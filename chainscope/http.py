"""Shared async HTTP client: per-host rate limiting, retry with backoff on
429/5xx/transport errors (honoring Retry-After), and an optional TTL cache for
GET requests. All providers fetch through this single client.
"""
from __future__ import annotations

import asyncio
import logging
from urllib.parse import urlsplit

import httpx

from .cache import MISS, TTLCache
from .ratelimit import RateLimiterRegistry

log = logging.getLogger("chainscope.http")

_RETRY_STATUS = {429, 500, 502, 503, 504}


class HttpError(Exception):
    def __init__(self, status: int, url: str, body: str = ""):
        self.status = status
        self.url = url
        self.body = body
        super().__init__(f"HTTP {status} for {url}: {body[:200]}")


class HttpClient:
    def __init__(self, timeout: float = 30.0, max_retries: int = 4,
                 user_agent: str = "chainscope/0.1"):
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers={"User-Agent": user_agent, "Accept": "application/json"},
            follow_redirects=True,
        )
        self._limiters = RateLimiterRegistry()
        self._cache = TTLCache()
        self.max_retries = max_retries

    @staticmethod
    def _host(url: str) -> str:
        return urlsplit(url).netloc.lower()

    async def _request(self, method, url, *, params=None, headers=None, json=None):
        limiter = await self._limiters.get(self._host(url))
        attempt = 0
        while True:
            attempt += 1
            await limiter.acquire()
            try:
                resp = await self._client.request(
                    method, url, params=params, headers=headers, json=json
                )
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                if attempt >= self.max_retries:
                    raise HttpError(0, url, str(exc)) from exc
                await asyncio.sleep(min(0.5 * 2 ** (attempt - 1), 8.0))
                continue

            if resp.status_code in _RETRY_STATUS and attempt < self.max_retries:
                retry_after = resp.headers.get("Retry-After")
                try:
                    delay = float(retry_after) if retry_after else min(0.5 * 2 ** (attempt - 1), 8.0)
                except ValueError:
                    delay = min(0.5 * 2 ** (attempt - 1), 8.0)
                log.debug("retry %s %s status=%s in %.1fs", method, url, resp.status_code, delay)
                await asyncio.sleep(delay)
                continue

            if resp.status_code >= 400:
                raise HttpError(resp.status_code, url, resp.text)
            return resp

    async def get_json(self, url, *, params=None, headers=None, cache_ttl: float | None = None):
        cache_key = None
        if cache_ttl:
            cache_key = (url, tuple(sorted((params or {}).items())))
            cached = await self._cache.get(cache_key)
            if cached is not MISS:
                return cached
        resp = await self._request("GET", url, params=params, headers=headers)
        data = resp.json()
        if cache_key is not None:
            await self._cache.set(cache_key, data, cache_ttl)
        return data

    async def post_json(self, url, *, json=None, headers=None, params=None):
        resp = await self._request("POST", url, params=params, headers=headers, json=json)
        return resp.json()

    async def aclose(self) -> None:
        await self._client.aclose()
