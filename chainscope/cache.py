"""Small in-memory TTL cache for GET responses (per-process)."""
from __future__ import annotations

import asyncio

_MISS = object()


class TTLCache:
    def __init__(self):
        self._data: dict = {}
        self._lock = asyncio.Lock()

    async def get(self, key):
        async with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return _MISS
            expires_at, value = entry
            if asyncio.get_event_loop().time() > expires_at:
                self._data.pop(key, None)
                return _MISS
            return value

    async def set(self, key, value, ttl: float) -> None:
        async with self._lock:
            self._data[key] = (asyncio.get_event_loop().time() + ttl, value)


MISS = _MISS
