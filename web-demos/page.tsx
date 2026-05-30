import type { Metadata } from "next";
import Link from "next/link";
import { Container } from "@/components/layout/Container";
import { CitationBlock } from "@/components/article/CitationBlock";
import { FigureWithCaption } from "@/components/article/FigureWithCaption";
import { ArrowUpRight, FileText, Github, Lock } from "lucide-react";
import CostFrontierDemo from "@/components/research/demos/dex/CostFrontierDemo";
import NullControlDemo from "@/components/research/demos/dex/NullControlDemo";
import CrashAvoidanceDemo from "@/components/research/demos/dex/CrashAvoidanceDemo";

const TITLE = "No Edge Without Information";
const SUBTITLE = "An empirical study of tradeability in decentralized-exchange-only cryptocurrencies";
const REPO_URL = "https://github.com/DaruFinance/dex-tradeability-study";

// Unlisted preview: keep out of search and out of site navigation until reviewed.
export const metadata: Metadata = {
  title: `${TITLE} (preview)`,
  robots: { index: false, follow: false },
};

const TLDR = [
  {
    label: "4,990 pairs · 27 chains",
    headline: "No price-based edge",
    body: "A long-only, realistically-costed trader has no exploitable edge from price information at any horizon. Real data performs worse than its own bar-shuffled null.",
  },
  {
    label: "fifteen analyses",
    headline: "The failure is structural",
    body: "It survives the intrabar exit convention, the selection regime, and the cost model, and it extends to hedging, liquidity provision, arbitrage, regime timing, portfolios, and order flow.",
  },
  {
    label: "one exception",
    headline: "Skill in the left tail",
    body: "A cross-sectional learner beats its null, but it predicts which coins crash, not which rise. A long-only trader cannot monetise it.",
  },
];

const METHOD = [
  "Long-only on the AMM (no native short), with a per-fill round-trip cost on both legs: swap fee plus constant-product slippage plus gas.",
  "Causal, walk-forward evaluation: parameters chosen in-sample, scored out-of-sample, with intrabar take-profit/stop using the bar high and low.",
  "A mandatory bar-shuffled null control: every claimed effect is re-run on a surrogate that destroys temporal order while keeping each coin's return distribution. An effect is real only if it beats the null net of cost.",
  "Full-distribution reporting (medians and the whole spread of outcomes), never a hand-picked winner, because the returns are extremely fat-tailed.",
  "Survivorship-aware throughout: the universe is sourced from currently-live pools, so every cross-sectional and holding metric is read as an upper bound.",
];

const apa =
  "Gatto, D. V. (2026). No Edge Without Information: An Empirical Study of Tradeability in Decentralized-Exchange-Only Cryptocurrencies. SSRN Working Paper (forthcoming).";
const bibtex = `@techreport{gatto2026dex,
  author      = {Gatto, Daniel V.},
  title       = {No Edge Without Information: An Empirical Study of
                 Tradeability in Decentralized-Exchange-Only Cryptocurrencies},
  year        = {2026},
  type        = {SSRN Working Paper (forthcoming)},
  note        = {Preprint}
}`;

export default function DexResearchPage() {
  return (
    <Container>
      <div className="py-16 md:py-20 max-w-5xl mx-auto">
        {/* unlisted preview banner */}
        <div className="mb-8 flex items-center gap-3 border border-[color:var(--color-border)] bg-[color:var(--color-surface)]/40 px-4 py-3">
          <Lock className="w-4 h-4 text-[color:var(--color-accent)] shrink-0" />
          <p className="font-mono text-[11px] text-[color:var(--color-fg-muted)] leading-relaxed">
            Unlisted preview. Not linked from the site and not indexed; shared for review ahead of
            the SSRN posting.
          </p>
        </div>

        <p className="font-mono text-[11px] uppercase tracking-[0.22em] text-[color:var(--color-accent)]">
          Research preview · extended study
        </p>
        <h1 className="mt-4 font-display text-4xl md:text-5xl font-semibold tracking-[-0.015em] leading-[1.08]">
          {TITLE}
        </h1>
        <p className="mt-3 font-display text-xl md:text-2xl text-[color:var(--color-fg-muted)] leading-snug">
          {SUBTITLE}
        </p>
        <p className="mt-6 text-lg text-[color:var(--color-fg-muted)] max-w-3xl leading-relaxed">
          This study makes a single argument. In decentralized-exchange markets of small, mostly
          non-CEX-listed tokens, an exploitable edge cannot be built from price information alone.
          Price-based timing fails, the failure is structural rather than methodological, the
          non-timing strategies a practitioner would reach for next fail too, and the only signal
          that beats a null is non-price and merely predicts crashes. The pieces below let you
          reproduce the core mechanics in the browser.
        </p>

        {/* action boxes: SSRN (coming soon) + repo */}
        <div className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-px bg-[color:var(--color-border)]">
          <div className="bg-[color:var(--color-bg)] p-6 opacity-70 cursor-default select-none">
            <div className="flex items-center justify-between">
              <FileText className="w-5 h-5 text-[color:var(--color-fg-subtle)]" />
              <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-[color:var(--color-fg-subtle)]">
                coming soon
              </span>
            </div>
            <h3 className="mt-3 font-display text-xl font-semibold text-[color:var(--color-fg-muted)]">
              View on SSRN
            </h3>
            <p className="mt-1 font-mono text-[11px] text-[color:var(--color-fg-subtle)]">
              working paper, not yet posted
            </p>
          </div>
          <a
            href={REPO_URL}
            target="_blank"
            rel="noreferrer"
            className="bg-[color:var(--color-bg)] hover:bg-[color:var(--color-surface)] transition-colors p-6 group"
          >
            <div className="flex items-center justify-between">
              <Github className="w-5 h-5 text-[color:var(--color-accent)]" />
              <ArrowUpRight className="w-4 h-4 text-[color:var(--color-fg-muted)] group-hover:text-[color:var(--color-fg)]" />
            </div>
            <h3 className="mt-3 font-display text-xl font-semibold">Code &amp; engines</h3>
            <p className="mt-1 font-mono text-[11px] text-[color:var(--color-fg-muted)]">
              github.com/DaruFinance/dex-tradeability-study
            </p>
          </a>
        </div>

        {/* TL;DR */}
        <div className="mt-16">
          <p className="font-mono text-[11px] uppercase tracking-[0.22em] text-[color:var(--color-fg-muted)]">
            The result in three lines
          </p>
          <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-px bg-[color:var(--color-border)]">
            {TLDR.map((card, i) => (
              <div key={i} className="bg-[color:var(--color-bg)] p-6">
                <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-[color:var(--color-accent)] tnum">
                  {card.label}
                </p>
                <p className="mt-3 font-display text-2xl font-semibold leading-snug tracking-[-0.01em]">
                  {card.headline}
                </p>
                <p className="mt-3 text-sm text-[color:var(--color-fg-muted)] leading-relaxed">
                  {card.body}
                </p>
              </div>
            ))}
          </div>
        </div>

        {/* body */}
        <div className="mt-16 grid grid-cols-1 md:grid-cols-12 gap-10">
          <div className="md:col-span-4">
            <h2 className="font-mono text-[11px] uppercase tracking-[0.22em] text-[color:var(--color-fg-muted)]">
              Overview
            </h2>
          </div>
          <div className="md:col-span-8 prose-daru">
            <p>
              DEX-only tokens are the part of the crypto market least exposed to professional
              high-frequency competition, which makes them a natural place to ask whether an
              ordinary participant can extract systematic profit. Two features make the question
              sharp: on an automated market maker there is no native short, so directional edge can
              only come from the long side; and the per-trade cost is large and endogenous to size,
              routinely 150 to 450 basis points round-trip.
            </p>
            <p>
              We assemble a survivorship-aware corpus of 4,990 trading pairs across 27 blockchains
              and run the evidence in five movements, each held to the same discipline: a bar-shuffled
              null control and full-distribution reporting. Price-based timing has no edge at any
              horizon, and real data is strictly worse than its own shuffle, because the return
              process is negative-drift, fat-tailed, volatility-clustered, and anti-momentum. That
              negative survives the obvious objections, and it extends to every non-timing strategy a
              practitioner would try next.
            </p>

            <h3 id="cost-wall">The cost wall</h3>
            <p>
              Edge has to clear cost before anything else. On an AMM the round-trip cost is not a
              fixed number; it is a U-shaped function of trade size, dominated by fixed gas on small
              trades and by constant-product slippage on large ones. The minimum, near the study&apos;s
              0.25%-of-reserve sizing, sits around 160 basis points on a deep pool and far higher on a
              thin one. Drag the pool depth and trade size and watch the three components trade off.
            </p>
            <CostFrontierDemo />
            <FigureWithCaption
              src="/figures/dex/cost-frontier.png"
              alt="Cost-versus-edge frontier: the median strategy's net return is negative at every cost level, and below the null at every point."
              number="1"
              caption="From the paper: the median strategy is net-negative even at zero cost, and below its null at every point on the cost frontier. There is no positive break-even cost."
              width={1311}
              height={862}
            />

            <h3 id="worse-than-noise">Worse than noise</h3>
            <p>
              The load-bearing finding is not merely that timing fails, but that real data performs
              worse than a bar-shuffled copy of itself. The shuffle keeps each coin&apos;s return
              distribution and destroys only the order of the bars. Because the order is what hurts a
              long position, clustered crashes on a downward, mean-reverting drift, shuffling it away
              helps. The demo runs a long-only momentum rule on a simulated market and on its own
              shuffle; the shuffle wins.
            </p>
            <NullControlDemo />
            <p className="text-sm text-[color:var(--color-fg-muted)] italic">
              The same pattern recurs across the alternative-channel studies, where several apparent
              &ldquo;edges&rdquo; turn out to be the shuffle paying the per-fill cost on destroyed
              structure, not alpha.
            </p>
            <FigureWithCaption
              src="/figures/dex/real-vs-null.png"
              alt="Real-versus-null decomposition: an unselected long-only bracket grid improves monotonically as temporal structure is destroyed."
              number="2"
              caption="An unselected long-only bracket grid improves monotonically as temporal structure is destroyed. Real is the worst case; the gap decomposes into crash-clustering and negative drift."
              width={1199}
              height={712}
            />

            <h3 id="one-signal">The one signal, and why it does not pay</h3>
            <p>
              A single test beats its null: a cross-sectional gradient-boosted ranker on time-varying
              causal features, with out-of-sample rank information coefficient well above its
              label-permutation baseline in every configuration. But its skill is crash avoidance,
              not return seeking. Sort the universe into deciles by the model&apos;s score and every
              decile&apos;s median forward return is negative; the top predicted decile is not the
              top-return decile. The information is in the left tail, which a long-only AMM trader
              cannot reach.
            </p>
            <CrashAvoidanceDemo />
            <FigureWithCaption
              src="/figures/dex/crash-decile.png"
              alt="Predicted decile versus forward return: every decile is negative; the model predicts which coins crash, not which rise."
              number="3"
              caption="From the paper: forward return by predicted decile. Every decile is negative and the top predicted decile is not the best performer. The signal identifies crashes, not winners."
              width={1498}
              height={936}
            />

            <h3>Method</h3>
            <ul className="list-disc pl-5 space-y-1">
              {METHOD.map((m, i) => (
                <li key={i}>{m}</li>
              ))}
            </ul>

            <h3>Reproducibility</h3>
            <p>
              The analysis engines and the source of the write-up are collected in a companion
              repository,{" "}
              <a href={REPO_URL} target="_blank" rel="noreferrer">
                dex-tradeability-study
              </a>
              . The interactive figures on this page are self-contained and seeded, so the mechanics
              they illustrate reproduce exactly on every load.
            </p>

            <h3>Cite</h3>
            <CitationBlock apa={apa} bibtex={bibtex} />

            <h3>See also</h3>
            <p>
              A companion methodological study on permutation testing and forward selection is at{" "}
              <Link href="/research">Research</Link>, and the narrative note{" "}
              <Link href="/articles/edge-from-process">The edge is in the process</Link> develops the
              selection-discipline theme that runs through both.
            </p>
          </div>
        </div>
      </div>
    </Container>
  );
}
