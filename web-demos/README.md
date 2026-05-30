# Web demos

Source for the interactive figures on the companion research preview:

- `demos/CostFrontierDemo.tsx`: round-trip AMM cost as a U-shaped function of trade size (fee + slippage + gas).
- `demos/NullControlDemo.tsx`: long-only strategy on a simulated market vs. its own bar-shuffle (the null-control signature).
- `demos/CrashAvoidanceDemo.tsx`: predicted-decile forward returns; the ranker's skill is in the left tail.
- `page.tsx`: the preview page that composes them.

React/TypeScript client components (Next.js). Self-contained and seeded.
