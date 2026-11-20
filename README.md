# GTA VI Trophy Rarity Forecaster v2

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/rezimx-btc/gta6-trophy-rarity-forecast-01-v2/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/rezimx-btc/gta6-trophy-rarity-forecast-01-v2/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts, for every one of the 55 PlayStation 5 trophies of Grand Theft Auto VI, its earn rate on Exophase one week after launch (2026-11-26T00:00:00Z). Entered on [make.fun](https://make.fun/bounty/gta6-trophy-rarity-forecast-01) for the "Forecast how many GTA VI players earn each trophy in week one" bounty.

## Method

A count-consistent dilution forecaster, built for how the Exophase page actually moves. The page refreshes on its own schedule: a refresh jumps the tracked-player count up and dilutes every rate, and the rates then recover as the new cohort plays. So the forecast fits each trophy's RATE with a saturating logistic curve in log-time — the rate approaches a per-trophy ceiling instead of growing without bound — while keeping earners equal to rate times players, so the forecast stays count-consistent. Two further guards: when the display has been frozen long enough that a refresh is overdue, the median shifts down by the expected log-dilution and the spread widens — the shift is strong only when the history has never seen a refresh at all, since once refreshes appear in the history the trend fit already prices their average effect; and every fit is robust (Huber reweighting), so a single outlying read cannot bend a line. Persistence shrinkage toward the last observed rate was tried and rejected by backtesting: it pulls the median toward stale pre-growth rates and scores worse than no shrinkage. The spread combines the fit's residual, a random-walk widening with horizon, and a term for the refreshes expected before the target, and it is never degenerate — the model never claims certainty it does not have.

## Why v2

The previous version ranked #3058 with loss 0.409 while the top 10 score 0.0002–0.012, because its pure log-linear extrapolation overshot at the 4.5-day horizon and its spread was never tuned. This version replaces the log-linear extrapolation with a saturating fit, adds robust fitting and a refresh-aware median shift and spread, and tunes the spread by backtesting on the kit's practice cases and on synthetic refresh scenarios — train on early history reads, predict held-out later reads — requiring the new model to beat the old one on the same holds before it ships. Measured on that backtest set (2 practice cases with proxied answers + 4 synthetic final-like scenarios): v2 mean pinball loss 0.196 vs the previous version's 0.340; on two held-out synthetic scenarios it also wins (0.262 vs 0.307 and 0.256 vs 0.289). On the practice proxy the v2 loss is 0.035/0.135 vs the previous version's 0.404/0.404.

## Run it

The program is `main.ts` at the repository root (Node 22, standard library only, no network). It reads every case file from `MAKEFUN_INPUT_DIR` and writes one forecast file per case to `MAKEFUN_OUTPUT_DIR`:

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Each output is one JSON object with a single `trophies` key: one row per trophy, each row exactly `{"id": n, "q": [q05, q25, q50, q75, q95]}`. The program is deterministic — two runs produce byte-identical output — and each run takes well under the 8-second limit.
