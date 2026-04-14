# Round 1 Trading Algorithm

This repository now includes:

- `trader.py`: the actual strategy logic
- `backtest.py`: a simple historical backtester you can run from the command line

## Strategy idea

The Round 1 data supports two different behaviors:

- `ASH_COATED_OSMIUM` is close to a stationary product around `10000`, so the strategy treats it as mean-reverting.
- `INTARIAN_PEPPER_ROOT` trends strongly within each day, so the strategy fits a rolling linear fair value and then trades short-term dislocations around that trend.

Both products also use top-of-book volume imbalance:

`imbalance = (bid_volume_1 - ask_volume_1) / (bid_volume_1 + ask_volume_1)`

That imbalance nudges fair value up when the bid looks stronger and down when the ask looks stronger.

## Backtest

Run:

```bash
python3 backtest.py --data-dir /Users/danielli/Downloads/ROUND1
```

If you keep the data in the default location above, `--data-dir` is optional:

```bash
python3 backtest.py
```

To export every simulated fill:

```bash
python3 backtest.py --export-trades trades.csv
```

## What the backtester does

- Loads all `prices_round_1_day_*.csv` files
- Skips rows with no usable best bid or ask
- Calls the strategy once per snapshot
- Fills only marketable orders against the displayed best bid/ask
- Marks any leftover inventory to the final mid price

This is intentionally simple and easy to edit. If you want, the next step would be to add:

- per-day metrics
- a trade log CSV export
- parameter sweeps
- a richer passive-fill model
