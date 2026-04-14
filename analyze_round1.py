from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


DATA_DIR = Path("/Users/danielli/Downloads/ROUND1")
HIDDEN_LOG = Path("/Users/danielli/Downloads/114063/114063.log")


def load_visible_prices() -> pd.DataFrame:
    frames = [pd.read_csv(path, sep=";") for path in sorted(DATA_DIR.glob("prices_round_1_day_*.csv"))]
    prices = pd.concat(frames, ignore_index=True)
    return prices[prices["bid_price_1"].notna() & prices["ask_price_1"].notna()].copy()


def load_visible_trades() -> pd.DataFrame:
    frames = []
    for path in sorted(DATA_DIR.glob("trades_round_1_day_*.csv")):
        day = int(path.stem.split("_")[-1])
        frame = pd.read_csv(path, sep=";")
        frame["day"] = day
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_hidden_prices() -> pd.DataFrame:
    payload = json.loads(HIDDEN_LOG.read_text())
    prices = pd.read_csv(pd.io.common.StringIO(payload["activitiesLog"]), sep=";")
    prices = prices[prices["product"].notna() & prices["bid_price_1"].notna() & prices["ask_price_1"].notna()].copy()
    prices["day"] = 1
    return prices


def add_features(prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.copy()
    prices["imbalance"] = (
        prices["bid_volume_1"].fillna(0) - prices["ask_volume_1"].fillna(0)
    ) / (prices["bid_volume_1"].fillna(0) + prices["ask_volume_1"].fillna(0))
    prices["spread"] = prices["ask_price_1"] - prices["bid_price_1"]
    return prices


def summarize(prices: pd.DataFrame) -> None:
    print("Visible Round 1 Summary")
    print("=======================")
    print(f"Rows: {len(prices)}")
    print(f"Products: {sorted(prices['product'].unique().tolist())}")
    print(f"Days: {sorted(prices['day'].unique().tolist())}")
    print()

    for product in sorted(prices["product"].unique()):
        product_rows = prices[prices["product"] == product].sort_values(["day", "timestamp"]).copy()
        print(product)
        print(f"  rows={len(product_rows)} spread_mean={product_rows['spread'].mean():.2f}")
        for horizon in [1, 2, 5, 10]:
            product_rows[f"future_{horizon}"] = (
                product_rows["mid_price"].shift(-horizon) - product_rows["mid_price"]
            )
            corr = product_rows["imbalance"].corr(product_rows[f"future_{horizon}"])
            print(f"  imbalance->future_{horizon}: {corr:.4f}")
        print()


def main() -> None:
    visible_prices = add_features(load_visible_prices())
    visible_trades = load_visible_trades()
    hidden_prices = add_features(load_hidden_prices())

    summarize(visible_prices)
    print(f"Visible trade rows: {len(visible_trades)}")
    print(f"Hidden price rows: {len(hidden_prices)}")


if __name__ == "__main__":
    main()
