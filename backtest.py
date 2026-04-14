from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from trader import MarketSnapshot, Round1Trader, SimOrder


@dataclass
class Fill:
    day: int
    timestamp: int
    product: str
    price: float
    quantity: int


class Backtester:
    def __init__(self, snapshots: Iterable[MarketSnapshot]) -> None:
        self.snapshots = list(snapshots)
        self.trader = Round1Trader()
        self.cash: Dict[str, float] = defaultdict(float)
        self.last_mid: Dict[str, float] = {}
        self.trades: List[Fill] = []

    def _execute(self, snapshot: MarketSnapshot, orders: List[SimOrder]) -> None:
        for order in orders:
            if order.quantity > 0:
                if order.price >= snapshot.ask_price_1:
                    fill_qty = min(order.quantity, snapshot.ask_volume_1)
                    if fill_qty > 0:
                        self.cash[order.product] -= fill_qty * snapshot.ask_price_1
                        self.trader.on_fill(order.product, fill_qty)
                        self.trades.append(
                            Fill(
                                day=snapshot.day,
                                timestamp=snapshot.timestamp,
                                product=order.product,
                                price=snapshot.ask_price_1,
                                quantity=fill_qty,
                            )
                        )
            elif order.quantity < 0:
                if order.price <= snapshot.bid_price_1:
                    fill_qty = min(-order.quantity, snapshot.bid_volume_1)
                    if fill_qty > 0:
                        self.cash[order.product] += fill_qty * snapshot.bid_price_1
                        self.trader.on_fill(order.product, -fill_qty)
                        self.trades.append(
                            Fill(
                                day=snapshot.day,
                                timestamp=snapshot.timestamp,
                                product=order.product,
                                price=snapshot.bid_price_1,
                                quantity=-fill_qty,
                            )
                        )

    def run(self) -> Dict[str, float]:
        for snapshot in self.snapshots:
            orders = self.trader.generate_orders(snapshot)
            self._execute(snapshot, orders)
            self.last_mid[snapshot.product] = snapshot.mid_price
            self.trader.on_snapshot_end(snapshot)

        per_product = {}
        for product, position in self.trader.position.items():
            per_product[product] = self.cash[product] + position * self.last_mid.get(product, 0.0)
        return per_product


def parse_int(value: str) -> int:
    if value == "" or value is None:
        return 0
    return int(float(value))


def parse_float(value: str) -> float:
    if value == "" or value is None:
        return math.nan
    return float(value)


def load_snapshots(data_dir: Path) -> List[MarketSnapshot]:
    snapshots: List[MarketSnapshot] = []
    for file_path in sorted(data_dir.glob("prices_round_1_day_*.csv")):
        with file_path.open(newline="") as handle:
            reader = csv.DictReader(handle, delimiter=";")
            for row in reader:
                bid_price = parse_float(row["bid_price_1"])
                ask_price = parse_float(row["ask_price_1"])
                if math.isnan(bid_price) or math.isnan(ask_price):
                    continue

                snapshots.append(
                    MarketSnapshot(
                        day=int(row["day"]),
                        timestamp=int(row["timestamp"]),
                        product=row["product"],
                        bid_price_1=bid_price,
                        bid_volume_1=parse_int(row["bid_volume_1"]),
                        ask_price_1=ask_price,
                        ask_volume_1=parse_int(row["ask_volume_1"]),
                        mid_price=parse_float(row["mid_price"]),
                    )
                )

    snapshots.sort(key=lambda s: (s.day, s.timestamp, s.product))
    return snapshots


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest the Round 1 strategy on historical price data.")
    parser.add_argument(
        "--data-dir",
        default="/Users/danielli/Downloads/ROUND1",
        help="Directory containing the Round 1 CSV files.",
    )
    parser.add_argument(
        "--export-trades",
        help="Optional path to write a CSV trade log.",
    )
    args = parser.parse_args()

    snapshots = load_snapshots(Path(args.data_dir))
    if not snapshots:
        raise SystemExit(f"No usable price snapshots found in {args.data_dir}")

    backtester = Backtester(snapshots)
    per_product = backtester.run()
    total = sum(per_product.values())

    print("Round 1 Backtest")
    print("================")
    print(f"Snapshots: {len(snapshots)}")
    print(f"Fills: {len(backtester.trades)}")
    print(f"Total PnL: {total:.2f}")
    print()
    print("Per product:")
    for product, pnl in sorted(per_product.items()):
        position = backtester.trader.position[product]
        print(f"  {product:24s} pnl={pnl:10.2f} final_position={position:4d}")

    if args.export_trades:
        export_path = Path(args.export_trades)
        with export_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["day", "timestamp", "product", "price", "quantity"])
            for fill in backtester.trades:
                writer.writerow([fill.day, fill.timestamp, fill.product, f"{fill.price:.2f}", fill.quantity])
        print()
        print(f"Trade log written to {export_path}")


if __name__ == "__main__":
    main()
