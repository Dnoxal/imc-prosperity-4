from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import trader as trader_module
from trader import Trader


@dataclass
class MockOrder:
    symbol: str
    price: int
    quantity: int


@dataclass
class MockOrderDepth:
    buy_orders: Dict[int, int] = field(default_factory=dict)
    sell_orders: Dict[int, int] = field(default_factory=dict)


@dataclass
class MockTradingState:
    timestamp: int
    order_depths: Dict[str, MockOrderDepth]
    position: Dict[str, int]


@dataclass
class MarketFrame:
    day: int
    timestamp: int
    order_depths: Dict[str, MockOrderDepth]
    mid_prices: Dict[str, float]


@dataclass
class Fill:
    day: int
    timestamp: int
    product: str
    price: float
    quantity: int


class Backtester:
    def __init__(self, frames: Iterable[MarketFrame]) -> None:
        self.frames = list(frames)
        trader_module.ExchangeOrder = MockOrder
        self.trader = Trader()
        self.cash: Dict[str, float] = defaultdict(float)
        self.position: Dict[str, int] = defaultdict(int)
        self.last_mid: Dict[str, float] = {}
        self.trades: List[Fill] = []

    def _execute_order(
        self,
        day: int,
        timestamp: int,
        product: str,
        order: MockOrder,
        order_depth: MockOrderDepth,
    ) -> None:
        if order.quantity == 0:
            return

        if order.quantity > 0:
            remaining = order.quantity
            for ask_price in sorted(order_depth.sell_orders):
                if ask_price > order.price or remaining <= 0:
                    break
                available = -order_depth.sell_orders[ask_price]
                fill_qty = min(remaining, available)
                if fill_qty <= 0:
                    continue
                order_depth.sell_orders[ask_price] += fill_qty
                self.cash[product] -= fill_qty * ask_price
                self.position[product] += fill_qty
                self.trades.append(Fill(day, timestamp, product, float(ask_price), fill_qty))
                remaining -= fill_qty
        else:
            remaining = -order.quantity
            for bid_price in sorted(order_depth.buy_orders, reverse=True):
                if bid_price < order.price or remaining <= 0:
                    break
                available = order_depth.buy_orders[bid_price]
                fill_qty = min(remaining, available)
                if fill_qty <= 0:
                    continue
                order_depth.buy_orders[bid_price] -= fill_qty
                self.cash[product] += fill_qty * bid_price
                self.position[product] -= fill_qty
                self.trades.append(Fill(day, timestamp, product, float(bid_price), -fill_qty))
                remaining -= fill_qty

    def run(self) -> Dict[str, float]:
        for frame in self.frames:
            state = MockTradingState(
                timestamp=frame.timestamp,
                order_depths=frame.order_depths,
                position=dict(self.position),
            )
            result, _, _ = self.trader.run(state)

            for product, orders in result.items():
                order_depth = frame.order_depths[product]
                for order in orders:
                    self._execute_order(frame.day, frame.timestamp, product, order, order_depth)

            self.last_mid.update(frame.mid_prices)

        per_product = {}
        for product in set(self.position) | set(self.last_mid):
            per_product[product] = self.cash[product] + self.position[product] * self.last_mid.get(product, 0.0)
        return per_product


def parse_int(value: str) -> int:
    if value == "" or value is None:
        return 0
    return int(float(value))


def parse_float(value: str) -> float:
    if value == "" or value is None:
        return math.nan
    return float(value)


def build_order_depth(row: Dict[str, str]) -> MockOrderDepth:
    buy_orders: Dict[int, int] = {}
    sell_orders: Dict[int, int] = {}

    for level in (1, 2, 3):
        bid_price = parse_float(row.get(f"bid_price_{level}", ""))
        bid_volume = parse_int(row.get(f"bid_volume_{level}", ""))
        ask_price = parse_float(row.get(f"ask_price_{level}", ""))
        ask_volume = parse_int(row.get(f"ask_volume_{level}", ""))

        if not math.isnan(bid_price) and bid_volume > 0:
            buy_orders[int(bid_price)] = bid_volume
        if not math.isnan(ask_price) and ask_volume > 0:
            sell_orders[int(ask_price)] = -ask_volume

    return MockOrderDepth(buy_orders=buy_orders, sell_orders=sell_orders)


def load_frames(data_dir: Path) -> List[MarketFrame]:
    grouped: Dict[Tuple[int, int], Dict[str, Tuple[MockOrderDepth, float]]] = defaultdict(dict)

    for file_path in sorted(data_dir.glob("prices_round_1_day_*.csv")):
        with file_path.open(newline="") as handle:
            reader = csv.DictReader(handle, delimiter=";")
            for row in reader:
                order_depth = build_order_depth(row)
                if not order_depth.buy_orders or not order_depth.sell_orders:
                    continue

                day = int(row["day"])
                timestamp = int(row["timestamp"])
                grouped[(day, timestamp)][row["product"]] = (order_depth, parse_float(row["mid_price"]))

    frames: List[MarketFrame] = []
    for (day, timestamp) in sorted(grouped):
        per_product = grouped[(day, timestamp)]
        frames.append(
            MarketFrame(
                day=day,
                timestamp=timestamp,
                order_depths={product: depth for product, (depth, _) in per_product.items()},
                mid_prices={product: mid for product, (_, mid) in per_product.items()},
            )
        )
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest the submission Trader.run(...) on historical price data.")
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

    frames = load_frames(Path(args.data_dir))
    if not frames:
        raise SystemExit(f"No usable price frames found in {args.data_dir}")

    backtester = Backtester(frames)
    per_product = backtester.run()
    total = sum(per_product.values())

    print("Round 1 Backtest")
    print("================")
    print(f"Frames: {len(frames)}")
    print(f"Fills: {len(backtester.trades)}")
    print(f"Total PnL: {total:.2f}")
    print()
    print("Per product:")
    for product, pnl in sorted(per_product.items()):
        position = backtester.position[product]
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
