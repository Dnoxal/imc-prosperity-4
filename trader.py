from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

try:
    from datamodel import Order as ExchangeOrder
    from datamodel import TradingState
except ImportError:
    ExchangeOrder = None
    TradingState = Any


@dataclass
class MarketSnapshot:
    day: int
    timestamp: int
    product: str
    bid_price_1: float
    bid_volume_1: int
    ask_price_1: float
    ask_volume_1: int
    mid_price: float


@dataclass
class SimOrder:
    product: str
    price: float
    quantity: int


@dataclass
class ProductConfig:
    position_limit: int
    max_clip: int
    trade_threshold: float
    inventory_penalty: float
    imbalance_weight: float
    anchor_price: Optional[float] = None
    regression_window: Optional[int] = None
    forecast_horizon: int = 1


class RollingLinearRegressor:
    def __init__(self, window: int) -> None:
        self.window = window
        self.xs: Deque[float] = deque()
        self.ys: Deque[float] = deque()

    def update(self, x: float, y: float) -> None:
        self.xs.append(x)
        self.ys.append(y)
        if len(self.xs) > self.window:
            self.xs.popleft()
            self.ys.popleft()

    def predict(self, x_next: float) -> Optional[float]:
        n = len(self.xs)
        if n < 5:
            if not self.ys:
                return None
            return self.ys[-1]

        mean_x = sum(self.xs) / n
        mean_y = sum(self.ys) / n
        denom = sum((x - mean_x) ** 2 for x in self.xs)
        if denom == 0:
            return self.ys[-1]

        slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(self.xs, self.ys)) / denom
        intercept = mean_y - slope * mean_x
        return intercept + slope * x_next


class Round1Trader:
    """
    Strategy summary:
    - ASH_COATED_OSMIUM behaves like a stable, mean-reverting product around 10_000.
    - INTARIAN_PEPPER_ROOT trends intraday, so we fit a rolling line and trade around
      that local fair value rather than a fixed anchor.
    - Both products also use top-of-book imbalance as a small directional signal.
    """

    def __init__(self) -> None:
        self.configs: Dict[str, ProductConfig] = {
            "ASH_COATED_OSMIUM": ProductConfig(
                position_limit=20,
                max_clip=10,
                trade_threshold=1.5,
                inventory_penalty=0.12,
                imbalance_weight=1.5,
                anchor_price=10_000.0,
            ),
            "INTARIAN_PEPPER_ROOT": ProductConfig(
                position_limit=20,
                max_clip=10,
                trade_threshold=2.0,
                inventory_penalty=0.10,
                imbalance_weight=2.0,
                regression_window=80,
                forecast_horizon=1,
            ),
        }
        self.position: Dict[str, int] = {product: 0 for product in self.configs}
        self.models: Dict[str, RollingLinearRegressor] = {
            product: RollingLinearRegressor(cfg.regression_window)
            for product, cfg in self.configs.items()
            if cfg.regression_window is not None
        }

    @staticmethod
    def _global_time(day: int, timestamp: int) -> int:
        return (day + 10) * 1_000_000 + timestamp

    @staticmethod
    def _imbalance(snapshot: MarketSnapshot) -> float:
        total = snapshot.bid_volume_1 + snapshot.ask_volume_1
        if total <= 0:
            return 0.0
        return (snapshot.bid_volume_1 - snapshot.ask_volume_1) / total

    def _fair_value(self, snapshot: MarketSnapshot) -> float:
        cfg = self.configs[snapshot.product]
        imbalance = self._imbalance(snapshot)

        if cfg.anchor_price is not None:
            fair = cfg.anchor_price
        else:
            model = self.models[snapshot.product]
            t_next = self._global_time(snapshot.day, snapshot.timestamp + 100 * cfg.forecast_horizon)
            fair = model.predict(t_next)
            if fair is None:
                fair = snapshot.mid_price

        fair += cfg.imbalance_weight * imbalance
        fair -= cfg.inventory_penalty * self.position[snapshot.product]
        return fair

    def generate_orders(self, snapshot: MarketSnapshot) -> List[SimOrder]:
        cfg = self.configs[snapshot.product]
        fair = self._fair_value(snapshot)
        position = self.position[snapshot.product]
        orders: List[SimOrder] = []

        if snapshot.ask_price_1 <= fair - cfg.trade_threshold and position < cfg.position_limit:
            buy_qty = min(cfg.max_clip, snapshot.ask_volume_1, cfg.position_limit - position)
            if buy_qty > 0:
                orders.append(
                    SimOrder(product=snapshot.product, price=snapshot.ask_price_1, quantity=buy_qty)
                )

        if snapshot.bid_price_1 >= fair + cfg.trade_threshold and position > -cfg.position_limit:
            sell_qty = min(cfg.max_clip, snapshot.bid_volume_1, cfg.position_limit + position)
            if sell_qty > 0:
                orders.append(
                    SimOrder(product=snapshot.product, price=snapshot.bid_price_1, quantity=-sell_qty)
                )

        return orders

    def on_fill(self, product: str, quantity: int) -> None:
        self.position[product] += quantity

    def on_snapshot_end(self, snapshot: MarketSnapshot) -> None:
        model = self.models.get(snapshot.product)
        if model is not None:
            model.update(self._global_time(snapshot.day, snapshot.timestamp), snapshot.mid_price)


class Trader:
    """
    Adapter for the IMC Prosperity submission interface.
    The simulator imports `Trader` and calls `run(state)` on every tick.
    """

    def __init__(self) -> None:
        self.strategy = Round1Trader()

    @staticmethod
    def _best_bid(order_depth: Any) -> Tuple[Optional[int], int]:
        if not order_depth.buy_orders:
            return None, 0
        price = max(order_depth.buy_orders)
        return price, int(order_depth.buy_orders[price])

    @staticmethod
    def _best_ask(order_depth: Any) -> Tuple[Optional[int], int]:
        if not order_depth.sell_orders:
            return None, 0
        price = min(order_depth.sell_orders)
        volume = -int(order_depth.sell_orders[price])
        return price, volume

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Any]], int, str]:
        result: Dict[str, List[Any]] = {}

        for product, order_depth in state.order_depths.items():
            bid_price, bid_volume = self._best_bid(order_depth)
            ask_price, ask_volume = self._best_ask(order_depth)
            if bid_price is None or ask_price is None:
                result[product] = []
                continue

            mid_price = (bid_price + ask_price) / 2
            snapshot = MarketSnapshot(
                day=0,
                timestamp=state.timestamp,
                product=product,
                bid_price_1=float(bid_price),
                bid_volume_1=bid_volume,
                ask_price_1=float(ask_price),
                ask_volume_1=ask_volume,
                mid_price=mid_price,
            )

            self.strategy.position[product] = state.position.get(product, 0)
            sim_orders = self.strategy.generate_orders(snapshot)
            exchange_orders: List[Any] = []

            for order in sim_orders:
                if ExchangeOrder is None:
                    exchange_orders.append(order)
                else:
                    exchange_orders.append(ExchangeOrder(order.product, int(order.price), int(order.quantity)))

            result[product] = exchange_orders
            self.strategy.on_snapshot_end(snapshot)

        conversions = 0
        trader_data = ""
        return result, conversions, trader_data
