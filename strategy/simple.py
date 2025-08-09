from collections import deque
from datetime import datetime, timedelta
from marketData.data_types import Tob, Tape


class SimpleStrategy:
    def __init__(self, contract, order_handler, position_handler, trade_size=10000, window_size=10, cooldown_sec=10):
        self.contract = contract
        self.order_handler = order_handler
        self.position_handler = position_handler
        self.trade_size = trade_size

        self.prices = deque(maxlen=window_size)
        self.window_size = window_size
        self.last_price = None
        self.last_trade_ts = datetime.min
        self.cooldown = timedelta(seconds=cooldown_sec)

    def _price_from_msg(self, msg) -> float | None:
        if isinstance(msg, Tob):
            return msg.mid()
        if isinstance(msg, Tape):
            return msg.price
        return None

    def on_market_data(self, msg):
        price = self._price_from_msg(msg)
        if price is None:
            return

        self.prices.append(price)
        if len(self.prices) < self.window_size:
            print(f'Waiting for Market Data... {len(self.prices)}, " | ", {self.window_size}')
            return

        # check cooldown
        now = datetime.utcnow()
        if now - self.last_trade_ts < self.cooldown:
            print(f"[Strategy] In cooldown ({(now - self.last_trade_ts).total_seconds():.2f}s)")
            return

        # Compute simple moving average
        ma = sum(self.prices) / self.window_size

        # Get current position for EURUSD
        position_obj = self.position_handler.positions.get(msg.symbol)
        current_position = position_obj.size if position_obj else 0

        # Log signal
        print(
            f"[Strategy] EURUSD Last Price: {self.last_price:.5f} | MA({self.window_size}): {ma:.5f} | Position: {current_position}")

        # BUY signal
        if current_position <= self.trade_size and self.last_price > ma:
            print("[Strategy] BUY signal triggered")
            order = self.order_handler.create_order('BUY', self.trade_size - current_position)
            self.order_handler.send_order(self.contract, order)
            self.last_trade_ts = now

        # SELL signal
        elif -self.trade_size < current_position and self.last_price < ma:
            print("[Strategy] SELL signal triggered")
            order = self.order_handler.create_order('SELL', self.trade_size + current_position)
            self.order_handler.send_order(self.contract, order)
            self.last_trade_ts = now