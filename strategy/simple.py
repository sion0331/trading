from collections import deque
from datetime import datetime, timedelta, timezone

from marketData.data_types import Tob, Tape
from orders.order_fx import notional_usd_from_qty


class SimpleStrategy:
    def __init__(self, contract, order_handler, position_handler, order_type="LMT", max_position=30_000,
                 position_throttle=30, window_size=10, refresh_pips=0.1, cooldown_sec=10):
        self.contract = contract
        self.order_handler = order_handler
        self.position_handler = position_handler
        self.max_position = max_position
        self.position_throttle = position_throttle
        self.refresh_pips = refresh_pips

        self.order_type = order_type
        self.prices = deque(maxlen=window_size)
        self.window_size = window_size
        self.last = {}
        self.last_trade_ts = datetime.now(timezone.utc)
        self.cooldown = timedelta(seconds=cooldown_sec)

    def _price_from_msg(self, msg) -> float | None:
        if isinstance(msg, Tob):
            self.last.update({"bid": msg.bid, "ask": msg.ask, "mid": msg.mid(), "ts": msg.ts})
            return msg.mid()
        if isinstance(msg, Tape):
            self.last.update({"last": msg.price, "ts": msg.ts})
            return msg.price
        return None

    def on_market_data(self, msg):
        price = self._price_from_msg(msg)
        if price is None:
            print("Unknown Msg: ", msg)
            return

        self.prices.append(price)
        if len(self.prices) < self.window_size:
            print(f'Waiting for Market Data... {len(self.prices)} | {self.window_size}')
            return

        # check cooldown
        now = datetime.now(timezone.utc)
        if now - self.last_trade_ts < self.cooldown:
            # print(f"[Strategy] In cooldown ({(now - self.last_trade_ts).total_seconds():.2f}s)")
            return

        # Compute simple moving average
        ma = sum(self.prices) / self.window_size

        # Get current position for EURUSD
        position_obj = self.position_handler.positions.get(msg.symbol)
        current_position_qty = position_obj.size if position_obj else 0
        current_position_usd = notional_usd_from_qty(current_position_qty, price)

        pend_buy_usd = self.order_handler.pending_notional_usd(msg.symbol, "BUY", ref_price=price)
        pend_sell_usd = self.order_handler.pending_notional_usd(msg.symbol, "SELL", ref_price=price)
        effective_usd = current_position_usd + pend_buy_usd - pend_sell_usd

        # print(f"[Strategy] EURUSD Last: {price:.5f} | MA({self.window_size}): {ma:.5f} "
        #       f"| FilledUSD: {current_position_usd:.0f} | Pending(+buy/-sell): {pend_buy_usd:.0f}/{pend_sell_usd:.0f} "
        #       f"| EffectiveUSD: {effective_usd:.0f}")

        # BUY signal
        order = None
        if price > ma:
            print(
                f"[Strategy] BUY signal triggered | EURUSD Last: {price:.5f} | MA({self.window_size}): {ma:.5f} | Position: {current_position_usd}")
            self.order_handler.cancel_all_for(self.contract.symbol, "SELL")

            if current_position_usd < self.max_position - self.position_throttle:

                if self.order_type == "MKT":
                    order = self.order_handler.create_order(
                        'BUY',
                        order_type='MKT',
                        usd_notional=self.max_position - current_position_usd,
                        ref_price=price,
                    )
                    self.order_handler.send_order(self.contract, order)
                    self.last_trade_ts = now

                elif self.order_type == "LMT":
                    trade_size = self.max_position - current_position_usd
                    lmt_price = self.last.get('bid')
                    best = self.order_handler.best_limit(self.contract.symbol, "BUY")
                    need_reprice = (best is None) or (
                            abs(best.lmt_price - lmt_price) >= self.refresh_pips * 0.0001)  # 0.5 pip
                    need_resize = abs(pend_buy_usd - trade_size) >= self.position_throttle

                    if not best or need_reprice or need_resize:
                        self.order_handler.request_replace_limit(
                            self.contract,
                            "BUY",
                            usd_notional=trade_size,
                            limit_price=lmt_price,
                            tif="DAY",
                            cancel_opposite=True,  # optional: also clear SELLs
                        )
                        self.last_trade_ts = now

        # SELL signal
        elif price < ma:
            print(
                f"[Strategy] SELL signal triggered | EURUSD Last: {price:.5f} | MA({self.window_size}): {ma:.5f} | Position: {current_position_usd}")
            self.order_handler.cancel_all_for(self.contract.symbol, "BUY")

            if current_position_usd > -self.max_position + self.position_throttle:

                if self.order_type == "MKT":
                    order = self.order_handler.create_order(
                        'SELL',
                        order_type='MKT',
                        usd_notional=self.max_position + current_position_usd,
                        ref_price=price,
                    )
                    self.order_handler.send_order(self.contract, order)
                    self.last_trade_ts = now

                elif self.order_type == "LMT":
                    trade_size = self.max_position + current_position_usd
                    lmt_price = self.last.get('ask')
                    best = self.order_handler.best_limit(self.contract.symbol, "SELL")
                    need_reprice = (best is None) or (
                            abs(best.lmt_price - lmt_price) >= self.refresh_pips * 0.0001)  # 0.5 pip
                    need_resize = abs(pend_sell_usd - trade_size) >= self.position_throttle

                    if not best or need_reprice or need_resize:
                        self.order_handler.request_replace_limit(
                            self.contract,
                            "SELL",
                            usd_notional=trade_size,
                            limit_price=lmt_price,
                            tif="DAY",
                            cancel_opposite=True,
                        )
                        self.last_trade_ts = now
