import json
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch

from orders.order_fx import notional_usd_from_qty
from strategy.transformer_simple.model import TinyTransformer
from utils.runtime_state import FrozenRuntimeState
from utils.utils_dt import _as_utc_dt


class TransformerStrategy:
    def __init__(self, contract, order_handler, position_handler, runtime_state,
                 model_dir: Path | None = None,
                 order_type="LMT", max_position=50_000, position_throttle=30,
                 lookback=120, cooldown_sec=5, ref_pips=0.1,  # lookback=120
                 last_trade_ts=datetime.now(timezone.utc)):
        self.contract = contract
        self.oh = order_handler
        self.ph = position_handler
        self.state = runtime_state or FrozenRuntimeState()

        self.max_position = max_position
        self.position_throttle = position_throttle
        self.order_type = order_type
        self.cooldown = timedelta(seconds=cooldown_sec)
        self.last_trade_ts = last_trade_ts

        self.lookback = lookback
        self.ref_pips = ref_pips * 1e-4
        self.buf = deque(maxlen=lookback)
        self.last_quote = None  # dict with bid/ask/mid/ts
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.control_params = {'order_type': None, 'max_position': 0.0, 'send_order': False}

        md = model_dir or (Path(__file__).resolve().parents[2] / "models" / "transformer_simple")
        with open(md / "eurusd_transformer.norm.json") as f:
            self.norm = json.load(f)
        self.mean = np.array(self.norm["feat_mean"], dtype=np.float32)
        self.std = np.array(self.norm["feat_std"], dtype=np.float32)

        self.model = TinyTransformer(in_dim=5, num_classes=3).to(self.device)
        sd = torch.load(md / "eurusd_transformer.pt", map_location=self.device)
        self.model.load_state_dict(sd["state_dict"])
        self.model.eval()

    def _append_feat(self, bid, ask, bid_sz, ask_sz, ts):
        mid = (bid + ask) / 2.0
        if self.last_quote is None:
            ret1 = 0.0;
            ret5 = 0.0
        else:
            prev_mid = self.last_quote["mid"]
            ret1 = (mid - prev_mid) / prev_mid if prev_mid > 0 else 0.0
            # rough ret5 using buffer
            last_mids = [m["mid"] for m in list(self.buf)[-5:] if "mid" in m]
            ret5 = ((mid - last_mids[0]) / last_mids[0]) if len(last_mids) >= 1 and last_mids[0] > 0 else 0.0

        spread_bp = ((ask - bid) / mid) * 1e4 if mid > 0 else 0.0
        imb = (bid_sz - ask_sz) / (bid_sz + ask_sz + 1e-9)
        tick_count = 1.0  # per tick approx; training used 1s agg, but this works ok

        self.last_quote = {"bid": bid, "ask": ask, "mid": mid, "ts": ts}
        # store feature vector and meta (for later ret5 calc)
        self.buf.append({"feat": np.array([ret1, ret5, spread_bp, imb, tick_count], dtype=np.float32),
                         "mid": mid})

    def _ready(self):
        return len(self.buf) >= self.lookback

    def _infer(self):
        X = np.stack([b["feat"] for b in self.buf], axis=0)  # [T,5]
        Xn = (X - self.mean) / self.std
        xt = torch.from_numpy(Xn).unsqueeze(0).to(self.device)  # [1,T,5]
        with torch.no_grad():
            logits = self.model(xt)
            prob = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            pred = int(prob.argmax())

            # print(logits, prob, pred)
        # map {0:DOWN,1:FLAT,2:UP} or {0:-1,1:0,2:1} depending on training
        # we used labels {-1,0,1} mapped to indices {0,1,2}. So:
        cls = [-1, 0, 1][pred]
        if prob.max() > 0.5:
            print(f"Direction:{'BUY' if cls > 0 else 'SELL' if cls < 0 else 'HOLD'} | Confidence: {prob.max()}")
        return cls, prob

    def _now_ok(self):
        now = datetime.now(timezone.utc)

        return now - self.last_trade_ts >= self.cooldown

    def on_market_data(self, msg):
        # Expecting your TOB wrapper carrying .bid, .ask, .bidSize, .askSize, .ts, .symbol
        self.update_params()
        bid = getattr(msg, "bid", None)
        ask = getattr(msg, "ask", None)
        bs = getattr(msg, "bidSize", getattr(msg, "bid_size", 0.0))
        asz = getattr(msg, "askSize", getattr(msg, "ask_size", 0.0))
        ts = getattr(msg, "ts", datetime.now(timezone.utc))
        if bid is None or ask is None: return

        # check cooldown
        ts = _as_utc_dt(msg.ts)
        if ts - self.last_trade_ts < self.cooldown:
            # print(f"[Strategy] In cooldown ({(ts - self.last_trade_ts).total_seconds():.2f}s)")
            return

        self._append_feat(float(bid), float(ask), float(bs or 0.0), float(asz or 0.0), ts)
        if not self._ready(): return

        cls, prob = self._infer()
        # position/pend
        price = (bid + ask) / 2.0
        # pos_obj = self.ph.positions.get(msg.symbol)
        # pos_qty = pos_obj.size if pos_obj else 0.0
        # # simple mapping: desire long on UP, short on DOWN, neutral otherwise
        # target_usd = 0.0
        # if cls > 0:  # UP
        #     target_usd = self.max_position
        # elif cls < 0:  # DOWN
        #     target_usd = -self.max_position
        # current_usd = pos_qty * price
        # delta_usd = target_usd - current_usd
        # if abs(delta_usd) < self.position_throttle:
        #     return  # nothing to do

        # Order placement (copying your SimpleStrategy approach)
        # side = "BUY" if delta_usd > 0 else "SELL"
        # lmt_price = bid if side == "BUY" else ask  # join best
        # order = self.oh.create_order(
        #     side,
        #     order_type="LMT" if self.order_type == "LMT" else "MKT",
        #     usd_notional=abs(delta_usd),
        #     limit_price=lmt_price if self.order_type == "LMT" else None,
        # )
        # self.oh.send_order(self.contract, order, _as_utc_dt(msg.ts))
        # self.last_trade_ts = datetime.now(timezone.utc)

        # Get current position for EURUSD
        position_obj = self.ph.positions.get(msg.symbol)
        current_position_qty = position_obj.size if position_obj else 0
        current_position_usd = notional_usd_from_qty(current_position_qty, price)

        pend_buy_usd = self.oh.pending_notional_usd(msg.symbol, "BUY", ref_price=price)
        pend_sell_usd = self.oh.pending_notional_usd(msg.symbol, "SELL", ref_price=price)
        effective_usd = current_position_usd + pend_buy_usd - pend_sell_usd

        # print(f"[Strategy] EURUSD Last: {price:.5f} | MA({self.window_size}): {ma:.5f} "
        #       f"| FilledUSD: {current_position_usd:.0f} | Pending(+buy/-sell): {pend_buy_usd:.0f}/{pend_sell_usd:.0f} "
        #       f"| EffectiveUSD: {effective_usd:.0f}")

        if cls > 0 and prob.max() > 0.5:
            # print(
            #     f"### {now} | SIGNAL BUY | {self.contract.symbol} Last: {price:.5f} | MA({self.window_size}): {ma:.5f} | Send: {self.control_params['send_order']} | Position: {current_position_usd}")
            # self.order_handler.cancel_all_for(self.contract.symbol, "SELL")

            if current_position_usd < self.control_params['max_position'] - self.position_throttle:

                if self.control_params['order_type'] == "MKT":
                    order = self.oh.create_order(
                        'BUY',
                        order_type='MKT',
                        usd_notional=self.control_params['max_position'] - current_position_usd,
                        ref_price=price,
                    )
                    if self.control_params['send_order']:
                        self.oh.send_order(self.contract, order, ts)
                    self.last_trade_ts = ts

                elif self.control_params['order_type'] == "LMT":
                    trade_size = self.control_params['max_position'] - current_position_usd
                    lmt_price = bid
                    best = self.oh.best_limit(self.contract.symbol, "BUY")
                    need_reprice = (best is None) or (
                            abs(best.lmt_price - lmt_price) >= self.ref_pips)
                    need_resize = abs(pend_buy_usd - trade_size) >= self.position_throttle

                    if not best or need_reprice or need_resize:
                        if self.control_params['send_order']:
                            self.oh.request_replace_limit(
                                self.contract,
                                "BUY",
                                usd_notional=trade_size,
                                limit_price=lmt_price,
                                tif="DAY",
                                cancel_opposite=True,  # optional: also clear SELLs
                                ts=ts
                            )
                        self.last_trade_ts = ts

        # SELL signal
        elif cls < 0 and prob.max() > 0.5:
            # print(
            #     f"### {now} | SIGNAL SELL | {self.contract.symbol} Last: {price:.5f} | MA({self.window_size}): {ma:.5f} | Send: {self.control_params['send_order']} | Position: {current_position_usd}")
            # self.order_handler.cancel_all_for(self.contract.symbol, "BUY")

            if current_position_usd > -self.control_params['max_position'] + self.position_throttle:

                if self.control_params['order_type'] == "MKT":
                    order = self.oh.create_order(
                        'SELL',
                        order_type='MKT',
                        usd_notional=self.control_params['max_position'] + current_position_usd,
                        ref_price=price,
                    )
                    if self.control_params['send_order']:
                        self.oh.send_order(self.contract, order, ts)
                    self.last_trade_ts = ts

                elif self.control_params['order_type'] == "LMT":
                    trade_size = self.control_params['max_position'] + current_position_usd
                    lmt_price = ask
                    best = self.oh.best_limit(self.contract.symbol, "SELL")
                    need_reprice = (best is None) or (
                            abs(best.lmt_price - lmt_price) >= self.ref_pips)  # 0.5 pip
                    need_resize = abs(pend_sell_usd - trade_size) >= self.position_throttle

                    if not best or need_reprice or need_resize:
                        if self.control_params['send_order']:
                            self.oh.request_replace_limit(
                                self.contract,
                                "SELL",
                                usd_notional=trade_size,
                                limit_price=lmt_price,
                                tif="DAY",
                                cancel_opposite=True,
                                ts=ts,
                            )
                        self.last_trade_ts = ts

    def update_params(self):
        # TODO - Cancel orders
        snap = self.state.get_snapshot()
        is_updated = False
        if self.control_params['order_type'] != snap['order_type'] and snap['order_type'] in ['LMT', 'MKT']:
            self.control_params['order_type'] = snap['order_type']
            is_updated = True
        if self.control_params['max_position'] != snap['max_position'] and snap['max_position'] >= 0:
            self.control_params['max_position'] = snap['max_position']
            is_updated = True
        if self.control_params['send_order'] != snap['send_order'] and snap['send_order'] in [True, False]:
            self.control_params['send_order'] = snap['send_order']
            is_updated = True
        if is_updated:
            print("### Updated Parameters: ", self.control_params)
