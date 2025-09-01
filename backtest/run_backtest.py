from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytz

from backtest.broker import SimIB, FakeConnection, BacktestOrderHandler, Contract, BacktestPositions
from backtest.replay import load_tob, replay_tob
from database.orders import OrderLogger
from strategy.transformer_simple.strategy import TransformerStrategy
from utils.runtime_state import FrozenRuntimeState

NY = pytz.timezone("America/New_York")


def fx_reset_window(date_ny_str: str) -> tuple[datetime, datetime]:
    y, m, d = map(int, date_ny_str.split("-"))
    day_ny = NY.localize(datetime(y, m, d))
    end_ny = day_ny.replace(hour=23, minute=59, second=0, microsecond=0)
    # start_ny = day_ny.replace(hour=16, minute=55, second=0, microsecond=0)
    start_ny = end_ny + timedelta(days=-1)
    return start_ny.astimezone(timezone.utc), end_ny.astimezone(timezone.utc)


def main():
    symbol = "EUR"
    date_ny = "2025-08-29"
    start_utc, end_utc = fx_reset_window(date_ny)
    start_iso, end_iso = start_utc.isoformat(), end_utc.isoformat()
    print("Running Backtest: ", start_iso, end_iso)

    # Strategy Parameters
    frozen_state = FrozenRuntimeState(order_type="MKT", max_position=50_000, send_order=True)

    # 1) Sim broker & connection
    ib_sim = SimIB()
    conn = FakeConnection(ib_sim)

    # 2) Positions object with the interface your strategy reads
    pos = BacktestPositions()

    # 3) Order handler that talks to the sim broker, writing to a separate orders DB
    backtest_logger = OrderLogger(ib_sim, db_path="./data/db/orders_backtest.db")
    oh = BacktestOrderHandler(conn, frozen_state, logger=backtest_logger, orders_db_path="./data/db/orders_backtest.db")
    oh.create_events()  # register OrderHandler listeners to ib_sim events

    # Also update positions when fills happen (so strategy sees them)
    def _on_exec(tr, execution, *args, **kwargs):
        # todo error if fail
        side = getattr(execution, "side", "BOT")
        qty = getattr(execution, "shares", 0.0)
        sym = getattr(tr.contract, "symbol", symbol)
        pos.apply_fill(sym, side, qty)

    ib_sim.execDetailsEvent += _on_exec
    ib_sim.connect()

    # 4) Strategy
    contract = Contract(symbol=symbol, exchange="SIM", currency="USD")
    strat = TransformerStrategy(contract, oh, pos, runtime_state=frozen_state, last_trade_ts=start_utc)

    # strat = SimpleStrategy(
    #     contract=contract,
    #     order_handler=oh,
    #     position_handler=pos,  # exposes .positions dict the same way
    #     runtime_state=frozen_state,
    #     position_throttle=30,
    #     window_size=10,
    #     cooldown_sec=10,
    #     last_trade_ts=start_utc
    # )

    # 5) Load TOB from your market.db
    df = load_tob(symbol, start_iso, end_iso)
    if df.empty:
        print("No TOB rows in this window. Adjust symbol/date.")
        return

    # 6) Replay
    print(f"Replaying {len(df)} TOB rows for {symbol} from {start_iso} to {end_iso} ...")
    replay_tob(df, strat, ib_sim, symbol)
    print("Backtest finished.")


if __name__ == "__main__":
    main()
