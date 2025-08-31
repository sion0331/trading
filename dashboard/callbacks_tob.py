from datetime import datetime, timezone, timedelta

import pytz
from dash import Input, Output

from dashboard.utils import plot_market_executions
from database.db_loader import load_tob_range, load_fills_range, load_commissions_range, ORDERS_DB
from utils.utils_dt import _to_utc_ts

NY = pytz.timezone("America/New_York")


def fx_reset_window_for_date(date_str: str):
    """
    For a YYYY-MM-DD date (NY), return (start_utc, end_utc) where the FX day is
    16:55 NY on that date -> 16:55 NY next date.
    """
    # parse selected date as NY-local midnight of that day
    y, m, d = map(int, date_str.split("-"))
    day_ny = NY.localize(datetime(y, m, d))
    end_ny = day_ny.replace(hour=23, minute=58, second=0, microsecond=0)
    # end_ny = day_ny.replace(hour=16, minute=55, second=0, microsecond=0)
    start_ny = end_ny + timedelta(days=-1)
    return start_ny.astimezone(timezone.utc), end_ny.astimezone(timezone.utc)


def fx_reset_start_utc(now_utc: datetime | None = None) -> datetime:
    """
    IBKR reset for FX: 16:55 America/New_York every trading day.
    Returns that instant in UTC for the current 'today' boundary.
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    now_ny = now_utc.astimezone(NY)
    # today 16:55 NY
    reset_today_ny = now_ny.replace(hour=16, minute=55, second=0, microsecond=0)
    if now_ny >= reset_today_ny:
        reset_ny = reset_today_ny
    else:
        reset_ny = reset_today_ny - timedelta(days=1)
    return reset_ny.astimezone(timezone.utc)


def register_callbacks(app):
    @app.callback(
        Output("refresh", "disabled"),
        Input("stop-refresh", "value")
    )
    def toggle_refresh(stop_values):
        return "stop" in stop_values  # True disables the interval

    @app.callback(
        Output("tob-graph", "figure"),
        Output("pnl-table", "data"),
        Input("refresh", "n_intervals"),
        Input("symbol-dropdown", "value"),
        Input("lookback-min", "value"),
        Input("trades-date", "date"),
    )
    def update_price_pnl(n_intervals, symbol, lookback_min, trades_date):
        is_today = datetime.today().strftime('%Y-%m-%d') == trades_date

        lookback_min = int(lookback_min or 30)
        lookback_iso = (datetime.now(timezone.utc) - timedelta(minutes=lookback_min)).isoformat() if is_today else None

        start_utc, end_utc = fx_reset_window_for_date(trades_date or str(datetime.now(NY).date()))
        start_iso = start_utc.isoformat()
        end_iso = end_utc.isoformat()
        # print(f'{start_iso} - {end_iso} | lookback: {lookback_iso}')

        # Load Data SQL
        df_mid = _to_utc_ts(load_tob_range(symbol, start_iso, end_iso))
        df_exec = _to_utc_ts(load_fills_range(symbol, start_iso, end_iso, ORDERS_DB))
        df_comm = _to_utc_ts(load_commissions_range(symbol, start_iso, end_iso, ORDERS_DB))

        fig, pnl_rows = plot_market_executions(symbol, df_mid, df_exec, df_comm, lookback_iso=lookback_iso)

        return fig, pnl_rows
