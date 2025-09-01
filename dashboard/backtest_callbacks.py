from datetime import datetime

import pytz
from dash import Input, Output

from dashboard.callbacks_tob import fx_reset_window_for_date
from dashboard.utils import plot_market_executions
from database.db_loader import load_tob_range, load_fills_range, BACKTEST_DB, load_commissions_range, HISTORY_DB
from utils.utils_dt import _to_utc_ts

NY = pytz.timezone("America/New_York")


def register_backtest_callbacks(app):
    @app.callback(
        Output("bt-graph", "figure"),
        Output("bt-pnl-table", "data"),
        Input("bt-symbol", "value"),
        Input("bt-date", "date"),
    )
    def update_backtest(symbol, date_str):
        # 1) time window
        if not date_str:
            date_str = str(datetime.now(NY).date())
        start_utc, end_utc = fx_reset_window_for_date(date_str)
        start_iso, end_iso = start_utc.isoformat(), end_utc.isoformat()
        print("Backtest: ", start_iso, end_iso)

        # Load Data SQL
        df_mid = _to_utc_ts(load_tob_range(symbol, start_iso, end_iso, db_path=HISTORY_DB))
        df_exec = _to_utc_ts(load_fills_range(symbol, start_iso, end_iso, BACKTEST_DB))
        df_comm = _to_utc_ts(load_commissions_range(symbol, start_iso, end_iso, BACKTEST_DB))

        fig, pnl_rows = plot_market_executions(symbol, df_mid, df_exec, df_comm)

        return fig, pnl_rows
