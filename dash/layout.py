# dash/layout.py
import datetime as _dt

import pytz
from dash import dash_table, html, dcc

NY = pytz.timezone("America/New_York")
_today_ny = _dt.datetime.now(NY).date()


def make_layout():
    return html.Div(
        style={"fontFamily": "system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial"},
        children=[
            html.H2("Top‑of‑Book (TOB) Viewer"),
            html.Div(
                style={"display": "flex", "gap": "12px", "alignItems": "center", "flexWrap": "wrap"},
                children=[
                    html.Label("Symbol"),
                    dcc.Dropdown(
                        id="symbol-dropdown",
                        options=[{"label": s, "value": s} for s in ["EUR", "BTC"]],
                        value="EUR",
                        clearable=False,
                        style={"width": "200px"},
                    ),
                    html.Label("Lookback (min)"),
                    dcc.Input(
                        id="lookback-min",
                        type="number",
                        min=1, max=1440, step=1,
                        value=30,  # 30 minutes default
                        style={"width": "120px"},
                    ),
                    dcc.Checklist(
                        id="stop-refresh",
                        options=[{"label": "Stop", "value": "stop"}],
                        value=[],  # empty list means unchecked
                        style={"display": "inline-block", "marginLeft": "10px"}
                    ),
                ],
            ),
            dcc.Graph(id="tob-graph", style={"height": "100vh"}),

            html.H4("PnL by Trade"),

            html.Div(
                style={"display": "flex", "gap": "12px", "alignItems": "center", "flexWrap": "wrap"},
                children=[
                    html.Label("Date"),
                    dcc.DatePickerSingle(
                        id="trades-date",
                        date=str(_today_ny),  # default: today (NY)
                        display_format="YYYY-MM-DD",
                        persistence=True,
                        style={"marginLeft": "16px"}
                    ),
                ],
            ),
            dash_table.DataTable(
                id="pnl-table",
                columns=[
                    {"name": "Time", "id": "ts"},
                    {"name": "Side", "id": "side"},
                    {"name": "USD", "id": "usd", "type": "numeric"},
                    {"name": "Qty", "id": "qty", "type": "numeric"},
                    {"name": "Price", "id": "price", "type": "numeric"},
                    {"name": "Mid@Fill", "id": "mid_exec", "type": "numeric"},
                    {"name": "Type", "id": "order_type"},
                    {"name": "Liq", "id": "liq"},
                    {"name": "Spread PnL", "id": "spread_pnl", "type": "numeric"},
                    {"name": "Commission", "id": "commission", "type": "numeric"},
                    {"name": "MTM Δ", "id": "mtm_incr", "type": "numeric"},
                    {"name": "Total", "id": "row_total", "type": "numeric"},
                ],
                fixed_rows={"headers": True, "data": 1},
                style_table={"height": "420px", "overflowY": "auto"},
                style_cell={"padding": "6px", "fontFamily": "monospace"},
                style_header={"fontWeight": "700"},
            ),

            dcc.Interval(id="refresh", interval=5_000, n_intervals=0),
        ],
    )
