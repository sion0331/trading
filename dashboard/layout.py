import datetime as _dt

import pytz
from dash import dash_table, html, dcc

NY = pytz.timezone("America/New_York")
_today_ny = _dt.datetime.now(NY).date()


def _live_tab():
    return html.Div(
        children=[
            html.H2("Top-of-Book (TOB) Viewer"),
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
                        value=10,
                        style={"width": "120px"},
                    ),
                    dcc.Checklist(
                        id="stop-refresh",
                        options=[{"label": "Stop", "value": "stop"}],
                        value=[],
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
                        date=str(_today_ny),
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
                    {"name": "Bid", "id": "bid", "type": "numeric"},
                    {"name": "Mid", "id": "mid", "type": "numeric"},
                    {"name": "Ask", "id": "ask", "type": "numeric"},
                    {"name": "Type", "id": "order_type"},
                    {"name": "Sprd PnL", "id": "spread_pnl", "type": "numeric"},
                    {"name": "Mkt PnL", "id": "market_pnl", "type": "numeric"},
                    {"name": "Fee", "id": "fee", "type": "numeric"},
                    {"name": "Total", "id": "row_total", "type": "numeric"},
                ],
                fixed_rows={"headers": True, "data": 1},
                style_table={"height": "420px", "overflowY": "auto"},
                style_cell={"padding": "6px", "fontFamily": "monospace"},
                style_header={"fontWeight": "700"},
            ),
            dcc.Interval(id="refresh", interval=5_000, n_intervals=0),
        ]
    )


def _backtest_tab():
    return html.Div(
        children=[
            html.H2("Backtest Viewer"),
            html.Div(
                style={"display": "flex", "gap": "12px", "alignItems": "center", "flexWrap": "wrap"},
                children=[
                    html.Label("Symbol"),
                    dcc.Dropdown(
                        id="bt-symbol",
                        options=[{"label": s, "value": s} for s in ["EUR", "BTC"]],
                        value="EUR",
                        clearable=False,
                        style={"width": "200px"},
                    ),
                    html.Label("Date"),
                    dcc.DatePickerSingle(
                        id="bt-date",
                        date=str(_today_ny),
                        display_format="YYYY-MM-DD",
                        persistence=True,
                        style={"marginLeft": "16px"}
                    ),
                ],
            ),
            dcc.Graph(id="bt-graph", style={"height": "90vh"}),
        ]
    )


def make_layout():
    return html.Div(
        style={"fontFamily": "system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial"},
        children=[
            dcc.Tabs(
                id="tabs",
                value="live",
                children=[
                    dcc.Tab(label="Live", value="live", children=_live_tab()),
                    dcc.Tab(label="Backtest", value="backtest", children=_backtest_tab()),
                ],
            ),
        ],
    )
