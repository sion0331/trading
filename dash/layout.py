# dash/layout.py
from dash import dash_table, html, dcc


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
                    # dcc.Input(
                    #     id="limit-input",
                    #     type="number",
                    #     min=10,
                    #     max=10000,
                    #     step=100,
                    #     value=300,
                    #     style={"width": "140px"},
                    #     placeholder="Rows",
                    # ),
                    # dcc.Checklist(
                    #     id="show-fills",
                    #     options=[{"label": "Show orders/fills", "value": "on"}],
                    #     value=[],
                    #     inline=True,
                    # ),
                    # dcc.Input(
                    #     id="fills-lookback-min",
                    #     type="number",
                    #     min=1, max=1440, step=1, value=120,  # last 120 minutes
                    #     style={"width": "140px"},
                    #     placeholder="Fills lookback (min)",
                    # ),
                ],
            ),
            dcc.Graph(id="tob-graph", style={"height": "100vh"}),
            html.H4("PnL by Trade"),
            dash_table.DataTable(
                id="pnl-table",
                page_size=12,
                sort_action="native",
                style_table={"overflowX": "auto"},
                style_cell={"fontSize": 13, "padding": "6px"},
                columns=[
                    {"name": "Time", "id": "ts"},
                    {"name": "Side", "id": "side"},
                    {"name": "Qty", "id": "qty", "type": "numeric", "format": {"specifier": ".0f"}},
                    {"name": "Price", "id": "price", "type": "numeric", "format": {"specifier": ".6f"}},
                    {"name": "Mid @ Exec", "id": "mid_exec", "type": "numeric", "format": {"specifier": ".6f"}},
                    {"name": "Commission", "id": "commission", "type": "numeric", "format": {"specifier": ".2f"}},
                    {"name": "Exec Impact PnL", "id": "exec_pnl", "type": "numeric", "format": {"specifier": ".2f"}},
                    {"name": "MTM PnL (after trade)", "id": "mtm_pnl", "type": "numeric",
                     "format": {"specifier": ".2f"}},
                ],
            ),

            # Poll the DB every 2 seconds
            dcc.Interval(id="refresh", interval=2000, n_intervals=0),
        ],
    )
