# dash/layout.py
from dash import html, dcc


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
                        options=[{"label": s, "value": s} for s in ["EUR", "EURUSD", "AAPL", "ETHUSD"]],
                        value="EUR",
                        clearable=False,
                        style={"width": "200px"},
                    ),
                    dcc.Input(
                        id="limit-input",
                        type="number",
                        min=100,
                        max=10000,
                        step=100,
                        value=3000,
                        style={"width": "140px"},
                        placeholder="Rows",
                    ),
                ],
            ),
            dcc.Graph(id="tob-graph", style={"height": "60vh"}),
            dcc.Graph(id="spread-graph", style={"height": "25vh", "marginTop": "8px"}),
            # Poll the DB every 2 seconds
            dcc.Interval(id="refresh", interval=2000, n_intervals=0),
        ],
    )
