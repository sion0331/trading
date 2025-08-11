import sqlite3
from datetime import datetime, timedelta

import dash
import pandas as pd
from dash import dcc, html

app = dash.Dash(__name__)


def load_latest_close(symbol="EUR", limit=1000, db_path="data/db/trading.db"):
    with sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES) as con:
        df = pd.read_sql_query("""
            SELECT ts, price
            FROM tape
            WHERE symbol = ?
            ORDER BY ts DESC
            LIMIT ?
        """, con, params=(symbol, limit))
    print("load tape: ", df)
    return df.sort_values("ts")


def load_latest_tob(symbol="EUR", limit=1000, db_path="data/db/trading.db"):
    """
    Load latest Top-of-Book rows for a symbol.
    Returns a DataFrame sorted ascending by ts with extra cols: mid, spread, spread_bps.
    """
    with sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES) as con:
        df = pd.read_sql_query("""
            SELECT
                ts,
                bid,
                ask,
                bid_size AS bidSize,
                ask_size AS askSize
            FROM tob
            WHERE symbol = ?
            ORDER BY ts DESC
            LIMIT ?
        """, con, params=(symbol, limit))
    print("load tob: ", df)
    if df.empty:
        return df

    # ensure datetime & chronological order
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.sort_values("ts").reset_index(drop=True)

    # handy derived metrics
    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    df["spread"] = df["ask"] - df["bid"]
    df["spread_bps"] = (df["spread"] / df["mid"]) * 1e4

    return df


# Hardcoded test data
def load_data():
    now = datetime.utcnow()
    timestamps = [now - timedelta(minutes=i) for i in reversed(range(10))]
    prices = [100 + i * 0.5 for i in range(10)]
    pnl = [i * 2 for i in range(10)]
    return pd.DataFrame({
        'timestamp': timestamps,
        'price': prices,
        'pnl': pnl
    })


app.layout = html.Div([
    html.H1("Trading Dashboard (Hardcoded Data)"),
    dcc.Interval(id='update', interval=30_000),  # update every 2s
    dcc.Graph(id='price-chart'),
    dcc.Graph(id='pnl-chart')
])


@app.callback(
    [dash.Output('price-chart', 'figure'),
     dash.Output('pnl-chart', 'figure')],
    dash.Input('update', 'n_intervals')
)
def update(_):
    df = load_data()
    print(df)
    price_fig = {
        'data': [{'x': df['timestamp'], 'y': df['price'], 'type': 'line', 'name': 'Price'}],
        'layout': {'title': 'Price Chart'}
    }
    pnl_fig = {
        'data': [{'x': df['timestamp'], 'y': df['pnl'], 'type': 'line', 'name': 'PnL'}],
        'layout': {'title': 'PnL Chart'}
    }
    load_latest_close()
    load_latest_tob()
    return price_fig, pnl_fig


if __name__ == '__main__':
    app.run(debug=True)
