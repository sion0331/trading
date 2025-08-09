import dash
from dash import dcc, html
import pandas as pd
from datetime import datetime, timedelta

app = dash.Dash(__name__)

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
    dcc.Interval(id='update', interval=2000),  # update every 2s
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
    return price_fig, pnl_fig

if __name__ == '__main__':
    app.run(debug=True)
