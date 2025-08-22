from dash import Dash

from backtest_callbacks import register_backtest_callbacks
from callbacks_tob import register_callbacks as register_live_callbacks
from layout import make_layout

# app = Dash(__name__, suppress_callback_exceptions=True, title="TOB Viewer")
app = Dash(__name__)
app.layout = make_layout()

# live callbacks
register_live_callbacks(app)
# backtest callbacks
register_backtest_callbacks(app)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8050, debug=True)
