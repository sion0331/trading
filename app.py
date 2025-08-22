from dash import Dash

from dashboard.backtest_callbacks import register_backtest_callbacks
from dashboard.callbacks_tob import register_callbacks as register_live_callbacks
from dashboard.layout import make_layout
from utils.config_loader import load_config

# app = Dash(__name__, suppress_callback_exceptions=True, title="TOB Viewer")
app = Dash(__name__)
app.layout = make_layout()

# live callbacks
register_live_callbacks(app)

# backtest callbacks
register_backtest_callbacks(app)

if __name__ == "__main__":
    config = load_config()

    dash_host = config['dash']['host']
    dash_port = config['dash']['port']

    app.run(host=dash_host, port=dash_port, debug=True)
