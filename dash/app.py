from dash import Dash

from callbacks_tob import register_callbacks
from layout import make_layout

# Create the app
app = Dash(__name__, suppress_callback_exceptions=True, title="TOB Viewer")
app.layout = make_layout()

# Wire callbacks
register_callbacks(app)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8050, debug=True)
