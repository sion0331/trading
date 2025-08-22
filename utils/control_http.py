from threading import Thread
from typing import Optional
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

class SetPayload(BaseModel):
    order_type: Optional[str] = None
    max_position: Optional[int] = None
    send_order: Optional[bool] = None

def make_app(state):
    app = FastAPI()

    @app.get("/state")
    def get_state():
        return state.get_snapshot()

    @app.post("/set")
    def set_state(p: SetPayload):
        updates = {k: v for k, v in p.dict().items() if v is not None}
        if updates:
            state.set(**updates)
        return {"ok": True, "state": state.get_snapshot()}

    @app.get("/", response_class=HTMLResponse)
    def index():
        return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Algo Control Panel</title>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 24px; }
    .card { max-width: 520px; padding: 20px; border: 1px solid #ddd; border-radius: 12px; box-shadow: 0 2px 6px rgba(0,0,0,.05); }
    .row { display: flex; gap: 12px; margin: 10px 0; align-items: center; }
    label { width: 140px; font-weight: 600; }
    input[type="number"] { width: 200px; }
    button { padding: 8px 14px; border: 0; border-radius: 8px; cursor: pointer; }
    .primary { background:#2563eb; color:#fff; }
    .muted { color:#666; font-size: 12px; }
    pre { background:#f7f7f7; padding:10px; border-radius:8px; overflow:auto; }
  </style>
</head>
<body>
  <h2>Algo Control Panel</h2>
  <div class="card">
    <div class="row">
      <label for="order_type">Order Type</label>
      <select id="order_type">
        <option value="LMT">LMT</option>
        <option value="MKT">MKT</option>
      </select>
    </div>
    <div class="row">
      <label for="max_position">Max Position</label>
      <input id="max_position" type="number" step="1000" min="0" />
    </div>
    <div class="row">
      <label for="send_order">Send Orders</label>
      <input id="send_order" type="checkbox" />
    </div>
    <div class="row">
      <button class="primary" id="apply">Apply</button>
      <button id="refresh">Refresh</button>
      <span id="msg" class="muted"></span>
    </div>
    <h4>Current State</h4>
    <pre id="state">loading…</pre>
  </div>

  <script>
    const stateEl = document.getElementById('state');
    const msgEl   = document.getElementById('msg');

    async function loadState() {
      const r = await fetch('/state');
      const s = await r.json();
      stateEl.textContent = JSON.stringify(s, null, 2);
      document.getElementById('order_type').value = s.order_type;
      document.getElementById('max_position').value = s.max_position;
      document.getElementById('send_order').checked = !!s.send_order;
    }

    async function apply() {
      msgEl.textContent = 'Applying…';
      const payload = {
        order_type: document.getElementById('order_type').value,
        max_position: Number(document.getElementById('max_position').value),
        send_order: document.getElementById('send_order').checked
      };
      const r = await fetch('/set', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify(payload)
      });
      const j = await r.json();
      stateEl.textContent = JSON.stringify(j.state || j, null, 2);
      msgEl.textContent = j.ok ? 'Updated' : 'Error';
    }

    document.getElementById('apply').addEventListener('click', apply);
    document.getElementById('refresh').addEventListener('click', loadState);

    loadState();
    setInterval(loadState, 5000); // auto-refresh
  </script>
</body>
</html>
        """

    return app

class ControlHTTPServer:
    def __init__(self, state, host="127.0.0.1", port=8788):
        self.state = state
        self.host = host
        self.port = port
        self._server = None
        self._thread = None
        self.app = make_app(self.state)

    def start(self):
        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="warning")
        self._server = uvicorn.Server(config)
        self._thread = Thread(target=self._server.run, daemon=True)
        self._thread.start()
