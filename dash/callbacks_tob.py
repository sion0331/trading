# dash/callbacks_tob.py
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output
# from data import load_latest_tob
# from data_orders import load_symbol_executions, load_symbol_latest_limit  # <-- NEW
from plotly.subplots import make_subplots

from data import load_tob_since
from data_orders import (
    load_executions_since,  # already used for plotting fills
    load_executions_raw_since,  # NEW: includes exec_id
    load_commissions_since,  # NEW
    compute_pnl_curve,
)


def register_callbacks(app):
    @app.callback(
        Output("tob-graph", "figure"),  # <- one Graph only
        Output("pnl-table", "data"),
        Input("refresh", "n_intervals"),
        Input("symbol-dropdown", "value"),
        Input("lookback-min", "value"),
    )
    def update_price_pnl(_, symbol, lookback_min):
        lookback_min = int(lookback_min or 30)

        # ---- load data
        df = load_tob_since(symbol, lookback_minutes=lookback_min)  # ts, bid, ask, mid, ...
        df_exec = load_executions_since(symbol, lookback_minutes=lookback_min)  # ts, price, qty, side
        df_exec_raw = load_executions_raw_since(symbol, lookback_minutes=lookback_min)  # + exec_id (for table)
        df_comm = load_commissions_since(lookback_min)  # exec_id, commission

        pnl_curve = compute_pnl_curve(
            df[["ts", "mid"]].copy() if not df.empty else df,
            df_exec if df_exec is not None else df_exec
        )

        # ---- figure with shared x-axis
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.06, row_heights=[0.7, 0.3],
            subplot_titles=(f"{symbol} Price & Executions", "PnL"),
        )

        # ----- Row 1: Price + executions
        if not df.empty:
            fig.add_trace(go.Scatter(x=df["ts"], y=df["bid"], name="Bid", mode="lines"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df["ts"], y=df["ask"], name="Ask", mode="lines"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df["ts"], y=df["mid"], name="Mid", mode="lines"), row=1, col=1)

        if df_exec is not None and not df_exec.empty:
            df_buy = df_exec[df_exec["side"].str.upper().isin(["BUY", "BOT"])]
            df_sell = df_exec[df_exec["side"].str.upper().isin(["SELL", "SLD"])]

            if not df_buy.empty:
                fig.add_trace(
                    go.Scatter(
                        x=df_buy["ts"], y=df_buy["price"], mode="markers",
                        name="Buy fills", marker=dict(symbol="triangle-up", size=10),
                        hovertemplate="BUY<br>%{x|%Y-%m-%d %H:%M:%S}<br>px=%{y:.6f}<extra></extra>",
                    ),
                    row=1, col=1
                )
            if not df_sell.empty:
                fig.add_trace(
                    go.Scatter(
                        x=df_sell["ts"], y=df_sell["price"], mode="markers",
                        name="Sell fills", marker=dict(symbol="triangle-down", size=10),
                        hovertemplate="SELL<br>%{x|%Y-%m-%d %H:%M:%S}<br>px=%{y:.6f}<extra></extra>",
                    ),
                    row=1, col=1
                )

        # optional: show most recent limit within window
        # lmt = load_latest_limit_since(symbol, lookback_minutes=lookback_min)
        # if lmt is not None:
        #     fig.add_hline(y=lmt, line_dash="dot", annotation_text=f"Limit {lmt}", row=1, col=1)

        # ----- Row 2: PnL
        if pnl_curve is not None and not pnl_curve.empty:
            fig.add_trace(
                go.Scatter(
                    x=pnl_curve["ts"], y=pnl_curve["pnl"], name="PnL", mode="lines",
                    hovertemplate="PnL=%{y:.2f}<br>%{x|%Y-%m-%d %H:%M:%S}<extra></extra>",
                ),
                row=2, col=1
            )
            # (optional) also show position:
            # fig.add_trace(go.Scatter(x=pnl_curve["ts"], y=pnl_curve["position"], name="Position", mode="lines"), row=2, col=1)

        # ---- layout / axes
        fig.update_layout(
            margin=dict(l=40, r=10, t=40, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0),
        )
        fig.update_yaxes(title_text=f"{symbol} Price", row=1, col=1)
        fig.update_yaxes(title_text="PnL (quote ccy)", row=2, col=1)
        fig.update_xaxes(title_text="Time (UTC)", row=2, col=1)

        # lock both to same visible range from market data (nice initial alignment)
        if not df.empty:
            x0, x1 = df["ts"].min(), df["ts"].max()
            fig.update_xaxes(range=[x0, x1], row=1, col=1)
            fig.update_xaxes(range=[x0, x1], row=2, col=1)

        # ---- Table: PnL per trade (consistent with MTM)
        pnl_rows = []
        if not df.empty and df_exec_raw is not None and not df_exec_raw.empty:
            # Mid at/just before each execution
            mid_for_merge = df[["ts", "mid"]].copy()
            exec_aligned = pd.merge_asof(
                df_exec_raw.sort_values("ts"),
                mid_for_merge.sort_values("ts"),
                on="ts",
                direction="backward"
            ).rename(columns={"mid": "mid_exec"})

            # commissions by exec_id (if any)
            comm_map = df_comm.set_index("exec_id")["commission"].to_dict() if not df_comm.empty else {}
            exec_aligned["commission"] = exec_aligned["exec_id"].map(comm_map).fillna(0.0)

            # exec impact pnl (slippage vs mid), consistent with MTM definition
            sign = exec_aligned["side"].str.upper().map({"BUY": 1.0, "SELL": -1.0})
            exec_aligned["exec_pnl"] = sign * (exec_aligned["mid_exec"] - exec_aligned["price"]) * exec_aligned[
                "qty"]
            exec_aligned["exec_pnl"] -= exec_aligned["commission"].astype(float)

            # sample MTM pnl after each trade time (from pnl_curve)
            if pnl_curve is not None and not pnl_curve.empty:
                mtm_sample = pd.merge_asof(
                    exec_aligned[["ts"]].sort_values("ts"),
                    pnl_curve[["ts", "pnl"]].sort_values("ts"),
                    on="ts", direction="backward"
                )["pnl"]
                exec_aligned["mtm_pnl"] = mtm_sample.values
            else:
                exec_aligned["mtm_pnl"] = None

            # final formatting
            view_cols = ["ts", "side", "qty", "price", "mid_exec", "commission", "exec_pnl", "mtm_pnl"]
            pnl_rows = exec_aligned[view_cols].to_dict("records")

        return fig, pnl_rows

# def register_callbacks(app):
#     @app.callback(
#         Output("tob-graph", "figure"),
#         Output("spread-graph", "figure"),   # we'll reuse this ID for PnL
#         Input("refresh", "n_intervals"),
#         Input("symbol-dropdown", "value"),
#         Input("lookback-min", "value"),
#     )
#     def update_graphs(_, symbol, lookback_min):
#         lookback_min = int(lookback_min or 30)
#
#         # --- load market data
#         df = load_tob_since(symbol, lookback_minutes=lookback_min)
#
#         # --- price figure (TOB)
#         fig_price = go.Figure()
#         if not df.empty:
#             fig_price.add_trace(go.Scatter(x=df["ts"], y=df["bid"], name="Bid", mode="lines"))
#             fig_price.add_trace(go.Scatter(x=df["ts"], y=df["ask"], name="Ask", mode="lines"))
#             fig_price.add_trace(go.Scatter(x=df["ts"], y=df["mid"], name="Mid", mode="lines"))
#
#         # --- executions overlay (always on)
#         df_exec = load_executions_since(symbol, lookback_minutes=lookback_min)
#         if not df_exec.empty:
#             df_buy = df_exec[df_exec["side"].str.upper().isin(["BUY","BOT"])]
#             df_sell = df_exec[df_exec["side"].str.upper().isin(["SELL","SLD"])]
#             if not df_buy.empty:
#                 fig_price.add_trace(go.Scatter(
#                     x=df_buy["ts"], y=df_buy["price"], mode="markers",
#                     name="Buy fills", marker=dict(symbol="triangle-up", size=10),
#                     hovertemplate="BUY<br>%{x|%Y-%m-%d %H:%M:%S}<br>px=%{y:.6f}<extra></extra>",
#                 ))
#             if not df_sell.empty:
#                 fig_price.add_trace(go.Scatter(
#                     x=df_sell["ts"], y=df_sell["price"], mode="markers",
#                     name="Sell fills", marker=dict(symbol="triangle-down", size=10),
#                     hovertemplate="SELL<br>%{x|%Y-%m-%d %H:%M:%S}<br>px=%{y:.6f}<extra></extra>",
#                 ))
#
#         # optional: last limit within window
#         # lmt = load_latest_limit_since(symbol, lookback_minutes=lookback_min)
#         # if lmt is not None:
#         #     fig_price.add_hline(y=lmt, line_dash="dot", annotation_text=f"Limit {lmt}")
#
#         fig_price.update_layout(
#             margin=dict(l=40, r=10, t=30, b=30),
#             xaxis_title="Time (UTC)", yaxis_title=f"{symbol} Price",
#             legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
#         )
#
#         # --- PnL figure (bottom)
#         fig_pnl = go.Figure()
#         if not df.empty:
#             # Build mid-only df for PnL helper
#             df_mid = df[["ts","mid"]].copy()
#             pnl_curve = compute_pnl_curve(df_mid, df_exec if df_exec is not None else df_exec)
#
#             if not pnl_curve.empty:
#                 fig_pnl.add_trace(go.Scatter(
#                     x=pnl_curve["ts"], y=pnl_curve["pnl"], mode="lines",
#                     name="PnL", hovertemplate="PnL=%{y:.2f}<br>%{x|%Y-%m-%d %H:%M:%S}<extra></extra>",
#                 ))
#                 # Optional: overlay position as a secondary series (scaled)
#                 # fig_pnl.add_trace(go.Scatter(x=pnl_curve["ts"], y=pnl_curve["position"], mode="lines", name="Position"))
#
#         fig_pnl.update_layout(
#             margin=dict(l=40, r=10, t=10, b=30),
#             xaxis_title="Time (UTC)", yaxis_title="PnL (quote ccy)",
#             showlegend=False,
#         )
#
#         return fig_price, fig_pnl


# def register_callbacks(app):
#     @app.callback(
#         Output("tob-graph", "figure"),
#         Output("spread-graph", "figure"),
#         Input("refresh", "n_intervals"),
#         Input("symbol-dropdown", "value"),
#         Input("lookback-min", "value"),
#     )
#     def update_graphs(_, symbol, lookback_min):
#         lookback_min = int(lookback_min or 30)
#
#         # --- TOB & derived ---
#         df = load_tob_since(symbol, lookback_minutes=lookback_min)
#
#         fig_price = go.Figure()
#         if not df.empty:
#             fig_price.add_trace(go.Scatter(x=df["ts"], y=df["bid"], name="Bid", mode="lines"))
#             fig_price.add_trace(go.Scatter(x=df["ts"], y=df["ask"], name="Ask", mode="lines"))
#             fig_price.add_trace(go.Scatter(x=df["ts"], y=df["mid"], name="Mid", mode="lines"))
#
#         # --- Executions overlay (always on) ---
#         df_exec = load_executions_since(symbol, lookback_minutes=lookback_min)
#         if not df_exec.empty:
#             df_buy = df_exec[df_exec["side"].str.upper() == "BUY"]
#             df_sell = df_exec[df_exec["side"].str.upper() == "SELL"]
#             if not df_buy.empty:
#                 fig_price.add_trace(go.Scatter(
#                     x=df_buy["ts"], y=df_buy["price"], mode="markers",
#                     name="Buy fills", marker=dict(symbol="triangle-up", size=10),
#                     hovertemplate="BUY<br>%{x|%Y-%m-%d %H:%M:%S}<br>px=%{y:.6f}<extra></extra>",
#                 ))
#             if not df_sell.empty:
#                 fig_price.add_trace(go.Scatter(
#                     x=df_sell["ts"], y=df_sell["price"], mode="markers",
#                     name="Sell fills", marker=dict(symbol="triangle-down", size=10),
#                     hovertemplate="SELL<br>%{x|%Y-%m-%d %H:%M:%S}<br>px=%{y:.6f}<extra></extra>",
#                 ))
#
#         # optional: last limit within window
#         # lmt = load_latest_limit_since(symbol, lookback_minutes=lookback_min)
#         # if lmt is not None:
#         #     fig_price.add_hline(y=lmt, line_dash="dot", annotation_text=f"Limit {lmt}")
#
#         fig_price.update_layout(
#             margin=dict(l=40, r=10, t=30, b=30),
#             xaxis_title="Time (UTC)", yaxis_title=f"{symbol} Price",
#             legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
#         )
#
#         # --- Spread chart (bps) ---
#         fig_spread = go.Figure()
#         if not df.empty:
#             fig_spread.add_trace(go.Scatter(x=df["ts"], y=df["spread_bps"], name="Spread (bps)", mode="lines"))
#         fig_spread.update_layout(
#             margin=dict(l=40, r=10, t=10, b=30),
#             xaxis_title="Time (UTC)", yaxis_title="Spread (bps)", showlegend=False,
#         )
#
#         return fig_price, fig_spread


#
# def register_callbacks(app):
#     @app.callback(
#         Output("tob-graph", "figure"),
#         Output("spread-graph", "figure"),
#         Input("refresh", "n_intervals"),
#         Input("symbol-dropdown", "value"),
#         Input("limit-input", "value"),
#         Input("show-fills", "value"),          # <-- NEW
#         Input("fills-lookback-min", "value"),  # <-- NEW
#     )
#     def update_graphs(_, symbol, limit, show_fills, lookback_min):
#         if not symbol:
#             return go.Figure(), go.Figure()
#         limit = int(limit or 3000)
#         lookback_min = int(lookback_min or 120)
#         df = load_latest_tob(symbol, limit=limit)
#
#         # --- Price figure (TOB) ---
#         f_price = go.Figure()
#         if not df.empty:
#             f_price.add_trace(go.Scatter(x=df["ts"], y=df["bid"], name="Bid", mode="lines"))
#             f_price.add_trace(go.Scatter(x=df["ts"], y=df["ask"], name="Ask", mode="lines"))
#             f_price.add_trace(go.Scatter(x=df["ts"], y=df["mid"], name="Mid", mode="lines"))
#         f_price.update_layout(
#             margin=dict(l=40, r=10, t=30, b=30),
#             xaxis_title="Time (UTC)",
#             yaxis_title=f"{symbol} Price",
#             legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
#         )
#
#         # --- Optional: overlay fills + latest limit price ---
#         if show_fills and "on" in (show_fills or []):
#             df_exec = load_symbol_executions(symbol, lookback_minutes=lookback_min)
#             if not df_exec.empty:
#                 # BUY and SELL markers
#                 df_buy = df_exec[df_exec["side"].str.upper() == "BUY"]
#                 df_sell = df_exec[df_exec["side"].str.upper() == "SELL"]
#
#                 if not df_buy.empty:
#                     f_price.add_trace(go.Scatter(
#                         x=df_buy["ts"], y=df_buy["price"], mode="markers",
#                         name="Buy fills",
#                         marker=dict(symbol="triangle-up", size=10),
#                         hovertemplate="BUY<br>%{x|%Y-%m-%d %H:%M:%S}<br>px=%{y:.5f}<extra></extra>",
#                     ))
#                 if not df_sell.empty:
#                     f_price.add_trace(go.Scatter(
#                         x=df_sell["ts"], y=df_sell["price"], mode="markers",
#                         name="Sell fills",
#                         marker=dict(symbol="triangle-down", size=10),
#                         hovertemplate="SELL<br>%{x|%Y-%m-%d %H:%M:%S}<br>px=%{y:.5f}<extra></extra>",
#                     ))
#
#                 # (Optional) marker size by qty:
#                 # size = 6 + 2 * np.sqrt(df_exec['qty'])
#
#             # (Optional) show latest limit price as a horizontal ref line
#             lmt = load_symbol_latest_limit(symbol)
#             if lmt is not None:
#                 f_price.add_hline(y=lmt, line_dash="dot", annotation_text=f"Last limit {lmt}")
#
#         # --- Spread figure (bps) ---
#         f_spread = go.Figure()
#         if not df.empty:
#             f_spread.add_trace(go.Scatter(x=df["ts"], y=df["spread_bps"], name="Spread (bps)", mode="lines"))
#         f_spread.update_layout(
#             margin=dict(l=40, r=10, t=10, b=30),
#             xaxis_title="Time (UTC)",
#             yaxis_title="Spread (bps)",
#             showlegend=False,
#         )
#
#         return f_price, f_spread
