# order_utils_fx.py
from ib_async import MarketOrder, LimitOrder


def _ensure_px(px: float | None, ctx_hint: str):
    if px is None or px <= 0:
        raise ValueError(f"{ctx_hint}: need a positive reference price")
    return float(px)


def _qty_from_notional_usd(usd_notional: float, ref_price: float) -> int:
    # EURUSD: price is USD per 1 EUR → qty (in EUR) = USD / (USD/EUR)
    return int(abs(usd_notional / ref_price))


def notional_usd_from_qty(qty: float, ref_price: float) -> float:
    return qty * ref_price


def create_fx_order(
        side: str,
        *,
        order_type: str = "MKT",
        qty: float | None = None,  # base units (EUR)
        usd_notional: float | None = None,  # quote units (USD)
        ref_price: float | None = None,  # use last/mid for MKT; limit_price for LMT if not provided
        limit_price: float | None = None,
        tif: str = "DAY",
) -> "Order":
    """
    Build an FX order (EURUSD) supporting USD notional via cashQty.
    - If usd_notional is given, totalQuantity is computed as usd_notional / ref_price (EUR),
      and order.cashQty is set to usd_notional (USD).
    - Otherwise, qty (in EUR) is used directly.
    """
    side = side.upper()
    if side not in ("BUY", "SELL"):
        raise ValueError("side must be BUY or SELL")

    ot = order_type.upper()
    if usd_notional is not None:
        # choose a reference price
        px = limit_price if (ot == "LMT" and limit_price is not None) else ref_price
        px = _ensure_px(px, "USD notional order")
        calc_qty = _qty_from_notional_usd(usd_notional, px)

        if ot == "MKT":
            o = MarketOrder(side, calc_qty)
        elif ot == "LMT":
            lp = _ensure_px(limit_price, "Limit order")
            o = LimitOrder(side, calc_qty, lp)
        else:
            raise ValueError(f"Unsupported order_type: {order_type}")

        o.tif = tif
        return o

    # No usd_notional: use qty directly (in base units)
    # if qty is None or qty <= 0:
    #     raise ValueError("Provide either usd_notional or qty > 0")
    #
    # if ot == "MKT":
    #     o = MarketOrder(side, float(qty))
    # elif ot == "LMT":
    #     lp = _ensure_px(limit_price, "Limit order")
    #     o = LimitOrder(side, float(qty), lp)
    # else:
    #     raise ValueError(f"Unsupported order_type: {order_type}")
    # o.tif = tif
    # return o
