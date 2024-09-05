from datetime import datetime

class Trade:
    def __init__(self, ts, exec_id, symbol, exchange,
                 side, shares, price, perm_id, client_id, order_id, liquidation, cum_qty, avg_price, order_ref, ev_rule
                 , ev_multiplier, model_code, last_liquidity, pending_price_revision, commission, commission_currency,
                 realized_pnl, yield_, yield_redemption_date):
        self.ts = ts
        self.execId = exec_id
        self.symbol = symbol
        self.exchange = exchange
        self.side = side  # todo - SLD -> mapping
        self.shares = shares
        self.price = price
        self.permId = perm_id
        self.clientId = client_id
        self.orderId = order_id
        self.liquidation = liquidation
        self.cumQty = cum_qty
        self.avgPrice = avg_price
        self.orderRef = order_ref
        self.evRule = ev_rule
        self.evMultiplier = ev_multiplier
        self.modelCode = model_code
        self.lastLiquidity = last_liquidity
        self.pendingPriceRevision = pending_price_revision
        self.commission = commission
        self.commissionCurrency = commission_currency
        self.realizedPNL = realized_pnl
        self.yield_ = yield_
        self.yieldRedemptionDate = yield_redemption_date
        print(self.ts)

    # def __hash__(self):
    #     return hash((self.ts, self.execId, self.symbol, self.side, self.shares, self.price, self.commission, self.realizedPNL))
    #
    # def __eq__(self, other):
    #     # if not isinstance(other, type(self)):
    #     # print(f'{self.ts==other.ts} | {self.ts} <> {other.ts}')
    #     print(f'{self.execId == other.execId} | {self.execId} <> {other.execId}')
    #     print(f'{self.symbol == other.symbol} | {self.symbol} <> {other.symbol}')
    #     print(f'{self.side == other.side} | {self.side} <> {other.side}')
    #     print(f'{self.shares == other.shares} | {self.shares} <> {other.shares}')
    #
    #     # return (self.execId== other.execId)
    #
    #     return ((self.execId, self.symbol, self.side, self.shares, self.price, self.commission, self.realizedPNL)
    #             == (other.execId, other.symbol, other.side, other.shares, other.price, other.commission, other.realizedPNL))

    # return ((self.ts, self.execId, self.symbol, self.side, self.shares, self.price, self.commission, self.realizedPNL)
    #         == (other.ts, other.execId, other.symbol, other.side, other.shares, other.price, other.commission, other.realizedPNL))

    @classmethod
    def from_ib(cls, fill):
        return cls(fill.execution.time, fill.execution.execId, fill.contract.symbol, fill.execution.exchange,
                   fill.execution.side, fill.execution.shares, fill.execution.price, fill.execution.permId,
                   fill.execution.clientId, fill.execution.orderId, fill.execution.liquidation, fill.execution.cumQty,
                   fill.execution.avgPrice, fill.execution.orderRef, fill.execution.evRule, fill.execution.evMultiplier,
                   fill.execution.modelCode, fill.execution.lastLiquidity, fill.execution.pendingPriceRevision,
                   fill.commissionReport.commission, fill.commissionReport.currency, fill.commissionReport.realizedPNL,
                   fill.commissionReport.yield_, fill.commissionReport.yieldRedemptionDate)

    @classmethod
    def from_data(cls, trade):
        return cls(datetime.strptime(trade['ts'], '%Y-%m-%d %H:%M:%S+00:00'), trade['execId'], trade['symbol'], trade['exchange'], trade['side'],
                   float(trade['shares']), float(trade['price']), int(trade['permId']), int(trade['clientId']),
                   int(trade['orderId']),
                   int(trade['liquidation']), float(trade['cumQty']), float(trade['avgPrice']), trade['orderRef'],
                   trade['evRule'],
                   float(trade['evMultiplier']), trade['modelCode'], float(trade['lastLiquidity']),
                   bool(trade['pendingPriceRevision']),
                   float(trade['commission']), trade['commissionCurrency'], float(trade['realizedPNL']),
                   float(trade['yield_']), trade['yieldRedemptionDate'])

    def to_csv(self):
        return [self.ts, self.execId, self.symbol, self.exchange, self.side, self.shares, self.price, self.permId,
                self.clientId, self.orderId, self.liquidation, self.cumQty, self.avgPrice, self.orderRef, self.evRule,
                self.evMultiplier, self.modelCode, self.lastLiquidity, self.pendingPriceRevision, self.commission,
                self.commissionCurrency, self.realizedPNL, self.yield_, self.yieldRedemptionDate]

    def log(self):
        print(
            f'{self.symbol} | {self.side} {self.shares} @ {self.price} | {self.realizedPNL} {self.commission} | {self.ts}')

    def __str__(self):
        return f'{self.symbol} | {self.side} {self.shares} @ {self.price} | {self.realizedPNL} {self.commission} | {self.ts}'
