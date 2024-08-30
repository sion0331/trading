from brokerage.ib_connection import IBConnection
from marketData.contracts import EURUSD
from marketData.market_data import MarketDataHandler
from orders.order_manager import OrderHandler
from position.position_manager import PositionHandler
from trades.trade_manager import TradeHandler
from utils.config_loader import load_config


def main():
    config = load_config()

    ib_host = config['api']['ib_host']
    ib_port = config['api']['ib_port']
    ib_client_id = config['api']['ib_client_id']

    market_data_handler = MarketDataHandler()
    trade_handler = TradeHandler()
    position_handler = PositionHandler()
    # order_handler = OrderHandler(ib_conn)

    ib_conn = IBConnection(ib_host, ib_port, ib_client_id, market_data_handler, trade_handler, position_handler)
    ib_conn.connect()

    ib_conn.subscribe_market_data(EURUSD)
    # ib_conn.subscribe_trades(trade_handler.on_trade)

    # contract = NFLX
    # order = order_manager.create_order(symbol='NFLX', qty=10)
    # order_manager.send_order(contract, order)

    ib_conn.run()


if __name__ == '__main__':
    main()
