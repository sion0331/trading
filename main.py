from brokerage.ib_connection import IBConnection
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

    ib_conn = IBConnection(ib_host, ib_port, ib_client_id)
    market_data_handler = MarketDataHandler(ib_conn)
    trade_handler = TradeHandler(ib_conn)
    position_handler = PositionHandler(ib_conn)
    order_handler = OrderHandler(ib_conn)

    market_data_handler.create_events()
    trade_handler.create_events()
    position_handler.create_events()
    order_handler.create_events()

    ib_conn.connect()

    position_handler.log_portfolio()
    trade_handler.load_trades()
    # ib_conn.subscribe_market_data(NFLX)

    # order = order_handler.create_order('BUY', 100)
    # order_handler.send_order(NFLX, order)

    ib_conn.run()


if __name__ == '__main__':
    main()
