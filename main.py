from brokerage.ib_connection import IBConnection
from marketData.contracts import EURUSD
from marketData.market_data import MarketDataHandler
from orders.order_manager import OrderHandler
from position.position_manager import PositionHandler
from strategy.simple import SimpleStrategy
from trades.trade_manager import TradeHandler
from utils.config_loader import load_config


def main():
    send_order = False
    contract = EURUSD
    max_position = 60_000
    order_type = "MKT"

    config = load_config()

    ib_host = config['api']['ib_host']
    ib_port = config['api']['ib_port']
    ib_client_id = config['api']['ib_client_id']

    ib_conn = IBConnection(ib_host, ib_port, ib_client_id)

    market_data_handler = MarketDataHandler(ib_conn, config['db']['path'])
    trade_handler = TradeHandler(ib_conn)
    position_handler = PositionHandler(ib_conn)
    order_handler = OrderHandler(ib_conn)
    strategy = SimpleStrategy(contract, order_handler, position_handler, send_order=send_order,
                              max_position=max_position, order_type=order_type)

    market_data_handler.create_events()
    trade_handler.create_events()
    position_handler.create_events()
    order_handler.create_events()

    market_data_handler.add_callback(strategy.on_market_data)

    ib_conn.connect()

    market_data_handler.subscribe(contract)
    # trade_handler.load_trades()
    position_handler.log_portfolio()

    ib_conn.run()


if __name__ == '__main__':
    main()
