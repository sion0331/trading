from brokerage.ib_connection import IBConnection
from marketData.contracts import EURUSD, BTC
from marketData.market_data import MarketDataHandler
from orders.order_manager import OrderHandler
from position.position_manager import PositionHandler
from strategy.simple import SimpleStrategy
from trades.trade_manager import TradeHandler
from utils.config_loader import load_config


def main():
    config = load_config()

    ib_host = config['api']['ib_host']
    ib_port = config['api']['ib_port']
    ib_client_id = config['api']['ib_client_id']

    ib_conn = IBConnection(ib_host, ib_port, ib_client_id)

    market_data_handler = MarketDataHandler(ib_conn)
    ib_conn.ib.pendingTickersEvent += market_data_handler.handle_msg  ## todo check
    trade_handler = TradeHandler(ib_conn)
    position_handler = PositionHandler(ib_conn)
    order_handler = OrderHandler(ib_conn)

    # market_data_handler.create_events()
    trade_handler.create_events()
    position_handler.create_events()
    order_handler.create_events()

    # TODO GPT - Setup strategy
    strategy = SimpleStrategy(BTC, order_handler, position_handler)
    market_data_handler.add_subscriber(strategy.on_market_data)

    ib_conn.connect()

    # TODO GPT - Subscribe to data
    market_data_handler.subscribe(BTC)

    # trade_handler.load_trades()
    position_handler.log_portfolio()
    # ib_conn.subscribe_market_data(NFLX)
    # order = order_handler.create_order('SELL', 100)
    # order_handler.send_order(GOOG, order)

    ib_conn.run()


if __name__ == '__main__':
    main()
