import logging
from ib_insync import Order, MarketOrder

class OrderHandler:
    def __init__(self, ib_connection):
        self.ib_connection = ib_connection
        logging.basicConfig(level=logging.INFO)

    def create_order(self, symbol, qty, order_type='MKT'):
        if order_type == 'MKT':
            order = MarketOrder(symbol, qty)
        else:
            # Implement other order types if needed
            raise NotImplementedError("Order type not supported")
        return order

    def send_order(self, contract, order):
        logging.info(f"Sending order: {order}")
        self.ib_connection.ib.placeOrder(contract, order)

    def cancel_order(self, order_id):
        logging.info(f"Cancelling order ID: {order_id}")
        self.ib_connection.ib.cancelOrder(order_id)

    def check_order_status(self, order_id):
        order = self.ib_connection.ib.openOrders()
        status = next((o for o in order if o.orderId == order_id), None)
        logging.info(f"Order status: {status}")
        return status