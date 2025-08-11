import logging

from ib_async import MarketOrder

from database.orders import OrderLogger


class OrderHandler:
    def __init__(self, ib_connection):
        self.ib_connection = ib_connection
        logging.basicConfig(level=logging.INFO)
        self.logger = OrderLogger(ib_connection.ib, db_path="./data/db/orders.db")

    def create_events(self):
        self.ib_connection.ib.orderStatusEvent += self.order_status
        self.ib_connection.ib.newOrderEvent += self.new_order

    def order_status(self, status):
        print(f"### Order Status: {status}")

    def new_order(self, order):
        print(f"### New Order: {order}")

    ###
    def create_order(self, side, qty, order_type='MKT'):
        if order_type == 'MKT':
            order = MarketOrder(side, qty)
        else:
            # Implement other order types if needed
            raise NotImplementedError("Order type not supported")
        return order

    def send_order(self, contract, order):
        logging.info(f"Sending order: {order}")
        self.logger.log_send_intent(contract, order)
        self.ib_connection.ib.placeOrder(contract, order)

    def cancel_order(self, order_id):
        logging.info(f"Cancelling order ID: {order_id}")
        self.logger.log_cancel_intent(order_id)
        self.ib_connection.ib.cancelOrder(order_id)

    def check_order_status(self, order_id):
        order = self.ib_connection.ib.openOrders()
        status = next((o for o in order if o.orderId == order_id), None)
        logging.info(f"Order status: {status}")
        return status
