import logging


class PositionHandler:
    def __init__(self):
        self.positions = {}
        logging.basicConfig(level=logging.INFO)

    def process_position(self, position):
        logging.info(f"Process position: {position}")
