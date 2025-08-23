from dataclasses import dataclass, field
from threading import RLock


@dataclass
class RuntimeState:
    order_type: str = "LMT"
    max_position: int = 60_000
    send_order: bool = False

    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def set(self, **kwargs):
        with self._lock:
            for k, v in kwargs.items():
                if hasattr(self, k):
                    setattr(self, k, v)

    def get_snapshot(self):
        with self._lock:
            return {
                "order_type": self.order_type,
                "max_position": self.max_position,
                "send_order": self.send_order,
            }


class FrozenRuntimeState:
    def __init__(self, order_type="LMT", max_position=60_000, send_order=False):
        self._snap = {
            "order_type": order_type,
            "max_position": max_position,
            "send_order": send_order,
        }

    def get_snapshot(self):
        # same interface as RuntimeState
        return dict(self._snap)

    def set(self, **kwargs):
        # either ignore or make it strict:
        # raise RuntimeError("State is frozen in backtests")
        return
