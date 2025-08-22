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
