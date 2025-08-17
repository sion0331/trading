from typing import Callable, List


class Event:
    """
    Minimal event multicaster: supports +=, -=, and emit(...)
    matching how you're using ib_async events.
    """

    def __init__(self, name: str):
        self._name = name
        self._subs: List[Callable] = []

    def __iadd__(self, func: Callable):
        self._subs.append(func)
        return self

    def __isub__(self, func: Callable):
        self._subs = [f for f in self._subs if f is not func]
        return self

    def emit(self, *args, **kwargs):
        for f in list(self._subs):
            try:
                f(*args, **kwargs)
            except Exception:
                # keep the sim running even if a listener fails
                import traceback;
                traceback.print_exc()
