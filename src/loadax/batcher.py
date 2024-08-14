from collections.abc import Callable
from typing import Generic, Protocol
from typing import TypeVar

T = TypeVar("T")
P = TypeVar("P")


class Batcher(Protocol, Generic[T, P]):
    def __init__(self, batch_fn: Callable[[list[T]], P], device: str | None = None):
        self.batch_fn = batch_fn
        self.device = device

    def batch(self, data: list[T]) -> P:
        return self.batch_fn(data)

    def __call__(self, data: list[T]) -> P:
        return self.batch(data)
