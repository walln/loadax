from abc import abstractmethod
from typing import Generic, Protocol

from jax import Array

from loadax.experimental.dataset.dataset import Dataset, Example


class Shuffleable(Protocol, Generic[Example]):
    """The shuffled dataset protocol.

    Any dataset that can be shuffled must implement the shuffleable protocol.
    Not all datasets can be shuffled, for example an iterable dataset that does
    not have a known size cannot be shuffled without a technique such as era-based
    permutation. To reduce the complexity, loadax datasets that do not implement
    shufflable cannot be shuffled.
    """

    @abstractmethod
    def shuffle(self, seed: Array) -> "Dataset[Example]":
        """Shuffle the dataset.

        Args:
            seed: The seed to use for the shuffle. This is a jax
            PRNGKey as all randomization in loadax is implemented using jax.random.

        Returns:
            The shuffled dataset.
        """
        pass
