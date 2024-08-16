"""Dataset that returns a range of integers."""

from loadax.dataset.protocol import Dataset


class RangeDataset(Dataset[int]):
    """Dataset that returns a range of integers.

    This dataset is useful for testing and debugging.

    Attributes:
        start: The start of the range.
        end: The end of the range.
        step: The step size of the range.
    """

    def __init__(self, start: int, end: int, step: int = 1):
        """Dataset that returns a range of integers.

        This dataset is useful for testing and debugging.

        Args:
            start: The start of the range.
            end: The end of the range.
            step: The step size of the range.
        """
        self.start = start
        self.end = end
        self.step = step

        if self.start >= self.end:
            raise ValueError("start must be less than end")
        if self.step <= 0:
            raise ValueError("step must be greater than 0")

    def get(self, index: int) -> int | None:
        """Get the item at the given index.

        Args:
            index: The index of the item to get.

        Returns:
            The item at the given index, or None if the index is out of range.
        """
        if index < 0:
            index = len(self) + index

        if 0 <= index < len(self):
            return self.start + index * self.step

        return None

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns:
            The length of the dataset.
        """
        return (self.end - self.start + self.step - 1) // self.step

    def __repr__(self) -> str:
        """Get a string representation of the dataset.

        Returns:
            A string representation of the dataset.
        """
        return f"RangeDataset(start={self.start}, end={self.end}, step={self.step})"
