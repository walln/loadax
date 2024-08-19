"""Progress metadata for a dataloader."""

from dataclasses import dataclass


@dataclass
class Progress:
    """Progress metadata for a dataloader.

    This metadata indicates how far the dataloader has progressed. This is useful for
    debugging and monitoring the progress of a dataloader.

    Attributes:
        items_processed: The number of items processed so far.
        items_total: The total number of items in the dataset.
    """

    items_processed: int
    items_total: int

    def __repr__(self) -> str:
        """Get a string representation of the progress.

        Returns:
            A string representation of the progress.
        """
        return f"Progress(items_processed={self.items_processed}, items_total={self.items_total})"  # noqa: E501
