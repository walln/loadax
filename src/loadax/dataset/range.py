from loadax.dataset.dataset import Dataset


class RangeDataset(Dataset[int]):
    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end

        if self.start >= self.end:
            raise ValueError("start must be less than end")

    def get(self, index: int) -> int | None:
        if index < 0:
            index = len(self) + index

        if 0 <= index < len(self):
            return self.start + index

        return None

    def __len__(self) -> int:
        return self.end - self.start

    def __repr__(self) -> str:
        return f"RangeDataset(start={self.start}, end={self.end})"
