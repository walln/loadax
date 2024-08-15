from loadax.dataset.protocol import Dataset


class RangeDataset(Dataset[int]):
    def __init__(self, start: int, end: int, step: int = 1):
        self.start = start
        self.end = end
        self.step = step

        if self.start >= self.end:
            raise ValueError("start must be less than end")
        if self.step <= 0:
            raise ValueError("step must be greater than 0")

    def get(self, index: int) -> int | None:
        if index < 0:
            index = len(self) + index

        if 0 <= index < len(self):
            return self.start + index * self.step

        return None

    def __len__(self) -> int:
        return (self.end - self.start + self.step - 1) // self.step

    def __repr__(self) -> str:
        return f"RangeDataset(start={self.start}, end={self.end}, step={self.step})"
