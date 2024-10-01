from typing import Any, Literal, TypeAlias, TypeVar

T = TypeVar("T")


class RequiredFieldValue:
    """A sentinel value to indicate that a field is required."""

    def __deepcopy__(self, _memo: Any) -> "RequiredFieldValue":
        return self

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> Literal["REQUIRED"]:
        return "REQUIRED"


REQUIRED = RequiredFieldValue()
Required: TypeAlias = RequiredFieldValue | T
