"""Format-independent physical storage layout descriptors."""

from dataclasses import dataclass
from typing import Iterable, Mapping, Tuple


LAYOUT_NAMES_KEY = "nn.storage_layout.names"
LAYOUT_OFFSETS_KEY = "nn.storage_layout.offsets"
LAYOUT_AXES_KEY = "nn.storage_layout.axes"


@dataclass(frozen=True)
class Layout:
    axes: Tuple[int, ...] = ()

    @classmethod
    def identity(cls) -> "Layout":
        return cls()

    @classmethod
    def permuted(cls, axes: Iterable[int]) -> "Layout":
        permutation = tuple(int(axis) for axis in axes)
        if not permutation or sorted(permutation) != list(range(len(permutation))):
            raise ValueError(f"Invalid axis permutation: {list(permutation)}")
        return cls(permutation)

    @property
    def is_identity(self) -> bool:
        return not self.axes or self.axes == tuple(range(len(self.axes)))


def write_layout_metadata(writer, layouts: Mapping[str, Layout]) -> None:
    non_identity = [(name, layout) for name, layout in sorted(layouts.items()) if not layout.is_identity]
    if not non_identity:
        return

    names = []
    offsets = [0]
    axes = []
    for name, layout in non_identity:
        if not name:
            raise ValueError("Layout tensor name cannot be empty")
        names.append(name)
        axes.extend(layout.axes)
        offsets.append(len(axes))

    writer.add_array(LAYOUT_NAMES_KEY, names)
    writer.add_array(LAYOUT_OFFSETS_KEY, offsets)
    writer.add_array(LAYOUT_AXES_KEY, axes)
