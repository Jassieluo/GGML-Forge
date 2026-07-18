"""Checked access to provider source state dictionaries."""

import re
from typing import Dict, Iterable, Optional, Pattern, Union


class Checkpoint:
    def __init__(self, state_dict: Dict[str, object]):
        self._state = state_dict
        self._consumed = set()
        self._ignored = set()

    def take(self, name: str):
        if name not in self._state:
            raise KeyError(f"Required checkpoint tensor is missing: {name}")
        if name in self._consumed:
            raise ValueError(f"Checkpoint tensor was consumed twice: {name}")
        self._consumed.add(name)
        return self._state[name]

    def optional(self, name: str):
        return self.take(name) if name in self._state else None

    def ignore(self, pattern: Union[str, Pattern[str]]) -> None:
        expression = re.compile(pattern) if isinstance(pattern, str) else pattern
        self._ignored.update(name for name in self._state if expression.fullmatch(name))

    def finish(self, allow_unconsumed: Iterable[str] = ()) -> None:
        allowed = set(allow_unconsumed)
        remaining = sorted(set(self._state) - self._consumed - self._ignored - allowed)
        if remaining:
            preview = ", ".join(remaining[:20])
            suffix = " ..." if len(remaining) > 20 else ""
            raise ValueError(f"Provider left {len(remaining)} checkpoint tensors unmapped: {preview}{suffix}")
