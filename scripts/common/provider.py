"""Minimal interface implemented by source-checkpoint providers."""

from abc import ABC, abstractmethod
from typing import Any

from .schema import ModelDefinition


class ModelProvider(ABC):
    @abstractmethod
    def load_source(self) -> Any:
        """Load provider-specific source artifacts."""

    @abstractmethod
    def build(self, source: Any) -> ModelDefinition:
        """Return canonical parameters and machine-independent metadata."""

    def definition(self) -> ModelDefinition:
        definition = self.build(self.load_source())
        if not definition.parameters:
            raise ValueError("Provider produced no model parameters")
        return definition
