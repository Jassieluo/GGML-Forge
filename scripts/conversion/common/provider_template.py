"""Copyable skeleton for a new model provider."""

from . import ModelDefinition, ModelProvider, Parameter
from .checkpoint import Checkpoint


class ExampleProvider(ModelProvider):
    def __init__(self, source_path: str):
        self.source_path = source_path

    def load_source(self):
        raise NotImplementedError("Load the framework-specific checkpoint here")

    def build(self, source) -> ModelDefinition:
        checkpoint = Checkpoint(source)
        model = ModelDefinition("example_model")
        model.string("general.name", "Example model")

        weight = checkpoint.take("source.projection.weight")
        model.parameter(Parameter("projection.weight", weight.detach().cpu().numpy()))

        checkpoint.finish()
        return model
