"""Common structural validation for exported GGUF artifacts."""

from collections import Counter
from dataclasses import dataclass
from typing import FrozenSet, Iterable

from gguf import GGUFReader

from .artifact import GGML_MAX_NAME_BYTES


@dataclass(frozen=True)
class ArtifactContract:
    required_metadata: FrozenSet[str] = frozenset()
    required_tensors: FrozenSet[str] = frozenset()

    @classmethod
    def create(cls, metadata: Iterable[str] = (), tensors: Iterable[str] = ()):
        return cls(frozenset(metadata), frozenset(tensors))


def validate_artifact(path: str, contract: ArtifactContract) -> None:
    reader = GGUFReader(path)
    metadata = set(reader.fields)
    tensor_names = [tensor.name for tensor in reader.tensors]
    duplicate_tensors = sorted(name for name, count in Counter(tensor_names).items() if count > 1)
    oversized_tensors = sorted(
        name for name in tensor_names if len(name.encode("utf-8")) > GGML_MAX_NAME_BYTES
    )
    missing_metadata = sorted(contract.required_metadata - metadata)
    missing_tensors = sorted(contract.required_tensors - set(tensor_names))
    if missing_metadata or missing_tensors or duplicate_tensors or oversized_tensors:
        raise ValueError(
            "GGUF artifact validation failed; "
            f"missing metadata={missing_metadata}, missing tensors={missing_tensors}, "
            f"duplicate tensors={duplicate_tensors}, oversized tensors={oversized_tensors}"
        )
