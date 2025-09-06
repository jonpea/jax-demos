import enum
from typing import Any, Literal, Protocol, TypeAlias, TypedDict

import jax
import numpy as np
from numpy.typing import NDArray
from PIL.Image import Image

ImageDType: TypeAlias = np.float32
LabelDType: TypeAlias = np.int32

type JaxArray = jax.Array
type JaxScalar = jax.Array  # scalar array with shape ()
type PyTree = Any
type Split = Literal["train", "test"]

type BatchedExamples = tuple[JaxArray, JaxArray]  # (images, labels)
type Metrics = tuple[JaxScalar, JaxScalar]  # (loss, accuracy)


class ApplyFn(Protocol):
    def __call__(self, vars: PyTree, x: JaxArray) -> JaxArray: ...


class RawBatch(TypedDict):
    """Raw labeled data (examples)."""

    image: list[Image]
    label: list[int]


class PreprocessedBatch(TypedDict):
    """Labeled data (examples) after batching/preprocessing."""

    image: NDArray[ImageDType]
    label: NDArray[LabelDType]


class Splits(enum.Enum):
    """Dataset splits associated with Hugging Face string literals."""

    TRAINING = "train"
    TESTING = "test"
    VALIDATION = "validation"
