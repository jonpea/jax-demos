from typing import TypeAlias, TypedDict

import jax
import numpy as np
from numpy.typing import NDArray
from PIL.Image import Image

ImageDType: TypeAlias = np.float32
LabelDType: TypeAlias = np.int32

type BatchedExamples = tuple[JaxArray, JaxArray]  # (images, labels)
type JaxArray = jax.Array


class RawBatch(TypedDict):
    """Raw labeled data (examples)."""

    image: list[Image]
    label: list[int]


class PreprocessedBatch(TypedDict):
    """Labeled data (examples) after batching/preprocessing."""

    image: NDArray[ImageDType]
    label: NDArray[LabelDType]
