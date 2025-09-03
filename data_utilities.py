import queue
import threading
from dataclasses import dataclass, field
from typing import Final, Iterator, TypeVar, cast

import jax
import jax.numpy as jnp
import numpy as np
from datasets import IterableDataset
from numpy.typing import NDArray

from custom_types import BatchedExamples, RawBatch, JaxArray, PreprocessedBatch, ImageDType, LabelDType


T = TypeVar("T")
_SENTINEL: Final = object()


def prefetch(batches: Iterator[T], size: int = 2) -> Iterator[T]:
    """Prefetch a few batches to device on a background thread."""

    if size <= 0:
        # No pre-fetch
        yield from batches
        return

    q: queue.Queue[T | object | Exception] = queue.Queue(maxsize=size)
    thread_done = threading.Event()

    def prefetch_worker() -> None:
        try:
            for b in batches:
                if thread_done.is_set():
                    break
                q.put(jax.device_put(b))
            q.put(_SENTINEL)
        except Exception as e:
            q.put(e)
        finally:
            thread_done.set()

    thread = threading.Thread(target=prefetch_worker, daemon=True)
    thread.start()

    try:
        while True:
            item = q.get()
            if isinstance(item, Exception):
                raise item
            if item is _SENTINEL:
                break
            yield cast(T, item)
    finally:
        thread_done.set()
        thread.join(timeout=1.0)


def make_huggingface_iterator(
    dataset: IterableDataset,
    shuffle: bool,  # training: True;  evaluation: False
    *,
    batch_size: int,
    data_key: JaxArray | None = None,
    shuffle_buffer_size: int | None = None,
) -> Iterator[BatchedExamples]:
    """Optimized pipeline with batched preprocessing."""

    if shuffle:
        if data_key is None:
            raise ValueError(
                "A JAX PRNGKey `data_key` must be provided when `shuffle=True`"
            )
        if shuffle_buffer_size is None:
            raise ValueError(
                "An integer `shuffle_buffer_size` must be provided when `shuffle=True`"
            )
        huggingface_seed: int = int(jax.random.randint(data_key, (), 0, 2**31 - 1))
        dataset = dataset.shuffle(
            seed=huggingface_seed, buffer_size=shuffle_buffer_size
        )

    def preprocess(batch: RawBatch) -> PreprocessedBatch:
        """Vectorized preprocessing for a batch of images and labels."""
        # fmt: off
        raw_images = np.stack([np.array(img) for img in batch["image"]])
        scale: ImageDType = ImageDType(2 / np.iinfo(raw_images.dtype).max)
        images: NDArray[ImageDType] = raw_images.astype(ImageDType) * scale - ImageDType(1)
        # fmt: on
        return {
            "image": images.reshape(images.shape[0], -1),
            "label": np.array(batch["label"], dtype=LabelDType),
        }

    # Apply preprocessing and set the format to "jax"
    dataset = dataset.map(preprocess, batched=True, batch_size=batch_size).with_format(
        "numpy"
    )

    for batch in dataset.iter(batch_size=batch_size):
        xb = jnp.asarray(batch["image"], dtype=jnp.float32)
        yb = jnp.asarray(batch["label"], dtype=jnp.int32)
        yield (xb, yb)


@dataclass
class Averaged:
    aggregate: float = 0.0
    count: int = 0

    def update(self, x: float, batch_size: int) -> None:
        self.aggregate += x * batch_size
        self.count += batch_size

    @property
    def mean(self) -> float:
        return np.nan if self.count == 0 else self.aggregate / self.count


@dataclass
class Metrics:
    loss: Averaged = field(default_factory=Averaged)
    accuracy: Averaged = field(default_factory=Averaged)

    def update(self, loss, accuracy, batch_size: int) -> None:
        self.loss.update(float(loss), batch_size)
        self.accuracy.update(float(accuracy), batch_size)
