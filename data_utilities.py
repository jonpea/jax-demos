import queue
import threading
from dataclasses import dataclass, field
from typing import Final, Iterator, TypeVar, cast

import jax
import jax.numpy as jnp
import numpy as np
from datasets import IterableDataset
from numpy.typing import NDArray

from custom_types import (
    BatchedExamples,
    ImageDType,
    JaxArray,
    LabelDType,
    PreprocessedBatch,
    RawBatch,
    RawExample,
)


T = TypeVar("T")
_SENTINEL: Final = object()


def prefetch(batches: Iterator[T], size: int = 2, timeout: float = 10.0) -> Iterator[T]:
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
            item = q.get(timeout=timeout)
            if isinstance(item, Exception):
                raise item
            if item is _SENTINEL:
                break
            yield cast(T, item)
    finally:
        thread_done.set()
        thread.join()  # blocking (no timeout)


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

    num_examples_counter: int = 0

    def preprocess(batch: RawBatch) -> PreprocessedBatch:
        """Vectorized preprocessing for a batch of images and labels."""
        # fmt: off
        raw_images = np.stack([np.array(img) for img in batch["image"]])
        scale: ImageDType = ImageDType(2 / np.iinfo(raw_images.dtype).max)
        images: NDArray[ImageDType] = raw_images.astype(ImageDType) * scale - ImageDType(1)
        this_batch_size = images.shape[0]
        nonlocal num_examples_counter
        num_examples_counter += this_batch_size
        assert 0 < this_batch_size <= batch_size  # final batch might be smaller
        # fmt: on
        return {
            "image": images.reshape(this_batch_size, -1),
            "label": np.array(batch["label"], dtype=LabelDType),
        }

    # Apply preprocessing; using fomat "numpy" (and later converting)
    # appears to be more efficient than using format "jax" directly(?)
    dataset = dataset.map(
        preprocess,
        batched=True,
        batch_size=batch_size,
    ).with_format("numpy")

    for i, batch in enumerate(dataset.iter(batch_size=batch_size)):
        xb = jnp.asarray(batch["image"], dtype=jnp.float32)
        yb = jnp.asarray(batch["label"], dtype=jnp.int32)
        yield (xb, yb)
    else:
        assert i == num_examples_counter // batch_size

    return


@dataclass
class RunningAverage:
    """Maintains a running average."""

    aggregate: float = 0.0
    count: int = 0

    def update(self, x: float, batch_size: int) -> None:
        self.aggregate += x * batch_size
        self.count += batch_size

    @property
    def mean(self) -> float:
        return np.nan if self.count == 0 else self.aggregate / self.count


@dataclass
class RunningMetrics:
    loss: RunningAverage = field(default_factory=RunningAverage)
    accuracy: RunningAverage = field(default_factory=RunningAverage)

    def update(self, loss, accuracy, batch_size: int) -> None:
        self.loss.update(float(loss), batch_size)
        self.accuracy.update(float(accuracy), batch_size)


def peek(dataset: IterableDataset) -> RawExample:
    """Peeks the first value in the lazy dataset."""
    # "creates a *new IterableDataset* with only the first n elements"
    # https://huggingface.co/docs/datasets/v4.0.0/en/package_reference/main_classes#datasets.IterableDataset.take
    first = next(iter(dataset.take(1)))
    return cast(RawExample, first)
