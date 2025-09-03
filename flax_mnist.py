import argparse
import functools
import logging
import queue
import shutil
import sys
import threading

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final, Iterator, Literal, Self, TypedDict, cast

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp

from datasets import load_dataset, IterableDataset, IterableDatasetDict
from flax.training import train_state
from numpy.typing import NDArray
from PIL.Image import Image

# Setup logging
logging.getLogger("absl").setLevel(logging.WARNING)
logging.getLogger("jax").setLevel(logging.WARNING)
logging.getLogger("orbax").setLevel(logging.WARNING)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(message)s"))
logger = logging.getLogger(__name__)
if logger.hasHandlers():
    logger.handlers.clear()
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Type aliases (PEP 695)
type JaxArray = jax.Array
type Split = Literal["train", "test"]
type BatchedExamples = tuple[JaxArray, JaxArray]  # (images, labels)


class RawBatch(TypedDict):
    """Raw labeled data (examples)."""

    image: list[Image]
    label: list[int]


ImageDType: Final = np.float32
LabelDType: Final = np.int32


class PreprocessedBatch(TypedDict):
    """Labeled data (examples) after batching/preprocessing."""

    image: NDArray[ImageDType]
    label: NDArray[LabelDType]


@dataclass(frozen=True)
class Configuration:
    """Learning configuration parameters."""

    batch_size: int = 128
    checkpoints_dir: Path = Path("./checkpoints")
    clean_start: bool = False
    early_stopping_patience: int = 3
    epochs: int = 5
    eval_every_n_epochs: int = 1
    learning_rate: float = 1e-3
    prefetch: int = 2
    seed: int = 0
    shuffle_buffer_size: int = 50_000

    def __post_init__(self):
        assert self.batch_size > 0
        assert self.epochs > 0
        assert self.learning_rate > 0
        assert self.prefetch >= 0
        assert self.eval_every_n_epochs > 0
        assert self.early_stopping_patience >= 0

    def serialization_workaround(self) -> dict[str, Any]:
        """orbax-serialization doesn't support pathlib.Path or even str(?)"""
        d = asdict(self)
        d.pop("checkpoints_dir")
        return d


_DEFAULTS: Final = Configuration()

# ------------------ Data utils (HF streaming) ------------------


def load_iterable_mnist(split: Split) -> IterableDataset:
    """Return an IterableDataset for the given split, with streaming enabled."""
    out = load_dataset("mnist", split=split, streaming=True)
    if isinstance(out, IterableDatasetDict):
        out = out[split]
    assert isinstance(out, IterableDataset)
    return out


def make_huggingface_iterator(
    dataset: IterableDataset,
    shuffle: bool,  # training: True;  evaluation: False
    *,
    batch_size: int = _DEFAULTS.batch_size,
    data_key: JaxArray | None = None,
    shuffle_buffer_size: int = _DEFAULTS.shuffle_buffer_size,
) -> Iterator[BatchedExamples]:
    """Optimized pipeline with batched preprocessing."""

    if shuffle:
        if data_key is None:
            raise ValueError(
                "A JAX PRNGKey `seed` must be provided when `shuffle=True`"
            )
        huggingface_seed: int = int(jax.random.randint(data_key, (), 0, 2**31 - 1))
        dataset = dataset.shuffle(
            seed=huggingface_seed, buffer_size=shuffle_buffer_size
        )

    def preprocess(batch: RawBatch) -> PreprocessedBatch:
        """Vectorized preprocessing for a batch of images and labels."""

        raw_images: NDArray[np.uint8 | np.uint16 | np.uint32 | np.uint64] = np.stack(
            [np.array(img) for img in batch["image"]]
        )
        assert raw_images.dtype in (np.uint8, np.uint16, np.uint32, np.uint64)

        # Normalize image pixel intensities to [-1, 1]
        scale: ImageDType = ImageDType(2 / np.iinfo(raw_images.dtype).max)
        images: NDArray[ImageDType] = raw_images.astype(
            ImageDType
        ) * scale - ImageDType(1)

        return {
            "image": images.reshape(images.shape[0], -1),  # (Batch Size, Num. Pixels)
            "label": np.array(batch["label"], dtype=LabelDType),
        }

    dataset = dataset.map(preprocess, batched=True, batch_size=batch_size).with_format(
        "numpy"
    )

    for batch in dataset.iter(batch_size=batch_size):
        xb = jnp.asarray(batch["image"], dtype=jnp.float32)
        yb = jnp.asarray(batch["label"], dtype=jnp.int32)
        yield (xb, yb)


# ------------------ Small prefetcher (host→device) ------------------
_SENTINEL: Final = object()


def prefetch(
    batches: Iterator[BatchedExamples], size: int = 2
) -> Iterator[BatchedExamples]:
    """Prefetch a few batches to device on a background thread."""

    if size <= 0:
        # No pre-fetch
        yield from batches
        return

    q: queue.Queue[BatchedExamples | object | Exception] = queue.Queue(maxsize=size)
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
            yield cast(BatchedExamples, item)
    finally:
        thread_done.set()
        thread.join(timeout=1.0)


# ------------------ Model ------------------
class MLP(nn.Module):
    """Multi-layer perceptron approximation architecture."""

    hidden_dim: int = 256
    num_classes: int = 10

    @nn.compact
    def __call__(self, x: JaxArray) -> JaxArray:
        x = nn.relu(nn.Dense(self.hidden_dim)(x))
        x = nn.Dense(self.num_classes)(x)
        return x


# ------------------ Train / Eval steps ------------------
def compute_metrics(
    apply_fn, params, x: JaxArray, y: JaxArray
) -> tuple[JaxArray, JaxArray]:
    logits: JaxArray = apply_fn({"params": params}, x)
    loss: JaxArray = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
    accuracy: JaxArray = (jnp.argmax(logits, axis=-1) == y).mean()
    return loss, accuracy


@functools.partial(jax.jit, donate_argnums=(0,))
def train_step(
    state: train_state.TrainState, x: JaxArray, y: JaxArray
) -> tuple[train_state.TrainState, JaxArray, JaxArray]:
    def _loss_fn(params):
        logits: JaxArray = state.apply_fn({"params": params}, x)
        loss: JaxArray = optax.softmax_cross_entropy_with_integer_labels(
            logits, y
        ).mean()
        accuracy: JaxArray = (jnp.argmax(logits, axis=-1) == y).mean()
        return loss, accuracy

    (loss, accuracy), grads = jax.value_and_grad(_loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, accuracy


@functools.partial(jax.jit, static_argnums=(1,))
def eval_step(params, apply_fn, x: JaxArray, y: JaxArray) -> tuple[JaxArray, JaxArray]:
    return compute_metrics(apply_fn, params, x, y)


# ------------------ Runner ------------------
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


def train_epoch(
    state: train_state.TrainState,
    examples: Iterator[BatchedExamples],
) -> tuple[train_state.TrainState, Metrics]:
    metrics = Metrics()
    for xb, yb in examples:
        state, loss, accuracy = train_step(state, xb, yb)
        metrics.update(float(loss), float(accuracy), xb.shape[0])
    return state, metrics


def evaluate(
    state: train_state.TrainState,
    examples: Iterator[BatchedExamples],
) -> Metrics:
    metrics = Metrics()
    for xb, yb in examples:
        loss, accuracy = eval_step(state.params, state.apply_fn, xb, yb)
        metrics.update(float(loss), float(accuracy), xb.shape[0])
    return metrics


def main(configuration: Configuration) -> None:
    """Main script driver."""

    # Prepare checkpointing
    checkpoints_dir: Path = configuration.checkpoints_dir.resolve()

    if configuration.clean_start and checkpoints_dir.exists():
        shutil.rmtree(checkpoints_dir)
        logger.info("Existing checkpoints removed. Starting from scratch.")

    checkpoints_dir.mkdir(exist_ok=True, parents=True)
    final_checkpoints_dir = checkpoints_dir / "final"
    final_checkpoints_dir.mkdir(exist_ok=True)

    key = jax.random.PRNGKey(configuration.seed)

    logger.info(f"Using {jax.device_count()} device(s): {jax.devices()}")

    checkpoints_options = ocp.CheckpointManagerOptions(max_to_keep=5, create=True)
    checkpoints_manager = ocp.CheckpointManager(
        checkpoints_dir, options=checkpoints_options
    )

    final_checkpoint_manager = ocp.CheckpointManager(
        final_checkpoints_dir,
        options=ocp.CheckpointManagerOptions(max_to_keep=1, create=True),
    )

    train_ds_size = 60_000
    steps_per_epoch = train_ds_size // configuration.batch_size
    num_training_steps = steps_per_epoch * configuration.epochs

    scheduler = optax.cosine_decay_schedule(
        init_value=configuration.learning_rate,
        decay_steps=num_training_steps,
    )

    model = MLP()
    initial_variables = model.init(key, jnp.ones((1, 784), jnp.float32))
    initial_state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=initial_variables["params"],
        tx=optax.adam(scheduler),
    )

    print("    initial_state:", type(initial_state))
    print("initial_variables:", type(initial_variables))

    # Define a default checkpoint structure
    default_checkpoint = {
        "state": initial_state,
        "best_loss": float("inf"),
        "patience": 0,
        "epoch": 0,
    }

    # Manual check for checkpoint existence to work around the Orbax bug
    latest_step = checkpoints_manager.latest_step()
    if latest_step is not None:
        logger.info(f"Restoring from checkpoint at step {latest_step}")
        restored_checkpoint = checkpoints_manager.restore(
            step=latest_step,
            args=ocp.args.StandardRestore(default_checkpoint),
        )
    else:
        logger.info("No checkpoints found. Starting from scratch.")
        restored_checkpoint = default_checkpoint

    # Check if the restored object is a dictionary, which it should be with StandardRestore
    if not isinstance(restored_checkpoint, dict):
        raise TypeError("Expected restored checkpoint to be a dictionary.")

    state = restored_checkpoint["state"]
    best_loss = restored_checkpoint.get("best_loss", float("inf"))
    patience_counter = restored_checkpoint.get("patience", 0)
    start_epoch = restored_checkpoint.get("epoch", 0) + 1

    train_dataset: IterableDataset = load_iterable_mnist("train")
    test_dataset: IterableDataset = load_iterable_mnist("test")

    training_completed: bool = False

    for epoch in range(start_epoch, configuration.epochs):
        training_completed = training_completed or True

        key, data_key = jax.random.split(key)

        train_iter = make_huggingface_iterator(
            dataset=train_dataset,
            shuffle=True,
            data_key=data_key,
            batch_size=configuration.batch_size,
            shuffle_buffer_size=configuration.shuffle_buffer_size,
        )
        train_iter_prefetched = prefetch(train_iter, size=configuration.prefetch)

        state, train_metrics = train_epoch(state, train_iter_prefetched)
        train_loss = train_metrics.loss.mean
        train_accuracy = train_metrics.accuracy.mean

        if epoch % configuration.eval_every_n_epochs == 0:
            eval_iter = make_huggingface_iterator(
                dataset=test_dataset,
                shuffle=False,
                batch_size=configuration.batch_size,
            )
            eval_iter_prefetched = prefetch(eval_iter, size=configuration.prefetch)

            eval_metrics = evaluate(state, eval_iter_prefetched)
            eval_loss = eval_metrics.loss.mean
            eval_accuracy = eval_metrics.accuracy.mean

            logger.info(
                f"[Flax] epoch {epoch} "
                f"loss={train_loss:.4f} accuracy={train_accuracy:.3f} "
                f"test_loss={eval_loss:.4f} test_accuracy={eval_accuracy:.3f}"
            )

            if eval_loss < best_loss:
                best_loss = eval_loss
                patience_counter = 0
                logger.info("Saving best model checkpoint...")

                checkpoint = {
                    "state": state,
                    "best_loss": best_loss,
                    "patience": patience_counter,
                    "epoch": epoch,
                }
                checkpoints_manager.save(epoch, args=ocp.args.StandardSave(checkpoint))
            else:
                patience_counter += 1
                if patience_counter >= configuration.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        checkpoint = {
            "state": state,
            "best_loss": best_loss,
            "patience": patience_counter,
            "epoch": epoch,
        }
        checkpoints_manager.save(epoch, args=ocp.args.StandardSave(checkpoint))

    if training_completed:
        final_checkpoint = {
            "state": state,
            "best_loss": best_loss,
            "final_epoch": epoch,
            "config": configuration.serialization_workaround(),
        }
        final_checkpoint_manager.save(
            epoch, args=ocp.args.StandardSave(final_checkpoint)
        )

    logger.info(f"Training completed. Best test loss: {best_loss:.4f}")


def parse_args() -> Configuration:
    """Processes command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Train digit classifier on MNIST database"
    )
    parser.add_argument(
        "--batch-size",
        default=_DEFAULTS.batch_size,
        help="Batch size for training",
        type=int,
    )
    parser.add_argument(
        "--checkpoints-dir",
        default=_DEFAULTS.checkpoints_dir,
        help="Directory to save checkpoints",
        type=Path,
    )
    parser.add_argument(
        "--clean-start",
        action="store_true",
        default=_DEFAULTS.clean_start,
        help="Remove existing checkpoints and start fresh",
    )
    parser.add_argument(
        "--epochs",
        default=_DEFAULTS.epochs,
        help="Number of training epochs",
        type=int,
    )
    parser.add_argument(
        "--learning-rate",
        default=_DEFAULTS.learning_rate,
        help="Learning rate",
        type=float,
    )
    parser.add_argument(
        "--seed",
        default=_DEFAULTS.seed,
        help="Seed for pseudo-random number generator",
        type=int,
    )

    args = parser.parse_args()
    return Configuration(**vars(args))


def cleanup():
    """Release JAX resources."""
    jax.clear_caches()
    # jax.clear_backends() was removed in JAX v0.4.36


if __name__ == "__main__":
    config = parse_args()
    try:
        main(config)
    finally:
        cleanup()
