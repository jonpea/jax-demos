import argparse
import functools
import pathlib
from pathlib import Path
import queue
import shutil
import threading
from dataclasses import dataclass
import logging
import sys
from typing import Any, Final, Iterator, Literal, Mapping, cast

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

import flax.linen as nn
from flax.training import train_state
import optax
import orbax.checkpoint as ocp

from datasets import load_dataset, IterableDataset, IterableDatasetDict

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

# ------------------ Types (PEP 695 aliases) ------------------
type JaxArray = jax.Array
type Split = Literal["train", "test"]

# A batch = (images, labels)
type BatchedExamples = tuple[JaxArray, JaxArray]
type RawBatch = Mapping[str, NDArray[Any]]
type PreprocessedBatch = Mapping[str, NDArray[Any]]


# ------------------ Data utils (HF streaming) ------------------
def load_iterable_mnist(split: Split) -> IterableDataset:
    """Return an IterableDataset for the given split, with streaming enabled."""
    out = load_dataset("mnist", split=split, streaming=True)
    if isinstance(out, IterableDatasetDict):
        out = out[split]
    assert isinstance(out, IterableDataset)
    return out


def make_huggingface_iterator(
    ds: IterableDataset,
    batch_size: int,
    *,
    shuffle: bool = True,
    seed: JaxArray | None = None,
    shuffle_buffer_size: int = 50_000,
) -> Iterator[BatchedExamples]:
    """Optimized pipeline with batched preprocessing."""

    hf_seed: int = 0 if seed is None else int(jax.random.key_data(seed)[0])

    def preprocess(batch: RawBatch) -> PreprocessedBatch:
        """Vectorized preprocessing for a batch of images and labels."""
        images = np.array(batch["image"], dtype=np.float32) / 127.5 - 1.0
        images = images.reshape(images.shape[0], -1)  # (B, 784)
        labels = np.array(batch["label"], dtype=np.int32)
        return {"image": images, "label": labels}

    if shuffle:
        ds = ds.shuffle(seed=hf_seed, buffer_size=shuffle_buffer_size)

    ds = ds.map(preprocess, batched=True, batch_size=batch_size).with_format("numpy")

    for batch in ds.iter(batch_size=batch_size):
        xb = jnp.asarray(batch["image"], dtype=jnp.float32)
        yb = jnp.asarray(batch["label"], dtype=jnp.int32)
        yield (xb, yb)


# ------------------ Small prefetcher (host→device) ------------------
class _Sentinel:
    pass


_SENTINEL: Final = _Sentinel()


def prefetch(it: Iterator[BatchedExamples], size: int = 2) -> Iterator[BatchedExamples]:
    """Prefetch a few batches to device on a background thread."""
    q: queue.Queue[BatchedExamples | _Sentinel | Exception] = queue.Queue(maxsize=size)
    thread_done = threading.Event()

    def _worker() -> None:
        try:
            for b in it:
                if thread_done.is_set():
                    break
                q.put(jax.device_put(b))
            q.put(_SENTINEL)
        except Exception as e:
            q.put(e)
        finally:
            thread_done.set()

    thread = threading.Thread(target=_worker, daemon=True)
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
    hidden_dim: int = 256
    num_classes: int = 10

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = nn.relu(nn.Dense(self.hidden_dim)(x))
        x = nn.Dense(self.num_classes)(x)
        return x


# ------------------ Train / Eval steps ------------------
def compute_metrics(
    apply_fn, params, x: jax.Array, y: jax.Array
) -> tuple[jax.Array, jax.Array]:
    logits = apply_fn({"params": params}, x)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
    accuracy = (jnp.argmax(logits, axis=-1) == y).mean()
    return loss, accuracy


@functools.partial(jax.jit, donate_argnums=(0,))
def train_step(
    state: train_state.TrainState, x: JaxArray, y: JaxArray
) -> tuple[train_state.TrainState, JaxArray, JaxArray]:
    def _loss_fn(params):
        logits = state.apply_fn({"params": params}, x)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
        accuracy = (jnp.argmax(logits, axis=-1) == y).mean()
        return loss, accuracy

    (loss, accuracy), grads = jax.value_and_grad(_loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, accuracy


@functools.partial(jax.jit, static_argnums=(1,))
def eval_step(params, apply_fn, x: JaxArray, y: JaxArray) -> tuple[JaxArray, JaxArray]:
    return compute_metrics(apply_fn, params, x, y)


# ------------------ Runner ------------------
@dataclass
class Metrics:
    loss: float = 0.0
    accuracy: float = 0.0
    count: int = 0

    def update(self, loss: float, accuracy: float, batch_size: int) -> None:
        self.loss += loss * batch_size
        self.accuracy += accuracy * batch_size
        self.count += batch_size

    def compute(self) -> tuple[float, float]:
        if self.count == 0:
            return 0.0, 0.0
        return self.loss / self.count, self.accuracy / self.count


@dataclass(frozen=True)
class Configuration:
    batch_size: int = 128
    checkpoints_dir: Path = Path("./checkpoints")
    epochs: int = 5
    learning_rate: float = 1e-3
    prefetch: int = 2
    seed: int = 0
    shuffle_buffer_size: int = 50_000
    eval_every_n_epochs: int = 1
    early_stopping_patience: int = 3
    clean_start: bool = False

    def __post_init__(self):
        assert self.batch_size > 0
        assert self.epochs > 0
        assert self.learning_rate > 0
        assert self.prefetch >= 0
        assert self.eval_every_n_epochs > 0
        assert self.early_stopping_patience >= 0


def train_epoch(
    state: train_state.TrainState,
    iterator: Iterator[BatchedExamples],
) -> tuple[train_state.TrainState, Metrics]:
    metrics = Metrics()
    for xb, yb in iterator:
        state, loss, accuracy = train_step(state, xb, yb)
        metrics.update(float(loss), float(accuracy), xb.shape[0])
    return state, metrics


def evaluate(
    state: train_state.TrainState,
    iterator: Iterator[BatchedExamples],
) -> Metrics:
    metrics = Metrics()
    for xb, yb in iterator:
        loss, accuracy = eval_step(state.params, state.apply_fn, xb, yb)
        metrics.update(float(loss), float(accuracy), xb.shape[0])
    return metrics


def main(configuration: Configuration) -> None:
    # Prepare checkpointing
    checkpoints_dir = configuration.checkpoints_dir.resolve()

    if configuration.clean_start and checkpoints_dir.exists():
        shutil.rmtree(checkpoints_dir)
        logger.info("Existing checkpoints removed. Starting from scratch.")

    checkpoints_dir.mkdir(exist_ok=True, parents=True)

    key = jax.random.PRNGKey(configuration.seed)

    logger.info(f"Using {jax.device_count()} device(s): {jax.devices()}")

    checkpoints_options = ocp.CheckpointManagerOptions(max_to_keep=5, create=True)
    checkpoints_manager = ocp.CheckpointManager(
        checkpoints_dir, options=checkpoints_options
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
        restored_ckpt = checkpoints_manager.restore(
            step=latest_step,
            args=ocp.args.StandardRestore(default_checkpoint),
        )
    else:
        logger.info("No checkpoints found. Starting from scratch.")
        restored_ckpt = default_checkpoint

    # Check if the restored object is a dictionary, which it should be with StandardRestore
    if not isinstance(restored_ckpt, dict):
        raise TypeError("Expected restored checkpoint to be a dictionary.")

    state = restored_ckpt["state"]
    best_loss = restored_ckpt.get("best_loss", float("inf"))
    patience_counter = restored_ckpt.get("patience", 0)
    start_epoch = restored_ckpt.get("epoch", 0) + 1

    train_ds = load_iterable_mnist("train")
    test_ds = load_iterable_mnist("test")

    for epoch in range(start_epoch, configuration.epochs):
        key, data_key = jax.random.split(key)

        train_iter = make_huggingface_iterator(
            train_ds,
            configuration.batch_size,
            shuffle=True,
            seed=data_key,
            shuffle_buffer_size=configuration.shuffle_buffer_size,
        )
        train_iter_prefetched = prefetch(train_iter, size=configuration.prefetch)

        state, train_metrics = train_epoch(state, train_iter_prefetched)
        train_loss, train_accuracy = train_metrics.compute()

        if epoch % configuration.eval_every_n_epochs == 0:
            eval_iter = make_huggingface_iterator(
                test_ds,
                configuration.batch_size,
                shuffle=False,
            )
            eval_iter_prefetched = prefetch(eval_iter, size=configuration.prefetch)

            eval_metrics = evaluate(state, eval_iter_prefetched)
            eval_loss, eval_accuracy = eval_metrics.compute()

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

    logger.info(f"Training completed. Best test loss: {best_loss:.4f}")


def parse_args() -> Configuration:
    defaults = Configuration()
    parser = argparse.ArgumentParser(description="Train MLP on MNIST with Flax")
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument(
        "--checkpoints-dir", type=pathlib.Path, default=defaults.checkpoints_dir
    )
    parser.add_argument(
        "--clean_start", action="store_true", default=defaults.clean_start
    )
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--seed", type=int, default=defaults.seed)

    args = parser.parse_args()
    return Configuration(**vars(args))


if __name__ == "__main__":
    config = parse_args()
    main(config)
