import argparse
import functools
import logging
import shutil
import sys

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Final, Iterator

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp

from datasets import load_dataset, IterableDataset, IterableDatasetDict

# from flax import jax_utils
# from flax.training import common_utils
from flax.training import train_state

from custom_types import (
    ApplyFn,
    BatchedExamples,
    JaxArray,
    JaxScalar,
    Metrics,
    PyTree,
    Split,
)
from data_utilities import RunningMetrics, prefetch, make_huggingface_iterator


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
# Silence noisy loggers
logging.getLogger("absl").setLevel(logging.WARNING)
logging.getLogger("jax").setLevel(logging.WARNING)
logging.getLogger("orbax").setLevel(logging.WARNING)


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


def load_iterable_mnist(split: Split) -> IterableDataset:
    """Return an IterableDataset for the given split, with streaming enabled."""
    out = load_dataset("mnist", split=split, streaming=True)
    if isinstance(out, IterableDatasetDict):
        out = out[split]
    assert isinstance(out, IterableDataset)
    return out


class MLP(nn.Module):
    """Multi-layer perceptron approximation architecture."""

    hidden_dim: int = 256
    num_classes: int = 10

    @nn.compact
    def __call__(self, x: JaxArray) -> JaxArray:
        x = nn.relu(nn.Dense(self.hidden_dim)(x))
        x = nn.Dense(self.num_classes)(x)
        return x


def compute_metrics(
    apply_fn: ApplyFn, params: PyTree, x: JaxArray, y: JaxArray
) -> Metrics:
    logits: JaxArray = apply_fn({"params": params}, x)
    loss: JaxScalar = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
    accuracy: JaxScalar = (jnp.argmax(logits, axis=-1) == y).mean()
    return loss, accuracy


@functools.partial(jax.jit, donate_argnums=(0,))
def train_step(
    state: train_state.TrainState, x: JaxArray, y: JaxArray
) -> tuple[train_state.TrainState, JaxScalar, JaxScalar]:
    """Evaluates training loss function and accuracy."""

    def _compute_metrics(params: PyTree) -> Metrics:
        """Partially applies the metrics function for various parameters."""
        return compute_metrics(state.apply_fn, params, x, y)

    loss: JaxScalar
    accuracy: JaxScalar
    grads: PyTree
    (loss, accuracy), grads = jax.value_and_grad(
        _compute_metrics,
        has_aux=True,
    )(state.params)
    next_state: train_state.TrainState = state.apply_gradients(grads=grads)
    return next_state, loss, accuracy


def jit_eval_step(apply_fn) -> Callable:
    """Generates the evaluation function."""

    @jax.jit
    def evaluate(params: PyTree, x: JaxArray, y: JaxArray) -> Metrics:
        return compute_metrics(apply_fn, params, x, y)

    return evaluate


def train_epoch(
    state: train_state.TrainState,
    batches: Iterator[BatchedExamples],
) -> tuple[train_state.TrainState, RunningMetrics]:
    metrics = RunningMetrics()
    for xb, yb in batches:
        state, loss, accuracy = train_step(state, xb, yb)
        metrics.update(float(loss), float(accuracy), xb.shape[0])
    return state, metrics


def evaluate(
    state: train_state.TrainState,
    batches: Iterator[BatchedExamples],
) -> RunningMetrics:
    metrics = RunningMetrics()
    eval_step: Callable = jit_eval_step(state.apply_fn)
    for xb, yb in batches:
        loss, accuracy = eval_step(state.params, xb, yb)
        metrics.update(float(loss), float(accuracy), xb.shape[0])
    return metrics


def main(configuration: Configuration) -> None:
    """Main script driver."""

    # Prepare checkpointing
    checkpoints_dir: Path = configuration.checkpoints_dir.resolve()

    if configuration.clean_start and checkpoints_dir.exists():
        shutil.rmtree(checkpoints_dir)
        logger.info("Starting from scratch - existing checkpoints removed")

    checkpoints_dir.mkdir(exist_ok=True, parents=True)
    final_checkpoints_dir: Path = checkpoints_dir / "final"
    final_checkpoints_dir.mkdir(exist_ok=True)

    key = jax.random.PRNGKey(configuration.seed)

    logger.info(f"Using {jax.device_count()} device(s): {jax.devices()}")
    logger.info(
        "Config | seed=%d batch=%d epochs=%d lr=%.2e prefetch=%d buffer=%d",
        configuration.seed,
        configuration.batch_size,
        configuration.epochs,
        configuration.learning_rate,
        configuration.prefetch,
        configuration.shuffle_buffer_size,
    )

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

    for epoch in range(start_epoch, configuration.epochs):
        key, data_key = jax.random.split(key)

        train_iter = make_huggingface_iterator(
            dataset=train_dataset,
            shuffle=True,
            data_key=data_key,
            batch_size=configuration.batch_size,
            shuffle_buffer_size=configuration.shuffle_buffer_size,
        )
        train_iter_prefetched = prefetch(train_iter, size=configuration.prefetch)
        # train_iter_prefetched = jax_utils.prefetch_to_device(
        #     train_iter, size=configuration.prefetch
        # )

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
            # eval_iter_prefetched = jax_utils.prefetch_to_device(
            #     eval_iter, size=configuration.prefetch
            # )

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

    else:
        # This block runs only if the for loop completes without a `break`
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
        "--early-stopping-patience",
        default=_DEFAULTS.early_stopping_patience,
        help="Early stopping patience",
        type=int,
    )
    parser.add_argument(
        "--eval-every-n-epochs",
        default=_DEFAULTS.eval_every_n_epochs,
        help="Evaluation frequency",
        type=int,
    )
    parser.add_argument(
        "--learning-rate",
        default=_DEFAULTS.learning_rate,
        help="Learning rate",
        type=float,
    )
    parser.add_argument(
        "--prefetch",
        default=_DEFAULTS.prefetch,
        help="Number of batches to prefetch",
        type=int,
    )
    parser.add_argument(
        "--seed",
        default=_DEFAULTS.seed,
        help="Seed for pseudo-random number generator",
        type=int,
    )
    parser.add_argument(
        "--shuffle-buffer-size",
        default=_DEFAULTS.shuffle_buffer_size,
        help="Shuffle buffer size",
        type=int,
    )

    args = parser.parse_args()
    return Configuration(**vars(args))


def cleanup():
    """Releases resources."""
    jax.clear_caches()
    # jax.clear_backends() was removed in JAX v0.4.36


if __name__ == "__main__":
    config = parse_args()
    try:
        main(config)
    finally:
        cleanup()
