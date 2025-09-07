# jax-demos


### Installation in vscode

1. Install the [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension `ms-vscode-remote.remote-containers`.
2. Open this repository's root folder in vscode.
3. Run the command `Dev Containers: Reopen in Container`.

---

## Testing

```bash
nvidia-smi
python -c "import jax; print(jax.default_backend(), jax.devices())"
```

---

## Creating a virtual environment

At the terminal (inside your newly-created container), run the following commands to create a virtual environment:

```bash
# from /workspace (your project root mounted by devcontainer)
python -m venv .venv --system-site-packages

source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --requirement requirements.txt
# or from pyproject.toml:
# python -m pip install --editable .
```

## Fitting the MLP model

```bash
python flax_mnist.py --help
python flax_mnist.py --clean-start
```

```
In [1]: %run flax_mnist.py --clean-start
Starting from scratch - existing checkpoints removed
Using 1 device(s):
  Device 0: cuda:0 (platform: gpu)
Config | seed=0 batch=128 epochs=5 lr=1.00e-03 prefetch=2 buffer=50000
No checkpoints found. Starting from scratch.
[Flax] epoch 1 loss=0.3567 accuracy=0.894 test_loss=0.1988 test_accuracy=0.940
Saving best model checkpoint...
[Flax] epoch 2 loss=0.1720 accuracy=0.949 test_loss=0.1362 test_accuracy=0.959
Saving best model checkpoint...
[Flax] epoch 3 loss=0.1210 accuracy=0.965 test_loss=0.1116 test_accuracy=0.968
Saving best model checkpoint...
[Flax] epoch 4 loss=0.0959 accuracy=0.973 test_loss=0.1007 test_accuracy=0.970
Saving best model checkpoint...
Training completed. Best test loss: 0.1007
```

## Viewing TensorBoard logs

```bash
tensorboard --logdir logs
```

View the TensorBoard dashboard at: `http://localhost:6006`

