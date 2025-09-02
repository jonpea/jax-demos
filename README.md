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
python -m pip install --requirements requirements.txt
# or from pyproject.toml:
# python -m pip install --editable .
```

