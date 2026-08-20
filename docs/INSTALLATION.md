# Installation and Deployment

SSAT requires Python 3.11 or later. PyTorch and Torchvision are the largest and most environment-sensitive dependencies, so choose CPU/CUDA wheels appropriate for the host before installing SSAT when you need explicit accelerator control.

## Docker Compose development workspace

The supported repository development environment uses the CUDA-enabled image defined in `.devcontainer/Dockerfile` and bind-mounts the repository at `/workspace`.

```bash
docker compose up -d --build region-sensitivity-workspace
docker compose exec region-sensitivity-workspace pip install --no-deps -e .
docker compose exec region-sensitivity-workspace pytest -q
docker compose exec region-sensitivity-workspace ssat --help
```

The image build already installs the packaged copy of SSAT and all requirements. The editable install makes the bind-mounted working tree authoritative after source changes.

`compose.yaml` requests all available GPUs and 32 GiB of shared memory. On a CPU-only machine, remove or override the service's `gpus: all` setting. The committed quickstart configuration explicitly uses `device: cpu`.

The VS Code Dev Container uses the same Dockerfile, GPU request, shared-memory allocation, and `/workspace` mount.

## Local Python installation

Create and activate a virtual environment, then install a PyTorch/Torchvision pair suitable for your platform. The CI-tested CPU pair is shown below:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  torch==2.8.0 torchvision==0.23.0
python -m pip install -e .
ssat --version
```

For CUDA or another platform, follow the PyTorch installation selector and then run `python -m pip install -e .`. `pyproject.toml` is the canonical package dependency declaration. `requirements.txt` mirrors it for Docker and CI, where PyTorch is installed first from the selected wheel index.

On a minimal Debian/Ubuntu host, the container build also installs common native runtime packages for OpenCV and Matplotlib:

```bash
sudo bash scripts/install_deps.sh
```

The script runs `apt-get`, so it requires root privileges and is not intended for non-Debian systems.

## Verify the installation

From the repository root:

```bash
ssat --help
ssat estimate configs/examples/quickstart.yaml
pytest -q
```

The synthetic quickstart does not download model weights. It uses committed fixture media and `weights: null`.

## Deployment image

`Dockerfile` creates an image whose entry point is `ssat`. `compose.deploy.yaml` mounts configuration and data read-only and stores dumps in a writable named volume or host directory.

```bash
docker compose -f compose.deploy.yaml build ssat
docker compose -f compose.deploy.yaml run --rm ssat \
  run /config/deploy/quickstart.yaml --output /dumps/quickstart
docker compose -f compose.deploy.yaml run --rm ssat \
  inspect /dumps/quickstart
```

The default mount sources are:

| Container path | Host source | Access |
| --- | --- | --- |
| `/config` | `${SSAT_CONFIG_DIR:-./configs}` | read-only |
| `/data` | `${SSAT_DATA_DIR:-./tests/fixtures/synthetic_classification}` | read-only |
| `/dumps` | `${SSAT_DUMP_DIR:-./dump}` | read-write |

Set the environment variables before invoking Compose to use real data and a chosen dump directory:

```bash
SSAT_CONFIG_DIR=/srv/ssat/configs \
SSAT_DATA_DIR=/srv/ssat/data \
SSAT_DUMP_DIR=/srv/ssat/dumps \
docker compose -f compose.deploy.yaml run --rm ssat \
  estimate /config/audit.yaml
```

The deployment Compose file also requests all GPUs and 32 GiB shared memory. Use a Compose override that removes the GPU reservation on a CPU-only host, and set the adapter's `device` to `cpu`.

## Offline operation

Framework-provided pretrained selectors can read caches or access the network. For an offline audit, use one of the following and ensure all media/configuration paths are mounted:

- Torchvision `weights: null` for seeded random initialization.
- timm `pretrained: false` for seeded random initialization.
- A trusted local `checkpoint.path` included in the configuration mount.

Randomly initialized weights validate the software path but do not produce scientifically meaningful sensitivity results.

## Troubleshooting

- If `ssat` is not found, verify that the active environment is the one where the package was installed, or use `python -m ssat`.
- If a pretrained selector fails offline, pre-populate the framework cache or switch to a local checkpoint.
- If CUDA initialization fails, use compatible PyTorch wheels and drivers or set `device: cpu`.
- If Compose reports that no GPU device driver is available, remove/override `gpus: all`; changing only the YAML adapter device is not sufficient for container creation.
- If OpenCV fails to import on a minimal Linux host, install the native packages used by `scripts/install_deps.sh`.
