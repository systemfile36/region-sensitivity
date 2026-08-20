import json
from pathlib import Path
import tomllib

import yaml


ROOT = Path(__file__).parents[2]


def test_package_and_deployment_metadata_are_parseable() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["version"] == "0.1.0"
    assert project["project"]["scripts"]["ssat"] == "ssat.cli:main"

    compose = yaml.safe_load((ROOT / "compose.deploy.yaml").read_text(encoding="utf-8"))
    service = compose["services"]["ssat"]
    assert service["shm_size"] == "32G"
    assert any(str(volume).endswith(":/data:ro") for volume in service["volumes"])
    assert any(str(volume).endswith(":/config:ro") for volume in service["volumes"])


def test_example_notebook_is_valid_v4_json() -> None:
    notebook = json.loads(
        (ROOT / "examples" / "inspect_dump.ipynb").read_text(encoding="utf-8")
    )
    assert notebook["nbformat"] == 4
    assert any(cell["cell_type"] == "code" for cell in notebook["cells"])
