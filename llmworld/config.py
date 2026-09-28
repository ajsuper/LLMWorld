"""Config and data loading."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: Path | None = None) -> dict:
    return yaml.safe_load((path or ROOT / "config.yaml").read_text())


def load_data(data_dir: Path | None = None) -> dict:
    d = data_dir or ROOT / "data"
    data = {name: yaml.safe_load((d / f"{name}.yaml").read_text()) for name in ("traits", "ambitions", "phrases", "agents")}
    for spec in data["agents"]:
        for t in spec.get("traits", []):
            if t not in data["traits"]:
                raise ValueError(f"agent {spec['name']} has unknown trait {t!r}")
        if spec.get("ambition") and spec["ambition"] not in data["ambitions"]:
            raise ValueError(f"agent {spec['name']} has unknown ambition {spec['ambition']!r}")
    return data
