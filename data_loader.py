import json
from pathlib import Path
from typing import List

import pandas as pd


def load_rig_config(path: Path) -> dict:
    with open(path, "r") as handle:
        return json.load(handle)


def load_all_rigs(directory: Path) -> List[dict]:
    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"Rig directory not found: {dir_path}")

    rigs = []
    for rig_path in sorted(dir_path.glob("*.json")):
        config = load_rig_config(rig_path)
        rigs.append({"path": rig_path, "config": config})
    return rigs


def load_difficulty_data(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)

