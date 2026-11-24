"""Validate a node-prediction submission.

CLI:   python competition/validate_submission.py predictions.csv
Format (CSV with header): id,y_pred

* exactly one row per id in data/public/test_nodes.csv, no duplicates
* y_pred is a finite number in [0, 1] (probability that the superpixel is polyp);
  a node counts as foreground when y_pred >= 0.5
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent


def _find_test_csv() -> Path:
    name = yaml.safe_load((HERE / "config.yaml").read_text()).get("test_file", "test_nodes.csv")
    for p in [Path("data/public") / name, HERE.parent / "data" / "public" / name,
              Path("PolypSegArena_Dataset/data/public") / name]:
        if p.exists():
            return p
    raise ValueError(f"Could not find data/public/{name}.")


def validate(pred_path, test_csv=None) -> bool:
    test_csv = Path(test_csv) if test_csv else _find_test_csv()
    try:
        preds = pd.read_csv(pred_path)
    except Exception as exc:
        raise ValueError(f"Could not read prediction file: {exc}")
    tests = pd.read_csv(test_csv, usecols=["id"])
    if "id" not in preds.columns or "y_pred" not in preds.columns:
        raise ValueError("Submission must contain 'id' and 'y_pred' columns.")
    if preds["id"].duplicated().any():
        raise ValueError("Duplicate id values found.")
    if set(preds["id"]) != set(tests["id"]):
        raise ValueError(f"id mismatch: expected {len(tests)} ids, got {len(preds)} "
                         f"({len(set(tests['id']) - set(preds['id']))} missing).")
    y = pd.to_numeric(preds["y_pred"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("y_pred must be finite numbers (no NaN / empty values).")
    if (y < 0).any() or (y > 1).any():
        raise ValueError("y_pred must be in [0, 1].")
    return True


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python validate_submission.py <pred_path>", file=sys.stderr)
        sys.exit(1)
    try:
        validate(sys.argv[1])
        print("VALID SUBMISSION")
    except Exception as exc:
        print(f"VALIDATION FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
