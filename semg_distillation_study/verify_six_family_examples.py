"""Run the bundled fold-0 checkpoints against 64 recorded feature histories."""

import json
from pathlib import Path

import numpy as np
import torch

from data import sha
from load_distilled import ElbowAngleModel
from models import FAMILIES
from verify_portable_models import validate_prediction_pair


def verify(folder):
    roster = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    models = roster["models"]
    expected = {(family, method) for family in FAMILIES for method in ("teacher", "direct", "kd")}
    if len(models) != 18 or {(r["family"], r["method"]) for r in models} != expected:
        raise ValueError("Need 18 distinct fold-0 examples")
    if sha(folder / "examples.npz") != roster["examples_sha256"]:
        raise ValueError("Example file changed")
    torch.set_num_threads(4)
    differences = []
    with np.load(folder / "examples.npz", allow_pickle=False) as examples:
        features = examples["X"]
        if features.shape != (64, 21, 12) or not np.isfinite(features).all():
            raise ValueError("Malformed input examples")
        for record in models:
            checkpoint = folder / record["family"] / record["method"] / "best.pt"
            if sha(checkpoint) != record["sha256"]:
                raise ValueError(f"Changed checkpoint: {checkpoint}")
            model = ElbowAngleModel(checkpoint)
            if model.family != record["family"] or model.role != ("teacher" if record["method"] == "teacher" else "student"):
                raise ValueError("Incorrect model identity")
            values = model.predict_degrees(features)
            differences.append(validate_prediction_pair(values, examples[record["example_key"]]))
    return dict(status="passed", n_models=len(differences), examples_per_model=64,
                maximum_difference_deg=max(differences))


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    proof = verify(root / "runs/six_family_kd_v1/portable")
    print(json.dumps(proof, ensure_ascii=False))
