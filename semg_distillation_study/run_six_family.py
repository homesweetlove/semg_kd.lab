"""Compare direct and fixed-weight distillation for every teacher family."""

import argparse
import datetime as dt
import itertools
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch

from data import prepare_data, sha
from models import make_model
from resume_personcv import check_checkpoints
from run_robustness import CODE
from select_teachers import verify_tree
from selection import (
    BASE, fit_model, normalized_fold, person_folds, row_hash, save, verify_proof,
    weight_hash,
)
from student_study import teacher_targets

TEACHERS = BASE / "runs/teacher_personcv_v1"
PREVIOUS = BASE / "runs/student_robustness_v2"
OUT = BASE / "runs/six_family_kd_v1"
FEATURES = BASE.parent / "semg_experiment/data/features.npz"
CACHE = BASE.parent.parent / "work/sequence_cache"
THREADS = 4
WEIGHTS = (0.0, 0.5)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def runtime():
    return dict(python=sys.version, torch=str(torch.__version__), numpy=np.__version__,
                platform=platform.platform(), cpu_count=os.cpu_count())


def prepare(features):
    teacher_manifest = read_json(TEACHERS / "manifest.json")
    student_manifest = read_json(PREVIOUS / "manifest.json")
    audit = read_json(PREVIOUS / "verification.json")
    for folder, manifest in ((TEACHERS, teacher_manifest), (PREVIOUS, student_manifest)):
        verify_tree(folder, manifest)
        for name, digest in manifest["configuration"]["code_sha256"].items():
            if sha(BASE / name) != digest:
                raise ValueError(f"Original implementation changed: {name}")
    if audit["status"] != "passed" or audit["manifest_sha256"] != sha(PREVIOUS / "manifest.json"):
        raise ValueError("Missing or mismatched parent audit")
    previous = student_manifest["configuration"]
    teachers = teacher_manifest["configuration"]
    features_hash = sha(features)
    if not (features_hash == previous["features_sha256"] == teachers["features_sha256"]):
        raise ValueError("Different source features")
    if previous["runtime"] != runtime() or previous["threads"] != THREADS:
        raise ValueError("Current environment differs from the reused experiment")
    if previous["teacher_manifest_sha256"] != sha(TEACHERS / "manifest.json"):
        raise ValueError("Previous students used different teachers")
    families = [row["family"] for row in read_json(TEACHERS / "selection.json")["ranking"]]
    reused = previous["families"]
    new = [family for family in families if family not in reused]
    if len(families) != 6 or len(set(families)) != 6 or len(new) != 3:
        raise ValueError("Need six teachers and three previously evaluated families")
    config = dict(
        families=families, reused_families=reused, new_families=new,
        source_people=previous["source_people"], student_seeds=previous["student_seeds"],
        weights=list(WEIGHTS), threads=THREADS, training=previous["training"],
        runtime=runtime(), features_sha256=features_hash,
        teacher_manifest_sha256=sha(TEACHERS / "manifest.json"),
        prior_student_manifest_sha256=sha(PREVIOUS / "manifest.json"),
        prior_audit_sha256=sha(PREVIOUS / "verification.json"),
        code_sha256={name: sha(BASE / name) for name in (*CODE, Path(__file__).name)},
        scope="Source-development comparison with fixed fold teachers. No independent test.",
    )
    protocol = OUT / "protocol.json"
    if protocol.exists():
        if read_json(protocol)["configuration"] != config:
            raise ValueError("Frozen protocol changed")
    else:
        save(protocol, dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                            configuration=config))
    return config


def path_for(fold, family, seed, weight):
    return OUT / f"fold{fold}" / family / f"s{seed}" / f"w{int(weight * 100):03d}" / "model"


def fit_fold(fold_number, family, features, cache):
    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(1)
    config = read_json(OUT / "protocol.json")["configuration"]
    for name, digest in config["code_sha256"].items():
        if sha(BASE / name) != digest:
            raise ValueError(f"Training code changed: {name}")
    if config["runtime"] != runtime() or config["features_sha256"] != sha(features):
        raise ValueError("Worker runtime or features differ from protocol")
    data = prepare_data(features, cache)
    fold = person_folds(config["source_people"], data["subject"])[fold_number]
    training_indices = fold["train_indices"]
    validation_indices = fold["validation_indices"]
    train_x, train_y, val_x, val_y, val_people, scaler = normalized_fold(data, fold)
    teacher = TEACHERS / f"fold{fold_number}" / family / "best.pt"
    target_dir = OUT / f"fold{fold_number}" / family / "targets"
    targets, target_proof = teacher_targets(
        target_dir, teacher, fold["train_people"], fold["validation_people"], family, scaler, train_x
    )
    if len(targets) != len(train_y) or not torch.isfinite(targets).all():
        raise ValueError("Invalid teacher predictions")
    common = dict(
        source_people=fold["train_people"], validation_people=fold["validation_people"],
        fold=fold_number, models_code_sha256=sha(BASE / "models.py"),
        configuration_sha256=sha(OUT / "protocol.json"),
        train_rows_sha256=row_hash(data["row_ids"][training_indices]),
        validation_rows_sha256=row_hash(data["row_ids"][validation_indices]),
    )
    records = []
    for seed in config["student_seeds"]:
        training = dict(config["training"], seed=seed + 1009 * fold_number)
        torch.manual_seed(training["seed"])
        initial = weight_hash(make_model(family, "student").state_dict())
        for weight in WEIGHTS:
            dest = path_for(fold_number, family, seed, weight)
            check_checkpoints(dest)
            metadata = dict(
                common, teacher_sha256=target_proof["teacher_sha256"] if weight else None,
                teacher_target_sha256=sha(target_dir / "targets.npz") if weight else None,
            )
            fit_model(dest, family, "student", train_x, train_y, val_x, val_y, val_people,
                      scaler, training, metadata, targets if weight else None, weight,
                      data["row_ids"][validation_indices])
            check_checkpoints(dest)
            best = torch.load(dest / "best.pt", map_location="cpu", weights_only=True)
            if best["metadata"]["initial_weights_sha256"] != initial:
                raise ValueError("Initial weights do not match seed")
            records.append(dict(
                fold=fold_number, family=family, seed=seed, weight=weight,
                model_folder=dest.relative_to(BASE).as_posix(),
                model_proof_sha256=sha(dest / "complete.json"),
                checkpoint_sha256=sha(dest / "best.pt"),
            ))
            print("CONDITION_COMPLETE", fold_number, family, seed, weight, flush=True)
    save(target_dir.parent / "conditions.json", records)


def run(args):
    if args.fold is not None:
        if args.family is None:
            raise ValueError("A fold worker needs a family")
        return fit_fold(args.fold, args.family, args.features, args.cache)
    config = prepare(args.features)
    prepare_data(args.features, args.cache)
    if (OUT / "fit_manifest.json").exists():
        verify_tree(OUT, read_json(OUT / "fit_manifest.json"))
        print("VERIFIED_COMPLETED_FITS", flush=True)
        return
    tasks = list(itertools.product(range(5), config["new_families"]))
    active = {}
    logs = OUT / "logs"
    logs.mkdir(exist_ok=True)
    expected = len(tasks) * len(config["student_seeds"]) * len(WEIGHTS)
    last_count = None
    try:
        while tasks or active:
            while tasks and len(active) < args.jobs:
                fold, family = tasks.pop(0)
                command = [
                    sys.executable, "-B", __file__, "--fold", str(fold), "--family", family,
                    "--features", str(args.features.resolve()), "--cache", str(args.cache.resolve()),
                ]
                env = dict(os.environ, PYTHONIOENCODING="utf-8", OMP_NUM_THREADS=str(THREADS),
                           MKL_NUM_THREADS=str(THREADS), OPENBLAS_NUM_THREADS=str(THREADS))
                with (logs / f"fold{fold}_{family}.log").open("a", encoding="utf-8") as log:
                    active[fold, family] = subprocess.Popen(
                        command, cwd=BASE, env=env, stdout=log, stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                    )
            for key, proc in list(active.items()):
                code = proc.poll()
                if code is not None:
                    del active[key]
                    if code:
                        raise RuntimeError(f"Worker {key} exited {code}; inspect {logs}")
            count = sum(1 for _ in OUT.glob("fold*/*/s*/w*/model/complete.json"))
            save(OUT / "progress.json", dict(status="running", completed_new=count,
                                            expected_new=expected, active=list(active)))
            if count != last_count:
                print("FIT_PROGRESS", count, "/", expected, flush=True)
                last_count = count
            if active:
                time.sleep(4)
        records = []
        for fold, family in itertools.product(range(5), config["new_families"]):
            local = read_json(OUT / f"fold{fold}" / family / "conditions.json")
            expected_keys = set(itertools.product(config["student_seeds"], WEIGHTS))
            keys = {(row["seed"], row["weight"]) for row in local}
            if len(local) != len(expected_keys) or keys != expected_keys:
                raise ValueError("Incomplete worker results")
            for row in local:
                verify_proof(BASE / row["model_folder"])
            records.extend(local)
        outputs = {"protocol.json": sha(OUT / "protocol.json")}
        outputs.update({
            path.relative_to(OUT).as_posix(): sha(path)
            for path in OUT.glob("fold*/**/*") if path.is_file() and path.suffix != ".tmp"
        })
        save(OUT / "fit_manifest.json", dict(status="complete", records=records,
                                           output_sha256=outputs, n_new_fits=len(records)))
        save(OUT / "progress.json", dict(status="trained", completed_new=expected,
                                        expected_new=expected, active=[]))
    finally:
        for proc in active.values():
            if proc.poll() is None:
                proc.terminate()
        for proc in active.values():
            proc.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=FEATURES)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--jobs", type=int, choices=range(1, 5), default=3)
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--family")
    run(parser.parse_args())
