"""Verify the complete six-family fit grid and measure trained checkpoint latency."""

import argparse
import itertools
from pathlib import Path

import numpy as np
import torch

from audit_personcv import benchmark, check_reloaded, recompute_predictions
from data import prepare_data, sha
from robustness import summarize
from run_six_family import BASE, CACHE, FEATURES, OUT, PREVIOUS, TEACHERS, THREADS, prepare, read_json
from select_teachers import verify_tree
from selection import normalized_fold, person_folds, row_hash, save, validate_teacher, verify_proof, write_csv
from student_study import teacher_targets


def collect_records(config):
    fresh = read_json(OUT / "fit_manifest.json")
    verify_tree(OUT, fresh)
    new = [dict(row, reused=False) for row in fresh["records"]]
    prior = []
    for fold in range(5):
        for row in verify_proof(PREVIOUS / f"fold{fold}")["records"]:
            if row["weight"] in config["weights"]:
                prior.append(dict(row, reused=True))
    all_records = new + prior
    expected = set(itertools.product(range(5), config["families"], config["student_seeds"], config["weights"]))
    actual = {(row["fold"], row["family"], row["seed"], row["weight"]) for row in all_records}
    if len(prior) != 90 or len(new) != 90 or len(all_records) != len(actual) or actual != expected:
        raise ValueError("The six-family comparison grid is incomplete")
    return all_records


def check_partition(pack, fold, data, scaler):
    metadata = pack["metadata"]
    if set(metadata["source_people"]) != set(fold["train_people"]):
        raise ValueError("Wrong training participants")
    if set(metadata["validation_people"]) != set(fold["validation_people"]):
        raise ValueError("Wrong validation participants")
    for subset in ("train", "validation"):
        if metadata[f"{subset}_rows_sha256"] != row_hash(data["row_ids"][fold[f"{subset}_indices"]]):
            raise ValueError(f"Wrong {subset} rows")
    for name, value in scaler.items():
        if not torch.equal(pack["scaler"][name], torch.as_tensor(value)):
            raise ValueError(f"Wrong normalization: {name}")


def add_simultaneous_intervals(paired, rows, config):
    """Bonferroni percentile intervals over the six prespecified contrasts."""
    mae = {(r["family"], r["subject"], r["seed"], r["weight"]): r["mae"] for r in rows}
    people = config["source_people"]
    seeds = config["student_seeds"]
    indices = np.random.default_rng(20261118).integers(0, len(people), size=(50000, len(people)))
    tail = .05 / (2 * len(paired))
    for row in paired:
        family = row["family"]
        changes = np.array([
            np.mean([mae[family, person, seed, 0.0] - mae[family, person, seed, 0.5] for seed in seeds])
            for person in people
        ])
        if not np.isclose(changes.mean(), row["mae_improvement"], rtol=0, atol=1e-10):
            raise ValueError("The two paired aggregations differ")
        low, high = np.quantile(changes[indices].mean(axis=1), [tail, 1 - tail])
        row.update(adjusted_ci_low=float(low), adjusted_ci_high=float(high),
                   adjusted_scope="Bonferroni across six families, conditional person bootstrap")


def audit(features, cache):
    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(1)
    config = prepare(features)
    records = collect_records(config)
    data = prepare_data(features, cache)
    rows, teachers, timings, fit_rows, checks = [], [], {}, [], []
    for number, fold in enumerate(person_folds(config["source_people"], data["subject"])):
        train_x, _, val_x, _, _, scaler = normalized_fold(data, fold)
        vi = fold["validation_indices"]
        for family in config["families"]:
            teacher_dir = TEACHERS / f"fold{number}" / family
            teacher_path = teacher_dir / "best.pt"
            teacher_proof = verify_proof(teacher_dir)
            pack, model, error = check_reloaded(teacher_dir, family, "teacher")
            check_partition(pack, fold, data, scaler)
            validate_teacher(pack["metadata"], fold["train_people"], fold["validation_people"], family)
            teacher_rows, _ = recompute_predictions(teacher_dir, data["row_ids"][vi], data, vi)
            teachers.extend(dict(family=family, fold=number, **row) for row in teacher_rows)
            timings[f"{number}/{family}/teacher"] = benchmark(model, val_x[:1], teacher_path)
            target_root = OUT if family in config["new_families"] else BASE / "runs/student_personcv_v1"
            target_dir = target_root / f"fold{number}" / family / "targets"
            verify_proof(target_dir)
            target, target_proof = teacher_targets(
                target_dir, teacher_path, fold["train_people"], fold["validation_people"], family, scaler, train_x
            )
            with torch.inference_mode():
                np.testing.assert_allclose(model(train_x[:64]).numpy(), target[:64].numpy(), atol=5e-5, rtol=0)
            fit_rows.append(dict(
                fold=number, family=family, seed="", role="teacher", weight="",
                optimization_seconds=teacher_proof["summary"]["optimization_seconds"],
                validation_seconds=teacher_proof["summary"]["validation_seconds"],
                teacher_target_seconds=target_proof["target_generation_seconds"],
                epochs=teacher_proof["summary"]["epochs_executed"], reused=True,
            ))
            teacher_hash = sha(teacher_path)
            target_hash = sha(target_dir / "targets.npz")
            for seed in config["student_seeds"]:
                pair = [r for r in records if (r["fold"], r["family"], r["seed"]) == (number, family, seed)]
                initial_hashes = set()
                for record in pair:
                    weight = record["weight"]
                    folder = BASE / record["model_folder"]
                    checkpoint = folder / "best.pt"
                    if sha(checkpoint) != record["checkpoint_sha256"]:
                        raise ValueError("Changed trained checkpoint")
                    if sha(folder / "complete.json") != record["model_proof_sha256"]:
                        raise ValueError("Changed completion proof")
                    proof = verify_proof(folder)
                    pack, student, error = check_reloaded(folder, family, "student")
                    check_partition(pack, fold, data, scaler)
                    metadata = pack["metadata"]
                    if metadata["distillation_weight"] != weight:
                        raise ValueError("Wrong distillation weight")
                    if (metadata["teacher_sha256"], metadata["teacher_target_sha256"]) != (
                        (teacher_hash, target_hash) if weight else (None, None)
                    ):
                        raise ValueError("Wrong teacher target")
                    if not record["reused"] and metadata["configuration_sha256"] != sha(OUT / "protocol.json"):
                        raise ValueError("Wrong new training protocol")
                    initial_hashes.add(metadata["initial_weights_sha256"])
                    person_rows, _ = recompute_predictions(folder, data["row_ids"][vi], data, vi)
                    rows.extend(dict(family=family, seed=seed, weight=weight, fold=number, **r) for r in person_rows)
                    key = f"{number}/{family}/{seed}/{weight}"
                    timings[key] = benchmark(student, val_x[:1], checkpoint)
                    stats = proof["summary"]
                    fit_rows.append(dict(
                        fold=number, family=family, seed=seed, role="student", weight=weight,
                        optimization_seconds=stats["optimization_seconds"],
                        validation_seconds=stats["validation_seconds"], teacher_target_seconds="",
                        epochs=stats["epochs_executed"], reused=record["reused"],
                    ))
                    checks.append(dict(key=key, fold=number, family=family, seed=seed, weight=weight,
                                       reloaded_difference_deg=error, reused=record["reused"]))
                if len(initial_hashes) != 1:
                    raise ValueError("Unpaired initial weights")
        print("AUDITED_FOLD", number, len(checks), flush=True)
    stats = summarize(rows, config["families"], config["student_seeds"], config["weights"],
                      config["source_people"])
    add_simultaneous_intervals(stats["paired"], rows, config)
    expected_counts = (1080, 180, 180, 210)
    if (len(rows), len(teachers), len(checks), len(timings)) != expected_counts:
        raise ValueError("Incomplete checkpoint audit")
    for name, table in (
        ("student_summary", stats["summary"]), ("per_seed", stats["per_seed"]),
        ("paired_improvement", stats["paired"]), ("per_person_seed", rows),
        ("teacher_per_person", teachers), ("training_times", fit_rows),
    ):
        write_csv(OUT / f"{name}.csv", table)
    save(OUT / "inference_timings.json", timings)
    save(OUT / "checkpoint_index.json", records)
    saved_files = (
        "student_summary.csv", "per_seed.csv", "paired_improvement.csv", "per_person_seed.csv",
        "teacher_per_person.csv", "training_times.csv", "inference_timings.json", "checkpoint_index.json",
    )
    save(OUT / "verification.json", dict(
        status="passed", n_new_fits=90, n_reused_fits=90, n_families=6, n_people=30,
        student_conditions=len(checks), student_metric_rows=len(rows), teacher_metric_rows=len(teachers),
        timed_checkpoints=len(timings), teacher_training_targets_checked=30 * 64,
        student_prediction_examples_per_checkpoint=64, teacher_seed_varied=False,
        independent_test_executed=False, checks=checks,
        audit_sha256=sha(__file__), fit_manifest_sha256=sha(OUT / "fit_manifest.json"),
        output_sha256={name: sha(OUT / name) for name in saved_files},
    ))
    print("SIX_FAMILY_AUDIT_PASSED", len(checks), len(rows), len(timings), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=FEATURES)
    parser.add_argument("--cache", type=Path, default=CACHE)
    args = parser.parse_args()
    audit(args.features, args.cache)
