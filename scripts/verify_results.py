"""Recompute the archived six-family score and timing summaries."""

import csv
import gzip
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "six-family"
sys.path.insert(0, str(ROOT / "semg_distillation_study"))
from robustness import summarize


def read_csv(name):
    with (RESULTS / name).open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def close(expected, actual, label):
    if not math.isclose(float(expected), float(actual), abs_tol=1e-10, rel_tol=0):
        raise ValueError(f"{label}: expected {expected}, found {actual}")


def main():
    config = json.loads((RESULTS / "protocol.json").read_text(encoding="utf-8"))["configuration"]
    families, seeds = config["families"], config["student_seeds"]
    weights, people = config["weights"], config["source_people"]
    students = read_csv("per_person_seed.csv")
    teachers = read_csv("teacher_per_person.csv")
    fold_for_person = {person: index // 6 for index, person in enumerate(people)}
    for row in students + teachers:
        if int(row["fold"]) != fold_for_person[row["subject"]]:
            raise ValueError("Inconsistent participant fold")
    stats = summarize(students, families, seeds, weights, people)
    expected = {(r["family"], r["weight"]): r for r in stats["summary"]}
    student_summary = read_csv("student_summary.csv")
    if len(student_summary) != len(expected):
        raise ValueError("Missing student summaries")
    for actual in student_summary:
        row = expected[actual["family"], float(actual["weight"])]
        for field in ("mae", "rmse", "seed_macro_mae_sd", "n_people", "n_student_seeds"):
            close(row[field], actual[field], field)

    expected = {r["family"]: r for r in stats["paired"]}
    person_mae = {(r["family"], r["subject"], int(r["seed"]), float(r["weight"])): float(r["mae"])
                  for r in students}
    resamples = np.random.default_rng(20261118).integers(0, len(people), (50000, len(people)))
    tail = .05 / (2 * len(families))
    paired = read_csv("paired_improvement.csv")
    if len(paired) != len(expected):
        raise ValueError("Missing paired comparisons")
    for actual in paired:
        family = actual["family"]
        for field in ("mae_improvement", "ci_low", "ci_high", "people_improved"):
            close(expected[family][field], actual[field], field)
        differences = np.array([
            np.mean([person_mae[family, p, seed, 0.0] - person_mae[family, p, seed, 0.5]
                     for seed in seeds])
            for p in people
        ])
        limits = np.quantile(differences[resamples].mean(axis=1), [tail, 1 - tail])
        close(limits[0], actual["adjusted_ci_low"], f"{family} lower adjusted bound")
        close(limits[1], actual["adjusted_ci_high"], f"{family} upper adjusted bound")

    with gzip.open(RESULTS / "inference_timings.json.gz", "rt", encoding="utf-8") as source:
        timing = json.load(source)
    for name, entry in timing.items():
        values = np.asarray(entry["raw_ms"])
        if len(values) != 200 or not np.isfinite(values).all() or not np.all(values > 0):
            raise ValueError(f"Invalid timing repeats: {name}")
        for helper, measured in (("median_ms", np.median(values)), ("p95_ms", np.quantile(values, .95))):
            close(entry[helper], measured, f"{name} {helper}")

    table = read_csv("comparison.csv")
    if len(table) != len(families) or {r["family"].lower() for r in table} != set(families):
        raise ValueError("Incomplete comparison table")
    for row in table:
        family = row["family"].lower()
        teacher = [r for r in teachers if r["family"] == family]
        if len(teacher) != len(people) or {r["subject"] for r in teacher} != set(people):
            raise ValueError("Incomplete teacher evaluation")
        for metric in ("mae", "rmse"):
            close(np.mean([float(r[metric]) for r in teacher]), row[f"teacher_{metric}"], metric)
            for method, weight in (("direct", 0.0), ("kd", 0.5)):
                actual = next(r for r in stats["summary"] if r["family"] == family and r["weight"] == weight)
                close(actual[metric], row[f"{method}_{metric}"], f"{family} {method} {metric}")
        for method, suffix, count in (("teacher", "/teacher", 5), ("direct", "/0.0", 15), ("kd", "/0.5", 15)):
            observations = [value for key, value in timing.items()
                            if key.split("/")[1] == family and key.endswith(suffix)]
            if len(observations) != count:
                raise ValueError(f"Incomplete inference log for {family} {method}")
            for metric in ("median_ms", "p95_ms"):
                close(np.median([v[metric] for v in observations]), row[f"{method}_{metric}"], metric)
        close(float(row["teacher_median_ms"]) / float(row["kd_median_ms"]),
              row["teacher_to_kd_speedup"], f"{family} speed ratio")

    if (len(students), len(teachers), len(timing)) != (1080, 180, 210):
        raise ValueError("Unexpected archive sizes")
    print("Verified six families: 1,080 student and 180 teacher metric rows, paired intervals, and 210 timing logs.")


if __name__ == "__main__":
    main()
