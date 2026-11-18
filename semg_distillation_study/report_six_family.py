"""Export readable results and figures for the matched six-family comparison."""

from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator
import numpy as np

from data import sha
from models import make_model, parameter_count
from run_six_family import BASE, OUT, read_json
from select_teachers import verify_tree
from selection import read_csv, save, write_csv


def combined_table(families):
    teachers = read_csv(OUT / "teacher_per_person.csv")
    students = read_csv(OUT / "student_summary.csv")
    paired = {row["family"]: row for row in read_csv(OUT / "paired_improvement.csv")}
    times = read_json(OUT / "inference_timings.json")
    combined = []
    for family in families:
        one = dict(family=family.upper())
        for method, suffix, role in (("teacher", "/teacher", "teacher"),
                                     ("direct", "/0.0", "student"), ("kd", "/0.5", "student")):
            if method == "teacher":
                data = [row for row in teachers if row["family"] == family]
                if len({row["subject"] for row in data}) != 30 or len(data) != 30:
                    raise ValueError("Incomplete teacher participant table")
                for metric in ("mae", "rmse"):
                    one[f"{method}_{metric}"] = float(np.mean([float(row[metric]) for row in data]))
            else:
                match = next(row for row in students if row["family"] == family
                             and float(row["weight"]) == (0.0 if method == "direct" else 0.5))
                for metric in ("mae", "rmse", "seed_macro_mae_sd"):
                    one[f"{method}_{metric}"] = float(match[metric])
            observations = [r for key, r in times.items()
                            if key.split("/")[1] == family and key.endswith(suffix)]
            if len(observations) != (5 if method == "teacher" else 15):
                raise ValueError("Incomplete timing table")
            for metric in ("median_ms", "p95_ms"):
                one[f"{method}_{metric}"] = float(np.median([r[metric] for r in observations]))
            one[f"{method}_parameters"] = parameter_count(make_model(family, role))
        for name in ("mae_improvement", "ci_low", "ci_high", "adjusted_ci_low", "adjusted_ci_high"):
            one[name] = float(paired[family][name])
        one["people_improved"] = int(paired[family]["people_improved"])
        one["teacher_to_kd_speedup"] = one["teacher_median_ms"] / one["kd_median_ms"]
        combined.append(one)
    return combined


def figure(rows):
    families = [row["family"] for row in rows]
    positions = np.arange(len(rows))
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.3), sharey=True)
    fig.subplots_adjust(left=.115, right=.983, top=.79, bottom=.19, wspace=.24)
    styles = (("teacher", "#236284", "o", "Large teacher"),
              ("direct", "#925424", "s", "Small direct"),
              ("kd", "#297452", "^", "Small KD, weight 0.5"))
    for offset, (method, color, marker, label) in zip((-0.19, 0, 0.19), styles):
        axes[0].scatter([r[f"{method}_mae"] for r in rows], positions + offset,
                        c=color, marker=marker, label=label, s=46)
        axes[2].scatter([r[f"{method}_median_ms"] for r in rows], positions + offset,
                        c=color, marker=marker, s=46)
    for index, row in enumerate(rows):
        axes[1].plot([row["adjusted_ci_low"], row["adjusted_ci_high"]], [index, index],
                     color="#98b1c1", lw=5, solid_capstyle="round")
        axes[1].plot([row["ci_low"], row["ci_high"]], [index, index], color="#285c79", lw=2)
        axes[1].scatter(row["mae_improvement"], index, c="#285c79", s=37)
    axes[1].axvline(0, color="#545c64", linestyle="--", lw=1)
    axes[0].set(xlabel="Participant mean MAE (degrees)", title="Angle error")
    axes[1].set(xlabel="Direct MAE minus KD MAE (degrees)", title="Paired KD effect")
    axes[2].set(xlabel="Median forward time (ms, log scale)", title="CPU inference", xscale="log")
    axes[2].xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    axes[2].xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axes[2].xaxis.set_minor_locator(NullLocator())
    for axis in axes:
        axis.set(yticks=positions, ylim=(len(rows)-0.5, -0.5), axisbelow=True)
        axis.grid(axis="x", color="#e6e9ec")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticklabels(families)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5, .925),
               ncol=3, frameon=False, fontsize=10)
    fig.suptitle("Six model families · 30 development participants · 3 student seeds", y=.975)
    fig.text(.5, .047, "Dark line: individual 95% interval   ·   Light line: adjusted for six comparisons",
             ha="center", fontsize=10, color="#273648")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"six_family_comparison.{ext}", dpi=220, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)


def training_summary(families):
    raw = read_csv(OUT / "training_times.csv")
    result = []
    for family in families:
        teachers = {r["fold"]: r for r in raw if r["family"] == family and r["role"] == "teacher"}
        direct = [r for r in raw if r["family"] == family and r["weight"] == "0.0"]
        kd = [r for r in raw if r["family"] == family and r["weight"] == "0.5"]
        if len(teachers) != 5 or len(direct) != 15 or len(kd) != 15:
            raise ValueError("Incomplete training timing table")

        def fit_seconds(record):
            return float(record["optimization_seconds"]) + float(record["validation_seconds"])

        kd_from_scratch = [
            fit_seconds(teachers[r["fold"]]) + float(teachers[r["fold"]]["teacher_target_seconds"])
            + fit_seconds(r) for r in kd
        ]
        result.append(dict(
            family=family.upper(),
            teacher_fit_seconds=float(np.median([fit_seconds(r) for r in teachers.values()])),
            target_generation_seconds=float(np.median([float(r["teacher_target_seconds"]) for r in teachers.values()])),
            direct_student_fit_seconds=float(np.median([fit_seconds(r) for r in direct])),
            kd_student_fit_seconds=float(np.median([fit_seconds(r) for r in kd])),
            kd_from_scratch_measured_stage_sum_seconds=float(np.median(kd_from_scratch)),
        ))
    return result


def report():
    protocol = read_json(OUT / "protocol.json")["configuration"]
    verified = read_json(OUT / "verification.json")
    verify_tree(OUT, verified)
    verify_tree(OUT, read_json(OUT / "fit_manifest.json"))
    if verified["status"] != "passed" or verified["fit_manifest_sha256"] != sha(OUT / "fit_manifest.json"):
        raise ValueError("Need the completed six-family audit")
    rows = combined_table(protocol["families"])
    write_csv(OUT / "comparison.csv", rows)
    train_times = training_summary(protocol["families"])
    write_csv(OUT / "training_summary.csv", train_times)
    figure(rows)
    lower_mae = sum(r["mae_improvement"] > 0 for r in rows)
    best_absolute = min(rows, key=lambda row: row["kd_mae"])
    adjusted = [r for r in rows if r["adjusted_ci_low"] > 0]
    survivors = ", ".join(r["family"] for r in adjusted) or "해당 모델 없음"
    lines = [
        "# sEMG 팔꿈치 각도 추정 — 6종 지식 증류 비교", "",
        "큰 모델 6종 각각을 교사로 사용해 같은 계열의 작은 학생을 학습했다. "
        "총 180개 학생 조건 중 LSTM·TCN·CNN1D의 90개를 새로 학습했고, "
        "GRU·MLP·Transformer의 이전 90개를 검증 후 재사용했다. "
        "교사 30개와 학생 180개 체크포인트에서 추론 시간을 다시 측정했다.", "",
        f"증류 후 평균 MAE가 낮아진 모델은 {lower_mae}/6종이다. "
        f"가장 낮은 증류 MAE는 {best_absolute['family']}의 {best_absolute['kd_mae']:.3f}°다. "
        f"여섯 비교를 함께 보정했을 때 개선 구간 전체가 0보다 큰 계열은 {survivors}였다.", "",
        "| 모델 | 큰 교사 MAE | 작은 직접 MAE | 작은 증류 MAE | 작은 직접 → 증류 RMSE | MAE 감소 (조건부 95% 구간) | 개선 인원 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['family']} | {r['teacher_mae']:.3f}° | {r['direct_mae']:.3f}° | "
            f"{r['kd_mae']:.3f}° | {r['direct_rmse']:.3f}° → {r['kd_rmse']:.3f}° | "
            f"{r['mae_improvement']:+.3f}° [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] | "
            f"{r['people_improved']}/30 |"
        )
    lines.extend([
        "",
        "MAE 감소는 직접 학습 MAE에서 증류 MAE를 뺀 값이다. 양수가 증류에 유리하다. "
        "각 사람의 seed별 차이를 먼저 평균한 다음 30명을 재표집했다. 표의 구간은 개별 95% 구간이다. "
        "아래에 여섯 비교를 함께 고려한 구간을 덧붙였다. 모델의 최적 체크포인트를 고른 분할에서 점수도 계산했다. "
        "독립 10·20명 평가는 추가 자료가 확보되면 진행할 수 있다.", "",
        "| 모델 | MAE 감소의 다중 비교 보정 구간 (°) |",
        "|---|---:|",
    ])
    for r in rows:
        lines.append(f"| {r['family']} | [{r['adjusted_ci_low']:+.3f}, {r['adjusted_ci_high']:+.3f}] |")
    lines.extend([
        "",
        "보정 구간은 사람 단위 부트스트랩 50,000회에 여섯 비교의 Bonferroni 보정을 적용했다. "
        "두 종류의 구간 모두 고정 교사와 개발 자료에 조건부이며, 체크포인트 선택 및 겹치는 훈련 분할의 "
        "불확실성은 반영하지 않는다.", "",
        "| 모델 | 큰 교사(ms) | 작은 직접(ms) | 작은 증류(ms) | 교사 / 증류 시간 |",
        "|---|---:|---:|---:|---:|",
    ])
    for r in rows:
        lines.append(f"| {r['family']} | {r['teacher_median_ms']:.3f} | {r['direct_median_ms']:.3f} | "
                     f"{r['kd_median_ms']:.3f} | {r['teacher_to_kd_speedup']:.2f}× |")
    lines.extend([
        "",
        "CPU 4스레드, float32, 배치 1에서 체크포인트마다 준비 30회와 측정 200회를 실행했다. "
        "표는 각 체크포인트 중앙값의 중앙값이다. 전처리와 센서 시간은 측정 범위 밖이다. "
        "교사와 학생의 시간 차이는 작은 구조의 속도 이점을 보여 준다. "
        "증류 효과의 주 비교는 동일 구조의 직접 학습과 증류다. "
        "비교표 CSV에는 파라미터 수와 p95 요약도 있다.", "",
        "![6종의 오차, 증류 효과, 추론 시간](six_family_comparison.png)", "",
        "| 모델 | 교사 학습(초) | 교사 예측값 생성(초) | 직접 학생 학습(초) | 증류 학생 학습(초) | 새 교사부터 증류까지 단계 합(초) |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    time_columns = ("teacher_fit_seconds", "target_generation_seconds", "direct_student_fit_seconds",
                    "kd_student_fit_seconds", "kd_from_scratch_measured_stage_sum_seconds")
    for row in train_times:
        cells = [row["family"], *(f"{row[key]:.1f}" for key in time_columns)]
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend([
        "",
        "기록된 최적화와 검증 시간을 합쳤다. 마지막 열은 같은 분할과 seed의 실제 기록에서 "
        "교사 학습·예측값 생성·증류 학생 학습 시간을 더한 뒤 중앙값을 구했다. "
        "병렬 실행의 영향을 받은 단계 시간이며, 프로그램 시작부터 끝까지 잰 시간과는 구분한다.", "",
        "## 설계와 재현", "",
        "- 한 분할의 24명으로 교사와 학생을 학습하고, 나머지 6명으로 개발 평가했다. "
        "다섯 분할에서 30명이 평가에 한 번씩 등장한다. 정규화는 분할별 학습 인원에서 계산했다.",
        "- 학생 seed 3개를 썼다. 같은 모델·분할·seed의 두 학생은 초기값과 배치 순서를 공유한다. "
        "증류 손실은 표준화한 각도의 정답 MSE 50%와 교사 예측 MSE 50%를 합했다.",
        "- 0.5는 이전 실험에서 공통으로 실행한 값이다. 기존 상위 3종의 여러 가중치 탐색은 "
        "별도 결과로 남아 있다. 이번 교사는 분할별로 고정됐다.",
        "- 1,080개 학생 사람·seed별 오차와 180개 교사 사람별 오차를 저장 예측으로 재계산했다. "
        "체크포인트마다 실제 입력 예제 64개로 모델 재실행을 확인했다.",
        "- 학습 시간은 training_times.csv에 교사 최적화, 교사 표적 생성, 학생 최적화, 검증으로 기록했다. "
        "병렬 실행과 측정 시점이 섞여 있어 전체 시간 절감의 통제 실험으로 쓰기 어렵다.",
        "- 프로젝트 폴더에서 python -B run_six_family.py, python -B audit_six_family.py, "
        "python -B report_six_family.py 순서로 실행한다. 원본 특징과 구조는 FEATURE_CONTRACT.md를 따른다.",
        "",
    ])
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    state_file = BASE / "research_state.json"
    state = read_json(state_file)
    state["six_family_distillation"] = dict(
        status="verified", directory=OUT.relative_to(BASE).as_posix(), n_families=6,
        new_fits=90, reused_fits=90, weight=0.5, student_seeds=3, independent_test_executed=False,
    )
    state["updated_utc"] = datetime.now(timezone.utc).isoformat()
    state["next_action"] = (
        "Freeze candidate models and evaluate on compatible independent participants. "
        "Use group-nested cross-validation for additional internal evidence if needed."
    )
    save(state_file, state)
    for destination in (BASE / "README.md", BASE / "AI_HANDOFF.md", BASE.parent / "README.md"):
        text = destination.read_text(encoding="utf-8")
        marker = "<!-- SIX_FAMILY_LATEST -->"
        if marker in text:
            continue
        link = Path("runs") / OUT.name / "README.md"
        if destination.parent != BASE:
            link = Path(BASE.name) / link
        note = (
            f"{marker}\n"
            f"**최신 결과는 [6종 비교 보고서]({link.as_posix()})에 있습니다.** "
            "LSTM·TCN·CNN1D를 추가했고, 여섯 계열의 직접 학습과 가중치 0.5 증류를 비교했습니다. "
            "아래 내용은 이전 3종 실험 기록입니다.\n\n"
        )
        destination.write_text(note + text, encoding="utf-8")
    print("SIX_FAMILY_REPORT_COMPLETE", flush=True)


if __name__ == "__main__":
    report()
