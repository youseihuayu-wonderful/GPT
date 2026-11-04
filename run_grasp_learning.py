"""Level-1 physical data -> random forest -> three-speed advice -> fresh execution."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path
import platform
import random
import shutil
import time

import joblib
import numpy as np

from benchmark import verify_pairing, write_report
from grasp_speed_model import (
    FEATURE_NAMES, FOREST_SETTINGS, SPEEDS_M_S,
    predict_candidates, select_speed, train_classifier,
)
from grasp_trial import REQUIRED_HOLD_S, SAMPLE_PERIOD_S, SETTLE_S, run_trial
from scenarios import SceneCase


ROOT = Path(__file__).resolve().parent
DEFAULT_SEED = 5644
DEFAULT_THRESHOLD = 0.9


def build_protocol(seed: int = DEFAULT_SEED, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Freeze all condition groups and the speed policy before study labels."""
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an unsigned 32-bit integer")
    select_speed([
        {"speed_m_s": speed, "predicted_success_probability": 0.5}
        for speed in SPEEDS_M_S
    ], threshold=threshold)
    grid = list(itertools.product((-.08, 0., .08), (-.08, 0., .08), (-.2, 0., .2)))
    rng = random.Random(seed)
    training, evaluation = [], []
    for target in ("bowl", "bottle"):
        training_indices = sorted(rng.sample(range(len(grid)), 18))
        for index in training_indices:
            x, y, yaw = grid[index]
            case = SceneCase(f"{target}.condition_{index:02d}", target, "training", seed, (x, y), yaw)
            training.append(case.to_dict())
        # New continuous poses avoid reusing the grid states observed during
        # development probes. Both objects still occur in both splits.
        seen = set(grid)
        for index in range(9):
            while True:
                values = (
                    round(rng.uniform(-.08, .08), 6),
                    round(rng.uniform(-.08, .08), 6),
                    round(rng.uniform(-.2, .2), 6),
                )
                if values not in seen:
                    seen.add(values)
                    break
            x, y, yaw = values
            case = SceneCase(f"{target}.evaluation_{index:02d}", target, "evaluation", seed, (x, y), yaw)
            evaluation.append(case.to_dict())
    train_states = {(c["target_object"], *c["offset_xy_m"], c["yaw_rad"]) for c in training}
    eval_states = {(c["target_object"], *c["offset_xy_m"], c["yaw_rad"]) for c in evaluation}
    if train_states & eval_states:
        raise ValueError("Training and evaluation conditions overlap")
    return {
        "schema_version": 1,
        "study": "proposal_level1_grasp_speed_pilot",
        "seed": seed,
        "speeds_m_s": list(SPEEDS_M_S),
        "threshold": threshold,
        "required_hold_s": REQUIRED_HOLD_S,
        "minimum_lift_m": 0.10,
        "settle_s": SETTLE_S,
        "trace_sample_period_s": SAMPLE_PERIOD_S,
        "fixed_initial_opening_m": {"bowl": .08, "bottle": .04},
        "fixed_close_trigger_height_m": 0.0,
        "feature_names": list(FEATURE_NAMES),
        "forest_settings": dict(FOREST_SETTINGS),
        "training_cases": training,
        "evaluation_cases": evaluation,
        "planned_training_trials": len(training) * len(SPEEDS_M_S),
        "planned_evaluation_trials": len(evaluation) * len(SPEEDS_M_S),
        "initial_sampling": "Seeded random selection of 18 of 27 condition groups per object; all three speeds within each selected group",
        "evaluation_sampling": "Nine independently sampled continuous XY/yaw conditions per object, excluding all development/grid states",
        "split_scope": "Held-out initial conditions of the SAME TWO objects; NOT an unseen-object test",
        "prediction_timing": "Persist all evaluation predictions and recommendations before any evaluation execution",
        "evaluation_order": "For each held-out condition execute the recommended speed first, then the other two speeds; never retrain on evaluation outcomes",
        "label_rule": "Controller completes and independent physical predicates remain true for two continuous simulation seconds; physical failures are negative, invalid setup is unlabelled, numerical/internal errors abort",
        "collision_rule": "Unexpected contacts are recorded separately; grasp-success labels are not robot-safety guarantees",
        "uncalibrated": True,
        "active_learning_comparison": False,
        "d_optimal_comparison": False,
        "physics_modified": False,
    }


def scene_case(values: dict) -> SceneCase:
    return SceneCase(**{**values, "offset_xy_m": tuple(values["offset_xy_m"])})


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance() -> dict:
    sources = [ROOT / name for name in (
        "run_grasp_learning.py", "grasp_trial.py", "grasp_speed_model.py", "benchmark.py",
        "simulation.py", "grasp.py", "storage.py", "scenarios.py", "collisions.py",
        "task_specs.py", "scene.xml", "pyproject.toml", "uv.lock",
    )]
    sources.extend(sorted(path for path in (ROOT / "vendor").rglob("*") if path.is_file()))
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in (
            "mujoco", "numpy", "scikit-learn", "joblib",
        )},
        "input_sha256": {str(path.relative_to(ROOT)): file_sha256(path) for path in sources},
    }


def outcome_counts(rows: list[dict]) -> dict:
    completed = [row for row in rows if row["trial_status"] == "completed"]
    durations = [row["measurement_wall_s"] for row in rows]
    return {
        "attempted": len(rows), "labelled": len(completed),
        "successes": sum(row["success"] is True for row in completed),
        "failures": sum(row["success"] is False for row in completed),
        "invalid_initial_states": sum(row["trial_status"] == "invalid_initial_state" for row in rows),
        "failure_reasons": dict(Counter(row["failure_reason"] for row in completed if not row["success"])),
        "invalid_initial_state_reasons": dict(Counter(
            row["failure_reason"] for row in rows if row["trial_status"] == "invalid_initial_state"
        )),
        "sum_collection_wall_s": sum(durations),
        "mean_collection_wall_s": float(np.mean(durations)) if durations else None,
        "min_collection_wall_s": min(durations) if durations else None,
        "max_collection_wall_s": max(durations) if durations else None,
        "unexpected_contact_trials": sum(row["collisions"]["episodes"] > 0 for row in completed),
        "numerical_warning_trials": sum(bool(row["warnings"]) for row in rows),
    }


def evaluate_predictions(rows: list[dict]) -> dict:
    labelled = [row for row in rows if row["trial_status"] == "completed"]
    if not labelled:
        return {"labelled": 0, "accuracy_at_0_5": None, "brier_score": None}
    labels = np.array([row["success"] for row in labelled], dtype=bool)
    probabilities = np.array([row["predicted_success_probability"] for row in labelled])
    predicted = probabilities >= .5
    return {
        "labelled": len(labelled),
        "accuracy_at_0_5": float(np.mean(predicted == labels)),
        "brier_score": float(np.mean((probabilities - labels.astype(float)) ** 2)),
        "true_negatives": int(np.sum(~labels & ~predicted)),
        "false_positives": int(np.sum(~labels & predicted)),
        "false_negatives": int(np.sum(labels & ~predicted)),
        "true_positives": int(np.sum(labels & predicted)),
        "scope": "Descriptive held-out-condition results, with three correlated speeds per condition; no unseen-object or calibrated-probability claim",
    }


def compare_speeds(rows: list[dict]) -> dict:
    groups = {}
    for row in rows:
        groups.setdefault(row["case"]["case_id"], []).append(row)
    matched = [group for group in groups.values() if (
        len(group) == len(SPEEDS_M_S)
        and all(row["trial_status"] == "completed" for row in group)
        and {row["parameters"]["descent_speed_m_s"] for row in group} == set(SPEEDS_M_S)
    )]
    methods = {}
    for method in ("recommended", *SPEEDS_M_S):
        selected = [next(row for row in group if (
            row["recommended"] if method == "recommended"
            else row["parameters"]["descent_speed_m_s"] == method
        )) for group in matched]
        successes = [row for row in selected if row["success"]]
        methods[str(method)] = {
            "attempted": len(selected), "successes": len(successes),
            "success_rate": len(successes) / len(selected) if selected else None,
            "mean_success_execution_sim_s": float(np.mean([
                row["execution_sim_s"] for row in successes
            ])) if successes else None,
            "meanings": "Mean duration is success-only; differing success sets are not a causal speedup comparison",
        }
    both_success_deltas = []
    faster_fails_slower_succeeds = []
    mixed_outcomes = []
    for group in matched:
        fastest = next(row for row in group if row["parameters"]["descent_speed_m_s"] == max(SPEEDS_M_S))
        chosen = next(row for row in group if row["recommended"])
        if fastest["success"] and chosen["success"]:
            both_success_deltas.append(chosen["execution_sim_s"] - fastest["execution_sim_s"])
        if len({row["success"] for row in group}) > 1:
            mixed_outcomes.append(group[0]["case"]["case_id"])
        if not fastest["success"] and any(row["success"] for row in group):
            faster_fails_slower_succeeds.append(group[0]["case"]["case_id"])
    return {
        "matched_valid_conditions": len(matched),
        "methods": methods,
        "speed_dependent_outcome_conditions": mixed_outcomes,
        "fastest_failed_but_slower_succeeded_conditions": faster_fails_slower_succeeds,
        "recommended_minus_fastest_on_joint_success": {
            "pairs": len(both_success_deltas),
            "mean_sim_s": float(np.mean(both_success_deltas)) if both_success_deltas else None,
        },
    }


def run_study(output: Path, *, seed: int = DEFAULT_SEED, threshold: float = DEFAULT_THRESHOLD,
              development_evidence: tuple[Path, ...] = ()) -> dict:
    protocol = build_protocol(seed, threshold)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {
        "kind": "proposal_level1_grasp_speed_pilot", "status": "running", "stage": "collect_training",
        "started_at": datetime.now(timezone.utc).isoformat(), "protocol": "protocol.json",
        "training": {}, "evaluation": {}, "selected_execution": {}, "development_evidence": [],
        "attempts_started": 0, "active_trial": None,
        "limitations": [
            "Only two existing object assets; held-out conditions are not held-out objects",
            "Random-forest probabilities are not calibrated and do not guarantee safety",
            "No active-learning, D-optimal, GPT, 27B, or MRIoC comparison is performed",
            "Controller timeouts and contact failures are retained and must be interpreted separately",
            "Numerical/internal errors are not negative grasp labels",
        ],
    }
    training_rows, evaluation_rows = [], []
    fingerprints = {}

    def checkpoint():
        report["elapsed_wall_s"] = time.monotonic() - started
        report["training"] = outcome_counts(training_rows)
        report["evaluation"] = outcome_counts(evaluation_rows)
        report["selected_execution"] = outcome_counts([row for row in evaluation_rows if row["recommended"]])
        write_report(output / "summary.json", report)

    def collect(case: SceneCase, speed: float, split: str, **prediction) -> None:
        trial_id = f"{case.case_id}.speed_{round(speed * 1000):03d}"
        trace = output / "traces" / f"{trial_id}.json.gz"
        report["active_trial"] = {
            "trial_id": trial_id, "case": case.to_dict(), "split": split,
            "speed_m_s": speed, "recommended": prediction.get("recommended", False),
            "trace_path": str(trace.relative_to(output)), "evidence_recorded": False,
        }
        report["attempts_started"] += 1
        checkpoint()
        trial_started = time.monotonic()
        row = run_trial(case, speed, trace_path=trace)
        row.update(trial_id=trial_id, split=split, **prediction)
        verify_pairing(case, row["initial_state_fingerprint"], fingerprints)
        if row["success"] is True and (
            row["verified_hold_s"] < REQUIRED_HOLD_S or not all(row["postconditions"].values())
        ):
            raise RuntimeError("Collector reported success without verified physical conditions")
        row["trace_path"] = str(trace.relative_to(output))
        row["measurement_wall_s"] = time.monotonic() - trial_started
        write_report(output / "trials" / f"{trial_id}.json", row)
        (training_rows if split == "training" else evaluation_rows).append(row)
        report["active_trial"]["evidence_recorded"] = True
        checkpoint()
        print(json.dumps({"stage": split, "trial_id": trial_id,
                          "success": row["success"], "status": row["trial_status"],
                          "reason": row["failure_reason"], "wall_s": row["measurement_wall_s"]},
                         ensure_ascii=False), flush=True)
        report["active_trial"] = None

    try:
        write_report(output / "protocol.json", protocol)
        write_report(output / "environment.json", provenance())
        for index, path in enumerate(development_evidence):
            source = Path(path)
            destination = output / "development" / f"{index:02d}_{source.name}"
            destination.parent.mkdir(exist_ok=True)
            shutil.copy2(source, destination)
            report["development_evidence"].append({
                "source": str(source), "copy": str(destination.relative_to(output)),
                "sha256": file_sha256(destination), "used_for_training": False,
                "note": "Earlier feasibility/development observations, excluded from held-out evaluation; retained for cost transparency",
            })
        checkpoint()
        for values in protocol["training_cases"]:
            case = scene_case(values)
            for speed in SPEEDS_M_S:
                collect(case, speed, "training")
                checkpoint()
        write_report(output / "training_data.json", {"rows": training_rows})
        valid_training = [row for row in training_rows if row["trial_status"] == "completed"]
        report["stage"] = "train_classifier"
        checkpoint()
        train_started = time.monotonic()
        model = train_classifier(valid_training, seed=seed)
        report["training_wall_s"] = time.monotonic() - train_started
        model_path = output / "random_forest.joblib"
        joblib.dump(model, model_path, compress=3)
        restored_model = joblib.load(model_path)  # Only load the artifact just created in this new directory.
        model_info = {
            "feature_names": list(FEATURE_NAMES), "settings": model.get_params(),
            "classes": model.classes_.tolist(), "training_labels": outcome_counts(training_rows),
            "calibrated": False, "sha256": file_sha256(model_path),
        }
        write_report(output / "model_metadata.json", model_info)
        report["model_sha256"] = model_info["sha256"]
        recommendations = []
        for values in protocol["evaluation_cases"]:
            case = scene_case(values)
            candidates = predict_candidates(restored_model, case)
            if candidates != predict_candidates(model, case):
                raise RuntimeError("Saved model does not reproduce the trained model's predictions")
            recommendations.append({
                "case": case.to_dict(), "candidates": candidates,
                **select_speed(candidates, threshold=threshold),
            })
        # This file is never rewritten with outcomes: all advice precedes evaluation labels.
        write_report(output / "recommendations.json", {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model_sha256": model_info["sha256"], "calibrated": False,
            "evaluation_has_started": False, "recommendations": recommendations,
        })
        report["recommendations"] = len(recommendations)
        report["unmet_threshold_recommendations"] = sum(not item["threshold_met"] for item in recommendations)
        report["stage"] = "execute_recommendations_and_speed_controls"
        checkpoint()
        for recommendation in recommendations:
            case = scene_case(recommendation["case"])
            selected_speed = recommendation["selected_speed_m_s"]
            probabilities = {item["speed_m_s"]: item["predicted_success_probability"]
                             for item in recommendation["candidates"]}
            order = [selected_speed, *(speed for speed in SPEEDS_M_S if speed != selected_speed)]
            for speed in order:
                collect(
                    case, speed, "evaluation", recommended=speed == selected_speed,
                    predicted_success_probability=probabilities[speed],
                    threshold=threshold, threshold_met=probabilities[speed] >= threshold,
                )
                checkpoint()
        write_report(output / "evaluation_data.json", {"rows": evaluation_rows})
        write_report(output / "recommendation_outcomes.json", {
            "predictions_source": "recommendations.json",
            "rows": [row for row in evaluation_rows if row["recommended"]],
        })
        report["prediction_metrics"] = evaluate_predictions(evaluation_rows)
        report["speed_comparison"] = compare_speeds(evaluation_rows)
        selected_valid = [row for row in evaluation_rows if row["recommended"] and row["trial_status"] == "completed"]
        report["level1_checks"] = {
            "real_positive_and_negative_training_labels": {row["success"] for row in valid_training} == {False, True},
            "trained_model_saved_and_reloaded": True,
            "three_probabilities_for_every_evaluation_condition": all(len(item["candidates"]) == 3 for item in recommendations),
            "recommendations_saved_before_evaluation": True,
            "all_recommendations_physically_checked": len(selected_valid) == len(recommendations),
            "at_least_one_recommended_grasp_physically_succeeded": any(row["success"] for row in selected_valid),
            "initial_states_match_across_speeds": True,
        }
        report["level1_completed"] = all(report["level1_checks"].values())
        report["status"] = "completed" if report["level1_completed"] else "completed_without_level1_acceptance"
        report["stage"] = "finished"
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        checkpoint()
        print("SUMMARY", json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)
        print("EVIDENCE", output / "summary.json", flush=True)
        return report
    except BaseException as exc:
        report["status"] = "aborted"
        report["error"] = {"type": type(exc).__name__, "message": str(exc),
                           "trial": report["active_trial"]}
        try:
            checkpoint()
        except Exception as save_error:
            exc.add_note(f"Could not save aborted report: {save_error}")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New output directory; existing directories are rejected")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--development-evidence", type=Path, action="append", default=[],
                        help="Archive an existing development record without training on it")
    args = parser.parse_args()
    report = run_study(args.output, seed=args.seed, threshold=args.threshold,
                       development_evidence=tuple(args.development_evidence))
    if not report["level1_completed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
