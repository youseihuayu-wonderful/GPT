"""Paired one-decision grasp comparison, not a closed-loop planning evaluation."""

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
from types import MappingProxyType

from experiments import SAMPLE_PERIOD_S, VALIDATION_HOLD_S
from grasp import GraspParameters
from planner import PlannerConfig, PlannerError, request_decision, validate_context, validate_decision
from scenarios import POSITION_OFFSETS, TARGETS, SceneCase, benchmark_cases
from simulation import Simulation
from task_specs import TASKS


ROOT = Path(__file__).resolve().parent
METHODS = ("fixed", "rule", "gpt")
DEFAULT_SEEDS = (101, 202, 303)
SETTLE_S = 0.5
COLLISION_FORCE_THRESHOLD_N = 0.1
# Frozen before evaluation: phase-3 bowl calibration and conservative bottle neck policy.
# Neither policy is learned from the evaluation matrix or previous benchmark rows.
HANDCRAFTED_PARAMETERS = MappingProxyType({
    "bowl": GraspParameters(0.08, 0.20, 0.004),
    "bottle": GraspParameters(0.04, 0.12, 0.0),
})


def initial_state_fingerprint(sim: Simulation) -> str:
    """Hash every qpos/qvel/ctrl value after the common settle, without rounding."""
    values = {name: getattr(sim.data, name).tolist() for name in ("qpos", "qvel", "ctrl")}
    serialized = json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def verify_pairing(case: SceneCase, fingerprint: str, fingerprints: dict[str, str]) -> None:
    previous = fingerprints.setdefault(case.case_id, fingerprint)
    if previous != fingerprint:
        raise RuntimeError(f"Paired initial state mismatch for {case.case_id}; comparison aborted")


def _mean(rows: list[dict], field: str) -> float | None:
    values = [row[field] for row in rows if row[field] is not None]
    return sum(values) / len(values) if values else None


def summarize_runs(runs: list[dict], methods: tuple[str, ...] = METHODS) -> list[dict]:
    summaries = []
    for method in methods:
        rows = [row for row in runs if row["method"] == method]
        successful = [row for row in rows if row["success"]]
        measured_collisions = [row["collisions"] for row in rows if row["collisions"] is not None]
        summaries.append({
            "method": method,
            "attempts": len(rows),
            "successes": len(successful),
            "success_rate": len(successful) / len(rows) if rows else None,
            "executed_runs": sum(row["execution_sim_s"] is not None for row in rows),
            "planning_failures": sum(row["status"] in ("planning_failed", "unsupported") for row in rows),
            "collision_runs": sum(value["episodes"] > 0 for value in measured_collisions)
            if measured_collisions else None,
            "total_collision_episodes": sum(value["episodes"] for value in measured_collisions)
            if measured_collisions else None,
            "mean_success_execution_sim_s": _mean(successful, "execution_sim_s"),
            "mean_execution_wall_s": _mean(rows, "execution_wall_s"),
            "mean_planning_wall_s": _mean(rows, "planning_wall_s"),
            "mean_end_to_end_wall_s": _mean(rows, "end_to_end_wall_s"),
            "mean_gpt_latency_s": _mean(rows, "gpt_latency_s"),
        })
    return summaries


def summarize_by_target(runs: list[dict], methods: tuple[str, ...] = METHODS) -> dict:
    return {
        target: summarize_runs([row for row in runs if row["case"]["target_object"] == target], methods)
        for target in TARGETS
    }


async def run_case(
    case: SceneCase, method: str, fingerprints: dict[str, str], experiments: dict, root: Path = ROOT,
) -> dict:
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    started = time.monotonic()
    row = {
        "case": case.to_dict(), "method": method, "status": "planning_failed", "success": False,
        "parameters": None, "decision": None, "model": None, "thread_id": None,
        "planning_wall_s": None, "gpt_latency_s": None,
        "execution_sim_s": None, "execution_wall_s": None, "validation_sim_s": None,
        "end_to_end_wall_s": None, "collisions": None, "failure_reason": None,
        "warnings": [], "initial_state_fingerprint": None,
    }
    sim = Simulation(case)
    try:
        sim.step(SETTLE_S)
        row["initial_state_fingerprint"] = initial_state_fingerprint(sim)
        verify_pairing(case, row["initial_state_fingerprint"], fingerprints)
        scene = sim.state()
        row["warnings"] = scene["warnings"]
        planning_started = time.monotonic()
        try:
            if method == "fixed":
                parameters = GraspParameters()
            elif method == "rule":
                parameters = HANDCRAFTED_PARAMETERS[case.target_object]
            else:
                config = PlannerConfig.load(root)
                row["model"] = config.model
                request_started = time.monotonic()
                try:
                    result = await request_decision(
                        config, f"抓起当前选定的{TARGETS[case.target_object].label}并稳定保持。",
                        scene, experiments,
                    )
                finally:
                    # Failed requests have observed wall latency, not an invented provider value.
                    row["gpt_latency_s"] = time.monotonic() - request_started
                row.update(decision=result.decision.to_dict(), model=result.model,
                           thread_id=result.thread_id, gpt_latency_s=result.latency_s)
                decision = validate_decision(result.decision.to_dict())
                if decision.decision == "unsupported":
                    row.update(status="unsupported", failure_reason=decision.reason)
                    return row
                if (decision.decision != "execute" or decision.goal != "hold_target"
                        or decision.skill != "grasp_and_hold"
                        or decision.target_object != case.target_object or decision.parameters is None):
                    raise PlannerError("Comparison requires one grasp_and_hold decision with goal hold_target")
                validate_context(decision, scene)
                parameters = decision.parameters
        except PlannerError as exc:
            row["failure_reason"] = str(exc)
            return row
        finally:
            row["planning_wall_s"] = time.monotonic() - planning_started

        row["parameters"] = parameters.to_dict()
        execution_started = time.monotonic()
        execution_sim_started = float(sim.data.time)
        sim.start_task("grasp", parameters)
        deadline = execution_sim_started + TASKS["grasp"].watchdog_s + SAMPLE_PERIOD_S
        while sim.skill.status == "running":
            if sim.data.time > deadline:
                raise RuntimeError("Controller did not terminate within its watchdog")
            sim.step(SAMPLE_PERIOD_S)
        row["execution_sim_s"] = float(sim.data.time) - execution_sim_started
        row["execution_wall_s"] = time.monotonic() - execution_started
        if sim.skill.status not in ("succeeded", "failed"):
            raise RuntimeError(f"Unexpected controller terminal status: {sim.skill.status}")

        validation_started = float(sim.data.time)
        if sim.skill.status == "succeeded":
            for _ in range(round(VALIDATION_HOLD_S / SAMPLE_PERIOD_S)):
                sim.step(SAMPLE_PERIOD_S)
                if sim.skill.status != "succeeded":
                    break
        row["validation_sim_s"] = float(sim.data.time) - validation_started
        state = sim.state()
        if state["skill"]["status"] not in ("succeeded", "failed"):
            raise RuntimeError(f"Unexpected validation status: {state['skill']['status']}")
        row.update(
            status=state["skill"]["status"], success=state["skill"]["status"] == "succeeded",
            failure_reason=state["skill"]["failure_reason"], warnings=state["warnings"],
            collisions=state["collisions"],
        )
        return row
    finally:
        sim.close()
        row["end_to_end_wall_s"] = time.monotonic() - started


def write_report(output: Path, report: dict) -> None:
    serialized = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    # Match experiments.py: readers never observe a partially written JSON checkpoint.
    with tempfile.TemporaryDirectory(dir=output.parent) as directory:
        pending = Path(directory) / output.name
        pending.write_text(serialized, encoding="utf-8")
        pending.replace(output)


def validate_methods(methods: tuple[str, ...]) -> None:
    if not methods or any(method not in METHODS for method in methods) or len(set(methods)) != len(methods):
        raise ValueError("Use distinct methods from fixed, rule, gpt")


async def run_benchmark(
    methods: tuple[str, ...] = METHODS, seeds: tuple[int, ...] = DEFAULT_SEEDS,
    output: Path = ROOT / "results" / "comparison.json", root: Path = ROOT,
) -> dict:
    validate_methods(methods)
    cases = benchmark_cases(seeds)
    calibration_path = root / "results" / "parameter_sweep.json"
    if output.resolve() == calibration_path.resolve():
        raise ValueError("Benchmark output must not overwrite the frozen calibration input")
    print(f"Planned runs: {len(cases) * len(methods)}; intended GPT calls: "
          f"{len(cases) if 'gpt' in methods else 0}", file=sys.stderr, flush=True)
    report = {
        "kind": "paired_grasp_comparison", "status": "running",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "methods": list(methods), "objects": list(TARGETS), "positions": list(POSITION_OFFSETS),
            "seeds": list(seeds), "planned_runs": len(cases) * len(methods),
            "settle_s": SETTLE_S, "validation_hold_s": VALIDATION_HOLD_S,
            "collision_force_threshold_n": COLLISION_FORCE_THRESHOLD_N,
            "note": (
                "Fresh simulation and identical settled qpos/qvel/ctrl fingerprint per paired case. "
                "Fixed: 80/100/0; frozen handcrafted rules: bowl 80/200/4 from phase-3 calibration, "
                "bottle 40/120/0 conservative neck policy (mm, mm/s, mm), not learned from evaluation. "
                "GPT chooses one grasp_and_hold action with goal hold_target using only current scene "
                "and existing parameter_sweep.json; one request per row. This is not a closed-loop "
                "evaluation: no retries, history feedback, stored-decision substitutes or fallbacks. "
                "Success requires an additional 5 s physical hold. Planning failures remain in the "
                "task-success denominator. Execution simulation time excludes validation; execution "
                "wall time excludes planning and validation; end-to-end wall includes setup, settle, "
                "planning, execution, validation and close. Failed GPT request latency is observed "
                "request wall time; no request means null. An aborted report includes only completed "
                "rows, never treats an internal error as method performance."
            ),
        },
        "runs": [], "summary": summarize_runs([], methods), "by_target": summarize_by_target([], methods),
    }
    fingerprints = {}
    active = None
    try:
        write_report(output, report)
        experiments = {}
        if "gpt" in methods:
            # Read once, before any evaluation; never feed benchmark results back to the planner.
            experiments = json.loads(calibration_path.read_text(encoding="utf-8"))
            if not isinstance(experiments, dict):
                raise ValueError("Calibration report must be a JSON object")
        for case in cases:
            for method in methods:
                active = {"case": case.to_dict(), "method": method}
                row = await run_case(case, method, fingerprints, experiments, root)
                report["runs"].append(row)
                report["summary"] = summarize_runs(report["runs"], methods)
                report["by_target"] = summarize_by_target(report["runs"], methods)
                write_report(output, report)
                print(f"{case.case_id} / {method}: {row['status']}"
                      + (f" — {row['failure_reason']}" if row["failure_reason"] else ""),
                      file=sys.stderr, flush=True)
                active = None
        report["status"] = "completed"
    except (Exception, KeyboardInterrupt, asyncio.CancelledError) as exc:
        report["status"] = "aborted"
        report["abort"] = {"active_run": active, "error_type": type(exc).__name__, "message": str(exc)}
    write_report(output, report)
    return report


def unsigned_seed(value: str) -> int:
    try:
        seed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("Seed must be an unsigned 32-bit integer") from None
    if not 0 <= seed < 2**32:
        raise argparse.ArgumentTypeError("Seed must be an unsigned 32-bit integer")
    return seed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=unsigned_seed, default=list(DEFAULT_SEEDS))
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "comparison.json")
    args = parser.parse_args(argv)
    try:
        validate_methods(tuple(args.methods))
        benchmark_cases(tuple(args.seeds))
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main() -> int:
    args = parse_args()
    report = asyncio.run(run_benchmark(tuple(args.methods), tuple(args.seeds), args.output))
    count = len(report["runs"])
    planned = report["protocol"]["planned_runs"]
    print(f"{report['status']}: {count}/{planned} rows saved to {args.output}", file=sys.stderr)
    if report["status"] == "aborted":
        print(f"Partial report; {report['abort']['error_type']}: {report['abort']['message']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
