"""Compare bounded grasp parameters in the same real MuJoCo reset state."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile

import mujoco

from grasp import GraspParameters
from simulation import Simulation


VALIDATION_HOLD_S = 5.0
SAMPLE_PERIOD_S = 0.02
CASES = (
    ("baseline", {}),
    ("smaller_opening", {"initial_opening_m": 0.05}),
    ("faster_descent", {"descent_speed_m_s": 0.16}),
    ("overlap", {"close_trigger_height_m": 0.004}),
    ("fast_overlap", {"descent_speed_m_s": 0.16, "close_trigger_height_m": 0.004}),
    ("higher_speed_overlap", {"descent_speed_m_s": 0.20, "close_trigger_height_m": 0.004}),
    ("combined", {
        "initial_opening_m": 0.05,
        "descent_speed_m_s": 0.16,
        "close_trigger_height_m": 0.004,
    }),
    ("early_closure", {
        "initial_opening_m": 0.05,
        "descent_speed_m_s": 0.16,
        "close_trigger_height_m": 0.02,
    }),
)


def run_case(name: str, parameters: GraspParameters) -> dict:
    sim = Simulation()
    try:
        sim.start_grasp(parameters)
        deadline = float(sim.data.time) + sim.skill.WATCHDOG + SAMPLE_PERIOD_S
        while sim.skill.status == "running":
            if sim.data.time > deadline:
                raise RuntimeError(f"{name}: controller did not terminate within its watchdog")
            sim.step(SAMPLE_PERIOD_S)

        validation_elapsed = 0.0
        if sim.skill.status == "succeeded":
            validation_start = float(sim.data.time)
            for _ in range(round(VALIDATION_HOLD_S / SAMPLE_PERIOD_S)):
                sim.step(SAMPLE_PERIOD_S)
                validation_elapsed = float(sim.data.time) - validation_start
                if sim.skill.status != "succeeded":
                    break

        state = sim.state()
        skill = state["skill"]
        fields = (
            "parameters", "status", "elapsed_s", "lift_height_m", "hold_elapsed_s",
            "descent_close_overlap_s", "close_started_s", "descent_completed_s",
            "phase_times_s", "failure_reason",
        )
        return {
            "name": name,
            **{field: skill[field] for field in fields},
            "validation_elapsed_s": validation_elapsed,
            "warnings": state["warnings"],
        }
    finally:
        sim.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Also save the measured JSON report to this path")
    args = parser.parse_args()
    runs = []
    for name, values in CASES:
        result = run_case(name, GraspParameters(**values))
        runs.append(result)
        print(
            f"{name}: {result['status']}, execution={result['elapsed_s']:.3f}s, "
            f"motion overlap={result['descent_close_overlap_s']:.3f}s"
            + (f", reason={result['failure_reason']}" if result["failure_reason"] else ""),
            file=sys.stderr, flush=True,
        )
    report = {
        "kind": "fixed_scene_parameter_sweep",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mujoco_version": mujoco.__version__,
        "validation_hold_s": VALIDATION_HOLD_S,
        "note": "同一固定初始场景，各配置仅执行一次；成功后继续物理验证 5 秒。不代表统计成功率或跨场景性能。",
        "runs": runs,
    }
    serialized = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Readers see either the previous complete report or the new complete report.
        with tempfile.TemporaryDirectory(dir=args.output.parent) as directory:
            pending = Path(directory) / args.output.name
            pending.write_text(serialized, encoding="utf-8")
            pending.replace(args.output)
    print(serialized, end="")


if __name__ == "__main__":
    main()
