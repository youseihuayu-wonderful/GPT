"""Run and save a real subscription-driven skill loop, authorizing continuous execution."""

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

from benchmark import write_report
from experiments import SAMPLE_PERIOD_S, VALIDATION_HOLD_S
from grasp import GraspParameters
from planner import PlannerError, validate_context, validate_decision
from planning import PlanningSession
from simulation import Simulation
from task_specs import SKILLS


ROOT = Path(__file__).resolve().parent
DEFAULT_TASK = "把当前黑碗放入已打开的下层抽屉，松开并撤出夹爪，实际推关抽屉，然后机械臂归位。"
TERMINAL = {"unconfigured", "idle", "cancelled", "unsupported", "failed", "succeeded"}


async def run_planning(
    task: str = DEFAULT_TASK, output: Path = ROOT / "results" / "planning_loop.json",
    *, start_held: bool = False, root: Path = ROOT,
) -> dict:
    """Authorize the first validated decision and retain every observed loop boundary."""
    calibration_path = root / "results" / "parameter_sweep.json"
    if output.resolve() == calibration_path.resolve():
        raise ValueError("Planning output must not overwrite the frozen calibration input")
    started = time.monotonic()
    report = {
        "kind": "subscription_live_skill_loop", "status": "running", "success": False,
        "generated_at": datetime.now(timezone.utc).isoformat(), "task": task,
        "protocol": {
            "authorization": "continuous", "start_held": start_held,
            "sample_period_s": SAMPLE_PERIOD_S, "validation_hold_s": VALIDATION_HOLD_S,
            "max_skills": 8, "max_requests": 9, "provider_retries": 0,
            "physics_during_provider_wait": True,
            "note": "Real Codex subscription decisions and MuJoCo execution; no provider substitutes. "
                    "Starting held uses an actual independent grasp before the first request. "
                    "Model wall latency and physics time are separate; final success requires "
                    "five additional simulated seconds and a local goal recheck.",
        },
        "initial_observation": None, "planning_initial_observation": None,
        "initial_grasp": None, "planning": None, "history": [], "final_state": None,
        "validation": {"status": "not_started", "sim_s": 0.0, "wall_s": 0.0,
                       "before": None, "after": None, "goal_verified": False},
        "error": None, "elapsed_wall_s": 0.0, "planning_sim_s": None,
    }
    sim = None
    session = None
    loop_started_sim = None
    validation_started_sim = None
    validation_started_wall = None

    def checkpoint(capture_physics=True):
        report["elapsed_wall_s"] = time.monotonic() - started
        if session is not None:
            state = session.state()
            report["history"] = state.pop("history")
            report["planning"] = state
        if sim is not None and capture_physics:
            report["final_state"] = sim.state()
        # Preserve nonfinite failure evidence explicitly while keeping the saved JSON strict.
        evidence = json.loads(json.dumps(report, ensure_ascii=False),
                              parse_constant=lambda token: {"nonfinite": token})
        write_report(output, evidence)

    def record_failure(exc, stage):
        detail = {"type": type(exc).__name__, "message": str(exc) or type(exc).__name__, "stage": stage}
        report.update(status="failed", success=False)
        if report["error"] is None:
            report["error"] = detail
        else:
            report.setdefault("secondary_errors", []).append(detail)

    try:
        checkpoint()
        experiments = json.loads(calibration_path.read_text(encoding="utf-8"))
        if not isinstance(experiments, dict):
            raise ValueError("Calibration report must be a JSON object")
        sim = Simulation()
        sim.step(0.5)
        report["initial_observation"] = sim.state()
        if start_held:
            grasp = {
                "skill": "grasp_and_hold", "parameters": GraspParameters().to_dict(),
                "status": "running", "observation": sim.state(), "result_observation": None,
                "execution_sim_s": 0.0, "execution_wall_s": 0.0,
            }
            report["initial_grasp"] = grasp
            grasp_started_sim = float(sim.data.time)
            grasp_started_wall = time.monotonic()
            try:
                sim.start_skill("grasp_and_hold", GraspParameters())
                checkpoint()
                while sim.skill.status == "running":
                    if float(sim.data.time) - grasp_started_sim > SKILLS["grasp_and_hold"].watchdog_s + SAMPLE_PERIOD_S:
                        raise RuntimeError("Initial grasp exceeded its controller watchdog")
                    sim.step(SAMPLE_PERIOD_S)
                    await asyncio.sleep(SAMPLE_PERIOD_S)
                if sim.skill.status != "succeeded":
                    raise RuntimeError(sim.state()["skill"]["failure_reason"] or "Initial grasp failed")
            finally:
                primary = sys.exc_info()[1]
                grasp["status"] = sim.skill.status
                grasp["execution_sim_s"] = float(sim.data.time) - grasp_started_sim
                grasp["execution_wall_s"] = time.monotonic() - grasp_started_wall
                try:
                    grasp["result_observation"] = sim.state()
                except Exception as exc:
                    grasp["result_observation_error"] = {"type": type(exc).__name__, "message": str(exc)}
                    if primary is None:
                        raise

        report["planning_initial_observation"] = sim.state()
        loop_started_sim = float(sim.data.time)
        session = PlanningSession(root, sim)
        if not session.start(task, experiments):
            raise PlannerError(session.state().get("error") or "Planning session could not start")
        checkpoint()
        authorized = False
        boundary = None
        while True:
            session.advance()
            state = session.state()
            marker = (state["status"], state["decision_id"], len(state["history"]))
            if marker != boundary:
                checkpoint()
                boundary = marker
            if state["status"] in TERMINAL:
                if state["status"] != "succeeded":
                    raise PlannerError(state.get("error") or f"Planning ended with status {state['status']}")
                break
            if state["status"] == "ready" and not authorized:
                if task == DEFAULT_TASK and state["decision"]["goal"] != "store_and_home":
                    raise PlannerError("Full storage task requires the store_and_home goal")
                session.execute(state["decision_id"], continuous=True)
                authorized = True
            sim.step(SAMPLE_PERIOD_S)
            session.advance()
            await asyncio.sleep(SAMPLE_PERIOD_S)

        report["planning_sim_s"] = float(sim.data.time) - loop_started_sim
        complete = validate_decision(session.state()["decision"])
        if complete.decision != "complete":
            raise PlannerError("Session success requires a locally checked model completion decision")
        validation = report["validation"]
        validation.update(status="running", before=sim.state())
        validation_started_sim = float(sim.data.time)
        validation_started_wall = time.monotonic()
        checkpoint()
        for _ in range(round(VALIDATION_HOLD_S / SAMPLE_PERIOD_S)):
            sim.step(SAMPLE_PERIOD_S)
            session.advance()
            if session.state()["status"] != "succeeded":
                raise PlannerError(session.state().get("error") or "Goal lost during independent final hold")
            await asyncio.sleep(SAMPLE_PERIOD_S)
        validate_context(complete, sim.state(), session.state()["goal"])
        validation.update(status="succeeded", after=sim.state(), goal_verified=True)
        report.update(status="succeeded", success=True)
    except (Exception, KeyboardInterrupt, asyncio.CancelledError) as exc:
        record_failure(exc, "execution")
        if report["validation"]["status"] == "running":
            report["validation"]["status"] = "failed"
    finally:
        if sim is not None and loop_started_sim is not None and report["planning_sim_s"] is None:
            report["planning_sim_s"] = float(sim.data.time) - loop_started_sim
        if validation_started_sim is not None:
            report["validation"]["sim_s"] = float(sim.data.time) - validation_started_sim
            report["validation"]["wall_s"] = time.monotonic() - validation_started_wall
        # Capture physics before disposal; cleanup cannot turn a failure into
        # success or erase the original error from the final artifact.
        if sim is not None:
            try:
                numerical_error = sim.numerical_error()
                if numerical_error is not None:
                    report["final_state"] = None
                    report["final_state_error"] = numerical_error
                else:
                    report["final_state"] = sim.state()
            except Exception as exc:
                report["final_state"] = None
                record_failure(exc, "final_observation")
        if session is not None:
            try:
                if sim.skill.status == "failed":
                    session.advance(allow_execution=False)
            except Exception as exc:
                record_failure(exc, "failure_feedback")
            try:
                await session.close()
            except (Exception, KeyboardInterrupt, asyncio.CancelledError) as exc:
                record_failure(exc, "session_cleanup")
        if sim is not None:
            try:
                sim.close()
            except Exception as exc:
                record_failure(exc, "simulation_cleanup")
        try:
            checkpoint(capture_physics=False)
        except (OSError, ValueError, TypeError):
            if report["error"] is not None:
                primary = report["error"]
                print(f"Run failed: {primary['type']}: {primary['message']}", file=sys.stderr)
            raise
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=DEFAULT_TASK, help="Natural-language goal; default is full bowl storage and home")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "planning_loop.json")
    parser.add_argument("--start-held", action="store_true", help="Physically grasp first, then ask GPT from the held state without reset")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    try:
        report = asyncio.run(run_planning(args.task, args.output, start_held=args.start_held))
    except (OSError, ValueError, TypeError) as exc:
        print(f"Cannot save planning evidence: {exc}", file=sys.stderr)
        return 1
    print(f"{report['status']}: {len(report['history'])} decisions saved to {args.output}", file=sys.stderr)
    if report["error"]:
        print(f"{report['error']['type']}: {report['error']['message']}", file=sys.stderr)
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
