"""Fresh-workcell grasp labels with independent, continuous two-second holds."""

import gzip
import json
import math
from pathlib import Path
import time

from benchmark import initial_state_fingerprint
from grasp import GraspParameters
from scenarios import SceneCase
from simulation import Simulation
from task_specs import TASKS


SETTLE_S = 1.0
REQUIRED_HOLD_S = 2.0
SAMPLE_PERIOD_S = 0.02
_PHYSICAL_PREDICATES = (
    "bilateral_grip", "clear_of_table", "target_lifted", "relative_pose_stable",
)


def _save_trace(path: Path, trace: dict) -> None:
    # Serialize before opening the destination so invalid evidence cannot leave
    # a seemingly valid, partially written JSON document.
    serialized = json.dumps(trace, ensure_ascii=False, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "xt", encoding="utf-8") as stream:
        stream.write(serialized)
        stream.write("\n")


def run_trial(case: SceneCase, speed_m_s: float, *, trace_path: Path | None = None) -> dict:
    """Execute one physical trial, or raise for internal/numerical failures.

    Invalid settled preconditions are unlabelled, not negative examples. A
    controller-reported physical failure is a completed negative example.
    The original one-second controller success is never itself a positive
    label: all physical hold predicates are observed at every physics step
    until two continuous seconds have elapsed since the first eligible
    observation. Trace samples are decimated to SAMPLE_PERIOD_S, with phase
    transitions and terminal observations retained as well.
    """
    started_wall = time.monotonic()
    sim = None
    trace = {
        "schema_version": 1,
        "settle_s": SETTLE_S,
        "sample_period_s": SAMPLE_PERIOD_S,
        "required_hold_s": REQUIRED_HOLD_S,
        "samples": [],
        "outcome": None,
    }
    if trace_path is not None:
        trace_path = Path(trace_path)
        if trace_path.exists():
            raise FileExistsError(trace_path)
    execution_started = None
    first_terminal_s = None
    hold_started = None
    verified_hold_s = 0.0
    last_sample_time = None
    last_sample_phase = None

    def sample(stage: str, predicates: dict, *, force: bool = False) -> None:
        nonlocal last_sample_time, last_sample_phase
        if trace_path is None:
            return
        now = float(sim.data.time)
        phase = (stage, sim.skill.phase, sim.skill.status)
        if (not force and phase == last_sample_phase and last_sample_time is not None
                and now - last_sample_time < SAMPLE_PERIOD_S - 1e-12):
            return
        observation = {
            "time_s": now,
            "execution_sim_s": None if execution_started is None else now - execution_started,
            "stage": stage,
            "phase": sim.skill.phase,
            "controller_status": sim.skill.status,
            "pose": sim.get_object_pose(f"{case.target_object}_object"),
            "tcp_position_m": sim.data.site_xpos[sim.skill.tcp].tolist(),
            "postconditions": predicates,
            "controller_hold_s": float(sim.skill.hold_elapsed),
            "verified_hold_s": verified_hold_s,
            "reference_z_m": sim.skill.reference_z,
        }
        # Keep only finite, JSON-safe snapshots; errors retain prior good ones.
        json.dumps(observation, allow_nan=False)
        trace["samples"].append(observation)
        last_sample_time = now
        last_sample_phase = phase

    def check_numerical() -> None:
        error = sim.numerical_error()
        if error is not None:
            raise RuntimeError(error)

    try:
        if not isinstance(case, SceneCase):
            raise ValueError("case must be a SceneCase")
        case.__post_init__()
        parameters = GraspParameters(
            initial_opening_m=0.08 if case.target_object == "bowl" else 0.04,
            descent_speed_m_s=speed_m_s,
            close_trigger_height_m=0.0,
        )
        trace.update(case=case.to_dict(), parameters=parameters.to_dict())
        sim = Simulation(case)
        dt = float(sim.model.opt.timestep)
        if not math.isfinite(dt) or dt <= 0:
            raise RuntimeError("Invalid physics timestep")
        settle_steps = round(SETTLE_S / dt)
        if settle_steps < 1 or not math.isclose(settle_steps * dt, SETTLE_S, abs_tol=1e-12):
            raise RuntimeError("Settle duration must be a multiple of the physics timestep")
        check_numerical()
        sample("settle", sim.skill.grasp_postconditions(), force=True)
        for _ in range(settle_steps):
            previous = float(sim.data.time)
            sim.step(dt)
            check_numerical()
            if not math.isclose(float(sim.data.time) - previous, dt, rel_tol=1e-7, abs_tol=1e-12):
                raise RuntimeError("Physics clock did not advance by one timestep")
            sample("settle", sim.skill.grasp_postconditions())

        initial_pose = sim.get_object_pose(f"{case.target_object}_object")
        fingerprint = initial_state_fingerprint(sim)
        # Reuse the controller's real support/velocity/empty-hand checks,
        # rather than inventing a geometry blacklist for difficult cases.
        initial_error = sim.skill.precondition_error("grasp_and_hold")
        check_numerical()
        trace.update(initial_pose=initial_pose, initial_state_fingerprint=fingerprint,
                     initial_precondition_error=initial_error)
        if initial_error is None:
            execution_started = float(sim.data.time)
            sim.start_task("grasp", parameters)
            deadline = TASKS["grasp"].watchdog_s + REQUIRED_HOLD_S + SAMPLE_PERIOD_S
            # Observe each physical step, not just trace samples. This preserves
            # the first terminal time even if success is lost within 20 ms.
            while True:
                check_numerical()
                now = float(sim.data.time)
                elapsed = now - execution_started
                status = sim.skill.status
                if status not in ("running", "succeeded", "failed"):
                    raise RuntimeError(f"Unexpected controller status: {status}")
                if status in ("succeeded", "failed") and first_terminal_s is None:
                    finished = sim.skill.finished_at
                    if finished is None or not math.isfinite(finished) or not execution_started <= finished <= now:
                        raise RuntimeError("Controller terminal time is missing or invalid")
                    first_terminal_s = float(finished) - execution_started
                predicates = sim.skill.grasp_postconditions()
                eligible = (
                    status in ("running", "succeeded")
                    and sim.skill.phase in ("hold", "done")
                    and all(predicates[key] for key in _PHYSICAL_PREDICATES)
                )
                if eligible:
                    if hold_started is None:
                        hold_started = now
                    verified_hold_s = now - hold_started
                else:
                    hold_started = None
                    verified_hold_s = 0.0
                complete = status == "failed" or (
                    status == "succeeded" and verified_hold_s >= REQUIRED_HOLD_S
                )
                sample("execution", predicates, force=complete)
                if complete:
                    break
                if elapsed > deadline:
                    raise RuntimeError("Controller/hold observer exceeded the trial watchdog")
                sim.step(dt)
                check_numerical()
                if not math.isclose(float(sim.data.time) - now, dt, rel_tol=1e-7, abs_tol=1e-12):
                    raise RuntimeError("Physics clock did not advance by one timestep")

        check_numerical()
        predicates = sim.skill.grasp_postconditions()
        result = {
            "case": case.to_dict(),
            "parameters": parameters.to_dict(),
            "trial_status": "completed" if initial_error is None else "invalid_initial_state",
            "success": sim.skill.status == "succeeded" if initial_error is None else None,
            "controller_status": sim.skill.status,
            "failure_reason": sim.skill.failure_reason if initial_error is None else initial_error,
            "initial_state_fingerprint": fingerprint,
            "initial_pose": initial_pose,
            "final_pose": sim.get_object_pose(f"{case.target_object}_object"),
            "controller_execution_sim_s": first_terminal_s,
            "execution_sim_s": float(sim.data.time) - execution_started if execution_started is not None else 0.0,
            "verified_hold_s": verified_hold_s,
            "required_hold_s": REQUIRED_HOLD_S,
            "wall_s": time.monotonic() - started_wall,
            "warnings": sim.warning_names(),
            "collisions": sim.collisions.state(),
            "postconditions": {**predicates, "verified_hold": verified_hold_s >= REQUIRED_HOLD_S},
            "controller_verified_hold": bool(predicates["verified_hold"]),
        }
        if trace_path is not None:
            result["trace_path"] = str(trace_path)
            sample("final", predicates, force=True)
        json.dumps(result, allow_nan=False)
        trace["outcome"] = result
        if trace_path is not None:
            _save_trace(trace_path, trace)
        return result
    except BaseException as exc:
        # A numerical fault may also set controller.status='failed'. It is
        # still an exception, never a physical negative training label.
        trace["outcome"] = None
        trace["error"] = {"type": type(exc).__name__, "message": str(exc)}
        trace["wall_s"] = time.monotonic() - started_wall
        if trace_path is not None:
            try:
                _save_trace(trace_path, trace)
            except Exception as trace_error:
                exc.add_note(f"Could not preserve trace at {trace_path}: {trace_error}")
        raise
    finally:
        if sim is not None:
            sim.close()
