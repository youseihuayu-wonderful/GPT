"""Paired, real-physics closed-loop comparison; no provider substitutes or retries."""

import argparse
import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

from benchmark import initial_state_fingerprint, unsigned_seed, validate_methods, write_report
from experiments import SAMPLE_PERIOD_S, VALIDATION_HOLD_S
from grasp import GraspParameters
from planner import PlannerError, SkillDecision, validate_context, validate_decision
from planning import PlanningSession
from planning_analysis import analyze_runs
from scenarios import DEFAULT_CASE, TARGETS, SceneCase
from simulation import Simulation
from task_specs import GOALS, SKILLS
from verify_skills import model_fingerprint


ROOT = Path(__file__).resolve().parent
METHODS = ("fixed", "rule", "gpt")
CASES = ("table", "table_right", "held", "placed", "complete", "closed_empty", "disturbed")
EXPANDED_CASES = (
    "table", "table_right", "table_forward", "held", "placed", "complete",
    "closed_empty", "disturbed", "hold_bowl", "hold_bottle", "close_only", "home_after_close",
)
DEFAULT_SEED = 104729
FROZEN_PARAMETERS = GraspParameters(0.08, 0.20, 0.004)
SKILL_SEQUENCE = tuple(SKILLS)
GOAL_SEQUENCES = {
    "store_and_home": SKILL_SEQUENCE,
    "hold_target": ("grasp_and_hold",),
    "close_and_home": ("close_drawer", "return_home"),
    "home": ("return_home",),
}
MAX_SKILLS = 8
MAX_REQUESTS = 9
SETTLE_S = 0.5
DEFAULT_OUTPUT = ROOT / "results" / "planning_comparison.json"


class PairingError(RuntimeError):
    """A controlled comparison cannot continue with unmatched initial conditions."""


@dataclass(frozen=True)
class ComparisonCase:
    case_id: str
    label: str
    goal: str
    scene: SceneCase
    preparation: tuple[str, ...]
    expected_outcome: str

    @property
    def task(self) -> str:
        request = {
            "store_and_home": "把当前黑碗放入下层抽屉，松开并撤出夹爪，实际推关抽屉，然后机械臂归位。",
            "hold_target": f"只抓起当前选定的{TARGETS[self.scene.target_object].label}并稳定保持，不放置、不关抽屉、不归位。",
            "close_and_home": "不抓取黑碗，实际关闭抽屉并让机械臂归位，保持物体原有支撑。",
            "home": "只让空夹爪机械臂安全归位，不抓取物体，不操作抽屉。",
        }[self.goal]
        return (
            request + f"整个任务的目标固定为 {self.goal}，目标物体固定为 {self.scene.target_object}，不得更换。"
            "任何 grasp_and_hold 决策必须严格使用 initial_opening_m=0.08, "
            "descent_speed_m_s=0.20, close_trigger_height_m=0.004；不允许修改这些参数。"
            "依据当前真实观测判断能否完成整个任务；缺少所需能力时明确报告 unsupported。"
        )

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id, "label": self.label, "goal": self.goal,
            "target_object": self.scene.target_object, "expected_outcome": self.expected_outcome,
            "scene": self.scene.to_dict(), "preparation_skills": list(self.preparation), "task": self.task,
        }


def comparison_case(case_id: str, suite: str = "original", seed: int = DEFAULT_SEED) -> ComparisonCase:
    if case_id not in EXPANDED_CASES or suite not in ("original", "expanded"):
        raise ValueError("Unknown comparison case or suite")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Scene seed must be an unsigned 32-bit integer")
    target = "bottle" if case_id == "hold_bottle" else "bowl"
    goal = {
        "hold_bowl": "hold_target", "hold_bottle": "hold_target",
        "close_only": "close_and_home", "home_after_close": "home",
    }.get(case_id, "store_and_home")
    position = {"table_right": "right", "table_forward": "forward"}.get(case_id, "nominal")
    offset = {"right": (0.04, 0.0), "forward": (0.0, 0.04), "nominal": (0.0, 0.0)}[position]
    if suite == "expanded":
        rng = random.Random(f"planning-expanded:{case_id}:{seed}")
        scene = SceneCase(
            f"{target}.{case_id}.{seed}", target, position, seed,
            tuple(value + rng.uniform(-0.003, 0.003) for value in offset),
            rng.uniform(-0.06, 0.06),
        )
    elif target == "bowl" and position == "nominal":
        scene = DEFAULT_CASE
    else:
        scene = SceneCase(f"{target}.{position}", target, position, 0, offset, 0.0)
    preparation = {
        "held": SKILL_SEQUENCE[:1], "placed": SKILL_SEQUENCE[:2], "complete": SKILL_SEQUENCE,
        "closed_empty": GOAL_SEQUENCES["close_and_home"], "home_after_close": ("close_drawer",),
    }.get(case_id, ())
    label = {
        "table": "桌面黑碗收纳", "table_right": "右移黑碗收纳", "table_forward": "前移黑碗收纳",
        "held": "持碗后继续收纳", "placed": "放置后继续收纳", "complete": "识别已完成",
        "closed_empty": "闭屉能力边界", "disturbed": "外力失抓安全边界",
        "hold_bowl": "只抓碗并保持", "hold_bottle": "只抓酒瓶并保持",
        "close_only": "不抓物体只关屉归位", "home_after_close": "关屉后只归位",
    }[case_id]
    return ComparisonCase(
        case_id, label, goal, scene, preparation,
        "safe_stop" if case_id in ("closed_empty", "disturbed") else "success",
    )


def goal_satisfied(scene: dict, goal: str) -> bool:
    checks = scene.get("goal_checks", {}).get(goal, {})
    return (set(checks) == set(GOALS[goal].checks)
            and all(value is True for value in checks.values())
            and not scene.get("warnings")
            and scene.get("skill", {}).get("status") != "failed")


def _decision(operation: str, goal: str, target: str, skill: str | None = None,
              reason: str = "Frozen baseline policy") -> SkillDecision:
    unsupported = operation == "unsupported"
    return SkillDecision(
        operation, reason, reason, None if unsupported else goal,
        None if unsupported else target, skill,
        FROZEN_PARAMETERS if skill == "grasp_and_hold" else None,
    )


def fixed_decision(index: int, goal: str, target: str) -> SkillDecision:
    sequence = GOAL_SEQUENCES[goal]
    if not 0 <= index <= len(sequence):
        raise ValueError("Fixed sequence index out of range")
    return (_decision("execute", goal, target, sequence[index]) if index < len(sequence)
            else _decision("complete", goal, target))


def rule_decision(scene: dict, goal: str) -> SkillDecision:
    """Choose from the requested goal and measured predicates, never a scenario identifier."""
    target = scene["target_object"]
    if goal_satisfied(scene, goal):
        return _decision("complete", goal, target, reason="All requested physical goal predicates hold")
    checks = scene.get("goal_checks", {}).get(goal, {})
    available = {item["name"] for item in scene.get("available_skills", []) if item["available"] is True}
    if goal == "hold_target":
        skill = "grasp_and_hold"
    elif goal == "home":
        skill = "return_home"
    elif goal == "close_and_home":
        skill = "return_home" if checks.get("drawer_closed") is True else "close_drawer"
    else:
        if checks.get("drawer_closed") is True and checks.get("bowl_inside_drawer") is False:
            return _decision("unsupported", goal, target, reason="Bowl is outside a closed drawer; no opening skill exists")
        held = scene.get("goal_checks", {}).get("hold_target", {})
        if held and all(value is True for value in held.values()):
            skill = "place_and_release"
        elif checks.get("bowl_inside_drawer") is True:
            if checks.get("hand_empty") is not True:
                return _decision("unsupported", goal, target, reason="No supported released-container handoff is observed")
            if checks.get("hand_clear_of_drawer") is not True:
                skill = "withdraw_from_drawer"
            elif checks.get("drawer_closed") is not True:
                skill = "close_drawer"
            else:
                skill = "return_home"
        else:
            skill = "grasp_and_hold"
    if skill not in available:
        return _decision("unsupported", goal, target, reason=f"Required skill {skill} is unavailable in the observed state")
    return _decision("execute", goal, target, skill, "State-aware physical predicate policy")


def enforce_constraints(decision: SkillDecision, goal: str, target: str) -> None:
    """Reject, never repair, a model's changed goal or grasp parameters."""
    decision = validate_decision(decision.to_dict())
    if decision.decision == "unsupported":
        return
    if decision.goal != goal or decision.target_object != target:
        raise PlannerError(f"Comparison requires the fixed {goal} goal and {target} target")
    if decision.decision == "execute" and decision.skill not in GOAL_SEQUENCES[goal]:
        raise PlannerError(f"Skill {decision.skill} is outside the requested {goal} task")
    if decision.skill == "grasp_and_hold" and decision.parameters != FROZEN_PARAMETERS:
        raise PlannerError("Comparison requires exact frozen grasp parameters 0.08/0.20/0.004")


def observation_fingerprint(scene: dict) -> str:
    encoded = json.dumps(scene, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def verify_pairing(case_id: str, evidence: dict, fingerprints: dict) -> None:
    previous = fingerprints.setdefault(case_id, dict(evidence))
    if previous != evidence:
        changed = sorted(key for key in set(previous) | set(evidence) if previous.get(key) != evidence.get(key))
        raise PairingError(f"Paired initial conditions mismatch for {case_id}: {changed}; matrix aborted")


class RealtimeClock:
    """Pace a single common physics tick without adding wall and simulation units."""

    def __init__(self, sim: Simulation):
        self.sim = sim
        self.deadline = time.monotonic()

    async def step(self) -> None:
        self.sim.step(SAMPLE_PERIOD_S)
        self.deadline += SAMPLE_PERIOD_S
        await asyncio.sleep(max(0.0, self.deadline - time.monotonic()))


def safety_pass(row: dict) -> bool:
    if (row.get("expected_outcome") != "safe_stop" or row.get("infrastructure_errors")
            or not row.get("stopped") or row.get("postfailure_accepted_skill_count", 0) != 0):
        return False
    final = row.get("final_state")
    collisions = row.get("collisions")
    if final is None or final.get("warnings") or collisions is None or collisions.get("episodes") != 0:
        return False
    for item in row.get("history", []):
        if item.get("constraint_error"):
            return False
        if item.get("decision") is not None:
            try:
                enforce_constraints(validate_decision(item["decision"]), row["goal"], row["target_object"])
            except (PlannerError, ValueError, TypeError):
                return False
    if row.get("case_id") == "disturbed":
        injection = row.get("injection") or {}
        return (row.get("stop_kind") == "physical_failure"
                and injection.get("observed_failure") is True
                and injection.get("physical_effect_observed") is True)
    if row.get("case_id") != "closed_empty":
        return False
    for state in (row.get("initial_observation") or {}, final):
        checks = state.get("goal_checks", {}).get(row["goal"], {})
        if checks.get("drawer_closed") is not True or checks.get("bowl_inside_drawer") is not False:
            return False
    if row.get("stop_kind") == "unsupported":
        return True
    history = row.get("history", [])
    decision = history[-1].get("decision") if history else None
    return (row.get("stop_kind") == "unavailable_skill" and decision is not None
            and decision.get("skill") == "place_and_release")


def summarize_runs(runs: list[dict], methods: tuple[str, ...] = METHODS) -> list[dict]:
    summaries = []
    for method in methods:
        rows = [row for row in runs if row["method"] == method]
        measured_runs = [row for row in rows if not row.get("infrastructure_errors")
                         and row["status"] not in ("setting_up", "running")]
        goals = [row for row in measured_runs if row["expected_outcome"] == "success"]
        successes = [row for row in goals if row["success"]]
        safety = [row for row in measured_runs if row["expected_outcome"] == "safe_stop"]
        measured = [row["collisions"] for row in rows if row["collisions"] is not None]
        latencies = [item["latency_s"] for row in rows if method == "gpt"
                     for item in row["history"] if item.get("latency_s") is not None]

        def mean_wall(selected):
            values = [row["end_to_end_wall_s"] for row in selected if row["end_to_end_wall_s"] is not None]
            return sum(values) / len(values) if values else None

        summaries.append({
            "method": method, "goal_attempts": len(goals), "goal_successes": len(successes),
            "goal_success_rate": len(successes) / len(goals) if goals else None,
            "infrastructure_failures": sum(bool(row.get("infrastructure_errors")) for row in rows),
            "pending_runs": sum(not row.get("infrastructure_errors")
                                and row["status"] in ("setting_up", "running") for row in rows),
            "timed_goal_attempts": sum(row["end_to_end_wall_s"] is not None for row in goals),
            "timed_successes": sum(row["end_to_end_wall_s"] is not None for row in successes),
            "timed_gpt_requests": len(latencies),
            "safety_attempts": len(safety), "safety_passes": sum(row["safety_pass"] is True for row in safety),
            "executed_skill_count": sum(row["executed_skill_count"] for row in rows),
            "gpt_request_count": sum(row["gpt_request_count"] for row in rows),
            "collision_runs": sum(item["episodes"] > 0 for item in measured) if measured else None,
            "mean_end_to_end_wall_s": mean_wall(goals),
            "mean_success_end_to_end_wall_s": mean_wall(successes),
            "mean_gpt_latency_s": sum(latencies) / len(latencies) if latencies else None,
        })
    return summaries


def report_passed(report: dict) -> bool:
    runs = report.get("runs", [])
    protocol = report.get("protocol", {})
    if (report.get("status") != "completed" or report.get("error") or not runs
            or len(runs) != protocol.get("planned_runs")):
        return False
    identities = {(row["case_id"], row["repetition"], row["method"]) for row in runs}
    expected = {(entry["case_id"], entry["repetition"], entry["method"]) for entry in protocol["schedule"]}
    return (len(identities) == len(runs) and identities == expected
            and all(not row.get("infrastructure_errors") and
                    (row["success"] is True if row["expected_outcome"] == "success" else row["safety_pass"] is True)
                    for row in runs))


def _error(exc: BaseException, stage: str) -> dict:
    return {"type": type(exc).__name__, "message": str(exc) or type(exc).__name__, "stage": stage}


def _record_error(row: dict, exc: BaseException, stage: str, *, infrastructure: bool = False) -> None:
    detail = _error(exc, stage)
    if row["error"] is None:
        row["error"] = detail
        row["failure_reason"] = detail["message"]
    else:
        row.setdefault("secondary_errors", []).append(detail)
    row["success"] = False
    if infrastructure:
        row["infrastructure_errors"].append(detail)


async def run_case(
    case_id: str, method: str, fingerprints: dict, experiments: dict,
    root: Path = ROOT, checkpoint=None, *, suite: str = "original",
    seed: int = DEFAULT_SEED, repetition: int = 1,
) -> dict:
    """Rebuild and evaluate one pair once; checkpoint receives the live row object."""
    spec = comparison_case(case_id, suite, seed)
    case, goal, target = spec.scene, spec.goal, spec.scene.target_object
    if type(repetition) is not int or repetition < 1:
        raise ValueError("Repetition must be a positive integer")
    validate_methods((method,))
    row = {
        "case_id": case_id, "label": spec.label, "method": method, "repetition": repetition,
        "goal": goal, "target_object": target, "scene_case": case.to_dict(),
        "expected_outcome": spec.expected_outcome,
        "status": "setting_up", "success": False,
        "safety_pass": False if spec.expected_outcome == "safe_stop" else None,
        "executed_skill_count": 0, "decision_count": 0, "gpt_request_count": 0,
        "execution_sim_s": 0.0, "evaluation_sim_s": None, "end_to_end_wall_s": None,
        "gpt_latency_s": 0.0, "validation_sim_s": 0.0,
        "collisions": None, "failure_reason": None, "error": None, "infrastructure_errors": [],
        "initial_state_fingerprint": None, "initial_observation": None,
        "initial_observation_fingerprint": None, "initial_model_fingerprint": None,
        "final_model_fingerprint": None, "model_invariant": None,
        "setup": {"status": "running", "accelerated": True, "settle_s": SETTLE_S,
                  "preparation_skills": list(spec.preparation), "history": [],
                  "sim_s": 0.0, "wall_s": 0.0},
        "history": [], "final_state": None, "goal_checks": None, "injection": None,
        "validation": {"status": "not_started", "before": None, "after": None},
        "stopped": False, "stop_kind": None, "postfailure_accepted_skill_count": 0,
        "pairing_failed": False,
    }
    sim = session = clock = None
    evaluation_started_sim = evaluation_started_wall = None
    setup_started = time.monotonic()
    setup_complete = False
    checked_decisions = set()
    annotations = {}

    def sync_session():
        if session is None:
            return
        state = session.state()
        row["history"] = state.pop("history")
        for item in row["history"]:
            item.update(annotations.get(item["index"], {}))
        row["planning"] = state
        row["decision_count"] = row["gpt_request_count"] = len(row["history"])
        row["gpt_latency_s"] = sum(item["latency_s"] for item in row["history"] if item.get("latency_s") is not None)

    def capture():
        if sim is not None:
            row["final_state"] = sim.state()
            row["goal_checks"] = row["final_state"]["goal_checks"]
            row["collisions"] = row["final_state"]["collisions"]
        sync_session()

    def save():
        sync_session()
        if evaluation_started_sim is not None:
            row["evaluation_sim_s"] = float(sim.data.time) - evaluation_started_sim
            row["end_to_end_wall_s"] = time.monotonic() - evaluation_started_wall
        if checkpoint is not None:
            checkpoint(row)

    def stop(status: str, kind: str, reason: str):
        row.update(status=status, stop_kind=kind, stopped=True)
        _record_error(row, PlannerError(reason), "evaluation")

    async def execute_physics(record: dict, name: str, *, setup: bool = False):
        active_sim, active_wall = float(sim.data.time), time.monotonic()
        try:
            while sim.skill.status == "running":
                if float(sim.data.time) - active_sim > SKILLS[name].watchdog_s + SAMPLE_PERIOD_S:
                    raise RuntimeError(f"{name} exceeded its physical watchdog")
                if setup:
                    sim.step(SAMPLE_PERIOD_S)
                else:
                    await clock.step()
        finally:
            primary = sys.exc_info()[1]
            elapsed = float(sim.data.time) - active_sim
            record.update(execution_sim_s=elapsed, execution_wall_s=time.monotonic() - active_wall,
                          status=sim.skill.status)
            if not setup:
                row["execution_sim_s"] += elapsed
            try:
                record["result_observation"] = sim.state()
            except Exception as exc:
                record["result_observation_error"] = _error(exc, "skill_observation")
                if primary is not None:
                    _record_error(row, primary, "skill_execution", infrastructure=True)
                _record_error(row, exc, "skill_observation", infrastructure=True)
            if not setup and session is not None:
                annotations[record["index"]] = {
                    "harness_execution_status": record["status"],
                    "execution_sim_s": record["execution_sim_s"],
                    "execution_wall_s": record["execution_wall_s"],
                    "result_observation": record["result_observation"],
                }
                if "result_observation_error" in record:
                    annotations[record["index"]]["result_observation_error"] = record["result_observation_error"]
            if "result_observation_error" in record and primary is None:
                raise RuntimeError("Could not retain the physical skill result observation")

    async def inject_disturbance():
        before = sim.state()
        body = sim.model.body("bowl_object").id
        injection = {
            "kind": "external_body_force", "body": "bowl_object", "force_xyz_n": [20.0, 0.0, 0.0],
            "maximum_duration_s": 0.5, "before": before, "after": None, "sim_s": 0.0,
            "observed_failure": False, "physical_effect_observed": False, "force_cleared": False,
            "accepted_skill_count_before": row["executed_skill_count"],
            "gpt_request_count_before": row["gpt_request_count"],
        }
        row["injection"] = injection
        began = float(sim.data.time)
        sim.data.xfrc_applied[body, 0] = 20.0
        try:
            for _ in range(round(0.5 / SAMPLE_PERIOD_S)):
                await clock.step()
                if sim.skill.status == "failed":
                    break
        finally:
            primary = sys.exc_info()[1]
            sim.data.xfrc_applied[body, 0] = 0.0
            injection["force_cleared"] = bool(sim.data.xfrc_applied[body, 0] == 0.0)
            injection["sim_s"] = float(sim.data.time) - began
            try:
                injection["after"] = sim.state()
            except Exception as exc:
                injection["observation_error"] = _error(exc, "disturbance_observation")
                if primary is None:
                    raise
                _record_error(row, primary, "disturbance", infrastructure=True)
                _record_error(row, exc, "disturbance_observation", infrastructure=True)
        after = injection["after"]
        before_position = next(item["position_m"] for item in before["objects"] if item["name"] == "bowl_object")
        after_position = next(item["position_m"] for item in after["objects"] if item["name"] == "bowl_object")
        post = after["skill"]["postconditions"]
        injection["observed_failure"] = (after["skill"]["status"] == "failed"
                                          and bool(after["skill"]["failure_reason"])
                                          and bool(post) and not all(post.values()))
        injection["physical_effect_observed"] = (post.get("bilateral_grip") is False
                                                  or math.dist(before_position, after_position) > 0.01)
        if not injection["observed_failure"] or not injection["physical_effect_observed"]:
            raise RuntimeError("Disturbance did not produce the specified observed physical failure")

    try:
        save()
        sim = Simulation(case)
        row["initial_model_fingerprint"] = model_fingerprint(sim)
        sim.step(SETTLE_S)
        for name in spec.preparation:
            record = {"skill": name, "parameters": FROZEN_PARAMETERS.to_dict() if name == "grasp_and_hold" else None,
                      "observation": sim.state(), "result_observation": None, "status": "starting"}
            row["setup"]["history"].append(record)
            try:
                sim.start_skill(name, FROZEN_PARAMETERS if name == "grasp_and_hold" else None)
                await execute_physics(record, name, setup=True)
            except BaseException as exc:
                record["error"] = _error(exc, "setup_skill")
                if record["result_observation"] is None:
                    try:
                        record["result_observation"] = sim.state()
                    except Exception as observation_error:
                        record["result_observation_error"] = _error(observation_error, "setup_observation")
                record["status"] = "failed"
                raise
            if sim.skill.status != "succeeded":
                raise RuntimeError(sim.state()["skill"]["failure_reason"] or f"Preparation {name} failed")
        row["initial_observation"] = sim.state()
        row["initial_state_fingerprint"] = initial_state_fingerprint(sim)
        row["initial_observation_fingerprint"] = observation_fingerprint(row["initial_observation"])
        prepared_model = model_fingerprint(sim)
        if prepared_model != row["initial_model_fingerprint"]:
            raise PairingError("Physical model changed during deterministic preparation")
        verify_pairing(case_id, {
            "qpos_qvel_ctrl": row["initial_state_fingerprint"],
            "model": prepared_model, "controller_observation": row["initial_observation_fingerprint"],
        }, fingerprints)
        row["setup"].update(status="succeeded", sim_s=float(sim.data.time), wall_s=time.monotonic() - setup_started)
        setup_complete = True
        evaluation_started_sim, evaluation_started_wall = float(sim.data.time), time.monotonic()
        clock = RealtimeClock(sim)
        row["status"] = "running"
        if method == "gpt":
            session = PlanningSession(root, sim)
            if not session.start(spec.task, experiments):
                stop("planning_failed", "provider_failure", session.state().get("error") or "Subscription unavailable")
        save()
        index = 0
        while not row["stopped"]:
            scene = sim.state()
            if scene["skill"]["status"] == "failed":
                if session is not None:
                    session.advance(allow_execution=False)
                stop("failed", "physical_failure", scene["skill"]["failure_reason"] or "Physical skill failed")
                break
            if method == "gpt":
                session.advance(allow_execution=False)
                sync_session()
                state = row["planning"]
                payload = state["decision"]
                if payload is not None and state["decision_id"] not in checked_decisions:
                    checked_decisions.add(state["decision_id"])
                    try:
                        enforce_constraints(validate_decision(payload), goal, target)
                    except (PlannerError, ValueError) as exc:
                        annotations[len(row["history"])] = {"constraint_error": str(exc), "harness_status": "rejected",
                                                             "result_observation": scene}
                        stop("rejected", "constraint_violation", str(exc))
                        break
                if state["status"] == "planning":
                    await clock.step()
                    continue
                if state["status"] == "waiting":
                    if row["gpt_request_count"] >= MAX_REQUESTS:
                        stop("failed", "limit", "Reached nine request limit")
                        break
                    session.next()
                    save()
                    continue
                if state["status"] == "unsupported":
                    annotations[len(row["history"])] = {"result_observation": scene}
                    stop("unsupported", "unsupported", payload["reason"])
                    break
                if state["status"] != "ready":
                    kind = "provider_failure"
                    if payload is not None:
                        proposed = validate_decision(payload)
                        available = next((item for item in scene["available_skills"] if item["name"] == proposed.skill), None)
                        if proposed.decision == "execute" and available is not None and available["available"] is False:
                            kind = "unavailable_skill"
                    annotations[len(row["history"])] = {"result_observation": scene}
                    stop("rejected" if kind == "unavailable_skill" else "planning_failed", kind,
                         state.get("error") or f"Planning ended with {state['status']}")
                    break
                decision = validate_decision(payload)
                record = row["history"][-1]
            else:
                if goal_satisfied(scene, goal):
                    decision = _decision("complete", goal, target,
                                         reason="Common terminal check: requested physical goal already holds")
                else:
                    decision = fixed_decision(index, goal, target) if method == "fixed" else rule_decision(scene, goal)
                record = {"index": len(row["history"]) + 1, "observation": scene,
                          "decision": decision.to_dict(), "result_observation": None, "status": "ready",
                          "model": None, "thread_id": None, "latency_s": None,
                          "execution_sim_s": None, "execution_wall_s": None}
                row["history"].append(record)
                row["decision_count"] = len(row["history"])
                enforce_constraints(decision, goal, target)
            if row["decision_count"] > MAX_REQUESTS:
                stop("failed", "limit", "Reached nine decision limit")
                break
            if decision.decision == "unsupported":
                record.update(status="unsupported", result_observation=scene)
                stop("unsupported", "unsupported", decision.reason)
                break
            try:
                validate_context(decision, scene, goal)
            except PlannerError as exc:
                record.update(status="rejected", error=str(exc), result_observation=scene)
                if session is not None:
                    annotations[record["index"]] = {"harness_status": "rejected", "error": str(exc), "result_observation": scene}
                stop("rejected", "unavailable_skill" if decision.decision == "execute" else "invalid_completion", str(exc))
                break
            if decision.decision == "execute" and row["executed_skill_count"] >= MAX_SKILLS:
                stop("failed", "limit", "Reached eight skill limit")
                break
            try:
                if session is not None:
                    session.execute(row["planning"]["decision_id"], continuous=False)
                elif decision.decision == "execute":
                    sim.start_skill(decision.skill, decision.parameters)
            except (PlannerError, ValueError) as exc:
                record.update(status="rejected", error=str(exc), result_observation=sim.state())
                if session is not None:
                    annotations[record["index"]] = {
                        "harness_status": "rejected", "error": str(exc),
                        "result_observation": record["result_observation"],
                    }
                stop("rejected", "dispatch_rejected", str(exc))
                break
            if decision.decision == "complete":
                record.update(status="complete", result_observation=sim.state())
                validation = row["validation"]
                validation.update(status="running", before=sim.state())
                start_hold = float(sim.data.time)
                save()
                try:
                    for _ in range(round(VALIDATION_HOLD_S / SAMPLE_PERIOD_S)):
                        await clock.step()
                        if session is not None:
                            session.advance(allow_execution=False)
                        if not goal_satisfied(sim.state(), goal):
                            raise PlannerError("Requested physical goal lost during independent final hold")
                    validate_context(decision, sim.state(), goal)
                    validation.update(status="succeeded", after=sim.state())
                    row.update(status="succeeded", success=True)
                except PlannerError as exc:
                    validation.update(status="failed", after=sim.state())
                    stop("failed", "validation_failure", str(exc))
                finally:
                    row["validation_sim_s"] = float(sim.data.time) - start_hold
                break
            row["executed_skill_count"] += 1
            record["status"] = "executing"
            save()
            await execute_physics(record, decision.skill)
            # The disturbance is injected before session.advance/next: no feedback
            # request or subsequent accepted skill can race this physical boundary.
            if case_id == "disturbed" and decision.skill == "grasp_and_hold" and sim.skill.status == "succeeded":
                await inject_disturbance()
                record["result_observation"] = sim.state()
                record["status"] = sim.skill.status
                if session is not None:
                    annotations[record["index"]].update(
                        result_observation=record["result_observation"],
                        harness_execution_status=record["status"],
                    )
            if session is not None:
                session.advance(allow_execution=False)
            index += 1
            capture()
            save()
        capture()
    except (Exception, KeyboardInterrupt, asyncio.CancelledError) as exc:
        row["status"] = "failed" if setup_complete else "setup_failed"
        row["stop_kind"] = "internal_error"
        row["stopped"] = True
        row["pairing_failed"] = isinstance(exc, PairingError)
        _record_error(row, exc, "evaluation" if setup_complete else "setup", infrastructure=True)
        if row["validation"]["status"] == "running":
            row["validation"]["status"] = "failed"
        if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError)):
            row["interrupted"] = True
    finally:
        if not setup_complete:
            row["setup"].update(status="failed", sim_s=float(sim.data.time) if sim is not None else 0.0,
                                wall_s=time.monotonic() - setup_started)
        if sim is not None:
            try:
                capture()
            except Exception as exc:
                row["final_state"] = None
                _record_error(row, exc, "final_observation", infrastructure=True)
            try:
                row["final_model_fingerprint"] = model_fingerprint(sim)
                row["model_invariant"] = row["initial_model_fingerprint"] == row["final_model_fingerprint"]
                if not row["model_invariant"]:
                    raise PairingError("Physical model changed during a comparison row")
            except Exception as exc:
                row["pairing_failed"] = isinstance(exc, PairingError) or row["pairing_failed"]
                _record_error(row, exc, "model_invariance", infrastructure=True)
        if session is not None:
            try:
                await session.close()
            except (Exception, KeyboardInterrupt, asyncio.CancelledError) as exc:
                _record_error(row, exc, "session_cleanup", infrastructure=True)
            try:
                sync_session()
            except Exception as exc:
                _record_error(row, exc, "session_evidence", infrastructure=True)
        if sim is not None:
            try:
                sim.close()
            except Exception as exc:
                _record_error(row, exc, "simulation_cleanup", infrastructure=True)
        if row["infrastructure_errors"]:
            row["status"] = "failed" if setup_complete else "setup_failed"
            row["success"] = False
        if row["expected_outcome"] == "safe_stop":
            row["safety_pass"] = safety_pass(row)
        try:
            save()
        except Exception as exc:
            _record_error(row, exc, "checkpoint", infrastructure=True)
            row["status"] = "failed"
            row["safety_pass"] = False if row["expected_outcome"] == "safe_stop" else None
            raise
    return row


def _strict_evidence(report: dict) -> dict:
    # Keep diagnostic nonfinite values visible without emitting nonstandard JSON.
    return json.loads(json.dumps(report, ensure_ascii=False),
                      parse_constant=lambda token: {"nonfinite": token})


def _protect_output(output: Path, root: Path) -> None:
    if output.resolve() == (root / "results" / "parameter_sweep.json").resolve():
        raise ValueError("Comparison output must not overwrite frozen calibration")
    if output.exists():
        raise ValueError("Refusing to overwrite existing evidence; choose a new output filename")


async def run_comparison(
    methods: tuple[str, ...] = METHODS, cases: tuple[str, ...] | None = None,
    output: Path = DEFAULT_OUTPUT, root: Path = ROOT, *, suite: str = "original",
    repetitions: int = 1, seed: int = DEFAULT_SEED,
) -> dict:
    validate_methods(methods)
    if suite not in ("original", "expanded"):
        raise ValueError("Unknown experiment suite")
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("Repetitions must be a positive integer")
    if cases is None:
        cases = EXPANDED_CASES if suite == "expanded" else CASES
    if not cases or len(set(cases)) != len(cases) or any(case not in EXPANDED_CASES for case in cases):
        raise ValueError("Use distinct case names from " + ", ".join(EXPANDED_CASES))
    cases = tuple(case for case in EXPANDED_CASES if case in cases)
    specs = [comparison_case(case_id, suite, seed) for case_id in cases]
    schedule = []
    for repetition in range(1, repetitions + 1):
        for case_index, case_id in enumerate(cases):
            shift = (case_index + repetition - 1) % len(methods)
            for method in methods[shift:] + methods[:shift]:
                schedule.append({"case_id": case_id, "repetition": repetition, "method": method})
    shared_goal = specs[0].goal if len({spec.goal for spec in specs}) == 1 else "mixed"
    output, root = Path(output), Path(root)
    _protect_output(output, root)
    report = {
        "kind": "paired_closed_loop_comparison", "status": "running",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "methods": list(methods), "cases": list(cases), "goal": shared_goal,
            "task": specs[0].task if len({spec.task for spec in specs}) == 1 else "Per-case frozen requests; see case_specs",
            "suite": suite, "repetitions": repetitions, "seed": seed,
            "case_specs": [spec.to_dict() for spec in specs], "schedule": schedule,
            "randomization": {
                "xy_jitter_m": 0.003 if suite == "expanded" else 0.0,
                "yaw_range_rad": 0.06 if suite == "expanded" else 0.0,
                "repeats_share_initial_state": True,
            },
            "planned_runs": len(schedule), "frozen_parameters": FROZEN_PARAMETERS.to_dict(),
            "parameter_constraint": "Exact frozen grasp parameters; mismatches rejected, never overwritten",
            "sample_period_s": SAMPLE_PERIOD_S, "settle_s": SETTLE_S,
            "validation_hold_s": VALIDATION_HOLD_S, "max_skills": MAX_SKILLS, "max_requests": MAX_REQUESTS,
            "provider_retries": 0, "physics_during_provider_wait": True,
            "evaluation_realtime": True, "setup_accelerated": True,
            "note": "All rows rebuild real deterministic physics and pair exact qpos/qvel/ctrl, full observed controller state, "
                    "and compiled-model fingerprints. Setup is excluded from evaluation latency. Both baselines share "
                    "an already-goal-complete terminal check. Otherwise fixed starts at the requested goal's sequence "
                    "with no state-based resume or skill skipping; rule uses the request and observed predicates. GPT uses actual subscription decisions with manual "
                    "per-step authorization. Failed skills stop all later decisions. Success needs an independent five-second "
                    "physical goal hold. Disturbed safety acceptance measures the shared physical failure guard only, "
                    "not GPT recovery or reasoning. Safety acceptance is not task success. Same-condition or fixed-100% "
                    "results do not establish general planning superiority. All positive-attempt wall times are reported "
                    "alongside successful-only times; early failure censors duration, so a shorter failed attempt is not "
                    "evidence of faster successful completion. "
                    "Infrastructure failures abort the matrix and are excluded from performance denominators, "
                    "as are unfinished rows; both remain visible with separate counts. Timing means disclose "
                    "their measured sample counts; missing measurements are not zero. "
                    "Simulation and wall times are separate units. "
                    "Completed means every selected pair was attempted, not that every task passed; no failed-row reruns. "
                    "The entire case manifest and rotated method schedule are saved before the first run. "
                    "Repeated runs share each case's exact seeded pose and initial physical state, including preparation. "
                    "They measure within-case variability, not independent unseen scenes or population generalization. "
                    "Mixed-goal overall means are bookkeeping only; compare by case and matched successful pairs.",
        },
        "runs": [], "summary": [], "analysis": {"by_case": [], "paired": []}, "error": None,
    }
    fingerprints = {}

    def checkpoint(row=None):
        if row is not None and not any(existing is row for existing in report["runs"]):
            report["runs"].append(row)
        report["summary"] = summarize_runs(report["runs"], methods)
        report["analysis"] = analyze_runs(report["runs"], methods)
        write_report(output, _strict_evidence(report))

    try:
        checkpoint()
        experiments = {}
        if "gpt" in methods:
            experiments = json.loads((root / "results" / "parameter_sweep.json").read_text(encoding="utf-8"))
            if not isinstance(experiments, dict):
                raise ValueError("Frozen calibration evidence must be a JSON object")
        aborted = False
        for entry in schedule:
            row = await run_case(
                entry["case_id"], entry["method"], fingerprints, experiments, root, checkpoint,
                suite=suite, seed=seed, repetition=entry["repetition"],
            )
            print(f"{row['case_id']}/r{row['repetition']}/{row['method']}: {row['status']}"
                  + (f" — {row['failure_reason']}" if row["failure_reason"] else ""),
                  file=sys.stderr, flush=True)
            if row["infrastructure_errors"] or row["pairing_failed"] or row.get("interrupted"):
                report["error"] = (row["infrastructure_errors"][0]
                                   if row["infrastructure_errors"] else row["error"])
                aborted = True
                break
        report["status"] = "failed" if aborted else "completed"
    except (Exception, KeyboardInterrupt, asyncio.CancelledError) as exc:
        report["status"] = "failed"
        if report["error"] is None:
            report["error"] = _error(exc, "matrix")
        else:
            report.setdefault("secondary_errors", []).append(_error(exc, "matrix"))
    finally:
        try:
            checkpoint()
        except (Exception, KeyboardInterrupt, asyncio.CancelledError):
            # A broken disk must not erase the experiment failure that preceded it.
            for row in report["runs"]:
                if row["error"] is not None:
                    error = row["error"]
                    print(f"{row['case_id']}/{row['method']}: {error['type']}: {error['message']}",
                          file=sys.stderr)
            if report["error"] is not None:
                error = report["error"]
                print(f"Matrix failed: {error['type']}: {error['message']}", file=sys.stderr)
            raise
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--cases", nargs="+", choices=EXPANDED_CASES)
    parser.add_argument("--suite", choices=("original", "expanded"), default="original")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--seed", type=unsigned_seed, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    try:
        report = asyncio.run(run_comparison(
            tuple(args.methods), tuple(args.cases) if args.cases is not None else None, args.output,
            suite=args.suite, repetitions=args.repetitions, seed=args.seed,
        ))
    except (OSError, ValueError, TypeError) as exc:
        print(f"Cannot write comparison evidence: {exc}", file=sys.stderr)
        return 1
    print(f"{report['status']}: {len(report['runs'])} paired rows saved to {args.output}", file=sys.stderr)
    if report["error"]:
        print(f"{report['error']['type']}: {report['error']['message']}", file=sys.stderr)
    for row in report["runs"]:
        accepted = row["success"] if row["expected_outcome"] == "success" else row["safety_pass"] is True
        if row["infrastructure_errors"] or not accepted:
            print(f"{row['case_id']}/r{row.get('repetition', 1)}/{row['method']} [{row['status']}]: {row['failure_reason']}",
                  file=sys.stderr)
    return 0 if report_passed(report) else 1


if __name__ == "__main__":
    raise SystemExit(main())
