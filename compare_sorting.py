"""Paired fixed-rule versus real subscription GPT sorting, with independent goals.

The baseline receives the declared mapping/priority as structured configuration;
GPT receives the corresponding natural-language task. Both use the same physical
controller. This small deterministic capability check is not a success-rate study.
"""

import argparse
import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from sorting import SortingSimulation
from sorting_contract import OBJECTS, SortStep, SortingPlan, validate_plan
from sorting_planning import SortingPlanningSession
from verify_skills import json_observations, model_fingerprint, physics_snapshot, require
from verify_sorting import STEP_S, execute_and_verify, observe_sorting

# The frozen session exposes safe messages, not exception classes. Only these
# explicit malformed-proposal outcomes are scored; unknown failures abort.
_EXPECTED_PROPOSAL_ERRORS = frozenset({
    "模型分拣计划不符合六物体安全约束，未执行动作。",
    "Codex 决策不是有效 JSON，未执行本次动作。",
})


ROOT = Path(__file__).resolve().parent
METHODS = ("fixed_rule", "gpt")


@dataclass(frozen=True)
class SortingCase:
    name: str
    destinations: tuple[str, str, str]
    cups_first: bool
    task: str

    @property
    def color_destinations(self):
        return dict(zip(("red", "blue", "yellow"), self.destinations))


CASES = (
    SortingCase(
        "standard", ("A", "B", "C"), False,
        "把全部六件碗杯按颜色分区：红色碗和杯放 A 区，蓝色碗和杯放 B 区，"
        "黄色碗和杯放 C 区。你根据当前物体位置选择处理顺序，全部完成后空手归位。"
        "每个区域的碗使用本地前排落点、杯使用后排落点，不堆叠。",
    ),
    SortingCase(
        "swapped_cups_first", ("B", "A", "C"), True,
        "更换分区规则：红色碗和杯放 B 区，蓝色碗和杯放 A 区，黄色碗和杯放 C 区。"
        "先完成所有三个杯子，再处理三个碗；每组内顺序由你根据当前状态决定。"
        "必须处理全部六件，最后空手归位。区域地面底色仅是标记，不改变这次明确指定的规则；"
        "碗使用本地前排落点、杯使用后排落点，不堆叠。",
    ),
)


def fixed_rule_plan(case):
    """A frozen rule receives the same objective, not the GPT-produced proposal."""
    objects = list(OBJECTS)
    if case.cups_first:
        objects.sort(key=lambda obj: obj.kind != "cup")
    destinations = case.color_destinations
    return SortingPlan(tuple(SortStep(obj.id, destinations[obj.color]) for obj in objects))


def require_matches_case(plan, case):
    """Benchmark authorization: enforce the objective declared before model access."""
    plan = validate_plan(plan.to_list())
    colors = {obj.id: obj.color for obj in OBJECTS}
    require(all(step.destination == case.color_destinations[colors[step.object_id]]
                for step in plan.steps), "Proposed destinations contradict the requested task")
    if case.cups_first:
        kinds = {obj.id: obj.kind for obj in OBJECTS}
        require([kinds[step.object_id] for step in plan.steps] == ["cup"] * 3 + ["bowl"] * 3,
                "Proposed queue violates the requested cups-before-bowls priority")


def pairing_evidence(sim):
    physics = physics_snapshot(sim)
    physics.update({name: getattr(sim.data, name).tolist()
                    for name in ("qacc_warmstart", "act", "mocap_pos", "mocap_quat")})
    model_hash = model_fingerprint(sim)
    encoded = json.dumps({"model": model_hash, "physics": physics}, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode()
    return {"sha256": hashlib.sha256(encoded).hexdigest(),
            "model_sha256": model_hash, "physics": physics}


async def run_case(case, method, record, expected_pairing=None, *, simulation_factory=SortingSimulation):
    if method not in METHODS:
        raise ValueError("Unknown sorting comparison method")
    record.update({"case_id": case.name, "method": method, "task": case.task,
                   "expected_destinations": case.color_destinations,
                   "cups_first": case.cups_first, "passed": False, "errors": [],
                   "status": "preparing", "planner": None, "final_state": None,
                   "final_physics": None, "independent_final_observation": None, "gpt_latency_s": None,
                   "planning_sim_s": 0.0, "execution_wall_s": None})
    started_wall = time.monotonic()
    sim = session = None
    primary_error, interrupted = None, False

    def cleanup_error(stage, exc):
        nonlocal primary_error, interrupted
        record["passed"] = False
        record.setdefault("failure_stage", stage)
        record["errors"].append({"stage": stage, "type": type(exc).__name__, "message": str(exc)})
        if primary_error is None:
            primary_error = exc
        else:
            primary_error.add_note(f"Cleanup also failed at {stage}: {exc!r}")
        interrupted = interrupted or not isinstance(exc, Exception)

    try:
        sim = simulation_factory()
        sim.step(0.5)
        record["pairing"] = pairing_evidence(sim)
        if expected_pairing is not None:
            require(record["pairing"]["sha256"] == expected_pairing,
                    "Initial physical state/model mismatch; no planning or motion authorized")
        record["status"] = "planning"
        if method == "gpt":
            record["status"] = "planning_setup"
            session = SortingPlanningSession(ROOT, sim)
            if not session.start(case.task):
                raise RuntimeError(session.state()["error"] or "GPT planning could not start")
            record["status"] = "planning"
            planning_start_sim = float(sim.data.time)
            planning_start_wall = time.monotonic()
            while session.state()["status"] == "planning":
                if time.monotonic() - planning_start_wall >= 120.0:
                    raise TimeoutError("Sorting planning exceeded the independent wall-time watchdog")
                step_wall = time.monotonic()
                sim.step(STEP_S)
                session.advance(allow_execution=False)
                require(not sim.warning_names() and sim.skill.status == "idle",
                        "Physics became invalid or executed before approval")
                require(not np.any(sim.data.xfrc_applied) and not np.any(sim.data.qfrc_applied),
                        "Planning wait applied external forces")
                await asyncio.sleep(max(0.0, STEP_S - (time.monotonic() - step_wall)))
            state = session.state()
            record["planning_sim_s"] = float(sim.data.time) - planning_start_sim
            record["gpt_latency_s"] = state["latency_s"]
            record["planner"] = state
            if state["status"] == "failed" and state["error"] not in _EXPECTED_PROPOSAL_ERRORS:
                record["status"] = "planning_provider"
                raise RuntimeError(state["error"] or "Unknown frozen planner failure")
            if state["status"] not in ("ready", "unsupported", "failed"):
                raise RuntimeError(f"Unexpected frozen planner state: {state['status']}")
            require(state["status"] == "ready", f"GPT proposal unavailable: {state['status']}: {state['error']}")
            plan = validate_plan(state["decision"]["steps"])
            # This automation substitutes only for explicit approval of a matching
            # proposal. It never repairs wrong model output or dispatches fallback.
            require_matches_case(plan, case)
            decision_id = state["decision_id"]
            start = lambda: session.execute(decision_id)
            after_step = lambda: session.advance(allow_execution=False)
        else:
            plan = fixed_rule_plan(case)
            require_matches_case(plan, case)
            start = lambda: sim.start_sorting(plan)
            after_step = None
        record["approved_plan"] = plan.to_list()
        require(model_fingerprint(sim) == record["pairing"]["model_sha256"],
                "Compiled model changed before execution")
        record["status"] = "executing"
        execution_start = time.monotonic()
        try:
            execute_and_verify(sim, record, color_destinations=case.color_destinations,
                               expected_plan=plan, start=start, after_step=after_step)
        finally:
            record["execution_wall_s"] = time.monotonic() - execution_start
        if session is not None:
            session.advance(allow_execution=False)
            require(session.state()["status"] == "succeeded",
                    "GPT session did not acknowledge the physical result")
        record["status"] = "succeeded"
    except BaseException as exc:
        primary_error = exc
        interrupted = not isinstance(exc, Exception)
        record["passed"] = False
        record["failure_stage"] = record["status"]
        record["status"] = "failed"
        record["errors"].append({"type": type(exc).__name__, "message": str(exc)})
    finally:
        if session is not None:
            try:
                record["planner"] = session.state()
                record["gpt_latency_s"] = record["planner"]["latency_s"]
            except BaseException as exc:
                cleanup_error("planner_state", exc)
            try:
                await session.close()
            except BaseException as exc:
                cleanup_error("planner_close", exc)
        if sim is not None:
            for key, observe in (
                ("final_state", sim.state),
                ("final_physics", lambda: physics_snapshot(sim)),
                ("independent_final_observation", lambda: observe_sorting(
                    sim, sim.state()["sorting"], case.color_destinations)),
            ):
                try:
                    record[key] = observe()
                except BaseException as exc:
                    cleanup_error(key, exc)
            try:
                sim.close()
            except BaseException as exc:
                cleanup_error("simulation_close", exc)
        if not record["passed"]:
            record["status"] = "failed"
        record["wall_s"] = time.monotonic() - started_wall
        if interrupted:
            raise primary_error


async def run_comparison(report, cases):
    for case in cases:
        expected_pairing = None
        for method in METHODS:
            record = {}
            report["runs"].append(record)
            await run_case(case, method, record, expected_pairing)
            if not record["passed"]:
                return
            expected_pairing = record["pairing"]["sha256"]
    report["passed"] = len(report["runs"]) == len(cases) * len(METHODS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", choices=[case.name for case in CASES],
                        default=[case.name for case in CASES])
    args = parser.parse_args()
    if len(set(args.cases)) != len(args.cases):
        parser.error("Cases must be distinct")
    cases = [next(case for case in CASES if case.name == name) for name in args.cases]
    report = {
        "kind": "paired_sorting_comparison", "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Two deterministic known-layout objectives; not success-rate or generalization evidence",
        "protocol": {
            "methods": list(METHODS), "cases": args.cases,
            "fixed_rule": "Predeclared structured color mapping/priority; does not parse natural language",
            "gpt": "Real subscription request; approve only an exact objective-compatible proposal; no retries/fallback",
            "pairing": "Full model plus qpos/qvel/ctrl/equalities/forces/solver warmstart/activation/mocap before planning",
            "planning_clock": "Physics advances while awaiting GPT; no manipulation before approval",
            "execution_clock": "Accelerated headless; wall time is not real-time throughput",
            "timing": "Execution simulation time, provider wall latency and headless execution wall time remain separate",
            "controls": "Identical physical controller and limits for both methods; additional independent 5 s hold",
        },
        "runs": [], "passed": False,
    }
    with args.output.open("x", encoding="utf-8") as output:
        try:
            asyncio.run(run_comparison(report, cases))
        finally:
            json.dump(json_observations(report), output, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            output.write("\n")
    print(json.dumps({"output": str(args.output), "passed": report["passed"], "runs": [
        {"case": run["case_id"], "method": run["method"], "passed": run["passed"],
         "execution_sim_s": (run["final_state"] or {}).get("sorting", {}).get("elapsed_s"),
         "gpt_latency_s": run["gpt_latency_s"], "errors": run["errors"]}
        for run in report["runs"]]}, ensure_ascii=False, indent=2, allow_nan=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
