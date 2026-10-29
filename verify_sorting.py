"""Execute fixed sorting and save independent geometry/contact observations.

Simulation time and accelerated headless wall time are recorded separately.
Positive runs never reset, teleport objects, attach them, or change the model.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

import mujoco
import numpy as np

from sorting import SortingSimulation
from sorting_contract import FIXED_PLAN, SortingPlan
from verify_skills import json_observations, model_fingerprint, physics_fingerprints, physics_snapshot, require

COLOR_DESTINATIONS = {"red": "A", "blue": "B", "yellow": "C"}
STEP_S = 0.02
SAMPLE_S = 0.10
MAX_RUN_S = 600.0


def free_joint(model, body):
    """Find the movable assembly's free joint, including a fixed object child."""
    while body:
        start = int(model.body_jntadr[body])
        for joint in range(start, start + int(model.body_jntnum[body])):
            if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE:
                return joint
        body = int(model.body_parentid[body])
    raise ValueError("Sorting object has no free joint")


def observe_sorting(sim, specification, color_destinations=None):
    """Measure against caller-specified goals, not a model's proposed destinations."""
    destinations = COLOR_DESTINATIONS if color_destinations is None else color_destinations
    model, data = sim.model, sim.data
    table = model.body("table").id
    tabletop = model.geom("tabletop").id
    table_top = float(data.geom_xpos[tabletop, 2] + np.abs(
        data.geom_xmat[tabletop].reshape(3, 3)[2]) @ model.geom_size[tabletop])
    fingers = {model.body("robot_left_finger").id, model.body("robot_right_finger").id}
    robot_bodies = {body for body in range(model.nbody)
                    if (model.body(body).name or "").startswith("robot_")}
    collidable = (model.geom_contype | model.geom_conaffinity) != 0
    body_ids = {obj["id"]: model.body(obj["body_name"]).id for obj in specification["objects"]}
    forces = {body: {"table": 0.0, **{str(finger): 0.0 for finger in fingers}}
              for body in body_ids.values()}
    touching_fingers = set()
    hand_empty = True
    force = np.zeros(6)
    for index, contact in enumerate(data.contact):
        a, b = (int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2]))
        for body, other in ((a, b), (b, a)):
            if body in forces and other in robot_bodies:
                hand_empty = False
            if body in forces and other in fingers:
                touching_fingers.add(body)
        if contact.efc_address < 0:
            continue
        mujoco.mj_contactForce(model, data, index, force)
        normal = max(0.0, float(force[0]))
        for body, other in ((a, b), (b, a)):
            if body not in forces:
                continue
            if other == table and tabletop in (contact.geom1, contact.geom2):
                upward = float(contact.frame[2]) * (1.0 if body == b else -1.0)
                forces[body]["table"] += normal * max(0.0, upward)
            elif other in fingers:
                forces[body][str(other)] += normal
    objects = []
    for obj in specification["objects"]:
        body = body_ids[obj["id"]]
        geoms = np.flatnonzero((model.geom_bodyid == body) & collidable)
        require(len(geoms) > 0, f"Missing collision geometry: {obj['id']}")
        rotation = data.geom_xmat[geoms].reshape(-1, 3, 3)
        bounds = model.geom_aabb[geoms]
        centers = data.geom_xpos[geoms] + np.einsum("nij,nj->ni", rotation, bounds[:, :3])
        extents = np.einsum("nij,nj->ni", np.abs(rotation), bounds[:, 3:])
        low, high = np.min(centers - extents, axis=0), np.max(centers + extents, axis=0)
        destination = destinations[obj["color"]]
        zone = model.geom(f"zone_{destination}").id
        center, half_size = data.geom_xpos[zone, :2], model.geom_size[zone, :2]
        slot = data.site(f"slot_{destination}_{0 if obj['kind'] == 'bowl' else 1}").xpos[:2]
        joint = free_joint(model, body)
        dof = int(model.jnt_dofadr[joint])
        velocity = data.qvel[dof:dof + 6]
        finger_forces = [forces[body][str(finger)] for finger in sorted(fingers)]
        objects.append({
            "id": obj["id"], "body_name": obj["body_name"], "color": obj["color"],
            "kind": obj["kind"], "expected_destination": destination,
            "expected_slot_xy_m": slot.tolist(),
            "position_m": data.xpos[body].tolist(), "velocity": velocity.tolist(),
            "collision_bounds_m": [low.tolist(), high.tolist()],
            "finger_forces_n": finger_forces, "table_force_n": forces[body]["table"],
            "bilateral_grip": all(value > 0.01 for value in finger_forces),
            "correct_region": bool(np.all(low[:2] >= center - half_size)
                                   and np.all(high[:2] <= center + half_size)),
            "slot_aligned": bool(np.linalg.norm(data.xpos[body, :2] - slot) < 0.025),
            "upright": bool(data.xmat[body].reshape(3, 3)[2, 2] > 0.95),
            "supported": bool(forces[body]["table"] > 0.01
                              and table_top - 0.003 <= low[2] <= table_top + 0.003),
            "stable": bool(np.linalg.norm(velocity[:3]) < 0.01 and np.linalg.norm(velocity[3:]) < 0.1),
            "detached": body not in touching_fingers,
        })
    arm_dofs = [int(model.joint(f"robot_joint{i}").dofadr[0]) for i in range(1, 8)]
    return {
        "time_s": float(data.time), "objects": objects, "hand_empty": hand_empty,
        "arm_home_error_rad": float(np.max(np.abs(data.qpos[sim.arm_qpos] - sim.arm_home))),
        "arm_max_speed_rad_s": float(np.max(np.abs(data.qvel[arm_dofs]))),
        "gripper_opening_m": float(data.qpos[sim.finger_qpos].sum()),
    }


def independent_complete(observation):
    objects = observation["objects"]
    return (len(objects) == 6
            and all(obj[key] for obj in objects for key in (
                "correct_region", "slot_aligned", "upright", "supported", "stable", "detached"))
            and observation["hand_empty"]
            and observation["arm_home_error_rad"] < 0.02
            and observation["arm_max_speed_rad_s"] < 0.05)


def validate_inventory(sim, specification, color_destinations=None):
    destinations = COLOR_DESTINATIONS if color_destinations is None else color_destinations
    require(set(destinations) == set(COLOR_DESTINATIONS) and set(destinations.values()) == set("ABC"),
            "Independent goal must assign each color to one distinct known zone")
    objects, plan = specification["objects"], specification["plan"]
    identities = {obj["id"] for obj in objects}
    require(len(objects) == len(identities) == 6, "Expected six distinct object IDs")
    require(len({sim.model.body(obj["body_name"]).id for obj in objects}) == 6,
            "Inventory must describe six distinct physical bodies")
    require({(obj["color"], obj["kind"]) for obj in objects}
            == {(color, kind) for color in COLOR_DESTINATIONS for kind in ("bowl", "cup")},
            "Expected exactly one bowl and one cup of each color")
    require(all(obj["id"] == f"{obj['color']}_{obj['kind']}"
                and obj["body_name"] == f"{obj['id']}_object" for obj in objects),
            "Inventory identity must match its physical body")
    require(len(plan) == 6 and {item["object_id"] for item in plan} == identities,
            "Queue must contain every physical object exactly once")
    by_id = {obj["id"]: obj for obj in objects}
    require(all(item["destination"] == destinations[by_id[item["object_id"]]["color"]]
                for item in plan), "Queue destination contradicts the independently requested rule")


class PhysicalHandlingSequence:
    """Infer handling order only from sampled contacts, heights and released support."""

    def __init__(self, expected_plan, initial_z):
        self.expected = [step.object_id for step in expected_plan.steps]
        self.initial_z = initial_z
        self.lifted = []
        self.placed = []

    def observe(self, observation):
        facts = {obj["id"]: obj for obj in observation["objects"]}
        require(set(facts) == set(self.expected), "Physical order observation has wrong identities")
        for identity in self.lifted:
            obj = facts[identity]
            if identity not in self.placed and all(obj[key] for key in (
                    "correct_region", "slot_aligned", "upright", "supported", "stable", "detached")):
                require(identity == self.expected[len(self.placed)],
                        f"Observed physical placement out of approved order: {identity}")
                self.placed.append(identity)
        for identity, obj in facts.items():
            lifted = (obj["bilateral_grip"] and not obj["supported"]
                      and obj["position_m"][2] - self.initial_z[identity] >= 0.10)
            if not lifted:
                continue
            require(identity not in self.placed, f"Previously placed object physically lifted again: {identity}")
            if identity in self.lifted:
                continue
            require(len(self.lifted) == len(self.placed),
                    "Next object physically lifted before the preceding placement was verified")
            require(identity == self.expected[len(self.lifted)],
                    f"Observed physical lift out of approved order: {identity}")
            self.lifted.append(identity)

    def require_complete(self):
        require(self.lifted == self.expected and self.placed == self.expected,
                "Missing independently observed ordered lift/release transitions")


def execute_and_verify(sim, record, *, color_destinations=None, expected_plan=None, start=None, after_step=None):
    """Run one already-prepared physical scene; authorization is supplied by caller."""
    record.update({"passed": False, "trace": [], "errors": []})
    expected_plan = FIXED_PLAN if expected_plan is None else SortingPlan(expected_plan.steps)
    record["expected_plan"] = expected_plan.to_list()
    record["initial_physics"] = physics_snapshot(sim)
    record["initial_model_sha256"] = model_fingerprint(sim)
    initial_parameters = physics_fingerprints(sim)
    initial = observe_sorting(sim, sim.state()["sorting"], color_destinations)
    initial_z = {obj["id"]: obj["position_m"][2] for obj in initial["objects"]}
    grasp_evidence = {obj["id"]: {"bilateral_seen": False, "lifted_while_gripped": False}
                      for obj in initial["objects"]}
    record["grasp_evidence"] = grasp_evidence
    sequence = PhysicalHandlingSequence(expected_plan, initial_z)
    record["physical_lift_order"] = sequence.lifted
    record["physical_placement_order"] = sequence.placed

    def checked_step():
        before = float(sim.data.time)
        sim.step(STEP_S)
        require(np.isclose(float(sim.data.time) - before, STEP_S, rtol=0, atol=1e-8),
                "Physics step did not advance by its requested duration")
        require(physics_fingerprints(sim) == initial_parameters,
                "Physics parameters changed during execution")
        require(np.array_equal(sim.data.eq_active, record["initial_physics"]["eq_active"]),
                "Equality activation changed during execution")
        require(not np.any(sim.data.xfrc_applied) and not np.any(sim.data.qfrc_applied),
                "Positive run applied external forces")
        if after_step is not None:
            after_step()

    (sim.start_sorting if start is None else start)()
    require(physics_snapshot(sim) == record["initial_physics"], "Starting a plan changed physics")
    specification = sim.state()["sorting"]
    record["specification"] = specification
    validate_inventory(sim, specification, color_destinations)
    require([{"object_id": item["object_id"], "destination": item["destination"]}
             for item in specification["plan"]] == expected_plan.to_list(),
            "Active queue differs from the explicitly approved plan")
    start_sim = float(sim.data.time)
    next_sample = start_sim
    previous_phase = None
    while sim.skill.status == "running":
        require(float(sim.data.time) - start_sim < MAX_RUN_S, "Sorting exceeded independent watchdog")
        checked_step()
        phase = sim.skill.phase
        if float(sim.data.time) + 1e-9 >= next_sample or phase != previous_phase:
            observed = observe_sorting(sim, specification, color_destinations)
            for obj in observed["objects"]:
                evidence = grasp_evidence[obj["id"]]
                evidence["bilateral_seen"] |= obj["bilateral_grip"]
                evidence["lifted_while_gripped"] |= (
                    obj["bilateral_grip"] and not obj["supported"]
                    and obj["position_m"][2] - initial_z[obj["id"]] >= 0.10)
            record["trace"].append({"phase": phase, "current_index": sim.skill.current_index,
                                   "observation": observed, "physics": physics_snapshot(sim)})
            sequence.observe(observed)
            next_sample = float(sim.data.time) + SAMPLE_S
            previous_phase = phase
    state = sim.state()["sorting"]
    require(sim.skill.status == "succeeded", f"Sorting failed: {state['failure_reason']}")
    require(all(all(item.values()) for item in grasp_evidence.values()),
            "A sorted object lacks observed bilateral grip and physical lift")
    final_plan = state["plan"]
    require({item["object_id"] for item in final_plan} == set(grasp_evidence)
            and len(final_plan) == 6 and all(item["status"] == "sorted" for item in final_plan),
            "Controller completed without finishing the entire queue")
    sequence.require_complete()
    require([item["object_id"] for item in state["item_results"] if item["status"] == "sorted"]
            == sequence.placed, "Controller log differs from independently observed physical order")
    require(independent_complete(observe_sorting(sim, specification, color_destinations)),
            "Controller completion disagrees with independent physical observations")
    record["extra_hold_s"] = 5.0
    independent_hold_start = float(sim.data.time)
    for _ in range(round(record["extra_hold_s"] / STEP_S)):
        checked_step()
        require(independent_complete(observe_sorting(sim, specification, color_destinations)),
                "Independent physical goal failed during extra hold")
        require(sim.skill.status == "succeeded", "Controller revoked completion")
    record["independent_final_hold_s"] = float(sim.data.time) - independent_hold_start
    require(record["independent_final_hold_s"] >= 5.0 - 1e-8,
            "Independent final hold was shorter than five simulated seconds")
    require(sim.state()["sorting"]["validation_hold_s"] >= 5.0 - 1e-8,
            "Missing full-batch five-second verification")
    require(sim.state()["collisions"]["episodes"] == 0, "Unexpected collision recorded")
    require(not sim.warning_names(), "MuJoCo warnings recorded")
    require(model_fingerprint(sim) == record["initial_model_sha256"], "Compiled model changed")
    record["passed"] = True


def run_positive(record):
    sim = None
    record.update({"kind": "positive_fixed_sorting", "passed": False, "trace": [], "errors": [],
                   "final_state": None, "final_physics": None})
    started_wall = perf_counter()
    try:
        sim = SortingSimulation()
        sim.step(0.5)
        execute_and_verify(sim, record)
    except BaseException as exc:
        record["errors"].append({"type": type(exc).__name__, "message": str(exc)})
        if not isinstance(exc, Exception):
            raise
    finally:
        record["wall_s"] = perf_counter() - started_wall
        if sim is not None:
            for key, observe in (("final_state", sim.state), ("final_physics", lambda: physics_snapshot(sim))):
                try:
                    record[key] = observe()
                except Exception as exc:
                    record["passed"] = False
                    record["errors"].append({"stage": key, "type": type(exc).__name__, "message": str(exc)})
            try:
                sim.close()
            except Exception as exc:
                record["passed"] = False
                record["errors"].append({"stage": "close", "type": type(exc).__name__, "message": str(exc)})
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.repeat <= 10:
        parser.error("--repeat must be between 1 and 10")
    # Reserve the result path before any run; never replace an old failure.
    with args.output.open("x", encoding="utf-8") as output:
        runs = []
        report = {"kind": "fixed_sorting_verification", "generated_at": datetime.now(timezone.utc).isoformat(),
                  "scope": "Known fixed layout; accelerated headless physics, not a real-time throughput benchmark",
                  "runs": runs, "passed": False}
        try:
            for _ in range(args.repeat):
                result = {}
                runs.append(result)
                run_positive(result)
                if not result["passed"]:
                    break
            report["passed"] = len(runs) == args.repeat and all(run["passed"] for run in runs)
        finally:
            json.dump(json_observations(report), output, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            output.write("\n")
    summary = {"output": str(args.output), "passed": report["passed"],
               "runs": [{"passed": run["passed"], "wall_s": run["wall_s"],
                         "sorting": {key: value for key, value in (run["final_state"] or {}).get("sorting", {}).items()
                                     if key in ("status", "phase", "sorted_count", "elapsed_s", "failure_reason")},
                         "errors": run["errors"]} for run in runs]}
    print(json.dumps(json_observations(summary), ensure_ascii=False, indent=2, allow_nan=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
