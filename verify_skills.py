"""Record real MuJoCo skill boundaries, contacts, and failure evidence.

Run: uv run python verify_skills.py --output results/skills_verification.json
The three positive runs never reset, teleport, edit the model, or apply external
forces. The separately labelled disturbance run deliberately applies a force.
Trace samples are observations, not a replay or a claim about unsampled times.
With --output, full compact JSON is saved and stdout contains only a summary.
Diagnostic failures are recorded separately without replacing execution errors.
"""

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
from time import perf_counter

import mujoco
import numpy as np

from grasp import GraspParameters
from scenarios import SceneCase
from simulation import ARM_HOME, Simulation


SKILL_CHAIN = (
    "grasp_and_hold", "place_and_release", "withdraw_from_drawer",
    "close_drawer", "return_home",
)
STEP_S = 0.02
TRACE_PERIOD_S = 0.10
BOUNDARY_HOLD_S = 1.0
FINAL_HOLD_S = 5.0
MAX_TRACE_SAMPLES = 10000
# These small arrays are monitored after every 20 ms physics advance. The full
# compiled MJB (including meshes and all other fields) is hashed at boundaries.
PHYSICS_ARRAYS = (
    "body_mass", "body_inertia", "body_pos", "body_quat", "body_gravcomp",
    "geom_size", "geom_pos", "geom_quat", "geom_type", "geom_contype",
    "geom_conaffinity", "geom_friction", "geom_solref", "geom_solimp",
    "geom_margin", "geom_gap", "site_pos", "site_quat", "site_size",
    "dof_damping", "dof_frictionloss", "dof_armature", "jnt_range",
    "jnt_limited", "jnt_stiffness", "actuator_gainprm", "actuator_biasprm",
    "actuator_dynprm", "actuator_gear", "actuator_ctrlrange",
    "actuator_forcerange", "actuator_ctrllimited", "actuator_forcelimited",
    "actuator_dyntype", "actuator_gaintype", "actuator_biastype",
    "eq_type", "eq_obj1id", "eq_obj2id", "eq_data", "eq_active0",
)
PHYSICS_OPTIONS = (
    "timestep", "gravity", "disableflags", "enableflags", "integrator",
    "solver", "iterations", "tolerance", "cone", "jacobian", "impratio",
)


def require(condition: bool, message: str) -> None:
    # Unlike a bare assert, evidence checks must also run under python -O.
    if not condition:
        raise AssertionError(message)


@contextmanager
def diagnostic_capture(record, stage):
    """Keep an in-flight execution error primary when final evidence also fails."""
    primary = sys.exc_info()[1]
    try:
        yield
    except BaseException as exc:
        record.setdefault("diagnostic_errors", []).append({
            "stage": stage, "type": type(exc).__name__, "message": str(exc),
        })
        if primary is None:
            raise


def physics_fingerprints(sim: Simulation) -> dict:
    fields = {name: getattr(sim.model, name) for name in PHYSICS_ARRAYS}
    fields.update({f"opt.{name}": getattr(sim.model.opt, name) for name in PHYSICS_OPTIONS})
    return {
        name: hashlib.sha256(np.asarray(value).tobytes()).hexdigest()
        for name, value in fields.items()
    }


def model_fingerprint(sim: Simulation) -> str:
    buffer = np.empty(mujoco.mj_sizeModel(sim.model), dtype=np.uint8)
    mujoco.mj_saveModel(sim.model, None, buffer)
    return hashlib.sha256(buffer).hexdigest()


def physics_snapshot(sim: Simulation) -> dict:
    return {
        "time_s": float(sim.data.time),
        "qpos": sim.data.qpos.tolist(),
        "qvel": sim.data.qvel.tolist(),
        "ctrl": sim.data.ctrl.tolist(),
        "eq_active": sim.data.eq_active.tolist(),
        "xfrc_applied": sim.data.xfrc_applied.tolist(),
        "qfrc_applied": sim.data.qfrc_applied.tolist(),
    }


def physical_observation(sim: Simulation) -> dict:
    """Measure constraints/geometry independently of controller success flags."""
    model, data = sim.model, sim.data
    target = model.body(f"{sim.case.target_object}_object").id
    table = model.body("table").id
    drawer = model.body("cabinet_cabinet_bottom").id
    fingers = {
        model.body("robot_left_finger").id,
        model.body("robot_right_finger").id,
    }
    collidable = (model.geom_contype | model.geom_conaffinity) != 0
    drawer_geoms = np.flatnonzero((model.geom_bodyid == drawer) & collidable)
    target_geoms = np.flatnonzero((model.geom_bodyid == target) & collidable)
    floor = int(min(drawer_geoms, key=lambda geom: model.geom_pos[geom, 2]))
    handle = int(min(drawer_geoms, key=lambda geom: model.geom_pos[geom, 1]))
    force = np.zeros(6)
    finger_forces = {str(body): 0.0 for body in sorted(fingers)}
    table_force = drawer_force = handle_force = 0.0
    for index, contact in enumerate(data.contact):
        if contact.efc_address < 0:
            continue
        a, b = int(contact.geom1), int(contact.geom2)
        body_a, body_b = int(model.geom_bodyid[a]), int(model.geom_bodyid[b])
        mujoco.mj_contactForce(model, data, index, force)
        normal_force = max(0.0, float(force[0]))
        if target in (body_a, body_b):
            other_body = body_b if body_a == target else body_a
            other_geom = b if body_a == target else a
            if other_body in fingers:
                finger_forces[str(other_body)] += normal_force
            if other_body == table:
                table_force += normal_force
            if other_geom == floor:
                drawer_force += normal_force
        if (a == handle and body_b in fingers) or (b == handle and body_a in fingers):
            handle_force += normal_force

    require(bool(np.all(model.geom_type[target_geoms] == mujoco.mjtGeom.mjGEOM_BOX)),
            "Independent containment observation requires the original box collision geometry")
    rotations = data.geom_xmat[target_geoms].reshape(-1, 3, 3)
    extents = np.einsum("nij,nj->ni", np.abs(rotations), model.geom_size[target_geoms])
    low = np.min(data.geom_xpos[target_geoms] - extents, axis=0)
    high = np.max(data.geom_xpos[target_geoms] + extents, axis=0)
    region = model.site("cabinet_bottom_region").id
    region_extent = np.abs(data.site_xmat[region].reshape(3, 3)) @ model.site_size[region]
    region_low = data.site_xpos[region] - region_extent
    region_high = data.site_xpos[region] + region_extent
    joint = model.joint("cabinet_bottom_level")
    arm_dof = [int(model.joint(f"robot_joint{i}").dofadr[0]) for i in range(1, 8)]
    target_joint = model.joint(f"{sim.case.target_object}_free")
    dof = int(target_joint.dofadr[0])
    return {
        "target_position_m": data.xpos[target].tolist(),
        "target_free_z_m": float(data.qpos[int(target_joint.qposadr[0]) + 2]),
        "target_velocity": data.qvel[dof:dof + 6].tolist(),
        "tcp_position_m": data.site("robot_grasp_center").xpos.tolist(),
        "finger_target_forces_n": finger_forces,
        "bilateral_grip": all(value > 0.01 for value in finger_forces.values()),
        "target_table_force_n": table_force,
        "target_drawer_floor_force_n": drawer_force,
        "finger_handle_force_n": handle_force,
        "target_collision_bounds_m": [low.tolist(), high.tolist()],
        "drawer_region_bounds_m": [region_low.tolist(), region_high.tolist()],
        "target_inside_drawer": bool(np.all(low >= region_low) and np.all(high <= region_high)),
        "drawer_position_m": float(data.qpos[int(joint.qposadr[0])]),
        "drawer_speed_m_s": float(data.qvel[int(joint.dofadr[0])]),
        "arm_home_error_rad": float(np.max(np.abs(data.qpos[sim.arm_qpos] - ARM_HOME))),
        "arm_max_speed_rad_s": float(np.max(np.abs(data.qvel[arm_dof]))),
    }


def boundary_snapshot(sim: Simulation) -> dict:
    return {
        "physics": physics_snapshot(sim),
        "state": deepcopy(sim.state()),
        "observed": physical_observation(sim),
        "model_sha256": model_fingerprint(sim),
        "physics_field_sha256": physics_fingerprints(sim),
    }


class EvidenceRun:
    def __init__(self, sim: Simulation, record: dict):
        self.sim, self.record = sim, record
        self.initial_model = model_fingerprint(sim)
        self.initial_fields = physics_fingerprints(sim)
        self.initial_eq_active = sim.data.eq_active.tolist()
        self.last_trace_time = -math.inf
        self.last_trace_phase = None
        self.peak_handle_force = 0.0
        self.reference = None
        self._invocation_starts = {}
        self.keep_target_on_table = False
        record.update({
            "invocations": [], "trace": [], "trace_samples_dropped": 0,
            "model_checks_executed": 0, "physics_field_checks_executed": 0,
            "physics_step_calls_executed": 0,
            "initial_model_sha256": self.initial_model,
            "initial_physics_field_sha256": self.initial_fields,
            "trace_period_s": TRACE_PERIOD_S, "physics_observation_period_s": STEP_S,
        })

    def check_model(self, full=False):
        actual = physics_fingerprints(self.sim)
        self.record["physics_field_checks_executed"] += 1
        changed = [key for key in self.initial_fields if actual[key] != self.initial_fields[key]]
        require(not changed, f"Physical model fields changed: {changed}")
        require(self.sim.data.eq_active.tolist() == self.initial_eq_active,
                "Runtime equality-constraint activation changed")
        if full:
            actual_model = model_fingerprint(self.sim)
            self.record["model_checks_executed"] += 1
            require(actual_model == self.initial_model, "Compiled physical model changed")

    def observe(self, action, force=False):
        state = self.sim.state()
        observed = physical_observation(self.sim)
        self.peak_handle_force = max(self.peak_handle_force, observed["finger_handle_force_n"])
        skill = state["skill"]
        phase = (skill["invocation_id"], skill["phase"], skill["status"])
        now = float(self.sim.data.time)
        if force or phase != self.last_trace_phase or now - self.last_trace_time >= TRACE_PERIOD_S - 1e-9:
            sample = {
                "time_s": now, "action": action, "invocation_id": skill["invocation_id"],
                "skill_name": skill["skill_name"], "status": skill["status"],
                "phase": skill["phase"], "postconditions": skill["postconditions"],
                "storage": skill["storage"], "observed": observed,
                "ctrl": self.sim.data.ctrl.tolist(),
                "collisions": state["collisions"], "warnings": state["warnings"],
            }
            if len(self.record["trace"]) < MAX_TRACE_SAMPLES:
                self.record["trace"].append(sample)
            else:
                self.record["trace_samples_dropped"] += 1
            self.last_trace_time, self.last_trace_phase = now, phase
        return state, observed

    def step(self, action, clean=True):
        before = float(self.sim.data.time)
        try:
            self.sim.step(STEP_S)
        finally:
            self.record["physics_step_calls_executed"] += 1
            with diagnostic_capture(self.record, f"observe:{action}"):
                state, observed = self.observe(action)
        require(abs(float(self.sim.data.time) - before - STEP_S) < 1e-8,
                "Physics time did not advance by the requested interval (possible hidden reset)")
        self.check_model()
        require(not state["warnings"], f"MuJoCo warnings: {state['warnings']}")
        if clean:
            require(state["collisions"]["episodes"] == 0, f"Unplanned contact: {state['collisions']}")
        if self.keep_target_on_table:
            require(observed["target_table_force_n"] > 0.01
                    and math.dist(observed["target_position_m"], self.reference["target_position_m"]) < 0.003,
                    "Independent close/home disturbed the table-supported target")
        return state, observed

    def settle(self):
        self.record["initial"] = boundary_snapshot(self.sim)
        for _ in range(round(1.0 / STEP_S)):
            self.step("settle_original_scene")
        self.reference = physical_observation(self.sim)
        self.record["settled"] = boundary_snapshot(self.sim)
        require(self.reference["target_table_force_n"] > 0.01, "Target did not settle on the table")
        self.check_model(full=True)

    def invoke(self, name, parameters=None, reject=False):
        invocation = {
            "requested_skill": name,
            "requested_parameters": parameters.to_dict() if isinstance(parameters, GraspParameters) else parameters,
            "expected": "rejection" if reject else "acceptance",
            "status": "attempting", "pre": boundary_snapshot(self.sim),
        }
        self.record["invocations"].append(invocation)
        began = perf_counter()
        try:
            self.sim.start_skill(name, parameters)
        except ValueError as exc:
            invocation.update({"status": "rejected", "reason": str(exc)})
            if not reject:
                raise
        except BaseException as exc:
            invocation.update({"status": "error", "error": {"type": type(exc).__name__, "message": str(exc)}})
            raise
        else:
            invocation["status"] = "accepted"
            self._invocation_starts[id(invocation)] = (float(self.sim.data.time), perf_counter())
            self.peak_handle_force = 0.0
        finally:
            invocation["call_wall_time_s"] = perf_counter() - began
            with diagnostic_capture(invocation, "post_call"):
                invocation["post_call"] = boundary_snapshot(self.sim)
            with diagnostic_capture(invocation, "invoke_observation"):
                self.observe("invoke_rejected" if invocation["status"] == "rejected" else "invoke", force=True)
        pre, post = invocation["pre"], invocation["post_call"]
        unchanged = pre["physics"] == post["physics"]
        invocation["call_preserved_physics"] = unchanged
        require(unchanged, f"{name!r} changed physics at the API boundary")
        self.check_model(full=True)
        if reject:
            require(invocation["status"] == "rejected", f"Unsafe request {name!r} was accepted")
            require(bool(invocation["reason"].strip()), "Refusal did not explain its precondition")
            require(pre == post, "Rejected call changed execution, collision evidence, model, or physics")
            invocation["verified"] = True
            return invocation
        require(invocation["status"] == "accepted", f"{name!r} refused: {invocation.get('reason')}")
        skill = post["state"]["skill"]
        require(skill["invocation_id"] == pre["state"]["skill"]["invocation_id"] + 1,
                "Accepted invocation did not increment invocation_id exactly once")
        require(skill["execution_mode"] == "skill" and skill["skill_name"] == name
                and skill["task_kind"] is None, "Standalone execution attribution is incorrect")
        require(pre["state"]["collisions"] == post["state"]["collisions"],
                "Handoff discarded collision evidence")
        if parameters is not None:
            require(skill["parameters"] == parameters.to_dict(), "Accepted grasp parameters were not applied")
        return invocation

    def complete(self, invocation):
        name = invocation["requested_skill"]
        start_time, began = self._invocation_starts[id(invocation)]
        try:
            spec = next(item for item in self.sim.state()["available_skills"] if item["name"] == name)
            watchdog = float(spec["watchdog_s"])
            require(math.isfinite(watchdog) and 0 < watchdog <= 120.0,
                    "Declared skill watchdog is outside this runner's 120 s safety budget")
            for _ in range(math.ceil(watchdog / STEP_S) + 1):
                if self.sim.state()["skill"]["status"] != "running":
                    break
                self.step(f"execute:{name}")
            self.assert_success(name)
            self.check_model(full=True)
            invocation["completion_verified"] = True
        except BaseException as exc:
            invocation["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            invocation["execution_wall_time_s"] = perf_counter() - began
            invocation["execution_simulation_time_s"] = float(self.sim.data.time) - start_time
            invocation["peak_observed_finger_handle_force_n"] = self.peak_handle_force
            with diagnostic_capture(invocation, "terminal"):
                invocation["terminal"] = boundary_snapshot(self.sim)

    def assert_success(self, name):
        state = self.sim.state()
        skill = state["skill"]
        require(skill["status"] == "succeeded", f"{name}: {skill['status']}: {skill['failure_reason']}")
        require(skill["skill_name"] == name, f"{name} automatically advanced to another skill")
        require(bool(skill["postconditions"]) and all(value is True for value in skill["postconditions"].values()),
                f"{name} has false or absent physical postconditions: {skill['postconditions']}")
        observed = physical_observation(self.sim)
        if name == "grasp_and_hold":
            require(observed["bilateral_grip"], "Grasp succeeded without real bilateral grip forces")
            require(observed["target_position_m"][2] - self.reference["target_position_m"][2] >= 0.10,
                    "Grasp succeeded without 0.10 m body lift")
            require(observed["target_free_z_m"] - self.reference["target_free_z_m"] >= 0.10,
                    "Grasp succeeded without 0.10 m free-joint lift")
        elif name in ("place_and_release", "withdraw_from_drawer"):
            require(observed["target_inside_drawer"] and observed["target_drawer_floor_force_n"] > 0.01,
                    "Placed target lacks measured containment or floor support")
            require(not any(force > 0.01 for force in observed["finger_target_forces_n"].values()),
                    "Released target still contacts fingers")
            require(observed["drawer_position_m"] <= -0.155, "Placement/withdrawal unexpectedly closed the drawer")
            require(skill["storage"]["hand_clear_of_drawer"] is (name == "withdraw_from_drawer"),
                    "Placement and withdrawal must have distinct physical hand-clearance outcomes")
        elif name == "close_drawer":
            require(abs(observed["drawer_position_m"]) <= 0.003 and abs(observed["drawer_speed_m_s"]) < 0.005,
                    "Drawer did not physically close and settle")
            require(self.peak_handle_force > 0.1, "No measured finger-handle pushing force")
            require(skill["storage"]["hand_clear_of_drawer"], "Close did not withdraw from the drawer")
            require(observed["arm_home_error_rad"] > 0.03, "Close silently ran return_home")
        elif name == "return_home":
            require(observed["arm_home_error_rad"] <= 0.03 and observed["arm_max_speed_rad_s"] < 0.03,
                    "Home did not meet measured joint-position and velocity criteria")
        require(not state["warnings"] and state["collisions"]["episodes"] == 0,
                "Success retained warnings or unplanned collisions")

    def hold(self, invocation, duration):
        began, start_time = perf_counter(), float(self.sim.data.time)
        hold = {"requested_simulation_time_s": duration, "status": "running"}
        invocation["hold"] = hold
        try:
            anchor = physical_observation(self.sim)
            for _ in range(round(duration / STEP_S)):
                self.step(f"hold:{invocation['requested_skill']}")
                self.assert_success(invocation["requested_skill"])
                if invocation["requested_skill"] == "grasp_and_hold":
                    observed = physical_observation(self.sim)
                    require(math.dist(anchor["target_position_m"], observed["target_position_m"]) <= 0.01,
                            "Held target drifted more than 0.01 m while waiting for handoff")
            self.check_model(full=True)
            hold["status"] = "passed"
        except BaseException as exc:
            hold["status"] = "failed" if isinstance(exc, Exception) else "interrupted"
            hold["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            hold["wall_time_s"] = perf_counter() - began
            hold["simulation_time_s"] = float(self.sim.data.time) - start_time
            with diagnostic_capture(hold, "post_hold"):
                hold["post"] = boundary_snapshot(self.sim)

    def execute(self, name, parameters=None, hold_s=BOUNDARY_HOLD_S):
        invocation = self.invoke(name, parameters)
        self.complete(invocation)
        self.hold(invocation, hold_s)
        return invocation


def independent_close_home(run: EvidenceRun):
    for name in ("place_and_release", "withdraw_from_drawer"):
        run.invoke(name, reject=True)
    original = run.reference
    run.keep_target_on_table = True
    run.execute("close_drawer")
    closed = physical_observation(run.sim)
    require(closed["drawer_position_m"] - original["drawer_position_m"] > 0.15,
            "Close did not move the originally open drawer through its physical travel")
    run.invoke("close_drawer", reject=True)
    run.execute("return_home", hold_s=FINAL_HOLD_S)
    final = physical_observation(run.sim)
    require(math.dist(final["target_position_m"], original["target_position_m"]) < 0.003
            and final["target_table_force_n"] > 0.01,
            "Independent close/home moved the table-supported bowl")
    require(abs(final["drawer_position_m"] - closed["drawer_position_m"]) < 0.003,
            "Home changed the drawer position")


def five_skill_chain(run: EvidenceRun, parameters: GraspParameters):
    grasp = run.execute("grasp_and_hold", parameters)
    overlap = grasp["terminal"]["state"]["skill"]["descent_close_overlap_s"]
    if parameters.close_trigger_height_m == 0.0:
        require(overlap == 0.0, "Baseline unexpectedly overlapped descent and closure")
    else:
        require(overlap > 0.0, "Requested early closure never overlapped physical descent")
    for name in ("close_drawer", "return_home"):
        run.invoke(name, reject=True)
    run.execute("place_and_release")
    for name in ("close_drawer", "return_home"):
        run.invoke(name, reject=True)
    run.execute("withdraw_from_drawer")
    before_close = physical_observation(run.sim)
    run.execute("close_drawer")
    require(physical_observation(run.sim)["drawer_position_m"] - before_close["drawer_position_m"] > 0.15,
            "Closure lacked physical slider travel")
    run.execute("return_home", hold_s=FINAL_HOLD_S)
    final = physical_observation(run.sim)
    require(final["target_inside_drawer"] and final["target_drawer_floor_force_n"] > 0.01,
            "Bowl did not remain contained and supported after home/extra hold")
    require(abs(final["drawer_position_m"]) <= 0.003, "Final hold lost drawer closure")


def invalid_requests(run: EvidenceRun):
    for name in (None, "", "unknown", [], 1):
        run.invoke(name, reject=True)
    for name in SKILL_CHAIN[1:]:
        run.invoke(name, GraspParameters(), reject=True)
    run.invoke("grasp_and_hold", {"initial_opening_m": 0.08}, reject=True)
    run.invoke("place_and_release", reject=True)
    run.invoke("withdraw_from_drawer", reject=True)
    run.execute("return_home")
    require(physical_observation(run.sim)["drawer_position_m"] <= -0.155,
            "Independent home unexpectedly closed the drawer")
    invocation = run.invoke("grasp_and_hold", GraspParameters())
    run.step("advance_before_busy_requests")
    for name in SKILL_CHAIN:
        run.invoke(name, reject=True)
    run.complete(invocation)
    run.hold(invocation, BOUNDARY_HOLD_S)


def disturbed_grasp(run: EvidenceRun):
    run.execute("grasp_and_hold", GraspParameters())
    anchor = physical_observation(run.sim)
    target = run.sim.model.body("bowl_object").id
    disturbance = {
        "kind": "external_body_force", "body": "bowl_object",
        "force_xyz_n": [20.0, 0.0, 0.0], "maximum_duration_s": 0.50,
        "pre": boundary_snapshot(run.sim),
    }
    run.record["disturbance"] = disturbance
    start_time = float(run.sim.data.time)
    run.sim.data.xfrc_applied[target, 0] = 20.0
    try:
        for _ in range(25):
            state, observed = run.step("external_force:20N_world_x", clean=False)
            if state["skill"]["status"] == "failed":
                break
        require(state["skill"]["status"] == "failed", "Disturbance did not invalidate physical completion")
        require(not observed["bilateral_grip"]
                or math.dist(anchor["target_position_m"], observed["target_position_m"]) > 0.01,
                "No independently observed grip loss or slip accompanied failure")
        require(bool(state["skill"]["failure_reason"]), "Invalidated success has no failure reason")
        require(not all(state["skill"]["postconditions"].values()),
                "Failed grasp still reports every physical postcondition true")
    finally:
        disturbance["observed_duration_s"] = float(run.sim.data.time) - start_time
        try:
            with diagnostic_capture(disturbance, "post_force"):
                disturbance["post_force"] = boundary_snapshot(run.sim)
        finally:
            run.sim.data.xfrc_applied[target, 0] = 0.0
    for name in SKILL_CHAIN:
        run.invoke(name, reject=True)


SCENARIOS = (
    ("independent_close_home", "Close original empty drawer, then home; bowl stays on table", None),
    ("baseline_five_skill_chain", "Five separate calls with one-second boundary holds and final five-second hold", GraspParameters(0.08, 0.10, 0.0)),
    ("overlap_five_skill_chain", "Five separate calls with overlapping descent/closure and physical boundary holds", GraspParameters(0.08, 0.20, 0.004)),
    ("invalid_requests", "Malformed, parameter-incompatible, premature, and busy requests preserve state", None),
    ("disturbed_grasp", "A deliberate 20 N external force must revoke grasp success and block restart", None),
)


def run_scenario(record: dict, parameters: GraspParameters | None):
    began = perf_counter()
    sim = None
    start_time = None
    try:
        scene = SceneCase(**{**record["scene"], "offset_xy_m": tuple(record["scene"]["offset_xy_m"])})
        sim = Simulation(scene)
        start_time = float(sim.data.time)
        run = EvidenceRun(sim, record)
        run.settle()
        if record["name"] == "independent_close_home":
            independent_close_home(run)
        elif record["name"] in ("baseline_five_skill_chain", "overlap_five_skill_chain"):
            five_skill_chain(run, parameters)
        elif record["name"] == "invalid_requests":
            invalid_requests(run)
        elif record["name"] == "disturbed_grasp":
            disturbed_grasp(run)
        else:
            raise ValueError(f"Unrecognized evidence scenario: {record['name']}")
        run.check_model(full=True)
        record["status"] = "passed"
    except BaseException as exc:
        record["status"] = "failed" if isinstance(exc, Exception) else "interrupted"
        record["error"] = {"type": type(exc).__name__, "message": str(exc)}
        if not isinstance(exc, Exception):
            raise
    finally:
        record["wall_time_s"] = perf_counter() - began
        if sim is not None:
            record["simulation_time_s"] = float(sim.data.time) - start_time
            try:
                record["final"] = boundary_snapshot(sim)
            except Exception as exc:
                record["final_snapshot_error"] = {"type": type(exc).__name__, "message": str(exc)}
                if record["status"] != "interrupted":
                    record["status"] = "failed"
            finally:
                try:
                    sim.close()
                except Exception as exc:
                    record["close_error"] = {"type": type(exc).__name__, "message": str(exc)}
                    if record["status"] != "interrupted":
                        record["status"] = "failed"


def json_observations(value):
    """Preserve numerical failures as explicit JSON tags, never invented zeros."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_observation": str(value)}
    if isinstance(value, dict):
        return {key: json_observations(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_observations(item) for item in value]
    return value


def save_report(report: dict, output: Path | None):
    serialized = json.dumps(json_observations(report), separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output.parent) as directory:
            pending = Path(directory) / output.name
            pending.write_text(serialized, encoding="utf-8")
            pending.replace(output)
    return serialized


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Atomically save partial and final JSON evidence")
    args = parser.parse_args()
    scene = SceneCase("bowl.nominal", "bowl", "nominal", 0, (0.0, 0.0), 0.0)
    report = {
        "kind": "independent_real_physics_skill_verification",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mujoco_version": mujoco.__version__, "status": "running",
        "scope": "Deterministic nominal bowl workcell only; no GPT, resets, retries, or statistical success-rate claims.",
        "sampling_limit": "API boundaries are exact; contacts/model fields are observed every 0.02 simulated seconds, traces at 0.1 seconds and phase changes. Full compiled model hashes are boundary observations, not an audit inside mj_step.",
        "nonfinite_encoding": "Invalid numeric observations are retained as {nonfinite_observation: nan/inf/-inf}, not replaced with valid measurements.",
        "parameter_limits": GraspParameters.limits(),
        "trace_sample_cap_per_case": MAX_TRACE_SAMPLES,
        "cases": [
            {"name": name, "definition": definition, "scene": scene.to_dict(),
             "parameters": parameters.to_dict() if parameters is not None else None,
             "status": "not_run"}
            for name, definition, parameters in SCENARIOS
        ],
    }
    began = perf_counter()
    exit_code = 1
    try:
        save_report(report, args.output)
        for record, (_, _, parameters) in zip(report["cases"], SCENARIOS):
            record["status"] = "running"
            save_report(report, args.output)
            run_scenario(record, parameters)
            save_report(report, args.output)
            print(f"{record['name']}: {record['status']}", file=sys.stderr, flush=True)
        report["status"] = "passed" if all(case["status"] == "passed" for case in report["cases"]) else "failed"
        exit_code = 0 if report["status"] == "passed" else 1
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        exit_code = 130
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        report["wall_time_s"] = perf_counter() - began
        report["simulation_time_s"] = sum(case.get("simulation_time_s", 0.0) for case in report["cases"])
        try:
            serialized = save_report(report, args.output)
        except OSError as exc:
            report["status"] = "failed"
            report["report_write_error"] = {"type": type(exc).__name__, "message": str(exc)}
            exit_code = 1
            serialized = save_report(report, None)
        if args.output is not None:
            summary = {
                "status": report["status"],
                "output": str(args.output) if "report_write_error" not in report else None,
                "cases": {case["name"]: case["status"] for case in report["cases"]},
                "wall_time_s": report["wall_time_s"],
                "simulation_time_s": report["simulation_time_s"],
            }
            for key in ("error", "report_write_error"):
                if key in report:
                    summary[key] = report[key]
            print(json.dumps(summary, ensure_ascii=False))
        else:
            print(serialized, end="")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
