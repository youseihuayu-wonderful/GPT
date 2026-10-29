"""Fixed ground-truth color sorting using force-driven Panda grasping only.

The six-item queue is immutable; physical observations gate every handoff.
No drawer observer, language model, attachment, or execution-time model edits
are involved. Reset alone initializes poses, then settles them through MuJoCo.
"""

from dataclasses import dataclass
import math

import mujoco
import numpy as np

from collisions import CollisionTracker
from grasp import GraspParameters, GraspSkill
from simulation import ARM_HOME, ROOT, Simulation


TABLE_Z = 0.72
COLORS = {"red": (0.85, 0.12, 0.10, 1.0),
          "blue": (0.10, 0.32, 0.90, 1.0),
          "yellow": (0.98, 0.75, 0.06, 1.0)}


@dataclass(frozen=True)
class SortingObject:
    id: str
    color: str
    kind: str
    destination: str
    source_xy_m: tuple[float, float]
    slot_xy_m: tuple[float, float]

    @property
    def body_name(self):
        return f"{self.id}_object"

    @property
    def label(self):
        return f"{self.color} {self.kind}"

    @property
    def grasp_offset_m(self):
        return (0.0, -0.047, 0.042) if self.kind == "bowl" else (0.0, 0.0, 0.025)


OBJECTS = tuple(
    SortingObject(f"{color}_{kind}", color, kind, destination,
                  (x, -0.08 if kind == "bowl" else -0.28),
                  (x, 0.10 if kind == "bowl" else 0.29))
    for color, destination, x in zip(COLORS, "ABC", (0.30, 0.47, 0.64))
    for kind in ("bowl", "cup")
)
ZONES = tuple({"id": destination, "label": f"{destination} · {color}",
               "color": color, "center_xy_m": [x, 0.195],
               "half_size_xy_m": [0.075, 0.16],
               "slots_xy_m": [[x, 0.10], [x, 0.29]]}
              for color, destination, x in zip(COLORS, "ABC", (0.30, 0.47, 0.64)))


def _cup_spec(color):
    """Open wall ring with a compound-box approximation of its flat circular base.

    Three overlapping boxes cover the inner floor without protruding beyond the
    outer radius. A non-colliding cylinder supplies the circular base visual.
    The base has fixed 13.2 g mass; wall density gives about 69 g total mass.
    """
    rgba = " ".join(map(str, COLORS[color]))
    walls = []
    for index in range(24):
        angle = index * 2 * math.pi / 24
        walls.append(
            f'<geom name="wall{index}" type="box" '
            f'pos="{0.033 * math.cos(angle)} {0.033 * math.sin(angle)} 0.002" '
            f'quat="{math.cos(angle / 2)} 0 0 {math.sin(angle / 2)}" '
            'size="0.003 0.0045 0.033"/>')
    return mujoco.MjSpec.from_string(
        '<mujoco model="hollow cup"><default><geom density="650" '
        f'rgba="{rgba}" friction="0.95 0.3 0.1" '
        'solimp="0.998 0.998 0.001" solref="0.001 1" group="0"/>'
        '</default><worldbody><body name="root"><freejoint name="free"/>'
        '<body name="object">'
        '<geom name="bottom_center" type="box" pos="0 0 -0.0325" size="0.025 0.025 0.0025" mass="0.0044"/>'
        '<geom name="bottom_x" type="box" pos="0 0 -0.0325" size="0.030 0.018 0.0025" mass="0.0044"/>'
        '<geom name="bottom_y" type="box" pos="0 0 -0.0325" size="0.018 0.030 0.0025" mass="0.0044"/>'
        '<geom name="bottom_visual" type="cylinder" pos="0 0 -0.0325" size="0.036 0.0025" '
        'contype="0" conaffinity="0" group="2" mass="0"/>'
        + "".join(walls) + '</body></body></worldbody></mujoco>')


class SortingCollisions(CollisionTracker):
    """Unexpected robot/object and object/object contact, with explicit supports."""

    def __init__(self, sim):
        self.sim = sim
        self.model, self.data = sim.model, sim.data
        self.robot_bodies = frozenset(
            body for body in range(self.model.nbody)
            if (self.model.body(body).name or "").startswith("robot_"))
        self.body_names = tuple(self.model.body(i).name or f"body_{i}"
                                for i in range(self.model.nbody))
        self.force = np.zeros(6)
        self.reset()

    def update(self, dt, skill):
        forces = {}
        active_body = skill.target_body
        for index, contact in enumerate(self.data.contact):
            if contact.efc_address < 0:
                continue
            a, b = (int(self.model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2))
            relevant = (a in self.robot_bodies or b in self.robot_bodies
                        or a in self.sim.object_bodies or b in self.sim.object_bodies)
            if not relevant:
                continue
            planned = False
            if self.sim.table in (a, b):
                other = b if a == self.sim.table else a
                if other in self.sim.object_bodies:
                    planned = other != active_body or not skill.cleared_table
                    if other == active_body and skill.phase in ("place", "release", "withdraw"):
                        planned = skill.placement_aligned()
                    if skill.phase in ("return_home", "verify_sorting", "done"):
                        planned = True
            if active_body in (a, b):
                other = b if a == active_body else a
                if other in skill.fingers and skill.phase in (
                        "descend", "close", "lift", "hold", "transfer", "place", "release"):
                    planned = True
            if planned:
                continue
            mujoco.mj_contactForce(self.model, self.data, index, self.force)
            pair = tuple(sorted((a, b), key=self.body_names.__getitem__))
            forces[pair] = forces.get(pair, 0.0) + max(0.0, float(self.force[0]))
        current = {pair for pair, force in forces.items() if force > self.FORCE_THRESHOLD_N}
        if current:
            self.duration_s += dt
        for pair in current:
            record = self.records.setdefault(pair, {
                "body_a": self.body_names[pair[0]], "body_b": self.body_names[pair[1]],
                "episodes": 0, "duration_s": 0.0, "peak_force_n": 0.0})
            record["episodes"] += int(pair not in self.active_pairs)
            record["duration_s"] += dt
            record["peak_force_n"] = max(record["peak_force_n"], forces[pair])
        self.active_pairs = current


class SortingSkill(GraspSkill):
    """Reuse the existing IK/quintic/force servo, not its storage state machine."""

    REQUIRED_VALIDATION = 5.0
    MESSAGES = {
        "idle": "Ready for the fixed red→A, blue→B, yellow→C queue",
        "prepare": "Checking stable source support and opening the gripper",
        "approach": "Approaching the selected bowl/cup from above",
        "descend": "Descending to the measured grasp pose",
        "close": "Establishing bilateral force-bearing finger contact",
        "lift": "Lifting with real finger contacts",
        "hold": "Verifying lift height and one-second stable grip",
        "transfer": "Transferring above the reserved destination slot",
        "place": "Descending until the table supports the object",
        "release": "Opening fingers and verifying stable contained support",
        "withdraw": "Withdrawing vertically from the released object",
        "return_home": "Returning the empty hand to its initial joint pose",
        "verify_sorting": "Checking the entire batch continuously for five simulated seconds",
        "done": "All six objects physically sorted; home and empty hand verified",
        "failed": "Execution stopped; explicit reset required",
    }

    def __init__(self, sim):
        self.sim, self.model, self.data = sim, sim.model, sim.data
        self.tcp = self.model.site("robot_grasp_center").id
        self.fingers = {self.model.body(f"robot_{side}_finger").id for side in ("left", "right")}
        self.arm_dof = np.array([self.model.joint(f"robot_joint{i}").dofadr[0] for i in range(1, 8)])
        self.finger_dof = np.array([self.model.joint(f"robot_finger_joint{i}").dofadr[0] for i in (1, 2)])
        self.joint_limits = np.array([self.model.joint(f"robot_joint{i}").range for i in range(1, 8)])
        self.work = mujoco.MjData(self.model)
        self.jacp, self.jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        self.reset()

    @property
    def target_body(self):
        return None if self.current_index is None else self.sim.body_ids[self.current_index]

    def reset(self):
        self.status = self.phase = "idle"
        self.current_index = None
        self.started_at = self.phase_started_at = float(self.data.time)
        self.finished_at = self.failure_reason = None
        self.phase_times = {}
        self.results = []
        self.plan_status = ["pending"] * len(OBJECTS)
        self.arm_target = self.data.qpos[self.sim.arm_qpos].copy()
        self.gripper_closed = False
        self.opening_target = 0.08
        self.orientation_target = np.array([0.0, 1.0, 0.0, 0.0])
        self.parameters = GraspParameters()
        self.stable_elapsed = self.hold_elapsed = self.validation_hold_s = 0.0
        self.lost_contact_elapsed = 0.0
        self.cleared_table = False
        self.reference_z = self.hold_relative = self.grasp_position = None
        self.motion_start = self.motion_target = self.target_position = None
        self.motion_duration = 0.0
        self.last_diagnostics = {}

    def _enter(self, phase):
        now = float(self.data.time)
        if self.phase not in ("idle", "done", "failed"):
            key = f"{self.current_index}:{self.phase}"
            self.phase_times[key] = self.phase_times.get(key, 0.0) + now - self.phase_started_at
        self.phase, self.phase_started_at = phase, now
        self.stable_elapsed = 0.0

    def fail(self, reason):
        if self.status == "failed":
            return
        previous_phase = self.phase
        self._enter("failed")
        self.status, self.failure_reason = "failed", str(reason)
        self.finished_at = float(self.data.time)
        self.validation_hold_s = 0.0
        self.arm_target = self.data.qpos[self.sim.arm_qpos].copy()
        active_item = (self.current_index is not None
                       and self.plan_status[self.current_index] == "active")
        if active_item:
            self.plan_status[self.current_index] = "failed"
        self.results.append({"index": self.current_index if active_item else None,
                             "status": "failed", "phase": previous_phase,
                             "time_s": self.finished_at, "reason": str(reason),
                             "diagnostics": dict(self.last_diagnostics)})

    def precondition_error(self):
        if self.status != "idle":
            return "A sorting run was already started; reset before another run"
        error = self.sim.numerical_error()
        if error:
            return error
        facts = self.sim.object_checks()
        if not all(obj["supported"] and obj["stable"] and obj["detached"] for obj in facts):
            return "All six source objects must be stably table-supported and detached"
        if not self.sim.hand_empty():
            return "Sorting requires an empty hand"
        if not self.sim.home_reached():
            return "Sorting requires the robot at home"
        for index in range(len(OBJECTS)):
            error = self.sim.slot_error(index)
            if error:
                return error
        return None

    def start(self):
        error = self.precondition_error()
        if error:
            raise ValueError(error)
        # Acceptance mutates controller bookkeeping only, never physics state.
        self.started_at = float(self.data.time)
        self.status = "running"
        self.current_index = 0
        self._begin_item()

    def _begin_item(self):
        self.plan_status[self.current_index] = "active"
        self.gripper_closed = False
        self.opening_target = 0.08
        self.cleared_table = False
        self.hold_elapsed = self.lost_contact_elapsed = 0.0
        self._enter("prepare")

    def placement_aligned(self):
        if self.current_index is None:
            return False
        return bool(np.linalg.norm(self.data.xpos[self.target_body, :2]
                                   - OBJECTS[self.current_index].slot_xy_m) < 0.012)

    def _grip_facts(self):
        contact = self.sim.contact_facts()[self.target_body]
        position = self.data.xpos[self.target_body]
        lower, _ = self.sim.object_bounds(self.current_index)
        if lower[2] > TABLE_Z + 0.003:
            self.cleared_table = True
        return contact, position, lower

    def verify_terminal(self):
        if self.status != "succeeded" or all(self.sim.goal_checks().values()):
            return
        for index, fact in enumerate(self.sim.object_checks()):
            if not fact["sorted"]:
                self.plan_status[index] = "failed"
        self.fail("Physical sorting postconditions lost after completion")

    def update(self):
        if self.status == "succeeded":
            self.verify_terminal()
            return
        if self.status != "running":
            return
        dt = float(self.model.opt.timestep)
        elapsed = float(self.data.time) - self.phase_started_at
        if self.data.time - self.started_at > 240.0:
            self.fail("Fixed sorting execution exceeded 240 simulated seconds")
            return
        facts = self.sim.object_checks()
        for index, fact in enumerate(facts):
            if self.plan_status[index] == "sorted" and not fact["sorted"]:
                self.plan_status[index] = "failed"
                self.fail(f"Previously sorted object disturbed: {OBJECTS[index].id}")
                return
        if self.phase in ("return_home", "verify_sorting"):
            self._finish_batch(elapsed, dt)
            return
        contact, position, lower = self._grip_facts()
        bilateral = contact["bilateral"]
        relative = position - self.data.site_xpos[self.tcp]
        self.last_diagnostics = {
            "object_id": OBJECTS[self.current_index].id,
            "bilateral_grip": bilateral, "finger_forces_n": contact["finger_forces_n"],
            "support_force_n": contact["support_force_n"],
            "lowest_collision_z_m": float(lower[2]),
            "lift_m": None if self.reference_z is None else float(position[2] - self.reference_z),
            "tcp_position_m": self.data.site_xpos[self.tcp].tolist(),
            "motion_target_m": None if self.target_position is None else self.target_position.tolist(),
            "motion_duration_s": self.motion_duration,
            "phase_elapsed_s": elapsed,
            "relative_drift_m": None if self.hold_relative is None else float(np.linalg.norm(relative - self.hold_relative)),
        }
        if self.phase in ("lift", "hold", "transfer", "place"):
            self.lost_contact_elapsed = 0.0 if bilateral else self.lost_contact_elapsed + dt
            if self.lost_contact_elapsed > 0.15:
                self.fail("Lost bilateral grip during lift/transfer/placement")
                return
        if self.phase in ("hold", "transfer") and self.hold_relative is not None:
            if np.linalg.norm(relative - self.hold_relative) >= 0.01:
                self.fail("Object slipped relative to the gripper")
                return
        if self.phase == "prepare":
            fact = facts[self.current_index]
            ready = fact["supported"] and fact["stable"] and abs(
                float(self.data.qpos[self.sim.finger_qpos].sum()) - 0.08) < 0.002
            self.stable_elapsed = self.stable_elapsed + dt if ready else 0.0
            if self.stable_elapsed >= 0.06:
                self.reference_z = float(position[2])
                rotation = self.data.xmat[self.target_body].reshape(3, 3)
                self.grasp_position = position + rotation @ OBJECTS[self.current_index].grasp_offset_m
                self._move("approach", self.grasp_position + [0, 0, 0.18])
            elif elapsed > 2.0:
                self.fail("Source did not settle or gripper did not open")
        elif self.phase in ("approach", "descend", "lift", "transfer", "withdraw"):
            phase = self.phase
            if not self._motion_done(elapsed) or self.status != "running":
                return
            if phase == "approach":
                self._move("descend", self.grasp_position)
            elif phase == "descend":
                self.gripper_closed = True
                self._enter("close")
            elif phase == "lift":
                self.hold_relative = relative.copy()
                self._enter("hold")
            elif phase == "transfer":
                error = self.sim.slot_error(self.current_index, exclude_current=True)
                if error:
                    self.fail(error)
                    return
                target = self.data.site_xpos[self.tcp].copy()
                target[2] -= lower[2] - TABLE_Z + 0.0005
                self._move("place", target)
            else:
                self._complete_item(facts[self.current_index])
        elif self.phase == "close":
            self.stable_elapsed = self.stable_elapsed + dt if bilateral else 0.0
            if self.stable_elapsed >= 0.10:
                self._move("lift", self.grasp_position + [0, 0, 0.18])
            elif elapsed > 2.0:
                self.fail("Gripper failed to establish stable bilateral contacts")
        elif self.phase == "hold":
            valid = (bilateral and contact["support_force_n"] <= 0.01
                     and lower[2] > TABLE_Z + 0.08
                     and position[2] - self.reference_z >= self.TARGET_LIFT)
            if not valid:
                self.fail("Lift/hold failed: no bilateral grip, clearance, or required lift height")
                return
            self.hold_elapsed += dt
            if self.hold_elapsed + 1e-9 >= self.REQUIRED_HOLD:
                target = self.data.site_xpos[self.tcp].copy()
                target[:2] += np.array(OBJECTS[self.current_index].slot_xy_m) - position[:2]
                self._move("transfer", target)
        elif self.phase == "place":
            error = self.sim.slot_error(self.current_index, exclude_current=True)
            if error:
                self.fail(error)
                return
            supported = contact["support_force_n"] > 0.01
            if supported:
                if not self.placement_aligned() or not facts[self.current_index]["correct_region"]:
                    self.fail("Table contact outside the assigned destination slot/region")
                    return
                # Stop lowering on actual support, not on a trajectory deadline.
                self.arm_target = self.data.qpos[self.sim.arm_qpos].copy()
                self.stable_elapsed += dt
                if self.stable_elapsed >= 0.10:
                    self.gripper_closed = False
                    self._enter("release")
            else:
                self.stable_elapsed = 0.0
                self._motion_done(elapsed)
                if self.status == "running" and elapsed > self.motion_duration + 1.0:
                    self.fail("Placement trajectory ended without positive table support")
        elif self.phase == "release":
            fact = facts[self.current_index]
            ready = all(fact[key] for key in ("correct_region", "supported", "stable", "detached"))
            self.stable_elapsed = self.stable_elapsed + dt if ready else 0.0
            if self.stable_elapsed >= 0.30:
                target = self.data.site_xpos[self.tcp].copy() + [0, 0, 0.18]
                self._move("withdraw", target)
            elif elapsed > 3.0:
                self.fail("Released object did not settle detached inside its assigned region")

    def _complete_item(self, fact):
        if not all(fact[key] for key in ("correct_region", "supported", "stable", "detached")):
            self.fail("Object lost valid placement while the hand withdrew")
            return
        self.plan_status[self.current_index] = "sorted"
        self.results.append({"index": self.current_index, "object_id": OBJECTS[self.current_index].id,
                             "status": "sorted", "time_s": float(self.data.time),
                             "position_m": fact["position_m"], "hold_s": self.hold_elapsed})
        if self.current_index + 1 < len(OBJECTS):
            self.current_index += 1
            self._begin_item()
        else:
            self.motion_start = self.data.qpos[self.sim.arm_qpos].copy()
            self.motion_target = self.sim.arm_home.copy()
            distance = float(np.max(np.abs(self.motion_target - self.motion_start)))
            self.motion_duration = max(1.0, 1.875 * distance / 0.6)
            self._enter("return_home")

    def _finish_batch(self, elapsed, dt):
        if self.phase == "return_home":
            u = min(1.0, elapsed / self.motion_duration)
            blend = u**3 * (10 - 15 * u + 6 * u**2)
            self.arm_target = self.motion_start + blend * (self.motion_target - self.motion_start)
            if elapsed >= self.motion_duration and self.sim.home_reached():
                self._enter("verify_sorting")
            elif elapsed > self.motion_duration + 3.0:
                self.fail("Robot did not return home within the servo deadline")
            return
        if not all(self.sim.goal_checks().values()):
            self.fail("Full-batch physical postconditions failed during final verification")
            return
        self.validation_hold_s += dt
        if self.validation_hold_s + 1e-9 >= self.REQUIRED_VALIDATION:
            self._enter("done")
            self.status = "succeeded"
            self.finished_at = float(self.data.time)

    def state(self):
        self.verify_terminal()
        objects = self.sim.object_checks()
        error = self.precondition_error()
        end = float(self.data.time) if self.finished_at is None else self.finished_at
        return {
            "status": self.status, "mode": "fixed", "can_start": error is None,
            "start_error": error, "current_index": self.current_index, "phase": self.phase,
            "message": self.failure_reason or self.MESSAGES[self.phase],
            "failure_reason": self.failure_reason,
            "elapsed_s": 0.0 if self.status == "idle" else end - self.started_at,
            "sorted_count": sum(obj["sorted"] for obj in objects), "object_count": len(OBJECTS),
            "validation_hold_s": self.validation_hold_s, "goal_checks": self.sim.goal_checks(),
            "plan": [{"index": i, "object_id": obj.id, "label": obj.label, "color": obj.color,
                      "kind": obj.kind, "destination": obj.destination,
                      "slot_xy_m": list(obj.slot_xy_m), "status": self.plan_status[i]}
                     for i, obj in enumerate(OBJECTS)],
            "objects": objects,
            "zones": [{**zone, "center_xy_m": list(zone["center_xy_m"]),
                       "half_size_xy_m": list(zone["half_size_xy_m"]),
                       "slots_xy_m": [list(slot) for slot in zone["slots_xy_m"]]} for zone in ZONES],
            "phase_times_s": dict(self.phase_times), "item_results": list(self.results),
            "diagnostics": dict(self.last_diagnostics), "gpt_called": False,
        }


class SortingSimulation(Simulation):
    """Independent sorting scene, sharing only general rendering/numerical helpers."""

    def __init__(self):
        scene = mujoco.MjSpec.from_file(str(ROOT / "sorting_scene.xml"))
        robot = mujoco.MjSpec.from_file(str(ROOT / "vendor/panda/panda.xml"))
        for key in list(robot.keys):
            robot.delete(key)
        for light in list(robot.lights):
            robot.delete(light)
        robot.body("hand").add_site(name="grasp_center", pos=[0, 0, 0.1034],
                                    size=[0.003] * 3, rgba=[0, 0, 0, 0])
        gripper = robot.actuator("actuator8")
        gripper.gainprm[:] = 0
        gripper.gainprm[0] = 1
        gripper.biasprm[:] = 0
        gripper.ctrlrange[:] = [-100, 100]
        scene.attach(robot, frame=scene.frame("robot_mount"), prefix="robot_")
        for zone in ZONES:
            rgba = list(COLORS[zone["color"]])
            rgba[3] = 0.28
            scene.worldbody.add_geom(name=f"zone_{zone['id']}", type=mujoco.mjtGeom.mjGEOM_BOX,
                                     pos=[*zone["center_xy_m"], TABLE_Z + 0.0002],
                                     size=[*zone["half_size_xy_m"], 0.0001],
                                     rgba=rgba, contype=0, conaffinity=0, group=2)
            for index, slot in enumerate(zone["slots_xy_m"]):
                scene.worldbody.add_site(name=f"slot_{zone['id']}_{index}",
                                         type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                                         pos=[*slot, TABLE_Z + 0.0004], size=[0.008, 0.0001, 0.0001],
                                         rgba=COLORS[zone["color"]])
        for obj in OBJECTS:
            if obj.kind == "bowl":
                part = mujoco.MjSpec.from_file(str(
                    ROOT / "vendor/libero/stable_scanned_objects/akita_black_bowl/akita_black_bowl.xml"))
                root = part.worldbody.bodies[0]
                root.name = "root"
                root.add_freejoint(name="free")
                part.body("object").pos[:] = 0
                for geom in part.geoms:
                    if geom.group == 1:
                        geom.material = ""
                        geom.rgba[:] = COLORS[obj.color]
                height = 0.73
            else:
                part = _cup_spec(obj.color)
                height = TABLE_Z + 0.04
            frame = scene.worldbody.add_frame(name=f"{obj.id}_mount", pos=[*obj.source_xy_m, height])
            scene.attach(part, frame=frame, prefix=f"{obj.id}_")
        self.model = scene.compile()
        self.data = mujoco.MjData(self.model)
        self.arm_qpos = np.array([self.model.joint(f"robot_joint{i}").qposadr[0] for i in range(1, 8)])
        self.arm_actuators = np.array([self.model.actuator(f"robot_actuator{i}").id for i in range(1, 8)])
        self.finger_qpos = np.array([self.model.joint(f"robot_finger_joint{i}").qposadr[0] for i in (1, 2)])
        self.gripper_actuator = self.model.actuator("robot_actuator8").id
        self.arm_home = ARM_HOME.copy()
        self.object_names = tuple(obj.body_name for obj in OBJECTS)
        self.body_ids = tuple(self.model.body(name).id for name in self.object_names)
        self.object_bodies = frozenset(self.body_ids)
        self.object_dofs = tuple(int(self.model.joint(f"{obj.id}_free").dofadr[0]) for obj in OBJECTS)
        collidable = (self.model.geom_contype | self.model.geom_conaffinity) != 0
        self.object_geoms = tuple(np.flatnonzero((self.model.geom_bodyid == body) & collidable)
                                  for body in self.body_ids)
        self.table = self.model.body("table").id
        self.tabletop = self.model.geom("tabletop").id
        self.renderer = None
        self.camera, self.options = mujoco.MjvCamera(), mujoco.MjvOption()
        self.options.geomgroup[0] = 0
        self.options.geomgroup[3] = 0
        # Cups use the actual visible collision solids; bowls retain their mesh.
        for obj in OBJECTS:
            if obj.kind == "cup":
                for geom in self.object_geoms[OBJECTS.index(obj)]:
                    self.model.geom_group[geom] = 2
        self._force = np.zeros(6)
        self._numerical_failure = None
        self.skill = SortingSkill(self)
        self.collisions = SortingCollisions(self)
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.arm_qpos] = self.arm_home
        self.data.qpos[self.finger_qpos] = 0.04
        self.data.ctrl[self.arm_actuators] = self.arm_home
        self._numerical_failure = None
        mujoco.mj_forward(self.model, self.data)
        self.skill.reset()
        self.collisions.reset()
        self.scene_revision = getattr(self, "scene_revision", 0) + 1
        # A reproducible physically settled starting scene, not manually zeroed velocities.
        self.step(0.6)
        self.skill.reset()
        self.collisions.reset()

    def start_sorting(self):
        self.skill.start()

    def step(self, duration_s=0.02):
        if not math.isfinite(duration_s):
            raise ValueError("Duration must be finite")
        steps = round(duration_s / self.model.opt.timestep)
        if steps < 1 or not np.isclose(steps * self.model.opt.timestep, duration_s):
            raise ValueError("Duration must be a positive multiple of the physics timestep")
        self._check_numerical_state()
        for _ in range(steps):
            self.skill.apply_control()
            self._check_numerical_state()
            mujoco.mj_step(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            self._check_numerical_state()
            if self.skill.status in ("running", "succeeded"):
                self.collisions.update(float(self.model.opt.timestep), self.skill)
                if self.collisions.active_pairs:
                    names = [f"{self.collisions.body_names[a]} / {self.collisions.body_names[b]}"
                             for a, b in self.collisions.active_pairs]
                    self.skill.fail("Unexpected collision: " + ", ".join(sorted(names)))
            self.skill.update()

    def object_bounds(self, index):
        """Exact world AABB for the scene's boxes and circular cylinder bottoms."""
        geoms = self.object_geoms[index]
        rotations = self.data.geom_xmat[geoms].reshape(-1, 3, 3)
        sizes = self.model.geom_size[geoms]
        kinds = self.model.geom_type[geoms]
        cylinders = kinds == mujoco.mjtGeom.mjGEOM_CYLINDER
        if np.any((kinds != mujoco.mjtGeom.mjGEOM_BOX) & ~cylinders):
            raise ValueError("Sorting bounds require box or cylinder collision geometry")
        extents = np.einsum("ijk,ik->ij", np.abs(rotations), sizes)
        if np.any(cylinders):
            axes = rotations[cylinders, :, 2]
            radii = sizes[cylinders, 0, None]
            heights = sizes[cylinders, 1, None]
            extents[cylinders] = radii * np.sqrt(np.maximum(0.0, 1 - axes**2)) + heights * np.abs(axes)
        positions = self.data.geom_xpos[geoms]
        return np.min(positions - extents, axis=0), np.max(positions + extents, axis=0)

    def contact_facts(self):
        result = {body: {"support_force_n": 0.0,
                         "finger_forces_n": {self.model.body(f).name: 0.0 for f in self.skill.fingers},
                         "finger_contact": False} for body in self.body_ids}
        for index, contact in enumerate(self.data.contact):
            a, b = (int(self.model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2))
            for body, other in ((a, b), (b, a)):
                if body not in result:
                    continue
                fact = result[body]
                if other in self.skill.fingers:
                    # A geometric finger contact is not detached even at zero force.
                    fact["finger_contact"] = True
                if contact.efc_address < 0:
                    continue
                mujoco.mj_contactForce(self.model, self.data, index, self._force)
                force = max(0.0, float(self._force[0]))
                if other in self.skill.fingers:
                    fact["finger_forces_n"][self.model.body(other).name] += force
                if other == self.table and self.tabletop in (contact.geom1, contact.geom2):
                    # Require upward normal load on the object, not a side touch.
                    normal_z = float(contact.frame[2]) * (1.0 if body == b else -1.0)
                    fact["support_force_n"] += force * max(0.0, normal_z)
        for fact in result.values():
            fact["bilateral"] = all(force > 0.01 for force in fact["finger_forces_n"].values())
        return result

    def object_checks(self):
        contacts = self.contact_facts()
        result = []
        for index, obj in enumerate(OBJECTS):
            body, dof = self.body_ids[index], self.object_dofs[index]
            lower, upper = self.object_bounds(index)
            zone = ZONES["ABC".index(obj.destination)]
            center, half = np.array(zone["center_xy_m"]), np.array(zone["half_size_xy_m"])
            contained = bool(np.all(lower[:2] >= center - half) and np.all(upper[:2] <= center + half))
            # Slots are distinct within a color zone, not interchangeable stacks.
            slot_aligned = bool(np.linalg.norm(self.data.xpos[body, :2] - obj.slot_xy_m) < 0.025)
            upright = bool(self.data.xmat[body].reshape(3, 3)[2, 2] > 0.95)
            supported = bool(contacts[body]["support_force_n"] > 0.01
                             and TABLE_Z - 0.003 <= lower[2] <= TABLE_Z + 0.003)
            velocity = self.data.qvel[dof:dof + 6]
            stable = bool(np.linalg.norm(velocity[:3]) < 0.01 and np.linalg.norm(velocity[3:]) < 0.1)
            detached = not contacts[body]["finger_contact"]
            correct = contained and slot_aligned and upright
            result.append({"id": obj.id, "body_name": obj.body_name, "label": obj.label,
                           "kind": obj.kind, "color": obj.color,
                           "position_m": self.data.xpos[body].tolist(),
                           "correct_region": correct, "supported": supported, "stable": stable,
                           "detached": detached,
                           "sorted": self.skill.plan_status[index] == "sorted" and correct and supported and stable and detached,
                           "contained": contained, "slot_aligned": slot_aligned, "upright": upright,
                           "bounds_min_m": lower.tolist(), "bounds_max_m": upper.tolist(),
                           "support_force_n": contacts[body]["support_force_n"],
                           "finger_forces_n": contacts[body]["finger_forces_n"],
                           "linear_speed_m_s": float(np.linalg.norm(velocity[:3])),
                           "angular_speed_rad_s": float(np.linalg.norm(velocity[3:]))})
        return result

    def slot_error(self, index, *, exclude_current=False):
        slot = np.array(OBJECTS[index].slot_xy_m)
        radius = 0.057 if OBJECTS[index].kind == "bowl" else 0.039
        for other, obj in enumerate(OBJECTS):
            if exclude_current and other == index:
                continue
            lower, upper = self.object_bounds(other)
            if np.all(upper[:2] > slot - radius) and np.all(lower[:2] < slot + radius):
                return f"Reserved slot for {OBJECTS[index].id} is occupied by {obj.id}"
        return None

    def hand_empty(self):
        robot_bodies = self.collisions.robot_bodies
        for contact in self.data.contact:
            a, b = (int(self.model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2))
            if ((a in self.object_bodies and b in robot_bodies)
                    or (b in self.object_bodies and a in robot_bodies)):
                return False
        return True

    def home_reached(self):
        return bool(np.max(np.abs(self.data.qpos[self.arm_qpos] - self.arm_home)) < 0.025
                    and np.max(np.abs(self.data.qvel[self.skill.arm_dof])) < 0.05)

    def goal_checks(self):
        objects = self.object_checks()
        complete = len(objects) == 6 and all(status == "sorted" for status in self.skill.plan_status)
        return {"all_correct": complete and all(obj["correct_region"] for obj in objects),
                "all_supported": bool(objects) and all(obj["supported"] for obj in objects),
                "all_stable": bool(objects) and all(obj["stable"] for obj in objects),
                "hand_empty": self.hand_empty(), "home_reached": self.home_reached()}

    def state(self):
        error = self.numerical_error()
        if error:
            self._numerical_failure = error
            self.skill.fail(error)
        sorting = self.skill.state()
        return {"time_s": float(self.data.time), "timestep_s": float(self.model.opt.timestep),
                "scene_revision": self.scene_revision,
                "scene_name": "Panda fixed color sorting · six bowls/cups",
                "robot_name": "Franka Panda · 7 joints · bilateral gripper",
                "source": "Menagerie Panda + recolored LIBERO bowls + hollow physical cups; ground-truth fixed planning; no vision/GPT",
                "objects": [self.get_object_pose(name) for name in self.object_names],
                "joint_positions_rad": self.data.qpos[self.arm_qpos].tolist(),
                "gripper_opening_m": float(self.data.qpos[self.finger_qpos].sum()),
                "warnings": self.warning_names(), "contacts": int(self.data.ncon),
                "collisions": self.collisions.state(), "sorting": sorting}
