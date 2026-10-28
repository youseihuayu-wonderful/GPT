"""Feedback-checked target grasp with bounded motion parameters and real contacts."""

from dataclasses import asdict, dataclass, field, fields
import math
from numbers import Real

import mujoco
import numpy as np

from task_specs import TASKS


@dataclass(frozen=True)
class GraspParameters:
    """SI-unit settings; inclusive bounds live in each field's metadata."""

    initial_opening_m: float = field(default=0.08, metadata={"min": TASKS["grasp"].minimum_opening_m, "max": 0.08})
    descent_speed_m_s: float = field(default=0.10, metadata={"min": 0.04, "max": 0.20})
    close_trigger_height_m: float = field(default=0.0, metadata={"min": 0.0, "max": 0.02})

    def __post_init__(self) -> None:
        for setting in fields(self):
            value = getattr(self, setting.name)
            lower, upper = setting.metadata["min"], setting.metadata["max"]
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not lower <= value <= upper or not math.isfinite(value)):
                raise ValueError(f"{setting.name} must be a finite number in [{lower}, {upper}]")
            object.__setattr__(self, setting.name, float(value))

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def limits(cls) -> dict[str, dict[str, float]]:
        return {setting.name: dict(setting.metadata) for setting in fields(cls)}


class GraspSkill:
    TARGET_LIFT = 0.10
    REQUIRED_HOLD = 1.0
    MESSAGES = {
        "idle": "等待开始；由本地物理控制器执行抓取",
        "prepare": "设置夹爪初始开度，确认{target}已在桌面稳定",
        "approach": "移动到{target}上方",
        "descend": "下降到抓取位置；按实际高度触发闭合",
        "close": "闭合夹爪，确认双侧真实接触",
        "lift": "保持夹紧并向上抬起",
        "hold": "检查离桌高度、双侧接触和持续保持",
        "transfer": "保持夹持，转腕并运送到全开抽屉上方",
        "place": "按碗的真实碰撞边界下降，确认抽屉底板支撑",
        "release": "张开夹爪，确认碗已稳定、完整入屉且脱离手指",
        "withdraw": "垂直撤出夹爪，避开碗和抽屉壁",
        "push_approach": "收拢夹爪并从抽屉把手前方接近",
        "push_drawer": "通过手指与把手的真实接触缓慢推关抽屉",
        "push_withdraw": "后撤离开把手，再抬升至安全高度",
        "return_home": "伺服机械臂返回初始关节位姿",
        "verify_store": "持续检查碗支撑、脱离夹爪、抽屉关闭与机械臂归位",
        "done": "{target}已抬起并稳定保持；重置后可再次执行",
        "failed": "抓取失败；重置后可再次执行",
    }

    def __init__(self, sim) -> None:
        self.sim = sim
        self.model = sim.model
        self.data = sim.data
        self.tcp = self.model.site("robot_grasp_center").id
        self.target = sim.target
        self.target_body = self.model.body(f"{self.target.name}_object").id
        self.fingers = {
            self.model.body("robot_left_finger").id,
            self.model.body("robot_right_finger").id,
        }
        self.table = self.model.body("table").id
        self.target_dof = self.model.joint(f"{self.target.name}_free").dofadr[0]
        self.arm_dof = np.array([
            self.model.joint(f"robot_joint{i}").dofadr[0] for i in range(1, 8)
        ])
        self.joint_limits = np.array([
            self.model.joint(f"robot_joint{i}").range for i in range(1, 8)
        ])
        self.work = mujoco.MjData(self.model)
        self.jacp = np.zeros((3, self.model.nv))
        self.jacr = np.zeros((3, self.model.nv))
        self.contact_force = np.zeros(6)
        self.tcp_velocity = np.zeros(6)
        self.finger_dof = np.array([
            self.model.joint(f"robot_finger_joint{i}").dofadr[0] for i in (1, 2)
        ])
        self.reset()

    def reset(self) -> None:
        self.status = "idle"
        self.phase = "idle"
        self.started_at = float(self.data.time)
        self.phase_started_at = self.started_at
        self.finished_at = None
        self.failure_reason = None
        self.phase_times = {}
        self.reference_z = None
        self.hold_elapsed = 0.0
        self.stable_elapsed = 0.0
        self.lost_contact_elapsed = 0.0
        self.hold_relative = None
        self.motion_start = None
        self.motion_target = None
        self.motion_duration = 0.0
        self.target_position = None
        self.grasp_position = None
        self.arm_target = self.data.ctrl[self.sim.arm_actuators].copy()
        self.gripper_closed = False
        self.parameters = GraspParameters()
        self.close_started_s = None
        self.descent_completed_s = None
        self.descent_close_overlap_s = 0.0
        self.task_kind = "grasp"
        self.storage = None
        self.opening_target = self.parameters.initial_opening_m
        self.orientation_target = np.array([0.0, 1.0, 0.0, 0.0])

    def start(self, task_kind: str, parameters: GraspParameters | None = None) -> None:
        if self.status != "idle":
            raise ValueError("请先重置场景，再开始新的抓取")
        if parameters is not None and not isinstance(parameters, GraspParameters):
            raise ValueError("parameters must be GraspParameters or None")
        if not isinstance(task_kind, str) or task_kind not in TASKS:
            raise ValueError("Unsupported task kind")
        if self.target.name not in TASKS[task_kind].target_objects:
            raise ValueError("Task does not support the selected target")
        storage = None
        if task_kind == "store_bowl":
            from storage import BowlStorage
            storage = BowlStorage(self)
            storage.validate(parameters or GraspParameters())
        self.parameters = parameters if parameters is not None else GraspParameters()
        self.task_kind = task_kind
        self.storage = storage
        self.opening_target = self.parameters.initial_opening_m
        self.status = "running"
        self.started_at = float(self.data.time)
        self.arm_target = self.data.qpos[self.sim.arm_qpos].copy()
        self._enter("prepare")
        self.gripper_closed = False

    def _enter(self, phase: str) -> None:
        now = float(self.data.time)
        if self.phase not in ("idle", "done", "failed"):
            self.phase_times[self.phase] = now - self.phase_started_at
        self.phase = phase
        self.phase_started_at = now
        self.stable_elapsed = 0.0

    def fail(self, reason: str) -> None:
        self._enter("failed")
        self.status = "failed"
        self.failure_reason = reason
        self.finished_at = float(self.data.time)
        # Stop progressing the trajectory; hold the arm where it actually is.
        self.arm_target = self.data.qpos[self.sim.arm_qpos].copy()

    def apply_control(self) -> None:
        # Bias-force feedforward offsets gravity-induced position-servo sag.
        # Commands still pass through the original actuator force/control limits.
        actuators = self.sim.arm_actuators
        compensation = self.data.qfrc_bias[self.arm_dof] / self.model.actuator_gainprm[actuators, 0]
        self.data.ctrl[actuators] = np.clip(
            self.arm_target + compensation,
            self.model.actuator_ctrlrange[actuators, 0],
            self.model.actuator_ctrlrange[actuators, 1],
        )
        # The tendon splits force equally between the two fingers. Use an
        # opening PD servo in free motion, then 4 N closing force with damping.
        length = float(self.data.qpos[self.sim.finger_qpos].mean())
        velocity = float(self.data.qvel[self.finger_dof].mean())
        force = (-4.0 if self.gripper_closed
                 else 100.0 * (self.opening_target / 2.0 - length))
        self.data.ctrl[self.sim.gripper_actuator] = np.clip(force - 10.0 * velocity, -100, 100)

    def _contacts(self) -> tuple[bool, bool]:
        touching_fingers = set()
        table_support = False
        for index, contact in enumerate(self.data.contact):
            a = int(self.model.geom_bodyid[contact.geom1])
            b = int(self.model.geom_bodyid[contact.geom2])
            if self.target_body not in (a, b):
                continue
            other = b if a == self.target_body else a
            if other == self.table:
                table_support = True
            if other in self.fingers and contact.efc_address >= 0:
                mujoco.mj_contactForce(self.model, self.data, index, self.contact_force)
                if self.contact_force[0] > 0.01:
                    touching_fingers.add(other)
        return touching_fingers == self.fingers, table_support

    def _inverse_kinematics(self, position: np.ndarray) -> np.ndarray:
        self.work.qpos[:] = self.data.qpos
        target_quat = self.orientation_target
        quat = np.empty(4)
        inverse = np.empty(4)
        delta = np.empty(4)
        rotation_error = np.empty(3)
        for _ in range(180):
            mujoco.mj_kinematics(self.model, self.work)
            mujoco.mj_comPos(self.model, self.work)
            mujoco.mju_mat2Quat(quat, self.work.site_xmat[self.tcp])
            mujoco.mju_negQuat(inverse, quat)
            mujoco.mju_mulQuat(delta, target_quat, inverse)
            mujoco.mju_quat2Vel(rotation_error, delta, 1.0)
            position_error = position - self.work.site_xpos[self.tcp]
            if np.linalg.norm(position_error) < 0.0003 and np.linalg.norm(rotation_error) < 0.003:
                return self.work.qpos[self.sim.arm_qpos].copy()
            mujoco.mj_jacSite(self.model, self.work, self.jacp, self.jacr, self.tcp)
            jac = np.vstack((self.jacp[:, self.arm_dof], 0.3 * self.jacr[:, self.arm_dof]))
            error = np.concatenate((position_error, 0.3 * rotation_error))
            update = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(6), error)
            update *= min(1.0, 0.08 / max(float(np.max(np.abs(update))), 1e-12))
            self.work.qpos[self.sim.arm_qpos] = np.clip(
                self.work.qpos[self.sim.arm_qpos] + update,
                self.joint_limits[:, 0] + 1e-4,
                self.joint_limits[:, 1] - 1e-4,
            )
        raise ValueError("目标抓取位姿不可达，未执行该段运动")

    def _move(self, phase: str, position: np.ndarray) -> None:
        try:
            target = self._inverse_kinematics(position)
        except ValueError as exc:
            self.fail(str(exc))
            return
        self.motion_start = self.data.qpos[self.sim.arm_qpos].copy()
        self.motion_target = target
        self.target_position = position.copy()
        distance = float(np.linalg.norm(position - self.data.site_xpos[self.tcp]))
        joint_distance = float(np.max(np.abs(target - self.motion_start)))
        # Quintic interpolation has peak derivative 1.875. These limits are
        # trajectory settings, not a claim of optimal timing or collision planning.
        speed = self.parameters.descent_speed_m_s if phase == "descend" else 0.10
        self.motion_duration = max(0.8, 1.875 * distance / speed, 1.875 * joint_distance / 0.8)
        self._enter(phase)

    def _motion_done(self, elapsed: float) -> bool:
        u = min(1.0, elapsed / self.motion_duration)
        blend = u**3 * (10.0 - 15.0 * u + 6.0 * u**2)
        self.arm_target = self.motion_start + blend * (self.motion_target - self.motion_start)
        arrived = np.linalg.norm(self.data.site_xpos[self.tcp] - self.target_position) < 0.004
        slow = np.max(np.abs(self.data.qvel[self.arm_dof])) < 0.08
        if elapsed > self.motion_duration + 2.0:
            self.fail("机械臂未在时限内到达目标位姿")
            return False
        return elapsed >= self.motion_duration and arrived and slow

    def _start_closing(self) -> None:
        if not self.gripper_closed:
            self.gripper_closed = True
            self.close_started_s = float(self.data.time) - self.started_at

    def _record_overlap(self, dt: float) -> None:
        # Only credit a completed physics interval while descent and closure
        # were both active. World-frame site velocity is [angular, linear].
        if self.phase != "descend" or not self.gripper_closed:
            return
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_SITE,
            self.tcp, self.tcp_velocity, 0,
        )
        if (self.tcp_velocity[5] < -1e-4
                and float(self.data.qvel[self.finger_dof].sum()) < -1e-4):
            self.descent_close_overlap_s += dt

    def update(self) -> None:
        if self.status not in ("running", "succeeded"):
            return
        dt = float(self.model.opt.timestep)
        self._record_overlap(dt)
        elapsed = float(self.data.time) - self.phase_started_at
        if self.status == "running" and self.data.time - self.started_at >= TASKS[self.task_kind].watchdog_s:
            self.fail("抓取执行超时")
            return
        if self.storage is not None and self.phase not in TASKS["grasp"].phases:
            self.storage.update(dt, elapsed)
            return
        bilateral, table_support = self._contacts()
        target_pos = self.data.xpos[self.target_body]
        if self.phase == "prepare":
            velocity = self.data.qvel[self.target_dof:self.target_dof + 6]
            ready = (table_support and np.linalg.norm(velocity[:3]) < 0.01
                     and np.linalg.norm(velocity[3:]) < 0.1
                     and abs(float(self.data.qpos[self.sim.finger_qpos].sum())
                             - self.parameters.initial_opening_m) < 0.002)
            self.stable_elapsed = self.stable_elapsed + dt if ready else 0.0
            if self.stable_elapsed >= 0.06:
                self.reference_z = float(target_pos[2])
                # Profiles are body-local: transform the rim/neck offset using
                # the measured settled orientation, not just the reset yaw.
                rotation = self.data.xmat[self.target_body].reshape(3, 3)
                self.grasp_position = target_pos + rotation @ self.target.grasp_offset_m
                self._move("approach", self.grasp_position + [0.0, 0.0, 0.16])
            elif elapsed > 2.0:
                self.fail(f"{self.target.label}未在桌面稳定，或夹爪未达到初始开度")
        elif self.phase in ("approach", "descend", "lift"):
            phase = self.phase
            arrived = self._motion_done(elapsed)
            if self.status != "running":
                return
            if (phase == "descend" and not arrived and not self.gripper_closed
                    and self.parameters.close_trigger_height_m > 0.0):
                tcp_position = self.data.site_xpos[self.tcp]
                remaining_height = float(tcp_position[2] - self.grasp_position[2])
                aligned = np.linalg.norm(tcp_position[:2] - self.grasp_position[:2]) < 0.004
                if aligned and 0.0 <= remaining_height <= self.parameters.close_trigger_height_m:
                    self._start_closing()
            if arrived:
                if phase == "approach":
                    self._move("descend", self.grasp_position)
                elif phase == "descend":
                    self.descent_completed_s = float(self.data.time) - self.started_at
                    self._enter("close")
                    self._start_closing()
                else:
                    self.hold_relative = self.data.xpos[self.target_body] - self.data.site_xpos[self.tcp]
                    self._enter("hold")
            if phase == "lift" and self.status == "running":
                self.lost_contact_elapsed = 0.0 if bilateral else self.lost_contact_elapsed + dt
                if self.lost_contact_elapsed > 0.15:
                    self.fail("抬升过程中失去双侧夹持接触")
        elif self.phase == "close":
            # The contact timeout starts at confirmed descent arrival, not early closure.
            self.stable_elapsed = self.stable_elapsed + dt if bilateral else 0.0
            if self.stable_elapsed >= 0.10:
                self._move("lift", self.grasp_position + [0.0, 0.0, 0.15])
            elif elapsed > 2.0:
                self.fail("夹爪未建立稳定的双侧接触")
        elif self.phase in ("hold", "done"):
            relative = target_pos - self.data.site_xpos[self.tcp]
            stable = np.linalg.norm(relative - self.hold_relative) < 0.01
            lifted = float(target_pos[2]) - self.reference_z >= self.TARGET_LIFT
            if not (bilateral and not table_support and lifted and stable):
                self.hold_elapsed = 0.0
                self.fail(f"{self.target.label}掉落、滑移或未达到要求的离桌高度")
                return
            if self.phase == "done":
                return
            self.hold_elapsed += dt
            if self.hold_elapsed + 1e-9 >= self.REQUIRED_HOLD:
                if self.storage is not None:
                    self.storage.begin()
                    return
                self._enter("done")
                self.status = "succeeded"
                self.finished_at = float(self.data.time)

    def state(self) -> dict:
        end = float(self.data.time) if self.finished_at is None else self.finished_at
        message = self.MESSAGES[self.phase].format(target=self.target.label)
        if self.storage is not None and self.phase == "done":
            message = "黑碗已入屉、抽屉已通过接触推关、机械臂已归位；持续验证中"
        return {
            "name": f"grasp_{self.target.name}" if self.storage is None else "store_bowl",
            "target_label": self.target.label,
            "status": self.status, "phase": self.phase,
            "task_kind": self.task_kind,
            "phase_order": list(TASKS[self.task_kind].phases),
            "storage": self.storage.state() if self.storage is not None else None,
            "message": self.failure_reason or message,
            "elapsed_s": end - self.started_at if self.status != "idle" else 0.0,
            "phase_elapsed_s": end - self.phase_started_at if self.status == "running" else 0.0,
            "lift_height_m": (float(self.data.xpos[self.target_body, 2]) - self.reference_z
                              if self.reference_z is not None else 0.0),
            "hold_elapsed_s": self.hold_elapsed,
            "target_lift_m": self.TARGET_LIFT, "required_hold_s": self.REQUIRED_HOLD,
            "phase_times_s": dict(self.phase_times), "failure_reason": self.failure_reason,
            "parameters": self.parameters.to_dict(),
            "parameter_limits": GraspParameters.limits(),
            "close_started_s": self.close_started_s,
            "descent_completed_s": self.descent_completed_s,
            "descent_close_overlap_s": self.descent_close_overlap_s,
        }
