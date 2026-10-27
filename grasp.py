"""Feedback-checked, fixed-script bowl grasp using position-actuated joints."""

import mujoco
import numpy as np


class GraspSkill:
    TARGET_LIFT = 0.10
    REQUIRED_HOLD = 1.0
    WATCHDOG = 20.0
    MESSAGES = {
        "idle": "等待开始；固定脚本，不使用 GPT",
        "prepare": "打开夹爪，确认黑碗已在桌面稳定",
        "approach": "移动到黑碗上方",
        "descend": "保持夹爪打开，下降到抓取位置",
        "close": "闭合夹爪，确认双侧真实接触",
        "lift": "保持夹紧并向上抬起",
        "hold": "检查离桌高度、双侧接触和持续保持",
        "done": "黑碗已抬起并稳定保持；重置后可再次执行",
        "failed": "抓取失败；重置后可再次执行",
    }

    def __init__(self, sim) -> None:
        self.sim = sim
        self.model = sim.model
        self.data = sim.data
        self.tcp = self.model.site("robot_grasp_center").id
        self.bowl = self.model.body("bowl_object").id
        self.fingers = {
            self.model.body("robot_left_finger").id,
            self.model.body("robot_right_finger").id,
        }
        self.table = self.model.body("table").id
        self.bowl_dof = self.model.joint("bowl_free").dofadr[0]
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

    def start(self) -> None:
        if self.status != "idle":
            raise ValueError("请先重置场景，再开始新的抓取")
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
        velocity = float(np.mean([
            self.data.qvel[self.model.joint(f"robot_finger_joint{i}").dofadr[0]]
            for i in (1, 2)
        ]))
        force = -4.0 if self.gripper_closed else 100.0 * (0.04 - length)
        self.data.ctrl[self.sim.gripper_actuator] = np.clip(force - 10.0 * velocity, -100, 100)

    def _contacts(self) -> tuple[bool, bool]:
        touching_fingers = set()
        table_support = False
        for index, contact in enumerate(self.data.contact):
            a = int(self.model.geom_bodyid[contact.geom1])
            b = int(self.model.geom_bodyid[contact.geom2])
            if self.bowl not in (a, b):
                continue
            other = b if a == self.bowl else a
            if other == self.table:
                table_support = True
            if other in self.fingers and contact.efc_address >= 0:
                mujoco.mj_contactForce(self.model, self.data, index, self.contact_force)
                if self.contact_force[0] > 0.01:
                    touching_fingers.add(other)
        return touching_fingers == self.fingers, table_support

    def _inverse_kinematics(self, position: np.ndarray) -> np.ndarray:
        self.work.qpos[:] = self.data.qpos
        target_quat = np.array([0.0, 1.0, 0.0, 0.0])
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
        self.motion_duration = max(0.8, 1.875 * distance / 0.10, 1.875 * joint_distance / 0.8)
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

    def update(self) -> None:
        if self.status != "running":
            return
        dt = float(self.model.opt.timestep)
        elapsed = float(self.data.time) - self.phase_started_at
        if self.data.time - self.started_at >= self.WATCHDOG:
            self.fail("抓取执行超时")
            return
        bilateral, table_support = self._contacts()
        bowl_pos = self.data.xpos[self.bowl]
        if self.phase == "prepare":
            velocity = self.data.qvel[self.bowl_dof:self.bowl_dof + 6]
            ready = (table_support and np.linalg.norm(velocity[:3]) < 0.01
                     and np.linalg.norm(velocity[3:]) < 0.1
                     and self.data.qpos[self.sim.finger_qpos].sum() > 0.078)
            self.stable_elapsed = self.stable_elapsed + dt if ready else 0.0
            if self.stable_elapsed >= 0.06:
                self.reference_z = float(bowl_pos[2])
                # The 107 mm bowl is wider than the 80 mm gripper: pinch its
                # near rim, with one finger inside and the other outside.
                self.grasp_position = bowl_pos.copy() + [0.0, -0.047, 0.042]
                self._move("approach", self.grasp_position + [0.0, 0.0, 0.16])
            elif elapsed > 2.0:
                self.fail("黑碗未在桌面稳定，或夹爪未能打开")
        elif self.phase in ("approach", "descend", "lift"):
            phase = self.phase
            if self._motion_done(elapsed):
                if phase == "approach":
                    self._move("descend", self.grasp_position)
                elif phase == "descend":
                    self._enter("close")
                    self.gripper_closed = True
                else:
                    self.hold_relative = self.data.xpos[self.bowl] - self.data.site_xpos[self.tcp]
                    self._enter("hold")
            if phase == "lift" and self.status == "running":
                self.lost_contact_elapsed = 0.0 if bilateral else self.lost_contact_elapsed + dt
                if self.lost_contact_elapsed > 0.15:
                    self.fail("抬升过程中失去双侧夹持接触")
        elif self.phase == "close":
            self.stable_elapsed = self.stable_elapsed + dt if bilateral else 0.0
            if self.stable_elapsed >= 0.10:
                self._move("lift", self.grasp_position + [0.0, 0.0, 0.15])
            elif elapsed > 2.0:
                self.fail("夹爪未建立稳定的双侧接触")
        elif self.phase == "hold":
            relative = bowl_pos - self.data.site_xpos[self.tcp]
            stable = np.linalg.norm(relative - self.hold_relative) < 0.01
            lifted = float(bowl_pos[2]) - self.reference_z >= self.TARGET_LIFT
            if not (bilateral and not table_support and lifted and stable):
                self.hold_elapsed = 0.0
                self.fail("黑碗掉落、滑移或未达到要求的离桌高度")
                return
            self.hold_elapsed += dt
            if self.hold_elapsed + 1e-9 >= self.REQUIRED_HOLD:
                self._enter("done")
                self.status = "succeeded"
                self.finished_at = float(self.data.time)

    def state(self) -> dict:
        end = float(self.data.time) if self.finished_at is None else self.finished_at
        return {
            "name": "grasp_bowl", "status": self.status, "phase": self.phase,
            "message": self.failure_reason or self.MESSAGES[self.phase],
            "elapsed_s": end - self.started_at if self.status != "idle" else 0.0,
            "phase_elapsed_s": end - self.phase_started_at if self.status == "running" else 0.0,
            "lift_height_m": (float(self.data.xpos[self.bowl, 2]) - self.reference_z
                              if self.reference_z is not None else 0.0),
            "hold_elapsed_s": self.hold_elapsed,
            "target_lift_m": self.TARGET_LIFT, "required_hold_s": self.REQUIRED_HOLD,
            "phase_times_s": dict(self.phase_times), "failure_reason": self.failure_reason,
        }
