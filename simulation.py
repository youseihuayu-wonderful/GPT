"""Panda workcell physics and world-frame ground-truth observations."""

import argparse
import base64
from io import BytesIO
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from grasp import GraspParameters, GraspSkill
from collisions import CollisionTracker
from scenarios import DEFAULT_CASE, TARGETS, SceneCase
from task_specs import SKILLS, TASKS

ROOT = Path(__file__).resolve().parent
ARM_HOME = np.array([0.0, -0.5, 0.0, -2.2, 0.0, 1.8, 0.7853981634])


class Simulation:
    def __init__(self, case: SceneCase = DEFAULT_CASE) -> None:
        if not isinstance(case, SceneCase):
            raise ValueError("case must be a SceneCase")
        # Validate before compiling or mutating any physics state.
        case.__post_init__()
        self.case = case
        self.target = TARGETS[case.target_object]
        scene = mujoco.MjSpec.from_file(str(ROOT / "scene.xml"))
        robot = mujoco.MjSpec.from_file(str(ROOT / "vendor" / "panda" / "panda.xml"))
        for key in list(robot.keys):
            robot.delete(key)
        for light in list(robot.lights):
            robot.delete(light)
        robot.body("hand").add_site(
            name="grasp_center", pos=[0, 0, 0.1034],
            size=[0.003, 0.003, 0.003], rgba=[0, 0, 0, 0],
        )
        # Keep the original tendon and force limits, but command its force.
        # A zero-width position servo loses squeeze force on a thin bowl rim.
        gripper = robot.actuator("actuator8")
        gripper.gainprm[:] = 0
        gripper.gainprm[0] = 1
        gripper.biasprm[:] = 0
        gripper.ctrlrange[:] = [-100, 100]
        scene.attach(robot, frame=scene.frame("robot_mount"), prefix="robot_")
        assets = ROOT / "vendor" / "libero"
        objects = (
            ("cabinet", "articulated_objects/white_cabinet.xml", False),
            ("bowl", "stable_scanned_objects/akita_black_bowl/akita_black_bowl.xml", True),
            ("bottle", "turbosquid_objects/wine_bottle/wine_bottle.xml", True),
            ("rack", "turbosquid_objects/wine_rack/wine_rack.xml", False),
        )
        for name, filename, movable in objects:
            part = mujoco.MjSpec.from_file(str(assets / filename))
            root = part.worldbody.bodies[0]
            root.name = "root"
            part.body("object").pos[:] = 0
            if movable:
                root.add_freejoint(name="free")
            scene.attach(part, frame=scene.frame(f"{name}_mount"), prefix=f"{name}_")
        bowl_material = scene.material("bowl_akita_black_bowl")
        bowl_material.rgba[:] = [0.10, 0.12, 0.15, 1.0]
        bowl_material.reflectance = 0
        bowl_material.specular = 0.1
        self.model = scene.compile()
        self.data = mujoco.MjData(self.model)
        self.arm_qpos = np.array([
            self.model.joint(f"robot_joint{i}").qposadr[0] for i in range(1, 8)
        ])
        self.arm_actuators = np.array([
            self.model.actuator(f"robot_actuator{i}").id for i in range(1, 8)
        ])
        self.finger_qpos = np.array([
            self.model.joint(f"robot_finger_joint{i}").qposadr[0] for i in (1, 2)
        ])
        self.gripper_actuator = self.model.actuator("robot_actuator8").id
        self.arm_home = ARM_HOME.copy()
        self.object_names = ("bowl_object", "bottle_object", "cabinet_cabinet_bottom")
        self.renderer = None
        self.camera = mujoco.MjvCamera()
        self.options = mujoco.MjvOption()
        # LIBERO collisions use group 0; Menagerie collisions use group 3.
        self.options.geomgroup[0] = 0
        self.options.geomgroup[3] = 0
        self.skill = GraspSkill(self)
        self.collisions = CollisionTracker(
            self.model, self.data, self.skill.target_body, self.skill.fingers, self.skill.table,
        )
        self.reset()

    def reset(self) -> None:
        self.case.__post_init__()
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.arm_qpos] = ARM_HOME
        self.data.qpos[self.finger_qpos] = 0.04
        self.data.ctrl[self.arm_actuators] = ARM_HOME
        self.data.ctrl[self.gripper_actuator] = 0
        self.data.qpos[self.model.joint("cabinet_bottom_level").qposadr[0]] = -0.16
        joint = self.model.joint(f"{self.target.name}_free")
        qpos = int(joint.qposadr[0])
        self.data.qpos[qpos:qpos + 2] += self.case.offset_xy_m
        half_yaw = self.case.yaw_rad / 2.0
        yaw = np.array([np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)])
        orientation = self.data.qpos[qpos + 3:qpos + 7].copy()
        mujoco.mju_mulQuat(self.data.qpos[qpos + 3:qpos + 7], yaw, orientation)
        mujoco.mj_forward(self.model, self.data)
        self.skill.reset()
        self.collisions.reset()
        self._numerical_failure = None

    def start_task(self, task_kind: str, parameters: GraspParameters | None = None) -> None:
        self.skill.start(task_kind, parameters)
        self.collisions.reset()

    def start_skill(self, skill_name: str, parameters: GraspParameters | None = None) -> None:
        self.skill.start_skill(skill_name, parameters)

    def step(self, duration_s: float = 0.02) -> None:
        steps = round(duration_s / self.model.opt.timestep)
        if steps < 1 or not np.isclose(steps * self.model.opt.timestep, duration_s):
            raise ValueError("Duration must be a positive multiple of the physics timestep")
        self._check_numerical_state()
        for _ in range(steps):
            self.skill.update()
            self.skill.apply_control()
            self._check_numerical_state()
            track_contacts = self.skill.status in ("running", "succeeded")
            mujoco.mj_step(self.model, self.data)
            # Refresh poses/contact forces for the controller at the next step.
            mujoco.mj_forward(self.model, self.data)
            self._check_numerical_state()
            if track_contacts:
                self.collisions.update(float(self.model.opt.timestep), self.skill)
                if (self.skill.task_kind == "store_bowl" or self.skill.execution_mode == "skill") and self.collisions.records:
                    self.skill.fail("Unplanned contact during skill execution")
                else:
                    self.skill.verify_terminal()

    def numerical_error(self) -> str | None:
        if self._numerical_failure is not None:
            return self._numerical_failure
        warnings = self.warning_names()
        invalid = [name for name in ("qpos", "qvel", "qacc", "ctrl")
                   if not np.isfinite(getattr(self.data, name)).all()]
        if warnings or invalid:
            return f"MuJoCo state is invalid: warnings={warnings}, nonfinite={invalid}"
        return None

    def _check_numerical_state(self) -> None:
        error = self.numerical_error()
        if error is not None:
            self._numerical_failure = error
            if self.skill.status != "failed":
                self.skill.fail(error)
            raise RuntimeError(error)

    def get_object_pose(self, name: str) -> dict:
        if name not in self.object_names:
            raise ValueError(f"Unknown scene object: {name}")
        body = self.data.body(name)
        return {
            "name": name,
            "position_m": body.xpos.tolist(),
            "quaternion_wxyz": body.xquat.tolist(),
        }

    def warning_names(self) -> list[str]:
        return [
            mujoco.mjtWarning(i).name
            for i, warning in enumerate(self.data.warning)
            if warning.number
        ]

    def state(self) -> dict:
        return {
            "time_s": float(self.data.time),
            "timestep_s": float(self.model.opt.timestep),
            "scene_name": "Panda 桌面工位 · 黑碗 / 酒瓶 / 抽屉柜 / 酒架",
            "robot_name": "Franka Panda · 7 轴 · 双指夹爪",
            "source": "Menagerie Panda + LIBERO 视频同款物体资产；重新布置的工位，非原任务回放",
            "target_object": self.target.name,
            "grasp_target": self.target.to_dict(),
            "scenario": self.case.to_dict(),
            "available_tasks": [task.to_dict() for task in TASKS.values()
                                if self.target.name in task.target_objects],
            "available_skills": self.skill.available_skills(),
            "collisions": self.collisions.state(),
            "objects": [self.get_object_pose(name) for name in self.object_names],
            "joint_positions_rad": self.data.qpos[self.arm_qpos].tolist(),
            "gripper_opening_m": float(self.data.qpos[self.finger_qpos].sum()),
            "contacts": int(self.data.ncon),
            "warnings": self.warning_names(),
            "skill": self.skill.state(),
        }

    def render(self, view: str = "overview") -> str:
        views = {
            "overview": (135.0, -25.0, 2.2),
            "front": (90.0, -18.0, 2.0),
            "top": (90.0, -80.0, 2.05),
        }
        self.camera.azimuth, self.camera.elevation, self.camera.distance = views[view]
        self.camera.lookat[:] = [0.3, 0.0, 0.95]
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=720, width=960)
        self.renderer.update_scene(self.data, camera=self.camera, scene_option=self.options)
        image = self.renderer.render()
        with BytesIO() as buffer:
            Image.fromarray(image).save(buffer, format="JPEG", quality=85)
            return base64.b64encode(buffer.getvalue()).decode("ascii")

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--image", type=Path, help="Save an actual MuJoCo-rendered JPEG")
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument("--task", choices=tuple(TASKS), help="Execute a fixed task using the shared skills")
    execution.add_argument("--skill", choices=tuple(SKILLS), nargs="+", help="Call these independent skills in order without resetting")
    parser.add_argument("--target", choices=tuple(TARGETS), default="bowl", help="Object to grasp")
    options = (
        ("--initial-opening-mm", "initial_opening_m", "Initial finger opening", "mm"),
        ("--descent-speed-mm-s", "descent_speed_m_s", "Nominal descent speed", "mm/s"),
        ("--close-trigger-height-mm", "close_trigger_height_m", "Remaining descent height to start closing", "mm"),
    )
    limits = GraspParameters.limits()
    defaults = GraspParameters().to_dict()
    for flag, name, description, unit in options:
        bounds = limits[name]
        parser.add_argument(
            flag, dest=name, type=float,
            help=(f"{description}: {bounds['min'] * 1000:g}–{bounds['max'] * 1000:g} {unit}, "
                  f"default {defaults[name] * 1000:g}; requires a grasp task/skill"),
        )
    args = parser.parse_args()
    supplied = {
        name: getattr(args, name) / 1000.0
        for _, name, _, _ in options if getattr(args, name) is not None
    }
    if supplied and not (args.task or (args.skill and "grasp_and_hold" in args.skill)):
        parser.error("Grasp parameter options require a grasp task or grasp_and_hold skill")
    try:
        parameters = GraspParameters(**supplied) if supplied else None
    except ValueError as exc:
        parser.error(str(exc))
    case = SceneCase(f"{args.target}.nominal", args.target, "nominal", 0, (0.0, 0.0), 0.0)
    sim = Simulation(case)
    try:
        if args.seconds:
            sim.step(args.seconds)
        if args.task:
            sim.start_task(args.task, parameters)
            while sim.skill.status == "running":
                sim.step()
        elif args.skill:
            for name in args.skill:
                sim.start_skill(name, parameters if name == "grasp_and_hold" else None)
                while sim.skill.status == "running":
                    sim.step()
                if sim.skill.status != "succeeded":
                    break
        print(json.dumps(sim.state(), indent=2, ensure_ascii=False))
        if args.image:
            args.image.write_bytes(base64.b64decode(sim.render()))
        if (args.task or args.skill) and sim.skill.status != "succeeded":
            raise SystemExit(1)
    finally:
        sim.close()


if __name__ == "__main__":
    main()
