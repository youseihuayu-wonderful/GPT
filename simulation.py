"""Panda workcell physics and world-frame ground-truth observations."""

import argparse
import base64
from io import BytesIO
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
ARM_HOME = np.array([0.0, -0.5, 0.0, -2.2, 0.0, 1.8, 0.7853981634])


class Simulation:
    def __init__(self) -> None:
        scene = mujoco.MjSpec.from_file(str(ROOT / "scene.xml"))
        robot = mujoco.MjSpec.from_file(str(ROOT / "vendor" / "panda" / "panda.xml"))
        for key in list(robot.keys):
            robot.delete(key)
        for light in list(robot.lights):
            robot.delete(light)
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
        self.object_names = ("bowl_object", "bottle_object", "cabinet_cabinet_bottom")
        self.renderer = None
        self.camera = mujoco.MjvCamera()
        self.options = mujoco.MjvOption()
        # LIBERO collisions use group 0; Menagerie collisions use group 3.
        self.options.geomgroup[0] = 0
        self.options.geomgroup[3] = 0
        self.reset()

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.arm_qpos] = ARM_HOME
        self.data.qpos[self.finger_qpos] = 0.04
        self.data.ctrl[self.arm_actuators] = ARM_HOME
        self.data.ctrl[self.gripper_actuator] = 255
        self.data.qpos[self.model.joint("cabinet_bottom_level").qposadr[0]] = -0.12
        mujoco.mj_forward(self.model, self.data)

    def step(self, duration_s: float = 0.02) -> None:
        steps = round(duration_s / self.model.opt.timestep)
        if steps < 1 or not np.isclose(steps * self.model.opt.timestep, duration_s):
            raise ValueError("Duration must be a positive multiple of the physics timestep")
        mujoco.mj_step(self.model, self.data, nstep=steps)
        # mj_step leaves Cartesian fields from its last forward pass; refresh them
        # so poses and rendering describe the final integrated state, not t - dt.
        mujoco.mj_forward(self.model, self.data)
        warnings = self.warning_names()
        if warnings or not np.isfinite(self.data.qpos).all():
            raise RuntimeError(f"MuJoCo state is invalid: {warnings}")

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
            "objects": [self.get_object_pose(name) for name in self.object_names],
            "joint_positions_rad": self.data.qpos[self.arm_qpos].tolist(),
            "gripper_opening_m": float(self.data.qpos[self.finger_qpos].sum()),
            "contacts": int(self.data.ncon),
            "warnings": self.warning_names(),
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
    args = parser.parse_args()
    sim = Simulation()
    try:
        if args.seconds:
            sim.step(args.seconds)
        print(json.dumps(sim.state(), indent=2, ensure_ascii=False))
        if args.image:
            args.image.write_bytes(base64.b64decode(sim.render()))
    finally:
        sim.close()


if __name__ == "__main__":
    main()
