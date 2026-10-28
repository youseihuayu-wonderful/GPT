"""Scene input boundaries and repeatable free-joint perturbations."""

from dataclasses import FrozenInstanceError, replace
import unittest

import numpy as np

from scenarios import DEFAULT_CASE, TARGETS, SceneCase, benchmark_cases
from simulation import Simulation


class SceneCaseTests(unittest.TestCase):
    def test_invalid_scene_fields_are_rejected(self) -> None:
        invalid = {
            "target_object": ("drawer", None, [], True),
            "case_id": ("", " ", "x" * 101, 5),
            "position_label": ("", None),
            "seed": (-1, 2**32, 1.5, True),
            "offset_xy_m": ([0.0, 0.0], (0.0,), (0.0, 0.0, 0.0),
                            (float("nan"), 0.0), (True, 0.0), (0.081, 0.0)),
            "yaw_rad": (float("inf"), float("nan"), True, "0", 0.201),
        }
        for field, values in invalid.items():
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    replace(DEFAULT_CASE, **{field: value})
        for case in (None, {}, "bottle", True):
            with self.subTest(case=case), self.assertRaises(ValueError):
                Simulation(case)

    def test_profiles_and_cases_are_immutable(self) -> None:
        with self.assertRaises(FrozenInstanceError):
            DEFAULT_CASE.yaw_rad = 0.1
        with self.assertRaises(FrozenInstanceError):
            TARGETS["bottle"].grasp_offset_m = (0.0, 0.0, 0.1)
        with self.assertRaises(TypeError):
            TARGETS["bottle"] = TARGETS["bowl"]

    def test_seeded_matrix_is_repeatable_distinct_and_bounded(self) -> None:
        cases = benchmark_cases()
        self.assertEqual(cases, benchmark_cases())
        self.assertEqual(len(cases), 18)
        self.assertEqual(len({case.case_id for case in cases}), 18)
        self.assertEqual({case.target_object for case in cases}, {"bowl", "bottle"})
        for target in TARGETS:
            for label, base in (("nominal", (0, 0)), ("right", (.04, 0)), ("forward", (0, .04))):
                group = [case for case in cases if case.target_object == target and case.position_label == label]
                self.assertEqual({case.seed for case in group}, {101, 202, 303})
                self.assertEqual(len({case.offset_xy_m for case in group}), 3)
                for case in group:
                    self.assertLessEqual(abs(case.yaw_rad), .06)
                    for actual, nominal in zip(case.offset_xy_m, base):
                        self.assertLessEqual(abs(actual - nominal), .003)
        for seeds in ((), (1, 1), (True,), (-1,), (2**32,)):
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                benchmark_cases(seeds)

    def test_reset_restores_same_perturbed_physics_and_settled_state(self) -> None:
        for target in TARGETS:
            with self.subTest(target=target):
                case = SceneCase(f"{target}.reset", target, "right", 101, (.04, -.002), .05)
                sim = Simulation(case)
                self.addCleanup(sim.close)
                initial = tuple(array.copy() for array in (sim.data.qpos, sim.data.qvel, sim.data.ctrl))
                adr = int(sim.model.joint(f"{target}_free").qposadr[0])
                np.testing.assert_allclose(initial[0][adr:adr + 2], sim.model.qpos0[adr:adr + 2] + case.offset_xy_m)
                np.testing.assert_allclose(initial[0][adr + 3:adr + 7], [np.cos(.025), 0, 0, np.sin(.025)])
                other = "bottle" if target == "bowl" else "bowl"
                other_adr = int(sim.model.joint(f"{other}_free").qposadr[0])
                np.testing.assert_array_equal(initial[0][other_adr:other_adr + 7], sim.model.qpos0[other_adr:other_adr + 7])
                sim.step(1.0)
                settled = tuple(array.copy() for array in (sim.data.qpos, sim.data.qvel, sim.data.ctrl))
                sim.start_task("grasp")
                sim.step(.2)
                sim.reset()
                for actual, expected in zip((sim.data.qpos, sim.data.qvel, sim.data.ctrl), initial):
                    np.testing.assert_array_equal(actual, expected)
                sim.step(1.0)
                for actual, expected in zip((sim.data.qpos, sim.data.qvel, sim.data.ctrl), settled):
                    np.testing.assert_array_equal(actual, expected)
                paired = Simulation(case)
                self.addCleanup(paired.close)
                paired.step(1.0)
                for actual, expected in zip((paired.data.qpos, paired.data.qvel, paired.data.ctrl), settled):
                    np.testing.assert_array_equal(actual, expected)
                self.assertEqual(sim.state()["scenario"], case.to_dict())

    def test_grasp_offset_uses_measured_world_rotation(self) -> None:
        case = replace(DEFAULT_CASE, case_id="bowl.rotated", yaw_rad=.15)
        sim = Simulation(case)
        self.addCleanup(sim.close)
        sim.step(1.0)
        sim.start_task("grasp")
        for _ in range(1000):
            sim.step(.002)
            if sim.skill.phase != "prepare":
                break
        self.assertEqual(sim.skill.phase, "approach")
        body = sim.skill.target_body
        expected = sim.data.xpos[body] + sim.data.xmat[body].reshape(3, 3) @ TARGETS["bowl"].grasp_offset_m
        np.testing.assert_allclose(sim.skill.grasp_position, expected, atol=1e-5)
        self.assertGreater(abs(sim.skill.grasp_position[0] - sim.data.xpos[body, 0]), .005)


if __name__ == "__main__":
    unittest.main()
