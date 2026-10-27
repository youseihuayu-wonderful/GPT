"""Headless, real-physics grasp behavior checks using stdlib unittest.

Run from the project root with the project's dependencies installed:
    python -m unittest discover -s tests -p 'test_grasp.py' -v

Contract assumptions: prepare settles the bowl without manipulating it, then
approach starts before any grip or lift. The first sample after prepare therefore
provides an independent settled reference, rather than trusting lift telemetry.
Contacts mean active, non-separated MuJoCo constraints between the actual bowl
and each finger. No rendering, wall-clock timing, or controller internals are used.

A 20 ms observation interval can miss at most one interval at hold onset. The
pre-success contact/lift history allows that sampling uncertainty; reported hold
must still reach 1.0 s. Another sixty seconds of physics after success must retain
both contacts, at least 0.10 m lift, and position within 0.01 m of the terminal
pose. This catches slow soft-contact drift that a one-second check misses.

Add a case as another test method; setUp creates a fresh model and registers
Simulation.close even when an assertion fails. Mutate only that case's model
when introducing a physical fault.
"""

from copy import deepcopy
import math
import unittest

from simulation import Simulation


class GraspBehaviorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sim = Simulation()
        self.addCleanup(self.sim.close)
        self.bowl_body = self.sim.model.body("bowl_object").id
        self.bowl_qpos = int(self.sim.model.joint("bowl_free").qposadr[0])
        self.finger_bodies = {
            self.sim.model.body("robot_left_finger").id,
            self.sim.model.body("robot_right_finger").id,
        }

    def _sample(self) -> dict:
        touching_fingers = set()
        for contact in self.sim.data.contact:
            if contact.efc_address < 0 or contact.dist > 0.0:
                continue
            body1 = int(self.sim.model.geom_bodyid[contact.geom1])
            body2 = int(self.sim.model.geom_bodyid[contact.geom2])
            if body1 == self.bowl_body and body2 in self.finger_bodies:
                touching_fingers.add(body2)
            elif body2 == self.bowl_body and body1 in self.finger_bodies:
                touching_fingers.add(body1)
        return {
            "time_s": float(self.sim.data.time),
            "skill": deepcopy(self.sim.state()["skill"]),
            "bowl_position": tuple(self.sim.data.body("bowl_object").xpos),
            "bowl_free_z": float(self.sim.data.qpos[self.bowl_qpos + 2]),
            "touching_fingers": touching_fingers,
        }

    def _step(self) -> dict:
        previous_time = float(self.sim.data.time)
        self.sim.step(0.02)
        self.assertAlmostEqual(self.sim.data.time - previous_time, 0.02, places=9)
        return self._sample()

    def _run_to_terminal(self) -> list[dict]:
        start_time = float(self.sim.data.time)
        self.assertEqual(self.sim.state()["skill"]["status"], "running")
        samples = []
        for _ in range(1000):
            sample = self._step()
            samples.append(sample)
            self.assertLessEqual(sample["time_s"] - start_time, 20.0 + 1e-9)
            if sample["skill"]["status"] in ("succeeded", "failed"):
                self.assertLessEqual(sample["skill"]["elapsed_s"], 20.0 + 1e-9)
                return samples
            self.assertEqual(sample["skill"]["status"], "running")
        self.fail("Grasp did not reach a terminal state within 20 simulated seconds")

    def _assert_idle(self) -> None:
        skill = self.sim.state()["skill"]
        self.assertEqual(skill["name"], "grasp_bowl")
        self.assertEqual(skill["status"], "idle")
        self.assertEqual(skill["phase"], "idle")
        self.assertEqual(skill["elapsed_s"], 0.0)
        self.assertEqual(skill["phase_elapsed_s"], 0.0)
        self.assertEqual(skill["lift_height_m"], 0.0)
        self.assertEqual(skill["hold_elapsed_s"], 0.0)
        self.assertEqual(skill["target_lift_m"], 0.10)
        self.assertEqual(skill["required_hold_s"], 1.0)
        for elapsed_s in skill["phase_times_s"].values():
            self.assertEqual(elapsed_s, 0.0)
        self.assertIsNone(skill["failure_reason"])

    def _has_physical_hold(self, sample: dict, baseline: dict) -> bool:
        return (
            sample["bowl_free_z"] - baseline["bowl_free_z"] >= 0.10
            and sample["bowl_position"][2] - baseline["bowl_position"][2] >= 0.10
            and sample["touching_fingers"] == self.finger_bodies
        )

    def _assert_successful_grasp(self, samples: list[dict]) -> None:
        terminal = samples[-1]
        skill = terminal["skill"]
        self.assertEqual(skill["status"], "succeeded")
        self.assertEqual(skill["phase"], "done")
        self.assertEqual(skill["target_lift_m"], 0.10)
        self.assertEqual(skill["required_hold_s"], 1.0)
        self.assertGreaterEqual(skill["lift_height_m"], 0.10)
        self.assertGreaterEqual(skill["hold_elapsed_s"], 1.0)
        self.assertIsNone(skill["failure_reason"])

        baseline = next(
            (sample for sample in samples if sample["skill"]["phase"] != "prepare"),
            None,
        )
        self.assertIsNotNone(baseline, "Grasp never left preparation")
        self.assertEqual(baseline["skill"]["phase"], "approach")
        self.assertGreaterEqual(terminal["bowl_free_z"] - baseline["bowl_free_z"], 0.10)
        self.assertGreaterEqual(
            terminal["bowl_position"][2] - baseline["bowl_position"][2], 0.10
        )
        self.assertEqual(terminal["touching_fingers"], self.finger_bodies)

        # Check the uninterrupted physical history, not only a timer or closed
        # fingers. One sample interval accounts for the unobserved hold onset.
        held_since = terminal["time_s"]
        for sample in reversed(samples):
            if not self._has_physical_hold(sample, baseline):
                break
            if math.dist(sample["bowl_position"], terminal["bowl_position"]) > 0.01:
                break
            held_since = sample["time_s"]
        self.assertGreaterEqual(terminal["time_s"] - held_since, 1.0 - 0.02 - 1e-9)

        # The watchdog applies to execution, not this post-terminal observation.
        # Keep advancing actual physics so a transient lift or dropped bowl cannot
        # be hidden by a latched succeeded status or stale Cartesian coordinates.
        for _ in range(3000):
            sample = self._step()
            self.assertEqual(sample["skill"]["status"], "succeeded")
            self.assertEqual(sample["skill"]["phase"], "done")
            self.assertGreaterEqual(sample["skill"]["hold_elapsed_s"], 1.0)
            self.assertGreaterEqual(sample["bowl_free_z"] - baseline["bowl_free_z"], 0.10)
            self.assertGreaterEqual(
                sample["bowl_position"][2] - baseline["bowl_position"][2], 0.10
            )
            self.assertEqual(sample["touching_fingers"], self.finger_bodies)
            self.assertLessEqual(
                math.dist(sample["bowl_position"], terminal["bowl_position"]), 0.01
            )
        self.assertAlmostEqual(self.sim.data.time - terminal["time_s"], 60.0, places=8)

    def test_fixed_scene_grasp_lifts_bowl_ten_centimeters_and_holds_one_second(self) -> None:
        self._assert_idle()
        self.sim.start_grasp()
        self._assert_successful_grasp(self._run_to_terminal())

    def test_reset_during_hold_restores_initial_physics_and_allows_complete_restart(self) -> None:
        initial_qpos = self.sim.data.qpos.tolist()
        initial_qvel = self.sim.data.qvel.tolist()
        initial_ctrl = self.sim.data.ctrl.tolist()
        self.assertEqual(self.sim.data.time, 0.0)
        self.sim.start_grasp()

        # Reset with a nonzero hold timer, not just during an untouched prepare
        # phase, so stale success progress and the displaced bowl must be cleared.
        for _ in range(1000):
            sample = self._step()
            self.assertLessEqual(sample["time_s"], 20.0 + 1e-9)
            self.assertEqual(sample["skill"]["status"], "running")
            if sample["skill"]["hold_elapsed_s"] > 0.0:
                break
        else:
            self.fail("Grasp never began a hold before the 20-second watchdog")
        self.assertLess(sample["skill"]["hold_elapsed_s"], 1.0)
        self.assertGreater(sample["skill"]["elapsed_s"], 0.0)
        self.assertNotEqual(self.sim.data.qpos.tolist(), initial_qpos)

        self.sim.reset()
        self._assert_idle()
        self.assertEqual(self.sim.data.time, 0.0)
        self.assertEqual(self.sim.data.qpos.tolist(), initial_qpos)
        self.assertEqual(self.sim.data.qvel.tolist(), initial_qvel)
        self.assertEqual(self.sim.data.ctrl.tolist(), initial_ctrl)

        self.sim.start_grasp()
        restarted = self.sim.state()["skill"]
        self.assertEqual(restarted["status"], "running")
        self.assertEqual(restarted["phase"], "prepare")
        self.assertEqual(restarted["elapsed_s"], 0.0)
        self.assertEqual(restarted["phase_elapsed_s"], 0.0)
        self.assertEqual(restarted["hold_elapsed_s"], 0.0)
        self._assert_successful_grasp(self._run_to_terminal())

    def test_duplicate_start_while_running_is_rejected_without_restarting_physics(self) -> None:
        self.sim.start_grasp()
        for _ in range(10):
            self._step()
        before = deepcopy(self.sim.state()["skill"])
        self.assertEqual(before["status"], "running")
        self.assertGreater(before["elapsed_s"], 0.0)
        before_qpos = self.sim.data.qpos.tolist()
        before_qvel = self.sim.data.qvel.tolist()
        before_ctrl = self.sim.data.ctrl.tolist()
        before_time = float(self.sim.data.time)
        progress_fields = (
            "status", "phase", "elapsed_s", "phase_elapsed_s", "hold_elapsed_s",
            "lift_height_m", "phase_times_s",
        )
        before_progress = {field: before[field] for field in progress_fields}

        with self.assertRaises(ValueError):
            self.sim.start_grasp()

        after = self.sim.state()["skill"]
        self.assertEqual({field: after[field] for field in progress_fields}, before_progress)
        self.assertEqual(self.sim.data.time, before_time)
        self.assertEqual(self.sim.data.qpos.tolist(), before_qpos)
        self.assertEqual(self.sim.data.qvel.tolist(), before_qvel)
        self.assertEqual(self.sim.data.ctrl.tolist(), before_ctrl)
        continued = self._step()
        self.assertEqual(continued["skill"]["status"], "running")
        self.assertGreater(continued["skill"]["elapsed_s"], before["elapsed_s"])

    def test_disabled_gripper_fails_without_achieving_a_one_second_hold(self) -> None:
        # setUp constructed a NEW Simulation: this real actuator fault cannot
        # leak into the successful-grasp or reset cases.
        self.sim.model.actuator_forcerange[self.sim.gripper_actuator] = [0, 0]
        self.sim.start_grasp()
        samples = self._run_to_terminal()
        self.assertEqual(samples[-1]["skill"]["status"], "failed")
        self.assertEqual(samples[-1]["skill"]["phase"], "failed")
        for sample in samples:
            self.assertLess(sample["skill"]["hold_elapsed_s"], 1.0)

    def test_losing_grip_after_success_invalidates_the_reported_success(self) -> None:
        self.sim.start_grasp()
        terminal = self._run_to_terminal()[-1]
        self.assertEqual(terminal["skill"]["status"], "succeeded")
        self.sim.model.actuator_forcerange[self.sim.gripper_actuator] = [0, 0]
        for _ in range(100):
            sample = self._step()
            if sample["skill"]["status"] == "failed":
                break
        else:
            self.fail("A lost grip left the live skill reporting success")
        self.assertEqual(sample["skill"]["phase"], "failed")
        self.assertEqual(sample["skill"]["hold_elapsed_s"], 0.0)
        self.assertIsNotNone(sample["skill"]["failure_reason"])


if __name__ == "__main__":
    unittest.main()
