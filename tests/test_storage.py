"""Real-contact end-to-end storage and fail-closed task-contract regressions."""

from copy import deepcopy
import unittest

import mujoco
import numpy as np

from grasp import GraspParameters
from scenarios import SceneCase
from simulation import ARM_HOME, Simulation
from task_specs import TASKS


class StorageTests(unittest.TestCase):
    def make_sim(self, target="bowl"):
        sim = Simulation(SceneCase(f"{target}.nominal", target, "nominal", 0, (0.0, 0.0), 0.0))
        self.addCleanup(sim.close)
        sim.step(1.0)
        return sim

    def run_until(self, sim, phase=None):
        seen = []
        for _ in range(4501):
            if not seen or seen[-1] != sim.skill.phase:
                seen.append(sim.skill.phase)
            if sim.skill.status != "running" or sim.skill.phase == phase:
                return seen
            sim.step(.02)
        self.fail("Storage exceeded its 90-second watchdog")

    def assert_stored(self, sim):
        state = sim.state()
        self.assertEqual(state["skill"]["status"], "succeeded", state["skill"])
        self.assertTrue(state["drawer"]["closed"])
        self.assertFalse(state["drawer"]["fully_open"])
        storage = state["skill"]["storage"]
        for key in ("bowl_inside_drawer", "bowl_supported", "gripper_clear", "drawer_closed", "home_reached", "push_contact_seen"):
            self.assertTrue(storage[key], (key, storage))
        self.assertGreater(storage["drawer_travel_m"], .15)
        self.assertLessEqual(abs(storage["drawer_position_m"]), .003)
        self.assertLessEqual(storage["home_error_rad"], .03)
        self.assertGreaterEqual(storage["verification_hold_s"], 1.0)
        self.assertEqual(state["collisions"]["episodes"], 0)
        self.assertEqual(state["warnings"], [])
        np.testing.assert_allclose(sim.data.qpos[sim.arm_qpos], ARM_HOME, atol=.03)
        # Independently inspect all live target/finger constraints after release.
        for contact in sim.data.contact:
            bodies = {int(sim.model.geom_bodyid[contact.geom1]), int(sim.model.geom_bodyid[contact.geom2])}
            if sim.skill.target_body in bodies:
                self.assertFalse(bodies & sim.skill.fingers)

    def test_default_and_fast_complete_sequence_remain_stored_after_extra_hold(self):
        for parameters in (GraspParameters(), GraspParameters(.08, .2, .004)):
            with self.subTest(parameters=parameters):
                sim = self.make_sim()
                physics = [array.copy() for array in (sim.model.geom_size, sim.model.geom_friction,
                           sim.model.body_mass, sim.model.actuator_forcerange)]
                sim.start_task("store_bowl", parameters)
                phases = self.run_until(sim)
                self.assertEqual(phases, list(TASKS["store_bowl"].phases) + ["done"])
                self.assert_stored(sim)
                self.assertGreaterEqual(sim.skill.hold_elapsed, 1.0)
                self.assertEqual(list(sim.skill.phase_times), list(TASKS["store_bowl"].phases))
                self.assertAlmostEqual(sum(sim.skill.phase_times.values()), sim.skill.state()["elapsed_s"], places=7)
                sim.step(5.0)
                self.assert_stored(sim)
                self.assertGreaterEqual(sim.skill.storage.verification_hold_s, 5.99)
                for actual, before in zip((sim.model.geom_size, sim.model.geom_friction,
                                          sim.model.body_mass, sim.model.actuator_forcerange), physics):
                    np.testing.assert_array_equal(actual, before)

    def test_reset_cancels_storage_and_clears_contact_and_verification_state(self):
        sim = self.make_sim()
        sim.start_task("store_bowl")
        self.run_until(sim, "push_drawer")
        self.assertEqual(sim.skill.phase, "push_drawer")
        self.assertTrue(sim.skill.storage.released)
        sim.reset()
        self.assertEqual(sim.skill.status, "idle")
        self.assertIsNone(sim.skill.storage)
        self.assertEqual(sim.state()["collisions"]["episodes"], 0)
        self.assertAlmostEqual(sim.data.qpos[sim.model.joint("cabinet_bottom_level").qposadr[0]], -.16)
        self.assertTrue(sim.state()["drawer"]["fully_open"])
        np.testing.assert_array_equal(sim.data.qpos[sim.arm_qpos], ARM_HOME)
        sim.start_task("store_bowl")
        self.assertEqual(sim.skill.phase, "prepare")
        self.assertFalse(sim.skill.storage.released)
        self.assertEqual(sim.skill.storage.verification_hold_s, 0)

    def test_idle_drawer_truth_distinguishes_full_open_partial_closed_and_moving(self):
        cases = (
            (-0.16, 0.0, True, False),
            (-0.155, 0.0, True, False),
            (-0.155 + 1e-6, 0.0, False, False),
            (-0.003, 0.0, False, True),
            (-0.003 - 1e-6, 0.0, False, False),
            (0.0, 0.005, False, False),
            (0.0, 0.005 - 1e-6, False, True),
        )
        for position, velocity, fully_open, closed in cases:
            with self.subTest(position=position, velocity=velocity):
                sim = self.make_sim()
                joint = sim.model.joint("cabinet_bottom_level")
                # Initial-condition fixtures, never a successful-motion shortcut.
                sim.data.qpos[joint.qposadr[0]] = position
                sim.data.qvel[joint.dofadr[0]] = velocity
                mujoco.mj_forward(sim.model, sim.data)
                before = deepcopy(sim.state())
                self.assertIsNone(before["skill"]["storage"])
                self.assertEqual(before["drawer"]["position_m"], position)
                self.assertEqual(before["drawer"]["velocity_m_s"], velocity)
                self.assertIs(before["drawer"]["fully_open"], fully_open)
                self.assertIs(before["drawer"]["closed"], closed)
                self.assertIs(before["goal_checks"]["store_and_home"]["drawer_closed"], closed)
                if fully_open:
                    sim.start_task("store_bowl")
                    self.assertEqual(sim.skill.status, "running")
                else:
                    with self.assertRaises(ValueError):
                        sim.start_task("store_bowl")
                    self.assertEqual(sim.state(), before)

    def test_invalid_task_target_opening_and_closed_drawer_refuse_before_motion(self):
        sim = self.make_sim()
        for task in (None, "unknown", [], 1):
            before = deepcopy(sim.state())
            with self.assertRaises(ValueError):
                sim.start_task(task)
            self.assertEqual(sim.state(), before)
        before = deepcopy(sim.state())
        with self.assertRaises(ValueError):
            sim.start_task("store_bowl", GraspParameters(initial_opening_m=.04))
        self.assertEqual(sim.state(), before)
        drawer = sim.model.joint("cabinet_bottom_level").qposadr[0]
        sim.data.qpos[drawer] = 0.0  # Deliberately invalid initial condition, not controller motion.
        mujoco.mj_forward(sim.model, sim.data)
        qpos, ctrl = sim.data.qpos.copy(), sim.data.ctrl.copy()
        with self.assertRaises(ValueError):
            sim.start_task("store_bowl")
        np.testing.assert_array_equal(sim.data.qpos, qpos)
        np.testing.assert_array_equal(sim.data.ctrl, ctrl)
        self.assertEqual(sim.skill.status, "idle")
        bottle = self.make_sim("bottle")
        with self.assertRaises(ValueError):
            bottle.start_task("store_bowl")
        self.assertEqual(bottle.skill.status, "idle")

    def test_impossible_original_region_fit_is_rejected(self):
        sim = self.make_sim()
        sim.model.site_size[sim.model.site("cabinet_bottom_region").id] *= .5
        before = deepcopy(sim.state())
        with self.assertRaisesRegex(ValueError, "does not fit"):
            sim.start_task("store_bowl")
        self.assertEqual(sim.state(), before)

    def test_external_drawer_disturbance_revokes_success_and_requires_reset(self):
        sim = self.make_sim()
        sim.start_task("store_bowl")
        self.run_until(sim)
        self.assert_stored(sim)
        sim.data.xfrc_applied[sim.skill.storage.drawer, 1] = -30.0
        sim.step(float(sim.model.opt.timestep))
        self.assertFalse(sim.skill.storage.state()["drawer_closed"])
        self.assertFalse(sim.state()["drawer"]["closed"])
        self.assertEqual(sim.skill.status, "failed")
        self.assertIsNotNone(sim.skill.failure_reason)
        with self.assertRaises(ValueError):
            sim.start_task("store_bowl")
        sim.reset()
        self.assertEqual(sim.skill.status, "idle")

    def test_only_explicit_floor_and_handle_geometries_are_planned_contacts(self):
        sim = self.make_sim()
        sim.start_task("store_bowl")
        storage = sim.skill.storage
        bowl_geom = int(storage.bowl_geoms[0])
        finger_geom = next(iter(storage.finger_geoms))
        hand = sim.model.body("robot_hand").id
        hand_geom = int(np.flatnonzero(sim.model.geom_bodyid == hand)[0])
        for phase in TASKS["store_bowl"].phases:
            self.assertFalse(storage.planned_contact(storage.handle, hand_geom, phase))
            self.assertFalse(storage.planned_contact(storage.floor, finger_geom, phase))
        self.assertFalse(storage.planned_contact(storage.floor, bowl_geom, "transfer"))
        self.assertTrue(storage.planned_contact(storage.floor, bowl_geom, "place"))
        self.assertTrue(storage.planned_contact(storage.floor, bowl_geom, "done"))
        self.assertFalse(storage.planned_contact(storage.handle, finger_geom, "push_approach"))
        self.assertTrue(storage.planned_contact(storage.handle, finger_geom, "push_drawer"))
        for finger in sim.skill.fingers:
            self.assertTrue(sim.collisions._unwanted(sim.skill.target_body, finger, allow_grip=False))


if __name__ == "__main__":
    unittest.main()
