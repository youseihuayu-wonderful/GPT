"""Real-contact regression tests for the five independent local skills.

Run from the project root:
    uv run python -m unittest discover -s tests -p 'test_skills.py' -v

Add a concrete physical case as a test method, using a fresh nominal scene and
accepted_call/refused_call to check exact API-boundary immutability. Successful
scenarios never reset or teleport. External-force fault injection is explicitly
labelled and only used to verify that stale physical success is revoked.
Observation helpers are shared with verify_skills.py, not controller predicates.
"""

from copy import deepcopy
import math
import unittest

import mujoco
import numpy as np

from grasp import GraspParameters
from scenarios import SceneCase
from simulation import Simulation
from verify_skills import (
    SKILL_CHAIN, model_fingerprint, physical_observation, physics_fingerprints,
    physics_snapshot,
)


class IndependentSkillTests(unittest.TestCase):
    def make_sim(self, target="bowl"):
        sim = Simulation(SceneCase(f"{target}.nominal", target, "nominal", 0, (0.0, 0.0), 0.0))
        self.addCleanup(sim.close)
        sim.step(1.0)
        self.assertGreater(physical_observation(sim)["target_table_force_n"], 0.01)
        return sim

    def accepted_call(self, sim, name, parameters=None):
        before = physics_snapshot(sim)
        model = model_fingerprint(sim)
        state = deepcopy(sim.state())
        spec = next(item for item in state["available_skills"] if item["name"] == name)
        self.assertTrue(spec["available"], spec)
        self.assertIsNone(spec["unavailable_reason"])
        sim.start_skill(name, parameters)
        self.assertEqual(physics_snapshot(sim), before, "An accepted call must not reset or move physics")
        self.assertEqual(model_fingerprint(sim), model)
        after = sim.state()
        self.assertEqual(after["collisions"], state["collisions"])
        self.assertEqual(after["skill"]["invocation_id"], state["skill"]["invocation_id"] + 1)
        self.assertEqual(after["skill"]["execution_mode"], "skill")
        self.assertEqual(after["skill"]["skill_name"], name)
        self.assertIsNone(after["skill"]["task_kind"])

    def refused_call(self, sim, name, parameters=None, task=False):
        before = deepcopy(sim.state())
        physics = physics_snapshot(sim)
        model = model_fingerprint(sim)
        with self.assertRaises(ValueError) as refusal:
            if task:
                sim.start_task(name, parameters)
            else:
                sim.start_skill(name, parameters)
        self.assertTrue(str(refusal.exception).strip(), "Refusals must give an actionable reason")
        self.assertEqual(sim.state(), before)
        self.assertEqual(physics_snapshot(sim), physics)
        self.assertEqual(model_fingerprint(sim), model)

    def assert_unavailable(self, sim, names):
        catalog = {item["name"]: item for item in sim.state()["available_skills"]}
        for name in names:
            with self.subTest(unavailable_skill=name):
                self.assertFalse(catalog[name]["available"], catalog[name])
                self.assertTrue(catalog[name]["unavailable_reason"], catalog[name])

    def run_until_terminal(self, sim, table_reference=None):
        state = sim.state()
        name = state["skill"]["skill_name"]
        spec = next(item for item in state["available_skills"] if item["name"] == name)
        watchdog = float(spec["watchdog_s"])
        self.assertGreater(watchdog, 0.0)
        self.assertLessEqual(watchdog, 120.0)
        model = physics_fingerprints(sim)
        seen = []
        peak_handle_force = 0.0
        for _ in range(math.ceil(watchdog / 0.02) + 1):
            state = sim.state()
            if not seen or seen[-1] != state["skill"]["phase"]:
                seen.append(state["skill"]["phase"])
            observed = physical_observation(sim)
            peak_handle_force = max(peak_handle_force, observed["finger_handle_force_n"])
            if table_reference is not None:
                self.assert_table_unchanged(observed, table_reference)
            if state["skill"]["status"] != "running":
                break
            before = float(sim.data.time)
            sim.step(0.02)
            self.assertAlmostEqual(float(sim.data.time) - before, 0.02, places=8)
            self.assertEqual(physics_fingerprints(sim), model)
        self.assert_success(sim, name)
        return seen, peak_handle_force

    def assert_success(self, sim, name):
        state = sim.state()
        skill = state["skill"]
        self.assertEqual(skill["status"], "succeeded", skill)
        self.assertEqual(skill["skill_name"], name)
        for key, value in skill["postconditions"].items():
            self.assertIs(value, True, (key, skill))
        self.assertIsNone(skill["failure_reason"])
        self.assertEqual(state["warnings"], [])
        self.assertEqual(state["collisions"]["episodes"], 0, state["collisions"])

    def hold(self, sim, name, seconds, reference=None, table_reference=None):
        anchor = physical_observation(sim)
        before_time = float(sim.data.time)
        before_model = model_fingerprint(sim)
        for _ in range(round(seconds / 0.02)):
            sim.step(0.02)
            self.assert_success(sim, name)
            observed = physical_observation(sim)
            if table_reference is not None:
                self.assert_table_unchanged(observed, table_reference)
            if name == "grasp_and_hold":
                self.assertTrue(observed["bilateral_grip"], observed)
                self.assertGreaterEqual(observed["target_position_m"][2] - reference["target_position_m"][2], 0.10)
                self.assertGreaterEqual(observed["target_free_z_m"] - reference["target_free_z_m"], 0.10)
                self.assertLessEqual(math.dist(observed["target_position_m"], anchor["target_position_m"]), 0.01)
        self.assertAlmostEqual(float(sim.data.time) - before_time, seconds, places=8)
        self.assertEqual(model_fingerprint(sim), before_model)

    def assert_table_unchanged(self, observed, reference):
        self.assertGreater(observed["target_table_force_n"], 0.01)
        self.assertLess(math.dist(observed["target_position_m"], reference["target_position_m"]), 0.003)

    def assert_supported_inside(self, sim):
        observed = physical_observation(sim)
        self.assertTrue(observed["target_inside_drawer"], observed)
        self.assertGreater(observed["target_drawer_floor_force_n"], 0.01)
        self.assertTrue(all(force <= 0.01 for force in observed["finger_target_forces_n"].values()))
        self.assertLess(np.linalg.norm(observed["target_velocity"][:3]), 0.01)
        self.assertLess(np.linalg.norm(observed["target_velocity"][3:]), 0.1)

    def test_close_then_home_without_grasp_keeps_bowl_on_table_and_never_resets(self):
        sim = self.make_sim()
        original = physical_observation(sim)
        model = model_fingerprint(sim)
        self.accepted_call(sim, "close_drawer")
        phases, peak_force = self.run_until_terminal(sim, table_reference=original)
        self.assertNotIn("return_home", phases)
        self.assertNotIn("prepare", phases)
        self.assertGreater(peak_force, 0.1, "Drawer closure must use real finger-handle force")
        closed = physical_observation(sim)
        self.assertGreater(closed["drawer_position_m"] - original["drawer_position_m"], 0.15)
        self.assertLessEqual(abs(closed["drawer_position_m"]), 0.003)
        self.assertLess(abs(closed["drawer_speed_m_s"]), 0.005)
        self.assertGreater(closed["arm_home_error_rad"], 0.03, "Close must not automatically home")
        self.assertTrue(sim.state()["skill"]["storage"]["hand_clear_of_drawer"])
        self.hold(sim, "close_drawer", 1.0, table_reference=original)
        self.assert_unavailable(sim, ("close_drawer",))
        self.refused_call(sim, "close_drawer")
        handoff_time = float(sim.data.time)
        self.accepted_call(sim, "return_home")
        phases, _ = self.run_until_terminal(sim, table_reference=original)
        self.assertNotIn("push_drawer", phases)
        self.hold(sim, "return_home", 5.0, table_reference=original)
        final = physical_observation(sim)
        self.assertGreater(float(sim.data.time), handoff_time + 5.0)
        self.assertLessEqual(final["arm_home_error_rad"], 0.03)
        self.assertLess(final["arm_max_speed_rad_s"], 0.03)
        self.assertLess(abs(final["drawer_position_m"] - closed["drawer_position_m"]), 0.003)
        self.assertGreater(final["target_table_force_n"], 0.01)
        self.assertLess(math.dist(final["target_position_m"], original["target_position_m"]), 0.003)
        self.assertEqual(model_fingerprint(sim), model)

    def test_return_home_from_original_empty_hand_does_not_close_open_drawer(self):
        sim = self.make_sim()
        original = physical_observation(sim)
        self.accepted_call(sim, "return_home")
        phases, _ = self.run_until_terminal(sim)
        self.assertNotIn("push_drawer", phases)
        self.hold(sim, "return_home", 1.0)
        observed = physical_observation(sim)
        self.assertLessEqual(observed["arm_home_error_rad"], 0.03)
        self.assertLess(observed["arm_max_speed_rad_s"], 0.03)
        self.assertLessEqual(observed["drawer_position_m"], -0.155)
        self.assertLess(abs(observed["drawer_position_m"] - original["drawer_position_m"]), 0.003)
        self.assertGreater(observed["target_table_force_n"], 0.01)

    def test_overlap_chain_holds_at_each_boundary_and_refuses_closing_over_released_hand(self):
        sim = self.make_sim()
        original = physical_observation(sim)
        model = model_fingerprint(sim)
        self.assert_unavailable(sim, ("place_and_release", "withdraw_from_drawer"))
        self.refused_call(sim, "place_and_release")
        self.assertTrue(sim.state()["drawer"]["fully_open"])
        self.refused_call(sim, "withdraw_from_drawer")
        self.accepted_call(sim, "grasp_and_hold", GraspParameters(0.08, 0.20, 0.004))
        self.run_until_terminal(sim)
        self.assertGreater(sim.state()["skill"]["descent_close_overlap_s"], 0.0)
        self.hold(sim, "grasp_and_hold", 5.0, original)
        self.assertTrue(sim.state()["drawer"]["fully_open"])
        self.assert_unavailable(sim, ("close_drawer", "return_home"))
        self.refused_call(sim, "close_drawer")
        self.refused_call(sim, "return_home")
        self.refused_call(sim, "grasp", task=True)
        self.accepted_call(sim, "place_and_release")
        phases, _ = self.run_until_terminal(sim)
        self.assertNotIn("withdraw", phases, "Placement must stop before withdrawal")
        self.assert_supported_inside(sim)
        storage = sim.state()["skill"]["storage"]
        self.assertTrue(storage["gripper_clear"])
        self.assertFalse(storage["hand_clear_of_drawer"], "Detachment is not geometric hand clearance")
        self.hold(sim, "place_and_release", 1.0)
        self.assert_unavailable(sim, ("close_drawer", "return_home"))
        self.refused_call(sim, "close_drawer")
        self.refused_call(sim, "return_home")
        self.accepted_call(sim, "withdraw_from_drawer")
        phases, _ = self.run_until_terminal(sim)
        self.assertNotIn("push_drawer", phases)
        self.assertTrue(sim.state()["skill"]["storage"]["hand_clear_of_drawer"])
        self.assert_supported_inside(sim)
        self.hold(sim, "withdraw_from_drawer", 1.0)
        self.assertLessEqual(physical_observation(sim)["drawer_position_m"], -0.155)
        self.accepted_call(sim, "close_drawer")
        _, peak_force = self.run_until_terminal(sim)
        self.assertGreater(peak_force, 0.1)
        self.hold(sim, "close_drawer", 1.0)
        self.accepted_call(sim, "return_home")
        self.run_until_terminal(sim)
        self.hold(sim, "return_home", 5.0)
        self.assert_supported_inside(sim)
        final = physical_observation(sim)
        self.assertLessEqual(abs(final["drawer_position_m"]), 0.003)
        self.assertLessEqual(final["arm_home_error_rad"], 0.03)
        self.assertLess(final["arm_max_speed_rad_s"], 0.03)
        self.assertEqual(sim.state()["skill"]["invocation_id"], 5)
        self.assertEqual(model_fingerprint(sim), model)

    def test_invalid_parameters_names_and_busy_restarts_preserve_all_state(self):
        sim = self.make_sim()
        for name in (None, "", "unknown", [], 1):
            with self.subTest(invalid_name=name):
                self.refused_call(sim, name)
        for name in SKILL_CHAIN[1:]:
            with self.subTest(unaccepted_parameters=name):
                self.refused_call(sim, name, GraspParameters())
        self.refused_call(sim, "grasp_and_hold", {"initial_opening_m": 0.08})
        self.accepted_call(sim, "grasp_and_hold", GraspParameters(0.08, 0.10, 0.0))
        sim.step(0.10)
        self.assertEqual(sim.state()["skill"]["status"], "running")
        self.assert_unavailable(sim, SKILL_CHAIN)
        for name in SKILL_CHAIN:
            with self.subTest(busy_skill=name):
                self.refused_call(sim, name)
        self.refused_call(sim, "grasp", task=True)
        self.refused_call(sim, "store_bowl", task=True)
        before = float(sim.data.time)
        self.run_until_terminal(sim)
        self.assertGreater(float(sim.data.time), before)
        self.assertEqual(sim.state()["skill"]["invocation_id"], 1)

    def test_external_force_revokes_held_grasp_success_and_blocks_every_restart(self):
        sim = self.make_sim()
        reference = physical_observation(sim)
        self.accepted_call(sim, "grasp_and_hold")
        self.run_until_terminal(sim)
        self.hold(sim, "grasp_and_hold", 1.0, reference)
        before = physical_observation(sim)
        target = sim.model.body("bowl_object").id
        # Deliberate physical fault, not a pose teleport or a successful scenario.
        sim.data.xfrc_applied[target, 0] = 20.0
        try:
            for _ in range(25):
                sim.step(0.02)
                if sim.state()["skill"]["status"] == "failed":
                    break
        finally:
            sim.data.xfrc_applied[target, 0] = 0.0
        after = physical_observation(sim)
        state = sim.state()
        self.assertTrue(not after["bilateral_grip"] or math.dist(before["target_position_m"], after["target_position_m"]) > 0.01)
        self.assertEqual(state["skill"]["status"], "failed", state["skill"])
        self.assertTrue(state["skill"]["failure_reason"])
        self.assertFalse(all(state["skill"]["postconditions"].values()))
        self.assert_unavailable(sim, SKILL_CHAIN)
        for name in SKILL_CHAIN:
            self.refused_call(sim, name)
        self.refused_call(sim, "grasp", task=True)

    def test_reopening_closed_drawer_with_force_invalidates_success_after_integration(self):
        sim = self.make_sim()
        self.accepted_call(sim, "close_drawer")
        self.run_until_terminal(sim)
        self.hold(sim, "close_drawer", 1.0)
        drawer = sim.model.body("cabinet_cabinet_bottom").id
        # Same real drawer-force disturbance as the existing storage regression.
        sim.data.xfrc_applied[drawer, 1] = -30.0
        try:
            sim.step(float(sim.model.opt.timestep))
        finally:
            sim.data.xfrc_applied[drawer, 1] = 0.0
        observed = physical_observation(sim)
        self.assertTrue(abs(observed["drawer_position_m"]) > 0.003 or abs(observed["drawer_speed_m_s"]) >= 0.005)
        state = sim.state()
        self.assertEqual(state["skill"]["status"], "failed", state["skill"])
        self.assertFalse(state["skill"]["storage"]["drawer_closed"])
        self.assertTrue(state["skill"]["failure_reason"])
        self.refused_call(sim, "return_home")
        self.refused_call(sim, "close_drawer")

    def test_bottle_hold_is_physical_but_bowl_placement_is_unavailable(self):
        sim = self.make_sim("bottle")
        reference = physical_observation(sim)
        self.accepted_call(sim, "grasp_and_hold", GraspParameters(0.04, 0.12, 0.0))
        self.run_until_terminal(sim)
        self.hold(sim, "grasp_and_hold", 2.0, reference)
        self.assert_unavailable(sim, ("place_and_release", "withdraw_from_drawer", "close_drawer", "return_home"))
        for name in SKILL_CHAIN[1:]:
            self.refused_call(sim, name)
        self.assert_success(sim, "grasp_and_hold")

    def test_handoff_rechecks_completed_skill_after_external_physics_advance(self):
        sim = self.make_sim()
        self.accepted_call(sim, "close_drawer")
        self.run_until_terminal(sim)
        drawer = sim.model.body("cabinet_cabinet_bottom").id
        sim.data.xfrc_applied[drawer, 1] = -30.0
        try:
            # Another physics owner advances a real step without this
            # controller's terminal check. Admission must inspect live state.
            mujoco.mj_step(sim.model, sim.data)
            mujoco.mj_forward(sim.model, sim.data)
        finally:
            sim.data.xfrc_applied[drawer, 1] = 0.0
        self.assertEqual(sim.skill.status, "succeeded")
        self.assertFalse(sim.skill.postconditions()["drawer_closed"])
        self.refused_call(sim, "return_home")

    def test_shifted_table_bowl_blocks_palm_approach_before_any_motion(self):
        sim = Simulation(SceneCase("bowl.right", "bowl", "right", 0, (.04, 0.0), 0.0))
        self.addCleanup(sim.close)
        sim.step(1.0)
        self.assertGreater(physical_observation(sim)["target_table_force_n"], 0.01)
        self.refused_call(sim, "close_drawer")

    def test_numerical_warning_latches_failure_before_more_motion(self):
        sim = self.make_sim()
        self.accepted_call(sim, "grasp_and_hold")
        sim.data.warning[int(mujoco.mjtWarning.mjWARN_BADQPOS)].number = 1
        before = physics_snapshot(sim)
        with self.assertRaises(RuntimeError):
            sim.step(.02)
        self.assertEqual(sim.skill.status, "failed")
        self.assertEqual(physics_snapshot(sim), before)
        self.refused_call(sim, "return_home")
        # Clearing the warning must not revive a failed numerical trajectory.
        # Only a full reset clears the failure latch.
        sim.data.warning[int(mujoco.mjtWarning.mjWARN_BADQPOS)].number = 0
        with self.assertRaises(RuntimeError):
            sim.step(.02)
        self.assertEqual(physics_snapshot(sim), before)
        sim.reset()
        sim.step(1.0)
        self.accepted_call(sim, "grasp_and_hold")
        self.run_until_terminal(sim)


if __name__ == "__main__":
    unittest.main()
