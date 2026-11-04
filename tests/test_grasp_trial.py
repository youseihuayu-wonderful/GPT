"""Collector invariants using explicit unit-test-only physics timelines.

Scripted timelines below are not experiment evidence or training data. The
integration test separately executes the original MuJoCo workcell.
"""

import gzip
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import grasp_trial
from scenarios import SceneCase


def case_for(target="bowl"):
    return SceneCase(f"{target}.trial", target, "nominal", 2026, (0.0, 0.0), 0.0)


class ScriptedSkill:
    def __init__(self, sim):
        self.sim = sim
        self.tcp = 0
        self.status = "idle"
        self.phase = "idle"
        self.hold_elapsed = 0.0
        self.reference_z = None
        self.finished_at = None
        self.failure_reason = None

    def precondition_error(self, name):
        return self.sim.initial_error

    def grasp_postconditions(self):
        holding = self.sim.execution_steps >= 3
        result = {
            "bilateral_grip": holding,
            "clear_of_table": holding,
            "target_lifted": holding,
            "relative_pose_stable": holding,
            "verified_hold": self.hold_elapsed >= 1.0,
        }
        if self.sim.lost_predicate is not None and self.status == "failed":
            result[self.sim.lost_predicate] = False
        if self.sim.permanently_missing is not None:
            result[self.sim.permanently_missing] = False
        return result


class ScriptedSimulation:
    """Deterministic step-level observations for testing the collector itself."""

    def __init__(self, case, *, initial_error=None, fail_step=None, lost_predicate=None,
                 fault_step=None, numerical=False, permanently_missing=None,
                 never_terminal=False, frozen_clock=False):
        self.case = case
        self.initial_error = initial_error
        self.fail_step = fail_step
        self.lost_predicate = lost_predicate
        self.fault_step = fault_step
        self.numerical = numerical
        self.permanently_missing = permanently_missing
        self.never_terminal = never_terminal
        self.frozen_clock = frozen_clock
        self.model = SimpleNamespace(opt=SimpleNamespace(timestep=0.002))
        self.data = SimpleNamespace(
            time=0.0, qpos=np.zeros(7), qvel=np.zeros(6), ctrl=np.zeros(8),
            site_xpos=np.array([[0.1, 0.2, 1.1]]),
        )
        self.skill = ScriptedSkill(self)
        self.collisions = SimpleNamespace(state=lambda: {
            "episodes": 1, "duration_s": 0.002, "peak_force_n": 0.3,
            "pairs": [{"body_a": "robot_hand", "body_b": "cabinet"}],
            "force_threshold_n": 0.1,
        })
        self.execution_steps = 0
        self.started_at = None
        self.parameters = None
        self.closed = False
        self.error = RuntimeError("test-only internal execution error")
        self._numerical_error = None

    def numerical_error(self):
        return self._numerical_error

    def warning_names(self):
        return []

    def start_task(self, kind, parameters):
        if kind != "grasp":
            raise AssertionError("Only grasp is in scope")
        self.parameters = parameters
        self.started_at = self.data.time
        self.skill.status = "running"
        self.skill.phase = "lift"
        self.skill.reference_z = 0.8

    def step(self, duration):
        if not self.frozen_clock:
            self.data.time = round(self.data.time + duration, 9)
        if self.started_at is None:
            return
        self.execution_steps += 1
        if self.execution_steps == self.fault_step:
            if not self.numerical:
                raise self.error
            self._numerical_error = "MuJoCo state is invalid: nonfinite=['qvel']"
            self.skill.status = "failed"
            self.skill.phase = "failed"
            self.skill.failure_reason = self._numerical_error
            self.skill.finished_at = self.data.time
            return
        if self.execution_steps >= 3:
            self.skill.phase = "hold"
            self.skill.hold_elapsed = min((self.execution_steps - 3) * duration, 1.0)
        if self.execution_steps >= 503 and not self.never_terminal:
            if self.skill.status != "succeeded":
                self.skill.finished_at = self.data.time
            self.skill.status = "succeeded"
            self.skill.phase = "done"
        if self.fail_step is not None and self.execution_steps >= self.fail_step:
            self.skill.status = "failed"
            self.skill.phase = "failed"
            self.skill.finished_at = self.data.time
            self.skill.failure_reason = "Physical postconditions lost after completion"
            self.skill.hold_elapsed = 0.0

    def get_object_pose(self, name):
        return {
            "name": name,
            "position_m": [0.0, 0.0, 0.95 if self.execution_steps >= 3 else 0.8],
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        }

    def close(self):
        self.closed = True


class GraspTrialTests(unittest.TestCase):
    def run_script(self, *, target="bowl", speed=0.1, trace_path=None, **settings):
        sim = ScriptedSimulation(case_for(target), **settings)
        with patch.object(grasp_trial, "Simulation", return_value=sim):
            result = grasp_trial.run_trial(sim.case, speed, trace_path=trace_path)
        self.assertTrue(sim.closed)
        return result, sim

    def test_legacy_one_second_success_is_not_a_two_second_label(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json.gz"
            row, sim = self.run_script(trace_path=path)
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                trace = json.load(stream)
        self.assertTrue(row["success"])
        self.assertEqual(row["trial_status"], "completed")
        self.assertGreaterEqual(row["verified_hold_s"], 2.0)
        # Hold begins at execution step 3, not at the start of its interval.
        self.assertGreaterEqual(row["execution_sim_s"], 3 * sim.model.opt.timestep + 2.0)
        self.assertAlmostEqual(row["controller_execution_sim_s"], 503 * sim.model.opt.timestep)
        early_success = [sample for sample in trace["samples"]
                         if sample["controller_status"] == "succeeded" and sample["verified_hold_s"] < 2.0]
        self.assertTrue(early_success)
        self.assertEqual(row["required_hold_s"], 2.0)
        self.assertTrue(all(row["postconditions"].values()))
        self.assertEqual(row["collisions"]["episodes"], 1)  # Diagnostics do not silently relabel.
        json.dumps(row, allow_nan=False)

    def test_failure_between_trace_samples_is_not_missed(self):
        row, sim = self.run_script(fail_step=504, lost_predicate="bilateral_grip")
        self.assertFalse(row["success"])
        self.assertEqual(row["trial_status"], "completed")
        self.assertEqual(row["controller_status"], "failed")
        self.assertEqual(row["failure_reason"], sim.skill.failure_reason)
        self.assertEqual(row["verified_hold_s"], 0.0)
        self.assertAlmostEqual(row["controller_execution_sim_s"], 1.006)
        self.assertAlmostEqual(row["execution_sim_s"], 1.008)
        self.assertFalse(row["postconditions"]["verified_hold"])

    def test_every_physical_hold_predicate_can_prevent_success(self):
        for predicate in ("bilateral_grip", "clear_of_table", "target_lifted", "relative_pose_stable"):
            with self.subTest(predicate=predicate):
                row, _ = self.run_script(fail_step=900, lost_predicate=predicate)
                self.assertFalse(row["success"])
                self.assertFalse(row["postconditions"][predicate])
                self.assertEqual(row["verified_hold_s"], 0.0)

    def test_unsubstantiated_controller_success_is_an_error_not_a_label(self):
        sim = ScriptedSimulation(case_for(), permanently_missing="target_lifted")
        with patch.object(grasp_trial, "Simulation", return_value=sim), patch.object(
            grasp_trial, "TASKS", {"grasp": SimpleNamespace(watchdog_s=0.1)},
        ):
            with self.assertRaises(RuntimeError):
                grasp_trial.run_trial(sim.case, 0.1)
        self.assertTrue(sim.closed)

    def test_invalid_initial_support_is_unlabelled_and_never_executes(self):
        row, sim = self.run_script(initial_error="Grasp requires a target stably supported on the table")
        self.assertEqual(row["trial_status"], "invalid_initial_state")
        self.assertIsNone(row["success"])
        self.assertIsNone(row["controller_execution_sim_s"])
        self.assertEqual(row["execution_sim_s"], 0.0)
        self.assertIsNone(sim.started_at)
        self.assertEqual(row["initial_pose"], row["final_pose"])


    def test_numerical_fault_cannot_become_a_negative_training_label(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "numerical.json.gz"
            sim = ScriptedSimulation(case_for(), fault_step=600, numerical=True)
            with patch.object(grasp_trial, "Simulation", return_value=sim):
                with self.assertRaises(RuntimeError):
                    grasp_trial.run_trial(sim.case, 0.1, trace_path=path)
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                trace = json.load(stream)
        self.assertTrue(sim.closed)
        self.assertIsNone(trace["outcome"])
        self.assertEqual(trace["error"]["type"], "RuntimeError")
        self.assertTrue(trace["samples"])
        self.assertTrue(any(sample["controller_status"] == "succeeded" for sample in trace["samples"]))

    def test_internal_error_is_preserved_even_when_trace_write_fails(self):
        sim = ScriptedSimulation(case_for(), fault_step=10)
        with patch.object(grasp_trial, "Simulation", return_value=sim), patch.object(
            grasp_trial, "_save_trace", side_effect=OSError("trace disk unavailable"),
        ):
            with self.assertRaises(RuntimeError) as caught:
                grasp_trial.run_trial(sim.case, 0.1, trace_path=Path("unused-test-trace.json.gz"))
        self.assertIs(caught.exception, sim.error)
        self.assertTrue(sim.closed)
        self.assertTrue(any("trace disk unavailable" in note for note in caught.exception.__notes__))

    def test_nonadvancing_physics_clock_raises_instead_of_hanging(self):
        sim = ScriptedSimulation(case_for(), frozen_clock=True)
        with patch.object(grasp_trial, "Simulation", return_value=sim):
            with self.assertRaises(RuntimeError):
                grasp_trial.run_trial(sim.case, 0.1)
        self.assertTrue(sim.closed)

    def test_invalid_speed_is_rejected_before_scene_construction(self):
        for speed in (True, 0.0, 0.21, float("nan"), float("inf")):
            with self.subTest(speed=speed), patch.object(grasp_trial, "Simulation") as simulation:
                with self.assertRaises(ValueError):
                    grasp_trial.run_trial(case_for(), speed)
                simulation.assert_not_called()


class GraspTrialPhysicsTests(unittest.TestCase):
    def test_original_workcell_outcomes_are_auditable_for_both_targets(self):
        for target in ("bowl", "bottle"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "physical.json.gz"
                row = grasp_trial.run_trial(case_for(target), 0.1, trace_path=path)
                self.assertEqual(row["trial_status"], "completed")
                self.assertIs(type(row["success"]), bool)
                self.assertEqual(row["warnings"], [])
                with gzip.open(path, "rt", encoding="utf-8") as stream:
                    trace = json.load(stream)
                self.assertTrue(row["success"])
                self.assertEqual(row["controller_status"], "succeeded")
                self.assertGreaterEqual(row["verified_hold_s"], 2.0)
                self.assertTrue(all(row["postconditions"].values()))
                execution = [sample for sample in trace["samples"] if sample["stage"] == "execution"]
                hold = [sample for sample in execution if sample["phase"] in ("hold", "done")]
                self.assertGreaterEqual(hold[-1]["time_s"] - hold[0]["time_s"], 2.0)
                self.assertTrue(all(all(sample["postconditions"][key] for key in (
                    "bilateral_grip", "clear_of_table", "target_lifted", "relative_pose_stable",
                )) for sample in hold))
                json.dumps(row, allow_nan=False)

    def test_real_motion_failure_is_a_negative_not_a_success_label(self):
        case = SceneCase("bottle.offset_failure", "bottle", "corner", 2026, (-.08, -.08), -.2)
        row = grasp_trial.run_trial(case, .2)
        self.assertEqual(row["trial_status"], "completed")
        self.assertFalse(row["success"])
        self.assertEqual(row["controller_status"], "failed")
        self.assertLess(row["verified_hold_s"], 2.0)
        self.assertEqual(row["warnings"], [])

    def test_faster_command_shortens_actual_nominal_grasp(self):
        slow = grasp_trial.run_trial(case_for(), .04)
        fast = grasp_trial.run_trial(case_for(), .2)
        self.assertTrue(slow["success"])
        self.assertTrue(fast["success"])
        self.assertLess(fast["execution_sim_s"], slow["execution_sim_s"])
        self.assertEqual(slow["initial_state_fingerprint"], fast["initial_state_fingerprint"])


if __name__ == "__main__":
    unittest.main()
