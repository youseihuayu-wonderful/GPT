"""Error-boundary tests; synthetic study rows are confined to temporary unit fixtures."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from grasp import GraspParameters
from grasp_trial import run_trial
import run_grasp_learning as study
from scenarios import DEFAULT_CASE
from simulation import Simulation


class TrialEvidenceSafetyTests(unittest.TestCase):
    def test_internal_ik_errors_propagate_instead_of_becoming_negative_labels(self):
        for error in (np.linalg.LinAlgError("injected solver failure"),
                      ValueError("injected unexpected shape error")):
            with self.subTest(error=type(error).__name__), patch("grasp.np.linalg.solve", side_effect=error):
                with self.assertRaises(type(error)) as caught:
                    run_trial(DEFAULT_CASE, .1)
                self.assertIs(caught.exception, error)

    def test_nonfinite_ik_update_cannot_become_an_unreachable_pose_label(self):
        with patch("grasp.np.linalg.solve", return_value=np.full(6, np.nan)):
            with self.assertRaises(FloatingPointError):
                run_trial(DEFAULT_CASE, .1)

    def test_bounded_nonconvergence_remains_an_explicit_controller_failure(self):
        sim = Simulation(DEFAULT_CASE)
        self.addCleanup(sim.close)
        sim.start_task("grasp", GraspParameters())
        sim.skill._move("approach", np.array([10., 10., 10.]))
        self.assertEqual(sim.skill.status, "failed")

    def test_existing_trace_survives_valid_and_invalid_new_requests(self):
        for speed in (.1, 0.):
            with self.subTest(speed=speed), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "prior.json.gz"
                original = b"previous experiment evidence sentinel"
                path.write_bytes(original)
                with self.assertRaises(FileExistsError):
                    run_trial(DEFAULT_CASE, speed, trace_path=path)
                self.assertEqual(path.read_bytes(), original)


class StudyEvidenceSafetyTests(unittest.TestCase):
    def small_protocol(self):
        protocol = study.build_protocol()
        for key in ("training_cases", "evaluation_cases"):
            protocol[key] = [next(case for case in protocol[key] if case["target_object"] == target)
                             for target in ("bowl", "bottle")]
        protocol["planned_training_trials"] = protocol["planned_evaluation_trials"] = 6
        return protocol

    @staticmethod
    def synthetic_trial(case, speed, *, trace_path):
        invalid = case.position_label == "evaluation" and case.target_object == "bottle"
        success = None if invalid else case.target_object == "bowl"
        return {
            "case": case.to_dict(), "parameters": {"descent_speed_m_s": speed},
            "trial_status": "invalid_initial_state" if invalid else "completed",
            "success": success, "failure_reason": None if success else "unit-only failure",
            "initial_state_fingerprint": case.case_id,
            "verified_hold_s": 2. if success else 0.,
            "postconditions": {"target_lifted": bool(success), "verified_hold": bool(success)},
            "execution_sim_s": 3. if success else 0.,
            "warnings": [], "collisions": {"episodes": 0},
        }

    def test_invalid_recommended_setup_prevents_complete_execution_acceptance(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), patch.object(
            study, "build_protocol", return_value=self.small_protocol(),
        ), patch.object(study, "provenance", return_value={"unit_fixture": True}), patch.object(
            study, "run_trial", side_effect=self.synthetic_trial,
        ):
            report = study.run_study(Path(directory) / "study")
        self.assertFalse(report["level1_completed"])
        self.assertEqual(report["selected_execution"]["successes"], 1)
        self.assertEqual(report["selected_execution"]["invalid_initial_states"], 1)
        self.assertFalse(report["level1_checks"]["all_recommendations_physically_checked"])

    def test_interrupted_attempt_keeps_identity_without_a_fake_negative(self):
        protocol = self.small_protocol()
        first = self.synthetic_trial(study.scene_case(protocol["training_cases"][0]), .04, trace_path=None)
        error = RuntimeError("unit-only internal failure")
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), patch.object(
            study, "build_protocol", return_value=protocol,
        ), patch.object(study, "provenance", return_value={"unit_fixture": True}), patch.object(
            study, "run_trial", side_effect=[first, error],
        ):
            output = Path(directory) / "study"
            with self.assertRaises(RuntimeError) as caught:
                study.run_study(output)
            report = json.loads((output / "summary.json").read_text())
        self.assertIs(caught.exception, error)
        self.assertEqual(report["status"], "aborted")
        self.assertEqual(report.get("attempts_started"), 2)
        self.assertEqual(report["training"]["attempted"], 1)
        self.assertEqual(report["training"]["failures"], 0)
        self.assertEqual(report["active_trial"]["speed_m_s"], .1)
        self.assertEqual(report["active_trial"]["case"]["case_id"], protocol["training_cases"][0]["case_id"])
        self.assertFalse(report["active_trial"]["evidence_recorded"])
        self.assertEqual(report["error"]["trial"], report["active_trial"])


if __name__ == "__main__":
    unittest.main()
