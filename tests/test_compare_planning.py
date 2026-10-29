"""Planner-comparison boundaries; fixture model replies are not performance evidence."""

from contextlib import redirect_stderr
from copy import deepcopy
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import compare_planning as comparison
from planner import PlannerConfig, PlannerError, validate_decision
from simulation import Simulation
from task_specs import SKILLS


def decision(**changes):
    payload = {
        "decision": "execute", "summary": "Regression fixture", "reason": "Regression fixture",
        "goal": "store_and_home", "target_object": "bowl", "skill": "grasp_and_hold",
        "parameters": comparison.FROZEN_PARAMETERS.to_dict(),
    }
    payload.update(changes)
    return validate_decision(payload)


class ConstraintTests(unittest.TestCase):
    def test_parameter_changes_and_goal_substitution_are_not_silently_corrected(self):
        altered = comparison.FROZEN_PARAMETERS.to_dict()
        altered["descent_speed_m_s"] = 0.1
        proposal = decision(parameters=altered)
        with self.assertRaises((PlannerError, ValueError)):
            comparison.enforce_constraints(proposal)
        self.assertEqual(proposal.parameters.descent_speed_m_s, 0.1)
        with self.assertRaises((PlannerError, ValueError)):
            comparison.enforce_constraints(decision(goal="hold_target"))
        with self.assertRaises((PlannerError, ValueError)):
            comparison.enforce_constraints(decision(
                decision="complete", goal="home", skill=None, parameters=None,
            ))

    def test_pairing_mismatch_aborts_instead_of_comparing_different_initial_states(self):
        first = {"physics": "original", "observation": "original", "model": "original"}
        fingerprints = {}
        comparison.verify_pairing("held", first, fingerprints)
        for field in first:
            changed = first | {field: "changed"}
            with self.subTest(field=field), self.assertRaises(comparison.PairingError):
                comparison.verify_pairing("held", changed, fingerprints)


class PhysicalRuleTests(unittest.TestCase):
    def setUp(self):
        self.sim = Simulation()
        self.addCleanup(self.sim.close)
        self.sim.step(0.5)

    def finish(self, name):
        parameters = comparison.FROZEN_PARAMETERS if name == "grasp_and_hold" else None
        self.sim.start_skill(name, parameters)
        deadline = float(self.sim.data.time) + SKILLS[name].watchdog_s + 0.02
        while self.sim.skill.status == "running" and self.sim.data.time <= deadline:
            self.sim.step(0.02)
        self.assertEqual(self.sim.skill.status, "succeeded", self.sim.skill.state())
        self.assertTrue(all(self.sim.skill.state()["postconditions"].values()))

    def test_rule_resumes_real_held_and_placed_states_then_recognizes_completion(self):
        self.finish("grasp_and_hold")
        for required in ("place_and_release", "withdraw_from_drawer", "close_drawer", "return_home"):
            scene = self.sim.state()
            proposal = comparison.rule_decision(scene)
            self.assertEqual(proposal.decision, "execute")
            self.assertEqual(proposal.skill, required)
            self.assertFalse(comparison.goal_satisfied(scene))
            self.finish(proposal.skill)
        final = self.sim.state()
        self.assertTrue(comparison.goal_satisfied(final))
        self.assertEqual(comparison.rule_decision(final).decision, "complete")
        incomplete = deepcopy(final)
        del incomplete["goal_checks"]["store_and_home"]["bowl_inside_drawer"]
        self.assertFalse(comparison.goal_satisfied(incomplete))
        unverified = deepcopy(final)
        unverified["goal_checks"]["store_and_home"]["bowl_inside_drawer"] = 1
        self.assertFalse(comparison.goal_satisfied(unverified))
        self.assertEqual(final["collisions"]["episodes"], 0)

    def test_rule_refuses_closed_empty_drawer_even_though_grasp_is_available(self):
        self.finish("close_drawer")
        self.finish("return_home")
        scene = self.sim.state()
        self.assertTrue(next(item["available"] for item in scene["available_skills"]
                             if item["name"] == "grasp_and_hold"))
        self.assertFalse(comparison.goal_satisfied(scene))
        self.assertEqual(comparison.rule_decision(scene).decision, "unsupported")


    def test_independent_physical_resets_have_exactly_pairable_initial_conditions(self):
        other = Simulation()
        self.addCleanup(other.close)
        other.step(0.5)
        fingerprints = {}
        for sim in (self.sim, other):
            comparison.verify_pairing("table", {
                "qpos_qvel_ctrl": comparison.initial_state_fingerprint(sim),
                "model": comparison.model_fingerprint(sim),
                "controller_observation": comparison.observation_fingerprint(sim.state()),
            }, fingerprints)


class ProviderFailureTests(unittest.IsolatedAsyncioTestCase):
    async def test_provider_outage_is_not_a_successful_safety_refusal(self):
        request = AsyncMock(side_effect=PlannerError("subscription unavailable in test"))
        with tempfile.TemporaryDirectory() as directory, \
                patch("planning.PlannerConfig.load", return_value=PlannerConfig("test-only-codex")), \
                patch("planning.request_decision", request):
            row = await comparison.run_case("closed_empty", "gpt", {}, {}, root=Path(directory))
        self.assertFalse(row["success"])
        self.assertFalse(row["safety_pass"])
        self.assertEqual(row["executed_skill_count"], 0)
        self.assertEqual(row["gpt_request_count"], 1)
        self.assertIn("subscription unavailable", row["failure_reason"])
        self.assertTrue(row["initial_observation"]["goal_checks"]["close_and_home"]["drawer_closed"])

    async def test_infrastructure_failure_aborts_without_scoring_the_method(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch("compare_planning.Simulation", side_effect=RuntimeError("setup unavailable")):
            report = await comparison.run_comparison(
                ("fixed", "rule"), ("table", "held"), Path(directory) / "evidence.json",
            )
        self.assertEqual(report["status"], "failed")
        self.assertEqual(len(report["runs"]), 1)
        self.assertEqual(report["summary"][0]["goal_attempts"], 0)
        self.assertEqual(report["summary"][0]["infrastructure_failures"], 1)
        self.assertIn("setup unavailable", report["error"]["message"])

    async def test_persistent_save_failure_keeps_primary_experiment_error_on_stderr(self):
        def fail_after_primary(output, report):
            if any(row["error"] is not None for row in report["runs"]):
                raise OSError("secondary disk failure")

        errors = StringIO()
        with tempfile.TemporaryDirectory() as directory, redirect_stderr(errors), \
                patch("compare_planning.Simulation", side_effect=RuntimeError("primary setup failure")), \
                patch("compare_planning.write_report", side_effect=fail_after_primary):
            with self.assertRaisesRegex(OSError, "secondary disk failure"):
                await comparison.run_comparison(("fixed",), ("table",), Path(directory) / "evidence.json")
        self.assertIn("primary setup failure", errors.getvalue())
        self.assertIn("table/fixed", errors.getvalue())


class AccountingTests(unittest.TestCase):
    def test_safety_cases_do_not_inflate_task_success_and_failed_attempt_times_remain(self):
        successful = {
            "case_id": "table", "method": "gpt", "expected_outcome": "success",
            "status": "succeeded", "success": True, "safety_pass": None,
            "executed_skill_count": 1, "decision_count": 2, "gpt_request_count": 2,
            "execution_sim_s": 10.0, "evaluation_sim_s": 19.0,
            "end_to_end_wall_s": 20.0, "gpt_latency_s": 6.0, "validation_sim_s": 5.0,
            "collisions": {"episodes": 0}, "infrastructure_errors": [],
            "history": [{"latency_s": 3.0}, {"latency_s": 3.0}],
        }
        failed = successful | {
            "case_id": "held", "status": "failed", "success": False,
            "executed_skill_count": 0, "decision_count": 1, "gpt_request_count": 1,
            "execution_sim_s": 0.0, "end_to_end_wall_s": 2.0, "gpt_latency_s": 2.0,
            "validation_sim_s": 0.0, "history": [{"latency_s": 2.0}],
        }
        safety = failed | {
            "case_id": "closed_empty", "expected_outcome": "safe_stop", "status": "unsupported",
            "safety_pass": True, "end_to_end_wall_s": 1.0, "gpt_latency_s": 1.0,
            "history": [{"latency_s": 1.0}],
        }
        rows = [successful, failed, safety]
        summary = comparison.summarize_runs(rows, ("gpt",))[0]
        self.assertEqual((summary["goal_attempts"], summary["goal_successes"]), (2, 1))
        self.assertEqual(summary["goal_success_rate"], 0.5)
        self.assertEqual((summary["safety_attempts"], summary["safety_passes"]), (1, 1))
        self.assertEqual(summary["mean_end_to_end_wall_s"], 11.0)
        self.assertEqual(summary["mean_success_end_to_end_wall_s"], 20.0)
        self.assertEqual(summary["mean_gpt_latency_s"], 2.25)
        self.assertFalse(comparison.report_passed({"status": "completed", "runs": rows}))
        invalid = failed | {
            "case_id": "placed", "status": "setup_failed", "end_to_end_wall_s": None,
            "infrastructure_errors": [{"type": "RuntimeError", "message": "setup failure"}],
            "gpt_request_count": 0, "history": [],
        }
        pending = failed | {
            "case_id": "complete", "status": "running", "end_to_end_wall_s": None,
            "history": [{"latency_s": None}],
        }
        unmeasured = failed | {"case_id": "table_right", "end_to_end_wall_s": None}
        partial = comparison.summarize_runs(rows + [invalid, pending, unmeasured], ("gpt",))[0]
        self.assertEqual((partial["goal_attempts"], partial["goal_successes"]), (3, 1))
        self.assertEqual((partial["infrastructure_failures"], partial["pending_runs"]), (1, 1))
        self.assertEqual(partial["timed_goal_attempts"], 2)
        self.assertEqual(partial["mean_end_to_end_wall_s"], 11.0)
        self.assertEqual(partial["timed_gpt_requests"], 5)
        self.assertEqual(partial["gpt_request_count"], 6)

    def test_absent_measurements_and_no_requests_are_not_reported_as_zero_latency(self):
        summary = comparison.summarize_runs([], ("gpt",))[0]
        self.assertEqual(summary["goal_attempts"], 0)
        self.assertIsNone(summary["goal_success_rate"])
        self.assertIsNone(summary["collision_runs"])
        self.assertIsNone(summary["mean_end_to_end_wall_s"])
        self.assertIsNone(summary["mean_gpt_latency_s"])

    def test_cli_identifies_failed_pair_and_reason_instead_of_only_saying_completed(self):
        row = {
            "case_id": "table", "method": "gpt", "expected_outcome": "success",
            "success": False, "safety_pass": None, "infrastructure_errors": [],
            "status": "planning_failed", "failure_reason": "subscription unavailable",
        }
        report = {"status": "completed", "runs": [row], "error": None}
        errors = StringIO()
        with redirect_stderr(errors), patch("sys.argv", ["compare_planning.py"]), \
                patch("compare_planning.run_comparison", new=AsyncMock(return_value=report)):
            self.assertEqual(comparison.main(), 1)
        self.assertIn("table/gpt", errors.getvalue())
        self.assertIn("subscription unavailable", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
