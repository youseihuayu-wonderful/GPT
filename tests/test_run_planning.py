"""Failure evidence regressions; model responses here are explicit test fixtures."""

from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from grasp import GraspParameters
from planner import DecisionResult, PlannerConfig, PlannerError, SkillDecision
from planning import PlanningSession
from simulation import Simulation
import run_planning


class PlanningEvidenceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.output = Path(directory.name) / "evidence.json"
        config = patch("planning.PlannerConfig.load", return_value=PlannerConfig("fixture-codex"))
        config.start()
        self.addCleanup(config.stop)
        provider = patch("planning.request_decision", new_callable=AsyncMock)
        self.provider = provider.start()
        self.addCleanup(provider.stop)
        self.provider.return_value = DecisionResult(
            SkillDecision("complete", "Fixture home", "Measured initial home", "home", "bowl", None, None),
            "fixture-model", "fixture-thread", 0.01,
        )
        hold = patch("run_planning.VALIDATION_HOLD_S", 0.04)
        hold.start()
        self.addCleanup(hold.stop)

    async def test_cleanup_failures_cannot_leave_successful_evidence(self):
        original_session_close = PlanningSession.close
        original_sim_close = Simulation.close

        async def failed_session_close(session):
            await original_session_close(session)
            raise RuntimeError("fixture provider cleanup failed")

        def failed_sim_close(sim):
            original_sim_close(sim)
            raise LookupError("fixture renderer cleanup failed")

        with patch.object(PlanningSession, "close", failed_session_close), patch.object(Simulation, "close", failed_sim_close):
            await run_planning.run_planning("空夹爪归位", self.output)
        saved = json.loads(self.output.read_text())
        self.assertFalse(saved["success"])
        self.assertEqual(saved["error"]["stage"], "session_cleanup")
        self.assertEqual(saved["secondary_errors"][0]["stage"], "simulation_cleanup")
        self.assertEqual(saved["validation"]["status"], "succeeded")

    async def test_interrupted_skill_is_saved_after_session_becomes_terminal(self):
        self.provider.return_value = DecisionResult(
            SkillDecision("execute", "Fixture grasp", "Test interruption", "hold_target", "bowl", "grasp_and_hold", GraspParameters()),
            "fixture-model", "fixture-thread", 0.01,
        )
        original_step = Simulation.step

        def fail_active(sim, duration):
            if sim.skill.status == "running":
                raise RuntimeError("fixture physics operation failed")
            original_step(sim, duration)

        with patch.object(Simulation, "step", fail_active):
            await run_planning.run_planning("抓起黑碗并保持", self.output)
        saved = json.loads(self.output.read_text())
        self.assertEqual(saved["error"]["message"], "fixture physics operation failed")
        self.assertEqual(saved["planning"]["status"], "cancelled")
        self.assertEqual(saved["history"][-1]["status"], "cancelled")
        self.assertIsNotNone(saved["history"][-1]["execution_wall_s"])
        self.assertFalse(saved["success"])

    async def test_initial_grasp_error_survives_failed_observation(self):
        original_step, original_state = Simulation.step, Simulation.state
        broken = False

        def fail_active(sim, duration):
            nonlocal broken
            if sim.skill.status == "running":
                broken = True
                raise RuntimeError("fixture primary physics failure")
            original_step(sim, duration)

        def fail_observation(sim):
            if broken:
                raise ValueError("fixture secondary observation failure")
            return original_state(sim)

        with patch.object(Simulation, "step", fail_active), patch.object(Simulation, "state", fail_observation):
            await run_planning.run_planning("收纳黑碗", self.output, start_held=True)
        saved = json.loads(self.output.read_text())
        self.assertEqual(saved["error"]["message"], "fixture primary physics failure")
        self.assertEqual(saved["initial_grasp"]["result_observation_error"]["type"], "ValueError")
        self.assertIsNotNone(saved["initial_grasp"]["execution_wall_s"])
        self.assertIsNone(saved["final_state"])
        self.provider.assert_not_awaited()

    async def test_final_write_failure_still_reports_provider_error(self):
        self.provider.side_effect = PlannerError("fixture primary provider failure")
        original_write = run_planning.write_report

        def fail_final_write(path, report):
            if report["status"] == "failed":
                raise OSError("fixture disk full")
            original_write(path, report)

        errors = io.StringIO()
        with patch("run_planning.write_report", fail_final_write), redirect_stderr(errors):
            with self.assertRaises(OSError):
                await run_planning.run_planning("空夹爪归位", self.output)
        self.assertIn("fixture primary provider failure", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
