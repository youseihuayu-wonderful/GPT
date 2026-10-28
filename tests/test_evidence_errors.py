"""Evidence error precedence and accounting, with explicit fault injection.

Run: uv run python -m unittest discover -s tests -p 'test_evidence_errors.py' -v
"""

from asyncio import CancelledError
from contextlib import ExitStack
import unittest
from unittest.mock import patch

from scenarios import SceneCase
from simulation import Simulation
from verify_skills import (
    EvidenceRun, STEP_S, boundary_snapshot, physical_observation, run_scenario,
)


class EvidenceErrorsTests(unittest.TestCase):
    def make_run(self):
        sim = Simulation(SceneCase("bowl.nominal", "bowl", "nominal", 0, (0.0, 0.0), 0.0))
        self.addCleanup(sim.close)
        return EvidenceRun(sim, {})

    def test_execution_and_cancellation_survive_secondary_diagnostics(self):
        for operation in ("invoke", "step", "complete", "hold"):
            for error_type in (RuntimeError, KeyboardInterrupt, CancelledError):
                with self.subTest(operation=operation, error=error_type.__name__):
                    run = self.make_run()
                    primary = error_type("execution stopped")
                    observation_error = ValueError("contact observation unavailable")
                    snapshot_error = LookupError("boundary unavailable")
                    invocation = None
                    if operation in ("complete", "hold"):
                        run.settle()
                        invocation = run.invoke("return_home")
                    pre = boundary_snapshot(run.sim)
                    with ExitStack() as faults:
                        faults.enter_context(patch.object(
                            run.sim, "start_skill" if operation == "invoke" else "step",
                            side_effect=primary,
                        ))
                        faults.enter_context(patch.object(run, "observe", side_effect=observation_error))
                        faults.enter_context(patch(
                            "verify_skills.boundary_snapshot",
                            side_effect=[pre, snapshot_error] if operation == "invoke" else snapshot_error,
                        ))
                        with self.assertRaises(error_type) as raised:
                            if operation == "invoke":
                                run.invoke("return_home")
                            elif operation == "step":
                                run.step("fault")
                            elif operation == "complete":
                                run.complete(invocation)
                            else:
                                run.hold(invocation, STEP_S)
                    self.assertIs(raised.exception, primary)
                    if operation == "step":
                        diagnostics = run.record["diagnostic_errors"]
                        self.assertEqual(run.record["physics_step_calls_executed"], 1)
                        self.assertEqual(diagnostics[0]["type"], "ValueError")
                    else:
                        result = run.record["invocations"][-1]
                        if operation == "hold":
                            result = result["hold"]
                        self.assertEqual(result["error"], {
                            "type": error_type.__name__, "message": "execution stopped",
                        })
                        diagnostics = result["diagnostic_errors"]
                        self.assertEqual(diagnostics[0]["type"], "LookupError")
                        if operation == "invoke":
                            self.assertEqual(diagnostics[1]["type"], "ValueError")
                        else:
                            self.assertEqual(run.record["diagnostic_errors"][0]["type"], "ValueError")

    def test_diagnostic_failure_without_execution_error_is_not_suppressed(self):
        run = self.make_run()
        error = ValueError("observation failed after successful physics step")
        with patch.object(run, "observe", side_effect=error):
            with self.assertRaises(ValueError) as raised:
                run.step("observation_failure")
        self.assertIs(raised.exception, error)
        self.assertAlmostEqual(run.sim.data.time, STEP_S)
        self.assertEqual(run.record["diagnostic_errors"][0]["type"], "ValueError")

    def test_scenario_report_keeps_execution_error_and_final_snapshot_failure(self):
        scene = SceneCase("bowl.nominal", "bowl", "nominal", 0, (0.0, 0.0), 0.0)
        record = {"name": "independent_close_home", "scene": scene.to_dict()}
        with patch.object(EvidenceRun, "settle", side_effect=RuntimeError("physics failed")):
            with patch("verify_skills.boundary_snapshot", side_effect=ValueError("snapshot failed")):
                run_scenario(record, None)
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["error"], {"type": "RuntimeError", "message": "physics failed"})
        self.assertEqual(record["final_snapshot_error"], {"type": "ValueError", "message": "snapshot failed"})

    def test_accepted_boundary_owns_timing_and_peak_but_rejections_do_not_reset(self):
        run = self.make_run()
        run.settle()
        run.peak_handle_force = 900.0  # Evidence left by a previous invocation.
        clock = [10.0]
        with patch("verify_skills.perf_counter", side_effect=lambda: clock[0]):
            invocation = run.invoke("return_home")
            accepted_time = invocation["post_call"]["physics"]["time_s"]
            self.assertLess(run.peak_handle_force, 900.0)
            clock[0] = 15.0
            run.step("advance_before_busy_requests")
            measured = physical_observation(run.sim)
            measured["finger_handle_force_n"] = 7.0
            with patch("verify_skills.physical_observation", return_value=measured):
                run.observe("injected_contact_measurement", force=True)
            clock[0] = 20.0
            run.invoke("grasp_and_hold", reject=True)
            self.assertEqual(run.peak_handle_force, 7.0)
            clock[0] = 30.0
            run.complete(invocation)
            self.assertAlmostEqual(invocation["execution_wall_time_s"], 20.0)
            self.assertAlmostEqual(
                invocation["execution_simulation_time_s"], run.sim.data.time - accepted_time,
            )
            self.assertGreater(invocation["execution_simulation_time_s"], STEP_S)
            self.assertEqual(invocation["peak_observed_finger_handle_force_n"], 7.0)
            run.invoke("grasp_and_hold")
            self.assertLess(run.peak_handle_force, 7.0)
            self.assertEqual(invocation["peak_observed_finger_handle_force_n"], 7.0)


if __name__ == "__main__":
    unittest.main()
