"""Benchmark accounting and safety boundaries; fixtures are not runtime GPT evidence."""

from contextlib import redirect_stderr
from io import StringIO
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import numpy as np

import benchmark
from grasp import GraspParameters
from planner import PlannerConfig, PlannerError, PlanResult, validate_plan
from scenarios import DEFAULT_CASE, benchmark_cases
from simulation import Simulation
from task_specs import TASKS


def result_fixture(target="bowl", decision="plan", task_kind="grasp"):
    plan = validate_plan({
        "decision": decision, "summary": "Test-only plan", "reason": "Test-only reason",
        "task_kind": task_kind if decision == "plan" else None,
        "target_object": target if decision == "plan" else None,
        "steps": [
            {"phase": phase, "instruction": phase}
            for phase in TASKS[task_kind].phases
        ] if decision == "plan" else [],
        "parameters": GraspParameters().to_dict() if decision == "plan" else None,
    })
    return PlanResult(plan, "fixture-not-gpt", "fixture-thread", 1.5)


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class ControllerFixture:
    """Exercise runner state transitions without pretending to measure physical performance."""

    def __init__(self, case, clock, drop=False, never_finish=False):
        self.case = case
        self.clock = clock
        self.drop = drop
        self.never_finish = never_finish
        self.closed = False
        self.started = False
        self.execution_steps = 0
        self.validation_steps = 0
        self.data = SimpleNamespace(time=0.0, qpos=np.array([1.0]), qvel=np.array([0.0]), ctrl=np.array([0.0]))
        self.skill = SimpleNamespace(status="idle")

    def step(self, duration):
        self.data.time += duration
        self.clock.now += duration / 10
        if self.skill.status == "running":
            self.execution_steps += 1
            if self.execution_steps == 2 and not self.never_finish:
                self.skill.status = "succeeded"
        elif self.skill.status == "succeeded":
            self.validation_steps += 1
            if self.drop:
                self.skill.status = "failed"

    def start_task(self, task_kind, parameters):
        if task_kind != "grasp":
            raise AssertionError("Comparison must execute only grasp tasks")
        self.task_kind = task_kind
        self.started = True
        self.parameters = parameters
        self.skill.status = "running"

    def state(self):
        return {
            "target_object": self.case.target_object,
            "scenario": self.case.to_dict(),
            "warnings": [],
            "skill": {"status": self.skill.status, "failure_reason": "lost grip" if self.drop else None},
            "collisions": {
                "episodes": self.validation_steps, "duration_s": self.validation_steps * 0.002,
                "peak_force_n": 2.0 if self.validation_steps else 0.0,
                "pairs": [], "force_threshold_n": 0.1,
            },
        }

    def close(self):
        self.closed = True
        self.clock.now += 0.03


class RunCaseTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.clock = Clock()
        self.simulations = []
        self.drop = False
        self.never_finish = False
        self.config = PlannerConfig("fixture-codex", "fixture-model")
        patches = (
            patch("benchmark.time.monotonic", self.clock),
            patch("benchmark.Simulation", side_effect=self.make_simulation),
            patch("benchmark.PlannerConfig.load", return_value=self.config),
        )
        for replacement in patches:
            replacement.start()
            self.addCleanup(replacement.stop)

    def make_simulation(self, case):
        sim = ControllerFixture(case, self.clock, self.drop, self.never_finish)
        self.simulations.append(sim)
        return sim

    async def test_expected_planning_failure_has_latency_but_no_physical_measurements(self):
        async def fail(*args):
            self.clock.now += 2.0
            raise PlannerError("subscription unavailable")

        with patch("benchmark.request_plan", new=AsyncMock(side_effect=fail)) as request:
            row = await benchmark.run_case(DEFAULT_CASE, "gpt", {}, {"calibration": "only"})
        request.assert_awaited_once()
        self.assertEqual(row["status"], "planning_failed")
        self.assertFalse(row["success"])
        self.assertAlmostEqual(row["planning_wall_s"], 2.0)
        self.assertAlmostEqual(row["gpt_latency_s"], 2.0)
        self.assertAlmostEqual(row["end_to_end_wall_s"], 2.08)
        for field in ("execution_sim_s", "execution_wall_s", "validation_sim_s", "collisions", "parameters"):
            self.assertIsNone(row[field])
        self.assertFalse(self.simulations[0].started)
        self.assertTrue(self.simulations[0].closed)

    async def test_configuration_error_does_not_fabricate_request_latency(self):
        with patch("benchmark.PlannerConfig.load", side_effect=PlannerError("bad config")), \
                patch("benchmark.request_plan", new_callable=AsyncMock) as request:
            row = await benchmark.run_case(DEFAULT_CASE, "gpt", {}, {})
        self.assertEqual(row["status"], "planning_failed")
        self.assertIsNone(row["gpt_latency_s"])
        request.assert_not_awaited()

    async def test_unsupported_wrong_target_and_storage_never_execute_or_use_defaults(self):
        for result in (
            result_fixture(decision="unsupported"), result_fixture(target="bottle"),
            result_fixture(task_kind="store_bowl"),
        ):
            with self.subTest(decision=result.plan.decision, task_kind=result.plan.task_kind), \
                    patch("benchmark.request_plan", new=AsyncMock(return_value=result)):
                row = await benchmark.run_case(DEFAULT_CASE, "gpt", {}, {})
            self.assertEqual(row["status"], "unsupported" if result.plan.decision == "unsupported" else "planning_failed")
            self.assertFalse(self.simulations[-1].started)
            self.assertIsNone(row["parameters"])
            self.assertIsNone(row["collisions"])
            self.assertEqual(row["plan"], result.plan.to_dict())
            summary = benchmark.summarize_runs([row], ("gpt",))[0]
            self.assertEqual(summary["attempts"], 1)
            self.assertEqual(summary["successes"], 0)
            self.assertEqual(summary["planning_failures"], 1)
            self.assertEqual(summary["executed_runs"], 0)

    async def test_success_accounts_for_planning_execution_and_validation_time_domains(self):
        calibration = {"runs": [{"name": "phase-3-only"}]}
        async def respond(config, task, scene, experiments):
            self.clock.now += 1.5
            return result_fixture()

        with patch("benchmark.request_plan", new=AsyncMock(side_effect=respond)) as request:
            row = await benchmark.run_case(DEFAULT_CASE, "gpt", {}, calibration)
        request.assert_awaited_once()
        self.assertTrue(row["success"])
        self.assertAlmostEqual(row["execution_sim_s"], 0.04)
        self.assertAlmostEqual(row["execution_wall_s"], 0.004)
        self.assertAlmostEqual(row["validation_sim_s"], 5.0)
        self.assertAlmostEqual(row["end_to_end_wall_s"], 0.05 + 1.5 + 0.004 + 0.5 + 0.03)
        self.assertAlmostEqual(row["planning_wall_s"], 1.5)

    async def test_post_success_drop_revokes_success_and_retains_validation_collisions(self):
        self.drop = True
        row = await benchmark.run_case(DEFAULT_CASE, "fixed", {}, {})
        self.assertEqual(row["status"], "failed")
        self.assertFalse(row["success"])
        self.assertEqual(row["failure_reason"], "lost grip")
        self.assertAlmostEqual(row["execution_sim_s"], 0.04)
        self.assertAlmostEqual(row["validation_sim_s"], 0.02)
        self.assertEqual(row["collisions"]["episodes"], 1)
        self.assertIsNone(row["gpt_latency_s"])

    async def test_pairing_mismatch_aborts_before_provider_or_execution(self):
        with patch("benchmark.request_plan", new_callable=AsyncMock) as request:
            with self.assertRaisesRegex(RuntimeError, "initial state mismatch"):
                await benchmark.run_case(DEFAULT_CASE, "gpt", {DEFAULT_CASE.case_id: "different"}, {})
        request.assert_not_awaited()
        self.assertFalse(self.simulations[0].started)
        self.assertTrue(self.simulations[0].closed)

    async def test_nonterminating_controller_is_not_a_method_failure(self):
        self.never_finish = True
        with self.assertRaisesRegex(RuntimeError, "watchdog"):
            await benchmark.run_case(DEFAULT_CASE, "fixed", {}, {})
        self.assertTrue(self.simulations[0].closed)



def accounting_row(method="gpt", target="bowl", success=True, executed=True):
    return {
        "case": {"target_object": target}, "method": method,
        "status": "succeeded" if success else ("failed" if executed else "planning_failed"),
        "failure_reason": None if success else "fixture failure",
        "success": success, "execution_sim_s": 3.0 if executed else None,
        "execution_wall_s": 0.3 if executed else None,
        "planning_wall_s": 2.0, "end_to_end_wall_s": 4.0 if executed else 2.0,
        "gpt_latency_s": 1.5 if method == "gpt" else None,
        "collisions": {"episodes": 2} if executed else None,
    }


class SummaryTests(unittest.TestCase):
    def test_failure_denominator_and_distinct_time_domains(self):
        rows = [accounting_row(), accounting_row(success=False, executed=False)]
        summary = benchmark.summarize_runs(rows, ("gpt",))[0]
        self.assertEqual(summary["attempts"], 2)
        self.assertEqual(summary["success_rate"], 0.5)
        self.assertEqual(summary["executed_runs"], 1)
        self.assertEqual(summary["planning_failures"], 1)
        self.assertEqual(summary["mean_success_execution_sim_s"], 3.0)
        self.assertEqual(summary["mean_execution_wall_s"], 0.3)
        self.assertEqual(summary["mean_end_to_end_wall_s"], 3.0)
        self.assertEqual(summary["collision_runs"], 1)
        self.assertEqual(summary["total_collision_episodes"], 2)

    def test_no_execution_does_not_claim_zero_collisions_or_zero_execution_time(self):
        summary = benchmark.summarize_runs([accounting_row(success=False, executed=False)], ("gpt",))[0]
        self.assertEqual(summary["success_rate"], 0.0)
        self.assertEqual(summary["executed_runs"], 0)
        for field in ("collision_runs", "total_collision_episodes", "mean_success_execution_sim_s", "mean_execution_wall_s"):
            self.assertIsNone(summary[field])
        empty = benchmark.summarize_runs([], ("fixed",))[0]
        self.assertIsNone(empty["success_rate"])
        self.assertIsNone(empty["mean_gpt_latency_s"])

    def test_grouping_preserves_target_and_selected_method_denominators(self):
        rows = [accounting_row("fixed"), accounting_row("fixed", "bottle", False),
                accounting_row("gpt", "bottle", False, False)]
        grouped = benchmark.summarize_by_target(rows, ("fixed", "gpt"))
        self.assertEqual([row["method"] for row in grouped["bottle"]], ["fixed", "gpt"])
        self.assertEqual(grouped["bowl"][0]["success_rate"], 1.0)
        self.assertIsNone(grouped["bowl"][1]["success_rate"])
        self.assertEqual(grouped["bottle"][0]["success_rate"], 0.0)
        self.assertEqual(grouped["bottle"][1]["planning_failures"], 1)


class ProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def test_internal_error_saves_partial_report_without_fabricated_failure_row(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "comparison.json"
            row = accounting_row("fixed")
            with patch("benchmark.run_case", new=AsyncMock(side_effect=[row, RuntimeError("programmer error")])) as run, \
                    redirect_stderr(StringIO()):
                await benchmark.run_benchmark(("fixed", "rule"), (101,), output)
            saved = json.loads(output.read_text())
        self.assertEqual(saved["status"], "aborted")
        self.assertEqual(saved["protocol"]["planned_runs"], 12)
        self.assertEqual(len(saved["runs"]), 1)
        self.assertEqual(saved["summary"][0]["attempts"], 1)
        self.assertEqual(saved["summary"][1]["attempts"], 0)
        self.assertEqual(saved["abort"]["active_run"]["method"], "rule")
        self.assertEqual(saved["abort"]["error_type"], "RuntimeError")
        self.assertEqual(run.await_count, 2)

    async def test_missing_calibration_aborts_instead_of_using_empty_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("benchmark.run_case", new_callable=AsyncMock) as run, redirect_stderr(StringIO()):
                report = await benchmark.run_benchmark(output=root / "comparison.json", root=root)
        self.assertEqual(report["protocol"]["planned_runs"], 54)
        self.assertEqual(report["status"], "aborted")
        self.assertEqual(report["runs"], [])
        self.assertEqual(report["abort"]["error_type"], "FileNotFoundError")
        run.assert_not_awaited()

    async def test_calibration_cannot_be_overwritten_by_output(self):
        with self.assertRaisesRegex(ValueError, "overwrite"):
            await benchmark.run_benchmark(output=benchmark.ROOT / "results" / "parameter_sweep.json")

    def test_invalid_cli_methods_and_seeds_rejected_before_any_simulation(self):
        invalid = (
            ["--methods", "unknown"], ["--methods", "fixed", "fixed"], ["--methods"],
            ["--seeds", "-1"], ["--seeds", str(2**32)], ["--seeds", "1.5"],
            ["--seeds", "101", "101"], ["--seeds"],
        )
        for arguments in invalid:
            with self.subTest(arguments=arguments), redirect_stderr(StringIO()), self.assertRaises(SystemExit) as error:
                benchmark.parse_args(arguments)
            self.assertEqual(error.exception.code, 2)
        args = benchmark.parse_args(["--methods", "rule", "fixed", "--seeds", "0", str(2**32 - 1)])
        self.assertEqual(args.methods, ["rule", "fixed"])
        self.assertEqual(args.seeds, [0, 2**32 - 1])


class PhysicalPairingTests(unittest.TestCase):
    def test_fresh_real_simulations_pair_after_settle_and_hash_all_state_vectors(self):
        case = benchmark_cases((101,))[0]
        fingerprints = {}
        for _ in range(2):
            sim = Simulation(case)
            try:
                sim.step(benchmark.SETTLE_S)
                fingerprint = benchmark.initial_state_fingerprint(sim)
                benchmark.verify_pairing(case, fingerprint, fingerprints)
                for field in ("qpos", "qvel", "ctrl"):
                    values = getattr(sim.data, field)
                    previous = values[0]
                    values[0] += 1e-8
                    self.assertNotEqual(benchmark.initial_state_fingerprint(sim), fingerprint, field)
                    values[0] = previous
            finally:
                sim.close()


if __name__ == "__main__":
    unittest.main()
