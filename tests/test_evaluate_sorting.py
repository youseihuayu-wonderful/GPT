"""Stored evaluation boundaries; all trial execution here is explicitly synthetic."""

import asyncio
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import evaluate_sorting as evaluation
from compare_sorting import fixed_rule_plan
from sorting_contract import OBJECTS


class EvaluationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "evaluation"
        self.protocol_path = self.root / "registered.json"
        self.protocol_path.write_bytes(evaluation.DEFAULT_PROTOCOL.read_bytes())
        self.protocol = evaluation.load_protocol(self.protocol_path)
        self.calls = []
        self.behavior = None
        self.frozen = patch.object(evaluation, "assert_frozen").start()
        self.addCleanup(patch.stopall)
        self.execution = patch.object(evaluation, "run_case", side_effect=self.fake_run).start()
        self.simulations = patch.object(evaluation, "make_simulation", side_effect=AssertionError(
            "Unit tests must not construct real physical scenes")).start()

    def successful_record(self, case, method):
        plan = fixed_rule_plan(case).to_list()
        order = [item["object_id"] for item in plan]
        observation = {
            "objects": [{"id": obj.id, "color": obj.color, "kind": obj.kind,
                         "expected_destination": case.color_destinations[obj.color],
                         **{key: True for key in ("correct_region", "slot_aligned", "upright",
                                                 "supported", "stable", "detached")}}
                        for obj in OBJECTS],
            "hand_empty": True, "arm_home_error_rad": 0.001, "arm_max_speed_rad_s": 0.001,
        }
        return {
            "case_id": case.name, "method": method, "status": "succeeded", "passed": True,
            "errors": [], "pairing": {"sha256": "1" * 64},
            "approved_plan": plan, "physical_lift_order": order,
            "physical_placement_order": order.copy(), "independent_final_hold_s": 5.0,
            "independent_final_observation": observation,
            "grasp_evidence": {obj.id: {"bilateral_seen": True, "lifted_while_gripped": True}
                               for obj in OBJECTS},
            "final_state": {"sorting": {"status": "succeeded", "sorted_count": 6,
                             "elapsed_s": 100.0 if method == "fixed_rule" else 101.0,
                             "validation_hold_s": 5.0,
                             "goal_checks": {key: True for key in (
                                 "all_correct", "all_supported", "all_stable", "hand_empty", "home_reached")}},
                            "collisions": {"episodes": 0}},
            "planner": {"model": self.protocol["requested_model"], "thread_id": "synthetic-thread"}
                       if method == "gpt" else None,
            "execution_wall_s": 2.0 if method == "fixed_rule" else 3.0,
            "wall_s": 5.0, "gpt_latency_s": 1.0 if method == "gpt" else None,
            "trace": [{"observation": observation, "raw_detail": "保留完整记录", "values": [1.25, -0.125]}],
        }

    async def fake_run(self, case, method, record, expected_pairing=None, *, simulation_factory):
        self.calls.append((case.name, method))
        record.update(self.successful_record(case, method))
        if self.behavior is not None:
            await self.behavior(case, method, record, simulation_factory)

    async def run_report(self):
        return await evaluation.run_evaluation(self.output, self.protocol_path)

    def stored(self):
        return json.loads((self.output / "summary.json").read_text())

    def replace_summary(self, report):
        (self.output / "summary.json").write_text(json.dumps(report))

    def fail_record(self, record, *, stage="executing", kind="AssertionError", message="Physical target not reached"):
        record.update(passed=False, status="failed", failure_stage=stage,
                      errors=[{"type": kind, "message": message}])

    async def test_registered_schedule_precedes_trials_and_accounts_each_cell_once(self):
        async def inspect_initial(case, method, record, factory):
            if len(self.calls) == 1:
                initial = evaluation.load_evaluation_report(self.output)
                self.assertEqual(initial["status"], "running")
                self.assertFalse(initial["passed"])
                self.assertEqual(initial["runs"], [])
                self.assertEqual(sum(item["pending"] for item in initial["summary"]), 16)
        self.behavior = inspect_initial
        report = await self.run_report()
        self.assertTrue(report["passed"])
        self.assertEqual(self.calls, [(entry["case_id"], entry["method"]) for entry in self.protocol["schedule"]])
        self.assertEqual(len(self.calls), 16)
        loaded = evaluation.load_evaluation_report(self.output)
        self.assertTrue(loaded["passed"])
        self.assertTrue(all(pair["execution_sim_delta_s"] == 1.0 for pair in loaded["paired"]))
        self.assertTrue(all(pair["execution_wall_delta_s"] == 1.0 for pair in loaded["paired"]))

    async def test_expected_planning_and_physical_failures_continue_and_remain_counted(self):
        async def failures(case, method, record, factory):
            if len(self.calls) == 1:
                self.fail_record(record, stage="planning", message="Proposed destinations contradict the requested task")
                record.update(approved_plan=None, physical_lift_order=None, physical_placement_order=None,
                              execution_wall_s=None, independent_final_hold_s=None)
                record["final_state"]["sorting"].update(status="idle", elapsed_s=0.0, sorted_count=0)
            elif len(self.calls) == 2:
                self.fail_record(record, message="Independent physical goal failed during extra hold")
        self.behavior = failures
        report = await self.run_report()
        self.assertEqual(len(self.calls), 16)
        self.assertEqual(report["status"], "completed")
        self.assertFalse(report["passed"])
        loaded = evaluation.load_evaluation_report(self.output)
        self.assertEqual(sum(item["failures"] for item in loaded["summary"]), 2)
        self.assertEqual(sum(item["pending"] for item in loaded["summary"]), 0)
        self.assertIsNone(loaded["runs"][0]["execution_sim_s"])
        self.assertIsNone(loaded["runs"][0]["execution_wall_s"])
        self.assertIsNone(loaded["paired"][0]["execution_sim_delta_s"])
        self.assertIn("destinations", loaded["runs"][0]["errors"][0]["message"])

    async def test_existing_directory_and_failed_artifact_are_never_overwritten(self):
        async def failure(case, method, record, factory):
            self.fail_record(record)
        self.behavior = failure
        report = await self.run_report()
        first = evaluation.evaluation_run_path(self.output, report["runs"][0]["run_id"])
        before = first.read_bytes()
        count = len(self.calls)
        with self.assertRaises(FileExistsError):
            await self.run_report()
        self.assertEqual(len(self.calls), count)
        with self.assertRaises(FileExistsError):
            with evaluation._save_record(self.output, report["runs"][0]["run_id"], {"passed": True}):
                self.fail("An existing record was overwritten")
        self.assertEqual(first.read_bytes(), before)

    async def test_unknown_internal_failure_aborts_and_preserves_partial_result(self):
        async def broken(case, method, record, factory):
            if len(self.calls) == 3:
                self.fail_record(record, kind="RuntimeError", message="unexpected internal failure")
        self.behavior = broken
        report = await self.run_report()
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(report["status"], "aborted")
        loaded = evaluation.load_evaluation_report(self.output)
        self.assertEqual(sum(row["pending"] for row in loaded["summary"]), 13)
        self.assertEqual(sum(row["infrastructure_failures"] for row in loaded["summary"]), 1)
        self.assertEqual(sum(row["failures"] for row in loaded["summary"]), 0)

    async def test_cleanup_failure_is_not_a_task_failure(self):
        async def cleanup(case, method, record, factory):
            record["passed"] = False
            record["status"] = "failed"
            record["errors"].append({"stage": "simulation_close", "type": "RuntimeError", "message": "close failed"})
        self.behavior = cleanup
        report = await self.run_report()
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(report["runs"][0]["failure_stage"], "simulation_close")
        self.assertTrue(evaluation.load_evaluation_report(self.output)["runs"][0]["infrastructure_failure"])

    async def test_initialization_failure_has_no_fabricated_execution_measurements(self):
        async def setup_failure(case, method, record, expected_pairing=None, *, simulation_factory):
            record.update(passed=False, status="failed", failure_stage="preparing", wall_s=0.01,
                          errors=[{"type": "ValueError", "message": "initialization failed"}])
        self.execution.side_effect = setup_failure
        report = await self.run_report()
        row = evaluation.load_evaluation_report(self.output)["runs"][0]
        self.assertEqual(report["status"], "aborted")
        self.assertTrue(row["infrastructure_failure"])
        self.assertIsNone(row["execution_sim_s"])
        self.assertIsNone(row["sorted_count"])

    async def test_pairing_is_enforced_across_wordings_not_only_methods(self):
        first_layout = self.protocol["cases"][0]["layout_id"]
        first_wording = self.protocol["cases"][0]["wording_id"]
        cases = {case["id"]: case for case in self.protocol["cases"]}
        async def mismatch(case, method, record, factory):
            registered = cases[case.name]
            if registered["layout_id"] == first_layout and registered["wording_id"] != first_wording:
                record["pairing"]["sha256"] = "2" * 64
        self.behavior = mismatch
        report = await self.run_report()
        self.assertLess(len(self.calls), 16)
        self.assertEqual(report["runs"][-1]["failure_stage"], "pairing")
        self.assertTrue(evaluation.load_evaluation_report(self.output)["runs"][-1]["infrastructure_failure"])

    async def test_model_substitution_aborts_without_retry(self):
        async def wrong_model(case, method, record, factory):
            if method == "gpt":
                record["planner"]["model"] = "different-model"
        self.behavior = wrong_model
        report = await self.run_report()
        self.assertEqual(sum(method == "gpt" for _, method in self.calls), 1)
        self.assertEqual(report["runs"][-1]["failure_stage"], "model_identity")
        self.assertEqual(evaluation.load_evaluation_report(self.output)["status"], "aborted")

    async def test_frozen_preflight_failure_persists_all_pending_and_executes_nothing(self):
        self.frozen.side_effect = ValueError("frozen source changed")
        with self.assertRaisesRegex(ValueError, "frozen source"):
            await self.run_report()
        self.execution.assert_not_called()
        report = evaluation.load_evaluation_report(self.output)
        self.assertEqual(report["status"], "aborted")
        self.assertEqual(sum(row["pending"] for row in report["summary"]), 16)

    async def test_frozen_postflight_failure_retains_trial_and_aborts(self):
        self.frozen.side_effect = [None, ValueError("frozen source changed")]
        report = await self.run_report()
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(report["runs"][0]["failure_stage"], "frozen_inputs")
        self.assertEqual(evaluation.load_evaluation_report(self.output)["status"], "aborted")

    async def test_cancellation_is_persisted_and_reraised(self):
        async def cancel(case, method, record, factory):
            if len(self.calls) == 2:
                raise asyncio.CancelledError("operator cancelled")
        self.behavior = cancel
        with self.assertRaises(asyncio.CancelledError):
            await self.run_report()
        report = evaluation.load_evaluation_report(self.output)
        self.assertEqual(report["status"], "aborted")
        self.assertEqual(len(report["runs"]), 2)
        self.assertEqual(report["runs"][-1]["failure_stage"], "interrupted")
        self.assertEqual(sum(row["pending"] for row in report["summary"]), 14)

    async def test_persistence_failure_does_not_replace_primary_cancellation(self):
        async def cancel(case, method, record, factory):
            raise asyncio.CancelledError("primary cancellation")
        self.behavior = cancel
        with patch.object(evaluation, "_save_record", side_effect=OSError("disk full")):
            with self.assertRaises(asyncio.CancelledError) as caught:
                await self.run_report()
        self.assertIn("disk full", " ".join(caught.exception.__notes__))
        self.assertEqual(self.stored()["status"], "aborted")

    async def test_success_flag_without_independent_evidence_becomes_infrastructure_error(self):
        async def no_evidence(case, method, record, factory):
            del record["independent_final_observation"]
        self.behavior = no_evidence
        report = await self.run_report()
        self.assertFalse(report["passed"])
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(report["runs"][0]["failure_stage"], "evidence")
        self.assertEqual(evaluation.load_evaluation_report(self.output)["status"], "aborted")

    async def test_nonfinite_trial_measurement_is_retained_without_inventing_zero(self):
        async def invalid_measurement(case, method, record, factory):
            record["execution_wall_s"] = float("nan")

        self.behavior = invalid_measurement
        await self.run_report()
        report = evaluation.load_evaluation_report(self.output)
        row = report["runs"][0]
        self.assertEqual(report["status"], "aborted")
        self.assertEqual(row["failure_stage"], "evidence")
        self.assertIsNone(row["execution_wall_s"])
        with gzip.open(self.output / row["record_path"], "rt") as stream:
            full = json.load(stream)
        self.assertEqual(full["execution_wall_s"], {"nonfinite_observation": "nan"})

    async def test_unsaved_attempt_is_not_misreported_as_pending(self):
        with patch.object(evaluation, "_save_record", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                await self.run_report()
        report = evaluation.load_evaluation_report(self.output)
        self.assertEqual(report["status"], "aborted")
        self.assertEqual(sum(row["attempted"] for row in report["summary"]), 1)
        self.assertEqual(sum(row["pending"] for row in report["summary"]), 15)
        row = report["runs"][0]
        self.assertEqual(row["failure_stage"], "persistence")
        self.assertIsNone(row["record_path"])
        self.assertIsNone(row["record_sha256"])
        with self.assertRaisesRegex(ValueError, "unsaved"):
            evaluation.evaluation_run_path(self.output, row["run_id"])

    async def test_integrity_assertion_is_infrastructure_but_plan_validation_is_not(self):
        async def failures(case, method, record, factory):
            if len(self.calls) == 1:
                self.fail_record(record, stage="planning", kind="ValueError",
                                 message="Sorting plan must be a list of exactly six steps")
            elif len(self.calls) == 2:
                self.fail_record(record, message="Physics parameters changed during execution")
        self.behavior = failures
        report = await self.run_report()
        self.assertEqual(len(self.calls), 2)
        self.assertFalse(report["runs"][0]["infrastructure_failure"])
        self.assertTrue(report["runs"][1]["infrastructure_failure"])
        self.assertEqual(evaluation.load_evaluation_report(self.output)["status"], "aborted")

    async def test_saved_artifact_without_persistence_fault_cannot_be_hidden(self):
        await self.run_report()
        report = self.stored()
        report["runs"][0]["record_path"] = None
        report["runs"][0]["record_sha256"] = None
        self.replace_summary(report)
        with self.assertRaises(ValueError):
            evaluation.load_evaluation_report(self.output)

    async def test_missing_protocol_is_invalid_not_an_absent_evaluation(self):
        await self.run_report()
        (self.output / "protocol.json").unlink()
        with self.assertRaisesRegex(ValueError, "copied protocol"):
            evaluation.load_evaluation_report(self.output)

    async def test_symlinked_runs_directory_is_rejected(self):
        report = await self.run_report()
        runs = self.output / "runs"
        outside = self.root / "outside-runs"
        runs.rename(outside)
        runs.symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ValueError):
            evaluation.evaluation_run_path(self.output, report["runs"][0]["run_id"])

    async def test_aggregate_counts_are_recomputed_not_trusted(self):
        await self.run_report()
        report = self.stored()
        report["summary"] = [{"method": "gpt", "successes": 10000, "pending": -20}]
        report["paired"] = [{"execution_sim_delta_s": -9999}]
        self.replace_summary(report)
        loaded = evaluation.load_evaluation_report(self.output)
        self.assertEqual([row["successes"] for row in loaded["summary"]], [8, 8])
        self.assertEqual(len(loaded["paired"]), 8)
        self.assertEqual(loaded["paired"][0]["execution_sim_delta_s"], 1.0)

    async def test_duplicate_missing_unknown_and_reordered_identities_rejected(self):
        await self.run_report()
        original = self.stored()
        mutations = []
        duplicate = deepcopy(original)
        duplicate["runs"][1] = deepcopy(duplicate["runs"][0])
        mutations.append(duplicate)
        missing = deepcopy(original)
        missing["runs"].pop()
        mutations.append(missing)
        unknown = deepcopy(original)
        unknown["runs"][0]["run_id"] = "unregistered"
        mutations.append(unknown)
        reordered = deepcopy(original)
        reordered["runs"][0], reordered["runs"][1] = reordered["runs"][1], reordered["runs"][0]
        mutations.append(reordered)
        wrong_schedule = deepcopy(original)
        wrong_schedule["schedule"].pop()
        mutations.append(wrong_schedule)
        for report in mutations:
            with self.subTest(report=report["runs"][0]["run_id"]):
                self.replace_summary(report)
                with self.assertRaises(ValueError):
                    evaluation.load_evaluation_report(self.output)

    async def test_contradictory_success_evidence_is_rejected_at_public_boundary(self):
        await self.run_report()
        original = self.stored()
        changes = [
            ("sorted_count", 5), ("independent_final_hold_s", 4.0),
            ("controller_validation_hold_s", None), ("physical_lift_order", []),
            ("physical_placement_order", list(reversed(original["runs"][0]["physical_placement_order"]))),
            ("final_goal_checks", {}), ("collision_episodes", 1),
            ("execution_sim_s", float("nan")), ("execution_wall_s", -1),
            ("grasp_evidence", {}), ("independent_final_observation", None),
            ("infrastructure_failure", True), ("status", "failed"),
            ("errors", [{"type": "AssertionError", "message": "not actually successful"}]),
        ]
        for field, value in changes:
            with self.subTest(field=field):
                report = deepcopy(original)
                report["runs"][0][field] = value
                self.replace_summary(report)
                with self.assertRaises(ValueError):
                    evaluation.load_evaluation_report(self.output)
        report = deepcopy(original)
        report["runs"][0]["independent_final_observation"]["objects"][0]["slot_aligned"] = False
        self.replace_summary(report)
        with self.assertRaises(ValueError):
            evaluation.load_evaluation_report(self.output)

    async def test_persisted_pairing_model_and_protocol_tampering_is_rejected(self):
        await self.run_report()
        original = self.stored()
        for mutate in (
            lambda report: report["runs"][1].update(initial_pairing_sha256="3" * 64),
            lambda report: next(row for row in report["runs"] if row["method"] == "gpt").update(model="replacement"),
            lambda report: report.update(protocol_sha256="0" * 64),
            lambda report: report.update(expected_runs=15),
            lambda report: report.update(passed=False),
        ):
            report = deepcopy(original)
            mutate(report)
            self.replace_summary(report)
            with self.assertRaises(ValueError):
                evaluation.load_evaluation_report(self.output)
        self.replace_summary(original)
        with (self.output / "protocol.json").open("ab") as stream:
            stream.write(b"\n")
        with self.assertRaisesRegex(ValueError, "digest"):
            evaluation.load_evaluation_report(self.output)

    async def test_artifact_lookup_rejects_traversal_unknown_missing_and_symlinks(self):
        report = await self.run_report()
        for run_id in ("../summary", "/etc/passwd", "a/b", "a\\b", "..", "unregistered"):
            with self.subTest(run_id=run_id), self.assertRaises(ValueError):
                evaluation.evaluation_run_path(self.output, run_id)
        row = report["runs"][0]
        artifact = self.output / row["record_path"]
        content = artifact.read_bytes()
        artifact.unlink()
        with self.assertRaises(ValueError):
            evaluation.evaluation_run_path(self.output, row["run_id"])
        outside = self.root / "outside.json.gz"
        outside.write_bytes(content)
        artifact.symlink_to(outside)
        with self.assertRaises(ValueError):
            evaluation.evaluation_run_path(self.output, row["run_id"])

    async def test_artifact_reference_cannot_escape_canonical_runs_directory(self):
        await self.run_report()
        report = self.stored()
        report["runs"][0]["record_path"] = "../registered.json"
        self.replace_summary(report)
        with self.assertRaises(ValueError):
            evaluation.load_evaluation_report(self.output)

    async def test_public_read_helpers_do_not_execute_or_decompress_trials(self):
        await self.run_report()
        self.execution.reset_mock()
        self.frozen.reset_mock()
        with patch.object(gzip, "open", side_effect=AssertionError("No trace decompression on refresh")):
            report = evaluation.load_evaluation_report(self.output)
            evaluation.evaluation_run_path(self.output, report["runs"][0]["run_id"])
        self.execution.assert_not_called()
        self.frozen.assert_not_called()
        self.simulations.assert_not_called()

    async def test_pending_cancellation_during_final_synchronous_trial_cannot_complete_batch(self):
        async def cancel_last(case, method, record, factory):
            if len(self.calls) == len(self.protocol["schedule"]):
                asyncio.current_task().cancel()

        self.behavior = cancel_last
        task = asyncio.create_task(self.run_report())
        with self.assertRaises(asyncio.CancelledError):
            await task
        report = evaluation.load_evaluation_report(self.output)
        self.assertEqual(report["status"], "aborted")
        self.assertFalse(report["passed"])
        self.assertEqual(len(report["runs"]), 16)

    async def test_published_evidence_remains_downloadable_when_temporary_cleanup_fails(self):
        unlink = Path.unlink

        def refuse_run_cleanup(path, *args, **kwargs):
            if path.name.startswith(".run-"):
                raise OSError("temporary cleanup unavailable")
            return unlink(path, *args, **kwargs)

        with patch.object(Path, "unlink", new=refuse_run_cleanup):
            with self.assertRaises(OSError):
                await self.run_report()
        report = evaluation.load_evaluation_report(self.output)
        self.assertEqual(report["status"], "aborted")
        row = report["runs"][0]
        self.assertTrue(row["infrastructure_failure"])
        artifact = evaluation.evaluation_run_path(self.output, row["run_id"])
        self.assertEqual(hashlib.sha256(artifact.read_bytes()).hexdigest(), row["record_sha256"])
        with gzip.open(artifact, "rt") as stream:
            self.assertEqual(json.load(stream)["case_id"], row["case_id"])

    async def test_temporary_cleanup_cannot_replace_primary_publication_error(self):
        primary = OSError("primary publication failure")
        unlink = Path.unlink

        def refuse_run_cleanup(path, *args, **kwargs):
            if path.name.startswith(".run-"):
                raise PermissionError("secondary temporary cleanup failure")
            return unlink(path, *args, **kwargs)

        with patch.object(evaluation.os, "link", side_effect=primary), patch.object(Path, "unlink", new=refuse_run_cleanup):
            with self.assertRaises(OSError) as caught:
                await self.run_report()
        self.assertIs(caught.exception, primary)
        report = evaluation.load_evaluation_report(self.output)
        self.assertEqual(report["status"], "aborted")
        self.assertEqual(sum(row["attempted"] for row in report["summary"]), 1)
        self.assertEqual(sum(row["pending"] for row in report["summary"]), 15)

    async def test_unrepresentable_numbers_and_invalid_copied_model_are_validation_errors(self):
        await self.run_report()
        report = self.stored()
        report["runs"][0]["execution_sim_s"] = 10 ** 400
        self.replace_summary(report)
        with self.assertRaises(ValueError):
            evaluation.load_evaluation_report(self.output)
        protocol = deepcopy(self.protocol)
        protocol["requested_model"] = "-invalid"
        (self.output / "protocol.json").write_text(json.dumps(protocol))
        with self.assertRaises(ValueError):
            evaluation.load_evaluation_report(self.output)

    def test_absence_and_invalid_existing_metadata_have_distinct_errors(self):
        with self.assertRaises(FileNotFoundError):
            evaluation.load_evaluation_report(self.output)
        self.output.mkdir()
        (self.output / "summary.json").write_text("{")
        with self.assertRaises(ValueError):
            evaluation.load_evaluation_report(self.output)
        (self.output / "summary.json").write_text(" " * 2_000_001)
        with self.assertRaisesRegex(ValueError, "size bound"):
            evaluation.load_evaluation_report(self.output)


if __name__ == "__main__":
    unittest.main()
