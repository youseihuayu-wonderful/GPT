"""Study integrity and reporting boundaries; synthetic rows are unit fixtures only."""

from pathlib import Path
import tempfile
import unittest

from run_grasp_learning import (
    build_protocol, compare_speeds, evaluate_predictions, outcome_counts, run_study,
)


def outcome(case_id, speed, success, seconds, *, recommended=False, probability=.8):
    return {
        "case": {"case_id": case_id},
        "parameters": {"descent_speed_m_s": speed},
        "trial_status": "completed" if success is not None else "invalid_initial_state",
        "success": success, "execution_sim_s": seconds, "measurement_wall_s": .1,
        "recommended": recommended, "predicted_success_probability": probability,
        "failure_reason": None if success else "unit-only failure",
        "collisions": {"episodes": 0}, "warnings": [],
    }


class ProtocolTests(unittest.TestCase):
    def test_seeded_condition_splits_do_not_leak_development_or_training_poses(self):
        protocol = build_protocol(5644)
        def state(case):
            return case["target_object"], *case["offset_xy_m"], case["yaw_rad"]
        training = {state(case) for case in protocol["training_cases"]}
        evaluation = {state(case) for case in protocol["evaluation_cases"]}
        self.assertFalse(training & evaluation)
        self.assertEqual(len(training), len(protocol["training_cases"]))
        self.assertEqual(len(evaluation), len(protocol["evaluation_cases"]))
        for case in protocol["evaluation_cases"]:
            x, y = case["offset_xy_m"]
            self.assertFalse(x in (-.08, 0., .08) and y in (-.08, 0., .08)
                             and case["yaw_rad"] in (-.2, 0., .2))
        self.assertEqual(protocol, build_protocol(5644))
        self.assertNotEqual(protocol["evaluation_cases"], build_protocol(5645)["evaluation_cases"])

    def test_existing_evidence_directory_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            sentinel = Path(directory) / "summary.json"
            sentinel.write_text('"old failed experiment"\n')
            with self.assertRaises(FileExistsError):
                run_study(Path(directory))
            self.assertEqual(sentinel.read_text(), '"old failed experiment"\n')


class ReportingTests(unittest.TestCase):
    def test_invalid_setup_is_not_a_negative_label_or_probability_error(self):
        rows = [
            outcome("valid_success", .2, True, 12., probability=.9),
            outcome("valid_failure", .2, False, 2., probability=.1),
            outcome("invalid", .2, None, 0., probability=.9),
        ]
        counts = outcome_counts(rows)
        self.assertEqual((counts["attempted"], counts["labelled"], counts["successes"],
                          counts["failures"], counts["invalid_initial_states"]), (3, 2, 1, 1, 1))
        metrics = evaluate_predictions(rows)
        self.assertEqual(metrics["labelled"], 2)
        self.assertAlmostEqual(metrics["brier_score"], .01)
        self.assertEqual(metrics["true_positives"], 1)
        self.assertEqual(metrics["true_negatives"], 1)

    def test_failed_fast_trials_cannot_create_a_spurious_speedup(self):
        rows = [
            outcome("a", .04, True, 20.),
            outcome("a", .10, True, 15., recommended=True),
            outcome("a", .20, False, 1.),
            outcome("b", .04, True, 18.),
            outcome("b", .10, True, 14.),
            outcome("b", .20, True, 12., recommended=True),
            outcome("invalid", .04, None, 0.),
            outcome("invalid", .10, None, 0.),
            outcome("invalid", .20, None, 0., recommended=True),
        ]
        result = compare_speeds(rows)
        self.assertEqual(result["matched_valid_conditions"], 2)
        self.assertEqual(result["methods"]["recommended"]["successes"], 2)
        self.assertEqual(result["methods"]["0.2"]["successes"], 1)
        self.assertEqual(result["methods"]["0.2"]["mean_success_execution_sim_s"], 12.)
        self.assertEqual(result["fastest_failed_but_slower_succeeded_conditions"], ["a"])
        paired = result["recommended_minus_fastest_on_joint_success"]
        self.assertEqual(paired["pairs"], 1)
        self.assertEqual(paired["mean_sim_s"], 0.)


if __name__ == "__main__":
    unittest.main()
