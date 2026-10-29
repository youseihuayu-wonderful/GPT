"""Behavioral accounting tests; synthetic rows are not experimental evidence."""

from copy import deepcopy
import math
import unittest

from planning_analysis import analyze_runs


def run(method="rule", repetition=1, **changes):
    row = {
        "case_id": "table", "method": method, "repetition": repetition,
        "goal": "store_and_home", "target_object": "bowl", "expected_outcome": "success",
        "status": "succeeded", "success": True, "safety_pass": None,
        "infrastructure_errors": [], "end_to_end_wall_s": 10.0, "collisions": None,
        "failure_reason": None, "history": [],
        "initial_state_fingerprint": "physics", "initial_observation_fingerprint": "observation",
        "initial_model_fingerprint": "model",
    }
    row.update(changes)
    return row


class AnalysisTests(unittest.TestCase):
    def test_pairing_uses_repetition_not_input_order_and_sample_variation(self):
        rows = [run("gpt", 2, end_to_end_wall_s=6), run("rule", 1),
                run("gpt", 1, end_to_end_wall_s=12), run("rule", 2)]
        original = deepcopy(rows)
        result = analyze_runs(rows, ("rule", "gpt"))
        self.assertEqual(rows, original)
        pair = result["paired"][0]
        self.assertEqual(pair["complete_pairs"], 2)
        self.assertEqual(pair["matched_success_pairs"], 2)
        delta = pair["delta_wall_s"]
        self.assertEqual(delta["n"], 2)
        self.assertEqual(delta["mean"], -1)
        self.assertAlmostEqual(delta["sd"], math.sqrt(18))
        self.assertEqual((delta["min"], delta["max"]), (-4, 2))
        self.assertEqual(result["by_case"][0]["wall_s"]["sd"], 0)

    def test_duplicate_identity_and_conflicting_case_metadata_are_errors(self):
        with self.assertRaises(ValueError):
            analyze_runs([run(), run()], ("rule", "gpt"))
        for changes in ({"goal": "home"}, {"target_object": "bottle"},
                        {"expected_outcome": "safe_stop"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                analyze_runs([run(), run("gpt", **changes)], ("rule", "gpt"))

    def test_mismatched_and_missing_fingerprints_cannot_supply_comparisons(self):
        for field in ("initial_state_fingerprint", "initial_observation_fingerprint",
                      "initial_model_fingerprint"):
            for value in ("different", None):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    analyze_runs([run(), run("gpt", **{field: value})], ("rule", "gpt"))
        # Even failed outcomes must not be compared under mismatched conditions.
        with self.assertRaises(ValueError):
            analyze_runs([run(success=False), run("gpt", success=False,
                         initial_state_fingerprint="different")], ("rule", "gpt"))

    def test_unavailable_measurements_are_not_zero_filled(self):
        unavailable = (None, float("nan"), float("inf"), -1, True, "3")
        for value in unavailable:
            with self.subTest(value=value):
                result = analyze_runs([run(end_to_end_wall_s=value), run("gpt", history=[
                    {"latency_s": value}, {"latency_s": 0, "status": "failed"},
                    {"latency_s": 2, "status": "failed"},
                ])], ("rule", "gpt"))
                rule, gpt = result["by_case"]
                self.assertEqual(rule["wall_s"], {"n": 0, "mean": None, "sd": None,
                                                  "min": None, "max": None})
                self.assertEqual(rule["success_wall_s"]["n"], 0)
                self.assertEqual(rule["request_latency_s"]["n"], 0)
                self.assertIsNone(rule["collision_runs"])
                self.assertEqual(gpt["request_latency_s"]["n"], 2)
                self.assertEqual(gpt["request_latency_s"]["mean"], 1)
                self.assertEqual(result["paired"][0]["matched_success_pairs"], 1)
                self.assertEqual(result["paired"][0]["delta_wall_s"]["n"], 0)
                self.assertIsNone(gpt["wall_s"]["sd"])

    def test_failed_attempts_retained_but_unmatched_successes_do_not_create_speed_advantage(self):
        rows = [run(), run("gpt", end_to_end_wall_s=12),
                run("rule", 2, end_to_end_wall_s=100),
                run("gpt", 2, success=False, status="failed", end_to_end_wall_s=3,
                    failure_reason="physical failure"),
                run("rule", 3, success=False, status="failed", end_to_end_wall_s=4),
                run("gpt", 3, end_to_end_wall_s=1),
                run("rule", 4, success=False, status="failed"),
                run("gpt", 4, success=False, status="planning_failed")]
        result = analyze_runs(rows, ("rule", "gpt"))
        rule, gpt = result["by_case"]
        self.assertEqual((rule["attempts"], gpt["attempts"]), (4, 4))
        self.assertEqual(gpt["wall_s"]["n"], 4)
        self.assertEqual(gpt["success_wall_s"]["n"], 2)
        pair = result["paired"][0]
        self.assertEqual(pair["complete_pairs"], 4)
        for key in ("matched_success_pairs", "rule_only_successes", "gpt_only_successes", "both_failed"):
            self.assertEqual(pair[key], 1)
        self.assertEqual(pair["delta_wall_s"]["mean"], 2)
        self.assertEqual(pair["delta_wall_s"]["n"], 1)
        self.assertEqual(gpt["failure_reasons"], [{"reason": "physical failure", "count": 1}])

    def test_infrastructure_pending_and_missing_pairs_have_explicit_denominators(self):
        rows = [run(), run("gpt", status="running"),
                run("rule", 2, infrastructure_errors=["setup"], status="setting_up"),
                run("gpt", 2), run("rule", 3, status="setting_up"), run("gpt", 4)]
        result = analyze_runs(rows, ("rule", "gpt"))
        rule, gpt = result["by_case"]
        self.assertEqual((rule["attempts"], rule["pending_runs"], rule["infrastructure_failures"]), (1, 1, 1))
        self.assertEqual((gpt["attempts"], gpt["pending_runs"]), (2, 1))
        self.assertEqual(rule["wall_s"]["n"], 1)
        self.assertEqual(result["paired"][0]["excluded_pairs"], 4)
        self.assertEqual(result["paired"][0]["complete_pairs"], 0)

    def test_safe_stops_are_not_task_successes_and_real_failures_remain_visible(self):
        rows = [run("gpt", expected_outcome="safe_stop", safety_pass=True,
                    failure_reason="expected stop", collisions={"episodes": 0}),
                run("gpt", 2, expected_outcome="safe_stop", safety_pass=False,
                    success=False, failure_reason="unsafe", collisions={"episodes": 2}),
                run("gpt", 3, expected_outcome="safe_stop", safety_pass=True,
                    infrastructure_errors=["shutdown"], failure_reason="shutdown")]
        result = analyze_runs(rows, ("rule", "gpt"))
        self.assertEqual(result["paired"], [])
        empty, safety = result["by_case"]
        self.assertEqual(empty["attempts"], 0)
        self.assertIsNone(empty["wall_s"]["mean"])
        self.assertEqual(safety["attempts"], 2)
        self.assertEqual(safety["successes"], 0)
        self.assertEqual(safety["safety_passes"], 1)
        self.assertEqual(safety["success_wall_s"]["n"], 0)
        self.assertEqual(safety["collision_runs"], 1)
        self.assertEqual(safety["failure_reasons"], [{"reason": "shutdown", "count": 1},
                                                    {"reason": "unsafe", "count": 1}])

    def test_distinct_task_groups_keep_first_seen_case_and_requested_method_order(self):
        rows = [run("gpt", case_id="hold_bottle", goal="hold_target", target_object="bottle",
                    end_to_end_wall_s=200), run("rule", end_to_end_wall_s=2)]
        result = analyze_runs(rows, ("rule", "gpt"))
        self.assertEqual([(item["case_id"], item["method"]) for item in result["by_case"]],
                         [("hold_bottle", "rule"), ("hold_bottle", "gpt"),
                          ("table", "rule"), ("table", "gpt")])
        self.assertEqual(result["by_case"][1]["wall_s"]["mean"], 200)
        self.assertEqual(result["by_case"][2]["wall_s"]["mean"], 2)
        self.assertEqual(result["paired"][0]["goal"], "hold_target")
        self.assertEqual(result["paired"][0]["target_object"], "bottle")
        self.assertEqual(result["paired"][1]["goal"], "store_and_home")
        self.assertTrue(all(pair["excluded_pairs"] == 1 for pair in result["paired"]))

    def test_legacy_rows_single_method_and_empty_input(self):
        row = run()
        for field in ("repetition", "goal", "target_object"):
            del row[field]
        result = analyze_runs([row], ("rule",))
        self.assertEqual(result["paired"], [])
        self.assertEqual(result["by_case"][0]["goal"], "store_and_home")
        self.assertEqual(result["by_case"][0]["target_object"], "bowl")
        self.assertEqual(analyze_runs([], ("rule", "gpt")), {"by_case": [], "paired": []})


if __name__ == "__main__":
    unittest.main()
