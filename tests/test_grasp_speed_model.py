"""Synthetic unit fixtures only: these labels are NOT physical experiment evidence."""

from copy import deepcopy
from dataclasses import replace
import unittest
from unittest.mock import Mock

import numpy as np

from grasp_speed_model import (
    FEATURE_NAMES, SPEEDS_M_S,
    predict_candidates, select_speed, train_classifier,
)
from scenarios import DEFAULT_CASE


def candidates(probabilities):
    return [
        {"speed_m_s": speed, "predicted_success_probability": probability}
        for speed, probability in zip(SPEEDS_M_S, probabilities)
    ]


def synthetic_unit_rows():
    # Deliberately invented labels exercise the estimator API, not grasp physics.
    return [
        {
            "trial_status": "completed",
            "success": success,
            "case": replace(DEFAULT_CASE, target_object=target, offset_xy_m=(offset, 0.)).to_dict(),
            "parameters": {"descent_speed_m_s": speed},
        }
        for target in ("bowl", "bottle")
        for offset, success in ((0., True), (.04, False))
        for speed in SPEEDS_M_S
    ]


class SpeedSelectionTests(unittest.TestCase):
    def test_threshold_equality_qualifies_and_fastest_wins(self):
        result = select_speed(candidates((.99, .95, .9)), threshold=.9)
        self.assertEqual(result, {
            "selected_speed_m_s": .2,
            "selected_success_probability": .9,
            "threshold": .9,
            "threshold_met": True,
            "selection_rule": "fastest_above_threshold",
        })

    def test_nonmonotonic_probabilities_are_not_assumed_ordered(self):
        result = select_speed(list(reversed(candidates((.95, .2, .91)))))
        self.assertEqual(result["selected_speed_m_s"], .2)
        self.assertTrue(result["threshold_met"])
        self.assertEqual(select_speed(candidates((.2, .94, .3)))["selected_speed_m_s"], .1)

    def test_below_threshold_maximum_and_slower_tie_are_explicit(self):
        for probabilities, expected in (((.5, .8, .8), .1), ((.8, .8, .8), .04), ((.2, .4, .8), .2)):
            with self.subTest(probabilities=probabilities):
                result = select_speed(candidates(probabilities))
                self.assertEqual(result["selected_speed_m_s"], expected)
                self.assertFalse(result["threshold_met"])
                self.assertEqual(result["selection_rule"], "highest_probability_slower_tiebreak")

    def test_threshold_endpoints(self):
        self.assertEqual(select_speed(candidates((0., 0., 0.)), threshold=0)["selected_speed_m_s"], .2)
        self.assertTrue(select_speed(candidates((1., .5, .5)), threshold=1)["threshold_met"])

    def test_invalid_probabilities_and_thresholds(self):
        for invalid in (float("nan"), float("inf"), -float("inf"), -.1, 1.1, True, "0.9", None):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    select_speed(candidates((.5, invalid, .9)))
                with self.assertRaises(ValueError):
                    select_speed(candidates((.5, .6, .9)), threshold=invalid)

    def test_invalid_candidate_sets(self):
        valid = candidates((.5, .6, .7))
        for invalid in (None, [], valid[:2], valid + valid[:1], [valid[0], valid[0], valid[2]],
                        [valid[0], {}, valid[2]], [valid[0], None, valid[2]]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                select_speed(invalid)
        for speed in (.05, True, None, float("nan"), float("inf"), "0.1"):
            invalid = deepcopy(valid)
            invalid[1]["speed_m_s"] = speed
            with self.subTest(speed=speed), self.assertRaises(ValueError):
                select_speed(invalid)


class SpeedClassifierTests(unittest.TestCase):
    def test_both_boolean_classes_required_without_silent_filtering(self):
        rows = synthetic_unit_rows()
        for invalid in ([], [row for row in rows if row["success"]],
                        [row for row in rows if not row["success"]]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                train_classifier(invalid)
        for label in (None, 0, 1, "true"):
            invalid = deepcopy(rows)
            invalid[0]["success"] = label
            with self.subTest(label=label), self.assertRaises(ValueError):
                train_classifier(invalid)
        invalid = deepcopy(rows)
        invalid[0]["trial_status"] = "invalid_initial_state"
        with self.assertRaises(ValueError):
            train_classifier(invalid)

    def test_trained_predictions_are_reproducible(self):
        first = train_classifier(synthetic_unit_rows(), seed=17)
        second = train_classifier(synthetic_unit_rows(), seed=17)
        actual = predict_candidates(first, DEFAULT_CASE)
        self.assertEqual(actual, predict_candidates(second, DEFAULT_CASE))
        self.assertEqual([item["speed_m_s"] for item in actual], list(SPEEDS_M_S))
        for item in actual:
            self.assertTrue(0 <= item["predicted_success_probability"] <= 1)

    def test_identifiers_and_post_execution_fields_do_not_leak(self):
        original = synthetic_unit_rows()
        changed = deepcopy(original)
        for index, row in enumerate(changed):
            row["case"].update(case_id=f"different-{index}", seed=index, position_label="ignored")
            row.update(final_pose={"z": index}, execution_sim_s=index, verified_hold_s=index,
                       failure_reason="ignored", initial_state_fingerprint=str(index))
            row["parameters"].update(initial_opening_m=999, close_trigger_height_m=999)
        first = train_classifier(original)
        second = train_classifier(changed)
        for case in (DEFAULT_CASE, replace(DEFAULT_CASE, target_object="bottle", offset_xy_m=(.04, 0.))):
            self.assertEqual(predict_candidates(first, case), predict_candidates(second, case))
        self.assertEqual(predict_candidates(first, DEFAULT_CASE), predict_candidates(
            first, replace(DEFAULT_CASE, case_id="another-id", seed=77, position_label="another-label")))

    def test_success_probability_uses_success_class_not_column_position(self):
        model = Mock()
        model.feature_names_ = FEATURE_NAMES
        model.classes_ = np.array([True, False])
        model.predict_proba.return_value = np.array([[.2, .8], [.4, .6], [.7, .3]])
        case = replace(DEFAULT_CASE, target_object="bottle", offset_xy_m=(.01, -.02), yaw_rad=.1)
        actual = predict_candidates(model, case)
        self.assertEqual([item["predicted_success_probability"] for item in actual], [.2, .4, .7])

    def test_invalid_feature_values_and_seed(self):
        for field, value in (("target_object", "drawer"), ("offset_xy_m", [0.]),
                             ("offset_xy_m", [True, 0.]), ("offset_xy_m", [.081, 0.]),
                             ("yaw_rad", float("nan")), ("yaw_rad", .201)):
            rows = synthetic_unit_rows()
            rows[0]["case"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                train_classifier(rows)
        rows = synthetic_unit_rows()
        rows[0]["parameters"]["descent_speed_m_s"] = .05
        with self.assertRaises(ValueError):
            train_classifier(rows)
        for seed in (True, -1, 2**32, 1.5):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                train_classifier(synthetic_unit_rows(), seed=seed)

    def test_invalid_model_outputs_fail_clearly(self):
        model = Mock()
        model.feature_names_ = FEATURE_NAMES
        model.classes_ = np.array([False, True])
        for probabilities in ([[.2, .8]], [[.2, float("nan")]] * 3,
                              [[-.1, 1.1]] * 3, [[.4, .4]] * 3,
                              [[float("inf"), .1]] * 3):
            model.predict_proba.return_value = probabilities
            with self.subTest(probabilities=probabilities), self.assertRaises(ValueError):
                predict_candidates(model, DEFAULT_CASE)
        model.classes_ = np.array([True])
        with self.assertRaises(ValueError):
            predict_candidates(model, DEFAULT_CASE)
        model.feature_names_ = ("final_pose",)
        with self.assertRaises(ValueError):
            predict_candidates(model, DEFAULT_CASE)


if __name__ == "__main__":
    unittest.main()
