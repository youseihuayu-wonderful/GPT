"""Independent task goals must not be rewritten by the model's own proposal."""

from copy import deepcopy
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from compare_sorting import CASES, pairing_evidence, require_matches_case
from sorting import SortingSimulation
from sorting_contract import FIXED_PLAN, OBJECTS, SortStep, SortingPlan
from verify_sorting import execute_and_verify, free_joint, observe_sorting


class SortingComparisonTests(unittest.TestCase):
    def make_sim(self):
        sim = SortingSimulation()
        self.addCleanup(sim.close)
        sim.step(0.5)
        return sim

    def test_safe_but_wrong_color_proposal_is_not_authorized(self):
        with self.assertRaisesRegex(AssertionError, "destinations"):
            require_matches_case(FIXED_PLAN, CASES[1])

    def test_correct_destinations_do_not_override_requested_priority(self):
        case = CASES[1]
        plan = SortingPlan(tuple(SortStep(obj.id, case.color_destinations[obj.color]) for obj in OBJECTS))
        with self.assertRaisesRegex(AssertionError, "priority"):
            require_matches_case(plan, case)

    def place_bowl_for_observer_test(self, sim, xy):
        body = sim.model.body("red_bowl_object").id
        joint = free_joint(sim.model, body)
        address = int(sim.model.jnt_qposadr[joint])
        # Explicit test setup for the independent oracle, not grasping evidence.
        sim.data.qpos[address:address + 2] += np.asarray(xy) - sim.data.xpos[body, :2]
        mujoco.mj_forward(sim.model, sim.data)
        sim.step(0.5)

    def test_independent_goal_ignores_controller_proposal_and_completion_flags(self):
        sim = self.make_sim()
        self.place_bowl_for_observer_test(sim, (0.30, 0.10))
        specification = deepcopy(sim.state()["sorting"])
        for item in specification["plan"]:
            item["destination"] = "B"
            item["status"] = "sorted"
        specification["goal_checks"] = {name: True for name in specification["goal_checks"]}
        standard = observe_sorting(sim, specification, CASES[0].color_destinations)["objects"][0]
        swapped = observe_sorting(sim, specification, CASES[1].color_destinations)["objects"][0]
        self.assertTrue(standard["correct_region"] and standard["slot_aligned"] and standard["supported"])
        self.assertFalse(swapped["correct_region"])
        self.assertFalse(swapped["slot_aligned"])

    def test_inside_region_is_not_enough_when_reserved_slot_is_missed(self):
        sim = self.make_sim()
        self.place_bowl_for_observer_test(sim, (0.30, 0.14))
        bowl = observe_sorting(sim, sim.state()["sorting"])["objects"][0]
        self.assertTrue(bowl["correct_region"] and bowl["supported"] and bowl["stable"])
        self.assertFalse(bowl["slot_aligned"])

    def test_pairing_detects_different_solver_warmstart_with_identical_poses(self):
        sim = self.make_sim()
        before = pairing_evidence(sim)
        sim.data.qacc_warmstart[0] += 0.1  # Deliberate latent-state mismatch.
        after = pairing_evidence(sim)
        for key in ("qpos", "qvel", "ctrl"):
            self.assertEqual(before["physics"][key], after["physics"][key])
        self.assertNotEqual(before["sha256"], after["sha256"])

    def test_wrong_physical_order_is_rejected_even_if_controller_metadata_agrees(self):
        sim = self.make_sim()
        wrong_order = SortingPlan((FIXED_PLAN.steps[1], FIXED_PLAN.steps[0], *FIXED_PLAN.steps[2:]))
        original_state = sim.state
        reported_queue = original_state()["sorting"]["plan"]

        def false_controller_metadata():
            state = original_state()
            state["sorting"]["plan"] = deepcopy(reported_queue)
            state["sorting"]["item_results"] = [
                {**result, "object_id": FIXED_PLAN.steps[index].object_id, "index": index}
                for index, result in enumerate(state["sorting"]["item_results"])
            ]
            return state

        record = {}
        with patch.object(sim, "state", side_effect=false_controller_metadata):
            with self.assertRaisesRegex(AssertionError, "physical lift out of approved order"):
                execute_and_verify(sim, record, expected_plan=FIXED_PLAN,
                                   start=lambda: sim.start_sorting(wrong_order))
        last = record["trace"][-1]["observation"]["objects"]
        cup = next(obj for obj in last if obj["id"] == "red_cup")
        self.assertTrue(cup["bilateral_grip"])
        self.assertFalse(cup["supported"])
        self.assertFalse(record["passed"])


if __name__ == "__main__":
    unittest.main()
