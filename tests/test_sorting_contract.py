"""Pure bounded-plan validation: no provider or MuJoCo required."""

from dataclasses import FrozenInstanceError
from itertools import permutations
import unittest

from sorting_contract import FIXED_PLAN, OBJECTS, SortStep, SortingPlan, validate_plan


class SortingContractTests(unittest.TestCase):
    def test_fixed_plan_preserves_exact_order_and_geometry(self):
        self.assertEqual(FIXED_PLAN.to_list(), [
            {"object_id": f"{color}_{kind}", "destination": destination}
            for color, destination in (("red", "A"), ("blue", "B"), ("yellow", "C"))
            for kind in ("bowl", "cup")
        ])
        for index, obj in enumerate(OBJECTS):
            step = FIXED_PLAN.assignment(obj.id)
            x = (0.30, 0.47, 0.64)[index // 2]
            self.assertEqual(obj.source_xy_m, (x, -0.08 if obj.kind == "bowl" else -0.28))
            self.assertEqual(step.slot_xy_m, (x, 0.10 if obj.kind == "bowl" else 0.29))
            self.assertEqual(step.object_index, index)
            self.assertFalse(hasattr(obj, "destination"))
            self.assertFalse(hasattr(obj, "slot_xy_m"))

    def test_all_orderings_and_color_bijections_are_valid(self):
        for destinations in permutations("ABC"):
            mapping = dict(zip(("red", "blue", "yellow"), destinations))
            for ordered in permutations(OBJECTS):
                payload = [{"object_id": obj.id, "destination": mapping[obj.color]}
                           for obj in ordered]
                plan = validate_plan(payload)
                self.assertEqual(plan.to_list(), payload)
                self.assertEqual(len({step.slot_xy_m for step in plan.steps}), 6)
                for step in plan.steps:
                    self.assertIs(plan.assignment(step.object_id), step)

    def test_plan_and_steps_are_immutable_and_payload_is_detached(self):
        payload = FIXED_PLAN.to_list()
        plan = validate_plan(payload)
        payload[0]["destination"] = "C"
        payload.clear()
        exported = plan.to_list()
        exported[0]["object_id"] = "unknown"
        self.assertEqual(plan, FIXED_PLAN)
        with self.assertRaises(FrozenInstanceError):
            plan.steps = ()
        with self.assertRaises(FrozenInstanceError):
            plan.steps[0].destination = "C"
        with self.assertRaises(TypeError):
            plan.steps[0] = plan.steps[1]

    def test_wire_format_rejects_nonlists_and_incomplete_or_duplicate_steps(self):
        for payload in (None, True, {}, "sort", tuple(FIXED_PLAN.to_list()), [],
                        FIXED_PLAN.to_list()[:-1], FIXED_PLAN.to_list() * 2,
                        [FIXED_PLAN.to_list()[0]] * 6):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                validate_plan(payload)

    def test_wire_format_rejects_unbounded_commands_and_wrong_types(self):
        class StringSubclass(str):
            pass

        invalid_steps = [
            {}, {"object_id": "red_bowl"},
            {"object_id": "red_bowl", "destination": "A", "coordinates": [0, 0, 0]},
            {"object_id": "red_bowl", "destination": "A", "speed": 1},
            {"object_id": "red_bowl", "destination": "A", "command": "move"},
            {"object_id": "unknown", "destination": "A"},
            {"object_id": "red_bowl", "destination": "D"},
            {"object_id": "red_bowl", "destination": "AA"},
            {"object_id": "red_bowl", "destination": ""},
            {"object_id": [], "destination": "A"},
            {"object_id": "red_bowl", "destination": True},
            {"object_id": StringSubclass("red_bowl"), "destination": "A"},
            {"object_id": "red_bowl", "destination": StringSubclass("A")},
            SortStep("red_bowl", "A"), None,
        ]
        for step in invalid_steps:
            payload = FIXED_PLAN.to_list()
            payload[0] = step
            with self.subTest(step=step), self.assertRaises(ValueError):
                validate_plan(payload)

    def test_color_pairs_and_destinations_cannot_conflict(self):
        split = FIXED_PLAN.to_list()
        split[0]["destination"] = "B"
        shared = FIXED_PLAN.to_list()
        for step in shared:
            if step["object_id"].startswith("red_"):
                step["destination"] = "B"
        for payload in (split, shared):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                validate_plan(payload)
            with self.assertRaises(ValueError):
                SortingPlan(tuple(SortStep(**step) for step in payload))

    def test_direct_construction_cannot_bypass_validation(self):
        class StepSubclass(SortStep):
            pass

        for steps in (list(FIXED_PLAN.steps), (), FIXED_PLAN.steps[:-1],
                      (FIXED_PLAN.steps[0],) * 6,
                      tuple(FIXED_PLAN.to_list()),
                      (StepSubclass("red_bowl", "A"),) + FIXED_PLAN.steps[1:]):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                SortingPlan(steps)
        for identity, destination in ((True, "A"), ("red_bowl", None),
                                      ("red_bowl", "AB"), ("red_cup ", "A")):
            with self.assertRaises(ValueError):
                SortStep(identity, destination)
        with self.assertRaises(ValueError):
            FIXED_PLAN.assignment("unknown")


if __name__ == "__main__":
    unittest.main()
