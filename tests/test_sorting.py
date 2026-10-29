"""Consumer-visible sorting transitions exercised with real MuJoCo physics."""

from copy import deepcopy
from itertools import permutations
import unittest

import mujoco
import numpy as np

from sorting import SortingSimulation
from sorting_contract import FIXED_PLAN, OBJECTS, SortStep, SortingPlan, validate_plan
from verify_skills import model_fingerprint, physics_snapshot
from verify_sorting import free_joint, independent_complete, observe_sorting


class SortingTests(unittest.TestCase):
    def make_sim(self):
        sim = SortingSimulation()
        self.addCleanup(sim.close)
        sim.step(0.5)
        return sim

    def test_start_and_duplicate_request_do_not_mutate_physics(self):
        sim = self.make_sim()
        before = physics_snapshot(sim)
        model = model_fingerprint(sim)
        sim.start_sorting()
        self.assertEqual(physics_snapshot(sim), before)
        self.assertEqual(model_fingerprint(sim), model)
        running = deepcopy(sim.state())
        with self.assertRaises(ValueError):
            sim.start_sorting()
        self.assertEqual(sim.state(), running)
        self.assertEqual(physics_snapshot(sim), before)

    def test_occupied_destination_is_rejected_before_motion(self):
        sim = self.make_sim()
        specification = sim.state()["sorting"]
        wrong = next(obj for obj in specification["objects"] if obj["color"] == "blue" and obj["kind"] == "cup")
        destination = next(zone for zone in specification["zones"] if zone["id"] == "A")
        body = sim.model.body(wrong["body_name"]).id
        joint = free_joint(sim.model, body)
        qpos = int(sim.model.jnt_qposadr[joint])
        # Explicit initial-scene fault, never used by successful task execution.
        sim.data.qpos[qpos:qpos + 2] += np.asarray(destination["slots_xy_m"][0]) - sim.data.xpos[body, :2]
        mujoco.mj_forward(sim.model, sim.data)
        sim.step(0.5)
        before = physics_snapshot(sim)
        with self.assertRaises(ValueError):
            sim.start_sorting()
        self.assertEqual(physics_snapshot(sim), before)
        self.assertEqual(sim.skill.status, "idle")

    def test_all_objects_physically_sorted_and_disturbance_revokes_success(self):
        sim = self.make_sim()
        specification = sim.state()["sorting"]
        initial = observe_sorting(sim, specification)
        initial_z = {obj["id"]: obj["position_m"][2] for obj in initial["objects"]}
        lifted = set()
        model = model_fingerprint(sim)
        sim.start_sorting()
        for _ in range(6000):
            if sim.skill.status != "running":
                break
            sim.step(0.1)
            for obj in observe_sorting(sim, specification)["objects"]:
                if (obj["bilateral_grip"] and not obj["supported"]
                        and obj["position_m"][2] - initial_z[obj["id"]] >= 0.10):
                    lifted.add(obj["id"])
        state = sim.state()
        self.assertEqual(sim.skill.status, "succeeded", state["sorting"])
        self.assertEqual(lifted, set(initial_z), "Every object must actually be gripped and lifted")
        self.assertTrue(independent_complete(observe_sorting(sim, specification)))
        self.assertEqual(state["sorting"]["sorted_count"], 6)
        self.assertGreaterEqual(state["sorting"]["validation_hold_s"] + 1e-8, 5.0)
        self.assertEqual(state["collisions"]["episodes"], 0, state["collisions"])
        self.assertEqual(model_fingerprint(sim), model)
        sim.step(1.0)
        self.assertTrue(independent_complete(observe_sorting(sim, specification)))
        self.assertEqual(sim.skill.status, "succeeded")
        body = sim.model.body(specification["objects"][0]["body_name"]).id
        # Deliberate post-success fault: no trajectory edits or attachment.
        sim.data.xfrc_applied[body, 0] = 20.0
        try:
            sim.step(0.04)
        finally:
            sim.data.xfrc_applied[body] = 0
        sim.step(0.1)
        self.assertEqual(sim.skill.status, "failed")
        self.assertFalse(independent_complete(observe_sorting(sim, specification)))
        affected = specification["objects"][0]["id"]
        plan = {item["object_id"]: item["status"] for item in sim.state()["sorting"]["plan"]}
        self.assertEqual(plan[affected], "failed", "Revoke the disturbed object's completion, not the last item")
        stopped = deepcopy(sim.state()["sorting"]["plan"])
        sim.step(0.1)
        self.assertEqual(sim.state()["sorting"]["plan"], stopped)
        with self.assertRaises(ValueError):
            sim.start_sorting()

    def test_lost_grasp_stops_before_next_object(self):
        sim = self.make_sim()
        specification = sim.state()["sorting"]
        first = specification["plan"][0]["object_id"]
        obj = next(item for item in specification["objects"] if item["id"] == first)
        body = sim.model.body(obj["body_name"]).id
        initial_z = float(sim.data.xpos[body, 2])
        sim.start_sorting()
        lifted = False
        for _ in range(450):
            sim.step(0.1)
            observed = next(item for item in observe_sorting(sim, specification)["objects"] if item["id"] == first)
            if (observed["bilateral_grip"] and observed["position_m"][2] - initial_z > 0.10
                    and sim.skill.phase == "hold"):
                lifted = True
                break
            if sim.skill.status != "running":
                break
        self.assertTrue(lifted, sim.state()["sorting"])
        before_relative = sim.data.xpos[body] - sim.data.site("robot_grasp_center").xpos
        disrupted = False
        sim.data.xfrc_applied[body, 0] = 20.0
        try:
            for _ in range(25):
                sim.step(0.02)
                observed = next(item for item in observe_sorting(sim, specification)["objects"] if item["id"] == first)
                relative = sim.data.xpos[body] - sim.data.site("robot_grasp_center").xpos
                disrupted |= not observed["bilateral_grip"] or np.linalg.norm(relative - before_relative) > 0.01
                if sim.skill.status == "failed":
                    break
        finally:
            sim.data.xfrc_applied[body] = 0
        sim.step(0.5)
        self.assertTrue(disrupted, "Fault injection must cause an observed loss of grip or relative slip")
        state = sim.state()["sorting"]
        self.assertEqual(state["status"], "failed", state)
        self.assertEqual(state["sorted_count"], 0)
        self.assertTrue(all(item["status"] == "pending" for item in state["plan"][1:]))

    def test_numerical_failure_stays_latched_until_reset(self):
        sim = self.make_sim()
        sim.start_sorting()
        sim.data.qvel[0] = np.nan
        with self.assertRaises(RuntimeError):
            sim.step()
        sim.data.qvel[0] = 0.0
        with self.assertRaises(RuntimeError):
            sim.step()
        with self.assertRaises(ValueError):
            sim.start_sorting()
        sim.reset()
        sim.step(0.5)
        sim.start_sorting()
        self.assertEqual(sim.skill.status, "running")

    def test_prospective_plan_does_not_change_assignments_or_controller(self):
        sim = self.make_sim()
        mapping = {"red": "C", "blue": "A", "yellow": "B"}
        plan = validate_plan([{"object_id": obj.id, "destination": mapping[obj.color]}
                              for obj in reversed(OBJECTS)])
        before = deepcopy(sim.state())
        physics = physics_snapshot(sim)
        model = model_fingerprint(sim)
        self.assertIsNone(sim.skill.precondition_error(plan))
        prospective = sim.object_checks(plan)
        self.assertEqual({obj["color"]: obj["destination"] for obj in prospective}, mapping)
        self.assertEqual(sim.state(), before)
        self.assertEqual(physics_snapshot(sim), physics)
        self.assertEqual(model_fingerprint(sim), model)
        self.assertIs(sim.plan, FIXED_PLAN)
        sim.start_sorting(plan)
        self.assertEqual(physics_snapshot(sim), physics)
        self.assertEqual(model_fingerprint(sim), model)
        state = sim.state()["sorting"]
        self.assertEqual(state["mode"], "planned")
        self.assertEqual(state["current_index"], 5)
        self.assertEqual(state["current_step_index"], 0)
        self.assertEqual(state["current_object_id"], "yellow_cup")
        self.assertEqual([item["object_index"] for item in state["plan"]], [5, 4, 3, 2, 1, 0])
        self.assertEqual([item["index"] for item in state["plan"]], list(range(6)))
        self.assertEqual(sim.skill.plan_status, ["pending"] * 5 + ["active"])
        self.assertNotIn("gpt_called", state)
        running = deepcopy(sim.state())
        with self.assertRaises(ValueError):
            sim.start_sorting(FIXED_PLAN)
        self.assertEqual(sim.state(), running)
        self.assertIs(sim.plan, plan)
        sim.reset()
        self.assertIs(sim.plan, FIXED_PLAN)
        self.assertEqual(sim.state()["sorting"]["mode"], "fixed")
        self.assertIsNone(sim.skill.current_step_index)
        sim.start_sorting()
        self.assertEqual(sim.skill.current_index, 0)
        self.assertEqual(sim.skill.mode, "fixed")

    def test_invalid_plans_cannot_mutate_physics_or_controller(self):
        sim = self.make_sim()
        incomplete = object.__new__(SortingPlan)
        object.__setattr__(incomplete, "steps", FIXED_PLAN.steps[:-1])
        mutable = object.__new__(SortingPlan)
        object.__setattr__(mutable, "steps", list(FIXED_PLAN.steps))
        invalid_step = object.__new__(SortStep)
        object.__setattr__(invalid_step, "object_id", "red_bowl")
        object.__setattr__(invalid_step, "destination", "D")
        corrupt = object.__new__(SortingPlan)
        object.__setattr__(corrupt, "steps", (invalid_step,) + FIXED_PLAN.steps[1:])
        for plan in (False, {}, FIXED_PLAN.to_list(), incomplete, mutable, corrupt):
            with self.subTest(plan=plan):
                before = deepcopy(sim.state())
                physics = physics_snapshot(sim)
                model = model_fingerprint(sim)
                self.assertIsNotNone(sim.skill.precondition_error(plan))
                with self.assertRaises(ValueError):
                    sim.start_sorting(plan)
                self.assertEqual(sim.state(), before)
                self.assertEqual(physics_snapshot(sim), physics)
                self.assertEqual(model_fingerprint(sim), model)
                self.assertIs(sim.plan, FIXED_PLAN)

    def test_occupied_prospective_destination_rejects_without_applying_plan(self):
        sim = self.make_sim()
        plan = validate_plan([
            {"object_id": obj.id, "destination": {"red": "C", "blue": "A", "yellow": "B"}[obj.color]}
            for obj in reversed(OBJECTS)
        ])
        body = sim.model.body("blue_cup_object").id
        joint = free_joint(sim.model, body)
        qpos = int(sim.model.jnt_qposadr[joint])
        # Deliberate scene fault at red's proposed slot, not its fixed slot.
        sim.data.qpos[qpos:qpos + 2] += np.asarray((0.64, 0.10)) - sim.data.xpos[body, :2]
        mujoco.mj_forward(sim.model, sim.data)
        sim.step(0.5)
        before = deepcopy(sim.state())
        physics = physics_snapshot(sim)
        self.assertIn("Reserved slot for red_bowl", sim.skill.precondition_error(plan))
        with self.assertRaisesRegex(ValueError, "Reserved slot for red_bowl"):
            sim.start_sorting(plan)
        self.assertEqual(sim.state(), before)
        self.assertEqual(physics_snapshot(sim), physics)
        self.assertIs(sim.plan, FIXED_PLAN)

    def test_all_destination_permutations_execute_reordered_real_contact_batches(self):
        orders = (
            ("yellow_cup", "blue_cup", "red_cup", "yellow_bowl", "blue_bowl", "red_bowl"),
            ("blue_cup", "yellow_bowl", "red_cup", "blue_bowl", "yellow_cup", "red_bowl"),
            ("yellow_bowl", "red_bowl", "blue_bowl", "red_cup", "yellow_cup", "blue_cup"),
        )
        sim = self.make_sim()
        for run, destinations in enumerate(permutations("ABC")):
            with self.subTest(destinations=destinations):
                if run:
                    sim.reset()
                    sim.step(0.5)
                expected = dict(zip(("red", "blue", "yellow"), destinations))
                ordered_ids = orders[run % len(orders)]
                plan = validate_plan([
                    {"object_id": identity, "destination": expected[identity.split("_")[0]]}
                    for identity in ordered_ids
                ])
                # Expectations are chosen independently before controller dispatch.
                specification = sim.state()["sorting"]
                initial = observe_sorting(sim, specification, expected)
                initial_z = {obj["id"]: obj["position_m"][2] for obj in initial["objects"]}
                lifted = set()
                model = model_fingerprint(sim)
                before = physics_snapshot(sim)
                sim.start_sorting(plan)
                self.assertEqual(physics_snapshot(sim), before)
                for _ in range(3000):
                    if sim.skill.status != "running":
                        break
                    sim.step(0.1)
                    for obj in observe_sorting(sim, specification, expected)["objects"]:
                        if (obj["bilateral_grip"] and not obj["supported"]
                                and obj["position_m"][2] - initial_z[obj["id"]] >= 0.10):
                            lifted.add(obj["id"])
                state = sim.state()["sorting"]
                self.assertEqual(sim.skill.status, "succeeded", state)
                self.assertEqual(lifted, set(ordered_ids))
                self.assertEqual([item["object_id"] for item in state["item_results"]], list(ordered_ids))
                self.assertEqual([item["object_id"] for item in state["plan"]], list(ordered_ids))
                self.assertEqual({obj["color"]: obj["destination"] for obj in state["objects"]}, expected)
                self.assertTrue(independent_complete(observe_sorting(sim, specification, expected)))
                self.assertGreaterEqual(state["validation_hold_s"] + 1e-8, 5.0)
                self.assertEqual(sim.collisions.state()["episodes"], 0)
                self.assertEqual(model_fingerprint(sim), model)
                sim.step(0.5)
                self.assertTrue(independent_complete(observe_sorting(sim, specification, expected)))


if __name__ == "__main__":
    unittest.main()
