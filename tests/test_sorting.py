"""Consumer-visible sorting transitions exercised with real MuJoCo physics."""

from copy import deepcopy
import unittest

import mujoco
import numpy as np

from sorting import SortingSimulation
from verify_skills import model_fingerprint, physics_snapshot
from verify_sorting import free_joint, independent_complete, observe_sorting


class FixedSortingTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
