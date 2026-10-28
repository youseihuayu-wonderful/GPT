"""Collision bookkeeping driven by real MuJoCo contacts, not fabricated forces."""

import unittest

import mujoco
import numpy as np

from collisions import CollisionTracker
from simulation import Simulation


class CollisionTrackerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = mujoco.MjModel.from_xml_string("""
        <mujoco>
          <option timestep="0.002"/>
          <worldbody>
            <body name="table" pos="0 0 -0.1"><geom type="box" size="3 3 .1"/></body>
            <body name="target" pos="0 0 .0499"><freejoint name="target_free"/>
              <geom type="box" size=".05 .05 .05" mass=".1"/></body>
            <body name="clutter" pos=".5 0 .0499"><freejoint name="clutter_free"/>
              <geom type="box" size=".05 .05 .05" mass=".1"/></body>
            <body name="robot_finger" pos="1 0 .5"><freejoint name="robot_finger_free"/>
              <geom type="box" size=".05 .05 .05" mass=".1"/></body>
            <body name="robot_hand" pos="1.5 0 .5"><freejoint name="robot_hand_free"/>
              <geom type="box" size=".05 .05 .05" mass=".1"/></body>
          </worldbody>
        </mujoco>
        """)
        self.data = mujoco.MjData(self.model)
        self.tracker = CollisionTracker(
            self.model, self.data, self.model.body("target").id,
            {self.model.body("robot_finger").id}, self.model.body("table").id,
        )
        self.dt = float(self.model.opt.timestep)
        mujoco.mj_forward(self.model, self.data)

    def _place(self, body: str, xyz) -> None:
        joint = self.model.joint(f"{body}_free")
        qpos, dof = int(joint.qposadr[0]), int(joint.dofadr[0])
        self.data.qpos[qpos:qpos + 3] = xyz
        self.data.qpos[qpos + 3:qpos + 7] = [1, 0, 0, 0]
        self.data.qvel[dof:dof + 6] = 0
        mujoco.mj_forward(self.model, self.data)

    def _step(self) -> dict:
        mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.tracker.update(self.dt)
        return self.tracker.state()

    def _pair_force(self, first: str, second: str) -> float:
        ids = {self.model.body(first).id, self.model.body(second).id}
        force = np.zeros(6)
        total = 0.0
        for i, contact in enumerate(self.data.contact):
            if {int(self.model.geom_bodyid[contact.geom1]), int(self.model.geom_bodyid[contact.geom2])} == ids:
                mujoco.mj_contactForce(self.model, self.data, i, force)
                total += max(0.0, float(force[0]))
        return total

    def test_initial_supports_and_finger_target_contacts_are_excluded(self) -> None:
        self.assertGreater(self._pair_force("target", "table"), .1)
        self.assertGreater(self._pair_force("clutter", "table"), .1)
        self.assertEqual(self._step()["episodes"], 0)
        self._place("target", (0, 0, .5))
        self._place("robot_finger", (.095, 0, .5))
        state = self._step()
        self.assertGreater(self._pair_force("target", "robot_finger"), .1)
        self.assertEqual(state["episodes"], 0)

    def test_table_clearance_requires_geometry_gap_not_contact_chatter(self) -> None:
        self._step()
        self.assertTrue(self.tracker.seen_table_support)
        self._place("target", (0, 0, .0501))
        self._step()
        self.assertFalse(self.tracker.cleared_table)
        self._place("target", (0, 0, .0499))
        self.assertEqual(self._step()["episodes"], 0)
        self._place("target", (0, 0, .055))
        self._step()
        self.assertTrue(self.tracker.cleared_table)
        self._place("target", (0, 0, .0499))
        state = self._step()
        self.assertEqual(state["episodes"], 1)
        self.assertEqual({state["pairs"][0]["body_a"], state["pairs"][0]["body_b"]}, {"target", "table"})
        self.assertGreater(state["peak_force_n"], .1)

    def test_pair_onsets_union_duration_and_force_sum_are_not_contact_point_counts(self) -> None:
        self._place("robot_hand", (1.5, 0, .0499))
        self._place("robot_finger", (1, 0, .0499))
        first = self._step()
        self.assertEqual(first["episodes"], 2)
        self.assertEqual(len(first["pairs"]), 2)
        self.assertAlmostEqual(first["duration_s"], self.dt)
        forces = []
        for pair in first["pairs"]:
            forces.append(self._pair_force(pair["body_a"], pair["body_b"]))
            self.assertAlmostEqual(pair["peak_force_n"], forces[-1])
            self.assertAlmostEqual(pair["duration_s"], self.dt)
        self.assertAlmostEqual(first["peak_force_n"], max(forces))
        second = self._step()
        self.assertEqual(second["episodes"], 2)
        self.assertAlmostEqual(second["duration_s"], 2 * self.dt)
        self._place("robot_hand", (1.5, 0, .5))
        self._place("robot_finger", (1, 0, .5))
        self._step()
        self._place("robot_hand", (1.5, 0, .0499))
        third = self._step()
        self.assertEqual(third["episodes"], 3)
        self.assertAlmostEqual(third["duration_s"], 3 * self.dt)
        self.tracker.reset()
        self.assertEqual(self.tracker.state()["pairs"], [])
        self.assertFalse(self.tracker.cleared_table)
        self.assertEqual(self._step()["episodes"], 1)

    def test_target_clutter_and_robot_self_contacts_count(self) -> None:
        self._place("target", (0, 0, .5))
        self._place("clutter", (.095, 0, .5))
        self._place("robot_finger", (1, 0, .5))
        self._place("robot_hand", (1.095, 0, .5))
        state = self._step()
        self.assertGreater(self._pair_force("target", "clutter"), .1)
        self.assertGreater(self._pair_force("robot_finger", "robot_hand"), .1)
        self.assertEqual(state["episodes"], 2)
        pairs = {(pair["body_a"], pair["body_b"]) for pair in state["pairs"]}
        self.assertEqual(pairs, {("clutter", "target"), ("robot_finger", "robot_hand")})
        self.assertAlmostEqual(state["duration_s"], self.dt)

    def test_subthreshold_real_constraint_force_is_not_an_episode(self) -> None:
        self.model.opt.gravity[:] = 0
        self._place("target", (0, 0, .5))
        self._place("clutter", (.1 - 1e-12, 0, .5))
        state = self._step()
        self.assertLess(self._pair_force("target", "clutter"), .1)
        self.assertEqual(state["episodes"], 0)
        self.assertEqual(state["duration_s"], 0.0)

    def test_returned_state_is_a_snapshot(self) -> None:
        self._place("robot_hand", (1.5, 0, .0499))
        state = self._step()
        state["pairs"][0]["episodes"] = 999
        self.assertEqual(self.tracker.state()["episodes"], 1)


class SimulationCollisionTests(unittest.TestCase):
    def test_idle_contacts_do_not_accumulate_and_reset_clears_execution_records(self) -> None:
        sim = Simulation()
        self.addCleanup(sim.close)
        hand = sim.model.body("robot_hand").id
        bottle = int(sim.model.joint("bottle_free").qposadr[0])
        # Move clutter into the real hand, without editing its mass or geometry.
        sim.data.qpos[bottle:bottle + 3] = sim.data.xpos[hand] - [0, 0, .05]
        mujoco.mj_forward(sim.model, sim.data)
        sim.step(.002)
        self.assertEqual(sim.state()["collisions"]["episodes"], 0)
        sim.start_task("grasp")
        sim.step(.02)
        self.assertGreater(sim.state()["collisions"]["episodes"], 0)
        before = sim.state()["collisions"]
        with self.assertRaises(ValueError):
            sim.start_task("grasp")
        self.assertEqual(sim.state()["collisions"], before)
        sim.reset()
        self.assertEqual(sim.state()["collisions"]["episodes"], 0)
        self.assertEqual(sim.state()["collisions"]["pairs"], [])

    def test_post_success_hold_still_measures_real_contacts(self) -> None:
        sim = Simulation()
        self.addCleanup(sim.close)
        sim.step(1.0)
        sim.start_task("grasp")
        for _ in range(1000):
            sim.step(.02)
            if sim.skill.status != "running":
                break
        self.assertEqual(sim.skill.status, "succeeded")
        before = sim.state()["collisions"]["episodes"]
        hand = sim.model.body("robot_hand").id
        bottle = int(sim.model.joint("bottle_free").qposadr[0])
        sim.data.qpos[bottle:bottle + 3] = sim.data.xpos[hand] - [0, 0, .05]
        mujoco.mj_forward(sim.model, sim.data)
        sim.step(.002)
        self.assertGreater(sim.state()["collisions"]["episodes"], before)


if __name__ == "__main__":
    unittest.main()
