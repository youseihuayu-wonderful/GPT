"""Frozen evaluation setup must remain reproducible and cannot change control."""

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from compare_sorting import pairing_evidence
from sorting import SortingSimulation
from sorting_contract import OBJECTS
from sorting_evaluation import assert_frozen, load_protocol, make_simulation, validate_protocol
from verify_skills import model_fingerprint
from verify_sorting import observe_sorting


class EvaluationProtocolTests(unittest.TestCase):
    def setUp(self):
        self.protocol = load_protocol()

    def test_missing_or_duplicate_schedule_cells_are_rejected(self):
        missing = deepcopy(self.protocol)
        missing["schedule"].pop()
        duplicate = deepcopy(self.protocol)
        duplicate["schedule"][-1] = deepcopy(duplicate["schedule"][0])
        with self.assertRaises(ValueError):
            validate_protocol(missing)
        with self.assertRaises(ValueError):
            validate_protocol(duplicate)

    def test_wording_pairs_cannot_change_goal_or_priority(self):
        for field in ("destinations", "cups_first"):
            with self.subTest(field=field):
                changed = deepcopy(self.protocol)
                case = changed["cases"][1]
                if field == "destinations":
                    case[field]["red"], case[field]["blue"] = case[field]["blue"], case[field]["red"]
                else:
                    case[field] = not case[field]
                with self.assertRaises(ValueError):
                    validate_protocol(changed)

    def test_unknown_object_identity_and_pose_commands_are_rejected(self):
        changed = deepcopy(self.protocol)
        poses = changed["layouts"][0]["poses"]
        poses["unknown_cup"] = poses.pop("red_cup")
        with self.assertRaises(ValueError):
            validate_protocol(changed)
        changed = deepcopy(self.protocol)
        changed["layouts"][0]["poses"]["red_cup"]["force_n"] = 100
        with self.assertRaises(ValueError):
            validate_protocol(changed)

    def test_nonfinite_and_boolean_poses_and_invalid_settling_are_rejected(self):
        for value in (True, math.inf, math.nan):
            with self.subTest(pose=value):
                changed = deepcopy(self.protocol)
                changed["layouts"][0]["poses"]["red_bowl"]["x_m"] = value
                with self.assertRaises(ValueError):
                    validate_protocol(changed)
        for value in (0, 0.001, True):
            with self.subTest(settle=value):
                changed = deepcopy(self.protocol)
                changed["setup"]["settle_s"] = value
                with self.assertRaises(ValueError):
                    validate_protocol(changed)

    def test_duplicate_json_keys_cannot_override_registered_model(self):
        text = '{"requested_model":"different-model",' + json.dumps(self.protocol)[1:]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            path.write_text(text, encoding="utf-8")
            with self.assertRaises(ValueError):
                load_protocol(path)

    def test_freeze_rejects_controller_policy_inventory_and_model_drift(self):
        for kind in ("controller", "policy", "inventory"):
            with self.subTest(kind=kind):
                changed = deepcopy(self.protocol)
                if kind == "controller":
                    changed["frozen_files_sha256"]["sorting.py"] = "0" * 64
                elif kind == "policy":
                    changed["baseline_functions_sha256"]["fixed_rule_plan"] = "0" * 64
                else:
                    changed["frozen_files_sha256"].pop(next(
                        path for path in changed["frozen_files_sha256"] if path.startswith("vendor/")))
                with self.assertRaises(ValueError):
                    assert_frozen(changed)
        with patch.dict(os.environ, {"CODEX_MODEL": "different-model"}):
            with self.assertRaises(ValueError):
                assert_frozen(self.protocol)

    def test_unknown_layout_cannot_create_a_scene(self):
        with self.assertRaises(ValueError):
            make_simulation(self.protocol, "not_registered")


class EvaluationInitialPhysicsTests(unittest.TestCase):
    def test_declared_poses_settle_supported_without_changing_compiled_model(self):
        protocol = load_protocol()
        baseline = SortingSimulation()
        self.addCleanup(baseline.close)
        frozen_model = model_fingerprint(baseline)
        initial_time = float(baseline.data.time)
        for layout in protocol["layouts"]:
            with self.subTest(layout=layout["id"]):
                sim = make_simulation(protocol, layout["id"])
                try:
                    self.assertEqual(model_fingerprint(sim), frozen_model)
                    self.assertAlmostEqual(float(sim.data.time), initial_time + protocol["setup"]["settle_s"], places=8)
                    facts = observe_sorting(sim, sim.state()["sorting"])
                    for obj, fact in zip(OBJECTS, facts["objects"]):
                        requested = layout["poses"][obj.id]
                        np.testing.assert_allclose(fact["position_m"][:2],
                                                   [requested["x_m"], requested["y_m"]], atol=0.002, rtol=0)
                        self.assertTrue(fact["supported"] and fact["stable"] and fact["detached"], fact)
                        joint = sim.model.joint(f"{obj.id}_free")
                        address = int(joint.qposadr[0])
                        yaw = requested["yaw_rad"]
                        expected = np.empty(4)
                        mujoco.mju_mulQuat(expected, np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]),
                                          sim.model.qpos0[address + 3:address + 7])
                        actual = sim.data.qpos[address + 3:address + 7]
                        self.assertGreater(abs(float(expected @ actual)), math.cos(0.02 / 2))
                    self.assertFalse(np.any(sim.data.xfrc_applied))
                    self.assertFalse(np.any(sim.data.qfrc_applied))
                finally:
                    sim.close()

    def test_fresh_creation_and_reset_reproduce_the_entire_paired_initial_state(self):
        protocol = load_protocol()
        for layout in protocol["layouts"]:
            with self.subTest(layout=layout["id"]):
                first = make_simulation(protocol, layout["id"])
                second = None
                try:
                    first.step(0.5)
                    expected = pairing_evidence(first)
                    first.step(0.02)
                    first.reset()
                    first.step(0.5)
                    self.assertEqual(pairing_evidence(first), expected)
                    second = make_simulation(protocol, layout["id"])
                    second.step(0.5)
                    self.assertEqual(pairing_evidence(second), expected)
                finally:
                    first.close()
                    if second is not None:
                        second.close()


if __name__ == "__main__":
    unittest.main()
