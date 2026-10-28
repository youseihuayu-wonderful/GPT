"""Per-body contact-onset accounting from refreshed MuJoCo constraint forces."""

import mujoco
import numpy as np


class CollisionTracker:
    FORCE_THRESHOLD_N = 0.1
    TABLE_CLEARANCE_M = 0.001

    def __init__(self, model, data, target_body: int, fingers: set[int], table: int) -> None:
        self.model = model
        self.data = data
        self.target_body = target_body
        self.fingers = frozenset(fingers)
        self.table = table
        self.robot_bodies = frozenset(
            body for body in range(model.nbody)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or "").startswith("robot_")
        )
        self.body_names = tuple(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or f"body_{body}"
            for body in range(model.nbody)
        )
        collidable = (model.geom_contype | model.geom_conaffinity) != 0
        self.target_geoms = np.flatnonzero((model.geom_bodyid == target_body) & collidable)
        self.table_geoms = np.flatnonzero((model.geom_bodyid == table) & collidable)
        # These workcell assets use box collision pieces, including the table.
        # Reject other shapes rather than silently using incorrect extents.
        for geoms in (self.target_geoms, self.table_geoms):
            if not len(geoms) or np.any(model.geom_type[geoms] != mujoco.mjtGeom.mjGEOM_BOX):
                raise ValueError("Table clearance requires box collision geometry")
        self.force = np.zeros(6)
        self.reset()

    def reset(self) -> None:
        self.active_pairs = set()
        self.records = {}
        self.duration_s = 0.0
        self.seen_table_support = False
        self.cleared_table = False

    def _unwanted(self, a: int, b: int, allow_grip: bool = True) -> bool:
        # Bilateral finger-target contacts implement the grasp, not a collision.
        # Initial target-table support is normal, but all later recontacts count.
        # Non-target supports (bottle-table, cabinet-table, etc.) are unrelated.
        # No robot self/environment/other-object contact is silently excluded.
        if self.target_body in (a, b):
            other = b if a == self.target_body else a
            if other in self.fingers and allow_grip:
                return False
            if other == self.table:
                return self.cleared_table
            return True
        return a in self.robot_bodies or b in self.robot_bodies

    def _clear_of_table(self) -> bool:
        # Exact world-Z support extents of every oriented collision box, not
        # body-origin height or a temporarily empty contact set. A 1 mm gap
        # (the asset contact-impedance scale) prevents sub-mm support chatter
        # during rim closure from being mislabeled as lift and recontact.
        def extents(geoms):
            rotations = self.data.geom_xmat[geoms].reshape(-1, 3, 3)
            return np.sum(np.abs(rotations[:, 2, :]) * self.model.geom_size[geoms], axis=1)

        lowest = np.min(self.data.geom_xpos[self.target_geoms, 2] - extents(self.target_geoms))
        highest = np.max(self.data.geom_xpos[self.table_geoms, 2] + extents(self.table_geoms))
        return bool(lowest - highest > self.TABLE_CLEARANCE_M)

    def update(self, dt: float, skill=None) -> None:
        """Call once after mj_forward for each completed execution/hold interval."""
        forces = {}
        table_contact = False
        storage = skill.storage if skill is not None else None
        allow_grip = storage is None or not storage.released
        for index, contact in enumerate(self.data.contact):
            a = int(self.model.geom_bodyid[contact.geom1])
            b = int(self.model.geom_bodyid[contact.geom2])
            if {a, b} == {self.target_body, self.table}:
                # Even a margin contact delays clearance: zero force alone is
                # not evidence that the object has physically left the table.
                table_contact = True
            planned = storage is not None and storage.planned_contact(
                int(contact.geom1), int(contact.geom2), skill.phase)
            if contact.efc_address < 0 or planned or not self._unwanted(a, b, allow_grip):
                continue
            mujoco.mj_contactForce(self.model, self.data, index, self.force)
            pair = tuple(sorted((a, b), key=self.body_names.__getitem__))
            forces[pair] = forces.get(pair, 0.0) + max(0.0, float(self.force[0]))
        if table_contact:
            self.seen_table_support = True
        elif self.seen_table_support and not self.cleared_table and self._clear_of_table():
            self.cleared_table = True
        current = {pair for pair, force in forces.items() if force > self.FORCE_THRESHOLD_N}
        if current:
            self.duration_s += dt
        for pair in current:
            record = self.records.setdefault(pair, {
                "body_a": self.body_names[pair[0]], "body_b": self.body_names[pair[1]],
                "episodes": 0, "duration_s": 0.0, "peak_force_n": 0.0,
            })
            record["episodes"] += int(pair not in self.active_pairs)
            record["duration_s"] += dt
            record["peak_force_n"] = max(record["peak_force_n"], forces[pair])
        self.active_pairs = current

    def state(self) -> dict:
        pairs = sorted(self.records.values(), key=lambda pair: (pair["body_a"], pair["body_b"]))
        return {
            "episodes": sum(pair["episodes"] for pair in pairs),
            "duration_s": self.duration_s,
            "peak_force_n": max((pair["peak_force_n"] for pair in pairs), default=0.0),
            "pairs": [dict(pair) for pair in pairs],
            "force_threshold_n": self.FORCE_THRESHOLD_N,
        }
