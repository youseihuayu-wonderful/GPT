"""Contact-driven bowl placement, drawer closure, and arm-home verification."""

import mujoco
import numpy as np

from task_specs import TASKS


class BowlStorage:
    PUSH_ORIENTATION = np.array([0.3535533906, -0.6123724357, -0.6123724357, -0.3535533906])

    def __init__(self, skill):
        self.skill = skill
        self.sim = skill.sim
        self.model, self.data = skill.model, skill.data
        self.drawer = self.model.body("cabinet_cabinet_bottom").id
        joint = self.model.joint("cabinet_bottom_level")
        self.drawer_qpos = int(joint.qposadr[0])
        self.drawer_dof = int(joint.dofadr[0])
        self.region = self.model.site("cabinet_bottom_region").id
        collidable = (self.model.geom_contype | self.model.geom_conaffinity) != 0
        self.bowl_body = self.model.body("bowl_object").id
        self.bowl_dof = int(self.model.joint("bowl_free").dofadr[0])
        self.bowl_geoms = np.flatnonzero((self.model.geom_bodyid == self.bowl_body) & collidable)
        hand_bodies = [self.model.body("robot_hand").id, *skill.fingers]
        self.hand_geoms = np.flatnonzero(np.isin(self.model.geom_bodyid, hand_bodies) & collidable)
        self.object_bodies = frozenset(self.model.body(name).id for name in ("bowl_object", "bottle_object"))
        self.drawer_geoms = np.flatnonzero((self.model.geom_bodyid == self.drawer) & collidable)
        self.object_geoms = {
            body: np.flatnonzero((self.model.geom_bodyid == body) & collidable)
            for body in self.object_bodies
        }
        # Identify individual original collision pieces, never the whole drawer.
        self.floor = int(min(self.drawer_geoms, key=lambda g: self.model.geom_pos[g, 2]))
        self.handle = int(min(self.drawer_geoms, key=lambda g: self.model.geom_pos[g, 1]))
        self.finger_geoms = frozenset(np.flatnonzero(np.isin(self.model.geom_bodyid, list(skill.fingers)) & collidable))
        self.released = False
        self.verification_hold_s = 0.0
        self.force = np.zeros(6)
        self.waypoints = []
        self.segment_started = 0.0
        self.initial_drawer_position = float(self.data.qpos[self.drawer_qpos])
        self.push_contact_seen = False
        self.transfer_aligned = False

    def bounds(self, geoms):
        geoms = np.asarray(geoms, dtype=int)
        rotation = self.data.geom_xmat[geoms].reshape(-1, 3, 3)
        local = self.model.geom_aabb[geoms]
        center = self.data.geom_xpos[geoms] + np.einsum("nij,nj->ni", rotation, local[:, :3])
        extent = np.einsum("nij,nj->ni", np.abs(rotation), local[:, 3:])
        return np.min(center - extent, axis=0), np.max(center + extent, axis=0)

    def region_bounds(self):
        rotation = self.data.site_xmat[self.region].reshape(3, 3)
        extent = np.abs(rotation) @ self.model.site_size[self.region]
        center = self.data.site_xpos[self.region]
        return center - extent, center + extent

    def drawer_fully_open(self):
        return bool(self.data.qpos[self.drawer_qpos] <= -0.155)

    def drawer_state(self):
        position = float(self.data.qpos[self.drawer_qpos])
        velocity = float(self.data.qvel[self.drawer_dof])
        return {
            "position_m": position,
            "velocity_m_s": velocity,
            "fully_open": self.drawer_fully_open(),
            "closed": bool(abs(position) <= 0.003 and abs(velocity) < 0.005),
        }

    def validate(self, parameters, held=False):
        if not self.drawer_fully_open():
            raise ValueError("Bottom drawer must initially be fully open")
        minimum = TASKS["store_bowl"].minimum_opening_m
        if parameters.initial_opening_m < minimum:
            raise ValueError(f"Bowl storage requires at least {minimum * 1000:g} mm initial opening")
        if np.any(self.model.geom_type[self.bowl_geoms] != mujoco.mjtGeom.mjGEOM_BOX):
            raise ValueError("Storage requires measured box collision geometry")
        low, high = self.bounds(self.bowl_geoms)
        region_low, region_high = self.region_bounds()
        floor_top = self.bounds([self.floor])[1][2]
        available = region_high - region_low
        available[2] = region_high[2] - floor_top
        # A held bowl tilts; the original path reorients it before insertion.
        # Fit its original collision dimensions, while keeping live containment
        # and contact checks for actual placement.
        size = high - low
        if held:
            rotation = self.data.xmat[self.bowl_body].reshape(3, 3)
            centers = (self.data.geom_xpos[self.bowl_geoms] - self.data.xpos[self.bowl_body]) @ rotation
            local_rotation = np.einsum(
                "ij,njk->nik", rotation.T, self.data.geom_xmat[self.bowl_geoms].reshape(-1, 3, 3))
            extent = np.einsum("nij,nj->ni", np.abs(local_rotation), self.model.geom_size[self.bowl_geoms])
            size = np.max(centers + extent, axis=0) - np.min(centers - extent, axis=0)
        if np.any(size + 0.003 > available):
            raise ValueError("Bowl collision geometry does not fit the original drawer")

    def planned_contact(self, a, b, phase):
        if self.floor in (a, b):
            other = b if a == self.floor else a
            if other in self.bowl_geoms:
                return phase in ("place", "release", "withdraw", "push_approach", "push_drawer", "push_withdraw", "return_home", "verify_store", "done")
        if self.handle in (a, b):
            other = b if a == self.handle else a
            return other in self.finger_geoms and phase in ("push_drawer", "push_withdraw")
        return False

    def support_and_clear(self):
        supported, clear = False, True
        for index, contact in enumerate(self.data.contact):
            a, b = int(contact.geom1), int(contact.geom2)
            if a not in self.bowl_geoms and b not in self.bowl_geoms:
                continue
            other = b if a in self.bowl_geoms else a
            if contact.efc_address < 0:
                continue
            mujoco.mj_contactForce(self.model, self.data, index, self.force)
            if other in self.finger_geoms and (contact.dist <= 0 or self.force[0] > 0.01):
                clear = False
            if other == self.floor and self.force[0] > 0.01:
                supported = True
        return supported, clear

    def empty_hand(self):
        for index, contact in enumerate(self.data.contact):
            bodies = {int(self.model.geom_bodyid[contact.geom1]), int(self.model.geom_bodyid[contact.geom2])}
            if bodies & self.skill.fingers and bodies & self.object_bodies and contact.efc_address >= 0:
                mujoco.mj_contactForce(self.model, self.data, index, self.force)
                if self.force[0] > 0.01:
                    return False
        return True

    def hand_clear(self):
        low, high = self.region_bounds()
        # Include the full interior sweep from its live position to closed.
        high = high.copy()
        high[1] += max(0.0, -float(self.data.qpos[self.drawer_qpos]))
        for geom in self.hand_geoms:
            hand_low, hand_high = self.bounds([geom])
            if np.all(hand_high >= low - 0.005) and np.all(hand_low <= high + 0.005):
                return False
        return True

    def bowl_settled(self):
        velocity = self.data.qvel[self.bowl_dof:self.bowl_dof + 6]
        return bool(np.linalg.norm(velocity[:3]) < 0.01 and np.linalg.norm(velocity[3:]) < 0.1)

    def support_preserved(self):
        velocity = self.data.qvel[self.skill.target_dof:self.skill.target_dof + 6]
        stable = np.linalg.norm(velocity[:3]) < 0.01 and np.linalg.norm(velocity[3:]) < 0.1
        _, on_table = self.skill._contacts()
        state = self.state()
        return bool(stable and (on_table or (
            self.skill.target.name == "bowl" and state["bowl_inside_drawer"]
            and state["bowl_supported"] and self.bowl_settled())))

    def precondition_error(self, name):
        state = self.state()
        if name == "place_and_release":
            if not all(self.skill.grasp_postconditions().values()):
                return "Placement requires a physically verified stable bowl grasp"
            try:
                self.validate(self.skill.parameters, held=True)
            except ValueError as exc:
                return str(exc)
        elif name == "withdraw_from_drawer":
            if not (state["bowl_inside_drawer"] and state["bowl_supported"]
                    and self.support_and_clear()[1] and self.empty_hand() and self.bowl_settled()):
                return "Withdrawal requires a contained, supported, settled and detached bowl"
            if state["hand_clear_of_drawer"]:
                return "Hand is already clear of the drawer"
            tcp = self.data.site_xpos[self.skill.tcp]
            low, high = self.region_bounds()
            if not (low[0] <= tcp[0] <= high[0] and low[1] <= tcp[1] <= high[1]
                    and tcp[2] <= high[2] + 0.15):
                return "Hand is outside the supported placement-to-withdrawal handoff"
        elif name in ("close_drawer", "return_home"):
            if not self.empty_hand() or self.skill.gripper_closed:
                return "This skill requires an empty released gripper"
            if not state["hand_clear_of_drawer"]:
                return "Withdraw the hand from the drawer before closing or returning home"
            if self.data.site_xpos[self.skill.tcp, 2] < 1.03:
                return "This workcell skill requires the verified elevated handoff (TCP >= 1.03 m)"
            if not self.support_preserved():
                return "Target must remain stably supported on the table or inside the drawer"
            if name == "close_drawer":
                if not self.drawer_fully_open():
                    return "Closing requires the fully open drawer; opening or partial-drawer approaches are not supported"
                if not self.closing_corridor_clear():
                    return "An object obstructs the drawer closing corridor"
                if not self.push_corridor_clear():
                    return "An object obstructs the fixed hand/palm pushing approach"
        return None

    def closing_corridor_clear(self):
        state = self.state()
        travel = max(0.0, -float(self.data.qpos[self.drawer_qpos]))
        # Sweep each original piece, not a solid envelope that fills the gap
        # beside the protruding handle and falsely intersects the table bowl.
        for drawer_geom in self.drawer_geoms:
            low, high = self.bounds([drawer_geom])
            high[1] += travel
            for body, geoms in self.object_geoms.items():
                if body == self.bowl_body and state["bowl_inside_drawer"] and state["bowl_supported"]:
                    continue
                for geom in geoms:
                    object_low, object_high = self.bounds([geom])
                    if np.all(object_high >= low - 0.003) and np.all(object_low <= high + 0.003):
                        return False
        return True

    def push_corridor_clear(self):
        # Bound the lowered approach/push, using the actual hand mesh boxes
        # transformed to the fixed pushing orientation plus tracking margin.
        # High-level waypoints and runtime contacts still bound this workcell
        # controller; this is not a general joint-space collision planner.
        target_rotation = np.empty(9)
        mujoco.mju_quat2Mat(target_rotation, self.PUSH_ORIENTATION)
        delta = target_rotation.reshape(3, 3) @ self.data.site_xmat[self.skill.tcp].reshape(3, 3).T
        contact_x = float(self.bounds([self.handle])[1][0] - 0.014)
        tcp_low = np.array([contact_x, -0.08, self.data.geom_xpos[self.handle, 2]])
        tcp_high = np.array([contact_x, 0.12, 1.05])
        state = self.state()
        for hand_geom in self.hand_geoms:
            rotation = self.data.geom_xmat[hand_geom].reshape(3, 3)
            box = self.model.geom_aabb[hand_geom]
            center = self.data.geom_xpos[hand_geom] + rotation @ box[:3]
            offset = delta @ (center - self.data.site_xpos[self.skill.tcp])
            extent = np.abs(delta @ rotation) @ box[3:]
            low, high = tcp_low + offset - extent - 0.01, tcp_high + offset + extent + 0.01
            for body, geoms in self.object_geoms.items():
                if body == self.bowl_body and state["bowl_inside_drawer"] and state["bowl_supported"]:
                    continue
                for geom in geoms:
                    object_low, object_high = self.bounds([geom])
                    if np.all(object_high >= low) and np.all(object_low <= high):
                        return False
        return True

    def state(self):
        low, high = self.bounds(self.bowl_geoms)
        region_low, region_high = self.region_bounds()
        inside = bool(np.all(low >= region_low) and np.all(high <= region_high))
        supported, clear = self.support_and_clear()
        drawer = self.drawer_state()
        q = drawer["position_m"]
        home_error = float(np.max(np.abs(self.data.qpos[self.sim.arm_qpos] - self.sim.arm_home)))
        return {
            "bowl_inside_drawer": inside, "bowl_supported": supported,
            "gripper_clear": clear and self.released,
            "drawer_position_m": q,
            "hand_clear_of_drawer": self.hand_clear(),
            "hand_empty": self.empty_hand(),
            "drawer_closed": drawer["closed"],
            "home_error_rad": home_error,
            "home_reached": bool(home_error <= 0.03 and np.max(np.abs(self.data.qvel[self.skill.arm_dof])) < 0.03),
            "verification_hold_s": self.verification_hold_s,
            "drawer_travel_m": q - self.initial_drawer_position,
            "push_contact_seen": self.push_contact_seen,
        }

    def move(self, phase, points):
        phase_started = self.skill.phase_started_at if self.skill.phase == phase else None
        self.waypoints = [np.array(point, dtype=float) for point in points]
        self.skill._move(phase, self.waypoints.pop(0))
        if phase_started is not None and self.skill.status == "running":
            self.skill.phase_started_at = phase_started
        self.segment_started = float(self.data.time)

    def motion_done(self):
        skill = self.skill
        if not skill._motion_done(float(self.data.time) - self.segment_started):
            return False
        if self.waypoints:
            # Internal waypoints belong to one public phase and one phase timer.
            phase_started = skill.phase_started_at
            skill._move(skill.phase, self.waypoints.pop(0))
            skill.phase_started_at = phase_started
            self.segment_started = float(self.data.time)
            return False
        return True

    def begin(self):
        tcp = self.data.site_xpos[self.skill.tcp].copy()
        # Across-drawer fingers keep the long palm clear of upper handles.
        # Raise before translating; the tilted held bowl cannot pass under them.
        self.skill.orientation_target = np.array([0.0, np.sqrt(0.5), np.sqrt(0.5), 0.0])
        self.move("transfer", [[tcp[0], tcp[1], 1.05]])

    def begin_withdrawal(self):
        tcp = self.data.site_xpos[self.skill.tcp].copy()
        tcp[2] = 1.05
        self.move("withdraw", [tcp])

    def begin_closing(self):
        skill = self.skill
        skill.gripper_closed = False
        skill.opening_target = 0.008
        skill.orientation_target = self.PUSH_ORIENTATION
        self.initial_drawer_position = float(self.data.qpos[self.drawer_qpos])
        self.push_contact_seen = False
        handle = self.data.geom_xpos[self.handle]
        # Push near the measured right end: a center push sweeps the wide palm
        # through the table bowl when closing independently of prior grasping.
        contact_x = float(self.bounds([self.handle])[1][0] - 0.014)
        self.move("push_approach", [[contact_x, -0.08, 1.05], [contact_x, -0.08, handle[2]]])

    def begin_home(self):
        skill = self.skill
        skill.motion_start = self.data.qpos[self.sim.arm_qpos].copy()
        skill.motion_target = self.sim.arm_home.copy()
        skill.motion_duration = max(2.0, 1.875 * np.max(np.abs(skill.motion_target - skill.motion_start)) / 0.6)
        skill._enter("return_home")

    def postconditions(self, name):
        state = self.state()
        if name == "place_and_release":
            return {
                "bowl_inside_drawer": state["bowl_inside_drawer"],
                "bowl_supported": state["bowl_supported"],
                "gripper_detached": state["gripper_clear"] and self.empty_hand(),
                "bowl_settled": self.bowl_settled(),
            }
        if name == "withdraw_from_drawer":
            return {
                **self.postconditions("place_and_release"),
                "hand_clear_of_drawer": state["hand_clear_of_drawer"],
                "elevated_handoff": bool(self.data.site_xpos[self.skill.tcp, 2] >= 1.03),
            }
        if name == "close_drawer":
            return {
                "drawer_closed": state["drawer_closed"],
                "push_contact_seen": self.push_contact_seen,
                "drawer_moved": state["drawer_travel_m"] >= 0.15,
                "hand_clear_of_drawer": state["hand_clear_of_drawer"],
                "hand_empty": state["hand_empty"],
                "support_preserved": self.support_preserved(),
            }
        if name == "return_home":
            return {
                "home_reached": state["home_reached"],
                "hand_clear_of_drawer": state["hand_clear_of_drawer"],
                "hand_empty": state["hand_empty"],
                "support_preserved": self.support_preserved(),
            }
        raise ValueError(f"Unknown storage skill: {name}")

    def final_valid(self):
        state = self.state()
        velocity = self.data.qvel[self.skill.target_dof:self.skill.target_dof + 6]
        return (all(state[key] for key in ("bowl_inside_drawer", "bowl_supported", "gripper_clear", "drawer_closed", "home_reached"))
                and np.linalg.norm(velocity[:3]) < 0.01 and np.linalg.norm(velocity[3:]) < 0.1
                and self.push_contact_seen and not self.sim.collisions.records)

    def update(self, dt, elapsed):
        skill = self.skill
        phase = skill.phase
        if phase in ("transfer", "place"):
            bilateral, _ = skill._contacts()
            skill.lost_contact_elapsed = 0.0 if bilateral else skill.lost_contact_elapsed + dt
            if skill.lost_contact_elapsed > 0.15:
                skill.fail("Lost bowl grip before placement")
                return
        if phase == "transfer" and self.motion_done():
            low, high = self.bounds(self.bowl_geoms)
            if not self.transfer_aligned:
                tcp = self.data.site_xpos[skill.tcp].copy()
                center = (low + high) / 2
                # Measured exposed-front corridor of the original fully-open
                # drawer. Center the complete held geometry, not the body origin.
                tcp[:2] += np.array([0.65328, 0.075]) - center[:2]
                self.transfer_aligned = True
                self.move("transfer", [tcp])
                return
            floor_top = self.bounds([self.floor])[1][2]
            tcp = self.data.site_xpos[skill.tcp].copy()
            # A 1 mm servo overtravel establishes real support; contact feedback
            # stops descent immediately, before opening either finger.
            tcp[2] += floor_top - low[2] - 0.001
            self.move("place", [tcp])
        elif phase == "place":
            arrived = self.motion_done()
            if skill.status != "running":
                return
            if self.support_and_clear()[0]:
                skill.arm_target = self.data.qpos[self.sim.arm_qpos].copy()
                skill.gripper_closed = False
                skill.opening_target = 0.050
                skill._enter("release")
            elif arrived:
                skill.fail("Placement reached its measured floor height without support")
        elif phase == "release":
            supported, clear = self.support_and_clear()
            velocity = self.data.qvel[skill.target_dof:skill.target_dof + 6]
            ready = supported and clear and np.linalg.norm(velocity[:3]) < 0.01 and np.linalg.norm(velocity[3:]) < 0.1
            skill.stable_elapsed = skill.stable_elapsed + dt if ready else 0.0
            if skill.stable_elapsed >= 0.3:
                self.released = True
                if not self.state()["bowl_inside_drawer"]:
                    skill.fail("Released bowl is not fully contained in drawer")
                    return
                skill.complete_skill()
            elif elapsed > 4:
                skill.fail("Bowl did not settle supported and detached")
        elif phase == "withdraw" and self.motion_done():
            skill.complete_skill()
        elif phase == "push_approach" and self.motion_done():
            tcp = self.data.site_xpos[skill.tcp].copy()
            tcp[1] = 0.12
            self.move("push_drawer", [tcp])
            # The original 3 kg / 50 N s/m drawer needs a slow contact push.
            # This reduces contact load without changing any physical limit.
            skill.motion_duration = max(skill.motion_duration, 1.875 * 0.20 / 0.025)
        elif phase == "push_drawer":
            for index, contact in enumerate(self.data.contact):
                geoms = (int(contact.geom1), int(contact.geom2))
                if (self.handle in geoms and self.planned_contact(*geoms, phase)
                        and contact.efc_address >= 0):
                    mujoco.mj_contactForce(self.model, self.data, index, self.force)
                    if self.force[0] > 0.1:
                        self.push_contact_seen = True
            if self.data.qpos[self.drawer_qpos] >= -0.001:
                tcp = self.data.site_xpos[skill.tcp].copy()
                tcp[1] -= 0.07
                self.move("push_withdraw", [tcp, [tcp[0], tcp[1], 1.05]])
            elif self.motion_done():
                skill.fail("Finger push did not close the drawer")
        elif phase == "push_withdraw" and self.motion_done():
            skill.complete_skill()
        elif phase == "return_home":
            u = min(1.0, elapsed / skill.motion_duration)
            blend = u**3 * (10 - 15 * u + 6 * u**2)
            skill.arm_target = skill.motion_start + blend * (skill.motion_target - skill.motion_start)
            if elapsed >= skill.motion_duration and self.state()["home_reached"]:
                skill.complete_skill()
            elif elapsed > skill.motion_duration + 2:
                skill.fail("Arm did not reach home")
        elif phase in ("verify_store", "done"):
            if not self.final_valid():
                self.verification_hold_s = 0.0
                skill.fail("Stored-state verification lost: containment/support/release/closure/home")
                return
            self.verification_hold_s += dt
            if phase == "verify_store" and self.verification_hold_s >= 1.0:
                skill._enter("done")
                skill.status = "succeeded"
                skill.finished_at = float(self.data.time)
