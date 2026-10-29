"""Deterministic authorization tests; provider fixtures are not live GPT evidence."""

import asyncio
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from planner import PlannerConfig, PlannerError
from sorting import SortingSimulation
from sorting_contract import FIXED_PLAN, OBJECTS, validate_plan
from sorting_planning import (
    SortingDecision, SortingDecisionResult, SortingPlanningSession, request_sorting_plan,
    sorting_scene_matches, validate_sorting_decision,
)


COLOR_MAP = {"red": "B", "blue": "C", "yellow": "A"}


def proposal():
    return {
        "decision": "plan", "summary": "Test-only reversed order and rotated destinations",
        "reason": "Controlled fixture, not evidence of language understanding or physical success.",
        "steps": [{"object_id": obj.id, "destination": COLOR_MAP[obj.color]} for obj in reversed(OBJECTS)],
    }


def result(payload=None):
    return SortingDecisionResult(
        validate_sorting_decision(payload if payload is not None else proposal()),
        "requested-fixture-model", "fixture-subscription-thread", 0.02, 2,
    )


class ObservedSortingSimulation:
    """State-machine fixture; the physical integration case below uses MuJoCo."""

    def __init__(self):
        self.plan = FIXED_PLAN
        self.skill = SimpleNamespace(status="idle", precondition_error=Mock(return_value=None))
        self.scene = {
            "time_s": 0.6, "scene_revision": 1, "warnings": [],
            "objects": [{"name": obj.body_name, "position_m": [0.1 * index, -0.1, 0.8],
                         "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]} for index, obj in enumerate(OBJECTS)],
            "joint_positions_rad": [0.0] * 7, "gripper_opening_m": 0.08,
            "sorting": {
                "status": "idle", "mode": "fixed", "elapsed_s": 0.0, "failure_reason": None,
                "goal_checks": {"all_correct": False, "all_supported": True, "all_stable": True,
                                "hand_empty": True, "home_reached": True},
                "objects": [{"id": obj.id, "body_name": obj.body_name, "kind": obj.kind, "color": obj.color,
                             "supported": True, "stable": True, "detached": True, "sorted": False}
                            for obj in OBJECTS],
            },
        }
        self.numerical_error = Mock(return_value=None)
        self.state = Mock(side_effect=self._state)
        self.start_sorting = Mock(side_effect=self._start_sorting)
        self.step = Mock(side_effect=AssertionError("Session must not tick physics"))

    def _state(self):
        self.scene["sorting"]["status"] = self.skill.status
        return deepcopy(self.scene)

    def _start_sorting(self, plan=None):
        self.plan = plan if plan is not None else FIXED_PLAN
        self.skill.status = "running"
        self.scene["sorting"]["mode"] = "planned" if plan is not None else "fixed"

    def succeed(self):
        self.skill.status = "succeeded"
        self.scene["time_s"] += 20
        self.scene["sorting"]["elapsed_s"] = 20.0
        self.scene["sorting"]["goal_checks"] = dict.fromkeys(self.scene["sorting"]["goal_checks"], True)
        for fact in self.scene["sorting"]["objects"]:
            fact["sorted"] = True


class SortingDecisionTests(unittest.TestCase):
    def test_complete_proposal_preserves_actual_order_and_destination_mapping(self):
        payload = proposal()
        decision = validate_sorting_decision(payload)
        self.assertEqual(decision.to_dict(), payload)
        self.assertEqual(decision.plan.steps[0].object_id, OBJECTS[-1].id)
        self.assertNotEqual(decision.plan, FIXED_PLAN)
        for obj in OBJECTS:
            self.assertEqual(decision.plan.assignment(obj.id).destination, COLOR_MAP[obj.color])
        payload["steps"][0]["destination"] = "C"
        self.assertNotEqual(payload, decision.to_dict())

    def test_unsupported_cannot_hide_a_partial_plan(self):
        payload = dict(proposal(), decision="unsupported", steps=[])
        self.assertIsNone(validate_sorting_decision(payload).plan)
        self.assertEqual(validate_sorting_decision(payload).to_dict(), payload)
        for steps in (proposal()["steps"], [proposal()["steps"][0]], None, (), {}):
            with self.subTest(steps=steps), self.assertRaises(PlannerError):
                validate_sorting_decision(dict(payload, steps=steps))

    def test_incomplete_duplicate_mixed_color_and_extra_commands_are_rejected(self):
        original = proposal()
        malformed = []
        for key in original:
            payload = deepcopy(original)
            del payload[key]
            malformed.append(payload)
        malformed.append(dict(original, joint_commands=[0] * 7))
        malformed.append(dict(original, steps=original["steps"][:-1]))
        malformed.append(dict(original, steps=original["steps"] + [original["steps"][0]]))
        for field, value in (("object_id", "unknown"), ("destination", "D"), ("position_m", [0, 0, 1])):
            payload = deepcopy(original)
            payload["steps"][0][field] = value
            malformed.append(payload)
        duplicate = deepcopy(original)
        duplicate["steps"][1] = duplicate["steps"][0]
        malformed.append(duplicate)
        mixed = deepcopy(original)
        mixed["steps"][0]["destination"] = "B"
        malformed.append(mixed)
        merged = deepcopy(original)
        for step in merged["steps"]:
            step["destination"] = "A"
        malformed.append(merged)
        for payload in malformed:
            with self.subTest(payload=payload), self.assertRaises(PlannerError):
                validate_sorting_decision(payload)

    def test_text_and_decision_types_are_bounded_even_on_direct_construction(self):
        for field, limit in (("summary", 300), ("reason", 600)):
            for value in (None, True, "", " \n", [], "字" * (limit + 1)):
                with self.subTest(field=field, value=value), self.assertRaises(PlannerError):
                    validate_sorting_decision(dict(proposal(), **{field: value}))
        for operation in (None, True, [], "execute", "complete"):
            with self.subTest(operation=operation), self.assertRaises(PlannerError):
                validate_sorting_decision(dict(proposal(), decision=operation))
        for operation, plan in (("plan", None), ("unsupported", FIXED_PLAN), ("plan", FIXED_PLAN.to_list())):
            with self.subTest(operation=operation), self.assertRaises(PlannerError):
                SortingDecision(operation, "Fixture", "Fixture", plan)


class SortingObservationTests(unittest.TestCase):
    def test_small_settling_and_quaternion_sign_are_not_scene_changes(self):
        before = ObservedSortingSimulation().state()
        after = deepcopy(before)
        after["time_s"] += 60
        after["objects"][0]["position_m"][0] += 0.001
        after["objects"][0]["quaternion_wxyz"] = [-1.0, 0.0, 0.0, 0.0]
        after["joint_positions_rad"][0] += 0.005
        after["gripper_opening_m"] -= 0.001
        self.assertTrue(sorting_scene_matches(before, after))

    def test_revision_identity_position_orientation_joint_and_opening_drift_are_stale(self):
        before = ObservedSortingSimulation().state()
        changes = [
            lambda scene: scene.update(scene_revision=2),
            lambda scene: scene["objects"][0].update(name="unknown"),
            lambda scene: scene["sorting"]["objects"][0].update(color="blue"),
            lambda scene: scene["objects"][0]["position_m"].__setitem__(0, 0.0051),
            lambda scene: scene["objects"][0].update(quaternion_wxyz=[math.cos(0.026), 0.0, 0.0, math.sin(0.026)]),
            lambda scene: scene["joint_positions_rad"].__setitem__(0, 0.0301),
            lambda scene: scene.update(gripper_opening_m=0.0769),
            lambda scene: scene["objects"][0]["position_m"].__setitem__(0, float("nan")),
            lambda scene: scene["objects"][0].update(quaternion_wxyz=[2.0, 0.0, 0.0, 0.0]),
        ]
        for change in changes:
            after = deepcopy(before)
            change(after)
            with self.subTest(change=change):
                self.assertFalse(sorting_scene_matches(before, after))


class SortingSubscriptionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.config = PlannerConfig("test-only-codex", "requested-fixture-model")
        self.events = [
            {"type": "thread.started", "thread_id": "test-completed-thread"},
            {"type": "turn.started"},
            {"type": "item.completed", "item": {"type": "agent_message", "text": "structured fixture"}},
            {"type": "turn.completed"},
        ]
        self.payload = json.dumps(proposal()).encode()
        self.login = (0, b"", b"Logged in using ChatGPT")
        self.code, self.stderr = 0, b""
        self.sent = None
        runner = patch("planner._run_cli", new=AsyncMock(side_effect=self.run_cli))
        self.runner = runner.start()
        self.addCleanup(runner.stop)

    async def run_cli(self, command, directory, environment, input_data=None, timeout=90):
        if command[1:3] == ["login", "status"]:
            return self.login
        self.sent = json.loads(input_data)
        if self.payload is not None:
            (directory / "plan.json").write_bytes(self.payload)
        return self.code, b"\n".join(json.dumps(event).encode() for event in self.events), self.stderr

    async def request(self):
        return await request_sorting_plan(self.config, "红B蓝C黄A，完整倒序", ObservedSortingSimulation().state())

    async def test_six_object_adapter_uses_real_transport_interface_and_preserves_evidence(self):
        self.events.insert(1, {"type": "item.completed", "item": {"type": "error", "message": "private-diagnostic"}})
        answer = await self.request()
        self.assertEqual(answer.decision.to_dict(), proposal())
        self.assertEqual(answer.model, self.config.model)
        self.assertEqual(answer.thread_id, "test-completed-thread")
        self.assertEqual(answer.diagnostic_count, 1)
        self.assertGreaterEqual(answer.latency_s, 0)
        self.assertEqual(len(self.sent["capabilities"]["objects"]), 6)
        self.assertEqual(self.sent["capabilities"]["reserved_slots"], {"bowl": 0, "cup": 1})
        self.assertTrue(self.sent["capabilities"]["approval_required"])
        self.assertNotIn("private-diagnostic", repr(answer))

    async def test_invalid_model_json_or_partial_plan_is_never_a_fallback(self):
        for payload in (None, b"{}", b'{"decision":"plan","decision":"unsupported"}',
                        b'{"steps":NaN}', json.dumps(dict(proposal(), steps=[])).encode()):
            self.payload = payload
            with self.subTest(payload=payload), self.assertRaises(PlannerError):
                await self.request()

    async def test_api_login_failed_turn_and_tool_events_are_rejected(self):
        self.login = (0, b"", b"Logged in using an API key")
        with self.assertRaises(PlannerError):
            await self.request()
        self.login = (0, b"", b"Logged in using ChatGPT")
        for event in ({"type": "turn.failed"},
                      {"type": "item.completed", "item": {"type": "command_execution"}}):
            original = self.events
            self.events = [*original[:2], event, original[-1]]
            with self.subTest(event=event), self.assertRaises(PlannerError):
                await self.request()
            self.events = original

    async def test_subscription_quota_failure_never_switches_to_api_or_retries(self):
        self.code, self.stderr = 1, b"429 quota private-provider-diagnostic"
        with self.assertRaises(PlannerError) as caught:
            await self.request()
        self.assertIn("不会改用 API", str(caught.exception))
        self.assertNotIn("private-provider-diagnostic", str(caught.exception))
        self.assertEqual(self.runner.await_count, 2)

    async def test_invalid_or_incomplete_scene_never_contacts_provider(self):
        scene = ObservedSortingSimulation().state()
        scene["objects"].pop()
        with self.assertRaises(PlannerError):
            await request_sorting_plan(self.config, "分拣", scene)
        self.runner.assert_not_awaited()


class SortingPlanningSessionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        config = patch("sorting_planning.PlannerConfig.load", return_value=PlannerConfig("test-only-codex"))
        self.loader = config.start()
        self.addCleanup(config.stop)
        self.answer = result()
        provider = patch("sorting_planning.request_sorting_plan", new=AsyncMock(return_value=self.answer))
        self.request = provider.start()
        self.addCleanup(provider.stop)
        self.sim = ObservedSortingSimulation()
        self.session = SortingPlanningSession(Path(__file__).parent, self.sim)
        self.addAsyncCleanup(self.session.close)

    async def ready(self):
        self.assertTrue(self.session.start("按实际预览批准完整六物体方案"))
        await self.session._pending
        self.assertEqual(self.session.state()["status"], "ready")
        return self.session.state()["decision_id"]

    async def test_construction_reset_and_fixed_simulation_never_request_provider(self):
        self.session.state()
        self.session.reset()
        self.sim.start_sorting()
        self.request.assert_not_awaited()
        self.assertEqual(self.session.state()["calls"], 0)
        self.assertEqual(self.session.state()["mode"], "sorting")

    async def test_no_motion_before_explicit_approval_and_no_duplicate_dispatch(self):
        decision_id = await self.ready()
        for allowed in (True, False, True):
            self.session.advance(allow_execution=allowed)
        self.sim.start_sorting.assert_not_called()
        self.sim.step.assert_not_called()
        state = self.session.state()
        self.assertEqual(state["decision"], proposal())
        self.assertTrue(state["locks_scene"])
        self.assertTrue(state["can_cancel"])
        for invalid_id in (None, True, str(decision_id), float(decision_id), decision_id - 1):
            with self.subTest(decision_id=invalid_id), self.assertRaises(ValueError):
                self.session.execute(invalid_id)
        self.session.execute(decision_id)
        self.sim.start_sorting.assert_called_once_with(self.answer.decision.plan)
        self.assertEqual(self.session.state()["status"], "executing")
        self.assertFalse(self.session.state()["can_cancel"])
        with self.assertRaises(ValueError):
            self.session.execute(decision_id)
        with self.assertRaisesRegex(ValueError, "暂停或重置"):
            self.session.cancel()
        self.assertEqual(self.session.state()["status"], "executing")

    async def test_request_and_final_observations_keep_proposal_and_timing_evidence(self):
        decision_id = await self.ready()
        state = self.session.state()
        self.assertEqual(state["calls"], 1)
        self.assertEqual(state["latency_s"], 0.02)
        self.assertEqual(state["total_latency_s"], 0.02)
        self.assertEqual(state["thread_id"], "fixture-subscription-thread")
        self.assertEqual(state["diagnostic_count"], 2)
        self.assertEqual(state["history"][0]["decision"], proposal())
        self.session.execute(decision_id)
        self.sim.succeed()
        final = self.session.state()
        self.assertEqual(final["status"], "succeeded")
        record = final["history"][0]
        self.assertEqual(record["observation"]["sorting"]["status"], "idle")
        self.assertEqual(record["result_observation"]["sorting"]["status"], "succeeded")
        self.assertEqual(record["execution_sim_s"], 20.0)
        self.assertGreaterEqual(record["execution_wall_s"], 0)
        record["decision"]["steps"].clear()
        self.assertEqual(self.session.state()["history"][0]["decision"], proposal())
        with self.assertRaises(ValueError):
            self.session.start("不能悄悄覆盖历史")
        self.request.assert_awaited_once()

    async def test_scene_drift_between_preview_and_approval_never_dispatches(self):
        decision_id = await self.ready()
        self.sim.scene["objects"][0]["position_m"][0] += 0.006
        with self.assertRaises(PlannerError):
            self.session.execute(decision_id)
        self.sim.start_sorting.assert_not_called()
        state = self.session.state()
        self.assertEqual(state["status"], "failed")
        self.assertIsNotNone(state["history"][0]["result_observation"])

    async def test_idle_status_cannot_bypass_live_support_home_empty_hand_or_slots(self):
        for failure in ("supported", "stable", "detached", "home_reached", "hand_empty", "occupied-slot"):
            self.session.reset()
            self.sim = ObservedSortingSimulation()
            self.session._simulation = self.sim
            decision_id = await self.ready()
            if failure in ("supported", "stable", "detached"):
                self.sim.scene["sorting"]["objects"][0][failure] = False
            elif failure == "occupied-slot":
                self.sim.skill.precondition_error.return_value = "Prospective destination is occupied"
            else:
                self.sim.scene["sorting"]["goal_checks"][failure] = False
            with self.subTest(failure=failure), self.assertRaises(PlannerError):
                self.session.execute(decision_id)
            self.sim.start_sorting.assert_not_called()
            self.assertEqual(self.session.state()["status"], "failed")

    async def test_ready_plan_is_checked_again_and_must_match_approved_preview(self):
        decision_id = await self.ready()
        self.session._result = replace(self.answer, decision=SortingDecision("plan", "Changed", "Changed", FIXED_PLAN))
        with self.assertRaises(PlannerError):
            self.session.execute(decision_id)
        self.sim.start_sorting.assert_not_called()
        self.assertEqual(self.session.state()["status"], "failed")

    async def test_unsupported_and_provider_failures_never_dispatch_or_retry(self):
        self.request.return_value = result(dict(proposal(), decision="unsupported", steps=[]))
        self.session.start("不支持的堆叠任务")
        await self.session._pending
        self.assertEqual(self.session.state()["status"], "unsupported")
        with self.assertRaises(ValueError):
            self.session.execute(self.session.state()["decision_id"])
        for failure in (PlannerError("subscription unavailable"), RuntimeError("private-bug-detail")):
            self.session.reset()
            self.request.side_effect = failure
            self.session.start("完整分拣")
            with self.assertLogs("sorting_planning", level="ERROR") if isinstance(failure, RuntimeError) else nullcontext():
                await self.session._pending
            self.assertEqual(self.session.state()["status"], "failed")
            self.assertNotIn("private-bug-detail", self.session.state()["error"])
            with self.assertRaises(ValueError):
                self.session.start("自动重试不允许")
        self.sim.start_sorting.assert_not_called()
        self.sim.step.assert_not_called()

    async def test_missing_configuration_and_bad_initial_physics_do_not_call_provider(self):
        self.loader.return_value = PlannerConfig(None)
        self.assertFalse(self.session.start("分拣"))
        self.assertEqual(self.session.state()["status"], "unconfigured")
        self.session.reset()
        self.sim.skill.precondition_error.return_value = "not home"
        with self.assertRaises(PlannerError):
            self.session.start("分拣")
        self.request.assert_not_awaited()
        self.sim.start_sorting.assert_not_called()

    async def test_cancel_and_reset_discard_even_cancellation_resistant_late_results(self):
        for operation in ("cancel", "reset"):
            self.session.reset()
            entered, release = asyncio.Event(), asyncio.Event()

            async def late(*args):
                entered.set()
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    await release.wait()
                return self.answer

            self.request.side_effect = late
            self.session.start("旧请求")
            pending = self.session._pending
            await entered.wait()
            old_id = self.session.state()["decision_id"]
            getattr(self.session, operation)()
            terminal = self.session.state()
            release.set()
            await pending
            with self.subTest(operation=operation):
                self.assertEqual(self.session.state(), terminal)
                self.assertNotEqual(self.session.state()["decision_id"], old_id)
                with self.assertRaises(ValueError):
                    self.session.execute(old_id)
            self.sim.start_sorting.assert_not_called()

    async def test_scene_drift_during_generation_invalidates_late_callback(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def delayed(*args):
            entered.set()
            await release.wait()
            return self.answer

        self.request.side_effect = delayed
        self.session.start("分拣")
        pending = self.session._pending
        await entered.wait()
        self.sim.scene["scene_revision"] += 1
        release.set()
        await pending
        self.assertEqual(self.session.state()["status"], "failed")
        self.sim.start_sorting.assert_not_called()

    async def test_close_awaits_cancel_cleanup_without_cancelling_it_twice(self):
        entered, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def cleanup(*args):
            entered.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                cleaning.set()
                await release.wait()
                raise

        self.request.side_effect = cleanup
        self.session.start("分拣")
        await entered.wait()
        self.session.cancel()
        await cleaning.wait()
        closing = asyncio.create_task(self.session.close())
        await asyncio.sleep(0)
        self.assertFalse(closing.done())
        release.set()
        await closing
        self.assertEqual(self.session.state()["status"], "cancelled")
        with self.assertRaises(ValueError):
            self.session.start("不能重开已关闭会话")
        self.sim.start_sorting.assert_not_called()

    async def test_physical_failure_and_post_success_goal_loss_revoke_session_success(self):
        for terminal in (False, True):
            self.session.reset()
            self.sim = ObservedSortingSimulation()
            self.session._simulation = self.sim
            self.session.execute(await self.ready())
            if terminal:
                self.sim.succeed()
                self.assertEqual(self.session.state()["status"], "succeeded")
                self.sim.scene["sorting"]["goal_checks"]["hand_empty"] = False
            else:
                self.sim.skill.status = "failed"
                self.sim.scene["sorting"]["failure_reason"] = "Collision fixture"
            state = self.session.state()
            with self.subTest(terminal=terminal):
                self.assertEqual(state["status"], "failed")
                self.assertEqual(state["history"][0]["status"], "failed")
                self.assertIsNotNone(state["history"][0]["result_observation"])
                self.sim.start_sorting.assert_called_once()

    async def test_numerical_failure_never_serializes_invalid_simulation_or_dispatches(self):
        await self.ready()
        self.sim.state.reset_mock()
        self.sim.numerical_error.return_value = "nonfinite qpos"
        state = self.session.state()
        self.assertEqual(state["status"], "failed")
        self.sim.state.assert_not_called()
        self.sim.start_sorting.assert_not_called()

    async def test_malformed_callback_and_dispatch_exception_cannot_leave_executable_preview(self):
        malformed = dict(proposal(), steps=[])
        self.request.return_value = replace(
            self.answer, decision=SimpleNamespace(to_dict=lambda: malformed),
        )
        self.session.start("分拣")
        await self.session._pending
        self.assertEqual(self.session.state()["status"], "failed")
        self.sim.start_sorting.assert_not_called()
        self.session.reset()
        self.request.return_value = self.answer
        decision_id = await self.ready()
        self.sim.start_sorting.side_effect = ValueError("Prospective slot rejected at dispatch")
        with self.assertRaises(ValueError):
            self.session.execute(decision_id)
        self.assertEqual(self.session.state()["status"], "failed")
        with self.assertRaises(ValueError):
            self.session.execute(decision_id)
        self.sim.start_sorting.assert_called_once()

    async def test_external_fixed_dispatch_or_plan_replacement_invalidates_authorization(self):
        decision_id = await self.ready()
        self.sim._start_sorting()
        with self.assertRaises(PlannerError):
            self.session.execute(decision_id)
        self.sim.start_sorting.assert_not_called()
        self.session.reset()
        self.sim = ObservedSortingSimulation()
        self.session._simulation = self.sim
        self.session.execute(await self.ready())
        self.sim.plan = FIXED_PLAN
        self.assertEqual(self.session.state()["status"], "failed")

    async def test_controller_success_label_without_complete_live_goals_is_not_success(self):
        self.session.execute(await self.ready())
        self.sim.skill.status = "succeeded"
        self.assertEqual(self.session.state()["status"], "failed")
        self.assertEqual(self.session.state()["history"][0]["status"], "failed")


class SortingPlanningPhysicsTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_scene_preview_and_approval_leave_physics_untouched_until_caller_ticks(self):
        simulation = SortingSimulation()
        with patch("sorting_planning.PlannerConfig.load", return_value=PlannerConfig("test-only-codex")), \
                patch("sorting_planning.request_sorting_plan", new=AsyncMock(return_value=result())):
            session = SortingPlanningSession(Path(__file__).parent, simulation)
            self.addAsyncCleanup(session.close)
            before = {name: getattr(simulation.data, name).copy() for name in ("qpos", "qvel", "ctrl")}
            before_time = float(simulation.data.time)
            self.assertTrue(session.start("先预览，明确批准后执行"))
            await session._pending
            session.advance()
            self.assertEqual(session.state()["status"], "ready")
            self.assertEqual(simulation.skill.status, "idle")
            self.assertEqual(simulation.plan, FIXED_PLAN)
            for name, values in before.items():
                self.assertTrue((getattr(simulation.data, name) == values).all())
            self.assertEqual(float(simulation.data.time), before_time)
            session.execute(session.state()["decision_id"])
            self.assertEqual(simulation.skill.status, "running")
            self.assertEqual(simulation.plan, validate_plan(proposal()["steps"]))
            self.assertEqual(simulation.skill.current_index, len(OBJECTS) - 1)
            for name, values in before.items():
                self.assertTrue((getattr(simulation.data, name) == values).all())
            self.assertEqual(float(simulation.data.time), before_time)


if __name__ == "__main__":
    unittest.main()
