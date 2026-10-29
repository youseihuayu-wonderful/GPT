"""Decision-loop safety fixtures and real local physics; no live GPT evidence."""

import asyncio
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from grasp import GraspParameters
from planner import DecisionResult, PlannerConfig, PlannerError, validate_decision
from planning import PlanningSession
from simulation import Simulation
from task_specs import GOALS, SKILLS


def decided_result(
    label="current", *, decision="execute", goal="hold_target",
    skill="grasp_and_hold", target="bowl",
) -> DecisionResult:
    if decision != "execute":
        skill = None
    if decision == "unsupported":
        goal = target = None
    parsed = validate_decision({
        "decision": decision, "summary": f"Test-only {label}",
        "reason": "Controlled model fixture, not real GPT output.",
        "goal": goal, "target_object": target, "skill": skill,
        "parameters": GraspParameters(0.08, 0.20, 0.004).to_dict()
        if skill == "grasp_and_hold" else None,
    })
    return DecisionResult(parsed, "test-only", f"fixture-{label}", 0.01, 2)


class ObservedSimulation:
    """Explicit state-machine fixture; physical integration uses Simulation below."""

    def __init__(self):
        self.skill = SimpleNamespace(status="idle", invocation_id=0)
        self.skill_name = "grasp_and_hold"
        self.postconditions = {}
        self.failure_reason = None
        self.elapsed_s = 0.0
        self.scene = {
            "time_s": 1.0, "scene_revision": 1, "target_object": "bowl",
            "scenario": {"name": "controlled-session-fixture"}, "warnings": [],
            "objects": [{"name": "bowl", "position_m": [0.0, 0.0, 1.0],
                         "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}],
            "joint_positions_rad": [0.0] * 7, "gripper_opening_m": 0.08,
            "available_skills": [{"name": name, "available": True,
                                  "unavailable_reason": None} for name in SKILLS],
            "goal_checks": {name: dict.fromkeys(spec.checks, False)
                            for name, spec in GOALS.items()},
        }
        self.numerical_error = Mock(return_value=None)
        self.state = Mock(side_effect=self._state)
        self.start_skill = Mock(side_effect=self._start_skill)
        self.start_task = Mock(side_effect=AssertionError("Whole-task dispatch is forbidden"))
        self.step = Mock(side_effect=AssertionError("The caller owns physics ticking"))

    def _state(self):
        return deepcopy(self.scene | {"skill": {
            "status": self.skill.status, "invocation_id": self.skill.invocation_id,
            "skill_name": self.skill_name, "postconditions": self.postconditions,
            "failure_reason": self.failure_reason, "elapsed_s": self.elapsed_s,
        }})

    def _start_skill(self, name, parameters=None):
        self.skill.status = "running"
        self.skill.invocation_id += 1
        self.skill_name = name
        self.elapsed_s = 0.0
        self.postconditions = {}

    def succeed(self):
        self.skill.status = "succeeded"
        self.postconditions = {"measured_fixture_postcondition": True}
        self.elapsed_s = 1.0
        self.scene["time_s"] += self.elapsed_s

    def satisfy_goal(self, goal):
        self.scene["goal_checks"][goal] = dict.fromkeys(GOALS[goal].checks, True)


class PlanningSessionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        config_patch = patch("planning.PlannerConfig.load", return_value=PlannerConfig("test-only-codex"))
        self.loader = config_patch.start()
        self.addCleanup(config_patch.stop)
        self.finished = asyncio.Event()
        self.result = decided_result()
        self.request = AsyncMock(side_effect=self.respond)
        request_patch = patch("planning.request_decision", self.request)
        request_patch.start()
        self.addCleanup(request_patch.stop)
        self.make_session()

    def make_session(self):
        self.sim = ObservedSimulation()
        self.session = PlanningSession(Path(__file__).parent, self.sim)
        self.addAsyncCleanup(self.session.close)

    async def respond(self, *args, **kwargs):
        self.finished.set()
        return self.result

    def state(self):
        return self.session.state()

    async def first(self):
        self.finished.clear()
        self.assertTrue(self.session.start("完成用户指定的完整目标", {}))
        await asyncio.wait_for(self.finished.wait(), 1)
        return self.state()["decision_id"]

    async def ready(self):
        decision_id = await self.first()
        self.assertEqual(self.state()["status"], "ready", self.state())
        return decision_id

    async def following(self, result):
        self.result = result
        self.finished.clear()
        self.session.next()
        await asyncio.wait_for(self.finished.wait(), 1)
        return self.state()["decision_id"]

    async def test_missing_cli_never_requests_or_actuates(self):
        self.loader.return_value = PlannerConfig(None)
        self.assertFalse(self.session.start("抓起黑碗", {}))
        self.request.assert_not_awaited()
        self.sim.start_skill.assert_not_called()
        self.sim.step.assert_not_called()
        self.assertEqual(self.state()["status"], "unconfigured")
        self.assertFalse(self.state()["locks_scene"])

    async def test_first_decision_never_actuates_without_explicit_authorization(self):
        decision_id = await self.ready()
        for _ in range(3):
            self.session.advance()
        self.sim.start_skill.assert_not_called()
        self.sim.step.assert_not_called()
        self.assertIsNone(self.state()["goal"])
        self.assertEqual(self.state()["decision"]["goal"], "hold_target")
        self.assertTrue(self.state()["locks_scene"])
        for invalid_id in (None, True, str(decision_id), float(decision_id), decision_id - 1):
            with self.subTest(decision_id=invalid_id), self.assertRaises(ValueError):
                self.session.execute(invalid_id)
        for continuous in (None, 1, "true"):
            with self.subTest(continuous=continuous), self.assertRaises(ValueError):
                self.session.execute(decision_id, continuous)
        self.session.execute(decision_id)
        self.sim.start_skill.assert_called_once_with("grasp_and_hold", self.result.decision.parameters)
        self.sim.start_task.assert_not_called()
        self.assertEqual(self.state()["goal"], "hold_target")
        self.assertEqual(self.state()["status"], "executing")
        with self.assertRaises(ValueError):
            self.session.execute(decision_id)
        self.assertEqual(self.state()["history"][0]["invocation_id"], 1)

    async def test_invalid_and_duplicate_requests_do_not_call_provider_again(self):
        for task in (None, [], " ", "字" * 1001):
            with self.subTest(task=task), self.assertRaises(ValueError):
                self.session.start(task, {})
        self.request.assert_not_awaited()
        started, release = asyncio.Event(), asyncio.Event()

        async def pending(*args, **kwargs):
            started.set()
            await release.wait()
            return self.result

        self.request.side_effect = pending
        self.session.start("抓起黑碗", {})
        await asyncio.wait_for(started.wait(), 1)
        with self.assertRaises(ValueError):
            self.session.start("重复请求", {})
        with self.assertRaises(ValueError):
            self.session.next()
        self.request.assert_awaited_once()
        release.set()
        await asyncio.sleep(0)
        with self.assertRaises(ValueError):
            self.session.start("替换待授权决策", {})

    async def test_stale_geometry_revision_or_invocation_is_terminal(self):
        mutations = {
            "reset": lambda sim: sim.scene.update(scene_revision=2),
            "object": lambda sim: sim.scene["objects"][0]["position_m"].__setitem__(0, 0.02),
            "orientation": lambda sim: sim.scene["objects"][0].update(quaternion_wxyz=[0, 0, 0, 1]),
            "joints": lambda sim: sim.scene["joint_positions_rad"].__setitem__(0, 0.1),
            "opening": lambda sim: sim.scene.update(gripper_opening_m=0.07),
            "target": lambda sim: sim.scene.update(target_object="bottle"),
            "scenario": lambda sim: sim.scene.update(scenario={"name": "another"}),
            "invocation": lambda sim: setattr(sim.skill, "invocation_id", 99),
        }
        for name, mutate in mutations.items():
            with self.subTest(change=name):
                self.make_session()
                decision_id = await self.ready()
                mutate(self.sim)
                with self.assertRaises(PlannerError):
                    self.session.execute(decision_id)
                self.assertEqual(self.state()["status"], "failed")
                self.sim.start_skill.assert_not_called()
                with self.assertRaises(ValueError):
                    self.session.execute(self.state()["decision_id"])

    async def test_live_clock_and_small_settling_do_not_require_frozen_physics(self):
        decision_id = await self.ready()
        self.sim.scene["time_s"] += 20.0
        self.sim.scene["objects"][0]["position_m"][0] += 0.001
        self.session.execute(decision_id)
        self.assertEqual(self.state()["status"], "executing")
        self.sim.step.assert_not_called()

    async def test_dispatch_rechecks_current_preconditions(self):
        decision_id = await self.ready()
        self.sim.scene["available_skills"][0].update(available=False, unavailable_reason="fixture loss")
        with self.assertRaises(PlannerError):
            self.session.execute(decision_id)
        self.assertEqual(self.state()["status"], "failed")
        self.sim.start_skill.assert_not_called()
        self.assertIsNone(self.state()["goal"])

    async def test_invalid_provider_decisions_never_become_executable(self):
        for changes in ({"skill": "unknown"}, {"target_object": "bottle"}, {"goal": "unknown"}):
            with self.subTest(changes=changes):
                self.make_session()
                valid = decided_result()
                self.result = replace(valid, decision=replace(valid.decision, **changes))
                await self.first()
                self.assertEqual(self.state()["status"], "failed")
                self.sim.start_skill.assert_not_called()

    async def test_premature_complete_is_refused_without_authorizing_a_goal(self):
        self.result = decided_result(decision="complete")
        await self.first()
        self.assertEqual(self.state()["status"], "failed")
        self.assertIsNone(self.state()["goal"])
        self.assertEqual(self.state()["history"][0]["status"], "failed")
        self.sim.start_skill.assert_not_called()

    async def test_local_complete_requires_authorization_and_continued_goal_truth(self):
        self.sim.satisfy_goal("home")
        self.result = decided_result(decision="complete", goal="home")
        decision_id = await self.ready()
        self.session.advance()
        self.assertEqual(self.state()["status"], "ready")
        self.session.execute(decision_id)
        self.assertEqual(self.state()["status"], "succeeded")
        self.assertEqual(self.state()["goal"], "home")
        self.assertEqual(self.state()["history"][0]["status"], "complete")
        self.assertIsNotNone(self.state()["history"][0]["result_observation"])
        self.sim.start_skill.assert_not_called()
        # Empty predicates must never count as vacuous success.
        self.sim.scene["goal_checks"]["home"] = {}
        self.session.advance(allow_execution=False)
        self.assertEqual(self.state()["status"], "failed")
        self.request.assert_awaited_once()

    async def test_complete_is_revalidated_again_at_authorization(self):
        self.sim.satisfy_goal("hold_target")
        self.result = decided_result(decision="complete")
        decision_id = await self.ready()
        predicate = GOALS["hold_target"].checks[0]
        self.sim.scene["goal_checks"]["hold_target"][predicate] = False
        with self.assertRaises(PlannerError):
            self.session.execute(decision_id)
        self.assertEqual(self.state()["status"], "failed")
        self.sim.start_skill.assert_not_called()

    async def test_manual_handoff_uses_actual_outcome_and_state_is_pure(self):
        self.session.execute(await self.ready())
        with self.assertRaises(ValueError):
            self.session.next()
        self.session.advance()
        self.request.assert_awaited_once()
        self.sim.succeed()
        before = self.state()
        self.assertEqual(self.session.state({"status": "failed"}), before)
        self.assertEqual(self.state()["status"], "executing")
        self.session.advance()
        self.assertEqual(self.state()["status"], "waiting")
        self.assertTrue(self.state()["locks_scene"])
        self.request.assert_awaited_once()
        record = self.state()["history"][0]
        self.assertEqual(record["status"], "succeeded")
        self.assertEqual(record["result_observation"]["skill"]["status"], "succeeded")
        self.assertEqual(record["execution_sim_s"], 1.0)
        self.assertGreaterEqual(record["execution_wall_s"], 0)
        self.sim.satisfy_goal("hold_target")
        next_id = await self.following(decided_result("finish", decision="complete"))
        request = self.request.await_args
        self.assertEqual(request.kwargs["goal"], "hold_target")
        self.assertEqual(request.kwargs["history"][0]["status"], "succeeded")
        self.assertEqual(request.args[2]["skill"]["invocation_id"], 1)
        self.session.execute(next_id)
        self.assertEqual(self.state()["status"], "succeeded")
        self.assertAlmostEqual(self.state()["total_latency_s"], 0.02)
        self.assertEqual(self.state()["diagnostic_count"], 2)
        self.assertEqual(self.sim.start_skill.call_count, 1)
        self.sim.step.assert_not_called()

    async def test_returned_and_provider_history_cannot_mutate_session_evidence(self):
        self.session.execute(await self.ready())
        self.sim.succeed()
        self.session.advance()
        returned = self.state()
        returned["history"][0]["result_observation"]["skill"]["status"] = "failed"
        self.sim.satisfy_goal("hold_target")
        await self.following(decided_result(decision="complete"))
        supplied = self.request.await_args.kwargs["history"]
        supplied[0]["observation"]["target_object"] = "bottle"
        supplied[0]["status"] = "failed"
        self.assertEqual(self.state()["history"][0]["status"], "succeeded")
        self.assertEqual(self.state()["history"][0]["observation"]["target_object"], "bowl")
        self.assertEqual(self.state()["history"][0]["result_observation"]["skill"]["status"], "succeeded")
        json.dumps(self.state(), allow_nan=False)

    async def test_continuous_mode_waits_for_success_and_pause_blocks_dispatch(self):
        self.result = decided_result(goal="store_and_home")
        self.session.execute(await self.ready(), continuous=True)
        self.session.advance()
        self.request.assert_awaited_once()
        self.sim.succeed()
        self.session.advance(allow_execution=False)
        self.assertEqual(self.state()["status"], "waiting")
        self.request.assert_awaited_once()
        self.result = decided_result("place", goal="store_and_home", skill="place_and_release")
        self.finished.clear()
        self.session.advance()
        await asyncio.wait_for(self.finished.wait(), 1)
        self.assertEqual(self.state()["status"], "ready")
        self.session.advance(allow_execution=False)
        self.assertEqual(self.sim.start_skill.call_count, 1)
        self.session.advance()
        self.assertEqual(self.sim.start_skill.call_count, 2)
        self.sim.start_skill.assert_called_with("place_and_release", None)
        self.assertEqual(self.state()["goal"], "store_and_home")
        self.sim.step.assert_not_called()

    async def test_established_goal_cannot_be_replaced_by_later_response(self):
        self.result = decided_result(goal="store_and_home")
        self.session.execute(await self.ready())
        self.sim.succeed()
        self.session.advance()
        await self.following(decided_result(goal="home", skill="return_home"))
        self.assertEqual(self.state()["status"], "failed")
        self.assertEqual(self.state()["goal"], "store_and_home")
        self.assertEqual(self.sim.start_skill.call_count, 1)

    async def test_unsupported_does_not_execute_or_lower_approved_goal(self):
        self.result = decided_result(goal="store_and_home")
        self.session.execute(await self.ready())
        self.sim.succeed()
        self.session.advance()
        await self.following(decided_result(decision="unsupported"))
        self.assertEqual(self.state()["status"], "unsupported")
        self.assertEqual(self.state()["goal"], "store_and_home")
        self.assertFalse(self.state()["locks_scene"])
        with self.assertRaises(ValueError):
            self.session.execute(self.state()["decision_id"])
        self.assertEqual(self.sim.start_skill.call_count, 1)

    async def test_cancel_ready_invalidates_authorization(self):
        decision_id = await self.ready()
        self.session.cancel()
        self.assertEqual(self.state()["status"], "cancelled")
        self.assertEqual(self.state()["history"][0]["status"], "cancelled")
        self.assertFalse(self.state()["locks_scene"])
        with self.assertRaises(ValueError):
            self.session.execute(decision_id)
        self.session.advance()
        self.sim.start_skill.assert_not_called()

    async def test_stop_finishes_current_physical_skill_without_claiming_task_success(self):
        self.session.execute(await self.ready(), continuous=True)
        self.session.cancel()
        self.assertEqual(self.state()["status"], "stopping")
        self.assertTrue(self.state()["locks_scene"])
        self.assertEqual(self.sim.skill.status, "running")
        with self.assertRaises(ValueError):
            self.session.start("不能抢占运动", {})
        self.session.advance()
        self.assertEqual(self.state()["status"], "stopping")
        self.sim.succeed()
        self.session.advance(allow_execution=False)
        self.assertEqual(self.state()["status"], "cancelled")
        self.assertEqual(self.state()["history"][0]["status"], "succeeded")
        self.assertFalse(self.state()["locks_scene"])
        self.request.assert_awaited_once()
        self.sim.step.assert_not_called()

    async def test_cancel_waiting_keeps_actual_success_without_requesting_again(self):
        self.session.execute(await self.ready())
        self.sim.succeed()
        self.session.advance()
        self.session.cancel()
        self.assertEqual(self.state()["status"], "cancelled")
        self.assertEqual(self.state()["history"][0]["status"], "succeeded")
        self.session.advance()
        self.request.assert_awaited_once()

    async def test_late_responses_and_errors_cannot_revive_cancelled_or_reset_session(self):
        for operation in ("cancel", "reset"):
            for late_error in (False, True):
                with self.subTest(operation=operation, late_error=late_error):
                    self.make_session()
                    started, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
                    obsolete = decided_result("obsolete")

                    async def delayed(*args, **kwargs):
                        started.set()
                        try:
                            await release.wait()
                        except asyncio.CancelledError:
                            await release.wait()
                        try:
                            if late_error:
                                raise PlannerError("obsolete failure")
                            return obsolete
                        finally:
                            finished.set()

                    self.request.side_effect = delayed
                    self.session.start("旧请求", {})
                    await asyncio.wait_for(started.wait(), 1)
                    old_id = self.state()["decision_id"]
                    getattr(self.session, operation)()
                    self.result = decided_result("replacement")
                    self.request.side_effect = self.respond
                    current_id = await self.ready()
                    release.set()
                    await asyncio.wait_for(finished.wait(), 1)
                    self.assertEqual(self.state()["status"], "ready")
                    self.assertEqual(self.state()["decision_id"], current_id)
                    self.assertEqual(self.state()["thread_id"], "fixture-replacement")
                    self.assertIsNone(self.state()["error"])
                    self.assertEqual(len(self.state()["history"]), 1)
                    with self.assertRaises(ValueError):
                        self.session.execute(old_id)
                    self.sim.start_skill.assert_not_called()

    async def test_physical_failure_invalidates_pending_response_even_while_paused(self):
        self.session.execute(await self.ready())
        self.sim.succeed()
        self.session.advance()
        started, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def delayed(*args, **kwargs):
            started.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                await release.wait()
            finished.set()
            return decided_result("late", skill="return_home")

        self.request.side_effect = delayed
        self.session.next()
        await asyncio.wait_for(started.wait(), 1)
        self.sim.skill.status = "failed"
        self.sim.failure_reason = "fixture contact loss"
        self.session.advance(allow_execution=False)
        self.assertEqual(self.state()["status"], "failed")
        release.set()
        await asyncio.wait_for(finished.wait(), 1)
        self.session.advance()
        self.assertEqual(self.state()["status"], "failed")
        self.assertEqual(self.state()["history"][-1]["error"], "fixture contact loss")
        self.assertEqual(self.sim.start_skill.call_count, 1)
        self.assertEqual(self.request.await_count, 2)

    async def test_failed_skill_or_lost_postconditions_never_requests_next(self):
        for failure in ("failed", "postconditions"):
            with self.subTest(failure=failure):
                self.make_session()
                self.session.execute(await self.ready(), continuous=True)
                if failure == "failed":
                    self.sim.skill.status = "failed"
                    self.sim.failure_reason = "fixture physical failure"
                else:
                    self.sim.succeed()
                    self.sim.postconditions["measured_fixture_postcondition"] = False
                calls = self.request.await_count
                self.session.advance(allow_execution=False)
                self.assertEqual(self.state()["status"], "failed")
                self.assertEqual(self.state()["history"][0]["status"], "failed")
                self.session.advance()
                self.assertEqual(self.request.await_count, calls)
                with self.assertRaises(ValueError):
                    self.session.next()

    async def test_numerical_errors_are_checked_before_serializing_scene(self):
        for operation in ("start", "execute", "advance"):
            with self.subTest(operation=operation):
                self.make_session()
                if operation != "start":
                    decision_id = await self.ready()
                self.sim.state.reset_mock()
                self.sim.numerical_error.return_value = "fixture invalid physics"
                if operation == "start":
                    with self.assertRaises(PlannerError):
                        self.session.start("不能读取无效状态", {})
                elif operation == "execute":
                    with self.assertRaises(PlannerError):
                        self.session.execute(decision_id)
                else:
                    self.session.advance(allow_execution=False)
                self.sim.state.assert_not_called()
                self.assertEqual(self.state()["status"], "failed")
                self.sim.start_skill.assert_not_called()

    async def test_bounded_loop_allows_eight_skills_and_ninth_completion_only(self):
        for finish in ("complete", "execute"):
            with self.subTest(ninth_decision=finish):
                self.make_session()
                self.result = decided_result()
                self.session.execute(await self.ready(), continuous=True)
                for completed in range(1, 9):
                    self.sim.succeed()
                    if completed == 8:
                        self.sim.satisfy_goal("hold_target")
                        self.result = decided_result("ninth", decision=finish)
                    self.finished.clear()
                    self.session.advance()
                    await asyncio.wait_for(self.finished.wait(), 1)
                    self.session.advance()
                expected = "succeeded" if finish == "complete" else "failed"
                self.assertEqual(self.state()["status"], expected)
                self.assertEqual(self.sim.start_skill.call_count, 8)
                self.assertEqual(len(self.state()["history"]), 9)
                calls = self.request.await_count
                self.session.advance()
                self.assertEqual(self.request.await_count, calls)
                self.assertAlmostEqual(self.state()["total_latency_s"], 0.09)

    async def test_provider_errors_are_terminal_and_have_no_fallback(self):
        async def fail(*args, **kwargs):
            self.finished.set()
            raise PlannerError("safe provider failure")

        self.request.side_effect = fail
        await self.first()
        self.session.advance()
        self.assertEqual(self.state()["status"], "failed")
        self.assertEqual(self.state()["history"][0]["error"], "safe provider failure")
        self.assertIsNone(self.state()["decision"])
        self.assertGreaterEqual(self.state()["total_latency_s"], 0)
        self.request.assert_awaited_once()
        self.sim.start_skill.assert_not_called()

    async def test_provider_cancellation_cannot_leave_session_waiting_forever(self):
        async def cancelled(*args, **kwargs):
            self.finished.set()
            raise asyncio.CancelledError

        self.request.side_effect = cancelled
        await self.first()
        self.assertEqual(self.state()["status"], "failed")
        self.assertFalse(self.state()["locks_scene"])
        self.session.advance()
        self.request.assert_awaited_once()
        self.sim.start_skill.assert_not_called()

    async def test_unexpected_provider_diagnostics_stay_private(self):
        private = "private-transport-diagnostic-test"

        async def fail(*args, **kwargs):
            self.finished.set()
            raise RuntimeError(private)

        self.request.side_effect = fail
        with self.assertLogs("planning", level="ERROR") as logs:
            await self.first()
        self.assertEqual(self.state()["status"], "failed")
        self.assertNotIn(private, "\n".join(logs.output))
        self.assertNotIn(private, json.dumps(self.state()))
        self.sim.start_skill.assert_not_called()

    async def test_dispatch_error_cannot_leave_an_automatic_retry_armed(self):
        decision_id = await self.ready()
        self.sim.start_skill.side_effect = ValueError("fixture precondition refusal")
        with self.assertRaises(ValueError):
            self.session.execute(decision_id, continuous=True)
        self.assertEqual(self.state()["status"], "failed")
        self.session.advance()
        self.sim.start_skill.assert_called_once()
        self.assertIsNone(self.state()["goal"])

    async def test_close_during_execution_records_stop_without_reading_invalid_physics(self):
        self.session.execute(await self.ready(), continuous=True)
        self.sim.state.reset_mock()
        self.sim.numerical_error.return_value = "fixture invalid physics"
        self.session.cancel()
        await self.session.close()
        self.sim.state.assert_not_called()
        self.assertEqual(self.state()["status"], "cancelled")
        record = self.state()["history"][0]
        self.assertEqual(record["status"], "cancelled")
        self.assertGreaterEqual(record["execution_wall_s"], 0)
        self.assertIsNone(record["result_observation"])
        self.assertEqual(self.sim.skill.status, "running")

    async def test_close_cancels_children_without_reading_physics_and_retains_terminals(self):
        started, cancelled = asyncio.Event(), asyncio.Event()

        async def pending(*args, **kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        self.request.side_effect = pending
        self.session.start("等待真实提供方", {})
        await asyncio.wait_for(started.wait(), 1)
        self.sim.state.reset_mock()
        await self.session.close()
        self.assertTrue(cancelled.is_set())
        self.sim.state.assert_not_called()
        self.assertEqual(self.state()["status"], "cancelled")
        self.assertEqual(self.state()["history"][0]["status"], "cancelled")
        self.make_session()
        self.request.side_effect = self.respond
        self.sim.satisfy_goal("home")
        self.result = decided_result(decision="complete", goal="home")
        self.session.execute(await self.ready())
        terminal = self.state()
        await self.session.close()
        self.assertEqual(self.state(), terminal)


class PhysicalPlanningSessionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.sim = Simulation()
        self.addCleanup(self.sim.close)
        self.sim.step(1.0)
        config_patch = patch("planning.PlannerConfig.load", return_value=PlannerConfig("test-only-codex"))
        config_patch.start()
        self.addCleanup(config_patch.stop)
        self.request = AsyncMock()
        request_patch = patch("planning.request_decision", self.request)
        request_patch.start()
        self.addCleanup(request_patch.stop)
        self.session = PlanningSession(Path(__file__).parent, self.sim)
        self.addAsyncCleanup(self.session.close)

    async def ready(self, result):
        self.request.return_value = result
        self.session.start("测试完整目标的真实物理交接", {})
        await asyncio.sleep(0)
        self.assertEqual(self.session.state()["status"], "ready", self.session.state())
        return self.session.state()["decision_id"]

    def finish_skill(self, maximum_s):
        until = float(self.sim.data.time) + maximum_s
        while self.sim.skill.status == "running" and self.sim.data.time < until:
            self.sim.step(0.02)
            self.session.advance()
        self.assertEqual(self.sim.skill.status, "succeeded", self.sim.skill.state())

    async def test_real_grasp_requires_observed_success_then_complete_and_detects_later_loss(self):
        decision_id = await self.ready(decided_result())
        before = self.sim.data.qpos.copy()
        self.session.execute(decision_id)
        self.assertEqual(self.sim.data.qpos.tolist(), before.tolist())
        self.assertEqual(self.sim.skill.execution_mode, "skill")
        self.assertIsNone(self.sim.skill.task_kind)
        with self.assertRaises(ValueError):
            self.session.next()
        self.finish_skill(21.0)
        self.assertEqual(self.session.state()["status"], "waiting")
        self.request.assert_awaited_once()
        self.assertGreaterEqual(self.sim.skill.state()["lift_height_m"], 0.10)
        self.assertTrue(all(self.sim.state()["goal_checks"]["hold_target"].values()))
        self.request.return_value = decided_result(decision="complete")
        self.session.next()
        await asyncio.sleep(0)
        self.session.execute(self.session.state()["decision_id"])
        self.assertEqual(self.session.state()["status"], "succeeded")
        history = self.session.state()["history"]
        self.assertEqual([record["status"] for record in history], ["succeeded", "complete"])
        self.assertGreater(history[0]["execution_sim_s"], 0)
        self.assertEqual(history[0]["result_observation"]["skill"]["invocation_id"], 1)
        # Explicit force fault injection checks revocation, not successful execution.
        self.sim.model.actuator_forcerange[self.sim.gripper_actuator] = [0, 0]
        for _ in range(100):
            self.sim.step(0.02)
            self.session.advance()
            if self.session.state()["status"] == "failed":
                break
        self.assertEqual(self.session.state()["status"], "failed")
        self.assertEqual(self.request.await_count, 2)

    async def test_supported_held_start_hands_off_to_place_without_reset_or_new_grasp(self):
        self.sim.start_skill("grasp_and_hold", GraspParameters(0.08, 0.20, 0.004))
        self.finish_skill(21.0)
        held_time = float(self.sim.data.time)
        held_revision = self.sim.state()["scene_revision"]
        held_invocation = self.sim.skill.invocation_id
        decision_id = await self.ready(decided_result(goal="store_and_home", skill="place_and_release"))
        self.assertEqual(float(self.sim.data.time), held_time)
        self.session.execute(decision_id)
        self.assertEqual(self.sim.skill.skill_name, "place_and_release")
        self.assertEqual(self.sim.skill.invocation_id, held_invocation + 1)
        self.finish_skill(31.0)
        self.assertEqual(self.session.state()["status"], "waiting")
        self.assertEqual(self.sim.state()["scene_revision"], held_revision)
        self.assertGreater(float(self.sim.data.time), held_time)
        self.assertTrue(all(self.sim.skill.state()["postconditions"].values()))
        self.assertEqual(self.session.state()["history"][0]["observation"]["skill"]["status"], "succeeded")
        self.assertEqual(self.session.state()["history"][0]["decision"]["skill"], "place_and_release")
        self.request.assert_awaited_once()

    async def test_overall_goal_loss_revokes_success_even_when_last_skill_stays_successful(self):
        decision_id = await self.ready(decided_result(goal="close_and_home", skill="close_drawer"))
        self.session.execute(decision_id)
        self.finish_skill(46.0)
        self.request.return_value = decided_result(goal="close_and_home", skill="return_home")
        self.session.next()
        await asyncio.sleep(0)
        self.session.execute(self.session.state()["decision_id"])
        self.finish_skill(19.0)
        self.request.return_value = decided_result(decision="complete", goal="close_and_home")
        self.session.next()
        await asyncio.sleep(0)
        self.session.execute(self.session.state()["decision_id"])
        self.assertEqual(self.session.state()["status"], "succeeded")
        self.assertTrue(all(self.sim.state()["goal_checks"]["close_and_home"].values()))
        # Pull the real drawer, leaving the table-supported bowl and home arm alone.
        drawer = self.sim.model.body("cabinet_cabinet_bottom").id
        self.sim.data.xfrc_applied[drawer, 1] = -30.0
        try:
            self.sim.step(float(self.sim.model.opt.timestep))
        finally:
            self.sim.data.xfrc_applied[drawer, 1] = 0.0
        self.assertEqual(self.sim.skill.status, "succeeded")
        self.assertTrue(all(self.sim.skill.postconditions().values()))
        self.assertFalse(self.sim.state()["goal_checks"]["close_and_home"]["drawer_closed"])
        self.assertEqual(self.session.state()["status"], "succeeded")
        self.session.advance(allow_execution=False)
        self.assertEqual(self.session.state()["status"], "failed")
        self.assertEqual(self.session.state()["history"][-1]["status"], "failed")
        self.assertEqual(self.request.await_count, 3)


if __name__ == "__main__":
    unittest.main()
