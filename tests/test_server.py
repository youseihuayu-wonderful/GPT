"""Local aiohttp lifecycle regressions; no model or external network calls."""

import asyncio
import gzip
import json
from pathlib import Path
import tempfile
import threading
import mujoco
import unittest
from unittest.mock import AsyncMock, patch

from aiohttp import WSMsgType
from aiohttp.test_utils import TestClient, TestServer

from grasp import GraspParameters
from planner import DecisionResult, PlannerConfig, PlannerError, SkillDecision
from sorting_contract import FIXED_PLAN, SortStep, SortingPlan
from sorting_planning import SortingDecision, SortingDecisionResult
import server


class ServerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        config = patch("planning.PlannerConfig.load", return_value=PlannerConfig("/fixture/codex"))
        config.start()
        self.addCleanup(config.stop)
        self.decision = SkillDecision(
            "execute", "Fixture grasp", "Explicit test fixture, not model evidence",
            "hold_target", "bowl", "grasp_and_hold", GraspParameters(),
        )
        provider = patch("planning.request_decision", new_callable=AsyncMock)
        self.provider = provider.start()
        self.provider.return_value = DecisionResult(self.decision, "fixture-model", "fixture-thread", 0.25)
        self.addCleanup(provider.stop)
        render = patch("server.Simulation.render", return_value="test-image")
        render.start()
        self.addCleanup(render.stop)
        self.app = server.create_app()
        self.app[server.CONTROLS].paused = True
        self.client = TestClient(TestServer(self.app))
        self.addAsyncCleanup(self.client.close)
        await self.client.start_server()
        self.sim = self.app[server.SIMULATION]
        self.planner = self.app[server.PLANNING]

    async def connect(self):
        ws = await self.client.ws_connect("/ws")
        initial = await asyncio.wait_for(ws.receive_json(), 2)
        self.assertIsNone(initial["state"]["runtime_error"])
        return ws

    async def receive_error(self, ws) -> dict:
        async with asyncio.timeout(2):
            while True:
                message = await ws.receive_json()
                if "error" in message:
                    return message

    async def receive_state(self, ws, predicate) -> dict:
        async with asyncio.timeout(2):
            while True:
                message = await ws.receive_json()
                self.assertNotIn("error", message, message)
                if predicate(message["state"]):
                    return message["state"]

    async def preview(self, ws) -> dict:
        self.sim.step(1.0)
        await ws.send_json({"action": "plan", "task": "抓起黑碗并稳定保持"})
        return await self.receive_state(ws, lambda state: state["planner"]["status"] == "ready")

    async def assert_fatal_close(self, ws) -> None:
        self.assertEqual(await self.receive_error(ws), {
            "error": server.WORKER_FAILURE_MESSAGE, "fatal": True,
        })
        message = await asyncio.wait_for(ws.receive(), 2)
        self.assertEqual(message.type, WSMsgType.CLOSE)
        self.assertEqual(ws.close_code, 1011)

    async def test_sorting_commands_rejected_in_storage_without_mutation(self) -> None:
        ws = await self.connect()
        before = (
            self.sim.data.time, self.sim.data.qpos.tolist(),
            self.sim.data.qvel.tolist(), self.sim.data.ctrl.tolist(),
        )
        for command in (
            {"action": "sort"}, {"action": "sort_plan", "task": "sort"},
            {"action": "sort_execute", "decision_id": 1}, {"action": "sort_cancel"},
        ):
            await ws.send_json(command)
            self.assertIn("仅适用于分拣", (await self.receive_error(ws))["error"])
            self.assertTrue(self.app[server.CONTROLS].paused)
            self.assertEqual(before, (
                self.sim.data.time, self.sim.data.qpos.tolist(),
                self.sim.data.qvel.tolist(), self.sim.data.ctrl.tolist(),
            ))
        self.provider.assert_not_awaited()

    async def test_paused_single_step_numerical_failure_stops_runtime(self) -> None:
        ws = await self.connect()
        self.sim.data.warning[int(mujoco.mjtWarning.mjWARN_BADQPOS)].number = 1
        before = float(self.sim.data.time)
        with self.assertLogs("server", level="ERROR"):
            await ws.send_json({"action": "step"})
            await self.assert_fatal_close(ws)
        self.assertEqual(self.sim.skill.status, "failed")
        self.assertEqual(float(self.sim.data.time), before)
        response = await self.client.get("/api/state")
        self.assertEqual(response.status, 503)
        self.assertEqual((await response.json())["runtime_error"], server.WORKER_FAILURE_MESSAGE)

    async def test_worker_failure_closes_clients_and_latches_without_state_reads(self) -> None:
        ws = await self.connect()
        failure = RuntimeError("original physics failure")
        with (
            patch.object(self.sim, "state", wraps=self.sim.state) as state_read,
            patch.object(self.sim, "render", return_value="test-image") as render,
            patch.object(self.sim, "step") as step,
            self.assertLogs("server", level="ERROR") as logs,
        ):
            def fail_step(_duration):
                state_read.side_effect = AssertionError("corrupt state must not be read")
                render.side_effect = AssertionError("corrupt simulation must not be rendered")
                raise failure

            step.side_effect = fail_step
            await ws.send_json({"action": "play"})
            await self.assert_fatal_close(ws)
            reads_after_failure = state_read.call_count
            renders_after_failure = render.call_count
            self.assertTrue(self.app[server.CONTROLS].paused)
            self.assertEqual(self.app[server.CONTROLS].runtime_error, server.WORKER_FAILURE_MESSAGE)
            self.assertTrue(any(record.exc_info and record.exc_info[1] is failure for record in logs.records))
            response = await self.client.get("/api/state")
            self.assertEqual(response.status, 503)
            self.assertEqual(await response.json(), {
                "paused": True, "runtime_error": server.WORKER_FAILURE_MESSAGE,
            })
            self.assertEqual(response.headers["Cache-Control"], "no-store")
            for _ in range(2):
                reconnect = await self.client.ws_connect("/ws")
                await self.assert_fatal_close(reconnect)
            await server.broadcast(self.app)
            self.assertFalse(self.app[server.CLIENTS])
            self.assertEqual(state_read.call_count, reads_after_failure)
            self.assertEqual(render.call_count, renders_after_failure)
            step.assert_called_once()
            # Cleanup must join the dead worker without rethrowing its logged failure.
            await self.client.close()

    async def test_buffered_command_cannot_resume_after_worker_failure(self) -> None:
        ws = await self.connect()
        latched = asyncio.Event()
        release = asyncio.Event()
        original_close = server.close_failed_websocket
        first_notification = True

        async def delay_worker_notification(app, connection):
            nonlocal first_notification
            if first_notification:
                first_notification = False
                latched.set()
                await release.wait()
                return
            await original_close(app, connection)

        with (
            patch("server.close_failed_websocket", side_effect=delay_worker_notification),
            patch.object(self.sim, "step", side_effect=RuntimeError("worker failure")),
            patch.object(self.sim, "start_skill") as start_skill,
            patch.object(self.planner, "reset") as reset,
            self.assertLogs("server", level="ERROR"),
        ):
            try:
                self.app[server.CONTROLS].paused = False
                await asyncio.wait_for(latched.wait(), 2)
                await ws.send_json({"action": "skill", "skill": "grasp_and_hold"})
                await self.assert_fatal_close(ws)
                self.assertTrue(self.app[server.CONTROLS].paused)
                start_skill.assert_not_called()
                reset.assert_not_called()
            finally:
                release.set()
                await self.client.close()

    async def test_client_notification_failure_does_not_hide_worker_failure(self) -> None:
        bad_client = AsyncMock()
        bad_client.send_json.side_effect = ConnectionError("disconnected")
        good_client = AsyncMock()
        app = server.create_app()
        app[server.SIMULATION] = self.sim
        app[server.PLANNING] = self.planner
        app[server.CLIENTS].update((bad_client, good_client))
        with (
            patch.object(self.sim, "step", side_effect=RuntimeError("primary worker error")),
            self.assertLogs("server", level="WARNING") as logs,
        ):
            await server.physics_loop(app)
        good_client.send_json.assert_awaited_once_with({
            "error": server.WORKER_FAILURE_MESSAGE, "fatal": True,
        })
        bad_client.close.assert_awaited_once()
        good_client.close.assert_awaited_once()
        self.assertFalse(app[server.CLIENTS])
        self.assertTrue(app[server.CONTROLS].paused)
        self.assertIn("primary worker error", "\n".join(logs.output))
        self.assertIn("disconnected", "\n".join(logs.output))

    async def test_first_request_preserves_live_physics_without_actuation(self) -> None:
        self.sim.step(1.0)
        ws = await self.connect()
        requested = asyncio.Event()
        release = asyncio.Event()

        async def delayed_decision(*args, **kwargs):
            requested.set()
            await release.wait()
            return DecisionResult(self.decision, "fixture-model", "fixture-thread", 0.25)

        self.provider.side_effect = delayed_decision
        with patch.object(self.sim, "start_skill", wraps=self.sim.start_skill) as start_skill:
            await ws.send_json({"action": "play"})
            running = await self.receive_state(ws, lambda state: not state["paused"])
            await ws.send_json({"action": "plan", "task": "抓起黑碗并稳定保持"})
            await asyncio.wait_for(requested.wait(), 2)
            live = await self.receive_state(ws, lambda state: (
                state["planner"]["status"] == "planning" and state["time_s"] > running["time_s"]
            ))
            self.assertTrue(live["planner"]["locks_scene"])
            self.assertEqual(live["skill"]["status"], "idle")
            release.set()
            preview = await self.receive_state(ws, lambda state: state["planner"]["status"] == "ready")
            self.assertFalse(preview["paused"])
            self.assertFalse(preview["planner"]["continuous"])
            self.assertEqual(preview["planner"]["latency_s"], 0.25)
            self.assertEqual(preview["planner"]["history"][0]["status"], "ready")
            start_skill.assert_not_called()

    async def test_manual_motion_locked_but_pause_play_and_single_tick_work(self) -> None:
        ws = await self.connect()
        preview = await self.preview(ws)
        decision_id = preview["planner"]["decision_id"]
        for command in (
            {"action": "grasp"}, {"action": "store"},
            {"action": "skill", "skill": "grasp_and_hold"},
        ):
            await ws.send_json(command)
            self.assertIn("锁定", (await self.receive_error(ws))["error"])
        await ws.send_json({"action": "step"})
        stepped = await self.receive_state(ws, lambda state: state["time_s"] > preview["time_s"])
        self.assertTrue(stepped["paused"])
        self.assertEqual(stepped["skill"]["status"], "idle")
        self.assertEqual(stepped["planner"]["decision_id"], decision_id)
        await ws.send_json({"action": "play"})
        await self.receive_state(ws, lambda state: not state["paused"])
        await ws.send_json({"action": "pause"})
        paused = await self.receive_state(ws, lambda state: state["paused"])
        for _ in range(2):
            current = await self.receive_state(ws, lambda state: True)
            self.assertEqual(current["time_s"], paused["time_s"])
            self.assertEqual(current["planner"]["status"], "ready")
        self.provider.assert_awaited_once()

    async def test_authorization_validates_arguments_and_preserves_pause(self) -> None:
        ws = await self.connect()
        preview = await self.preview(ws)
        decision_id = preview["planner"]["decision_id"]
        for command in (
            {"action": "execute_plan", "plan_id": decision_id},
            {"action": "execute_decision", "decision_id": True},
            {"action": "execute_decision", "decision_id": decision_id + 1},
            {"action": "execute_decision", "decision_id": decision_id, "continuous": "true"},
        ):
            await ws.send_json(command)
            self.assertNotIn("fatal", await self.receive_error(ws))
            self.assertEqual(self.sim.skill.status, "idle")
        await ws.send_json({"action": "execute_decision", "decision_id": decision_id, "continuous": True})
        executing = await self.receive_state(ws, lambda state: state["planner"]["status"] == "executing")
        self.assertTrue(executing["paused"])
        self.assertTrue(executing["planner"]["continuous"])
        self.assertEqual(executing["skill"]["skill_name"], "grasp_and_hold")
        self.assertEqual(executing["time_s"], preview["time_s"])
        for _ in range(2):
            current = await self.receive_state(ws, lambda state: True)
            self.assertEqual(current["time_s"], preview["time_s"])
            self.assertEqual(current["skill"]["invocation_id"], executing["skill"]["invocation_id"])
        self.provider.assert_awaited_once()
        await ws.send_json({"action": "plan_next"})
        self.assertNotIn("fatal", await self.receive_error(ws))
        self.provider.assert_awaited_once()
        await ws.send_json({"action": "cancel_plan"})
        stopping = await self.receive_state(ws, lambda state: state["planner"]["status"] == "stopping")
        self.assertTrue(stopping["paused"])
        self.assertTrue(stopping["planner"]["locks_scene"])
        await ws.send_json({"action": "skill", "skill": "return_home"})
        self.assertIn("锁定", (await self.receive_error(ws))["error"])

    async def test_cancelled_preview_unlocks_only_accepted_manual_motion(self) -> None:
        ws = await self.connect()
        await self.preview(ws)
        await ws.send_json({"action": "cancel_plan"})
        cancelled = await self.receive_state(ws, lambda state: state["planner"]["status"] == "cancelled")
        self.assertFalse(cancelled["planner"]["locks_scene"])
        self.assertTrue(cancelled["paused"])
        await ws.send_json({"action": "skill", "skill": "unknown_skill"})
        self.assertNotIn("fatal", await self.receive_error(ws))
        self.assertEqual(self.planner.state()["status"], "cancelled")
        await ws.send_json({"action": "skill", "skill": "grasp_and_hold"})
        manual = await self.receive_state(ws, lambda state: state["skill"]["status"] == "running")
        self.assertFalse(manual["paused"])
        self.assertEqual(manual["planner"]["status"], "idle")
        self.assertEqual(manual["planner"]["history"], [])

    async def test_one_step_feedback_requires_next_request_and_local_completion(self) -> None:
        ws = await self.connect()
        preview = await self.preview(ws)
        await ws.send_json({
            "action": "execute_decision", "decision_id": preview["planner"]["decision_id"],
            "continuous": False,
        })
        executing = await self.receive_state(ws, lambda state: state["planner"]["status"] == "executing")
        for _ in range(1500):
            if self.sim.skill.status != "running":
                break
            self.sim.step(0.02)
        self.assertEqual(self.sim.skill.status, "succeeded")
        feedback = await self.receive_state(ws, lambda state: state["planner"]["status"] == "waiting")
        self.assertTrue(feedback["paused"])
        self.assertTrue(all(feedback["goal_checks"]["hold_target"].values()))
        record = feedback["planner"]["history"][0]
        self.assertEqual(record["invocation_id"], executing["skill"]["invocation_id"])
        self.assertEqual(record["result_observation"]["skill"]["status"], "succeeded")
        self.assertGreater(record["execution_sim_s"], 0)
        self.provider.assert_awaited_once()
        self.provider.return_value = DecisionResult(
            SkillDecision("complete", "Fixture complete", "Measured hold checks",
                          "hold_target", "bowl", None, None),
            "fixture-model", "fixture-completion-thread", 0.5,
        )
        await ws.send_json({"action": "plan_next"})
        complete = await self.receive_state(ws, lambda state: (
            state["planner"]["status"] == "ready"
            and state["planner"]["decision_id"] != preview["planner"]["decision_id"]
        ))
        self.assertEqual(self.provider.await_count, 2)
        self.assertEqual(complete["planner"]["history"][1]["observation"]["skill"]["status"], "succeeded")
        await ws.send_json({
            "action": "execute_decision", "decision_id": complete["planner"]["decision_id"],
            "continuous": False,
        })
        succeeded = await self.receive_state(ws, lambda state: state["planner"]["status"] == "succeeded")
        self.assertEqual(succeeded["skill"]["invocation_id"], executing["skill"]["invocation_id"])
        self.assertEqual(succeeded["planner"]["total_latency_s"], 0.75)
        self.assertEqual(succeeded["planner"]["history"][1]["status"], "complete")

    async def test_reset_invalidates_pending_decision_before_resetting_scene(self) -> None:
        requested = asyncio.Event()

        async def late_decision(*args, **kwargs):
            requested.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return DecisionResult(self.decision, "fixture-model", "late-thread", 0.25)

        self.provider.side_effect = late_decision
        ws = await self.connect()
        revision = self.sim.state()["scene_revision"]
        await ws.send_json({"action": "plan", "task": "抓起黑碗并稳定保持"})
        await asyncio.wait_for(requested.wait(), 2)
        await ws.send_json({"action": "reset"})
        reset = await self.receive_state(ws, lambda state: state["scene_revision"] != revision)
        self.assertEqual(reset["planner"]["status"], "idle")
        self.assertEqual(reset["planner"]["history"], [])
        self.assertTrue(reset["paused"])
        for _ in range(2):
            current = await self.receive_state(ws, lambda state: True)
            self.assertEqual(current["planner"]["status"], "idle")
            self.assertIsNone(current["planner"]["decision"])
            self.assertEqual(current["skill"]["status"], "idle")

    async def test_provider_failure_is_visible_without_fallback_or_retry(self) -> None:
        self.provider.side_effect = PlannerError("fixture provider failure")
        ws = await self.connect()
        await ws.send_json({"action": "plan", "task": "抓起黑碗并稳定保持"})
        failed = await self.receive_state(ws, lambda state: state["planner"]["status"] == "failed")
        self.assertIn("fixture provider failure", failed["planner"]["error"])
        self.assertEqual(failed["planner"]["history"][0]["status"], "failed")
        self.assertEqual(failed["skill"]["status"], "idle")
        self.assertTrue(failed["paused"])
        self.provider.assert_awaited_once()

    async def test_fatal_failure_cancels_pending_provider(self) -> None:
        requested = asyncio.Event()
        cancelled = asyncio.Event()

        async def pending_decision(*args, **kwargs):
            requested.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        self.provider.side_effect = pending_decision
        ws = await self.connect()
        await ws.send_json({"action": "plan", "task": "抓起黑碗并稳定保持"})
        await asyncio.wait_for(requested.wait(), 2)
        with (
            patch.object(self.sim, "step", side_effect=RuntimeError("fatal during model wait")),
            self.assertLogs("server", level="ERROR"),
        ):
            await ws.send_json({"action": "step"})
            await self.assert_fatal_close(ws)
        await asyncio.wait_for(cancelled.wait(), 2)
        self.provider.assert_awaited_once()

    async def test_client_shutdown_timeout_still_releases_renderer_and_other_clients(self):
        class RendererLease:
            released = False

            def close(self):
                self.released = True

        renderer = RendererLease()
        self.sim.renderer = renderer
        failed, remaining = AsyncMock(), AsyncMock()
        failed.close.side_effect = asyncio.TimeoutError()
        self.app[server.CLIENTS].update((failed, remaining))
        with self.assertLogs("server", level="ERROR"):
            await self.client.close()
        self.assertTrue(renderer.released)
        self.assertIsNone(self.sim.renderer)
        remaining.close.assert_awaited_once()

    async def test_report_batches_preserve_distinct_outcomes_and_download_names_without_actuation(self):
        invocation = self.sim.skill.invocation_id
        physical_state = (
            self.sim.data.time, self.sim.data.qpos.tolist(),
            self.sim.data.qvel.tolist(), self.sim.data.ctrl.tolist(),
        )
        with tempfile.TemporaryDirectory() as directory, patch("server.ROOT", Path(directory)):
            results = Path(directory) / "results"
            results.mkdir()
            for batch, filename, outcome in (
                ("initial", "planning_comparison.json", "planning_failed"),
                ("followup", "planning_comparison_followup.json", "succeeded"),
                ("expanded", "planning_comparison_expanded.json", "running"),
                ("drawer_state", "planning_comparison_drawer_state.json", "unsupported"),
            ):
                (results / filename).write_text(json.dumps({
                    "kind": "paired_closed_loop_comparison",
                    "runs": [{"case_id": "held", "method": "gpt", "status": outcome}],
                }), encoding="utf-8")
            for batch, filename, outcome in (
                ("initial", "planning_comparison.json", "planning_failed"),
                ("followup", "planning_comparison_followup.json", "succeeded"),
                ("expanded", "planning_comparison_expanded.json", "running"),
                ("drawer_state", "planning_comparison_drawer_state.json", "unsupported"),
                (None, "planning_comparison.json", "planning_failed"),
            ):
                response = await self.client.get(
                    "/api/planning-comparison", params={} if batch is None else {"batch": batch},
                )
                self.assertEqual(response.status, 200)
                self.assertEqual((await response.json())["runs"][0]["status"], outcome)
                self.assertIn(f'filename="{filename}"', response.headers["Content-Disposition"])
                self.assertEqual(response.headers["Cache-Control"], "no-store")
        self.assertEqual(self.sim.skill.invocation_id, invocation)
        self.assertEqual(physical_state, (
            self.sim.data.time, self.sim.data.qpos.tolist(),
            self.sim.data.qvel.tolist(), self.sim.data.ctrl.tolist(),
        ))
        self.provider.assert_not_awaited()

    async def test_expanded_report_preserves_repeated_multigoal_rows_and_incomplete_status(self):
        cases = (
            "table", "table_right", "table_forward", "held", "placed", "complete",
            "closed_empty", "disturbed", "hold_bowl", "hold_bottle", "close_only", "home_after_close",
        )
        goals = {
            "hold_bowl": "hold_target", "hold_bottle": "hold_target",
            "close_only": "close_and_home", "home_after_close": "home",
        }
        runs = [
            {
                "case_id": case, "repetition": repetition, "method": method,
                "goal": goals.get(case, "store_and_home"),
                "target_object": "bottle" if case == "hold_bottle" else "bowl",
                "status": "setting_up", "success": False,
            }
            for repetition in range(1, 4) for case in cases for method in ("rule", "gpt")
        ]
        report = {
            "kind": "paired_closed_loop_comparison", "status": "running",
            "protocol": {"suite": "expanded", "repetitions": 3, "planned_runs": 72},
            "runs": runs,
        }
        with tempfile.TemporaryDirectory() as directory, patch("server.ROOT", Path(directory)):
            results = Path(directory) / "results"
            results.mkdir()
            (results / "planning_comparison_expanded.json").write_text(json.dumps(report), encoding="utf-8")
            response = await self.client.get("/api/planning-comparison?batch=expanded")
            self.assertEqual(response.status, 200)
            self.assertEqual(await response.json(), report)
            self.assertEqual(len(runs), 72)
            self.assertEqual((await self.client.get("/api/planning-comparison")).status, 404)
        self.provider.assert_not_awaited()

    async def test_missing_or_corrupt_selected_batch_never_falls_back_to_old_results(self):
        with tempfile.TemporaryDirectory() as directory, patch("server.ROOT", Path(directory)):
            results = Path(directory) / "results"
            results.mkdir()
            (results / "planning_comparison.json").write_text(
                '{"kind":"paired_closed_loop_comparison","runs":[]}', encoding="utf-8",
            )
            for batch in ("followup", "expanded", "drawer_state"):
                response = await self.client.get("/api/planning-comparison", params={"batch": batch})
                self.assertEqual(response.status, 404)
                filename = server.COMPARISON_REPORTS[batch]
                for invalid in ("{", "[]", '{"kind":"fixed_scene_parameter_sweep"}'):
                    with self.subTest(batch=batch, invalid=invalid):
                        (results / filename).write_text(invalid, encoding="utf-8")
                        response = await self.client.get("/api/planning-comparison", params={"batch": batch})
                        self.assertEqual(response.status, 500)
            response = await self.client.get("/api/planning-comparison?batch=initial")
            self.assertEqual(response.status, 200)
        self.provider.assert_not_awaited()

    async def test_unknown_paths_and_ambiguous_batch_queries_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory, patch("server.ROOT", Path(directory)):
            (Path(directory) / "private.json").write_text(
                '{"kind":"paired_closed_loop_comparison","private":"not_a_report"}', encoding="utf-8",
            )
            for batch in ("", "../private.json", str(Path(directory) / "private.json"), "parameter_sweep"):
                response = await self.client.get("/api/planning-comparison", params={"batch": batch})
                self.assertEqual(response.status, 400)
                self.assertNotIn("not_a_report", await response.text())
            response = await self.client.get(
                "/api/planning-comparison", params=[("batch", "initial"), ("batch", "expanded")],
            )
            self.assertEqual(response.status, 400)
        self.provider.assert_not_awaited()


class SortingServerTests(unittest.IsolatedAsyncioTestCase):
    connect = ServerTests.connect
    receive_error = ServerTests.receive_error
    receive_state = ServerTests.receive_state
    assert_fatal_close = ServerTests.assert_fatal_close

    async def asyncSetUp(self) -> None:
        forbidden = patch("server.PlanningSession", side_effect=AssertionError("no storage session in sorting"))
        forbidden.start()
        self.addCleanup(forbidden.stop)
        config = patch("sorting_planning.PlannerConfig.load", return_value=PlannerConfig("/fixture/codex"))
        config.start()
        self.addCleanup(config.stop)
        legacy_provider = patch("planning.request_decision", new_callable=AsyncMock)
        self.legacy_provider = legacy_provider.start()
        self.addCleanup(legacy_provider.stop)
        self.addCleanup(self.legacy_provider.assert_not_awaited)
        provider = patch("sorting_planning.request_sorting_plan", new_callable=AsyncMock)
        self.provider = provider.start()
        self.addCleanup(provider.stop)
        # Deliberately different mapping and queue, not model or benchmark evidence.
        destinations = {"A": "C", "B": "A", "C": "B"}
        self.plan = SortingPlan(tuple(
            SortStep(step.object_id, destinations[step.destination])
            for step in reversed(FIXED_PLAN.steps)
        ))
        self.decision = SortingDecision("plan", "Fixture proposal", "Explicit test fixture", self.plan)
        self.result = SortingDecisionResult(self.decision, "fixture-model", "fixture-thread", 0.25)
        self.provider.return_value = self.result
        render = patch("sorting.SortingSimulation.render", return_value="test-image")
        self.render = render.start()
        self.addCleanup(render.stop)
        self.app = server.create_app(scene="sorting")
        self.app[server.CONTROLS].paused = True
        self.client = TestClient(TestServer(self.app))
        self.addAsyncCleanup(self.client.close)
        await self.client.start_server()
        self.sim = self.app[server.SIMULATION]
        self.planner = self.app[server.PLANNING]

    def physical_state(self):
        return (
            self.sim.data.time, self.sim.data.qpos.tolist(),
            self.sim.data.qvel.tolist(), self.sim.data.ctrl.tolist(),
        )

    async def test_frozen_report_refresh_and_download_are_read_only_and_responsive(self) -> None:
        before = self.physical_state()
        planner_before = self.planner.state()
        reading, release = threading.Event(), threading.Event()
        with tempfile.TemporaryDirectory() as directory, patch("server.ROOT", Path(directory)):
            output = Path(directory) / "results" / "sorting_frozen_evaluation"
            output.mkdir(parents=True)
            record = output / "fixture-run.json.gz"
            record.write_bytes(gzip.compress(b'{"fixture":"not model evidence"}', mtime=0))

            def load(path):
                reading.set()
                if not release.wait(5):
                    raise TimeoutError("Blocked report reader was never released")
                return {"status": "running"}

            with patch("evaluate_sorting.load_evaluation_report", side_effect=load):
                pending = asyncio.create_task(self.client.get("/api/sorting-evaluation"))
                try:
                    self.assertTrue(await asyncio.to_thread(reading.wait, 5))
                    self.assertFalse(pending.done())
                    state = await asyncio.wait_for(self.client.get("/api/state"), timeout=2)
                    self.assertEqual(state.status, 200)
                finally:
                    release.set()
                response = await pending
                self.assertEqual(response.status, 200)
                self.assertEqual(response.headers["Cache-Control"], "no-store")
            with patch("evaluate_sorting.evaluation_run_path", return_value=record):
                response = await self.client.get("/api/sorting-evaluation/runs/fixture-run")
                self.assertEqual(response.status, 200)
                self.assertEqual(response.headers["Content-Type"], "application/gzip")
                self.assertEqual(response.headers["Cache-Control"], "no-store")
                await response.read()
        self.assertEqual(self.physical_state(), before)
        self.assertEqual(self.planner.state(), planner_before)
        self.assertTrue(self.app[server.CONTROLS].paused)
        self.provider.assert_not_awaited()

    async def test_frozen_report_missing_invalid_and_unsafe_downloads_never_fall_back(self) -> None:
        before = self.physical_state()
        with tempfile.TemporaryDirectory() as directory, patch("server.ROOT", Path(directory)):
            results = Path(directory) / "results"
            results.mkdir()
            (results / "sorting_comparison.json").write_text('{"old":"not_a_fallback"}')
            response = await self.client.get("/api/sorting-evaluation")
            self.assertEqual(response.status, 404)
            output = results / "sorting_frozen_evaluation"
            output.mkdir()
            (output / "summary.json").write_text("{")
            response = await self.client.get("/api/sorting-evaluation")
            self.assertEqual(response.status, 500)
            self.assertNotIn("not_a_fallback", await response.text())
            self.assertEqual(response.headers["Cache-Control"], "no-store")
            with patch("evaluate_sorting.evaluation_run_path") as locate:
                for run_id in ("..", "%2E%2E%2Fprivate", "private.json", "x%22", "%2Fetc%2Fpasswd"):
                    response = await self.client.get(f"/api/sorting-evaluation/runs/{run_id}")
                    self.assertEqual(response.status, 404)
                locate.assert_not_called()
            for error in (FileNotFoundError("missing"), ValueError("unregistered or unsafe")):
                with patch("evaluate_sorting.evaluation_run_path", side_effect=error):
                    response = await self.client.get("/api/sorting-evaluation/runs/unregistered")
                    self.assertEqual(response.status, 404)
            self.assertEqual((await self.client.post("/api/sorting-evaluation")).status, 405)
            self.assertEqual((await self.client.post("/api/sorting-evaluation/runs/fixture")).status, 405)
        self.assertEqual(self.physical_state(), before)
        self.assertTrue(self.app[server.CONTROLS].paused)
        self.provider.assert_not_awaited()

    async def test_declared_layout_changes_real_positions_and_survives_reset(self) -> None:
        from sorting_evaluation import load_protocol

        protocol = load_protocol()
        layout_id = protocol["layouts"][0]["id"]
        with patch("sorting_planning.PlannerConfig.load",
                   return_value=PlannerConfig("/fixture/codex", protocol["requested_model"])):
            app = server.create_app(scene="sorting", sorting_layout=layout_id)
            app[server.CONTROLS].paused = True
            client = TestClient(TestServer(app))
            self.addAsyncCleanup(client.close)
            await client.start_server()
            body = app[server.SIMULATION].model.body("red_bowl_object").id
            self.assertAlmostEqual(float(app[server.SIMULATION].data.xpos[body, 0]),
                                   protocol["layouts"][0]["poses"]["red_bowl"]["x_m"], delta=0.002)
            response = await client.get("/api/state")
            observed = await response.json()
            self.assertEqual(observed["sorting_layout"], layout_id)
            self.assertNotIn("sorting_layout", observed["planner"])
            self.assertNotIn("sorting_layout", app[server.SIMULATION].state())
            initial = app[server.SIMULATION].data.qpos.tolist()
            ws = await client.ws_connect("/ws")
            await ws.receive_json()
            await ws.send_json({"action": "reset"})
            reset = await self.receive_state(ws, lambda state: state["scene_revision"] != observed["scene_revision"])
            self.assertEqual(reset["sorting_layout"], layout_id)
            self.assertTrue(reset["paused"])
            self.assertEqual(app[server.SIMULATION].data.qpos.tolist(), initial)
            await ws.close()
        self.provider.assert_not_awaited()

    async def test_layout_validation_and_default_sorting_start_without_protocol(self) -> None:
        with patch("sorting_evaluation.load_protocol", side_effect=AssertionError("default must not load")):
            app = server.create_app(scene="sorting")
            app[server.CONTROLS].paused = True
            client = TestClient(TestServer(app))
            self.addAsyncCleanup(client.close)
            await client.start_server()
            response = await client.get("/api/state")
            self.assertEqual(response.status, 200)
            self.assertIsNone((await response.json())["sorting_layout"])
            with self.assertRaises(ValueError):
                server.create_app(sorting_layout="right18")
        for layout in ("unknown", "", "../right18"):
            with self.assertRaises(ValueError):
                server.create_app(scene="sorting", sorting_layout=layout)
        self.provider.assert_not_awaited()

    async def test_sorting_state_uses_own_session_without_legacy_skill(self) -> None:
        observed = self.sim.state()
        observed.pop("skill", None)
        with patch.object(self.sim, "state", return_value=observed):
            response = await self.client.get("/api/state")
            self.assertEqual(response.status, 200)
            state = await response.json()
        self.assertEqual(state["sorting"]["mode"], "fixed")
        self.assertEqual(state["sorting"]["object_count"], 6)
        self.assertEqual(len(state["sorting"]["plan"]), 6)
        self.assertEqual(state["planner"]["status"], "idle")
        self.assertEqual(state["planner"]["mode"], "sorting")
        self.assertTrue(state["planner"]["configured"])
        self.assertEqual(state["planner"]["calls"], 0)
        self.assertEqual(state["planner"]["history"], [])
        self.provider.assert_not_awaited()

    async def test_incompatible_commands_reject_before_mutating_sorting_scene(self) -> None:
        ws = await self.connect()
        before = self.physical_state()
        before_sorting = self.sim.state()["sorting"]
        for command in (
            {"action": "grasp"}, {"action": "store"},
            {"action": "skill", "skill": "return_home"},
            {"action": "plan", "task": "sort objects"},
            {"action": "execute_decision", "decision_id": 1},
            {"action": "plan_next"}, {"action": "cancel_plan"},
        ):
            with self.subTest(command=command):
                await ws.send_json(command)
                self.assertIn("分拣场景不支持", (await self.receive_error(ws))["error"])
                self.assertEqual(self.physical_state(), before)
                self.assertEqual(self.sim.state()["sorting"], before_sorting)
                self.assertTrue(self.app[server.CONTROLS].paused)
        self.provider.assert_not_awaited()

    async def test_sort_starts_unpaused_and_repeated_request_preserves_paused_state(self) -> None:
        self.sim.step(1.0)
        self.assertTrue(self.sim.state()["sorting"]["can_start"])
        ws = await self.connect()
        await ws.send_json({"action": "sort"})
        started = await self.receive_state(ws, lambda state: state["sorting"]["status"] == "running")
        self.assertFalse(started["paused"])
        await ws.send_json({"action": "pause"})
        await self.receive_state(ws, lambda state: state["paused"])
        before = self.physical_state()
        await ws.send_json({"action": "sort"})
        self.assertTrue((await self.receive_error(ws))["error"])
        self.assertTrue(self.app[server.CONTROLS].paused)
        self.assertEqual(self.physical_state(), before)
        await ws.send_json({"action": "reset"})
        reset = await self.receive_state(ws, lambda state: state["sorting"]["status"] == "idle")
        self.assertTrue(reset["paused"])
        self.assertEqual(reset["sorting"]["sorted_count"], 0)
        self.provider.assert_not_awaited()

    async def test_pause_step_camera_and_reset_keep_existing_transport(self) -> None:
        ws = await self.connect()
        before = float(self.sim.data.time)
        await ws.send_json({"action": "step"})
        stepped = await self.receive_state(ws, lambda state: state["time_s"] > before)
        self.assertTrue(stepped["paused"])
        self.assertAlmostEqual(stepped["time_s"] - before, 0.02, places=5)
        await ws.send_json({"action": "camera", "view": "top"})
        async with asyncio.timeout(2):
            while self.app[server.CONTROLS].view != "top":
                await ws.receive_json()
        self.render.assert_called_with("top")
        unchanged = self.physical_state()
        await ws.send_json({"action": "camera", "view": "invalid"})
        self.assertIn("未知视角", (await self.receive_error(ws))["error"])
        self.assertEqual(self.physical_state(), unchanged)
        self.assertEqual(self.app[server.CONTROLS].view, "top")
        await ws.send_json({"action": "play"})
        await self.receive_state(ws, lambda state: not state["paused"])
        await ws.send_json({"action": "pause"})
        paused = await self.receive_state(ws, lambda state: state["paused"])
        await ws.send_json({"action": "reset"})
        reset = await self.receive_state(ws, lambda state: state["scene_revision"] != paused["scene_revision"])
        self.assertTrue(reset["paused"])
        self.provider.assert_not_awaited()

    async def test_sort_rejection_does_not_unpause_or_mutate(self) -> None:
        ws = await self.connect()
        before = self.physical_state()
        with patch.object(self.sim, "start_sorting", side_effect=ValueError("objects not stable")):
            await ws.send_json({"action": "sort"})
            self.assertEqual((await self.receive_error(ws))["error"], "objects not stable")
        self.assertEqual(self.physical_state(), before)
        self.assertTrue(self.app[server.CONTROLS].paused)
        self.provider.assert_not_awaited()

    async def test_sorting_runtime_failure_latches_and_closes_session(self) -> None:
        ws = await self.connect()
        with (
            patch.object(self.sim, "step", side_effect=RuntimeError("sorting numerical failure")),
            self.assertLogs("server", level="ERROR"),
        ):
            await ws.send_json({"action": "step"})
            await self.assert_fatal_close(ws)
        self.assertTrue(self.app[server.CONTROLS].paused)
        self.assertEqual((await self.client.get("/api/state")).status, 503)
        await self.client.close()
        self.provider.assert_not_awaited()

    async def test_dispatch_failure_after_real_acceptance_latches_runtime(self) -> None:
        ws = await self.connect()
        preview = await self.preview(ws)
        start = self.sim.start_sorting

        def partially_accept(plan):
            start(plan)
            raise RuntimeError("Injected failure after physical controller acceptance")

        with patch.object(self.sim, "start_sorting", side_effect=partially_accept), \
                self.assertLogs(level="ERROR"):
            await ws.send_json({"action": "sort_execute", "decision_id": preview["planner"]["decision_id"]})
            failure = await self.receive_error(ws)
        self.assertTrue(failure.get("fatal"), failure)
        self.assertTrue(self.app[server.CONTROLS].paused)
        frozen = self.physical_state()
        self.assertEqual((await self.client.get("/api/state")).status, 503)
        await self.client.close()
        self.assertEqual(self.physical_state(), frozen)

    async def preview(self, ws) -> dict:
        self.sim.step(1.0)
        await ws.send_json({"action": "sort_plan", "task": "Fixture reordered sorting task"})
        return await self.receive_state(ws, lambda state: state["planner"]["status"] == "ready")

    async def test_proposal_is_preview_only_with_real_result_metadata(self) -> None:
        ws = await self.connect()
        self.sim.step(1.0)
        before = self.physical_state()
        await ws.send_json({"action": "sort_plan", "task": "Fixture sorting task"})
        preview = await self.receive_state(ws, lambda state: state["planner"]["status"] == "ready")
        self.assertEqual(self.physical_state(), before)
        self.assertEqual(preview["sorting"]["status"], "idle")
        self.assertEqual(preview["sorting"]["mode"], "fixed")
        self.assertTrue(preview["paused"])
        self.assertTrue(preview["planner"]["locks_scene"])
        self.assertEqual(preview["planner"]["decision"]["steps"], self.plan.to_list())
        self.assertEqual(preview["planner"]["latency_s"], 0.25)
        self.assertEqual(preview["planner"]["thread_id"], "fixture-thread")
        self.assertEqual(preview["planner"]["history"][0]["decision"], self.decision.to_dict())
        self.provider.assert_awaited_once()

    async def test_approval_rejects_bad_ids_and_executes_exact_proposal_once(self) -> None:
        ws = await self.connect()
        preview = await self.preview(ws)
        decision_id = preview["planner"]["decision_id"]
        before = self.physical_state()
        for invalid_id in (None, True, str(decision_id), decision_id + 1):
            await ws.send_json({"action": "sort_execute", "decision_id": invalid_id})
            self.assertNotIn("fatal", await self.receive_error(ws))
            self.assertEqual(self.physical_state(), before)
            self.assertTrue(self.app[server.CONTROLS].paused)
            self.assertEqual(self.sim.skill.status, "idle")
        with patch.object(self.sim, "start_sorting", wraps=self.sim.start_sorting) as start:
            await ws.send_json({"action": "sort_execute", "decision_id": decision_id})
            executing = await self.receive_state(ws, lambda state: state["planner"]["status"] == "executing")
            self.assertFalse(executing["paused"])
            self.assertEqual(executing["sorting"]["mode"], "planned")
            self.assertEqual(self.sim.plan, self.plan)
            self.assertEqual(executing["sorting"]["current_object_id"], self.plan.steps[0].object_id)
            self.assertEqual(executing["sorting"]["current_step_index"], 0)
            self.assertFalse(executing["planner"]["can_cancel"])
            await ws.send_json({"action": "pause"})
            await self.receive_state(ws, lambda state: state["paused"])
            for action in ("sort_execute", "sort_cancel", "sort"):
                await ws.send_json({"action": action, "decision_id": decision_id})
                self.assertNotIn("fatal", await self.receive_error(ws))
                self.assertTrue(self.app[server.CONTROLS].paused)
            start.assert_called_once_with(self.plan)
        self.provider.assert_awaited_once()

    async def test_scene_drift_rejects_approval_without_actuation(self) -> None:
        ws = await self.connect()
        preview = await self.preview(ws)
        # Change the robot's live joint position after the observation.
        self.sim.data.qpos[self.sim.arm_qpos[0]] += 0.1
        mujoco.mj_forward(self.sim.model, self.sim.data)
        before = self.physical_state()
        await ws.send_json({"action": "sort_execute", "decision_id": preview["planner"]["decision_id"]})
        self.assertNotIn("fatal", await self.receive_error(ws))
        self.assertEqual(self.physical_state(), before)
        self.assertEqual(self.sim.skill.status, "idle")
        self.assertTrue(self.app[server.CONTROLS].paused)
        self.assertNotEqual(self.planner.state()["status"], "ready")

    async def test_cancel_keeps_history_and_requires_reset_for_fixed_comparison(self) -> None:
        ws = await self.connect()
        preview = await self.preview(ws)
        before = self.physical_state()
        await ws.send_json({"action": "sort"})
        self.assertIn("锁定", (await self.receive_error(ws))["error"])
        await ws.send_json({"action": "sort_cancel"})
        cancelled = await self.receive_state(ws, lambda state: state["planner"]["status"] == "cancelled")
        self.assertEqual(self.physical_state(), before)
        self.assertEqual(len(cancelled["planner"]["history"]), 1)
        for command in (
            {"action": "sort"},
            {"action": "sort_execute", "decision_id": preview["planner"]["decision_id"]},
        ):
            await ws.send_json(command)
            self.assertNotIn("fatal", await self.receive_error(ws))
            self.assertEqual(self.physical_state(), before)
        await ws.send_json({"action": "reset"})
        reset = await self.receive_state(ws, lambda state: state["scene_revision"] != preview["scene_revision"])
        self.assertEqual(reset["planner"]["history"], [])
        self.assertEqual(reset["sorting"]["mode"], "fixed")
        self.sim.step(1.0)
        await ws.send_json({"action": "sort"})
        fixed = await self.receive_state(ws, lambda state: state["sorting"]["status"] == "running")
        self.assertEqual(fixed["sorting"]["mode"], "fixed")
        self.assertEqual(fixed["planner"]["calls"], 0)
        self.provider.assert_awaited_once()

    async def test_provider_failure_never_falls_back_or_discards_evidence(self) -> None:
        self.provider.side_effect = PlannerError("fixture subscription failure")
        ws = await self.connect()
        self.sim.step(1.0)
        before = self.physical_state()
        await ws.send_json({"action": "sort_plan", "task": "Fixture task"})
        failed = await self.receive_state(ws, lambda state: state["planner"]["status"] == "failed")
        self.assertIn("fixture subscription failure", failed["planner"]["error"])
        self.assertEqual(failed["planner"]["history"][0]["status"], "failed")
        self.assertEqual(self.physical_state(), before)
        await ws.send_json({"action": "sort"})
        self.assertIn("重置", (await self.receive_error(ws))["error"])
        self.assertEqual(self.sim.skill.status, "idle")
        self.provider.assert_awaited_once()

    async def test_unsupported_reply_is_not_executable_or_successful(self) -> None:
        self.provider.return_value = SortingDecisionResult(
            SortingDecision("unsupported", "Fixture unsupported", "Outside bounded task", None),
            "fixture-model", "fixture-thread", 0.3,
        )
        ws = await self.connect()
        self.sim.step(1.0)
        await ws.send_json({"action": "sort_plan", "task": "Fixture unsupported task"})
        unsupported = await self.receive_state(ws, lambda state: state["planner"]["status"] == "unsupported")
        self.assertEqual(unsupported["planner"]["decision"]["steps"], [])
        before = self.physical_state()
        await ws.send_json({"action": "sort_execute", "decision_id": unsupported["planner"]["decision_id"]})
        self.assertNotIn("fatal", await self.receive_error(ws))
        self.assertEqual(self.physical_state(), before)
        self.assertEqual(self.sim.skill.status, "idle")

    async def test_pending_request_reset_discards_late_provider_reply(self) -> None:
        requested = asyncio.Event()

        async def late_result(*args, **kwargs):
            requested.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return self.result

        self.provider.side_effect = late_result
        ws = await self.connect()
        self.sim.step(1.0)
        revision = self.sim.state()["scene_revision"]
        await ws.send_json({"action": "sort_plan", "task": "Fixture pending task"})
        await asyncio.wait_for(requested.wait(), 2)
        await ws.send_json({"action": "sort"})
        self.assertIn("锁定", (await self.receive_error(ws))["error"])
        await ws.send_json({"action": "reset"})
        reset = await self.receive_state(ws, lambda state: state["scene_revision"] != revision)
        self.assertTrue(reset["paused"])
        for _ in range(2):
            current = await self.receive_state(ws, lambda state: True)
            self.assertEqual(current["planner"]["status"], "idle")
            self.assertEqual(current["planner"]["history"], [])
            self.assertIsNone(current["planner"]["decision"])
            self.assertEqual(current["sorting"]["status"], "idle")

    async def test_fixed_execution_rejects_gpt_request_without_provider_call(self) -> None:
        ws = await self.connect()
        self.sim.step(1.0)
        await ws.send_json({"action": "sort"})
        await self.receive_state(ws, lambda state: state["sorting"]["status"] == "running")
        await ws.send_json({"action": "pause"})
        await self.receive_state(ws, lambda state: state["paused"])
        before = self.physical_state()
        await ws.send_json({"action": "sort_plan", "task": "Fixture task"})
        self.assertNotIn("fatal", await self.receive_error(ws))
        self.assertEqual(self.physical_state(), before)
        self.assertEqual(self.sim.plan, FIXED_PLAN)
        self.provider.assert_not_awaited()

    async def test_ready_preview_allows_step_camera_without_authorizing(self) -> None:
        ws = await self.connect()
        preview = await self.preview(ws)
        await ws.send_json({"action": "step"})
        stepped = await self.receive_state(ws, lambda state: state["time_s"] > preview["time_s"])
        self.assertTrue(stepped["paused"])
        self.assertEqual(stepped["planner"]["status"], "ready")
        self.assertEqual(stepped["sorting"]["status"], "idle")
        await ws.send_json({"action": "camera", "view": "front"})
        async with asyncio.timeout(2):
            while self.app[server.CONTROLS].view != "front":
                await ws.receive_json()
        self.render.assert_called_with("front")
        self.assertEqual(self.planner.state()["decision_id"], preview["planner"]["decision_id"])
        self.provider.assert_awaited_once()

    async def test_pending_cancel_discards_late_result_without_unlocking_fixed_path(self) -> None:
        requested = asyncio.Event()

        async def late_result(*args, **kwargs):
            requested.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return self.result

        self.provider.side_effect = late_result
        ws = await self.connect()
        self.sim.step(1.0)
        before = self.physical_state()
        await ws.send_json({"action": "sort_plan", "task": "Fixture pending task"})
        await asyncio.wait_for(requested.wait(), 2)
        await ws.send_json({"action": "sort_cancel"})
        await self.receive_state(ws, lambda state: state["planner"]["status"] == "cancelled")
        for _ in range(2):
            current = await self.receive_state(ws, lambda state: True)
            self.assertEqual(current["planner"]["status"], "cancelled")
            self.assertEqual(current["sorting"]["status"], "idle")
            self.assertEqual(self.physical_state(), before)
        await ws.send_json({"action": "sort"})
        self.assertIn("重置", (await self.receive_error(ws))["error"])
        self.provider.assert_awaited_once()

    async def test_reset_replaces_approval_id_and_rejects_old_client_authorization(self) -> None:
        ws = await self.connect()
        first = await self.preview(ws)
        await ws.send_json({"action": "reset"})
        await self.receive_state(ws, lambda state: state["scene_revision"] != first["scene_revision"])
        second = await self.preview(ws)
        self.assertNotEqual(first["planner"]["decision_id"], second["planner"]["decision_id"])
        before = self.physical_state()
        await ws.send_json({"action": "sort_execute", "decision_id": first["planner"]["decision_id"]})
        self.assertNotIn("fatal", await self.receive_error(ws))
        self.assertEqual(self.physical_state(), before)
        self.assertEqual(self.planner.state()["status"], "ready")
        self.assertTrue(self.app[server.CONTROLS].paused)

    async def test_unknown_scene_rejected_by_factory(self) -> None:
        with self.assertRaises(ValueError):
            server.create_app(scene="unknown")


if __name__ == "__main__":
    unittest.main()
