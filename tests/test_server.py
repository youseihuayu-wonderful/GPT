"""Local aiohttp lifecycle regressions; no model or external network calls."""

import asyncio
import mujoco
import unittest
from unittest.mock import AsyncMock, patch

from aiohttp import WSMsgType
from aiohttp.test_utils import TestClient, TestServer

from grasp import GraspParameters
from planner import DecisionResult, PlannerConfig, PlannerError, SkillDecision
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


if __name__ == "__main__":
    unittest.main()
