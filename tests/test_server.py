"""Local aiohttp lifecycle regressions; no model or external network calls."""

import asyncio
import mujoco
import unittest
from unittest.mock import AsyncMock, patch

from aiohttp import WSMsgType
from aiohttp.test_utils import TestClient, TestServer

from planner import PlannerConfig
import server


class ServerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        config = patch("planning.PlannerConfig.load", return_value=PlannerConfig(None))
        config.start()
        self.addCleanup(config.stop)
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

    async def test_manual_skill_lock_and_acceptance_attribution_are_preserved(self) -> None:
        self.sim.step(1.0)
        ws = await self.connect()
        self.planner._status = "ready"
        revision = self.planner._revision
        with patch.object(self.sim, "start_skill", wraps=self.sim.start_skill) as start_skill:
            await ws.send_json({"action": "skill", "skill": "grasp_and_hold"})
            self.assertIn("锁定", (await self.receive_error(ws))["error"])
            start_skill.assert_not_called()
            self.assertEqual(self.planner._revision, revision)
            self.assertEqual(self.planner._status, "ready")
            self.planner._status = "unsupported"
            await ws.send_json({"action": "skill", "skill": "unknown_skill"})
            self.assertNotIn("fatal", await self.receive_error(ws))
            self.assertEqual(self.planner._revision, revision)
            self.assertEqual(self.planner._status, "unsupported")
            await ws.send_json({"action": "skill", "skill": "grasp_and_hold"})
            async with asyncio.timeout(2):
                while self.planner._revision == revision:
                    message = await ws.receive_json()
                    self.assertNotIn("error", message, message)
            self.assertFalse(self.app[server.CONTROLS].paused)
        await ws.close()


if __name__ == "__main__":
    unittest.main()
