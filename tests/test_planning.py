"""Session safety tests; controlled API replies are not live GPT evidence."""

import asyncio
from pathlib import Path
import unittest
from unittest.mock import AsyncMock, patch

from planner import PlannerConfig, PlannerError, PlanResult, validate_plan
from planning import PlanningSession
from simulation import Simulation


def planned_result(label: str = "current") -> PlanResult:
    plan = validate_plan({
        "decision": "plan",
        "summary": f"Test-only {label} grasp",
        "reason": "Deterministic session fixture, not GPT output.",
        "target_object": "bowl",
        "steps": [
            {"phase": phase, "instruction": phase}
            for phase in ("prepare", "approach", "descend", "close", "lift", "hold")
        ],
        "parameters": {
            "initial_opening_m": 0.08,
            "descent_speed_m_s": 0.20,
            "close_trigger_height_m": 0.004,
        },
    })
    return PlanResult(plan, "test-only", f"fixture-{label}", 0.01)


class PlanningSessionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.config = PlannerConfig("test-only-codex")
        config_patch = patch("planning.PlannerConfig.load", return_value=self.config)
        self.loader = config_patch.start()
        self.addCleanup(config_patch.stop)
        self.sim = Simulation()
        self.addCleanup(self.sim.close)
        self.session = PlanningSession(Path(__file__).parent, self.sim)
        self.addAsyncCleanup(self.session.close)
        self.finished = asyncio.Event()
        self.result = planned_result()
        self.request = AsyncMock(side_effect=self.respond)
        request_patch = patch("planning.request_plan", self.request)
        request_patch.start()
        self.addCleanup(request_patch.stop)

    async def respond(self, *args) -> PlanResult:
        self.finished.set()
        return self.result

    def state(self) -> dict:
        return self.session.state(self.sim.skill.state())

    async def ready(self) -> int:
        self.finished.clear()
        self.assertTrue(self.session.start("抓起黑碗并保持", {}))
        await asyncio.wait_for(self.finished.wait(), 1)
        self.assertEqual(self.state()["status"], "ready")
        self.assertEqual(self.sim.skill.status, "idle")
        return self.state()["plan_id"]

    async def test_missing_cli_does_not_call_provider_or_start_physics(self) -> None:
        self.loader.return_value = PlannerConfig(None)
        self.assertFalse(self.session.start("抓起黑碗", {}))
        self.request.assert_not_awaited()
        self.assertEqual(self.state()["status"], "unconfigured")
        self.assertIsNone(self.state()["plan"])
        self.assertEqual(self.sim.skill.status, "idle")
        self.assertEqual(self.sim.data.time, 0)
        with self.assertRaises(ValueError):
            self.session.execute(self.state()["plan_id"])

    async def test_plan_is_preview_only_and_execute_is_single_use(self) -> None:
        plan_id = await self.ready()
        before = self.sim.data.qpos.copy()
        for invalid_id in (None, True, str(plan_id), float(plan_id), plan_id - 1):
            with self.subTest(plan_id=invalid_id), self.assertRaises(ValueError):
                self.session.execute(invalid_id)
        self.assertEqual(self.sim.skill.status, "idle")
        self.assertEqual(self.sim.data.qpos.tolist(), before.tolist())
        self.session.execute(plan_id)
        self.assertEqual(self.sim.skill.status, "running")
        with self.assertRaises(ValueError):
            self.session.execute(plan_id)
        with self.assertRaises(ValueError):
            self.session.cancel()
        with self.assertRaises(ValueError):
            self.session.start("重新抓碗", {})

    async def test_cancel_ready_plan_invalidates_execution(self) -> None:
        plan_id = await self.ready()
        self.session.cancel()
        with self.assertRaises(ValueError):
            self.session.execute(plan_id)
        self.assertFalse(self.session.locks_scene)
        self.assertIsNone(self.state()["plan"])
        self.assertEqual(self.sim.skill.status, "idle")

    async def test_duplicate_requests_and_invalid_tasks_do_not_charge_again(self) -> None:
        started, release = asyncio.Event(), asyncio.Event()

        async def pending(*args):
            started.set()
            await release.wait()
            self.finished.set()
            return self.result

        self.request.side_effect = pending
        for invalid_task in (None, [], " ", "字" * 1001):
            with self.subTest(task=invalid_task), self.assertRaises(ValueError):
                self.session.start(invalid_task, {})
        self.request.assert_not_awaited()
        self.session.start("抓起黑碗", {})
        await asyncio.wait_for(started.wait(), 1)
        with self.assertRaises(ValueError):
            self.session.start("重复抓碗", {})
        self.assertTrue(self.session.locks_scene)
        self.request.assert_awaited_once()
        release.set()
        await asyncio.wait_for(self.finished.wait(), 1)
        with self.assertRaises(ValueError):
            self.session.start("替换尚未执行的计划", {})
        self.request.assert_awaited_once()

    async def test_cancel_and_reset_cannot_be_revived_by_late_response_or_error(self) -> None:
        for operation in ("cancel", "reset"):
            with self.subTest(operation=operation):
                self.session.reset()
                started, release, late_finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
                old_result = planned_result("obsolete")

                async def delayed(*args):
                    started.set()
                    try:
                        await release.wait()
                    except asyncio.CancelledError:
                        # Deliberately misbehaving transport exercises the revision barrier.
                        await release.wait()
                    try:
                        if operation == "reset":
                            raise PlannerError("obsolete request failure")
                        return old_result
                    finally:
                        late_finished.set()

                self.request.side_effect = delayed
                self.session.start("旧任务", {})
                await asyncio.wait_for(started.wait(), 1)
                old_id = self.state()["plan_id"]
                getattr(self.session, operation)()
                self.result = planned_result("replacement")
                self.request.side_effect = self.respond
                new_id = await self.ready()
                release.set()
                await asyncio.wait_for(late_finished.wait(), 1)
                current = self.state()
                self.assertEqual(current["status"], "ready")
                self.assertEqual(current["plan_id"], new_id)
                self.assertEqual(current["thread_id"], "fixture-replacement")
                self.assertIsNone(current["error"])
                with self.assertRaises(ValueError):
                    self.session.execute(old_id)
                self.assertEqual(self.sim.skill.status, "idle")

    async def test_unsupported_task_never_becomes_executable(self) -> None:
        unsupported = validate_plan({
            "decision": "unsupported", "summary": "放入抽屉不受支持",
            "reason": "不能只执行所请求任务的抓碗子集。", "target_object": None,
            "steps": [], "parameters": None,
        })
        self.result = PlanResult(unsupported, "test-only", "fixture-unsupported", 0.01)
        self.session.start("抓碗放入抽屉", {})
        await asyncio.wait_for(self.finished.wait(), 1)
        with self.assertRaises(ValueError):
            self.session.execute(self.state()["plan_id"])
        self.assertEqual(self.state()["status"], "unsupported")
        self.assertFalse(self.session.locks_scene)
        self.assertEqual(self.sim.skill.status, "idle")

    async def test_provider_failure_releases_scene_without_actuating(self) -> None:
        async def fail(*args):
            self.finished.set()
            raise PlannerError("test provider failure")

        self.request.side_effect = fail
        self.session.start("抓起黑碗", {})
        await asyncio.wait_for(self.finished.wait(), 1)
        self.assertEqual(self.state()["status"], "failed")
        self.assertFalse(self.session.locks_scene)
        self.assertIsNone(self.state()["plan"])
        self.assertEqual(self.sim.skill.status, "idle")
        with self.assertRaises(ValueError):
            self.session.execute(self.state()["plan_id"])

    async def test_unexpected_failure_does_not_expose_private_diagnostics(self) -> None:
        private_diagnostic = "private-transport-diagnostic-test"

        async def fail(*args):
            self.finished.set()
            raise RuntimeError(private_diagnostic)

        self.request.side_effect = fail
        with self.assertLogs("planning", level="ERROR") as logs:
            self.session.start("抓起黑碗", {})
            await asyncio.wait_for(self.finished.wait(), 1)
        self.assertNotIn(private_diagnostic, "\n".join(logs.output))
        self.assertNotIn(private_diagnostic, self.state()["error"])
        self.assertEqual(self.state()["status"], "failed")
        self.assertEqual(self.sim.skill.status, "idle")

    async def test_real_physical_success_and_later_loss_override_planner_success(self) -> None:
        self.session.execute(await self.ready())
        while self.sim.skill.status == "running" and self.sim.data.time < 21:
            self.sim.step(0.02)
        self.assertEqual(self.state()["status"], "succeeded")
        self.assertGreaterEqual(self.sim.skill.state()["lift_height_m"], 0.10)
        self.sim.model.actuator_forcerange[self.sim.gripper_actuator] = [0, 0]
        for _ in range(100):
            self.sim.step(0.02)
            if self.state()["status"] == "failed":
                break
        self.assertEqual(self.state()["status"], "failed")
        self.assertIsNotNone(self.state()["error"])


if __name__ == "__main__":
    unittest.main()
