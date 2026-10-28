"""Deterministic contract tests, not evidence of live GPT or physical success."""

import asyncio
from contextlib import suppress
from copy import deepcopy
from dataclasses import FrozenInstanceError
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from grasp import GraspParameters
from planner import (
    PlannerConfig, PlannerError, _run_cli, _subscription_environment, request_plan, validate_plan,
)
from task_specs import TASKS


def plan_payload(task_kind: str = "grasp", target: str = "bowl") -> dict:
    return {
        "decision": "plan",
        "task_kind": task_kind,
        "summary": f"Test-only {task_kind} for {target}",
        "reason": "Deterministic validation fixture; not a physical success report.",
        "target_object": target,
        "steps": [
            {"phase": phase, "instruction": f"执行固定阶段 {phase}"}
            for phase in TASKS[task_kind].phases
        ],
        "parameters": {
            name: 0.08 if name == "initial_opening_m" else (bounds["min"] + bounds["max"]) / 2
            for name, bounds in GraspParameters.limits().items()
        },
    }


def unsupported_payload() -> dict:
    return {
        "decision": "unsupported", "summary": "无法完成整项任务",
        "reason": "没有打开已关闭抽屉的能力，不能只执行抓取部分。",
        "task_kind": None, "target_object": None, "steps": [], "parameters": None,
    }


class PlanValidationTests(unittest.TestCase):
    def test_validated_plan_is_immutable_and_detached_from_input_and_serialization(self) -> None:
        payload = plan_payload()
        expected = deepcopy(payload)
        plan = validate_plan(payload)
        self.assertEqual(plan.to_dict(), expected)
        self.assertIsInstance(plan.steps, tuple)
        for target, name, value in (
            (plan, "target_object", "bottle"),
            (plan, "task_kind", "store_bowl"),
            (plan.steps[0], "phase", "release"),
            (plan.parameters, "initial_opening_m", 100),
        ):
            with self.subTest(name=name), self.assertRaises(FrozenInstanceError):
                setattr(target, name, value)
        payload["steps"][0]["phase"] = "release"
        payload["parameters"]["initial_opening_m"] = 100
        encoded = plan.to_dict()
        encoded["steps"].clear()
        encoded["parameters"].clear()
        self.assertEqual(plan.to_dict(), expected)

    def test_parameter_boundaries_are_inclusive_without_clipping(self) -> None:
        for endpoint in ("min", "max"):
            payload = plan_payload()
            payload["parameters"] = {name: bounds[endpoint] for name, bounds in GraspParameters.limits().items()}
            with self.subTest(endpoint=endpoint):
                self.assertEqual(validate_plan(payload).parameters.to_dict(), payload["parameters"])

    def test_missing_and_extra_fields_cannot_add_commands_or_get_defaults(self) -> None:
        for section in ("root", "step", "parameters"):
            payload = plan_payload()
            target = payload if section == "root" else payload["steps"][0] if section == "step" else payload["parameters"]
            for key in list(target):
                damaged = deepcopy(payload)
                candidate = damaged if section == "root" else damaged["steps"][0] if section == "step" else damaged["parameters"]
                del candidate[key]
                with self.subTest(section=section, missing=key), self.assertRaises(PlannerError):
                    validate_plan(damaged)
            target["joint_commands"] = [0, 1, 2]
            with self.subTest(section=section, extra=True), self.assertRaises(PlannerError):
                validate_plan(payload)

    def test_unsafe_parameter_values_are_rejected(self) -> None:
        for name, bounds in GraspParameters.limits().items():
            for value in (
                True, False, "0.06", None, [], {}, complex(1, 0),
                float("nan"), float("inf"), float("-inf"), 10**400,
                bounds["min"] - 1e-9, bounds["max"] + 1e-9,
            ):
                payload = plan_payload()
                payload["parameters"][name] = value
                with self.subTest(name=name, value=value), self.assertRaises(PlannerError):
                    validate_plan(payload)

    def test_unknown_reordered_skipped_and_repeated_phases_are_rejected(self) -> None:
        original = plan_payload()["steps"]
        candidates = (
            original[::-1], original[:-1], original + [original[-1]],
            [original[0]] + original[:-1],
            [{"phase": "release", "instruction": "放下"}] + original[1:],
        )
        for steps in candidates:
            payload = plan_payload()
            payload["steps"] = steps
            with self.subTest(steps=steps), self.assertRaises(PlannerError):
                validate_plan(payload)


    def test_storage_opening_clearance_is_checked_before_plan_can_execute(self) -> None:
        payload = plan_payload("store_bowl")
        payload["parameters"]["initial_opening_m"] = 0.059
        with self.assertRaises(PlannerError):
            validate_plan(payload)
        payload["parameters"]["initial_opening_m"] = 0.06
        self.assertEqual(validate_plan(payload).parameters.initial_opening_m, 0.06)
        payload = plan_payload("grasp")
        payload["parameters"]["initial_opening_m"] = 0.02
        self.assertEqual(validate_plan(payload).parameters.initial_opening_m, 0.02)

    def test_cross_kind_truncation_mixing_and_storage_reordering_are_rejected(self) -> None:
        storage = plan_payload("store_bowl")
        grasp = plan_payload()
        mixed = deepcopy(storage["steps"])
        mixed[-1] = grasp["steps"][-1]
        reordered = deepcopy(storage["steps"])
        reordered[-2], reordered[-1] = reordered[-1], reordered[-2]
        for kind, steps in (
            ("store_bowl", grasp["steps"]), ("grasp", storage["steps"]),
            ("store_bowl", mixed), ("store_bowl", reordered),
        ):
            with self.subTest(kind=kind, steps=steps), self.assertRaises(PlannerError):
                validate_plan(dict(plan_payload(kind), steps=steps))

    def test_bottle_storage_is_rejected_even_with_complete_storage_steps(self) -> None:
        with self.assertRaises(PlannerError):
            validate_plan(plan_payload("store_bowl", "bottle"))

    def test_wrong_target_decision_and_container_types_are_rejected(self) -> None:
        for field, values in {
            "target_object": (None, "drawer", ["bowl"], True),
            "task_kind": (None, "place", "", True, [], {}),
            "decision": (None, "execute", True, []),
            "steps": (None, {}, "prepare", tuple(plan_payload()["steps"])),
            "parameters": (None, [], "preset"),
        }.items():
            for value in values:
                payload = plan_payload()
                payload[field] = value
                with self.subTest(field=field, value=value), self.assertRaises(PlannerError):
                    validate_plan(payload)
        for value in (None, [], "{}", 3):
            with self.subTest(root=value), self.assertRaises(PlannerError):
                validate_plan(value)

    def test_text_fields_require_nonblank_bounded_strings(self) -> None:
        for name, limit in (("summary", 300), ("reason", 600), ("instruction", 240)):
            for value in (None, [], 1, "", " \n", "字" * (limit + 1)):
                payload = plan_payload()
                target = payload["steps"][0] if name == "instruction" else payload
                target[name] = value
                with self.subTest(name=name, value=value), self.assertRaises(PlannerError):
                    validate_plan(payload)

    def test_unsupported_is_valid_only_without_any_executable_actions(self) -> None:
        payload = unsupported_payload()
        plan = validate_plan(payload)
        self.assertEqual(plan.to_dict(), payload)
        self.assertIsNone(plan.parameters)
        self.assertEqual(plan.steps, ())
        executable = plan_payload()
        for field in ("task_kind", "target_object", "steps", "parameters"):
            damaged = dict(payload, **{field: executable[field]})
            with self.subTest(field=field), self.assertRaises(PlannerError):
                validate_plan(damaged)


class PlannerConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)

    def test_model_override_precedence_does_not_expand_private_environment(self) -> None:
        (self.root / ".env").write_text("CODEX_MODEL=${PRIVATE_VALUE}\n", encoding="utf-8")
        with patch.dict(os.environ, {"PRIVATE_VALUE": "private-secret"}):
            config = PlannerConfig.load(self.root)
            self.assertEqual(config.model, "${PRIVATE_VALUE}")
            self.assertNotIn("private-secret", repr(config))
            self.assertNotIn("CODEX_MODEL", os.environ)
        with patch.dict(os.environ, {"CODEX_MODEL": "environment-model"}):
            self.assertEqual(PlannerConfig.load(self.root).model, "environment-model")

    def test_api_credentials_cannot_replace_missing_cli(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-not-a-real-key"}):
            with patch("planner.shutil.which", return_value=None):
                self.assertFalse(PlannerConfig.load(self.root).configured)

    def test_unreadable_config_has_safe_error_without_raw_exception(self) -> None:
        for error in (PermissionError("private-path-and-secret"), UnicodeError("private-content")):
            with patch.object(Path, "open", side_effect=error):
                with self.assertRaises(PlannerError) as caught:
                    PlannerConfig.load(self.root)
            self.assertNotIn("private-", str(caught.exception))
            self.assertTrue(caught.exception.__suppress_context__)

    def test_invalid_model_cannot_become_an_option_or_multiline_argument(self) -> None:
        for model in ("--dangerously-bypass-approvals-and-sandbox", "x\nmodel", "x\x00", "x\tmodel", "模型", "two models", None):
            with self.subTest(model=model), self.assertRaises(PlannerError):
                PlannerConfig("test-codex", model)




class PlannerRequestTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.config = PlannerConfig("test-only-codex", "test-requested-model")
        self.events = [
            {"type": "thread.started", "thread_id": "test-completed-thread"},
            {"type": "turn.started"},
            {"type": "item.completed", "item": {"type": "agent_message", "text": "structured plan"}},
            {"type": "turn.completed"},
        ]
        self.plan = json.dumps(plan_payload()).encode()
        self.login = (0, b"", b"Logged in using ChatGPT")
        self.exit_code = 0
        self.stderr = b""
        runner = patch("planner._run_cli", new=AsyncMock(side_effect=self.run_cli))
        runner.start()
        self.addCleanup(runner.stop)

    async def run_cli(self, command, directory, environment, input_data=None, timeout=90):
        if command[1:3] == ["login", "status"]:
            return self.login
        self.schema = json.loads((directory / "schema.json").read_text())
        self.prompt = json.loads(input_data)
        if self.plan is not None:
            (directory / "plan.json").write_bytes(self.plan)
        return self.exit_code, b"\n".join(json.dumps(event).encode() for event in self.events), self.stderr

    async def request(self, target="bowl"):
        return await request_plan(self.config, "抓起选定物体", {"target_object": target}, {})

    async def test_schema_and_capabilities_are_scoped_to_selected_target(self) -> None:
        self.plan = json.dumps(plan_payload("grasp", "bottle")).encode()
        result = await self.request("bottle")
        self.assertEqual(result.plan.target_object, "bottle")
        properties = self.schema["properties"]
        self.assertIn("task_kind", self.schema["required"])
        self.assertFalse(self.schema["additionalProperties"])
        self.assertEqual(properties["task_kind"]["anyOf"][0]["enum"], ["grasp"])
        self.assertEqual(properties["target_object"]["anyOf"][0]["enum"], ["bottle"])
        self.assertNotIn("push_drawer", properties["steps"]["items"]["properties"]["phase"]["enum"])
        self.assertEqual(set(self.prompt["task_specs"]), {"grasp"})
        self.plan = json.dumps(plan_payload("store_bowl")).encode()
        result = await self.request()
        self.assertEqual(result.plan.task_kind, "store_bowl")
        self.assertEqual(set(self.prompt["task_specs"]), {"grasp", "store_bowl"})
        self.assertIn("verify_store", self.schema["properties"]["steps"]["items"]["properties"]["phase"]["enum"])

    async def test_valid_plan_for_other_scene_target_is_rejected(self) -> None:
        for scene_target, plan_target in (("bowl", "bottle"), ("bottle", "bowl")):
            self.plan = json.dumps(plan_payload("grasp", plan_target)).encode()
            with self.subTest(target=scene_target), self.assertRaises(PlannerError):
                await self.request(scene_target)

    async def test_missing_or_unknown_scene_target_cannot_start_cli(self) -> None:
        for scene in ({}, {"target_object": None}, {"target_object": "drawer"}):
            with patch("planner._run_cli", new_callable=AsyncMock) as runner:
                with self.subTest(scene=scene), self.assertRaises(PlannerError):
                    await request_plan(self.config, "抓起物体", scene, {})
                runner.assert_not_awaited()

    async def test_api_login_or_signed_out_cannot_produce_an_executable_plan(self) -> None:
        for login in ((0, b"", b"Logged in using an API key"), (1, b"", b"Not logged in")):
            self.login = login
            with self.subTest(login=login), self.assertRaises(PlannerError):
                await self.request()

    async def test_plan_file_cannot_bypass_incomplete_failed_or_tool_using_turn(self) -> None:
        original = deepcopy(self.events)
        candidates = [
            [], original[:-1], original[1:], original + [original[-1]],
            [original[0], original[0], *original[1:]],
            [original[0], {"type": "turn.failed", "error": {"message": "failure"}}],
            [original[0], {"type": "error", "message": "failure"}],
            [original[0], {"type": "item.completed", "item": {"type": "agent_message"}}, *original[1:]],
        ]
        for tool in ("command_execution", "file_change", "mcp_tool_call", "web_search", "collab_tool_call"):
            candidates.append([*original[:2], {"type": "item.completed", "item": {"type": tool}}, original[-1]])
        for events in candidates:
            self.events = events
            with self.subTest(events=events), self.assertRaises(PlannerError):
                await self.request()

    async def test_nonfatal_diagnostics_remain_visible_without_exposing_contents(self) -> None:
        private = "private-startup-diagnostic"
        diagnostic = {"type": "item.completed", "item": {"type": "error", "message": private}}
        self.events.insert(1, diagnostic)
        self.events.insert(-1, diagnostic)
        result = await self.request()
        self.assertEqual(result.diagnostic_count, 2)
        self.assertNotIn(private, repr(result))
        self.events[-1] = {"type": "turn.failed", "error": {"message": private}}
        with self.assertRaises(PlannerError) as caught:
            await self.request()
        self.assertNotIn(private, str(caught.exception))

    async def test_completed_turn_cannot_bypass_missing_or_invalid_plan_file(self) -> None:
        unsafe = plan_payload()
        unsafe["parameters"]["initial_opening_m"] = 100
        for plan in (
            None, b"", b"```json\n{}\n```", b'{"decision":"plan","decision":"unsupported"}',
            b'{"value":NaN}', b"\xff", json.dumps(unsafe).encode(),
        ):
            self.plan = plan
            with self.subTest(plan=plan), self.assertRaises(PlannerError):
                await self.request()

    async def test_failed_cli_never_returns_its_plan_or_private_diagnostic(self) -> None:
        self.exit_code = 1
        for message in (b"401 private-token", b"429 private-token", b"unknown private-token"):
            self.stderr = message
            with self.subTest(message=message), self.assertRaises(PlannerError) as caught:
                await self.request()
            self.assertNotIn("private-token", str(caught.exception))

    async def test_unexpected_errors_are_not_converted_to_synthetic_plans(self) -> None:
        with patch("planner._run_cli", side_effect=RuntimeError("test-programming-error")):
            with self.assertRaises(RuntimeError):
                await self.request()


class CliProcessTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_child_cannot_receive_api_billing_credentials(self) -> None:
        keys = ("OPENAI_API_KEY", "CODEX_API_KEY", "CODEX_ACCESS_TOKEN", "OPENAI_BASE_URL")
        injected = {key: "private-test-value" for key in keys}
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, injected):
            code, stdout, stderr = await _run_cli(
                [sys.executable, "-c", f"import os,json; print(json.dumps([k for k in {keys!r} if k in os.environ]))"],
                Path(directory), _subscription_environment(),
            )
        self.assertEqual((code, json.loads(stdout), stderr), (0, [], b""))

    async def test_cancel_and_timeout_stop_real_process_group_including_descendants(self) -> None:
        for cancel in (True, False):
            with self.subTest(cancel=cancel):
                connected, disconnected = set(), set()
                ready, stopped = asyncio.Event(), asyncio.Event()

                async def connection(reader, writer):
                    role = (await reader.readline()).decode().strip()
                    connected.add(role)
                    if connected == {"parent", "child"}:
                        ready.set()
                    await reader.read()
                    disconnected.add(role)
                    if disconnected == {"parent", "child"}:
                        stopped.set()
                    writer.close()
                    await writer.wait_closed()

                server = await asyncio.start_server(connection, "127.0.0.1", 0)
                port = server.sockets[0].getsockname()[1]
                child = (
                    "import signal,socket\n"
                    + ("signal.signal(signal.SIGTERM, signal.SIG_IGN)\n" if cancel else "")
                    + f"s=socket.create_connection(('127.0.0.1',{port}))\n"
                    + "s.sendall(b'child\\n')\nsignal.pause()\n"
                )
                parent = (
                    "import signal,socket,subprocess,sys\n"
                    + ("signal.signal(signal.SIGTERM, signal.SIG_IGN)\n" if cancel else "")
                    + f"s=socket.create_connection(('127.0.0.1',{port}))\ns.sendall(b'parent\\n')\n"
                    + f"subprocess.Popen([sys.executable,'-c',{child!r}])\nsignal.pause()\n"
                )
                async with server:
                    with tempfile.TemporaryDirectory() as directory:
                        task = asyncio.create_task(_run_cli(
                            [sys.executable, "-c", parent], Path(directory), os.environ.copy(),
                            timeout=20 if cancel else 5,
                        ))
                        try:
                            await asyncio.wait_for(ready.wait(), 3)
                            if cancel:
                                task.cancel()
                            with self.assertRaises(asyncio.CancelledError if cancel else TimeoutError):
                                await task
                            await asyncio.wait_for(stopped.wait(), 2)
                            self.assertEqual(disconnected, {"parent", "child"})
                        finally:
                            if not task.done():
                                task.cancel()
                            with suppress(asyncio.CancelledError, TimeoutError):
                                await task




if __name__ == "__main__":
    unittest.main()
