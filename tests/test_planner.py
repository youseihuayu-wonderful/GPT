"""Deterministic contract tests, not evidence of live GPT or physical success."""

import asyncio
from contextlib import suppress
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from grasp import GraspParameters
from planner import (
    PlannerConfig, PlannerError, _run_cli, _subscription_environment,
    request_decision, scene_matches, validate_context, validate_decision,
)
from task_specs import GOALS, SKILLS


def decision_payload(goal="hold_target", target="bowl", skill="grasp_and_hold", decision="execute"):
    return {
        "decision": decision,
        "summary": "Test-only single-skill decision",
        "reason": "Deterministic validation fixture; not a physical success report.",
        "goal": goal if decision != "unsupported" else None,
        "target_object": target if decision != "unsupported" else None,
        "skill": skill if decision == "execute" else None,
        "parameters": GraspParameters().to_dict()
        if decision == "execute" and skill == "grasp_and_hold" else None,
    }


def scene_fixture(target="bowl"):
    return {
        "scene_revision": 1, "time_s": 0.5,
        "target_object": target, "scenario": {"target_object": target, "case_id": "fixture"},
        "skill": {"invocation_id": 0, "status": "idle"},
        "available_skills": [
            {"name": name, "available": name == "grasp_and_hold", "unavailable_reason": "Fixture precondition"}
            for name in SKILLS
        ],
        "goal_checks": {
            name: {check: False for check in spec.checks} for name, spec in GOALS.items()
        },
        "objects": [{"name": "bowl_object", "position_m": [0.3, 0.0, 0.8],
                     "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}],
        "joint_positions_rad": [0.0] * 7, "gripper_opening_m": 0.08,
    }


class DecisionValidationTests(unittest.TestCase):
    def test_parameter_boundaries_are_inclusive_without_clipping(self):
        for endpoint in ("min", "max"):
            payload = decision_payload()
            payload["parameters"] = {name: bounds[endpoint] for name, bounds in GraspParameters.limits().items()}
            with self.subTest(endpoint=endpoint):
                self.assertEqual(validate_decision(payload).parameters.to_dict(), payload["parameters"])

    def test_missing_and_extra_action_fields_cannot_inject_commands(self):
        for section in ("root", "parameters"):
            payload = decision_payload()
            fields = payload if section == "root" else payload["parameters"]
            for key in fields:
                damaged = deepcopy(payload)
                candidate = damaged if section == "root" else damaged["parameters"]
                del candidate[key]
                with self.subTest(section=section, missing=key), self.assertRaises(PlannerError):
                    validate_decision(damaged)
            fields["joint_commands"] = [0, 1, 2]
            with self.subTest(section=section, extra=True), self.assertRaises(PlannerError):
                validate_decision(payload)

    def test_unsafe_parameter_values_are_rejected(self):
        for name, bounds in GraspParameters.limits().items():
            for value in (
                True, False, "0.06", None, [], {}, complex(1, 0),
                float("nan"), float("inf"), float("-inf"), 10**400,
                bounds["min"] - 1e-9, bounds["max"] + 1e-9,
            ):
                payload = decision_payload()
                payload["parameters"][name] = value
                with self.subTest(name=name, value=value), self.assertRaises(PlannerError):
                    validate_decision(payload)

    def test_storage_grasp_requires_clearance_and_supported_target(self):
        payload = decision_payload(goal="store_and_home")
        payload["parameters"]["initial_opening_m"] = 0.059
        with self.assertRaises(PlannerError):
            validate_decision(payload)
        payload["parameters"]["initial_opening_m"] = 0.06
        self.assertEqual(validate_decision(payload).parameters.initial_opening_m, 0.06)
        with self.assertRaises(PlannerError):
            validate_decision(decision_payload(goal="store_and_home", target="bottle"))

    def test_only_grasp_accepts_parameters(self):
        for skill in SKILLS:
            payload = decision_payload(goal="store_and_home", skill=skill)
            if skill == "grasp_and_hold":
                payload["parameters"] = None
            else:
                payload["parameters"] = GraspParameters().to_dict()
            with self.subTest(skill=skill), self.assertRaises(PlannerError):
                validate_decision(payload)

    def test_wrong_goals_skills_targets_and_decisions_are_rejected(self):
        for field, values in {
            "target_object": (None, "drawer", ["bowl"], True),
            "goal": (None, "open_drawer", "", True, [], {}),
            "skill": (None, "teleport", "", True, [], {}),
            "decision": (None, "plan", True, []),
            "parameters": (None, [], "preset"),
        }.items():
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaises(PlannerError):
                    validate_decision(dict(decision_payload(), **{field: value}))
        for value in (None, [], "{}", 3):
            with self.subTest(root=value), self.assertRaises(PlannerError):
                validate_decision(value)

    def test_text_fields_require_nonblank_bounded_strings(self):
        for name, limit in (("summary", 300), ("reason", 600)):
            for value in (None, [], 1, "", " \n", "字" * (limit + 1)):
                with self.subTest(name=name, value=value), self.assertRaises(PlannerError):
                    validate_decision(dict(decision_payload(), **{name: value}))

    def test_complete_and_unsupported_cannot_smuggle_executable_actions(self):
        for kind in ("complete", "unsupported"):
            payload = decision_payload(decision=kind)
            validate_decision(payload)
            for field, value in (("skill", "grasp_and_hold"), ("parameters", GraspParameters().to_dict())):
                with self.subTest(kind=kind, field=field), self.assertRaises(PlannerError):
                    validate_decision(dict(payload, **{field: value}))
        for field, value in (("goal", "hold_target"), ("target_object", "bowl")):
            with self.subTest(field=field), self.assertRaises(PlannerError):
                validate_decision(dict(decision_payload(decision="unsupported"), **{field: value}))


class ContextValidationTests(unittest.TestCase):
    def test_premature_completion_requires_every_measured_predicate(self):
        complete = validate_decision(decision_payload(decision="complete"))
        scene = scene_fixture()
        checks = scene["goal_checks"]["hold_target"]
        for name in checks:
            checks[name] = True
        validate_context(complete, scene, "hold_target")
        for name in checks:
            checks[name] = False
            with self.subTest(predicate=name), self.assertRaises(PlannerError):
                validate_context(complete, scene, "hold_target")
            checks[name] = True
        first = next(iter(checks))
        for invalid in (1, "true", None):
            checks[first] = invalid
            with self.subTest(value=invalid), self.assertRaises(PlannerError):
                validate_context(complete, scene, "hold_target")
        checks[first] = True
        checks["invented_evidence"] = True
        with self.assertRaises(PlannerError):
            validate_context(complete, scene, "hold_target")
        del scene["goal_checks"]["hold_target"]
        with self.assertRaises(PlannerError):
            validate_context(complete, scene, "hold_target")

    def test_available_skill_selected_target_and_bound_goal_are_required(self):
        action = validate_decision(decision_payload())
        validate_context(action, scene_fixture())
        with self.assertRaises(PlannerError):
            validate_context(action, scene_fixture("bottle"))
        with self.assertRaises(PlannerError):
            validate_context(action, scene_fixture(), "store_and_home")
        scene = scene_fixture()
        scene["available_skills"][0]["available"] = False
        with self.assertRaises(PlannerError):
            validate_context(action, scene)
        unsupported = validate_decision(decision_payload(decision="unsupported"))
        validate_context(unsupported, scene_fixture(), "store_and_home")

    def test_failed_or_numerically_warned_scene_cannot_accept_a_decision(self):
        action = validate_decision(decision_payload())
        for field, value in (
            ("warnings", ["mjWARN_BADQPOS"]),
            ("skill", {"status": "failed", "invocation_id": 0}),
        ):
            with self.subTest(field=field), self.assertRaises(PlannerError):
                validate_context(action, dict(scene_fixture(), **{field: value}))

    def test_stale_reset_target_invocation_or_physical_state_is_rejected(self):
        before = scene_fixture()
        changes = (
            ("scene_revision", 2), ("target_object", "bottle"),
            ("scenario", {"case_id": "another"}),
            ("skill", {"invocation_id": 1, "status": "idle"}),
            ("skill", {"invocation_id": 0, "status": "failed"}),
            ("joint_positions_rad", [0.04] + [0.0] * 6),
            ("gripper_opening_m", 0.07),
            ("objects", [{"name": "bowl_object", "position_m": [0.31, 0.0, 0.8],
                          "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}]),
            ("objects", [{"name": "bowl_object", "position_m": [0.3, 0.0, 0.8],
                          "quaternion_wxyz": [0.9987502604, 0.0, 0.0, 0.0499791693]}]),
        )
        for field, value in changes:
            with self.subTest(field=field, value=value):
                self.assertFalse(scene_matches(before, dict(before, **{field: value})))

    def test_clock_and_small_settling_do_not_make_a_decision_stale(self):
        before = scene_fixture()
        after = deepcopy(before)
        after["time_s"] += 60
        after["objects"][0]["position_m"][0] += 0.001
        after["joint_positions_rad"][0] += 0.005
        after["gripper_opening_m"] -= 0.001
        self.assertTrue(scene_matches(before, after))


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
            {"type": "item.completed", "item": {"type": "agent_message", "text": "structured decision"}},
            {"type": "turn.completed"},
        ]
        self.response = json.dumps(decision_payload()).encode()
        self.login = (0, b"", b"Logged in using ChatGPT")
        self.exit_code = 0
        self.stderr = b""
        runner = patch("planner._run_cli", new=AsyncMock(side_effect=self.run_cli))
        runner.start()
        self.addCleanup(runner.stop)

    async def run_cli(self, command, directory, environment, input_data=None, timeout=90):
        if command[1:3] == ["login", "status"]:
            return self.login
        if self.response is not None:
            (directory / "plan.json").write_bytes(self.response)
        return self.exit_code, b"\n".join(json.dumps(event).encode() for event in self.events), self.stderr

    async def request(self, target="bowl", *, goal=None):
        return await request_decision(
            self.config, "抓起选定物体", scene_fixture(target), {}, goal=goal,
        )

    async def test_provider_decision_requires_live_skill_and_original_goal(self):
        for payload, goal in (
            (decision_payload(skill="place_and_release"), None),
            (decision_payload(decision="complete"), None),
            (decision_payload(), "store_and_home"),
        ):
            self.response = json.dumps(payload).encode()
            with self.subTest(payload=payload, goal=goal), self.assertRaises(PlannerError):
                await self.request(goal=goal)

    async def test_valid_decision_for_other_scene_target_is_rejected(self):
        for scene_target, decision_target in (("bowl", "bottle"), ("bottle", "bowl")):
            self.response = json.dumps(decision_payload(target=decision_target)).encode()
            with self.subTest(target=scene_target), self.assertRaises(PlannerError):
                await self.request(scene_target)

    async def test_missing_or_unknown_scene_target_cannot_start_cli(self) -> None:
        for scene in ({}, {"target_object": None}, {"target_object": "drawer"}):
            with patch("planner._run_cli", new_callable=AsyncMock) as runner:
                with self.subTest(scene=scene), self.assertRaises(PlannerError):
                    await request_decision(self.config, "抓起物体", scene, {})
                runner.assert_not_awaited()

    async def test_api_login_or_signed_out_cannot_produce_an_executable_decision(self) -> None:
        for login in ((0, b"", b"Logged in using an API key"), (1, b"", b"Not logged in")):
            self.login = login
            with self.subTest(login=login), self.assertRaises(PlannerError):
                await self.request()

    async def test_decision_file_cannot_bypass_incomplete_failed_or_tool_using_turn(self) -> None:
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

    async def test_completed_turn_cannot_bypass_missing_or_invalid_decision_file(self) -> None:
        unsafe = decision_payload()
        unsafe["parameters"]["initial_opening_m"] = 100
        for response in (
            None, b"", b"```json\n{}\n```", b'{"decision":"execute","decision":"unsupported"}',
            b'{"value":NaN}', b"\xff", json.dumps(unsafe).encode(),
        ):
            self.response = response
            with self.subTest(response=response), self.assertRaises(PlannerError):
                await self.request()

    async def test_failed_cli_never_returns_its_decision_or_private_diagnostic(self) -> None:
        self.exit_code = 1
        for message in (b"401 private-token", b"429 private-token", b"unknown private-token"):
            self.stderr = message
            with self.subTest(message=message), self.assertRaises(PlannerError) as caught:
                await self.request()
            self.assertNotIn("private-token", str(caught.exception))

    async def test_unexpected_errors_are_not_converted_to_synthetic_decisions(self) -> None:
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
