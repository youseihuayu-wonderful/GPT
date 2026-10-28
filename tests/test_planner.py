"""Deterministic contract tests, not evidence of live GPT or physical success."""

import asyncio
from copy import deepcopy
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp

from grasp import GraspParameters
from planner import PlannerConfig, PlannerError, request_plan, validate_plan


def plan_payload() -> dict:
    return {
        "decision": "plan",
        "summary": "夹取黑碗碗沿并保持",
        "reason": "仅使用已有碗沿抓取能力，实际结果由仿真判定。",
        "target_object": "bowl",
        "steps": [
            {"phase": phase, "instruction": instruction}
            for phase, instruction in (
                ("prepare", "设置开度并等待稳定"),
                ("approach", "移动至碗沿上方"),
                ("descend", "下降并按高度触发闭合"),
                ("close", "闭合并检查接触"),
                ("lift", "抬起黑碗"),
                ("hold", "检查稳定保持"),
            )
        ],
        "parameters": {
            name: (bounds["min"] + bounds["max"]) / 2
            for name, bounds in GraspParameters.limits().items()
        },
    }


def unsupported_payload() -> dict:
    return {
        "decision": "unsupported", "summary": "无法完成整项任务",
        "reason": "没有放置能力，不能只执行抓取部分。",
        "target_object": None, "steps": [], "parameters": None,
    }


def response_payload(plan: dict | None = None) -> dict:
    return {
        "id": "resp_test_only", "model": "test-provider-snapshot",
        "status": "completed", "error": None, "incomplete_details": None,
        "output": [{
            "type": "message", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": json.dumps(plan if plan is not None else plan_payload())}],
        }],
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

    def test_wrong_target_decision_and_container_types_are_rejected(self) -> None:
        for field, values in {
            "target_object": (None, "bottle", "drawer", ["bowl"], True),
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
        for field in ("target_object", "steps", "parameters"):
            damaged = dict(payload, **{field: executable[field]})
            with self.subTest(field=field), self.assertRaises(PlannerError):
                validate_plan(damaged)


class PlannerConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.environment = patch.dict("os.environ", {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_environment_wins_without_interpolation_or_mutating_environment(self) -> None:
        (self.root / ".env").write_text(
            "OPENAI_API_KEY='file-${UNRELATED_SECRET}'\nOPENAI_MODEL=file-model\n", encoding="utf-8",
        )
        with patch.dict("os.environ", {"UNRELATED_SECRET": "private-other-value"}):
            config = PlannerConfig.load(self.root)
            self.assertEqual(config.api_key, "file-${UNRELATED_SECRET}")
            self.assertNotIn("private-other-value", repr(config))
        with patch.dict("os.environ", {"OPENAI_API_KEY": "environment-key", "OPENAI_MODEL": "environment-model"}):
            config = PlannerConfig.load(self.root)
            self.assertEqual(config.api_key, "environment-key")
            self.assertEqual(config.model, "environment-model")
            self.assertNotIn("environment-key", repr(config))
        with patch.dict("os.environ", {"OPENAI_API_KEY": ""}):
            self.assertFalse(PlannerConfig.load(self.root).configured)

    def test_missing_file_is_unconfigured_and_reloading_reads_new_key(self) -> None:
        self.assertFalse(PlannerConfig.load(self.root).configured)
        (self.root / ".env").write_text("OPENAI_API_KEY=test-local-only\n", encoding="utf-8")
        self.assertTrue(PlannerConfig.load(self.root).configured)
        (self.root / ".env").write_text("OPENAI_API_KEY=''\n", encoding="utf-8")
        self.assertFalse(PlannerConfig.load(self.root).configured)

    def test_unreadable_config_has_safe_error_without_raw_exception(self) -> None:
        for error in (PermissionError("private-path-and-secret"), UnicodeError("private-content")):
            with patch.object(Path, "open", side_effect=error):
                with self.assertRaises(PlannerError) as caught:
                    PlannerConfig.load(self.root)
            self.assertNotIn("private-", str(caught.exception))
            self.assertTrue(caught.exception.__suppress_context__)

    def test_key_header_injection_is_rejected_before_network(self) -> None:
        for key in ("secret\r\nX-Injected: true", "secret\x00", "secret\t", "密钥", "secret key"):
            with self.subTest(key=key), self.assertRaises(PlannerError):
                PlannerConfig(key)


class PlannerRequestTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.config = PlannerConfig("test-key-never-real", "test-requested-model")
        self.scene = {"object": {"name": "bowl", "position": [0.1, 0.2, 0.3]}}
        self.experiments = {"runs": [{"success": False, "elapsed_s": 8.5}]}
        self.response = MagicMock()
        self.response.status = 200
        self.response.read = AsyncMock(return_value=json.dumps(response_payload()).encode())
        self.response.__aenter__ = AsyncMock(return_value=self.response)
        self.response.__aexit__ = AsyncMock(return_value=False)
        self.session = MagicMock()
        self.session.post.return_value = self.response
        self.session.__aenter__ = AsyncMock(return_value=self.session)
        self.session.__aexit__ = AsyncMock(return_value=False)
        self.factory_patch = patch("planner.aiohttp.ClientSession", return_value=self.session)
        self.factory = self.factory_patch.start()
        self.addCleanup(self.factory_patch.stop)

    async def request(self):
        return await request_plan(self.config, "抓起黑碗并保持", self.scene, self.experiments)

    def set_response(self, payload: object) -> None:
        self.response.read.return_value = json.dumps(payload).encode()

    async def test_missing_key_never_opens_a_network_session(self) -> None:
        with self.assertRaises(PlannerError):
            await request_plan(PlannerConfig(""), "抓碗", self.scene, self.experiments)
        self.factory.assert_not_called()

    async def test_unsupported_response_is_not_replaced_with_an_executable_plan(self) -> None:
        self.set_response(response_payload(unsupported_payload()))
        result = await self.request()
        self.assertEqual(result.plan.to_dict(), unsupported_payload())
        self.assertIsNone(result.plan.parameters)

    async def test_reasoning_metadata_is_ignored_but_not_treated_as_a_plan(self) -> None:
        payload = response_payload()
        reasoning = {"type": "reasoning", "summary": [{"text": "not a plan"}]}
        payload["output"].insert(0, reasoning)
        self.set_response(payload)
        self.assertEqual((await self.request()).plan.target_object, "bowl")
        payload["output"] = [reasoning]
        self.set_response(payload)
        with self.assertRaises(PlannerError):
            await self.request()

    async def test_refusal_and_incomplete_have_distinct_safe_failures(self) -> None:
        refusal = response_payload()
        refusal["output"][0]["content"] = [{"type": "refusal", "refusal": self.config.api_key}]
        incomplete = response_payload()
        incomplete["status"] = "incomplete"
        incomplete["incomplete_details"] = {"reason": "max_output_tokens"}
        messages = []
        for payload in (refusal, incomplete):
            self.set_response(payload)
            with self.assertRaises(PlannerError) as caught:
                await self.request()
            messages.append(str(caught.exception))
            self.assertNotIn(self.config.api_key, messages[-1])
        self.assertNotEqual(*messages)

    async def test_empty_ambiguous_unknown_and_unfinished_output_is_rejected(self) -> None:
        original = response_payload()["output"][0]
        candidates = (
            [], None, [None], [original, original],
            [dict(original, role="user")], [dict(original, status="in_progress")],
            [dict(original, content=[])], [dict(original, content=[None])],
            [dict(original, content=original["content"] * 2)],
            [dict(original, content=[{"type": "output_text", "text": " "}])],
            [dict(original, content=[{"type": "output_text", "text": {}}])],
            [dict(original, content=[{"type": "tool_call", "text": "{}"}])],
            [original, {"type": "function_call", "name": "move_robot"}],
        )
        for output in candidates:
            self.set_response(dict(response_payload(), output=output))
            with self.subTest(output=output), self.assertRaises(PlannerError):
                await self.request()

    async def test_invalid_envelope_and_missing_metadata_are_rejected(self) -> None:
        candidates = [None, [], "response"]
        for field, value in (
            ("status", "failed"), ("status", None), ("model", ""), ("id", None),
            ("model", 1), ("error", {"message": self.config.api_key}),
        ):
            candidates.append(dict(response_payload(), **{field: value}))
        for payload in candidates:
            self.set_response(payload)
            with self.subTest(payload=payload), self.assertRaises(PlannerError) as caught:
                await self.request()
            self.assertNotIn(self.config.api_key, str(caught.exception))

    async def test_malformed_duplicate_and_nonfinite_json_cannot_pass_validation(self) -> None:
        valid = json.dumps(plan_payload())
        texts = (
            valid[:-1], "```json\n" + valid + "\n```", "[]", "null",
            valid[:-1] + ', "decision": "unsupported"}',
            valid.replace('"phase": "prepare"', '"phase": "prepare", "phase": "release"'),
            valid.replace('"initial_opening_m": 0.05', '"initial_opening_m": NaN'),
            valid.replace('"initial_opening_m": 0.05', '"initial_opening_m": Infinity'),
            valid.replace('"initial_opening_m": 0.05', '"initial_opening_m": 1e999'),
        )
        for text in texts:
            payload = response_payload()
            payload["output"][0]["content"][0]["text"] = text
            self.set_response(payload)
            with self.subTest(text=text), self.assertRaises(PlannerError):
                await self.request()
        for raw in (b"not-json", b"\xff", b'{"status":"completed","status":"incomplete"}'):
            self.response.read.return_value = raw
            with self.subTest(raw=raw), self.assertRaises(PlannerError):
                await self.request()

    async def test_http_errors_are_secret_safe_and_not_retried_or_decoded(self) -> None:
        messages = {}
        for status in (301, 400, 401, 403, 404, 422, 429, 500, 503):
            self.response.status = status
            self.response.read.return_value = self.config.api_key.encode()
            self.response.read.reset_mock()
            self.session.post.reset_mock()
            with self.subTest(status=status), self.assertRaises(PlannerError) as caught:
                await self.request()
            messages[status] = str(caught.exception)
            self.assertNotIn(self.config.api_key, messages[status])
            self.response.read.assert_not_awaited()
            self.session.post.assert_called_once()
        self.assertNotEqual(messages[401], messages[429])
        self.assertNotEqual(messages[429], messages[500])

    async def test_network_timeout_and_unsafe_plan_are_distinct_and_sanitized(self) -> None:
        messages = []
        for error in (aiohttp.ClientConnectionError(self.config.api_key), TimeoutError(self.config.api_key)):
            self.response.read.side_effect = error
            with self.assertRaises(PlannerError) as caught:
                await self.request()
            messages.append(str(caught.exception))
            self.assertNotIn(self.config.api_key, messages[-1])
            self.assertTrue(caught.exception.__suppress_context__)
        self.response.read.side_effect = None
        unsafe = plan_payload()
        unsafe["target_object"] = "bottle"
        self.set_response(response_payload(unsafe))
        with self.assertRaises(PlannerError) as caught:
            await self.request()
        messages.append(str(caught.exception))
        self.assertEqual(len(set(messages)), 3)

    async def test_cancellation_propagates_and_closes_the_request_contexts(self) -> None:
        reading = asyncio.Event()

        async def blocked_read():
            reading.set()
            await asyncio.Future()

        self.response.read.side_effect = blocked_read
        task = asyncio.create_task(self.request())
        await reading.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.response.__aexit__.assert_awaited_once()
        self.session.__aexit__.assert_awaited_once()
        self.session.post.assert_called_once()

    async def test_unexpected_programmer_errors_are_not_converted_to_fallbacks(self) -> None:
        self.response.read.side_effect = RuntimeError("programmer-error")
        with self.assertRaises(RuntimeError):
            await self.request()


if __name__ == "__main__":
    unittest.main()
