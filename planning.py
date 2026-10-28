"""Bind a validated GPT plan to one frozen simulation scene."""

import asyncio
from contextlib import suppress
import logging
from pathlib import Path
import secrets
import time
import traceback

from planner import DEFAULT_MODEL, PlannerConfig, PlannerError, PlanResult, request_plan
from simulation import Simulation


LOGGER = logging.getLogger(__name__)


class PlanningSession:
    def __init__(self, root: Path, simulation: Simulation) -> None:
        self._root = root
        self._simulation = simulation
        self._config: PlannerConfig | None = None
        self._pending: asyncio.Task | None = None
        self._tasks: set[asyncio.Task] = set()
        # Distinguish restarted servers while keeping IDs exact in JavaScript.
        self._revision = secrets.randbits(52)
        self._status = "unconfigured"
        self._task_text = ""
        self._result: PlanResult | None = None
        self._latency_s: float | None = None
        self._error: str | None = None
        self._load_config()

    @property
    def locks_scene(self) -> bool:
        return self._status in ("planning", "ready")

    def _load_config(self) -> bool:
        try:
            self._config = PlannerConfig.load(self._root)
        except PlannerError as exc:
            self._config = None
            self._status = "failed"
            self._error = str(exc)
            return False
        self._status = "idle" if self._config.configured else "unconfigured"
        self._error = None
        return self._config.configured

    def _invalidate(self, status: str) -> None:
        self._revision += 1
        if self._pending is not None:
            self._pending.cancel()
            self._pending = None
        self._result = None
        self._latency_s = None
        self._error = None
        self._status = status

    def start(self, task: object, experiments: dict) -> bool:
        if not isinstance(task, str) or not 1 <= len(task.strip()) <= 1000:
            raise ValueError("任务必须是 1–1000 字的文本")
        if self.locks_scene:
            raise ValueError("已有规划正在生成或等待执行，请先取消或重置")
        if self._simulation.skill.status != "idle":
            raise ValueError("请先重置场景，再生成新的 GPT 计划")
        self._invalidate("idle")
        self._task_text = task.strip()
        if not self._load_config():
            if self._status == "unconfigured":
                self._error = "未找到官方 Codex CLI；请安装并运行 codex login 使用当前 ChatGPT 订阅。"
            return False
        scene = self._simulation.state()
        self._status = "planning"
        pending = asyncio.create_task(
            self._generate(self._revision, self._config, self._task_text, scene, experiments),
            name=f"gpt-plan-{self._revision}",
        )
        self._pending = pending
        self._tasks.add(pending)
        pending.add_done_callback(self._tasks.discard)
        return True

    async def _generate(
        self, revision: int, config: PlannerConfig, task: str, scene: dict, experiments: dict,
    ) -> None:
        started = time.monotonic()
        try:
            result = await request_plan(config, task, scene, experiments)
            if revision == self._revision and self._status == "planning":
                self._result = result
                self._latency_s = result.latency_s
                self._status = "ready" if result.plan.decision == "plan" else "unsupported"
        except asyncio.CancelledError:
            raise
        except PlannerError as exc:
            if revision == self._revision and self._status == "planning":
                self._latency_s = time.monotonic() - started
                self._status = "failed"
                self._error = str(exc)
        except Exception as exc:
            LOGGER.error(
                "GPT planning failed unexpectedly (%s):\n%s",
                type(exc).__name__, "".join(traceback.format_tb(exc.__traceback__)),
            )
            if revision == self._revision and self._status == "planning":
                self._latency_s = time.monotonic() - started
                self._status = "failed"
                self._error = "规划器发生内部错误，未执行任何动作；请查看服务日志。"
        finally:
            if revision == self._revision:
                self._pending = None

    def execute(self, plan_id: object) -> None:
        if type(plan_id) is not int or plan_id != self._revision:
            raise ValueError("计划已过期，请使用当前已校验的计划")
        if self._status != "ready" or self._result is None:
            raise ValueError("没有等待执行的已校验计划")
        if self._simulation.skill.status != "idle":
            raise ValueError("场景已开始其他任务，请先重置并重新规划")
        parameters = self._result.plan.parameters
        if parameters is None:
            raise ValueError("该计划不包含可执行的抓取动作")
        self._simulation.start_grasp(parameters)
        self._status = "executing"

    def cancel(self) -> None:
        if not self.locks_scene:
            raise ValueError("当前没有可取消的规划；已开始的物理执行请使用暂停或重置")
        self._invalidate("cancelled")

    def reset(self) -> None:
        self._invalidate("idle")
        self._task_text = ""
        self._load_config()

    def state(self, skill: dict) -> dict:
        if self._status in ("executing", "succeeded"):
            if skill["status"] == "succeeded":
                self._status = "succeeded"
            elif skill["status"] == "failed":
                self._status = "failed"
                self._error = skill["failure_reason"]
        return {
            "status": self._status,
            "configured": self._config is not None and self._config.configured,
            "model": self._result.model if self._result else (
                self._config.model if self._config else DEFAULT_MODEL
            ),
            "task": self._task_text,
            "plan_id": self._revision,
            "plan": self._result.plan.to_dict() if self._result else None,
            "latency_s": self._latency_s,
            "thread_id": self._result.thread_id if self._result else None,
            "diagnostic_count": self._result.diagnostic_count if self._result else 0,
            "error": self._error,
        }

    async def close(self) -> None:
        self._invalidate("cancelled")
        tasks = tuple(self._tasks)
        for task in tasks:
            task.cancel()
        for task in tasks:
            with suppress(asyncio.CancelledError):
                await task
