"""Authorize and observe single GPT-selected skills; callers own the physics clock."""

import asyncio
from contextlib import suppress
from copy import deepcopy
import logging
from pathlib import Path
import secrets
import time
import traceback

from planner import (
    DEFAULT_MODEL, DecisionResult, PlannerConfig, PlannerError, request_decision,
    scene_matches, validate_context, validate_decision,
)
from simulation import Simulation


LOGGER = logging.getLogger(__name__)
MAX_SKILLS = 8
MAX_REQUESTS = MAX_SKILLS + 1


class PlanningSession:
    def __init__(self, root: Path, simulation: Simulation) -> None:
        self._root = root
        self._simulation = simulation
        self._config: PlannerConfig | None = None
        self._pending: asyncio.Task | None = None
        self._tasks: set[asyncio.Task] = set()
        # Distinguish restarted servers while keeping IDs exact in JavaScript.
        self._revision = secrets.randbits(52)
        self._closed = False
        self._clear()
        self._load_config()

    @property
    def locks_scene(self) -> bool:
        return self._status in ("planning", "ready", "executing", "waiting", "stopping")

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

    def _clear(self) -> None:
        self._status = "idle"
        self._task_text = ""
        self._experiments: dict = {}
        self._result: DecisionResult | None = None
        self._goal: str | None = None
        self._continuous = False
        self._history: list[dict] = []
        self._skill_count = 0
        self._scene_revision: int | None = None
        self._invocation_id: int | None = None
        self._request_started_wall: float | None = None
        self._execution_started_wall: float | None = None
        self._latency_s: float | None = None
        self._total_latency_s = 0.0
        self._error: str | None = None

    def _set_latency(self, latency: float) -> None:
        self._latency_s = latency
        self._total_latency_s += latency
        self._history[-1]["latency_s"] = latency
        self._request_started_wall = None

    def _stop_pending(self) -> None:
        self._revision += 1
        if self._request_started_wall is not None:
            self._set_latency(time.monotonic() - self._request_started_wall)
        if self._pending is not None:
            self._pending.cancel()
            self._pending = None

    def _record_execution(self, scene: dict) -> None:
        record = self._history[-1]
        record["result_observation"] = deepcopy(scene)
        if self._execution_started_wall is not None:
            record["execution_sim_s"] = scene["skill"]["elapsed_s"]
            record["execution_wall_s"] = time.monotonic() - self._execution_started_wall
            self._execution_started_wall = None

    def _fail(self, error: str, scene: dict | None = None) -> None:
        self._stop_pending()
        self._status = "failed"
        self._error = error
        if self._history:
            record = self._history[-1]
            if scene is not None:
                self._record_execution(scene)
            elif self._execution_started_wall is not None:
                record["execution_wall_s"] = time.monotonic() - self._execution_started_wall
                self._execution_started_wall = None
            record["status"] = "failed"
            record["error"] = error

    def _observe(self) -> dict:
        # Never serialize a numerically invalid simulation, even to report failure.
        error = self._simulation.numerical_error()
        if error is not None:
            self._fail(error)
            raise PlannerError(error)
        scene = deepcopy(self._simulation.state())
        skill = scene["skill"]
        if skill["status"] == "failed":
            error = skill["failure_reason"] or "物理技能执行失败，已停止后续决策。"
        elif (skill["status"] == "succeeded" and
              (not skill["postconditions"] or not all(skill["postconditions"].values()))):
            error = "已完成技能的真实物理后置条件丢失，已停止后续决策。"
        elif self._scene_revision is not None and (
            scene["scene_revision"] != self._scene_revision or
            skill["invocation_id"] != self._invocation_id
        ):
            error = "规划期间场景或技能调用已被替换，本次决策已过期。"
        if error is not None:
            self._fail(error, scene)
            raise PlannerError(error)
        return scene

    def start(self, task: object, experiments: dict) -> bool:
        if not isinstance(task, str) or not 1 <= len(task.strip()) <= 1000:
            raise ValueError("任务必须是 1–1000 字的文本")
        if self._closed:
            raise ValueError("规划会话已关闭")
        if self.locks_scene:
            raise ValueError("已有单步决策会话，请先停止或重置")
        error = self._simulation.numerical_error()
        if error is not None:
            self._fail(error)
            raise PlannerError(error)
        if self._simulation.skill.status not in ("idle", "succeeded"):
            raise ValueError("当前物理技能尚未成功结束，不能开始新的决策")
        self._stop_pending()
        self._clear()
        self._task_text = task.strip()
        self._experiments = deepcopy(experiments)
        if not self._load_config():
            if self._status == "unconfigured":
                self._error = "未找到官方 Codex CLI；请安装并运行 codex login 使用当前 ChatGPT 订阅。"
            return False
        scene = self._observe()
        self._scene_revision = scene["scene_revision"]
        self._invocation_id = scene["skill"]["invocation_id"]
        self._queue(scene)
        return True

    def _queue(self, scene: dict) -> None:
        if len(self._history) >= MAX_REQUESTS:
            error = "已达到 9 次模型请求上限，未继续规划或执行。"
            self._fail(error, scene)
            raise PlannerError(error)
        history = deepcopy(self._history)
        self._revision += 1
        self._result = None
        self._latency_s = None
        self._error = None
        self._status = "planning"
        self._history.append({
            "index": len(self._history) + 1, "decision_id": self._revision,
            "observation": deepcopy(scene), "result_observation": None,
            "decision": None, "model": self._config.model, "thread_id": None,
            "latency_s": None, "diagnostic_count": 0, "status": "planning",
            "error": None, "invocation_id": None,
            "execution_sim_s": None, "execution_wall_s": None,
        })
        self._request_started_wall = time.monotonic()
        pending = asyncio.create_task(
            self._generate(self._revision, deepcopy(scene), history),
            name=f"gpt-decision-{self._revision}",
        )
        self._pending = pending
        self._tasks.add(pending)
        pending.add_done_callback(self._tasks.discard)

    async def _generate(self, revision: int, scene: dict, history: list[dict]) -> None:
        started = time.monotonic()
        try:
            result = await request_decision(
                self._config, self._task_text, scene, deepcopy(self._experiments),
                goal=self._goal, history=history,
            )
            if revision != self._revision or self._status != "planning":
                return
            self._pending = None
            self._result = result
            self._set_latency(result.latency_s)
            record = self._history[-1]
            record.update({
                "decision": result.decision.to_dict(), "model": result.model,
                "thread_id": result.thread_id, "diagnostic_count": result.diagnostic_count,
            })
            validate_context(result.decision, record["observation"], self._goal)
            self._status = "unsupported" if result.decision.decision == "unsupported" else "ready"
            record["status"] = self._status
        except asyncio.CancelledError:
            if revision == self._revision and self._status == "planning":
                self._pending = None
                self._fail("模型请求已取消，未执行本次动作。")
            raise
        except PlannerError as exc:
            if revision == self._revision and self._status == "planning":
                self._pending = None
                if self._request_started_wall is not None:
                    self._set_latency(time.monotonic() - started)
                self._fail(str(exc))
        except Exception as exc:
            LOGGER.error(
                "GPT decision failed unexpectedly (%s):\n%s",
                type(exc).__name__, "".join(traceback.format_tb(exc.__traceback__)),
            )
            if revision == self._revision and self._status == "planning":
                self._pending = None
                if self._request_started_wall is not None:
                    self._set_latency(time.monotonic() - started)
                self._fail("规划器发生内部错误，未执行本次动作；请查看服务日志。")
        finally:
            if revision == self._revision:
                self._pending = None

    def execute(self, decision_id: object, continuous: bool = False) -> None:
        if type(decision_id) is not int or decision_id != self._revision:
            raise ValueError("决策已过期，请使用当前已校验的决策")
        if type(continuous) is not bool:
            raise ValueError("连续执行授权必须是布尔值")
        if self._status != "ready" or self._result is None:
            raise ValueError("没有等待执行的已校验决策")
        scene = self._observe()
        try:
            decision = validate_decision(self._result.decision.to_dict())
            if not scene_matches(self._history[-1]["observation"], scene):
                raise PlannerError("真实场景已偏离决策观测，本次决策已过期；未执行动作。")
            validate_context(decision, scene, self._goal)
            if decision.decision == "unsupported":
                raise PlannerError("模型拒绝了该任务，不能执行。")
            if decision.decision == "execute":
                if self._skill_count >= MAX_SKILLS:
                    raise PlannerError("已达到 8 次技能调用上限，未执行更多动作。")
                self._simulation.start_skill(decision.skill, decision.parameters)
        except (PlannerError, ValueError) as exc:
            self._fail(str(exc), scene)
            raise
        except Exception as exc:
            LOGGER.error(
                "Skill dispatch failed unexpectedly (%s):\n%s",
                type(exc).__name__, "".join(traceback.format_tb(exc.__traceback__)),
            )
            error = "技能启动发生内部错误，已停止后续决策；请查看服务日志。"
            self._fail(error, scene)
            raise PlannerError(error) from None
        self._goal = decision.goal
        self._continuous = continuous
        record = self._history[-1]
        if decision.decision == "complete":
            self._status = "succeeded"
            record["status"] = "complete"
            record["result_observation"] = deepcopy(scene)
            return
        self._skill_count += 1
        self._invocation_id = self._simulation.skill.invocation_id
        self._execution_started_wall = time.monotonic()
        self._status = "executing"
        record["status"] = "executing"
        record["invocation_id"] = self._invocation_id

    def next(self) -> None:
        if self._status != "waiting":
            raise ValueError("只有上一技能真实成功后，才能请求下一步决策")
        self._queue(self._observe())

    def advance(self, allow_execution: bool = True) -> None:
        if self._status not in ("planning", "ready", "executing", "waiting", "stopping", "succeeded"):
            return
        scene = None
        try:
            scene = self._observe()
            if self._status == "succeeded":
                validate_context(self._result.decision, scene, self._goal)
                return
            if self._status in ("executing", "stopping"):
                if scene["skill"]["status"] == "running":
                    return
                if scene["skill"]["status"] != "succeeded":
                    raise PlannerError("技能未得到真实成功结果，已停止后续决策。")
                stopping = self._status == "stopping"
                self._record_execution(scene)
                self._history[-1]["status"] = "succeeded"
                self._status = "cancelled" if stopping else "waiting"
            if self._status == "waiting" and self._continuous and allow_execution:
                self._queue(scene)
            elif self._status == "ready" and self._continuous and allow_execution:
                self.execute(self._revision, continuous=True)
        except (PlannerError, ValueError) as exc:
            if self._status != "failed":
                self._fail(str(exc), scene)

    def cancel(self) -> None:
        if not self.locks_scene:
            raise ValueError("当前没有可停止的决策会话")
        self._stop_pending()
        self._continuous = False
        if self._status in ("executing", "stopping"):
            self._status = "stopping"
            return
        if self._status in ("planning", "ready"):
            self._history[-1]["status"] = "cancelled"
        self._status = "cancelled"

    def reset(self) -> None:
        self._stop_pending()
        self._clear()
        self._load_config()

    def state(self, skill: dict | None = None) -> dict:
        return {
            "status": self._status,
            "configured": self._config is not None and self._config.configured,
            "model": self._result.model if self._result else (
                self._config.model if self._config else DEFAULT_MODEL
            ),
            "task": self._task_text, "decision_id": self._revision,
            "decision": self._result.decision.to_dict() if self._result else None,
            "goal": self._goal, "continuous": self._continuous, "locks_scene": self.locks_scene,
            "latency_s": self._latency_s, "total_latency_s": self._total_latency_s,
            "thread_id": self._result.thread_id if self._result else None,
            "diagnostic_count": self._result.diagnostic_count if self._result else 0,
            "history": deepcopy(self._history), "max_skills": MAX_SKILLS, "error": self._error,
        }

    async def close(self) -> None:
        self._closed = True
        if self.locks_scene:
            self._stop_pending()
            if self._status != "waiting":
                self._history[-1]["status"] = "cancelled"
            if self._execution_started_wall is not None:
                self._history[-1]["execution_wall_s"] = time.monotonic() - self._execution_started_wall
                self._execution_started_wall = None
            self._status = "cancelled"
            self._continuous = False
        tasks = tuple(self._tasks)
        for task in tasks:
            task.cancel()
        for task in tasks:
            with suppress(asyncio.CancelledError):
                await task
