"""Local live view of MuJoCo physics, not a recorded trajectory player."""

import argparse
import asyncio
import json
import logging
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
import time

from aiohttp import WSMsgType, web

from grasp import GraspParameters
from planner import PlannerError
from planning import PlanningSession
from scenarios import DEFAULT_CASE, TARGETS, SceneCase
from simulation import Simulation


@dataclass
class Controls:
    paused: bool = False
    view: str = "overview"
    runtime_error: str | None = None


ROOT = Path(__file__).resolve().parent
COMPARISON_REPORTS = {
    "initial": "planning_comparison.json",
    "followup": "planning_comparison_followup.json",
}
SIMULATION = web.AppKey("simulation", Simulation)
CLIENTS = web.AppKey("clients", set)
CONTROLS = web.AppKey("controls", Controls)
PLANNING = web.AppKey("planning", PlanningSession)
SCENE_CASE = web.AppKey("scene_case", SceneCase)
FRAME_PERIOD = 0.05
LOGGER = logging.getLogger(__name__)
WORKER_FAILURE_MESSAGE = "仿真工作线程已停止；请查看服务日志并重启本地服务后重新连接。"


def public_state(app: web.Application) -> dict:
    if app[CONTROLS].runtime_error is not None:
        # Failed physics/rendering may leave simulation state unreadable.
        return {"paused": True, "runtime_error": app[CONTROLS].runtime_error}
    state = app[SIMULATION].state()
    state["paused"] = app[CONTROLS].paused
    state["runtime_error"] = None
    state["planner"] = app[PLANNING].state(state["skill"])
    return state


def packet(app: web.Application) -> dict:
    return {"state": public_state(app), "image": app[SIMULATION].render(app[CONTROLS].view)}


async def broadcast(app: web.Application) -> None:
    if app[CONTROLS].runtime_error is not None or not app[CLIENTS]:
        return
    message = packet(app)
    for client in tuple(app[CLIENTS]):
        if app[CONTROLS].runtime_error is not None:
            return
        try:
            await asyncio.wait_for(client.send_json(message), timeout=1)
        except (ConnectionError, asyncio.TimeoutError):
            app[CLIENTS].discard(client)
            await client.close()


async def close_failed_websocket(app: web.Application, ws: web.WebSocketResponse) -> None:
    try:
        await asyncio.wait_for(
            ws.send_json({"error": app[CONTROLS].runtime_error, "fatal": True}), timeout=1,
        )
    except Exception:
        LOGGER.warning("Could not notify client of fatal simulation failure", exc_info=True)
    try:
        await ws.close(code=1011, message=b"Simulation failed; restart server")
    except Exception:
        LOGGER.warning("Could not close failed simulation connection", exc_info=True)


async def stop_failed_runtime(app: web.Application) -> None:
    if app[CONTROLS].runtime_error is not None:
        return
    app[CONTROLS].runtime_error = WORKER_FAILURE_MESSAGE
    app[CONTROLS].paused = True
    LOGGER.exception("Physics execution stopped; restart the local server to recover")
    try:
        await app[PLANNING].close()
    except Exception:
        LOGGER.exception("Could not stop planner after fatal simulation failure")
    for client in tuple(app[CLIENTS]):
        await close_failed_websocket(app, client)
        app[CLIENTS].discard(client)


async def physics_loop(app: web.Application) -> None:
    try:
        while app[CONTROLS].runtime_error is None:
            started = time.monotonic()
            allow_execution = bool(app[CLIENTS]) and not app[CONTROLS].paused
            if allow_execution:
                app[SIMULATION].step(FRAME_PERIOD)
            app[PLANNING].advance(allow_execution=allow_execution)
            await broadcast(app)
            await asyncio.sleep(max(0, FRAME_PERIOD - (time.monotonic() - started)))
    except asyncio.CancelledError:
        raise
    except Exception:
        await stop_failed_runtime(app)


async def lifecycle(app: web.Application):
    app[SIMULATION] = Simulation(app[SCENE_CASE])
    app[PLANNING] = PlanningSession(ROOT, app[SIMULATION])
    task = asyncio.create_task(physics_loop(app), name="mujoco-physics")
    try:
        yield
    finally:
        task.cancel()
        try:
            with suppress(asyncio.CancelledError):
                await task
        finally:
            try:
                await app[PLANNING].close()
            finally:
                try:
                    for client in tuple(app[CLIENTS]):
                        try:
                            await client.close(code=1001, message=b"Server shutdown")
                        except (ConnectionError, asyncio.TimeoutError):
                            LOGGER.exception("Could not close a client during server shutdown")
                finally:
                    app[SIMULATION].close()


@web.middleware
async def local_only(request: web.Request, handler):
    if request.url.host not in ("127.0.0.1", "localhost"):
        raise web.HTTPForbidden(text="Only localhost access is allowed")
    return await handler(request)


async def websocket(request: web.Request) -> web.WebSocketResponse:
    # The server is local-only; reject cross-origin browser control requests.
    origin = request.headers.get("Origin")
    if origin is not None and origin != f"http://{request.host}":
        raise web.HTTPForbidden(text="Cross-origin simulation control is not allowed")
    ws = web.WebSocketResponse(heartbeat=20, max_msg_size=16384)
    await ws.prepare(request)
    app = request.app
    if app[CONTROLS].runtime_error is not None:
        await close_failed_websocket(app, ws)
        return ws
    app[CLIENTS].add(ws)
    try:
        await ws.send_json(packet(app))
        async for msg in ws:
            if app[CONTROLS].runtime_error is not None:
                await close_failed_websocket(app, ws)
                break
            if msg.type == WSMsgType.ERROR:
                break
            if msg.type != WSMsgType.TEXT:
                continue
            try:
                command = msg.json()
                if not isinstance(command, dict):
                    raise ValueError("命令必须是 JSON 对象")
                action = command.get("action")
                planner = app[PLANNING]
                if action in ("grasp", "store", "skill") and planner.state()["locks_scene"]:
                    raise ValueError("GPT 决策循环已锁定场景；请先停止后续决策并等待当前技能结束，或重置")
                if action == "pause":
                    app[CONTROLS].paused = True
                elif action == "play":
                    app[CONTROLS].paused = False
                elif action == "reset":
                    planner.reset()
                    app[SIMULATION].reset()
                    app[CONTROLS].paused = True
                elif action in ("grasp", "store"):
                    parameters = command.get("parameters", {})
                    if not isinstance(parameters, dict):
                        raise ValueError("抓取参数必须是 JSON 对象")
                    task_kind = "store_bowl" if action == "store" else "grasp"
                    app[SIMULATION].start_task(task_kind, GraspParameters(**parameters))
                    planner.reset()
                    app[CONTROLS].paused = False
                elif action == "skill":
                    skill_name = command.get("skill")
                    if not isinstance(skill_name, str):
                        raise ValueError("技能名称必须是字符串")
                    parameters = command.get("parameters")
                    if parameters is not None:
                        if skill_name != "grasp_and_hold":
                            raise ValueError("仅 grasp_and_hold 接受抓取参数；其他技能请省略 parameters")
                        if not isinstance(parameters, dict):
                            raise ValueError("抓取参数必须是 JSON 对象")
                        parameters = GraspParameters(**parameters)
                    app[SIMULATION].start_skill(skill_name, parameters)
                    planner.reset()
                    app[CONTROLS].paused = False
                elif action == "step":
                    if not app[CONTROLS].paused:
                        raise ValueError("请先暂停，再单步推进")
                    app[SIMULATION].step(0.02)
                    planner.advance(allow_execution=False)
                elif action == "plan":
                    try:
                        report = load_experiment_report()
                    except (OSError, json.JSONDecodeError) as exc:
                        raise ValueError("无法读取本地参数实验记录，未调用 GPT；请检查 results/parameter_sweep.json") from exc
                    if not isinstance(report, dict):
                        raise ValueError("参数实验报告必须是 JSON 对象，未调用 GPT")
                    planner.start(command.get("task"), report)
                elif action == "execute_decision":
                    continuous = command.get("continuous", False)
                    if not isinstance(continuous, bool):
                        raise ValueError("continuous 必须是布尔值")
                    planner.execute(command.get("decision_id"), continuous=continuous)
                elif action == "plan_next":
                    planner.next()
                elif action == "cancel_plan":
                    planner.cancel()
                elif action == "camera":
                    view = command.get("view")
                    if view not in ("overview", "front", "top"):
                        raise ValueError("未知视角")
                    app[CONTROLS].view = view
                else:
                    raise ValueError("未知命令")
            except (ValueError, TypeError, PlannerError) as exc:
                await ws.send_json({"error": str(exc)})
                continue
            except RuntimeError:
                await stop_failed_runtime(app)
                break
            await broadcast(app)
    finally:
        app[CLIENTS].discard(ws)
    return ws


async def state(request: web.Request) -> web.Response:
    return web.json_response(
        public_state(request.app),
        status=503 if request.app[CONTROLS].runtime_error is not None else 200,
        headers={"Cache-Control": "no-store"},
    )


def load_experiment_report() -> dict:
    return json.loads((ROOT / "results" / "parameter_sweep.json").read_text(encoding="utf-8"))


async def experiments(request: web.Request) -> web.Response:
    try:
        report = load_experiment_report()
    except FileNotFoundError as exc:
        raise web.HTTPNotFound(text="尚无参数实验结果；请先运行 experiments.py") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise web.HTTPInternalServerError(text=f"读取参数实验结果失败：{exc}") from exc
    return web.json_response(report, headers={"Cache-Control": "no-store"})


async def planning_comparison(request: web.Request) -> web.Response:
    batches = request.query.getall("batch", [])
    if len(batches) > 1:
        raise web.HTTPBadRequest(text="一次只能选择一个实验批次")
    batch = batches[0] if batches else "initial"
    if batch not in COMPARISON_REPORTS:
        raise web.HTTPBadRequest(text="未知实验批次；请选择 initial 或 followup")
    filename = COMPARISON_REPORTS[batch]
    try:
        report = json.loads((ROOT / "results" / filename).read_text(encoding="utf-8"))
        if not isinstance(report, dict) or report.get("kind") != "paired_closed_loop_comparison":
            raise ValueError("不是闭环规划对照报告")
    except FileNotFoundError as exc:
        raise web.HTTPNotFound(text=f"所选批次 {batch} 尚无结果；请先运行对应的 compare_planning.py 批次") from exc
    except (OSError, ValueError) as exc:
        raise web.HTTPInternalServerError(text=f"读取闭环对照结果失败：{exc}") from exc
    return web.json_response(report, headers={
        "Cache-Control": "no-store",
        "Content-Disposition": f'attachment; filename="{filename}"',
    })


async def asset(request: web.Request) -> web.FileResponse:
    name = request.match_info.get("name", "index.html")
    if name not in ("index.html", "app.js", "style.css"):
        raise web.HTTPNotFound()
    return web.FileResponse(ROOT / "web" / name, headers={"Cache-Control": "no-cache"})


def create_app(case: SceneCase = DEFAULT_CASE) -> web.Application:
    app = web.Application(middlewares=[local_only])
    app[CLIENTS] = set()
    app[CONTROLS] = Controls()
    app[SCENE_CASE] = case
    app.cleanup_ctx.append(lifecycle)
    app.router.add_get("/ws", websocket)
    app.router.add_get("/api/state", state)
    app.router.add_get("/api/experiments", experiments)
    app.router.add_get("/api/planning-comparison", planning_comparison)
    app.router.add_get("/", asset)
    app.router.add_get("/{name}", asset)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--target", choices=tuple(TARGETS), default="bowl")
    args = parser.parse_args()
    case = SceneCase(f"{args.target}.nominal", args.target, "nominal", 0, (0.0, 0.0), 0.0)
    web.run_app(create_app(case), host="127.0.0.1", port=args.port, access_log=None)


if __name__ == "__main__":
    main()
