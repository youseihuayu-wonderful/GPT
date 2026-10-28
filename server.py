"""Local live view of MuJoCo physics, not a recorded trajectory player."""

import argparse
import asyncio
import json
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
import time

from aiohttp import WSMsgType, web

from grasp import GraspParameters
from planning import PlanningSession
from simulation import Simulation


@dataclass
class Controls:
    paused: bool = False
    view: str = "overview"


ROOT = Path(__file__).resolve().parent
SIMULATION = web.AppKey("simulation", Simulation)
CLIENTS = web.AppKey("clients", set)
CONTROLS = web.AppKey("controls", Controls)
PLANNING = web.AppKey("planning", PlanningSession)
FRAME_PERIOD = 0.05


def public_state(app: web.Application) -> dict:
    state = app[SIMULATION].state()
    state["paused"] = app[CONTROLS].paused
    state["planner"] = app[PLANNING].state(state["skill"])
    return state


def packet(app: web.Application) -> dict:
    return {"state": public_state(app), "image": app[SIMULATION].render(app[CONTROLS].view)}


async def broadcast(app: web.Application) -> None:
    if not app[CLIENTS]:
        return
    message = packet(app)
    for client in tuple(app[CLIENTS]):
        try:
            await asyncio.wait_for(client.send_json(message), timeout=1)
        except (ConnectionError, asyncio.TimeoutError):
            app[CLIENTS].discard(client)
            await client.close()


async def physics_loop(app: web.Application) -> None:
    try:
        while True:
            started = time.monotonic()
            if app[CLIENTS] and not app[CONTROLS].paused:
                app[SIMULATION].step(FRAME_PERIOD)
            await broadcast(app)
            await asyncio.sleep(max(0, FRAME_PERIOD - (time.monotonic() - started)))
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        app[CONTROLS].paused = True
        for client in tuple(app[CLIENTS]):
            with suppress(ConnectionError):
                await client.send_json({"error": f"仿真停止：{exc}"})
                await client.close(code=1011, message=b"Simulation failed")
        raise


async def lifecycle(app: web.Application):
    app[SIMULATION] = Simulation()
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
                for client in tuple(app[CLIENTS]):
                    await client.close(code=1001, message=b"Server shutdown")
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
    app[CLIENTS].add(ws)
    try:
        await ws.send_json(packet(app))
        async for msg in ws:
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
                if action in ("play", "step", "grasp") and planner.locks_scene:
                    raise ValueError("规划中的场景已锁定，请先取消计划或重置，再手动运行")
                if action == "pause":
                    app[CONTROLS].paused = True
                elif action == "play":
                    app[CONTROLS].paused = False
                elif action == "reset":
                    planner.reset()
                    app[SIMULATION].reset()
                    app[CONTROLS].paused = True
                elif action == "grasp":
                    parameters = command.get("parameters", {})
                    if not isinstance(parameters, dict):
                        raise ValueError("抓取参数必须是 JSON 对象")
                    app[SIMULATION].start_grasp(GraspParameters(**parameters))
                    app[CONTROLS].paused = False
                elif action == "step":
                    if not app[CONTROLS].paused:
                        raise ValueError("请先暂停，再单步推进")
                    app[SIMULATION].step(0.02)
                elif action == "plan":
                    try:
                        report = load_experiment_report()
                    except (OSError, json.JSONDecodeError) as exc:
                        raise ValueError("无法读取本地参数实验记录，未调用 GPT；请检查 results/parameter_sweep.json") from exc
                    if not isinstance(report, dict):
                        raise ValueError("参数实验报告必须是 JSON 对象，未调用 GPT")
                    if planner.start(command.get("task"), report):
                        app[CONTROLS].paused = True
                elif action == "execute_plan":
                    planner.execute(command.get("plan_id"))
                    app[CONTROLS].paused = False
                elif action == "cancel_plan":
                    planner.cancel()
                    app[CONTROLS].paused = True
                elif action == "camera":
                    view = command.get("view")
                    if view not in ("overview", "front", "top"):
                        raise ValueError("未知视角")
                    app[CONTROLS].view = view
                else:
                    raise ValueError("未知命令")
            except (ValueError, TypeError) as exc:
                await ws.send_json({"error": str(exc)})
                continue
            await broadcast(app)
    finally:
        app[CLIENTS].discard(ws)
    return ws


async def state(request: web.Request) -> web.Response:
    return web.json_response(public_state(request.app), headers={"Cache-Control": "no-store"})


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


async def asset(request: web.Request) -> web.FileResponse:
    name = request.match_info.get("name", "index.html")
    if name not in ("index.html", "app.js", "style.css"):
        raise web.HTTPNotFound()
    return web.FileResponse(ROOT / "web" / name, headers={"Cache-Control": "no-cache"})


def create_app() -> web.Application:
    app = web.Application(middlewares=[local_only])
    app[CLIENTS] = set()
    app[CONTROLS] = Controls()
    app.cleanup_ctx.append(lifecycle)
    app.router.add_get("/ws", websocket)
    app.router.add_get("/api/state", state)
    app.router.add_get("/api/experiments", experiments)
    app.router.add_get("/", asset)
    app.router.add_get("/{name}", asset)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    web.run_app(create_app(), host="127.0.0.1", port=args.port, access_log=None)


if __name__ == "__main__":
    main()
