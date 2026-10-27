"""Local live view of MuJoCo physics, not a recorded trajectory player."""

import argparse
import asyncio
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
import time

from aiohttp import WSMsgType, web

from simulation import Simulation


@dataclass
class Controls:
    paused: bool = False
    view: str = "overview"


ROOT = Path(__file__).resolve().parent
SIMULATION = web.AppKey("simulation", Simulation)
CLIENTS = web.AppKey("clients", set)
CONTROLS = web.AppKey("controls", Controls)
FRAME_PERIOD = 0.05


def packet(app: web.Application) -> dict:
    state = app[SIMULATION].state()
    state["paused"] = app[CONTROLS].paused
    return {"state": state, "image": app[SIMULATION].render(app[CONTROLS].view)}


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
    task = asyncio.create_task(physics_loop(app), name="mujoco-physics")
    try:
        yield
    finally:
        task.cancel()
        try:
            with suppress(asyncio.CancelledError):
                await task
        finally:
            for client in tuple(app[CLIENTS]):
                await client.close(code=1001, message=b"Server shutdown")
            app[SIMULATION].close()


async def websocket(request: web.Request) -> web.WebSocketResponse:
    # The server is local-only; reject cross-origin browser control requests.
    origin = request.headers.get("Origin")
    if origin is not None and origin != f"http://{request.host}":
        raise web.HTTPForbidden(text="Cross-origin simulation control is not allowed")
    ws = web.WebSocketResponse(heartbeat=20, max_msg_size=4096)
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
                if action == "pause":
                    app[CONTROLS].paused = True
                elif action == "play":
                    app[CONTROLS].paused = False
                elif action == "reset":
                    app[SIMULATION].reset()
                    app[CONTROLS].paused = True
                elif action == "grasp":
                    app[SIMULATION].start_grasp()
                    app[CONTROLS].paused = False
                elif action == "step":
                    if not app[CONTROLS].paused:
                        raise ValueError("请先暂停，再单步推进")
                    app[SIMULATION].step(0.02)
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
    result = request.app[SIMULATION].state()
    result["paused"] = request.app[CONTROLS].paused
    return web.json_response(result, headers={"Cache-Control": "no-store"})


async def asset(request: web.Request) -> web.FileResponse:
    name = request.match_info.get("name", "index.html")
    if name not in ("index.html", "app.js", "style.css"):
        raise web.HTTPNotFound()
    return web.FileResponse(ROOT / "web" / name, headers={"Cache-Control": "no-cache"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    app = web.Application()
    app[CLIENTS] = set()
    app[CONTROLS] = Controls()
    app.cleanup_ctx.append(lifecycle)
    app.router.add_get("/ws", websocket)
    app.router.add_get("/api/state", state)
    app.router.add_get("/", asset)
    app.router.add_get("/{name}", asset)
    web.run_app(app, host="127.0.0.1", port=args.port, access_log=None)


if __name__ == "__main__":
    main()
