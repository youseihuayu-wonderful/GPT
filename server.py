"""Local live view of MuJoCo physics, not a recorded trajectory player."""

import argparse
import asyncio
import json
import logging
import re
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
import time

from aiohttp import WSMsgType, web

from grasp import GraspParameters
from planner import PlannerError
from planning import PlanningSession
from sorting_planning import SortingPlanningSession
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
    "expanded": "planning_comparison_expanded.json",
    "drawer_state": "planning_comparison_drawer_state.json",
}
SIMULATION = web.AppKey("simulation", Simulation)
CLIENTS = web.AppKey("clients", set)
CONTROLS = web.AppKey("controls", Controls)
PLANNING = web.AppKey[PlanningSession | SortingPlanningSession]("planning")
SCENE_CASE = web.AppKey("scene_case", SceneCase)
SCENE = web.AppKey("scene", str)
SORTING_LAYOUT = web.AppKey[str | None]("sorting_layout")
SORTING_PROTOCOL = web.AppKey("sorting_protocol", dict)
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
    state["planner"] = app[PLANNING].state(
        state["sorting"] if app[SCENE] == "sorting" else state["skill"]
    )
    if app[SCENE] == "sorting":
        state["sorting_layout"] = app[SORTING_LAYOUT]
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
        if PLANNING in app:
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
            if PLANNING in app:
                app[PLANNING].advance(allow_execution=allow_execution)
            await broadcast(app)
            await asyncio.sleep(max(0, FRAME_PERIOD - (time.monotonic() - started)))
    except asyncio.CancelledError:
        raise
    except Exception:
        await stop_failed_runtime(app)


async def lifecycle(app: web.Application):
    if app[SCENE] == "sorting":
        from sorting import SortingSimulation

        if app[SORTING_LAYOUT] is None:
            app[SIMULATION] = SortingSimulation()
        else:
            from sorting_evaluation import make_simulation

            app[SIMULATION] = make_simulation(app[SORTING_PROTOCOL], app[SORTING_LAYOUT])
        app[PLANNING] = SortingPlanningSession(ROOT, app[SIMULATION])
    else:
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
                if PLANNING in app:
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
                sorting = app[SCENE] == "sorting"
                if sorting and action not in (
                    "sort", "sort_plan", "sort_execute", "sort_cancel",
                    "pause", "play", "reset", "step", "camera",
                ):
                    raise ValueError("分拣场景不支持收纳规划、抓取或独立技能命令")
                if not sorting and action in ("sort", "sort_plan", "sort_execute", "sort_cancel"):
                    raise ValueError("分拣命令仅适用于分拣场景")
                planner = app.get(PLANNING)
                if action in ("grasp", "store", "skill") and planner.state()["locks_scene"]:
                    raise ValueError("GPT 决策循环已锁定场景；请先停止后续决策并等待当前技能结束，或重置")
                if action == "pause":
                    app[CONTROLS].paused = True
                elif action == "play":
                    app[CONTROLS].paused = False
                elif action == "reset":
                    if planner is not None:
                        planner.reset()
                    app[SIMULATION].reset()
                    app[CONTROLS].paused = True
                elif action == "sort":
                    planning_state = planner.state()
                    if planning_state["locks_scene"]:
                        raise ValueError("GPT 分拣提案已锁定场景；请取消或重置")
                    if planning_state["task"] or planning_state["history"] or planning_state["calls"]:
                        raise ValueError("GPT 分拣尝试后须先重置，才能开始固定对照；历史不会自动清除")
                    app[SIMULATION].start_sorting()
                    app[CONTROLS].paused = False
                elif action == "sort_plan":
                    task = command.get("task")
                    if not isinstance(task, str):
                        raise ValueError("分拣任务必须是字符串")
                    planner.start(task)
                elif action == "sort_execute":
                    decision_id = command.get("decision_id")
                    if type(decision_id) is not int:
                        raise ValueError("decision_id 必须是整数提案编号")
                    planner.execute(decision_id)
                    app[CONTROLS].paused = False
                elif action == "sort_cancel":
                    planner.cancel()
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
                    if planner is not None:
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
        raise web.HTTPBadRequest(text="未知实验批次；请选择 initial、followup、expanded 或 drawer_state")
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


async def sorting_evaluation_report(request: web.Request) -> web.Response:
    from evaluate_sorting import load_evaluation_report

    try:
        report = await asyncio.to_thread(
            load_evaluation_report, ROOT / "results" / "sorting_frozen_evaluation",
        )
    except FileNotFoundError as exc:
        raise web.HTTPNotFound(
            text="尚无冻结版本分拣评估报告；不会显示旧批次或启动试验。",
            headers={"Cache-Control": "no-store"},
        ) from exc
    except (OSError, ValueError) as exc:
        raise web.HTTPInternalServerError(
            text=f"冻结版本分拣评估报告无效：{exc}",
            headers={"Cache-Control": "no-store"},
        ) from exc
    return web.json_response(report, headers={
        "Cache-Control": "no-store",
        "Content-Disposition": 'attachment; filename="sorting_frozen_evaluation_summary.json"',
    })


async def sorting_evaluation_run(request: web.Request) -> web.FileResponse:
    from evaluate_sorting import evaluation_run_path

    run_id = request.match_info["run_id"]
    headers = {"Cache-Control": "no-store"}
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", run_id) is None:
        raise web.HTTPNotFound(text="未知或不安全的评估记录编号", headers=headers)
    try:
        path = await asyncio.to_thread(
            evaluation_run_path, ROOT / "results" / "sorting_frozen_evaluation", run_id,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise web.HTTPNotFound(text="评估记录不存在或未通过完整性验证", headers=headers) from exc
    except OSError as exc:
        raise web.HTTPInternalServerError(text="无法读取评估记录", headers=headers) from exc
    return web.FileResponse(path, headers={
        **headers,
        "Content-Type": "application/gzip",
        "Content-Disposition": f'attachment; filename="{run_id}.json.gz"',
    })


async def asset(request: web.Request) -> web.FileResponse:
    name = request.match_info.get("name", "index.html")
    if name not in ("index.html", "app.js", "style.css", "localization.js"):
        raise web.HTTPNotFound()
    return web.FileResponse(ROOT / "web" / name, headers={"Cache-Control": "no-cache"})


def create_app(
    case: SceneCase = DEFAULT_CASE, *, scene: str = "storage", sorting_layout: str | None = None,
) -> web.Application:
    if scene not in ("storage", "sorting"):
        raise ValueError("scene 必须是 storage 或 sorting")
    protocol = None
    if sorting_layout is not None:
        if scene != "sorting":
            raise ValueError("sorting_layout 仅适用于 sorting 场景")
        from sorting_evaluation import load_protocol

        protocol = load_protocol()
        if not any(layout["id"] == sorting_layout for layout in protocol["layouts"]):
            raise ValueError(f"未知分拣布局：{sorting_layout}")
    app = web.Application(middlewares=[local_only])
    app[CLIENTS] = set()
    app[CONTROLS] = Controls()
    app[SCENE_CASE] = case
    app[SCENE] = scene
    app[SORTING_LAYOUT] = sorting_layout
    if protocol is not None:
        app[SORTING_PROTOCOL] = protocol
    app.cleanup_ctx.append(lifecycle)
    app.router.add_get("/ws", websocket)
    app.router.add_get("/api/state", state)
    app.router.add_get("/api/experiments", experiments)
    app.router.add_get("/api/planning-comparison", planning_comparison)
    app.router.add_get("/api/sorting-evaluation", sorting_evaluation_report)
    app.router.add_get("/api/sorting-evaluation/runs/{run_id}", sorting_evaluation_run)
    app.router.add_get("/", asset)
    app.router.add_get("/{name}", asset)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--target", choices=tuple(TARGETS), default="bowl")
    parser.add_argument("--scene", choices=("storage", "sorting"), default="storage")
    parser.add_argument("--sorting-layout", help="sorting 场景的预注册初始布局 ID")
    args = parser.parse_args()
    case = SceneCase(f"{args.target}.nominal", args.target, "nominal", 0, (0.0, 0.0), 0.0)
    if args.sorting_layout is not None and args.scene != "sorting":
        parser.error("--sorting-layout 仅适用于 --scene sorting")
    web.run_app(
        create_app(case, scene=args.scene, sorting_layout=args.sorting_layout),
        host="127.0.0.1", port=args.port, access_log=None,
    )


if __name__ == "__main__":
    main()
