"""Registered evaluation-only initial conditions; frozen skills remain unchanged."""

import ast
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re

import mujoco
import numpy as np

from planner import PlannerConfig, PlannerError
from simulation import ROOT
from sorting import SortingSimulation
from sorting_contract import OBJECTS


DEFAULT_PROTOCOL = ROOT / "sorting_evaluation_protocol.json"
DEFAULT_OUTPUT = ROOT / "results" / "sorting_frozen_evaluation"
CORE_FILES = frozenset({
    "collisions.py", "grasp.py", "simulation.py", "scenarios.py", "task_specs.py",
    "storage.py", "planning.py", "planner.py", "sorting.py", "sorting_contract.py",
    "sorting_planning.py", "verify_skills.py", "verify_sorting.py",
    "sorting_scene.xml", "scene.xml", "pyproject.toml", "uv.lock",
})
BASELINE_FUNCTIONS = frozenset({"fixed_rule_plan", "require_matches_case"})
OBJECT_IDS = frozenset(obj.id for obj in OBJECTS)
METHODS = ("fixed_rule", "gpt")
WORDINGS = frozenset({"direct", "paraphrase"})
_SLUG = re.compile(r"[a-z0-9][a-z0-9_-]{0,99}\Z")
_HASH = re.compile(r"[0-9a-f]{64}\Z")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _fields(value, names, context):
    _require(type(value) is dict and set(value) == set(names), f"Invalid {context} fields")


def _text(value):
    return type(value) is str and bool(value.strip())


def _slug(value):
    return type(value) is str and _SLUG.fullmatch(value) is not None


def _finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, f"Duplicate protocol key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"Non-finite JSON number: {value}")


def _relative_file(value):
    if not _text(value) or "\\" in value:
        return False
    path = Path(value)
    return not path.is_absolute() and value == path.as_posix() and ".." not in path.parts


def validate_protocol(protocol):
    """Validate identities, paired goals and the entire predeclared schedule."""
    _fields(protocol, {
        "kind", "schema_version", "frozen_commit", "requested_model", "mujoco_version",
        "frozen_files_sha256", "baseline_functions_sha256", "methods", "repetitions",
        "scope", "setup", "layouts", "cases", "schedule",
    }, "protocol")
    _require(protocol["kind"] == "frozen_sorting_evaluation_protocol", "Unknown protocol kind")
    _require(type(protocol["schema_version"]) is int and protocol["schema_version"] == 1,
             "Unknown protocol version")
    _require(type(protocol["frozen_commit"]) is str
             and re.fullmatch(r"[0-9a-f]{40}", protocol["frozen_commit"]), "Invalid frozen commit")
    model = protocol["requested_model"]
    try:
        _require(_text(model) and PlannerConfig(None, model).model == model, "Invalid registered model")
    except PlannerError as exc:
        raise ValueError("Invalid registered model") from exc
    _require(_text(protocol["mujoco_version"]) and _text(protocol["scope"]), "Missing protocol scope/version")
    _require(protocol["methods"] == list(METHODS), "Evaluation requires fixed_rule and gpt")
    _require(type(protocol["repetitions"]) is int and protocol["repetitions"] == 1,
             "This protocol permits exactly one attempt per cell")
    hashes = protocol["frozen_files_sha256"]
    _require(type(hashes) is dict and CORE_FILES <= set(hashes), "Frozen source inventory is incomplete")
    _require(all(_relative_file(path) and type(digest) is str and _HASH.fullmatch(digest)
                 for path, digest in hashes.items()), "Invalid frozen file path/hash")
    functions = protocol["baseline_functions_sha256"]
    _require(type(functions) is dict and set(functions) == BASELINE_FUNCTIONS
             and all(type(value) is str and _HASH.fullmatch(value) for value in functions.values()),
             "Invalid frozen baseline policy hashes")
    setup = protocol["setup"]
    _fields(setup, {"settle_s", "initialization", "timing"}, "setup")
    settle = setup["settle_s"]
    _require(_finite(settle) and settle > 0 and _finite(settle / 0.002)
             and math.isclose(round(settle / 0.002) * 0.002, settle, rel_tol=0, abs_tol=1e-9),
             "Settling must be a positive multiple of the frozen 2 ms timestep")
    _require(_text(setup["initialization"]) and _text(setup["timing"]), "Missing setup/timing definition")
    layouts = protocol["layouts"]
    _require(type(layouts) is list and bool(layouts), "Missing evaluation layouts")
    layout_ids = set()
    for layout in layouts:
        _fields(layout, {"id", "description", "poses"}, "layout")
        _require(_slug(layout["id"]) and layout["id"] not in layout_ids, "Invalid/duplicate layout ID")
        layout_ids.add(layout["id"])
        _require(_text(layout["description"]), "Missing layout description")
        poses = layout["poses"]
        _require(type(poses) is dict and set(poses) == OBJECT_IDS, "Layout must initialize exactly six known objects")
        for pose in poses.values():
            _fields(pose, {"x_m", "y_m", "yaw_rad"}, "initial pose")
            _require(all(_finite(value) for value in pose.values()), "Initial pose must contain finite numbers")
    cases = protocol["cases"]
    _require(type(cases) is list and len(cases) == len(layouts) * len(WORDINGS), "Missing wording cases")
    case_ids, wording_cells, goals = set(), set(), {}
    for case in cases:
        _fields(case, {"id", "layout_id", "wording_id", "destinations", "cups_first", "task"}, "case")
        _require(_slug(case["id"]) and case["id"] not in case_ids, "Invalid/duplicate case ID")
        case_ids.add(case["id"])
        _require(type(case["layout_id"]) is str and case["layout_id"] in layout_ids, "Unknown case layout")
        _require(type(case["wording_id"]) is str and case["wording_id"] in WORDINGS, "Unknown task wording")
        cell = (case["layout_id"], case["wording_id"])
        _require(cell not in wording_cells, "Duplicate layout/wording cell")
        wording_cells.add(cell)
        destinations = case["destinations"]
        _require(type(destinations) is dict and set(destinations) == {"red", "blue", "yellow"}
                 and all(type(value) is str for value in destinations.values())
                 and set(destinations.values()) == set("ABC"), "Invalid independent color objective")
        _require(type(case["cups_first"]) is bool, "Priority must be boolean")
        _require(_text(case["task"]) and len(case["task"]) <= 1000, "Task must contain 1–1000 characters")
        goal = (tuple(destinations[color] for color in ("red", "blue", "yellow")), case["cups_first"])
        _require(case["layout_id"] not in goals or goals[case["layout_id"]] == goal,
                 "Equivalent wordings must have identical independently specified objectives")
        goals[case["layout_id"]] = goal
    _require(wording_cells == {(layout, wording) for layout in layout_ids for wording in WORDINGS},
             "Incomplete layout/wording matrix")
    schedule = protocol["schedule"]
    _require(type(schedule) is list and len(schedule) == len(cases) * len(METHODS), "Incomplete run schedule")
    run_ids, scheduled_cells = set(), set()
    for run in schedule:
        _fields(run, {"run_id", "case_id", "method"}, "schedule entry")
        _require(_slug(run["run_id"]) and run["run_id"] not in run_ids, "Invalid/duplicate run ID")
        run_ids.add(run["run_id"])
        _require(type(run["case_id"]) is str and run["case_id"] in case_ids, "Unknown scheduled case")
        _require(type(run["method"]) is str and run["method"] in METHODS, "Unknown scheduled method")
        cell = (run["case_id"], run["method"])
        _require(cell not in scheduled_cells, "Duplicate case/method execution")
        scheduled_cells.add(cell)
    _require(scheduled_cells == {(case, method) for case in case_ids for method in METHODS},
             "Incomplete case/method schedule")


def load_protocol(path=DEFAULT_PROTOCOL):
    with Path(path).open(encoding="utf-8") as stream:
        protocol = json.load(stream, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    validate_protocol(protocol)
    return protocol


def assert_frozen(protocol, root=ROOT):
    """Reject source, asset, baseline-policy, model or engine drift before scoring."""
    validate_protocol(protocol)
    root = Path(root).resolve()
    inventory = set(CORE_FILES)
    inventory.update(path.relative_to(root).as_posix() for path in (root / "vendor").rglob("*")
                     if path.is_file() and not any(part.startswith(".") for part in path.relative_to(root).parts))
    _require(inventory == set(protocol["frozen_files_sha256"]), "Frozen source/asset inventory changed")
    for relative, expected in protocol["frozen_files_sha256"].items():
        path = (root / relative).resolve()
        _require(path.is_relative_to(root), f"Frozen file escapes project: {relative}")
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        _require(actual == expected, f"Frozen file changed: {relative}")
    source = (root / "compare_sorting.py").read_text(encoding="utf-8")
    functions = {node.name: hashlib.sha256(ast.get_source_segment(source, node).encode()).hexdigest()
                 for node in ast.parse(source).body
                 if isinstance(node, ast.FunctionDef) and node.name in BASELINE_FUNCTIONS}
    _require(functions == protocol["baseline_functions_sha256"], "Frozen fixed-rule/authorization policy changed")
    _require(mujoco.__version__ == protocol["mujoco_version"], "Frozen MuJoCo version changed")
    _require(PlannerConfig.load(root).model == protocol["requested_model"], "Registered model configuration changed")


@dataclass(frozen=True, slots=True)
class _InitialPose:
    object_id: str
    x_m: float
    y_m: float
    yaw_rad: float


class _EvaluationSimulation(SortingSimulation):
    """Only explicit reset changes initial poses; inherited control stays frozen."""

    def __init__(self, layout_id, poses, settle_s):
        self.evaluation_layout = layout_id
        self._initial_poses = poses
        self._settle_s = settle_s
        super().__init__()

    def reset(self):
        super().reset()
        for pose in self._initial_poses:
            joint = self.model.joint(f"{pose.object_id}_free")
            address, dof = int(joint.qposadr[0]), int(joint.dofadr[0])
            self.data.qpos[address:address + 7] = self.model.qpos0[address:address + 7]
            self.data.qpos[address:address + 2] = (pose.x_m, pose.y_m)
            yaw = np.array([math.cos(pose.yaw_rad / 2), 0.0, 0.0, math.sin(pose.yaw_rad / 2)])
            mujoco.mju_mulQuat(self.data.qpos[address + 3:address + 7], yaw,
                              self.model.qpos0[address + 3:address + 7])
            self.data.qvel[dof:dof + 6] = 0.0
            self.data.qacc_warmstart[dof:dof + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.step(self._settle_s)
        self.skill.reset()
        self.collisions.reset()


def make_simulation(protocol, layout_id):
    validate_protocol(protocol)
    _require(type(layout_id) is str, "Evaluation layout ID must be a string")
    layout = next((item for item in protocol["layouts"] if item["id"] == layout_id), None)
    _require(layout is not None, f"Unknown evaluation layout: {layout_id}")
    assert_frozen(protocol)
    poses = tuple(_InitialPose(obj.id, **layout["poses"][obj.id]) for obj in OBJECTS)
    return _EvaluationSimulation(layout_id, poses, protocol["setup"]["settle_s"])
