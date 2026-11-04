"""Test-only working-tree snapshots; never rewrite the registered experiment."""

import ast
import hashlib
from pathlib import Path

import mujoco

from planner import PlannerConfig
from sorting_evaluation import load_protocol


def current_protocol():
    """Keep the declared layouts, but exercise freeze guards against today's code."""
    root = Path(__file__).resolve().parents[1]
    protocol = load_protocol()
    protocol["frozen_commit"] = "0" * 40  # Explicitly not a historical commit or experiment.
    protocol["scope"] = "Temporary unit fixture; not experimental evidence"
    protocol["mujoco_version"] = mujoco.__version__
    protocol["requested_model"] = PlannerConfig.load(root).model
    for relative in protocol["frozen_files_sha256"]:
        with (root / relative).open("rb") as stream:
            protocol["frozen_files_sha256"][relative] = hashlib.file_digest(stream, "sha256").hexdigest()
    source = (root / "compare_sorting.py").read_text()
    protocol["baseline_functions_sha256"] = {
        node.name: hashlib.sha256(ast.get_source_segment(source, node).encode()).hexdigest()
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name in protocol["baseline_functions_sha256"]
    }
    return protocol
