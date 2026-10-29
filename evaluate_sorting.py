"""Run the preregistered frozen sorting schedule once, retaining every outcome.

Only this CLI executes trials. The report/download helpers read bounded metadata;
opening the UI never creates a simulation, requests a plan, or resumes a run.
"""

import argparse
import asyncio
from contextlib import contextmanager
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile

from compare_sorting import SortingCase, require_matches_case, run_case
from sorting_contract import OBJECTS, validate_plan
from sorting_evaluation import DEFAULT_OUTPUT, DEFAULT_PROTOCOL, assert_frozen, load_protocol, make_simulation
from verify_skills import json_observations
from verify_sorting import independent_complete


_KIND = "frozen_sorting_evaluation"
_SLUG = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]*\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_GOALS = {"all_correct", "all_supported", "all_stable", "hand_empty", "home_reached"}
_MEASUREMENTS = ("execution_sim_s", "execution_wall_s", "gpt_latency_s", "wall_s",
                 "independent_final_hold_s", "controller_validation_hold_s")
# These assertions protect the experiment, rather than score task performance.
_INTEGRITY_ERRORS = (
    "Initial physical state/model mismatch", "Compiled model changed",
    "Physics parameters changed", "Equality activation changed",
    "Positive run applied external forces", "Planning wait applied external forces",
    "Physics step did not advance", "Starting a plan changed physics",
    "Physics became invalid or executed before approval",
)


def _case(case):
    return SortingCase(case["id"], tuple(case["destinations"][color]
                                       for color in ("red", "blue", "yellow")),
                       case["cups_first"], case["task"])


def _schedule(protocol):
    cases = {case["id"]: case for case in protocol["cases"]}
    return [{**entry, "layout_id": cases[entry["case_id"]]["layout_id"],
             "wording_id": cases[entry["case_id"]]["wording_id"]}
            for entry in protocol["schedule"]]


def _aggregates(report):
    rows = report["runs"]
    summary = []
    for method in ("fixed_rule", "gpt"):
        selected = [row for row in rows if row["method"] == method]
        scheduled = sum(item["method"] == method for item in report["schedule"])
        infra = sum(row["infrastructure_failure"] for row in selected)
        successes = sum(row["passed"] for row in selected)
        summary.append({"method": method, "scheduled": scheduled, "attempted": len(selected),
                        "successes": successes, "failures": len(selected) - successes - infra,
                        "pending": scheduled - len(selected), "infrastructure_failures": infra})
    by_id = {row["run_id"]: row for row in rows}
    paired = []
    for case_id in dict.fromkeys(item["case_id"] for item in report["schedule"]):
        entries = [item for item in report["schedule"] if item["case_id"] == case_id]
        identities = {item["method"]: item["run_id"] for item in entries}
        rule, gpt = (by_id.get(identities[method]) for method in ("fixed_rule", "gpt"))
        both_passed = rule is not None and gpt is not None and rule["passed"] and gpt["passed"]
        status = ("infrastructure_error" if any(row and row["infrastructure_failure"]
                                               for row in (rule, gpt)) else
                  "pending" if rule is None or gpt is None else
                  "succeeded" if both_passed else "failed")
        pair = {"case_id": case_id, "layout_id": entries[0]["layout_id"],
                "wording_id": entries[0]["wording_id"], "status": status,
                "fixed_rule_run_id": identities["fixed_rule"], "gpt_run_id": identities["gpt"]}
        for measurement in ("execution_sim_s", "execution_wall_s"):
            pair[measurement.removesuffix("_s") + "_delta_s"] = (
                gpt[measurement] - rule[measurement] if both_passed else None)
        paired.append(pair)
    report["summary"], report["paired"] = summary, paired
    report["passed"] = (report["status"] == "completed"
                        and len(rows) == report["expected_runs"]
                        and all(row["passed"] for row in rows) and not report["errors"])
    return report


def _cleanup_temporary(path, primary_error):
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except BaseException as exc:
        if primary_error is None:
            raise
        primary_error.add_note(f"Temporary evidence cleanup also failed: {exc!r}")


def _atomic_summary(directory, report):
    _aggregates(report)
    temporary = primary_error = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory,
                                         prefix=".summary-", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, directory / "summary.json")
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        _cleanup_temporary(temporary, primary_error)


@contextmanager
def _save_record(directory, run_id, record):
    destination = directory / "runs" / f"{run_id}.json.gz"
    temporary = primary_error = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".run-", delete=False) as raw:
            temporary = Path(raw.name)
            with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8") as stream:
                    json.dump(json_observations(record), stream, ensure_ascii=False, allow_nan=False,
                              separators=(",", ":"))
            raw.flush()
            os.fsync(raw.fileno())
        with temporary.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        # Publish exclusively, then expose its identity before fallible cleanup.
        os.link(temporary, destination)
        yield f"runs/{run_id}.json.gz", digest
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        _cleanup_temporary(temporary, primary_error)


def _error(record, stage, exc):
    record["passed"] = False
    record["status"] = "failed"
    record["failure_stage"] = stage
    record.setdefault("errors", []).append({"stage": stage, "type": type(exc).__name__,
                                             "message": str(exc)})


def _infrastructure_failure(record):
    if record.get("failure_stage") not in (None, "planning", "executing"):
        return True
    for error in record.get("errors", []):
        if error.get("stage") is not None:
            return True
        if error.get("type") not in ("AssertionError", "ValueError"):
            return True
        if error.get("type") == "ValueError" and record.get("failure_stage") != "planning":
            return True
        if any(message in error.get("message", "") for message in _INTEGRITY_ERRORS):
            return True
    return False


def _compact(entry, record):
    final = record.get("final_state") or {}
    sorting = final.get("sorting") or {}
    planner = record.get("planner") or {}
    infrastructure = _infrastructure_failure(record)
    return {**entry, "status": "infrastructure_error" if infrastructure else
            "succeeded" if record.get("passed") else "failed",
            "passed": record.get("passed", False) and not infrastructure,
            "infrastructure_failure": infrastructure, "errors": record.get("errors", []),
            "failure_stage": record.get("failure_stage"),
            "initial_pairing_sha256": (record.get("pairing") or {}).get("sha256"),
            "sorted_count": sorting.get("sorted_count"),
            "execution_sim_s": sorting.get("elapsed_s") if sorting.get("status") not in (None, "idle") else None,
            "execution_wall_s": record.get("execution_wall_s"),
            "gpt_latency_s": record.get("gpt_latency_s"), "wall_s": record.get("wall_s"),
            "model": planner.get("model"), "thread_id": planner.get("thread_id"),
            "approved_plan": record.get("approved_plan"),
            "physical_lift_order": record.get("physical_lift_order"),
            "physical_placement_order": record.get("physical_placement_order"),
            "independent_final_hold_s": record.get("independent_final_hold_s"),
            "independent_final_observation": record.get("independent_final_observation"),
            "grasp_evidence": record.get("grasp_evidence"),
            "controller_validation_hold_s": sorting.get("validation_hold_s"),
            "controller_status": sorting.get("status"),
            "final_goal_checks": sorting.get("goal_checks"),
            "collision_episodes": (final.get("collisions") or {}).get("episodes")}


def _success_evidence(row, case):
    if row["sorted_count"] != 6 or row.get("controller_status") != "succeeded":
        raise ValueError("Success lacks six sorted objects and controller completion")
    plan = validate_plan(row.get("approved_plan"))
    try:
        require_matches_case(plan, _case(case))
    except AssertionError as exc:
        raise ValueError(str(exc)) from exc
    order = [step.object_id for step in plan.steps]
    if row.get("physical_lift_order") != order or row.get("physical_placement_order") != order:
        raise ValueError("Success lacks independently observed approved handling order")
    for field in ("independent_final_hold_s", "controller_validation_hold_s"):
        if row.get(field) is None or row[field] < 5.0 - 1e-8:
            raise ValueError("Success lacks the full independent/controller final hold")
    goals = row.get("final_goal_checks")
    if not isinstance(goals, dict) or set(goals) != _GOALS or any(value is not True for value in goals.values()):
        raise ValueError("Success contradicts final physical goals")
    if row.get("collision_episodes") != 0:
        raise ValueError("Success has missing or nonzero collision evidence")
    evidence = row.get("grasp_evidence")
    if (not isinstance(evidence, dict) or set(evidence) != set(order)
            or any(facts.get("bilateral_seen") is not True or facts.get("lifted_while_gripped") is not True
                   for facts in evidence.values())):
        raise ValueError("Success lacks independent bilateral grasp/lift evidence")
    observation = row.get("independent_final_observation")
    try:
        facts = observation["objects"]
        if (len(facts) != 6 or {obj["id"] for obj in facts} != set(order)
                or any(obj["expected_destination"] != case["destinations"][obj["color"]]
                       or obj["color"] != next(item.color for item in OBJECTS if item.id == obj["id"])
                       for obj in facts)
                or not independent_complete(observation)):
            raise ValueError("Success contradicts independent final observation")
        for obj in facts:
            if any(obj[key] is not True for key in
                   ("correct_region", "slot_aligned", "upright", "supported", "stable", "detached")):
                raise ValueError("Independent goal facts must be booleans")
        if observation["hand_empty"] is not True:
            raise ValueError("Independent hand-empty fact must be true")
        for field in ("arm_home_error_rad", "arm_max_speed_rad_s"):
            value = observation[field]
            if not _nonnegative_number(value):
                raise ValueError("Invalid independent arm measurement")
    except (KeyError, TypeError, StopIteration) as exc:
        raise ValueError("Success lacks independent final observation") from exc
    for field in ("execution_sim_s", "execution_wall_s", "wall_s"):
        if row.get(field) is None:
            raise ValueError("Successful execution is missing measurements")


def _nonnegative_number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value) and value >= 0
    except OverflowError:
        return False


def _finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite report measurement")
    if isinstance(value, dict):
        for item in value.values():
            _finite(item)
    elif isinstance(value, list):
        for item in value:
            _finite(item)


def _validate_row(row, entry, case, model):
    if not isinstance(row, dict) or any(row.get(key) != value for key, value in entry.items()):
        raise ValueError("Run identity differs from the registered schedule")
    if type(row.get("passed")) is not bool or type(row.get("infrastructure_failure")) is not bool:
        raise ValueError("Run flags must be booleans")
    expected = "infrastructure_error" if row["infrastructure_failure"] else "succeeded" if row["passed"] else "failed"
    if row.get("status") != expected or row["passed"] and row["infrastructure_failure"]:
        raise ValueError("Contradictory run status")
    errors = row.get("errors")
    if not isinstance(errors, list) or any(not isinstance(error, dict) or
            not isinstance(error.get("type"), str) or not isinstance(error.get("message"), str) for error in errors):
        raise ValueError("Run errors must retain typed reasons")
    if (row["passed"] and (errors or row.get("failure_stage") is not None)
            or not row["passed"] and (not errors or not isinstance(row.get("failure_stage"), str))):
        raise ValueError("Contradictory success/failure evidence")
    if _infrastructure_failure(row) != row["infrastructure_failure"]:
        raise ValueError("Failure classification contradicts its stage/reason")
    for field in _MEASUREMENTS:
        value = row.get(field)
        if value is not None and not _nonnegative_number(value):
            raise ValueError(f"Invalid {field}")
    for field in ("sorted_count", "collision_episodes"):
        value = row.get(field)
        if value is not None and (type(value) is not int or value < 0 or field == "sorted_count" and value > 6):
            raise ValueError(f"Invalid {field}")
    pairing = row.get("initial_pairing_sha256")
    if pairing is None and not row["infrastructure_failure"] or pairing is not None and not _SHA.fullmatch(str(pairing)):
        raise ValueError("Missing/invalid initial pairing evidence")
    if row["method"] == "gpt":
        if row.get("model") != model and not (row["infrastructure_failure"] and
                (row.get("model") is None or any(error.get("stage") == "model_identity" for error in errors))):
            raise ValueError("Reported GPT model differs from protocol")
        if row["passed"] and (not isinstance(row.get("thread_id"), str) or not row["thread_id"]
                              or row.get("gpt_latency_s") is None):
            raise ValueError("Successful GPT run lacks provider identity/latency")
    elif any(row.get(field) is not None for field in ("model", "thread_id", "gpt_latency_s")):
        raise ValueError("Fixed-rule run has GPT metadata")
    if row["passed"]:
        _success_evidence(row, case)


def _artifact_path(directory, run_id):
    if not isinstance(run_id, str) or not _SLUG.fullmatch(run_id):
        raise ValueError("Unsafe run ID")
    directory = directory.resolve()
    runs = directory / "runs"
    path = runs / f"{run_id}.json.gz"
    if runs.is_symlink() or path.is_symlink() or path.resolve().parent != runs or not path.is_file():
        raise ValueError("Missing or unsafe run artifact")
    return path


def _small_json(path, maximum):
    if path.is_symlink():
        raise ValueError("Report metadata cannot be a symlink")
    with path.open("rb") as stream:
        raw = stream.read(maximum + 1)
    if len(raw) > maximum:
        raise ValueError("Report metadata exceeds its size bound")
    try:
        return json.loads(raw), raw
    except (ValueError, UnicodeError) as exc:
        raise ValueError("Invalid report JSON") from exc


def load_evaluation_report(directory: Path = DEFAULT_OUTPUT) -> dict:
    """Validate bounded stored metadata, never decompress traces or run physics."""
    directory = Path(directory)
    report, _ = _small_json(directory / "summary.json", 2_000_000)
    if not isinstance(report, dict) or report.get("kind") != _KIND or type(report.get("schema_version")) is not int or report["schema_version"] != 1:
        raise ValueError("Unknown evaluation report schema")
    try:
        _, protocol_bytes = _small_json(directory / "protocol.json", 1_000_000)
        protocol = load_protocol(directory / "protocol.json")
    except FileNotFoundError as exc:
        raise ValueError("Existing evaluation is missing its copied protocol") from exc
    try:
        for key in ("frozen_commit", "requested_model", "scope"):
            if report.get(key) != protocol[key]:
                raise ValueError(f"Report {key} differs from copied protocol")
        if report.get("protocol_sha256") != hashlib.sha256(protocol_bytes).hexdigest():
            raise ValueError("Copied protocol digest mismatch")
        schedule = _schedule(protocol)
        if report.get("schedule") != schedule or type(report.get("expected_runs")) is not int or report["expected_runs"] != len(schedule):
            raise ValueError("Report schedule/count differs from preregistration")
        if report.get("status") not in ("running", "completed", "aborted") or type(report.get("passed")) is not bool:
            raise ValueError("Invalid report status")
        if not isinstance(report.get("errors"), list) or any(not isinstance(error, str) for error in report["errors"]):
            raise ValueError("Invalid report errors")
        rows = report.get("runs")
        if not isinstance(rows, list) or len(rows) > len(schedule):
            raise ValueError("Invalid attempted run count")
        cases = {case["id"]: case for case in protocol["cases"]}
        pairing = {}
        for row, entry in zip(rows, schedule):
            _validate_row(row, entry, cases[entry["case_id"]], protocol["requested_model"])
            layout, digest = row["layout_id"], row.get("initial_pairing_sha256")
            if digest is not None:
                if layout in pairing and pairing[layout] != digest:
                    if not row["infrastructure_failure"] or not any(error.get("stage") == "pairing" for error in row["errors"]):
                        raise ValueError("Same-layout initial states are not exactly paired")
                else:
                    pairing[layout] = digest
            if row.get("record_path") is None:
                if (row.get("record_sha256") is not None or not row["infrastructure_failure"]
                        or row["failure_stage"] != "persistence"):
                    raise ValueError("Missing artifact without explicit persistence failure")
            else:
                if row["record_path"] != f"runs/{entry['run_id']}.json.gz" or not _SHA.fullmatch(str(row.get("record_sha256", ""))):
                    raise ValueError("Invalid canonical artifact reference")
                _artifact_path(directory, entry["run_id"])
        if any(row["infrastructure_failure"] for row in rows[:-1]):
            raise ValueError("Execution continued after infrastructure failure")
        if report["status"] == "completed" and (len(rows) != len(schedule) or report["errors"] or any(row["infrastructure_failure"] for row in rows)):
            raise ValueError("Completion conceals pending work or infrastructure failure")
        if report["status"] == "aborted" and not report["errors"]:
            raise ValueError("Aborted evaluation lacks its reason")
        if report["status"] != "aborted" and (report["errors"] or any(row["infrastructure_failure"] for row in rows)):
            raise ValueError("Infrastructure failure must be reported as aborted")
        _finite(report)
        claimed = report["passed"]
        _aggregates(report)
        if claimed != report["passed"]:
            raise ValueError("Contradictory overall success evidence")
        return report
    except (KeyError, TypeError, AttributeError, OverflowError) as exc:
        raise ValueError("Malformed evaluation report") from exc


def evaluation_run_path(directory: Path, run_id: str) -> Path:
    """Return only a registered, persisted gzip record from a valid report."""
    if not isinstance(run_id, str) or not _SLUG.fullmatch(run_id):
        raise ValueError("Unsafe run ID")
    report = load_evaluation_report(directory)
    if run_id not in {row["run_id"] for row in report["runs"] if row.get("record_path") is not None}:
        raise ValueError("Unknown, pending, or unsaved run ID")
    return _artifact_path(Path(directory), run_id)


async def run_evaluation(output: Path, protocol_path: Path = DEFAULT_PROTOCOL) -> dict:
    """Reserve a new directory and attempt the registered schedule without retry."""
    protocol_path, output = Path(protocol_path), Path(output)
    protocol = load_protocol(protocol_path)
    protocol_bytes = protocol_path.read_bytes()
    output.mkdir(parents=True, exist_ok=False)
    with (output / "protocol.json").open("xb") as stream:
        stream.write(protocol_bytes)
    # Validate the exact bytes retained, not a potentially changed source path.
    protocol = load_protocol(output / "protocol.json")
    (output / "runs").mkdir()
    report = {"kind": _KIND, "schema_version": 1, "status": "running", "passed": False,
              **{key: protocol[key] for key in ("frozen_commit", "requested_model", "scope")},
              "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
              "expected_runs": len(protocol["schedule"]), "schedule": _schedule(protocol),
              "runs": [], "summary": [], "paired": [], "errors": []}
    _atomic_summary(output, report)
    cases = {case["id"]: case for case in protocol["cases"]}
    pairing = {}
    try:
        for entry in report["schedule"]:
            await asyncio.sleep(0)
            # A preflight fault is not an attempted method execution.
            assert_frozen(protocol)
            case = cases[entry["case_id"]]
            record = {**entry, "passed": False, "errors": []}
            interrupted = None

            def factory():
                return make_simulation(protocol, entry["layout_id"])

            try:
                await run_case(_case(case), entry["method"], record,
                               pairing.get(entry["layout_id"]), simulation_factory=factory)
                await asyncio.sleep(0)
            except BaseException as exc:
                interrupted = exc
                _error(record, "interrupted" if not isinstance(exc, Exception) else "runner", exc)
            try:
                assert_frozen(protocol)
            except BaseException as exc:
                _error(record, "frozen_inputs", exc)
                if not isinstance(exc, Exception) and interrupted is None:
                    interrupted = exc
            digest = (record.get("pairing") or {}).get("sha256")
            if digest is not None and entry["layout_id"] in pairing and digest != pairing[entry["layout_id"]]:
                _error(record, "pairing", ValueError("Same-layout initial physical state differs across methods/wordings"))
            elif digest is not None:
                pairing[entry["layout_id"]] = digest
            if entry["method"] == "gpt" and record.get("planner") is not None:
                if record["planner"].get("model") != protocol["requested_model"]:
                    _error(record, "model_identity", ValueError("Actual GPT model differs from preregistered model"))
            if not record.get("passed") and not record.get("failure_stage"):
                record["failure_stage"] = next((error.get("stage") for error in record["errors"] if error.get("stage")), "runner")
            row = _compact(entry, record)
            try:
                _validate_row(row, entry, case, protocol["requested_model"])
                _finite(row)
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                _error(record, "evidence", exc)
                row = _compact(entry, record)
                # Invalid measurements remain in the full record, never become zeros.
                for field in _MEASUREMENTS:
                    value = row.get(field)
                    if value is not None and not _nonnegative_number(value):
                        row[field] = None
                for field in ("sorted_count", "collision_episodes"):
                    value = row.get(field)
                    if value is not None and (type(value) is not int or value < 0
                                              or field == "sorted_count" and value > 6):
                        row[field] = None
                if not _SHA.fullmatch(str(row.get("initial_pairing_sha256"))):
                    row["initial_pairing_sha256"] = None
                if row["method"] == "fixed_rule":
                    for field in ("model", "thread_id", "gpt_latency_s"):
                        row[field] = None
                row = json_observations(row)
            record["infrastructure_failure"] = row["infrastructure_failure"]
            record["outcome_status"] = row["status"]
            row_saved = False
            try:
                with _save_record(output, entry["run_id"], record) as artifact:
                    row["record_path"], row["record_sha256"] = artifact
                report["runs"].append(row)
                row_saved = True
                if row["infrastructure_failure"]:
                    report["status"] = "aborted"
                    report["errors"].append(f"{entry['run_id']}: infrastructure failure at {row['failure_stage']}")
                _atomic_summary(output, report)
            except BaseException as saving_error:
                if not row_saved:
                    row.update(status="infrastructure_error", passed=False,
                               infrastructure_failure=True, failure_stage="persistence")
                    row.setdefault("record_path", None)
                    row.setdefault("record_sha256", None)
                    row["errors"].append({"stage": "persistence", "type": type(saving_error).__name__,
                                          "message": str(saving_error)})
                    report["runs"].append(row)
                report["status"] = "aborted"
                report["errors"].append(f"{entry['run_id']}: evidence persistence failed: {saving_error}")
                if interrupted is not None:
                    interrupted.add_note(f"Evidence persistence also failed: {saving_error!r}")
                    raise interrupted from saving_error
                if record.get("errors"):
                    saving_error.add_note(f"Primary trial evidence: {record['errors']!r}")
                    report["errors"].append(f"{entry['run_id']}: unsaved trial errors: {record['errors']!r}")
                raise
            if interrupted is not None:
                raise interrupted
            await asyncio.sleep(0)
            if row["infrastructure_failure"]:
                return report
        report["status"] = "completed"
        _atomic_summary(output, report)
        await asyncio.sleep(0)
        return report
    except BaseException as exc:
        report["status"] = "aborted"
        report["errors"].append(f"{type(exc).__name__}: {exc}")
        try:
            _atomic_summary(output, report)
        except BaseException as saving_error:
            exc.add_note(f"Terminal summary persistence also failed: {saving_error!r}")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL, type=Path)
    args = parser.parse_args()
    report = asyncio.run(run_evaluation(args.output, args.protocol))
    print(json.dumps({key: report[key] for key in ("status", "passed", "summary", "errors")}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
