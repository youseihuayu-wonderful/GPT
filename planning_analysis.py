"""Descriptive repeated-run accounting; paired deltas are not population estimates."""

from collections import Counter
import math
from statistics import mean, stdev


_PENDING = ("setting_up", "running")
_FINGERPRINTS = (
    "initial_state_fingerprint", "initial_observation_fingerprint", "initial_model_fingerprint",
)


def _measured(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0)


def _distribution(values):
    return {
        "n": len(values), "mean": mean(values) if values else None,
        "sd": stdev(values) if len(values) > 1 else None,
        "min": min(values) if values else None, "max": max(values) if values else None,
    }


def _valid(row):
    return not row.get("infrastructure_errors") and row["status"] not in _PENDING


def _success(row):
    return _valid(row) and row["expected_outcome"] == "success" and row.get("success") is True


def _safety_pass(row):
    return (_valid(row) and row["expected_outcome"] == "safe_stop"
            and row.get("safety_pass") is True)


def analyze_runs(runs: list[dict], methods: tuple[str, ...]) -> dict:
    """Group observed rows without imputing missing measurements or planned runs.

    Legacy rows imply repetition one and the original bowl storage goal. Conflicting
    case metadata, duplicate identities, and unverified complete pairs are errors.
    Durations of failed attempts remain in all-attempt distributions; speed deltas
    use only the same repetitions where both methods succeeded and were timed.
    """
    if len(set(methods)) != len(methods):
        raise ValueError("Duplicate methods")
    cases = {}
    indexed = {}
    groups = {}
    repetitions = {}
    for row in runs:
        case_id, method = row["case_id"], row["method"]
        if method not in methods:
            raise ValueError(f"Unexpected method: {method}")
        repetition = row.get("repetition", 1)
        if not isinstance(repetition, int) or isinstance(repetition, bool) or repetition < 1:
            raise ValueError(f"Invalid repetition: {repetition!r}")
        identity = (case_id, repetition, method)
        if identity in indexed:
            raise ValueError(f"Duplicate run: {identity}")
        metadata = {
            "case_id": case_id, "goal": row.get("goal", "store_and_home"),
            "target_object": row.get("target_object", "bowl"),
            "expected_outcome": row["expected_outcome"],
        }
        if metadata["expected_outcome"] not in ("success", "safe_stop"):
            raise ValueError(f"Invalid expected outcome for {case_id}")
        if case_id in cases and cases[case_id] != metadata:
            raise ValueError(f"Conflicting case metadata: {case_id}")
        cases[case_id] = metadata
        indexed[identity] = row
        groups.setdefault((case_id, method), []).append(row)
        repetitions.setdefault(case_id, set()).add(repetition)

    by_case = []
    paired = []
    for case_id, metadata in cases.items():
        for method in methods:
            rows = groups.get((case_id, method), [])
            valid = [row for row in rows if _valid(row)]
            successes = [row for row in valid if _success(row)]
            collisions = [row["collisions"]["episodes"] for row in rows
                          if isinstance(row.get("collisions"), dict)
                          and _measured(row["collisions"].get("episodes"))]
            reasons = Counter(row["failure_reason"] for row in rows
                              if isinstance(row.get("failure_reason"), str)
                              and row["failure_reason"] and not _safety_pass(row))
            by_case.append({
                **metadata, "method": method, "attempts": len(valid),
                "successes": len(successes),
                "safety_passes": sum(_safety_pass(row) for row in valid),
                "infrastructure_failures": sum(bool(row.get("infrastructure_errors")) for row in rows),
                "pending_runs": sum(not row.get("infrastructure_errors")
                                    and row["status"] in _PENDING for row in rows),
                "collision_runs": sum(value > 0 for value in collisions) if collisions else None,
                "wall_s": _distribution([row["end_to_end_wall_s"] for row in valid
                                         if _measured(row.get("end_to_end_wall_s"))]),
                "success_wall_s": _distribution([row["end_to_end_wall_s"] for row in successes
                                                 if _measured(row.get("end_to_end_wall_s"))]),
                "request_latency_s": _distribution([
                    item["latency_s"] for row in rows if method == "gpt"
                    for item in row.get("history", []) if _measured(item.get("latency_s"))
                ]),
                "failure_reasons": [{"reason": reason, "count": reasons[reason]}
                                    for reason in sorted(reasons)],
            })
        if metadata["expected_outcome"] != "success" or not {"rule", "gpt"}.issubset(methods):
            continue
        pair = {key: metadata[key] for key in ("case_id", "goal", "target_object")}
        pair.update(complete_pairs=0, matched_success_pairs=0, rule_only_successes=0,
                    gpt_only_successes=0, both_failed=0, excluded_pairs=0)
        deltas = []
        for repetition in sorted(repetitions[case_id]):
            rule = indexed.get((case_id, repetition, "rule"))
            gpt = indexed.get((case_id, repetition, "gpt"))
            if rule is None or gpt is None or not _valid(rule) or not _valid(gpt):
                pair["excluded_pairs"] += 1
                continue
            for field in _FINGERPRINTS:
                if not rule.get(field) or not gpt.get(field) or rule[field] != gpt[field]:
                    raise ValueError(f"Unverified pair {case_id}/{repetition}: {field}")
            pair["complete_pairs"] += 1
            rule_success, gpt_success = _success(rule), _success(gpt)
            if rule_success and gpt_success:
                pair["matched_success_pairs"] += 1
                if (_measured(rule.get("end_to_end_wall_s"))
                        and _measured(gpt.get("end_to_end_wall_s"))):
                    deltas.append(gpt["end_to_end_wall_s"] - rule["end_to_end_wall_s"])
            elif rule_success:
                pair["rule_only_successes"] += 1
            elif gpt_success:
                pair["gpt_only_successes"] += 1
            else:
                pair["both_failed"] += 1
        pair["delta_wall_s"] = _distribution(deltas)
        paired.append(pair)
    return {"by_case": by_case, "paired": paired}
