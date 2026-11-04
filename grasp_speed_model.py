"""Level-1 pre-execution random forest and explicit threshold speed policy.

Probabilities are uncalibrated random-forest estimates, not safety guarantees.
Only completed physical collector rows belong in experiment training data.
"""

import math
from numbers import Real
from types import MappingProxyType

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from scenarios import SceneCase


SPEEDS_M_S = (.04, .10, .20)
FEATURE_NAMES = (
    "target_is_bowl", "target_is_bottle", "offset_x_m", "offset_y_m",
    "yaw_rad", "descent_speed_m_s",
)
FOREST_SETTINGS = MappingProxyType({
    "n_estimators": 128,
    "criterion": "gini",
    "max_depth": 6,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": None,
    "bootstrap": True,
    "class_weight": None,
    "n_jobs": 1,
})


def _finite_number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _speed(value) -> float:
    speed = _finite_number(value, "speed_m_s")
    if speed not in SPEEDS_M_S:
        raise ValueError(f"speed_m_s must be one of {SPEEDS_M_S}")
    return speed


def _features(case: dict, speed) -> list[float]:
    if not isinstance(case, dict):
        raise ValueError("case must be a collector case dictionary")
    target = case.get("target_object")
    if not isinstance(target, str) or target not in ("bowl", "bottle"):
        raise ValueError("target_object must be bowl or bottle")
    offset = case.get("offset_xy_m")
    if not isinstance(offset, (list, tuple)) or len(offset) != 2:
        raise ValueError("offset_xy_m must be an XY pair")
    x = _finite_number(offset[0], "offset_x_m")
    y = _finite_number(offset[1], "offset_y_m")
    yaw = _finite_number(case.get("yaw_rad"), "yaw_rad")
    if abs(x) > .08 or abs(y) > .08 or abs(yaw) > .2:
        raise ValueError("Scene perturbation exceeds the evaluation workspace")
    return [float(target == "bowl"), float(target == "bottle"), x, y, yaw, _speed(speed)]


def train_classifier(rows: list[dict], *, seed: int = 2026) -> RandomForestClassifier:
    """Fit both binary classes without resampling or outcome-derived features.

    ``feature_names_`` records the fixed positional encoding; ``get_params()``
    records all forest settings, including the supplied random seed. Invalid
    rows are rejected, never silently filtered into a different training set.
    """
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an unsigned 32-bit integer")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Training requires completed rows with both binary classes")
    features, labels = [], []
    for index, row in enumerate(rows):
        if (not isinstance(row, dict) or row.get("trial_status") != "completed"
                or type(row.get("success")) is not bool):
            raise ValueError(f"Training row {index} must be completed with boolean success")
        parameters = row.get("parameters")
        if not isinstance(parameters, dict):
            raise ValueError(f"Training row {index} needs a parameters dictionary")
        features.append(_features(row.get("case"), parameters.get("descent_speed_m_s")))
        labels.append(row["success"])
    if len(set(labels)) != 2:
        raise ValueError("Training requires both binary classes (success and failure)")
    model = RandomForestClassifier(random_state=seed, **FOREST_SETTINGS)
    model.fit(np.asarray(features, dtype=float), np.asarray(labels, dtype=bool))
    model.feature_names_ = FEATURE_NAMES
    return model


def predict_candidates(model, case: SceneCase) -> list[dict]:
    """Predict each of the three commanded speeds in fixed ascending order."""
    if not isinstance(case, SceneCase):
        raise ValueError("case must be a SceneCase")
    if tuple(getattr(model, "feature_names_", ())) != FEATURE_NAMES:
        raise ValueError("Model feature encoding does not match FEATURE_NAMES")
    classes = np.asarray(getattr(model, "classes_", []))
    if classes.shape != (2,) or set(classes.tolist()) != {False, True}:
        raise ValueError("Model must contain both binary classes")
    features = [_features(case.to_dict(), speed) for speed in SPEEDS_M_S]
    probabilities = np.asarray(model.predict_proba(features), dtype=float)
    if (probabilities.shape != (3, 2) or not np.all(np.isfinite(probabilities))
            or np.any(probabilities < 0) or np.any(probabilities > 1)
            or not np.allclose(probabilities.sum(axis=1), 1, rtol=0, atol=1e-12)):
        raise ValueError("Model returned invalid binary probabilities")
    success_column = classes.tolist().index(True)
    return [
        {"speed_m_s": speed, "predicted_success_probability": float(probabilities[i, success_column])}
        for i, speed in enumerate(SPEEDS_M_S)
    ]


def select_speed(candidates: list[dict], *, threshold: float = .9) -> dict:
    """Choose fastest qualifying speed, else best probability with slower ties."""
    threshold = _finite_number(threshold, "threshold")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must lie in [0, 1]")
    if not isinstance(candidates, list) or len(candidates) != len(SPEEDS_M_S):
        raise ValueError("Exactly three speed candidates are required")
    values = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("Each speed candidate must be a dictionary")
        speed = _speed(candidate.get("speed_m_s"))
        probability = _finite_number(candidate.get("predicted_success_probability"), "probability")
        if not 0 <= probability <= 1:
            raise ValueError("probability must lie in [0, 1]")
        if speed in values:
            raise ValueError("Duplicate speed candidate")
        values[speed] = probability
    qualifying = [speed for speed, probability in values.items() if probability >= threshold]
    if qualifying:
        selected = max(qualifying)
        rule = "fastest_above_threshold"
    else:
        selected = max(values, key=lambda speed: (values[speed], -speed))
        rule = "highest_probability_slower_tiebreak"
    return {
        "selected_speed_m_s": selected,
        "selected_success_probability": values[selected],
        "threshold": threshold,
        "threshold_met": bool(qualifying),
        "selection_rule": rule,
    }
