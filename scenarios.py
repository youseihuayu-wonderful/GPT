"""Shared target geometry and deterministic paired evaluation scenes."""

from dataclasses import asdict, dataclass
import math
from numbers import Real
import random
from types import MappingProxyType


@dataclass(frozen=True)
class GraspTarget:
    name: str
    label: str
    grasp_offset_m: tuple[float, float, float]
    strategy: str

    def to_dict(self) -> dict:
        return asdict(self)


TARGETS = MappingProxyType({
    "bowl": GraspTarget(
        "bowl", "黑碗", (0.0, -0.047, 0.042),
        "107 mm 碗宽于 80 mm 夹爪，夹取近侧碗沿，一指在碗内、一指在碗外。",
    ),
    "bottle": GraspTarget(
        # Neck reaches 0.158 m; grasp near its top to clear the Panda palm.
        "bottle", "酒瓶", (0.0, 0.0, 0.145),
        "从上方接近直立酒瓶，双指夹住瓶颈后竖直抬升；不执行放置或倾倒。",
    ),
})


@dataclass(frozen=True)
class SceneCase:
    case_id: str
    target_object: str
    position_label: str
    seed: int
    offset_xy_m: tuple[float, float]
    yaw_rad: float

    def __post_init__(self) -> None:
        if not isinstance(self.target_object, str) or self.target_object not in TARGETS:
            raise ValueError("Unknown grasp target")
        if any(not isinstance(value, str) or not value.strip() or len(value) > 100
               for value in (self.case_id, self.position_label)):
            raise ValueError("Scene identifiers must be nonempty bounded strings")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("Scene seed must be an unsigned 32-bit integer")
        if not isinstance(self.offset_xy_m, tuple) or len(self.offset_xy_m) != 2:
            raise ValueError("Scene offset must be an immutable XY pair")
        values = (*self.offset_xy_m, self.yaw_rad)
        if any(isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
               for value in values):
            raise ValueError("Scene pose must be finite numeric values")
        if any(abs(value) > 0.08 for value in self.offset_xy_m) or abs(self.yaw_rad) > 0.2:
            raise ValueError("Scene perturbation exceeds the evaluation workspace")

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_CASE = SceneCase("bowl.nominal", "bowl", "nominal", 0, (0.0, 0.0), 0.0)
POSITION_OFFSETS = MappingProxyType({
    "nominal": (0.0, 0.0),
    "right": (0.04, 0.0),
    "forward": (0.0, 0.04),
})


def benchmark_cases(seeds: tuple[int, ...] = (101, 202, 303)) -> tuple[SceneCase, ...]:
    if (not seeds or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("Use distinct unsigned 32-bit scene seeds")
    cases = []
    for target in TARGETS:
        for label, position in POSITION_OFFSETS.items():
            for seed in seeds:
                rng = random.Random(f"{target}:{label}:{seed}")
                offset = tuple(value + rng.uniform(-0.003, 0.003) for value in position)
                cases.append(SceneCase(
                    f"{target}.{label}.{seed}", target, label, seed, offset,
                    rng.uniform(-0.06, 0.06),
                ))
    return tuple(cases)
