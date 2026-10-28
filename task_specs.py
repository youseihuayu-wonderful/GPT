"""Executable task contracts shared by planning and physical controllers."""

from dataclasses import asdict, dataclass
from types import MappingProxyType

from scenarios import TARGETS


GRASP_PHASES = ("prepare", "approach", "descend", "close", "lift", "hold")
STORAGE_PHASES = GRASP_PHASES + (
    "transfer", "place", "release", "withdraw", "push_approach", "push_drawer",
    "push_withdraw", "return_home", "verify_store",
)


@dataclass(frozen=True)
class TaskSpec:
    kind: str
    label: str
    target_objects: tuple[str, ...]
    phases: tuple[str, ...]
    watchdog_s: float
    minimum_opening_m: float

    def to_dict(self) -> dict:
        return asdict(self)


TASKS = MappingProxyType({
    "grasp": TaskSpec("grasp", "抓取并保持", tuple(TARGETS), GRASP_PHASES, 20.0, 0.02),
    "store_bowl": TaskSpec("store_bowl", "黑碗入抽屉、关抽屉并归位", ("bowl",), STORAGE_PHASES, 90.0, 0.06),
})
