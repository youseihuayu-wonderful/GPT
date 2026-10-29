"""Executable task contracts shared by planning and physical controllers."""

from dataclasses import asdict, dataclass
from types import MappingProxyType

from scenarios import TARGETS


GRASP_PHASES = ("prepare", "approach", "descend", "close", "lift", "hold")
PLACE_PHASES = ("transfer", "place", "release")
WITHDRAW_PHASES = ("withdraw",)
CLOSE_PHASES = ("push_approach", "push_drawer", "push_withdraw")
HOME_PHASES = ("return_home",)
STORAGE_PHASES = GRASP_PHASES + PLACE_PHASES + WITHDRAW_PHASES + CLOSE_PHASES + HOME_PHASES + ("verify_store",)


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


@dataclass(frozen=True)
class SkillSpec:
    name: str
    label: str
    target_objects: tuple[str, ...]
    phases: tuple[str, ...]
    watchdog_s: float
    accepts_grasp_parameters: bool
    preconditions: tuple[str, ...]
    postconditions: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


SKILLS = MappingProxyType({
    "grasp_and_hold": SkillSpec(
        "grasp_and_hold", "抓取并保持", tuple(TARGETS), GRASP_PHASES, 20.0, True,
        ("夹爪空闲", "目标在桌面稳定且位于支持的工位"),
        ("双侧真实夹持", "离桌至少 100 mm", "稳定保持至少 1 s"),
    ),
    "place_and_release": SkillSpec(
        "place_and_release", "放置并松开", ("bowl",), PLACE_PHASES, 30.0, False,
        ("已稳定持有黑碗", "下层抽屉全开", "持物碰撞几何适合已验证的放置通道"),
        ("黑碗完整入屉", "底板真实支撑", "夹爪脱离且碗稳定"),
    ),
    "withdraw_from_drawer": SkillSpec(
        "withdraw_from_drawer", "撤出夹爪", ("bowl",), WITHDRAW_PHASES, 15.0, False,
        ("黑碗已完整入屉并稳定支撑", "夹爪已松开", "仍在支持的放置交接位姿"),
        ("手部离开抽屉运动区域", "黑碗保持支撑和静稳"),
    ),
    "close_drawer": SkillSpec(
        "close_drawer", "关抽屉并撤离", tuple(TARGETS), CLOSE_PHASES, 45.0, False,
        ("抽屉全开且目标物体稳定支撑", "空夹爪已撤离并处于高位"),
        ("真实接触推动抽屉", "抽屉关闭且停止", "手部撤离"),
    ),
    "return_home": SkillSpec(
        "return_home", "机械臂归位", tuple(TARGETS), HOME_PHASES, 18.0, False,
        ("空夹爪已离开抽屉", "机械臂处于已验证的高位回程区域"),
        ("关节位姿到达 home", "关节速度低", "手部仍为空且已撤离"),
    ),
})

TASK_SKILLS = MappingProxyType({
    "grasp": ("grasp_and_hold",),
    "store_bowl": tuple(SKILLS),
})


@dataclass(frozen=True)
class GoalSpec:
    name: str
    label: str
    target_objects: tuple[str, ...]
    checks: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


_EMPTY_SUPPORTED = ("hand_empty", "hand_clear_of_drawer", "support_preserved")
GOALS = MappingProxyType({
    "hold_target": GoalSpec(
        "hold_target", "抓起当前目标并稳定保持", tuple(TARGETS),
        ("bilateral_grip", "clear_of_table", "target_lifted", "relative_pose_stable", "verified_hold"),
    ),
    "store_and_home": GoalSpec(
        "store_and_home", "黑碗入屉、关抽屉并归位", ("bowl",),
        ("bowl_inside_drawer", "bowl_supported", "bowl_settled", "drawer_closed", "home_reached") + _EMPTY_SUPPORTED,
    ),
    "close_and_home": GoalSpec(
        "close_and_home", "关抽屉并归位，保持物体支撑", tuple(TARGETS),
        ("drawer_closed", "home_reached") + _EMPTY_SUPPORTED,
    ),
    "home": GoalSpec(
        "home", "空夹爪安全归位", tuple(TARGETS), ("home_reached",) + _EMPTY_SUPPORTED,
    ),
})
