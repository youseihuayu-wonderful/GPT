"""Bounded six-object sorting proposals, independent of physics and providers."""

from dataclasses import dataclass


TABLE_Z = 0.72
COLORS = {"red": (0.85, 0.12, 0.10, 1.0),
          "blue": (0.10, 0.32, 0.90, 1.0),
          "yellow": (0.98, 0.75, 0.06, 1.0)}


@dataclass(frozen=True)
class SortingObject:
    id: str
    color: str
    kind: str
    source_xy_m: tuple[float, float]

    @property
    def body_name(self):
        return f"{self.id}_object"

    @property
    def label(self):
        return f"{self.color} {self.kind}"

    @property
    def grasp_offset_m(self):
        return (0.0, -0.047, 0.042) if self.kind == "bowl" else (0.0, 0.0, 0.025)


OBJECTS = tuple(
    SortingObject(f"{color}_{kind}", color, kind,
                  (x, -0.08 if kind == "bowl" else -0.28))
    for color, x in zip(COLORS, (0.30, 0.47, 0.64))
    for kind in ("bowl", "cup")
)
ZONES = tuple({"id": destination, "label": f"{destination} · {color}",
               "color": color, "center_xy_m": [x, 0.195],
               "half_size_xy_m": [0.075, 0.16],
               "slots_xy_m": [[x, 0.10], [x, 0.29]]}
              for color, destination, x in zip(COLORS, "ABC", (0.30, 0.47, 0.64)))
_OBJECT_INDICES = {obj.id: index for index, obj in enumerate(OBJECTS)}


@dataclass(frozen=True, slots=True)
class SortStep:
    object_id: str
    destination: str

    def __post_init__(self):
        if type(self.object_id) is not str or self.object_id not in _OBJECT_INDICES:
            raise ValueError("Sorting step requires a known object_id")
        if type(self.destination) is not str or self.destination not in ("A", "B", "C"):
            raise ValueError("Sorting step requires destination A, B, or C")

    @property
    def object_index(self) -> int:
        return _OBJECT_INDICES[self.object_id]

    @property
    def slot_xy_m(self) -> tuple[float, float]:
        slot = 0 if OBJECTS[self.object_index].kind == "bowl" else 1
        return tuple(ZONES["ABC".index(self.destination)]["slots_xy_m"][slot])


@dataclass(frozen=True, slots=True)
class SortingPlan:
    steps: tuple[SortStep, ...]

    def __post_init__(self):
        if type(self.steps) is not tuple or len(self.steps) != len(OBJECTS):
            raise ValueError("Sorting plan requires exactly six immutable steps")
        identities = set()
        destinations = {}
        for step in self.steps:
            if type(step) is not SortStep:
                raise ValueError("Sorting plan entries must be SortStep values")
            # Recheck entries rather than trust a previously constructed instance.
            step.__post_init__()
            if step.object_id in identities:
                raise ValueError("Sorting plan must include every object exactly once")
            identities.add(step.object_id)
            color = OBJECTS[step.object_index].color
            if color in destinations and destinations[color] != step.destination:
                raise ValueError("A color's bowl and cup must share one destination")
            destinations[color] = step.destination
        if len(set(destinations.values())) != len(COLORS):
            raise ValueError("Each color must have a distinct destination")

    def assignment(self, object_id: str) -> SortStep:
        for step in self.steps:
            if step.object_id == object_id:
                return step
        raise ValueError(f"Unknown sorting object: {object_id}")

    def to_list(self) -> list[dict[str, str]]:
        return [{"object_id": step.object_id, "destination": step.destination}
                for step in self.steps]


def validate_plan(payload: object) -> SortingPlan:
    """Accept only the complete, bounded wire format; never motion parameters."""
    if type(payload) is not list or len(payload) != len(OBJECTS):
        raise ValueError("Sorting plan must be a list of exactly six steps")
    steps = []
    for item in payload:
        if (type(item) is not dict or set(item) != {"object_id", "destination"}
                or any(type(key) is not str for key in item)):
            raise ValueError("Sorting steps permit only object_id and destination")
        steps.append(SortStep(item["object_id"], item["destination"]))
    return SortingPlan(tuple(steps))


FIXED_PLAN = SortingPlan(tuple(
    SortStep(obj.id, {"red": "A", "blue": "B", "yellow": "C"}[obj.color])
    for obj in OBJECTS
))
