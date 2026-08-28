from dataclasses import dataclass


@dataclass(frozen=True)
class ObstacleStatus:
    obstacle_ahead: bool
    emergency: bool
    left_clear: bool
    right_clear: bool
    front_min: float
    left_min: float
    right_min: float
    front_count: int
    left_count: int
    right_count: int


@dataclass(frozen=True)
class OvertakeDecision:
    state: str
    offset: float
    motion_enabled: bool