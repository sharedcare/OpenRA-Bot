"""Build-order goal library with goal-conditioned reward.

Each goal is a target composition (building + unit counts). At episode start,
a goal is sampled and encoded into the observation scalar. The reward is
decomposed into goal-aligned and goal-agnostic components, so the agent is
rewarded for making progress toward its assigned target rather than guessing
an implicit objective.

Design follows DI-star / AlphaStar z-conditioning: the goal `z` is a
one-hot or embedding that conditions the policy and value function.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class BuildOrderGoal:
    """A target composition for a specific strategic objective."""
    name: str
    # Target building counts (type_name -> count)
    target_buildings: Dict[str, int] = field(default_factory=dict)
    # Target unit counts (type_name -> count)
    target_units: Dict[str, int] = field(default_factory=dict)
    # Priority weights for reward components under this goal
    building_weight: float = 1.0
    unit_weight: float = 0.3
    # Goal embedding index
    index: int = 0


# ---------------------------------------------------------------------------
# Predefined goals for RA mod (development-only, no combat)
# ---------------------------------------------------------------------------

GOALS: List[BuildOrderGoal] = [
    BuildOrderGoal(
        name="economy",
        index=0,
        target_buildings={"powr": 2, "proc": 2, "fact": 1},
        target_units={"harv": 3},
        building_weight=1.0,
        unit_weight=0.1,
    ),
    BuildOrderGoal(
        name="infantry",
        index=1,
        target_buildings={"powr": 2, "proc": 1, "barr": 1, "fact": 1},
        target_units={"e1": 15, "e3": 3},
        building_weight=0.5,
        unit_weight=0.5,
    ),
    BuildOrderGoal(
        name="vehicle",
        index=2,
        target_buildings={"powr": 2, "proc": 2, "weap": 1, "fact": 1},
        target_units={"harv": 2, "1tnk": 5, "jeep": 3},
        building_weight=0.3,
        unit_weight=0.7,
    ),
    BuildOrderGoal(
        name="balanced",
        index=3,
        target_buildings={"powr": 2, "proc": 2, "barr": 1, "weap": 1, "fact": 1},
        target_units={"harv": 2, "e1": 10, "1tnk": 3, "jeep": 2},
        building_weight=0.5,
        unit_weight=0.5,
    ),
]

# Human-readable names for scalar encoding
GOAL_NAMES = [g.name for g in GOALS]


class GoalLibrary:
    """Manages goal sampling and goal-conditioned reward computation.

    Usage:
        lib = GoalLibrary()
        goal = lib.sample()                        # episode start
        goal_vec = lib.encode_scalar(goal)          # append to obs scalar
        reward = lib.goal_reward(goal, owned)       # per-step reward
    """

    def __init__(self, goals: Optional[List[BuildOrderGoal]] = None) -> None:
        self._goals = goals or GOALS
        self._num_goals = len(self._goals)

    @property
    def num_goals(self) -> int:
        return self._num_goals

    def sample(self) -> BuildOrderGoal:
        """Uniformly sample a goal for an episode."""
        return random.choice(self._goals)

    def get_by_index(self, idx: int) -> BuildOrderGoal:
        return self._goals[idx % self._num_goals]

    def encode_scalar(self, goal: BuildOrderGoal) -> List[float]:
        """Encode goal as a fixed-size scalar vector for observation.

        Returns [one_hot_over_goals, building_weight, unit_weight].
        Total dim = num_goals + 2.
        """
        one_hot = [0.0] * self._num_goals
        one_hot[goal.index] = 1.0
        return one_hot + [goal.building_weight, goal.unit_weight]

    @staticmethod
    def scalar_dim() -> int:
        """Extra scalar dimensions added by goal conditioning."""
        return len(GOALS) + 2  # one-hot + 2 weights

    def goal_reward(
        self,
        goal: BuildOrderGoal,
        owned_buildings: Dict[str, int],
        owned_units: Dict[str, int],
    ) -> Tuple[float, float, float]:
        """Compute goal-aligned reward components.

        Returns (building_progress, unit_progress, total_goal_reward).
        Each is in [0, 1] range: fraction of target achieved.
        """
        bld_progress = 0.0
        for btype, target in goal.target_buildings.items():
            current = owned_buildings.get(btype, 0)
            bld_progress += min(current / max(target, 1), 1.0) / max(len(goal.target_buildings), 1)

        unit_progress = 0.0
        for utype, target in goal.target_units.items():
            current = owned_units.get(utype, 0)
            unit_progress += min(current / max(target, 1), 1.0) / max(len(goal.target_units), 1)

        total = goal.building_weight * bld_progress + goal.unit_weight * unit_progress
        return bld_progress, unit_progress, total