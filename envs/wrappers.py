"""
Gym wrappers for augmented state and shaped reward.

Wrapping order (inner to outer):
    env = OpenRAEnv(...)
    env = ShapedRewardWrapper(env)
    env = AugmentedStateWrapper(env)

ShapedRewardWrapper provides a multi-component reward that is dense enough
for an RL agent to bootstrap from random exploration:
    - building count change (+1 per new building, -1 per lost)
    - production start bonus (encourages queuing)
    - deploy bonus (encourages MCV deployment)
    - idle penalty (discourages doing nothing)

AugmentedStateWrapper enriches the vector observation with temporal context
so that the RL agent (PPO + LSTM) can learn delayed credit assignment:
    - frame stacking (last k observations)
    - action history (last k actions, one-hot by action type)
    - state delta (current obs minus previous obs)
    - time-since-action counter per action type
"""

from __future__ import annotations

import sys
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

try:
    import gym
    from gym import spaces
except ImportError:
    import gymnasium as gym
    from gymnasium import spaces


def _get_base_env(env: gym.Env):
    """Walk the wrapper chain to reach the underlying OpenRAEnv."""
    inner = env
    while hasattr(inner, "env"):
        if type(inner).__name__ == "OpenRAEnv":
            return inner
        inner = inner.env
    return inner


class ShapedRewardWrapper(gym.Wrapper):
    """Multi-component shaped reward for bootstrapping macro play.

    Reward components (per step):
        +1.0  per new building
        -1.0  per lost building
        +0.3  per new unit (non-building)
        -0.3  per lost unit
        +0.1  when a new production item is queued
        +0.5  one-time bonus when MCV deploys (fact appears for the first time)
        -0.01 per step where the agent takes noop and has idle mobile units

    All values are configurable via ``reward_weights``.
    """

    def __init__(self, env: gym.Env, verbose: bool = False) -> None:
        super().__init__(env)
        self.verbose = verbose

        self.reward_weights: Dict[str, float] = {
            "building": 1.0,
            "unit": 0.3,
            "production_start": 0.1,
            "deploy_bonus": 0.5,
            "idle_penalty": 0.01,
        }

        # State tracking.
        self._prev_building_count: int = 0
        self._prev_unit_count: int = 0
        self._prev_prod_keys: Set[Tuple[int, str]] = set()
        self._deployed_once: bool = False
        self._step_idx: int = 0

    def reset(self, **kwargs: Any) -> Tuple[Any, Dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        buildings, units = self._count_actors()
        self._prev_building_count = buildings
        self._prev_unit_count = units
        self._prev_prod_keys = self._production_keys()
        self._deployed_once = buildings > 0  # already deployed if fact exists at start
        self._step_idx = 0
        if self.verbose:
            print(f"[reward] reset  buildings={buildings} units={units} deployed={self._deployed_once}",
                  file=sys.stderr)
        return obs, info

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        obs, _orig_reward, terminated, truncated, info = self.env.step(action)
        self._step_idx += 1
        reward = 0.0

        buildings, units = self._count_actors()

        # --- building diff ---
        db = buildings - self._prev_building_count
        reward += self.reward_weights["building"] * db

        # --- unit diff ---
        du = units - self._prev_unit_count
        reward += self.reward_weights["unit"] * du

        # --- deploy bonus (one-time) ---
        if not self._deployed_once and buildings > 0:
            reward += self.reward_weights["deploy_bonus"]
            self._deployed_once = True

        # --- production start ---
        cur_prod = self._production_keys()
        new_starts = cur_prod - self._prev_prod_keys
        if new_starts:
            reward += self.reward_weights["production_start"] * len(new_starts)

        # --- idle penalty ---
        action_type = int(action[0]) if hasattr(action, "__len__") else int(action)
        if action_type == 0 and units > 0:  # noop with idle units
            reward -= self.reward_weights["idle_penalty"]

        if self.verbose and reward != 0.0:
            print(f"[reward] step={self._step_idx} r={reward:+.3f} "
                  f"db={db} du={du} starts={len(new_starts)} buildings={buildings} units={units}",
                  file=sys.stderr)

        # Periodic diagnostic dump (every 50 steps) to debug build availability.
        if self.verbose and self._step_idx % 20 == 0:
            self._dump_build_diagnostic()

        self._prev_building_count = buildings
        self._prev_unit_count = units
        self._prev_prod_keys = cur_prod
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------

    def _dump_build_diagnostic(self) -> None:
        """Print production queue and placeable_areas state for debugging."""
        raw = getattr(self.env, "_last_raw_state", None)
        if raw is None:
            print(f"[diag] step={self._step_idx} raw_state=NONE", file=sys.stderr)
            return
        prod = raw.get("production") or {}
        queues = prod.get("Queues") or []
        pa = raw.get("placeable_areas") or {}

        # Show actor summary for player to verify the fact exists
        my_owner = int(raw.get("my_owner", -1))
        actors = raw.get("actors") or []
        my_actors = [
            f"{a.get('type','?')}(id={a.get('id','?')})"
            for a in actors if int(a.get("owner", -1)) == my_owner and not bool(a.get("dead", False))
        ]

        queue_summary = []
        for q in queues:
            q_id = int(q.get("ActorId", -1))
            q_type = str(q.get("Type", "")).lower() or "?"
            enabled = bool(q.get("Enabled", False))
            items = q.get("Items") or []
            item_strs = []
            for it in items:
                name = str(it.get("Item") or it.get("Name") or "?")
                done = bool(it.get("Done", False))
                prog = it.get("Progress", 0)
                item_strs.append(f"{name}({'DONE' if done else f'{prog}%'})")
            queue_summary.append(f"q{q_id}/{q_type}[{'on' if enabled else 'off'}]:{','.join(item_strs) or 'empty'}")

        pa_summary = {k: len(v) for k, v in pa.items() if v}

        print(f"[diag] step={self._step_idx} "
              f"actors={my_actors or 'NONE'} "
              f"queues=[{' | '.join(queue_summary) or 'EMPTY'}] "
              f"placeable={pa_summary or 'EMPTY'}",
              file=sys.stderr)

    def _count_actors(self) -> Tuple[int, int]:
        """Return (building_count, mobile_unit_count) for my alive actors."""
        raw = getattr(self.env, "_last_raw_state", None)
        if raw is None:
            return 0, 0
        actors = raw.get("actors") or []
        my_owner = int(raw.get("my_owner", -1))
        buildings = 0
        units = 0
        for u in actors:
            if int(u.get("owner", -1)) != my_owner:
                continue
            if bool(u.get("dead", False)):
                continue
            orders = {str(x).lower() for x in (u.get("available_orders") or [])}
            if "move" not in orders:
                buildings += 1
            else:
                units += 1
        return buildings, units

    def _production_keys(self) -> Set[Tuple[int, str]]:
        """Extract set of (queue_actor_id, item_name) for in-progress items."""
        raw = getattr(self.env, "_last_raw_state", None)
        if raw is None:
            return set()
        keys: Set[Tuple[int, str]] = set()
        prod = raw.get("production") or {}
        for q in prod.get("Queues") or []:
            q_id = int(q.get("ActorId", -1))
            for it in q.get("Items") or []:
                name = str(it.get("Item") or it.get("Name") or "")
                if name and not bool(it.get("Done", False)):
                    keys.add((q_id, name))
        return keys


# Keep the old name as an alias for backward compatibility.
StateDiffRewardWrapper = ShapedRewardWrapper


class AugmentedStateWrapper(gym.Wrapper):
    """Augment vector observations with temporal context.

    The augmented observation is the concatenation of:
        [frame_stack (k * obs_dim),
         action_history (k * num_action_types),
         state_delta (obs_dim),
         time_since_action (num_action_types)]

    Parameters
    ----------
    env : gym.Env
        Must use observation_type="vector".
    frame_stack_k : int
        Number of past frames (including current) to stack.
    """

    def __init__(self, env: gym.Env, frame_stack_k: int = 8) -> None:
        super().__init__(env)
        self.k = frame_stack_k

        # Resolve the underlying OpenRAEnv to read action_types.
        inner = env
        while hasattr(inner, "env"):
            if hasattr(inner, "action_types"):
                break
            inner = inner.env
        self.num_action_types: int = len(getattr(inner, "action_types", []))
        if self.num_action_types == 0:
            raise ValueError("Cannot determine num_action_types from wrapped env")

        # Base observation dimension (flat vector).
        self.base_obs_dim: int = int(env.observation_space.shape[0])

        # Augmented dimension breakdown:
        #   k * obs_dim          frame stack
        #   k * num_action_types action history (one-hot)
        #   obs_dim              state delta
        #   num_action_types     time since each action type
        self.aug_obs_dim = (
            self.k * self.base_obs_dim
            + self.k * self.num_action_types
            + self.base_obs_dim
            + self.num_action_types
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.aug_obs_dim,),
            dtype=np.float32,
        )

        # Internal buffers (initialized in reset).
        self._obs_history: deque[np.ndarray] = deque(maxlen=self.k)
        self._action_history: deque[np.ndarray] = deque(maxlen=self.k)
        self._prev_obs: Optional[np.ndarray] = None
        self._time_since_action = np.zeros(self.num_action_types, dtype=np.float32)

    @property
    def augmentation_config(self) -> Dict[str, int]:
        """Metadata consumed by ``AugmentedVectorEncoder`` / ``ActorCritic``."""
        return {
            "base_obs_dim": self.base_obs_dim,
            "frame_stack_k": self.k,
            "num_action_types": self.num_action_types,
        }

    # ------------------------------------------------------------------

    def reset(self, **kwargs: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)

        zero_obs = np.zeros(self.base_obs_dim, dtype=np.float32)
        zero_act = np.zeros(self.num_action_types, dtype=np.float32)

        self._obs_history.clear()
        self._action_history.clear()
        for _ in range(self.k - 1):
            self._obs_history.append(zero_obs.copy())
            self._action_history.append(zero_act.copy())
        self._obs_history.append(obs.copy())
        self._action_history.append(zero_act.copy())

        self._prev_obs = obs.copy()
        # Large initial value: no action has ever been taken.
        self._time_since_action = np.full(
            self.num_action_types, float(self.k), dtype=np.float32
        )

        return self._build_augmented(obs), info

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        # --- update action history ---
        action_type = int(action[0]) if hasattr(action, "__len__") else int(action)
        onehot = np.zeros(self.num_action_types, dtype=np.float32)
        if 0 <= action_type < self.num_action_types:
            onehot[action_type] = 1.0
        self._action_history.append(onehot)

        # --- update time-since-action ---
        self._time_since_action += 1.0
        if 0 <= action_type < self.num_action_types:
            self._time_since_action[action_type] = 0.0

        # --- update observation history ---
        self._obs_history.append(obs.copy())

        aug = self._build_augmented(obs)
        self._prev_obs = obs.copy()
        return aug, reward, terminated, truncated, info

    # ------------------------------------------------------------------

    def _build_augmented(self, current_obs: np.ndarray) -> np.ndarray:
        # 1) Frame stack: oldest to newest.
        frames = np.concatenate(list(self._obs_history), axis=0)

        # 2) Action history: oldest to newest, one-hot.
        actions = np.concatenate(list(self._action_history), axis=0)

        # 3) State delta: current minus previous.
        if self._prev_obs is not None:
            delta = current_obs - self._prev_obs
        else:
            delta = np.zeros_like(current_obs)

        # 4) Time since each action type, normalized by k.
        time_feat = self._time_since_action / max(1.0, float(self.k))

        return np.concatenate([frames, actions, delta, time_feat]).astype(np.float32)
