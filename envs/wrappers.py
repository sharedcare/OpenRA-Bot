"""
Gym wrappers for augmented state and diagnostic logging.

Wrapping order (inner to outer):
    env = OpenRAEnv(...)
    env = ShapedRewardWrapper(env)
    env = AugmentedStateWrapper(env)

ShapedRewardWrapper is now a transparent pass-through — all reward shaping
lives in OpenRAEnv._compute_reward. The wrapper only adds optional verbose
diagnostics for debugging reward signals.

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
from typing import Any, Dict, Optional, Tuple

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
    """Transparent reward pass-through with optional diagnostic logging.

    All reward shaping now lives in OpenRAEnv._compute_reward so this
    wrapper only adds verbose diagnostics for debugging reward signals.
    """

    def __init__(self, env: gym.Env, verbose: bool = False) -> None:
        super().__init__(env)
        self.verbose = verbose
        self._step_idx: int = 0

    def reset(self, **kwargs: Any) -> Tuple[Any, Dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        self._step_idx = 0
        if self.verbose:
            raw = getattr(self.env, "_last_raw_state", None)
            if raw is not None:
                my_owner = int(raw.get("my_owner", -1))
                actors = raw.get("actors") or []
                my_actors = [
                    f"{a.get('type','?')}(id={a.get('id','?')})"
                    for a in actors if int(a.get("owner", -1)) == my_owner and not bool(a.get("dead", False))
                ]
                print(f"[reward] reset  actors={my_actors}", file=sys.stderr)
        return obs, info

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_idx += 1

        if self.verbose and reward != 0.0:
            print(f"[reward] step={self._step_idx} r={reward:+.3f}", file=sys.stderr)

        if self.verbose and self._step_idx % 20 == 0:
            self._dump_build_diagnostic()

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
