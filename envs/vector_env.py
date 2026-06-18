"""Subprocess-based vectorized OpenRA environment.

Each env runs in its own process (multiprocessing 'spawn' start method) so each
loads an independent .NET CLR + headless OpenRA engine. 'spawn' is mandatory:
pythonnet's CLR cannot be safely used across a fork.

Workers auto-reset on episode end (standard for on-policy PPO); the terminal
observation is returned in info['terminal_observation'].

Observations are stacked across envs:
  - dict obs (entity mode): {k: np.stack([... per env ...])}
  - array obs (vector mode): np.stack(...)

info stays a per-env list (each carries its own 'action_mask').
"""
from __future__ import annotations

import multiprocessing as mp
from typing import Any, Callable, Dict, List, Tuple

import numpy as np


class EnvFactory:
    """Picklable env factory for spawn workers (closures are not picklable).

    Holds make_env kwargs; calling it builds a fresh OpenRAEnv in the worker
    process. headless is forced on (parallel requires no window/GL).
    """

    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)
        self.kwargs["headless"] = True

    def __call__(self):
        # Imported inside the worker so each process loads its own module state.
        try:
            from envs.openra_env import make_env
        except ImportError:
            from OpenRA.Bot.envs.openra_env import make_env  # type: ignore
        return make_env(**self.kwargs)


def _worker(remote, parent_remote, env_fn):
    parent_remote.close()
    env = env_fn()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                obs, reward, terminated, truncated, info = env.step(data)
                if terminated or truncated:
                    info = dict(info or {})
                    info["terminal_observation"] = obs
                    obs, reset_info = env.reset()
                    # keep the fresh-episode action_mask for the next step
                    info["action_mask"] = (reset_info or {}).get("action_mask", info.get("action_mask"))
                remote.send((obs, reward, terminated, truncated, info))
            elif cmd == "reset":
                obs, info = env.reset()
                remote.send((obs, info))
            elif cmd == "action_space":
                remote.send((env.action_space, env.observation_space, list(env.action_types)))
            elif cmd == "close":
                try:
                    env.close()
                except Exception:
                    pass
                remote.close()
                break
            else:
                raise ValueError(f"Unknown command: {cmd}")
    except (KeyboardInterrupt, EOFError):
        pass


class SubprocVecEnv:
    """Vectorized env running each OpenRAEnv in its own spawned process."""

    def __init__(self, env_fns: List[Callable[[], Any]]):
        self.n_envs = len(env_fns)
        ctx = mp.get_context("spawn")
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.n_envs)])
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            p = ctx.Process(target=_worker, args=(work_remote, remote, env_fn), daemon=True)
            p.start()
            work_remote.close()
            self.processes.append(p)

        self.remotes[0].send(("action_space", None))
        self.action_space, self.observation_space, self.action_types = self.remotes[0].recv()
        self.closed = False

    def reset(self) -> Tuple[Any, List[Dict[str, Any]]]:
        for remote in self.remotes:
            remote.send(("reset", None))
        results = [remote.recv() for remote in self.remotes]
        obs = [r[0] for r in results]
        infos = [r[1] for r in results]
        return self._stack_obs(obs), list(infos)

    def step(self, actions) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        results = [remote.recv() for remote in self.remotes]
        obs, rewards, terminated, truncated, infos = zip(*results)
        return (
            self._stack_obs(list(obs)),
            np.array(rewards, dtype=np.float32),
            np.array(terminated, dtype=bool),
            np.array(truncated, dtype=bool),
            list(infos),
        )

    def close(self):
        if self.closed:
            return
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except Exception:
                pass
        for p in self.processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        self.closed = True

    @staticmethod
    def _stack_obs(obs_list: List[Any]) -> Any:
        if isinstance(obs_list[0], dict):
            return {k: np.stack([o[k] for o in obs_list]) for k in obs_list[0].keys()}
        return np.stack(obs_list)
