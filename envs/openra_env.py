from typing import List, Dict, Any, Optional
import os
import sys

try:
    import gym
    from gym import spaces
except ImportError:
    import gymnasium as gym
    from gymnasium import spaces

# Make sibling utils package importable when running without installed package
_CUR_DIR = os.path.dirname(__file__)
_PKG_ROOT = os.path.abspath(os.path.join(_CUR_DIR, os.pardir))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from utils.engine import ensure_engine
from utils.net import join_remote, host_local, get_connection_state, is_connected_to_lobby, wait_for_connection
from utils.obs import build_observation
from utils.actions import send_actions
from agent.agent import BaseAgent


class OpenRAEnv(gym.Env):
    """
    Minimal Gym wrapper around OpenRA's PythonAPI.

    - Uses pythonnet (clr) to call into OpenRA.Game.dll
    - Exposes reset/step/close, with a simple dict-based action format

    Action format (list of dicts):
      [
        { 'order': 'Move', 'subject': <actor_id>, 'target_cell': (x, y), 'queued': False },
        { 'order': 'Attack', 'subject': <actor_id>, 'target_actor': <actor_id>, 'queued': False },
        { 'order': 'Stop', 'subject': <actor_id> }
      ]

    Observation: the raw RLState from PythonAPI.GetState() converted to a python dict:
      {
        'world_tick': int,
        'net_frame': int,
        'local_frame': int,
        'actors': [
          { 'id': int, 'type': str, 'owner': int, 'cell_bits': int, 'hp': int, 'max_hp': int, 'dead': bool }
        ]
      }

    Reward/Termination: left as placeholders; customize for your training.
    """

    metadata = { 'render.modes': [] }

    def __init__(
        self,
        bin_dir: str,
        mod_id: str,
        map_uid: str,
        ticks_per_step: int = 1,
        max_episode_ticks: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.bin_dir = bin_dir
        self.mod_id = mod_id
        self.map_uid = map_uid
        self.ticks_per_step = max(1, int(ticks_per_step))
        self.max_episode_ticks = max_episode_ticks

        # Lazy init of pythonnet/OpenRA until first reset()
        self._initialized = False
        self._openra = None

        # Spaces are left generic; customize to your needs
        self.observation_space = spaces.Dict({})
        self.action_space = spaces.Dict({})

        self._last_obs = None

        # Remote join settings (optional)
        self.remote_host: Optional[str] = None
        self.remote_port: Optional[int] = None
        self.remote_password: str = ""
        self.remote_slot: Optional[str] = None
        self.remote_spectator: bool = False

        # Host-local settings (optional)
        self.host_local: bool = False
        self.host_options: List[str] = []

    # --- Engine bootstrap ---

    def _ensure_engine(self) -> None:
        if self._initialized:
            return

        self._openra = ensure_engine(self.bin_dir)

        self._initialized = True

    # --- Remote helpers ---

    def configure_remote(self, host: str, port: int, password: str = "", slot: Optional[str] = None, spectator: bool = False) -> None:
        self.remote_host = host
        self.remote_port = int(port)
        self.remote_password = password or ""
        self.remote_slot = slot
        self.remote_spectator = bool(spectator)
        self.host_local = False

    def _join_remote(self) -> None:
        if not self.remote_host or not self.remote_port:
            raise RuntimeError("Remote host/port not configured. Call configure_remote().")
        join_remote(
            self._openra,
            mod_id=self.mod_id,
            host=self.remote_host,
            port=int(self.remote_port),
            password=self.remote_password,
            bin_dir=self.bin_dir,
            slot=self.remote_slot,
            spectator=self.remote_spectator,
        )

    # --- Host-local helpers ---

    def configure_host(self, options: Optional[List[str]] = None) -> None:
        """Host a local server for `map_uid`. `options` is a list of raw lobby commands.
        Example:
          [
            "option gamespeed default",
            "name PythonAgent",
            "slot Multi0",
            "state 1"
          ]
        """
        self.host_local = True
        self.host_options = list(options or [])
        self.remote_host = None
        self.remote_port = None

    def _host_local(self) -> None:
        host_local(self._openra, self.mod_id, self.map_uid, self.bin_dir, self.host_options)

    # --- Gym API ---

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):  # type: ignore[override]
        super().reset(seed=seed)
        self._ensure_engine()

        api = self._openra['PythonAPI']
        if self.host_local:
            self._host_local()
        elif self.remote_host and self.remote_port:
            self._join_remote()
        else:
            api.StartLocalGame(self.mod_id, self.map_uid, self.bin_dir)

        obs = self._get_obs()
        self._last_obs = obs
        info = {}
        return obs, info

    def step(self, action: Any):  # type: ignore[override]
        self._ensure_engine()
        self.send_actions(action)

        api = self._openra['PythonAPI']
        for _ in range(self.ticks_per_step):
            api.Step()

        new_obs = self._get_obs()

        reward = 0.0
        terminated = False
        truncated = False

        if self.max_episode_ticks is not None and new_obs['world_tick'] >= self.max_episode_ticks:
            truncated = True

        info = {}
        self._last_obs = new_obs
        return new_obs, reward, terminated, truncated, info

    def render(self):  # type: ignore[override]
        return None

    def close(self):  # type: ignore[override]
        pass

    # --- Helpers ---

    def _get_obs(self) -> Dict[str, Any]:
        return build_observation(self._openra)

    def send_actions(self, actions: List[Dict[str, Any]]) -> None:
        send_actions(self._openra, actions)

    # --- Additional connection helpers ---

    def get_connection_state(self) -> str:
        """Get current connection state as string."""
        if not self._initialized:
            return "NotInitialized"
        return get_connection_state(self._openra)

    def get_lobby_info(self) -> Dict[str, Any]:
        if not self._initialized:
            return {}
        return dict(self._openra['PythonAPI'].GetLobbyInfo())

    def is_connected_to_lobby(self) -> bool:
        if not self._initialized:
            return False
        return is_connected_to_lobby(self._openra)

    def wait_for_connection(self, timeout_ms: int = 10000) -> bool:
        if not self._initialized:
            return False
        return wait_for_connection(self._openra, timeout_ms)


# Convenience factory
def make_env(bin_dir: str, mod_id: str, map_uid: str, **kwargs) -> OpenRAEnv:
    return OpenRAEnv(bin_dir=bin_dir, mod_id=mod_id, map_uid=map_uid, **kwargs)

if __name__ == "__main__":
    # Keep the module free of side-effects; see scripts/ for runnable examples.
    pass
