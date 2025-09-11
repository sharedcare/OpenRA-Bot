import os
import sys
import random
from typing import List, Dict, Any, Optional, Tuple
import pythonnet

# Optional: use gymnasium if preferred
try:
    import gym
    from gym import spaces
except ImportError:
    import gymnasium as gym
    from gymnasium import spaces


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

        # Ensure BIN dir is current working dir and import path
        pythonnet.load("coreclr", runtime_config=r"F:\\Projects\\OpenRA\\bin\\OpenRA.runtimeconfig.json")

        os.makedirs(self.bin_dir, exist_ok=True)
        os.chdir(self.bin_dir)
        if self.bin_dir not in sys.path:
            sys.path.append(self.bin_dir)

        import clr  # type: ignore  # pylint: disable=import-error
        clr.AddReference('OpenRA.Game')  # pylint: disable=no-member
        clr.AddReference('OpenRA.Utility')  # pylint: disable=no-member
        clr.AddReference('OpenRA.Platforms.Default')  # pylint: disable=no-member

        from OpenRA import PythonAPI, CPos, RLAction, RLTarget, Game  # type: ignore  # pylint: disable=import-error

        self._openra = {
            'PythonAPI': PythonAPI,
            'CPos': CPos,
            'RLAction': RLAction,
            'RLTarget': RLTarget,
            'Game': Game,
        }

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
        api = self._openra['PythonAPI']
        if not self.remote_host or not self.remote_port:
            raise RuntimeError("Remote host/port not configured. Call configure_remote().")

        api.JoinServer(self.mod_id, self.remote_host, int(self.remote_port), self.remote_password, self.bin_dir)

        # Optional: claim slot or spectate
        if self.remote_spectator:
            api.SetSpectator(True)
        else:
            slot = self.remote_slot
            if not slot:
                slots = list(api.GetAvailableSlots())
                if slots:
                    slot = slots[0]
            if slot:
                api.ClaimSlot(slot)

        # Ready up (only if not spectating)
        if not self.remote_spectator:
            api.SetReady(True)

        # Wait until game starts; tick the network/logic while waiting
        for _ in range(6000):
            if api.IsInGame():
                break
            api.Step()

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
        api = self._openra['PythonAPI']
        setup = list(self.host_options)
        if not any(s.startswith("state ") for s in setup):
            setup.append("state 1")
        api.CreateAndStartLocalServer(self.mod_id, self.map_uid, self.bin_dir, setup)

        for _ in range(6000):
            if api.IsInGame():
                break
            api.Step()

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

        first_obs = self._get_obs()
        self._last_obs = first_obs
        first_info = {}
        return first_obs, first_info

    def step(self, action: Any):  # type: ignore[override]
        self._ensure_engine()
        self._send_actions(action)

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
        api = self._openra['PythonAPI']
        state = api.GetState()

        actors = []
        for a in state.Actors:
            actors.append({
                'id': int(a.ActorId),
                'type': str(a.Type) if a.Type is not None else '',
                'owner': int(a.OwnerIndex),
                'cell_bits': int(a.CellBits),
                'hp': int(a.HP),
                'max_hp': int(a.MaxHP),
                'dead': bool(a.IsDead),
                'available_orders': list(a.AvailableOrders) if getattr(a, 'AvailableOrders', None) is not None else [],
            })

        obs = {
            'world_tick': int(state.WorldTick),
            'net_frame': int(state.NetFrame),
            'local_frame': int(state.LocalFrame),
            'actors': actors,
        }

        my_owner = int(self._openra['Game'].LocalClientId)
        available = set()
        for u in actors:
            if u['owner'] == my_owner and not u['dead']:
                for oid in u.get('available_orders', []):
                    available.add(oid)
        obs['valid_action_mask'] = sorted(list(available))

        return obs

    def pick_agent_actions(self, owner_index: Optional[int] = None) -> List[Dict[str, Any]]:
        if self._last_obs is None:
            return []

        obs = self._last_obs
        actors = obs.get('actors', [])
        if owner_index is None:
            owner_index = int(self._openra['Game'].LocalClientId)

        my_units = [a for a in actors if a['owner'] == owner_index and not a['dead']]
        if not my_units:
            return []

        a = random.choice(my_units)
        return [{
            'order': 'Move',
            'subject': a['id'],
            'target_cell': (10, 10),
            'queued': False
        }]

    def _send_actions(self, action: Any) -> None:
        if action is None:
            return

        if isinstance(action, dict):
            actions_list = [action]
        elif isinstance(action, (list, tuple)):
            actions_list = list(action)
        else:
            raise ValueError("Unsupported action type. Expect dict or list of dicts.")

        RLAction = self._openra['RLAction']
        RLTarget = self._openra['RLTarget']
        PythonAPI = self._openra['PythonAPI']
        CPos = self._openra['CPos']

        def make_cell_target(xy: Tuple[int, int], subcell: int = 0):
            x, y = int(xy[0]), int(xy[1])
            t = RLTarget()
            t.Type = "Cell"
            t.CellBits = CPos(x, y).Bits
            t.SubCell = int(subcell)
            return t

        def make_actor_target(actor_id: int):
            t = RLTarget()
            t.Type = "Actor"
            t.ActorId = int(actor_id)
            return t

        rl_actions = []
        for a in actions_list:
            if not isinstance(a, dict):
                continue

            order = str(a.get('order', ''))
            subject = a.get('subject')
            if subject is None:
                continue

            queued = bool(a.get('queued', False))
            target = None

            if 'target_cell' in a:
                target = make_cell_target(a['target_cell'], int(a.get('subcell', 0)))
            elif 'target_actor' in a:
                target = make_actor_target(a['target_actor'])

            target_string = a.get('target_string')
            extra_data = int(a.get('extra_data', 0))

            ra = RLAction()
            ra.Order = order
            ra.SubjectActorId = int(subject)
            ra.Queued = bool(queued)
            ra.Target = target
            ra.TargetString = target_string
            ra.ExtraData = int(extra_data)
            rl_actions.append(ra)

        if rl_actions:
            PythonAPI.SendActions(rl_actions)


# Convenience factory

def make_env(bin_dir: str, mod_id: str, map_uid: str, **kwargs) -> OpenRAEnv:
    return OpenRAEnv(bin_dir=bin_dir, mod_id=mod_id, map_uid=map_uid, **kwargs)

if __name__ == "__main__":
    env = make_env(bin_dir="F:/Projects/OpenRA/bin", mod_id="ra", map_uid="b53e25e007666442dbf62b87eec7bfbe8160ef3f")
    # Host a local game the agent can play in (others can connect to watch)
    env.configure_host(options=["option gamespeed default", "name PythonAgent", "slot Multi0", "state 1"])  # uncomment to host
    # Or join a remote lobby instead:
    # env.configure_remote(host="10.10.10.120", port=1234, password="1234", spectator=False)
    first_obs, first_info = env.reset()
    print(first_obs)
    print(first_info)
    for _ in range(1000):
        actions = env.pick_agent_actions()
        env.step(actions)
    env.close()
