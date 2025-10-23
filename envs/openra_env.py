from typing import List, Dict, Any, Optional, Tuple
import os
import sys
from collections import deque
import numpy as np
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

try:
    from OpenRA.Bot.utils.engine import ensure_engine  # type: ignore
    from OpenRA.Bot.utils.net import (  # type: ignore
        join_remote,
        host_local,
        get_connection_state,
        is_connected_to_lobby,
        wait_for_connection,
    )
    from OpenRA.Bot.utils.obs import build_observation  # type: ignore
    from OpenRA.Bot.utils.actions import send_actions  # type: ignore
except ImportError:  # pragma: no cover - fallback when running from source
    from utils.engine import ensure_engine
    from utils.net import (
        join_remote,
        host_local,
        get_connection_state,
        is_connected_to_lobby,
        wait_for_connection,
    )
    from utils.obs import build_observation
    from utils.actions import send_actions


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
        observation_type: str = "vector",
        enable_actions: Optional[List[str]] = None,
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

        # Config
        self.observation_type = observation_type
        default_actions = ['noop', 'move', 'attack', 'produce', 'build', 'deploy']
        self.action_types: List[str] = enable_actions if enable_actions else default_actions

        # Unit type mappings sourced from actors.csv if available
        self.unit_types: Dict[str, int] = {}
        # self._init_unit_types_from_csv()
        if not self.unit_types:
            # Fallback minimal set if CSV is unavailable
            self.unit_types = {
                'powr': 0,
                'proc': 1,
                'tent': 2,
                'barr': 3,
                'fact': 4,
                'mcv': 5,
                'e1': 6,
            }
        self.reverse_unit_types = {v: k for k, v in self.unit_types.items()}

        # Spaces
        self._setup_spaces()

        self._last_obs = None
        self._my_unit_ids: List[int] = []
        self._enemy_unit_ids: List[int] = []
        self._recent_actions: deque[Tuple] = deque(maxlen=256)
        self._action_ttl_steps: int = 8

        # Remote join settings (optional)
        self.remote_host: Optional[str] = None
        self.remote_port: Optional[int] = None
        self.remote_password: str = ""
        self.remote_slot: Optional[str] = None
        self.remote_spectator: bool = False

        # Host-local settings (optional)
        self.host_local: bool = False
        self.host_options: List[str] = []

    def _setup_spaces(self) -> None:
        max_units = 100
        max_coord = 128
        max_unit_types = len(self.unit_types)
        # Action space: [action_type, unit_idx, target_x, target_y, target_idx, unit_type]
        self.action_space = spaces.MultiDiscrete([
            len(self.action_types),
            max_units,
            max_coord,
            max_coord,
            max_units,
            max_unit_types,
        ])

        if self.observation_type == "vector":
            obs_dim = (
                max_units * 6 +
                max_units * 5 +
                7 +
                2
            )
            self.observation_space = spaces.Box(low=-1, high=max_coord, shape=(obs_dim,), dtype=np.float32)
        elif self.observation_type == "image":
            self.observation_space = spaces.Box(low=0, high=255, shape=(128, 128, 10), dtype=np.uint8)
        else:
            self.observation_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

    # --- Engine bootstrap ---

    def _ensure_engine(self) -> None:
        if self._initialized:
            return

        self._openra = ensure_engine(self.bin_dir)

        self._initialized = True

    def _init_unit_types_from_csv(self) -> None:
        """Populate unit_types mapping by reading actors.csv at repo root.
        Falls back silently if file is missing or unreadable.
        """
        try:
            # Try repo root first (../actors.csv relative to OpenRA.Bot)
            repo_root = os.path.abspath(os.path.join(_PKG_ROOT, os.pardir))
            candidates = [
                os.path.join(repo_root, 'actors.csv'),
                os.path.join(self.bin_dir or '', 'actors.csv') if self.bin_dir else '',
            ]
            csv_path = next((p for p in candidates if p and os.path.isfile(p)), '')
            if not csv_path:
                return
            names: List[str] = []
            with open(csv_path, 'r', encoding='utf-8') as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    if s.lower() == 'string':
                        # header
                        continue
                    # Keep raw actor codes as-is
                    names.append(s)
            # Build name -> index mapping
            self.unit_types = {name: idx for idx, name in enumerate(names)}
        except Exception:
            # Leave unit_types empty; caller will fallback
            self.unit_types = {}

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
            api.StartLocalGame(self.mod_id, self.map_uid, self.bin_dir, addBotOpponent=True, explored=True, fog=False)

        # Build first observation and info
        raw = self._get_raw_state()
        obs = self._state_to_observation(raw)
        self._last_obs = obs
        info = self._make_info(raw)
        return obs, info

    def step(self, action: Any):  # type: ignore[override]
        self._ensure_engine()
        # Accept dict/list (legacy) or MultiDiscrete ndarray (new)
        if isinstance(action, np.ndarray):
            self._execute_multidiscrete_action(action)
        elif isinstance(action, (list, tuple, dict)) or action is None:
            self.send_actions(action)
        else:
            # Unknown type: ignore
            pass

        api = self._openra['PythonAPI']
        for _ in range(self.ticks_per_step):
            api.Step()

        raw = self._get_raw_state()
        new_obs = self._state_to_observation(raw)

        reward = 0.0
        terminated = False
        truncated = False

        if self.max_episode_ticks is not None and new_obs['world_tick'] >= self.max_episode_ticks:
            truncated = True

        info = self._make_info(raw)
        self._last_obs = new_obs
        return new_obs, reward, terminated, truncated, info

    def render(self):  # type: ignore[override]
        return None

    def close(self):  # type: ignore[override]
        pass

    # --- Helpers ---

    def _get_raw_state(self) -> Dict[str, Any]:
        return build_observation(self._openra)

    def send_actions(self, actions: List[Dict[str, Any]]) -> None:
        send_actions(self._openra, actions)

    # --- MultiDiscrete helpers ---

    def _execute_multidiscrete_action(self, action: np.ndarray) -> None:
        atype_idx, unit_idx, tx, ty, target_idx, unit_type_idx = [int(x) for x in action.tolist()]
        atype = self.action_types[atype_idx] if 0 <= atype_idx < len(self.action_types) else 'noop'
        actions: List[Dict[str, Any]] = []

        if atype == 'move':
            # Move to cell
            actor_id = self._resolve_my_unit_id(unit_idx)
            actions.append({
                'order': 'Move',
                'subject': actor_id,
                'target_cell': (int(tx), int(ty)),
                'queued': False,
            })
        elif atype == 'attack':
            actor_id = self._resolve_my_unit_id(unit_idx)
            target_id = self._resolve_enemy_unit_id(target_idx)
            actions.append({
                'order': 'Attack',
                'subject': actor_id,
                'target_actor': target_id,
                'queued': False,
            })
        elif atype == 'deploy':
            actor_id = self._resolve_my_unit_id(unit_idx)
            actions.append({
                'order': 'DeployTransform',
                'subject': actor_id,
                'queued': False,
            })
        elif atype == 'produce':
            # Queue production at a producer (queue actor id)
            producer_id = self._resolve_queue_actor_id(unit_idx)
            unit_type_name = self.reverse_unit_types.get(int(unit_type_idx), 'e1')
            actions.append({
                'order': 'StartProduction',
                'subject': producer_id,
                'target_string': unit_type_name,
                'queued': True,
            })
        elif atype == 'build':
            # Place finished building items using PlaceBuilding order via PlayerActor
            try:
                player_actor_id = int(self._openra['Game'].OrderManager.World.LocalPlayer.PlayerActor.ActorID)
            except Exception:
                player_actor_id = self._resolve_my_unit_id(unit_idx)
            unit_type_name = self.reverse_unit_types.get(int(unit_type_idx), 'barr')
            actions.append({
                'order': 'PlaceBuilding',
                'subject': player_actor_id,
                'target_cell': (int(tx), int(ty)),
                'target_string': unit_type_name,
                'extra_data': int(self._resolve_queue_actor_id(unit_idx)),
                'queued': False,
            })
        elif atype == 'noop':
            pass

        # Deduplicate recent identical actions
        if actions:
            deduped: List[Dict[str, Any]] = []
            for a in actions:
                sig = self._action_signature(a)
                if not self._is_duplicate_action(sig):
                    deduped.append(a)
            actions = deduped

        if actions:
            self.send_actions(actions)

    def _action_signature(self, a: Dict[str, Any]) -> Tuple:
        return (
            a.get('order', ''),
            int(a.get('subject', -1)),
            int((a.get('target_cell') or ([-1, -1]))[0]) if a.get('target_cell') else -1,
            int((a.get('target_cell') or ([-1, -1]))[1]) if a.get('target_cell') else -1,
            int(a.get('target_actor', -1)),
        )

    def _is_duplicate_action(self, sig: Tuple) -> bool:
        now = int(self._openra['PythonAPI'].GetState().WorldTick)
        # prune by TTL in steps converted to ticks_per_step buckets
        self._recent_actions = deque([(s, t) for (s, t) in self._recent_actions if (now - t) <= (self.ticks_per_step * self._action_ttl_steps)], maxlen=256)
        for s, _ in self._recent_actions:
            if s == sig:
                return True
        return False

    def _record_action(self, sig: Tuple) -> None:
        now = int(self._openra['PythonAPI'].GetState().WorldTick)
        self._recent_actions.append((sig, now))

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


    # --- Observation helpers ---

    def _state_to_observation(self, raw: Dict[str, Any]):
        if self.observation_type == 'vector':
            return self._state_to_vector(raw)
        elif self.observation_type == 'image':
            return self._state_to_image(raw)
        return raw

    def _make_info(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        my_owner = int(raw.get('my_owner', -1))
        my_units = [u for u in (raw.get('actors') or []) if u.get('owner') == my_owner and not u.get('dead', False)]
        ally_units = []  # not available from RLState directly
        enemy_units = [u for u in (raw.get('actors') or []) if u.get('owner') != my_owner and not u.get('dead', False)]

        # Cache id lists for action resolution
        self._my_unit_ids = [int(u.get('id')) for u in my_units]
        self._enemy_unit_ids = [int(u.get('id')) for u in enemy_units]

        info = {
            'tick': int(raw.get('world_tick', 0)),
            'my_unit_count': len(my_units),
            'ally_unit_count': len(ally_units),
            'enemy_unit_count': len(enemy_units),
        }
        # Include production overview if available from raw
        info['production'] = raw.get('production', {}) or {}
        # Cache queue actor ids to resolve produce/build indices
        self._queue_actor_ids = []
        try:
            for q in (info['production'].get('Queues', []) or []):
                aid = q.get('ActorId')
                if isinstance(aid, int):
                    self._queue_actor_ids.append(aid)
        except Exception:
            self._queue_actor_ids = []
        # Include placeable areas map if present
        info['placeable_areas'] = raw.get('placeable_areas', {}) or {}
        info['action_mask'] = self._get_action_mask(my_units, enemy_units)
        return info

    def _get_action_mask(self, my_units: List[Dict[str, Any]], enemy_units: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        max_units = 100
        move_mask = np.zeros((max_units,), dtype=np.uint8)
        deploy_mask = np.zeros((max_units,), dtype=np.uint8)
        attack_mask = np.zeros((max_units, max_units), dtype=np.uint8)

        # Gate by per-unit available_orders from obs. Supported: Move, Attack, DeployTransform, PlaceBuilding
        for i, u in enumerate(my_units[:max_units]):
            orders = set(str(x).lower() for x in (u.get('available_orders') or []))
            if 'move' in orders:
                move_mask[i] = 1
            if 'deploytransform' in orders:
                deploy_mask[i] = 1

        if enemy_units:
            for i, u in enumerate(my_units[:max_units]):
                orders = set(str(x).lower() for x in (u.get('available_orders') or []))
                if 'attack' not in orders:
                    continue
                for j, _ in enumerate(enemy_units[:max_units]):
                    attack_mask[i, j] = 1

        # Start with all action types enabled then prune based on masks
        action_type_mask = np.ones((len(self.action_types),), dtype=np.uint8)

        def _disable_if_empty(name: str, cond: bool) -> None:
            if name in self.action_types and not cond:
                action_type_mask[self.action_types.index(name)] = 0

        _disable_if_empty('move', bool(move_mask.any()))
        _disable_if_empty('attack', bool(attack_mask.any()))
        _disable_if_empty('deploy', bool(deploy_mask.any()))

        mask = {
            'action_type': action_type_mask,
            'move_mask': move_mask,
            'attack_mask': attack_mask,
            'deploy_mask': deploy_mask,
        }
        if 'produce' in self.action_types:
            produce_mask = np.zeros((max_units,), dtype=np.uint8)
            n = min(len(getattr(self, '_queue_actor_ids', []) or []), max_units)
            if n > 0:
                produce_mask[:n] = 1
            mask['produce_mask'] = produce_mask
        if 'build' in self.action_types:
            build_mask = np.zeros((max_units,), dtype=np.uint8)
            # Enable only when PlaceBuilding is available on at least one unit
            has_place_order = any('placebuilding' in set(str(x).lower() for x in (u.get('available_orders') or [])) for u in my_units)
            # Enable indices if we have queues; precise cell-level validity is user logic
            n = min(len(getattr(self, '_queue_actor_ids', []) or []), max_units)
            if n > 0:
                build_mask[:n] = 1
            mask['build_mask'] = build_mask
            _disable_if_empty('build', bool(build_mask.any()) and has_place_order)
        return mask

    def _state_to_vector(self, raw: Dict[str, Any]) -> np.ndarray:
        max_units = 100
        actors = raw.get('actors') or []
        my_owner = int(raw.get('my_owner', -1))
        my_units = [u for u in actors if u.get('owner') == my_owner and not u.get('dead', False)]
        enemy_units = [u for u in actors if u.get('owner') != my_owner and not u.get('dead', False)]

        # Determine map size if available from any actor positions (fallback 128x128)
        map_width = max([int(u.get('cell_x', 0)) for u in actors] + [128])
        map_height = max([int(u.get('cell_y', 0)) for u in actors] + [128])
        map_width = max(1, map_width)
        map_height = max(1, map_height)

        obs = np.zeros(max_units * 6 + max_units * 5 + 7 + 2, dtype=np.float32)

        # My units
        for i, u in enumerate(my_units[:max_units]):
            idx = i * 6
            obs[idx:idx+6] = [
                int(u.get('id', 0)) / 1000.0,
                self.unit_types.get(str(u.get('type', '')).lower(), 0) / max(1, len(self.unit_types)),
                int(u.get('cell_x', 0)) / map_width,
                int(u.get('cell_y', 0)) / map_height,
                int(u.get('hp', 0)) / max(1, int(u.get('max_hp', 1))),
                1.0 if ('idle' in ' '.join(map(str, u.get('available_orders', []))).lower()) else 0.0,
            ]

        # Enemy units
        start_idx = max_units * 6
        for i, u in enumerate(enemy_units[:max_units]):
            idx = start_idx + i * 5
            obs[idx:idx+5] = [
                int(u.get('id', 0)) / 1000.0,
                self.unit_types.get(str(u.get('type', '')).lower(), 0) / max(1, len(self.unit_types)),
                int(u.get('cell_x', 0)) / map_width,
                int(u.get('cell_y', 0)) / map_height,
                int(u.get('hp', 0)) / max(1, int(u.get('max_hp', 1))),
            ]

        # Resources/Power placeholders (not available from RLState directly)
        resource_idx = max_units * 6 + max_units * 5
        obs[resource_idx:resource_idx+7] = [
            0.0,  # cash
            0.0,  # resource_fill
            0.0,  # power provided
            0.0,  # power drained
            1.0,  # power normal
            0.0,  # power low
            0.0,  # power critical
        ]

        map_idx = resource_idx + 7
        obs[map_idx:map_idx+2] = [map_width / 128.0, map_height / 128.0]
        return obs

    def _state_to_image(self, raw: Dict[str, Any]) -> np.ndarray:
        img = np.zeros((128, 128, 10), dtype=np.uint8)
        actors = raw.get('actors') or []
        # Fallback map size approximation
        max_x = max([int(u.get('cell_x', 0)) for u in actors] + [1])
        max_y = max([int(u.get('cell_y', 0)) for u in actors] + [1])
        scale_x = 128.0 / max(1, max_x)
        scale_y = 128.0 / max(1, max_y)

        my_owner = int(raw.get('my_owner', -1))
        for u in actors:
            x = int(int(u.get('cell_x', 0)) * scale_x)
            y = int(int(u.get('cell_y', 0)) * scale_y)
            if not (0 <= x < 128 and 0 <= y < 128):
                continue
            is_enemy = int(u.get('owner', -1)) != my_owner
            is_infantry = 'infantry' in str(u.get('type', '')).lower()
            if int(u.get('owner', -1)) == my_owner:
                img[y, x, 0 if is_infantry else 1] = 255
            elif is_enemy:
                img[y, x, 3 if is_infantry else 4] = 255

        # Resource density not available -> leave at zero
        return img

    def _resolve_my_unit_id(self, index: int) -> int:
        if not self._my_unit_ids:
            return int(index)
        i = int(max(0, min(index, len(self._my_unit_ids) - 1)))
        return self._my_unit_ids[i]

    def _resolve_enemy_unit_id(self, index: int) -> int:
        if not self._enemy_unit_ids:
            return int(index)
        i = int(max(0, min(index, len(self._enemy_unit_ids) - 1)))
        return self._enemy_unit_ids[i]

    def _resolve_queue_actor_id(self, index: int) -> int:
        ids = getattr(self, '_queue_actor_ids', None) or []
        if not ids:
            return self._resolve_my_unit_id(index)
        i = int(max(0, min(index, len(ids) - 1)))
        return ids[i]


# Convenience factory
def make_env(bin_dir: str, mod_id: str, map_uid: str, **kwargs) -> OpenRAEnv:
    return OpenRAEnv(bin_dir=bin_dir, mod_id=mod_id, map_uid=map_uid, **kwargs)

if __name__ == "__main__":
    # Keep the module free of side-effects; see scripts/ for runnable examples.
    pass
