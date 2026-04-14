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
        if self.observation_type not in ['feature', 'vector', 'image']:
            raise ValueError(f"Invalid observation type: {self.observation_type}")
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
        self._last_raw_state: Optional[Dict[str, Any]] = None
        self._my_unit_ids: List[int] = []
        self._enemy_unit_ids: List[int] = []
        self._recent_actions: deque[Tuple] = deque(maxlen=256)
        self._action_ttl_steps: int = 8
        self.auto_place: bool = False
        self._placement_spiral = self._build_placement_spiral(radius=6)
        # Reward shaping config and previous metrics snapshot
        self.reward_weights = {
            'unit': 0.5,
            'building': 1.0,
            'low_cash_penalty': 0.2,
            'min_cash': 1500.0,
            # Production shaping
            'produce_start': 0.05,           # small reward when a new item is queued
            'produce_cancel_penalty': 0.1,   # penalty multiplier when an in-progress item is canceled
            'produce_queue_threshold': 6,     # queue length after which start reward is damped
            'produce_queue_damp': 0.5,       # start reward scale when beyond threshold
        }
        self._prev_metrics: Dict[str, Optional[float]] = {
            'units': None,
            'buildings': None,
            'cash': None,

        }
        # Track previous production items to detect new starts and cancels
        self._prev_prod_items: Dict[Tuple[int, str], float] = {}

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
            api.StartLocalGame(self.mod_id, self.map_uid, self.bin_dir, addBotOpponent=False, explored=True, fog=False)

        # Build first observation and info
        raw = self._get_raw_state()
        self._last_raw_state = raw
        obs = self._state_to_observation(raw)
        self._last_obs = obs
        # Initialize previous metrics for reward shaping
        m = self._compute_development_metrics(raw)
        self._prev_metrics = {
            'units': float(m.get('units', 0.0)),
            'buildings': float(m.get('buildings', 0.0)),
            'cash': float(m['cash']) if m.get('cash') is not None else None,
        }
        # Initialize production item snapshot
        self._prev_prod_items = self._extract_production_items(raw)
        # Clear action dedup history so stale entries from the previous
        # episode (with higher world_tick) don't suppress actions in the
        # new episode where world_tick resets to 0.
        self._recent_actions.clear()
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
        self._last_raw_state = raw
        new_obs = self._state_to_observation(raw)

        # Reward shaping: encourage development and sufficient cash reserve
        reward = self._compute_reward(raw)
        terminated = False
        truncated = False

        if self.max_episode_ticks is not None and int(raw.get('world_tick', 0)) >= self.max_episode_ticks:
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

    def _get_current_world_tick(self) -> int:
        if self._last_raw_state is not None:
            try:
                return int(self._last_raw_state.get('world_tick', 0))
            except Exception:
                pass
        try:
            return int(self._openra['PythonAPI'].GetState().WorldTick)
        except Exception:
            return 0

    # --- Reward shaping helpers ---

    def _compute_development_metrics(self, raw: Dict[str, Any]) -> Dict[str, Optional[float]]:
        actors = raw.get('actors') or []
        my_owner = int(raw.get('my_owner', -1))
        my_alive = [u for u in actors if int(u.get('owner', -1)) == my_owner and not bool(u.get('dead', False))]
        units = 0
        buildings = 0
        for u in my_alive:
            orders = set(str(x).lower() for x in (u.get('available_orders') or []))
            # Heuristic: actors that can Move are units; others are buildings
            if 'move' in orders:
                units += 1
            else:
                buildings += 1
        cash = raw.get('cash')
        try:
            cash_val: Optional[float] = float(cash) if cash is not None else None
        except Exception:
            cash_val = None

        # Production progress
        prod = raw.get('production') or {}
        progress_sum = 0.0
        try:
            for q in (prod.get('Queues', []) or []):
                for it in (q.get('Items', []) or []):
                    progress_sum += float(it.get('Progress', 0))
        except Exception:
            progress_sum = 0.0

        # Exploration: count new visited cells this step (not part of metrics state; used in reward)
        return {
            'units': float(units),
            'buildings': float(buildings),
            'cash': cash_val,
        }

    def _compute_reward(self, raw: Dict[str, Any]) -> float:
        metrics = self._compute_development_metrics(raw)
        # Initialize if first call
        if self._prev_metrics['units'] is None:
            self._prev_metrics['units'] = float(metrics.get('units') or 0.0)
        if self._prev_metrics['buildings'] is None:
            self._prev_metrics['buildings'] = float(metrics.get('buildings') or 0.0)
        if self._prev_metrics['cash'] is None and metrics.get('cash') is not None:
            self._prev_metrics['cash'] = float(metrics.get('cash') or 0.0)

        du = float(metrics.get('units') or 0.0) - float(self._prev_metrics.get('units') or 0.0)
        db = float(metrics.get('buildings') or 0.0) - float(self._prev_metrics.get('buildings') or 0.0)

        # Reward for new units/buildings
        rw = self.reward_weights['unit'] * du + self.reward_weights['building'] * db

        # Production shaping: reward new starts (small), penalize cancels by progress
        starts, cancels = self._diff_production_events(raw)
        if starts > 0:
            rw += self._production_start_reward(raw, starts)
        if cancels > 0:
            rw -= self._production_cancel_penalty(cancels)

        # Penalty for low cash (only if cash is reported by engine)
        cash_now = metrics.get('cash')
        if cash_now is not None:
            min_cash = float(self.reward_weights['min_cash'])
            if cash_now < min_cash:
                # Scale penalty by deficit ratio
                deficit = (min_cash - float(cash_now)) / max(1.0, min_cash)
                rw -= self.reward_weights['low_cash_penalty'] * float(deficit)

        # Update previous metrics
        self._prev_metrics['units'] = float(metrics.get('units') or 0.0)
        self._prev_metrics['buildings'] = float(metrics.get('buildings') or 0.0)
        if cash_now is not None:
            self._prev_metrics['cash'] = float(cash_now)

        # Update production snapshot
        self._prev_prod_items = self._extract_production_items(raw)

        return float(rw)

    def _extract_production_items(self, raw: Dict[str, Any]) -> Dict[Tuple[int, str], float]:
        items: Dict[Tuple[int, str], float] = {}
        prod = raw.get('production') or {}
        try:
            for q in (prod.get('Queues', []) or []):
                q_actor = int(q.get('ActorId', -1))
                for it in (q.get('Items', []) or []):
                    name = str(it.get('Item') or it.get('Name') or '')
                    if not name:
                        continue
                    prog = float(it.get('Progress', 0.0))
                    done = bool(it.get('Done', False))
                    if done:
                        # Completed items are not tracked as active production
                        continue
                    items[(q_actor, name)] = prog
        except Exception:
            return {}
        return items

    def _diff_production_events(self, raw: Dict[str, Any]) -> Tuple[int, int]:
        """Return (num_starts, weighted_cancels), where cancels is the count weighted by progress fraction."""
        prev = self._prev_prod_items or {}
        cur = self._extract_production_items(raw)
        # New starts: items present now but not before
        starts = 0
        for key in cur.keys():
            if key not in prev:
                starts += 1
        # Cancels: items that disappeared and were not done; weight by last progress fraction
        cancels_weighted = 0.0
        for key, prog in prev.items():
            if key not in cur:
                cancels_weighted += max(0.0, float(prog) / 100.0)
        return int(starts), int(round(cancels_weighted * 1000))  # return scaled int; actual penalty computed later

    def _production_start_reward(self, raw: Dict[str, Any], starts: int) -> float:
        base = float(self.reward_weights.get('produce_start', 0.0))
        if base <= 0.0 or starts <= 0:
            return 0.0
        # Dampen when queue is long to discourage overproduction
        prod = raw.get('production') or {}
        q_len = 0
        try:
            for q in (prod.get('Queues', []) or []):
                q_len += len(q.get('Items', []) or [])
        except Exception:
            q_len = 0
        thresh = int(self.reward_weights.get('produce_queue_threshold', 6))
        damp = float(self.reward_weights.get('produce_queue_damp', 0.5)) if q_len >= thresh else 1.0
        # Scale by cash ratio to avoid rewarding when broke
        cash = self._prev_metrics.get('cash')
        min_cash = float(self.reward_weights.get('min_cash', 0.0))
        cash_scale = 1.0
        if cash is not None and min_cash > 0.0:
            cash_scale = max(0.1, min(1.0, float(cash) / min_cash))
        return starts * base * damp * cash_scale

    def _production_cancel_penalty(self, cancels_scaled: int) -> float:
        # Convert back to progress-weighted count
        cancels_weighted = float(cancels_scaled) / 1000.0
        coef = float(self.reward_weights.get('produce_cancel_penalty', 0.0))
        return coef * cancels_weighted

    def send_actions(self, actions: List[Dict[str, Any]]) -> None:
        send_actions(self._openra, actions)

    # --- MultiDiscrete helpers ---

    def _execute_multidiscrete_action(self, action: np.ndarray) -> None:
        atype_idx, unit_idx, tx, ty, target_idx, unit_type_idx = [int(x) for x in action.tolist()]
        atype = self.action_types[atype_idx] if 0 <= atype_idx < len(self.action_types) else 'noop'
        actions: List[Dict[str, Any]] = []

        # Auto-place completed production items BEFORE the agent's action.
        # This prevents Done items from being overwritten by new produce orders.
        if self.auto_place:
            actions.extend(self._auto_place_done_items())

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
            # Place finished building item using PlaceBuilding order.
            # Subject and ExtraData both use the queue actor ID, matching
            # RuleBasedAgent and BaseBuilderQueueManager.
            queue_actor_id = self._resolve_queue_actor_id(unit_idx)
            unit_type_name = self.reverse_unit_types.get(int(unit_type_idx), 'powr')
            actions.append({
                'order': 'PlaceBuilding',
                'subject': int(queue_actor_id),
                'target_cell': (int(tx), int(ty)),
                'target_string': unit_type_name,
                'extra_data': int(queue_actor_id),
                'queued': False,
            })
        elif atype == 'noop':
            pass

        # Debug logging (controlled by env attribute)
        if getattr(self, '_debug_actions', False) and atype not in ('noop',):
            import sys
            for a in actions:
                order = a.get('order', '')
                subj = a.get('subject', '?')
                tgt = a.get('target_string', '') or a.get('target_cell', '')
                print(f"[action] {atype} → {order} subj={subj} tgt={tgt}", file=sys.stderr)

        # Deduplicate recent identical actions
        if actions:
            deduped: List[Dict[str, Any]] = []
            for a in actions:
                sig = self._action_signature(a)
                if not self._is_duplicate_action(sig):
                    deduped.append(a)
                elif getattr(self, '_debug_actions', False):
                    import sys
                    print(f"[dedup] BLOCKED {a.get('order','')} tgt={a.get('target_string','')}", file=sys.stderr)
            actions = deduped

        if actions:
            if getattr(self, '_debug_actions', False):
                import sys
                for a in actions:
                    print(f"[send] {a.get('order','')} subj={a.get('subject','?')} "
                          f"tgt={a.get('target_string','') or a.get('target_cell','')}",
                          file=sys.stderr)
            self.send_actions(actions)
            for a in actions:
                self._record_action(self._action_signature(a))

    def _action_signature(self, a: Dict[str, Any]) -> Tuple:
        return (
            a.get('order', ''),
            int(a.get('subject', -1)),
            int((a.get('target_cell') or ([-1, -1]))[0]) if a.get('target_cell') else -1,
            int((a.get('target_cell') or ([-1, -1]))[1]) if a.get('target_cell') else -1,
            int(a.get('target_actor', -1)),
            str(a.get('target_string', '')),
        )

    def _is_duplicate_action(self, sig: Tuple) -> bool:
        now = self._get_current_world_tick()
        # prune by TTL in steps converted to ticks_per_step buckets
        self._recent_actions = deque([(s, t) for (s, t) in self._recent_actions if (now - t) <= (self.ticks_per_step * self._action_ttl_steps)], maxlen=256)
        for s, _ in self._recent_actions:
            if s == sig:
                return True
        return False

    def _record_action(self, sig: Tuple) -> None:
        now = self._get_current_world_tick()
        self._recent_actions.append((sig, now))

    # --- Auto-placement helpers ---

    @staticmethod
    def _build_placement_spiral(radius: int = 6) -> List[Tuple[int, int]]:
        """Offsets sorted by Chebyshev distance from origin."""
        offsets: List[Tuple[int, int]] = []
        for r in range(1, radius + 1):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if max(abs(dx), abs(dy)) == r:
                        offsets.append((dx, dy))
        return offsets

    def _find_base_position(self) -> Optional[Tuple[int, int]]:
        """Return (cell_x, cell_y) of the first alive building we own."""
        raw = self._last_raw_state
        if raw is None:
            return None
        my_owner = int(raw.get("my_owner", -1))
        for u in raw.get("actors") or []:
            if int(u.get("owner", -1)) != my_owner:
                continue
            if bool(u.get("dead", False)):
                continue
            orders = {str(x).lower() for x in (u.get("available_orders") or [])}
            if "move" not in orders:
                return (int(u.get("cell_x", 0)), int(u.get("cell_y", 0)))
        return None

    def _auto_place_done_items(self) -> List[Dict[str, Any]]:
        """Generate PlaceBuilding orders for all Done production items."""
        raw = self._last_raw_state
        if raw is None:
            return []

        prod = raw.get("production") or {}
        queues = prod.get("Queues") or []
        if not queues:
            return []

        base_pos = self._find_base_position()
        if base_pos is None:
            return []

        try:
            api = self._openra['Game']
            player_actor_id = int(
                api.OrderManager.World.LocalPlayer.PlayerActor.ActorID
            )
        except Exception:
            return []

        place_orders: List[Dict[str, Any]] = []
        for q in queues:
            q_id = int(q.get("ActorId", -1))
            for it in q.get("Items") or []:
                if not bool(it.get("Done", False)):
                    continue
                name = str(it.get("Item") or it.get("Name") or "")
                if not name:
                    continue
                # Try positions in a spiral around the base.
                # Send several candidates; the engine accepts the first
                # valid one and ignores the rest.
                for dx, dy in self._placement_spiral[:12]:
                    cx = base_pos[0] + dx
                    cy = base_pos[1] + dy
                    place_orders.append({
                        "order": "PlaceBuilding",
                        "subject": player_actor_id,
                        "target_cell": (cx, cy),
                        "target_string": name,
                        "extra_data": q_id,
                        "queued": False,
                    })
        return place_orders

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

    def _split_visible_actors(self, raw: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        actors = raw.get('actors') or []
        my_owner = int(raw.get('my_owner', -1))
        my_units = [u for u in actors if int(u.get('owner', -1)) == my_owner and not bool(u.get('dead', False))]
        enemy_units = [u for u in actors if int(u.get('owner', -1)) != my_owner and not bool(u.get('dead', False))]
        return my_units, enemy_units

    def _estimate_map_size(self, raw: Dict[str, Any]) -> Tuple[int, int]:
        xs = [128]
        ys = [128]
        for u in raw.get('actors') or []:
            xs.append(int(u.get('cell_x', 0)) + 1)
            ys.append(int(u.get('cell_y', 0)) + 1)
        for r in raw.get('resources') or []:
            xs.append(int(r.get('cell_x', 0)) + 1)
            ys.append(int(r.get('cell_y', 0)) + 1)
        return max(1, max(xs)), max(1, max(ys))

    def _economy_features(self, raw: Dict[str, Any]) -> List[float]:
        cash = max(0.0, float(raw.get('cash', 0) or 0.0))
        resources_total = max(0.0, float(raw.get('resources_total', 0) or 0.0))
        resource_capacity = max(0.0, float(raw.get('resource_capacity', 0) or 0.0))
        power = raw.get('power') or {}
        power_provided = max(0.0, float(power.get('provided', 0) or 0.0))
        power_drained = max(0.0, float(power.get('drained', 0) or 0.0))
        power_state = str(power.get('state', '') or '').lower()

        cash_norm = min(cash / 10000.0, 1.0)
        resource_fill = min(resources_total / max(1.0, resource_capacity), 1.0)
        power_provided_norm = min(power_provided / 500.0, 1.0)
        power_drained_norm = min(power_drained / 500.0, 1.0)
        power_is_critical = 1.0 if 'critical' in power_state else 0.0
        power_is_low = 1.0 if 'low' in power_state else 0.0
        power_is_normal = 1.0 if power_is_low == 0.0 and power_is_critical == 0.0 else 0.0

        return [
            cash_norm,
            resource_fill,
            power_provided_norm,
            power_drained_norm,
            power_is_normal,
            power_is_low,
            power_is_critical,
        ]

    @staticmethod
    def _actor_health_ratio(actor: Dict[str, Any]) -> float:
        return float(int(actor.get('hp', 0)) / max(1, int(actor.get('max_hp', 1))))

    @staticmethod
    def _looks_like_infantry(actor: Dict[str, Any]) -> bool:
        actor_type = str(actor.get('type', '')).lower()
        return actor_type.startswith('e') or 'infantry' in actor_type or actor_type in {'dog', 'spy', 'thf'}

    @staticmethod
    def _project_to_image(cell_x: int, cell_y: int, map_width: int, map_height: int) -> Tuple[int, int]:
        x = min(127, max(0, int(cell_x * (127.0 / max(1, map_width - 1)))))
        y = min(127, max(0, int(cell_y * (127.0 / max(1, map_height - 1)))))
        return x, y

    @staticmethod
    def _scalar_channel(value: float) -> np.uint8:
        return np.uint8(np.clip(round(value * 255.0), 0, 255))

    def _make_info(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        my_units, enemy_units = self._split_visible_actors(raw)
        ally_units = []  # not available from RLState directly

        # Cache id lists for action resolution
        self._my_unit_ids = [int(u.get('id')) for u in my_units]
        self._enemy_unit_ids = [int(u.get('id')) for u in enemy_units]

        info = {
            'tick': int(raw.get('world_tick', 0)),
            'my_unit_count': len(my_units),
            'ally_unit_count': len(ally_units),
            'enemy_unit_count': len(enemy_units),
        }
        info["actors"] = [
            actor["type"] + str(actor["id"]) for actor in raw.get("actors", [])
        ]
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
        info['action_mask'] = self._get_action_mask(raw, my_units, enemy_units)
        return info

    def _get_action_mask(
        self, raw: Dict[str, Any], my_units: List[Dict[str, Any]], enemy_units: List[Dict[str, Any]]
    ) -> Dict[str, np.ndarray]:
        max_units = 100
        move_mask = np.zeros((max_units,), dtype=np.uint8)
        deploy_mask = np.zeros((max_units,), dtype=np.uint8)
        attack_mask = np.zeros((max_units, max_units), dtype=np.uint8)
        produce_queue_mask = np.zeros((max_units,), dtype=np.uint8)
        build_mask = np.zeros((max_units,), dtype=np.uint8)
        target_idx_mask = np.zeros((max_units,), dtype=np.uint8)
        target_x_mask = np.zeros((max_units, 128), dtype=np.uint8)
        target_y_mask = np.zeros((max_units, 128), dtype=np.uint8)
        production = raw.get('production') or {}
        queues = production.get('Queues') or []

        # Gate by per-unit available_orders from obs and, when available, engine-side
        # order feasibility checks to better match what the game will actually accept.
        for i, u in enumerate(my_units[:max_units]):
            orders = set(str(x).lower() for x in (u.get('available_orders') or []))
            if 'move' in orders and self._unit_has_feasible_move(u, fallback_to_orders=True):
                move_mask[i] = 1
                self._fill_move_target_masks(target_x_mask[i], target_y_mask[i], u)
            # HACK: Allow deploy for non-building actors (e.g. MCV→fact).
            # Exclude known building types from undeploying because the
            # current agent spams deploy and constantly cancels in-progress
            # production.  In a real game undeploying to relocate is a valid
            # strategy; this restriction should be removed once the reward
            # signal is strong enough for the agent to learn the cost of
            # interrupting production on its own.
            # TODO: Remove this building-type guard and let the agent learn via reward.
            if 'deploytransform' in orders and self._unit_can_deploy(u):
                unit_type = str(u.get('type', '')).lower()
                # Block deploy for buildings that would lose production queues
                _building_types = {'fact', 'afld', 'weap', 'tent', 'barr', 'spen', 'syrd'}
                if unit_type not in _building_types:
                    deploy_mask[i] = 1

        if enemy_units:
            for i, u in enumerate(my_units[:max_units]):
                orders = set(str(x).lower() for x in (u.get('available_orders') or []))
                if 'attack' not in orders:
                    continue
                for j, enemy in enumerate(enemy_units[:max_units]):
                    if self._check_order_feasibility(
                        int(u.get('id', -1)),
                        'Attack',
                        target_type='Actor',
                        target_actor_id=int(enemy.get('id', -1)),
                    ):
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
        queue_infos: List[Dict[str, Any]] = []
        for q in queues:
            try:
                queue_infos.append({
                    'actor_id': int(q.get('ActorId', -1)),
                    'type': str(q.get('Type', '')).lower(),
                    'enabled': bool(q.get('Enabled', False)),
                    'items': list(q.get('Items') or []),
                    'producible': list(q.get('Producible') or []),
                })
            except Exception:
                continue

        # Identify queue types (categories) that already have items.
        # E.g. if the "Building" queue has an in-progress/Done item, all
        # produce actions targeting building-type units should be blocked,
        # even if a different queue (like "Defense") on the same actor is
        # empty.  This prevents the engine from routing a building-type
        # StartProduction to an occupied building queue.
        occupied_queue_types: set = set()
        for qi in queue_infos:
            if qi['enabled'] and len(qi['items']) >= 1 and qi['type']:
                occupied_queue_types.add(qi['type'])

        if 'produce' in self.action_types:
            # Producible unit types mask using global catalog if available
            produce_unit_type_mask = np.zeros((len(self.unit_types),), dtype=np.uint8)
            allowed_names = set()
            try:
                catalog = raw.get('producible_catalog') or []
                for b in catalog:
                    nm = str(b.get('Name') or '').lower()
                    if nm:
                        allowed_names.add(nm)
            except Exception:
                allowed_names = set()

            # Build a set of unit type names whose queues are ALL occupied.
            # A unit type is blocked if every queue that can produce it has
            # its queue-type in occupied_queue_types.
            blocked_unit_names: set = set()
            if occupied_queue_types:
                for nm in allowed_names:
                    # Find all queues that list this item as producible
                    all_occupied = True
                    found_queue = False
                    for qi in queue_infos:
                        if not qi['enabled']:
                            continue
                        producible_names = {str(it.get('Name', '')).lower() for it in qi['producible']}
                        if nm not in producible_names:
                            continue
                        found_queue = True
                        if qi['type'] not in occupied_queue_types:
                            all_occupied = False
                            break
                    if found_queue and all_occupied:
                        blocked_unit_names.add(nm)

            if allowed_names:
                for idx, name in self.reverse_unit_types.items():
                    nm = str(name).lower()
                    if nm in allowed_names and nm not in blocked_unit_names:
                        i = int(idx)
                        if 0 <= i < produce_unit_type_mask.shape[0]:
                            produce_unit_type_mask[i] = 1
            queue_actor_ids = list(getattr(self, '_queue_actor_ids', []) or [])
            for idx, actor_id in enumerate(queue_actor_ids[:max_units]):
                info = next((q for q in queue_infos if q['actor_id'] == int(actor_id)), None)
                if info is None or not info['enabled']:
                    continue
                # Block this queue if it has items OR its type is occupied
                if len(info['items']) >= 1:
                    continue
                if info['type'] in occupied_queue_types:
                    continue
                if allowed_names:
                    producible_names = {str(it.get('Name', '')).lower() for it in info['producible']}
                    if not (allowed_names & producible_names):
                        continue
                produce_queue_mask[idx] = 1
            mask['produce_queue_mask'] = produce_queue_mask
            mask['produce_unit_type_mask'] = produce_unit_type_mask

            _disable_if_empty('produce', bool(produce_unit_type_mask.any()) and bool(produce_queue_mask.any()))
        build_unit_type_mask = np.zeros((len(self.unit_types),), dtype=np.uint8)
        if 'build' in self.action_types:
            # PlaceBuilding is issued via queue actor, not regular units, so
            # we no longer gate on has_place_order (it was always False).
            queue_actor_ids = list(getattr(self, '_queue_actor_ids', []) or [])
            placeable_names = {str(k).lower() for k in (raw.get('placeable_areas') or {}).keys()}

            # Collect names of Done items so build_unit_type_mask only enables
            # types the agent can actually place right now.
            done_item_names: set = set()
            for qi in queue_infos:
                if not qi['enabled']:
                    continue
                for it in qi['items']:
                    if bool(it.get('Done', False)):
                        nm = str(it.get('Item') or it.get('Name') or '').lower()
                        if nm:
                            done_item_names.add(nm)

            # Only enable placeable cells for Done item types.
            buildable_names = done_item_names & placeable_names

            for idx, actor_id in enumerate(queue_actor_ids[:max_units]):
                info = next((q for q in queue_infos if q['actor_id'] == int(actor_id)), None)
                if info is None or not info['enabled']:
                    continue
                has_done_item = any(bool(it.get('Done', False)) for it in info['items'])
                if not has_done_item:
                    continue
                if not buildable_names:
                    continue
                build_mask[idx] = 1
                # Only populate target cells for the Done item types
                for unit_type, cells in (raw.get('placeable_areas') or {}).items():
                    if str(unit_type).lower() not in done_item_names:
                        continue
                    if not cells:
                        continue
                    for cell in cells:
                        try:
                            cx, cy = int(cell[0]), int(cell[1])
                        except Exception:
                            continue
                        if 0 <= cx < 128:
                            target_x_mask[idx, cx] = 1
                        if 0 <= cy < 128:
                            target_y_mask[idx, cy] = 1
            mask['build_mask'] = build_mask
            for idx, name in self.reverse_unit_types.items():
                if str(name).lower() in buildable_names:
                    build_unit_type_mask[int(idx)] = 1
            _disable_if_empty('build', bool(build_mask.any()) and bool(build_unit_type_mask.any()))

        if enemy_units:
            target_idx_mask[:min(len(enemy_units), max_units)] = 1

        unit_idx_mask = np.maximum.reduce([
            move_mask,
            deploy_mask,
            produce_queue_mask,
            build_mask,
            attack_mask.any(axis=1).astype(np.uint8),
        ])

        unit_type_mask = np.zeros((len(self.unit_types),), dtype=np.uint8)
        if 'produce_unit_type_mask' in mask:
            unit_type_mask = np.maximum(unit_type_mask, mask['produce_unit_type_mask'])
        if 'build' in self.action_types:
            unit_type_mask = np.maximum(unit_type_mask, build_unit_type_mask)
            mask['build_unit_type_mask'] = build_unit_type_mask

        mask['unit_idx'] = unit_idx_mask
        mask['target_idx'] = target_idx_mask
        mask['target_x'] = target_x_mask
        mask['target_y'] = target_y_mask
        mask['unit_type'] = unit_type_mask
        return mask

    def _check_order_feasibility(
        self,
        subject_actor_id: int,
        order_id: str,
        target_type: str = 'None',
        cell_x: int = 0,
        cell_y: int = 0,
        target_actor_id: int = 0,
        force_attack: bool = False,
        force_queue: bool = False,
        force_move: bool = False,
    ) -> bool:
        try:
            api = self._openra['PythonAPI']
            cell_bits = 0
            if target_type.lower() == 'cell':
                cell_bits = int(self._openra['CPos'](int(cell_x), int(cell_y)).Bits)
            return bool(api.CheckOrderFeasibility(
                int(subject_actor_id),
                str(order_id),
                str(target_type),
                int(cell_bits),
                0,
                int(target_actor_id),
                bool(force_attack),
                bool(force_queue),
                bool(force_move),
            ))
        except Exception:
            return False

    def _unit_has_feasible_move(self, unit: Dict[str, Any], fallback_to_orders: bool = False) -> bool:
        subject_id = int(unit.get('id', -1))
        ux = int(unit.get('cell_x', 0))
        uy = int(unit.get('cell_y', 0))
        for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0), (1, -1), (-1, 1)):
            tx = max(0, ux + dx)
            ty = max(0, uy + dy)
            if self._check_order_feasibility(
                subject_id,
                'Move',
                target_type='Cell',
                cell_x=tx,
                cell_y=ty,
                force_move=True,
            ):
                return True
        if fallback_to_orders:
            orders = set(str(x).lower() for x in (unit.get('available_orders') or []))
            return 'move' in orders
        return False

    def _unit_can_deploy(self, unit: Dict[str, Any]) -> bool:
        subject_id = int(unit.get('id', -1))
        if self._check_order_feasibility(
            subject_id,
            'DeployTransform',
            target_type='None',
        ):
            return True
        orders = set(str(x).lower() for x in (unit.get('available_orders') or []))
        return 'deploytransform' in orders

    def _fill_move_target_masks(self, target_x_mask: np.ndarray, target_y_mask: np.ndarray, unit: Dict[str, Any]) -> None:
        subject_id = int(unit.get('id', -1))
        ux = int(unit.get('cell_x', 0))
        uy = int(unit.get('cell_y', 0))
        found_any = False
        for dx in range(-4, 5):
            for dy in range(-4, 5):
                if dx == 0 and dy == 0:
                    continue
                tx = max(0, min(127, ux + dx))
                ty = max(0, min(127, uy + dy))
                if self._check_order_feasibility(
                    subject_id,
                    'Move',
                    target_type='Cell',
                    cell_x=tx,
                    cell_y=ty,
                    force_move=True,
                ):
                    target_x_mask[tx] = 1
                    target_y_mask[ty] = 1
                    found_any = True

        # Fallback to a small local window so masks stay non-empty even if strict
        # feasibility probing fails transiently during remote synchronization.
        if not found_any:
            for tx in range(max(0, ux - 2), min(128, ux + 3)):
                target_x_mask[tx] = 1
            for ty in range(max(0, uy - 2), min(128, uy + 3)):
                target_y_mask[ty] = 1

    def _state_to_vector(self, raw: Dict[str, Any]) -> np.ndarray:
        max_units = 100
        my_units, enemy_units = self._split_visible_actors(raw)
        map_width, map_height = self._estimate_map_size(raw)
        unit_type_scale = max(1, len(self.unit_types))

        obs = np.zeros(max_units * 6 + max_units * 5 + 7 + 2, dtype=np.float32)

        # My units
        for i, u in enumerate(my_units[:max_units]):
            idx = i * 6
            obs[idx:idx+6] = [
                int(u.get('id', 0)) / 1000.0,
                self.unit_types.get(str(u.get('type', '')).lower(), 0) / unit_type_scale,
                int(u.get('cell_x', 0)) / map_width,
                int(u.get('cell_y', 0)) / map_height,
                self._actor_health_ratio(u),
                1.0 if ('idle' in ' '.join(map(str, u.get('available_orders', []))).lower()) else 0.0,
            ]

        # Enemy units
        start_idx = max_units * 6
        for i, u in enumerate(enemy_units[:max_units]):
            idx = start_idx + i * 5
            obs[idx:idx+5] = [
                int(u.get('id', 0)) / 1000.0,
                self.unit_types.get(str(u.get('type', '')).lower(), 0) / unit_type_scale,
                int(u.get('cell_x', 0)) / map_width,
                int(u.get('cell_y', 0)) / map_height,
                self._actor_health_ratio(u),
            ]

        # Economy and power
        resource_idx = max_units * 6 + max_units * 5
        obs[resource_idx:resource_idx+7] = self._economy_features(raw)

        map_idx = resource_idx + 7
        obs[map_idx:map_idx+2] = [map_width / 128.0, map_height / 128.0]
        return obs

    def _state_to_image(self, raw: Dict[str, Any]) -> np.ndarray:
        img = np.zeros((128, 128, 10), dtype=np.uint8)
        my_units, enemy_units = self._split_visible_actors(raw)
        map_width, map_height = self._estimate_map_size(raw)

        resource_cells = raw.get('resources') or []
        max_resource_density = max([int(r.get('density', 0)) for r in resource_cells] + [1])
        for r in resource_cells:
            x, y = self._project_to_image(int(r.get('cell_x', 0)), int(r.get('cell_y', 0)), map_width, map_height)
            density = int(r.get('density', 0))
            value = int(np.clip(round(255.0 * density / max_resource_density), 0, 255))
            img[y, x, 5] = max(img[y, x, 5], value)

        for actors, base_channel, low_hp_channel in ((my_units, 0, 8), (enemy_units, 3, 9)):
            for u in actors:
                x, y = self._project_to_image(int(u.get('cell_x', 0)), int(u.get('cell_y', 0)), map_width, map_height)
                channel = base_channel if self._looks_like_infantry(u) else base_channel + 1
                img[y, x, channel] = 255
                if self._actor_health_ratio(u) < 0.5:
                    img[y, x, low_hp_channel] = 255

        econ = self._economy_features(raw)
        power_balance = min(
            max((float((raw.get('power') or {}).get('provided', 0) or 0) - float((raw.get('power') or {}).get('drained', 0) or 0)) / 500.0, 0.0),
            1.0,
        )
        img[:, :, 6] = self._scalar_channel(power_balance)
        img[:, :, 7] = self._scalar_channel(econ[0])
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
