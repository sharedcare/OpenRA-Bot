"""
OpenRA Environment for Reinforcement Learning
Provides Gym-compatible interface for training RL agents on OpenRA
"""

import numpy as np
import requests
import websocket
import json
import threading
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque
import gymnasium as gym
from gymnasium import spaces


@dataclass
class ActorInfo:
    """Information about a game actor (unit/building)"""
    id: int
    type: str
    x: int
    y: int
    health: int
    max_health: int
    is_idle: bool = True


@dataclass
class ResourceCell:
    """Information about a resource cell on the map"""
    x: int
    y: int
    type: str
    density: int


@dataclass
class GameState:
    """Complete game state information"""
    tick: int
    my_units: List[ActorInfo]
    enemy_units: List[ActorInfo]
    cash: int
    resources: int
    resource_capacity: int
    power_provided: int
    power_drained: int
    power_state: str
    map_width: int
    map_height: int
    ally_units: List[ActorInfo] = field(default_factory=list)
    resource_cells: List[ResourceCell] = field(default_factory=list)
    production: Dict = field(default_factory=dict)
    placeable_areas: Dict[str, List[Tuple[int, int]]] = field(default_factory=dict)


class OpenRAEnvironment(gym.Env):
    """
    OpenRA Gymnasium Environment for RL training
    
    Action Space:
    - Type 0: Move unit (unit_id, target_x, target_y)
    - Type 1: Attack target (unit_id, target_id)
    - Type 2: Produce unit (building_id, unit_type)
    - Type 3: Build structure (builder_id, structure_type, target_x, target_y)
    """
    
    def __init__(self, 
                 api_host: str = "localhost",
                 api_port: int = 8081,
                 ws_port: int = 8081,
                 max_episode_steps: int = 5000,
                 observation_type: str = "vector",
                 enable_actions: Optional[List[str]] = None):
        
        super().__init__()
        
        self.api_host = api_host
        self.api_port = api_port
        self.ws_port = ws_port
        self.max_episode_steps = max_episode_steps
        self.observation_type = observation_type
        # Configure enabled actions (reduced set by default)
        # Include a 'noop' that performs no action, useful while waiting for production to complete
        default_actions = ['noop', 'move', 'attack', 'deploy']
        self.action_types: List[str] = enable_actions if enable_actions else default_actions
        
        # API endpoints
        self.api_base = f"http://{api_host}:{api_port}/api"
        self.ws_url = f"ws://{api_host}:{ws_port}"
        
        # Game state
        self.current_state: Optional[GameState] = None
        self.step_count = 0
        self.episode_reward = 0
        self.previous_enemy_count = 0
        self.previous_my_count = 0
        
        # Real-time update queue
        self.ws = None
        self.ws_messages = deque(maxlen=1000)
        self.ws_connected = False
        
        # Unit type mappings (game-specific)
        # Include RA actor codes needed by build/produce API (powr, proc, tent/barr, fact, mcv, e1)
        # Additional entries remain for generic categories; unknown types default to 0
        self.unit_types = {
            # Buildings
            'powr': 0,        # Power Plant
            'proc': 1,        # Refinery
            'tent': 2,        # Allied Barracks (RA)
            'barr': 3,        # Soviet Barracks (RA)
            'fact': 4,        # Construction Yard
            # Vehicles / Units
            'mcv': 5,
            'e1': 6,          # Rifle Infantry (RA)
            # Generic fallbacks (kept for compatibility in observations)
            'infantry': 7,
            'tank': 8,
            'artillery': 9,
            'aircraft': 10,
            'barracks': 11,
            'factory': 12,
            'power_plant': 13,
            'refinery': 14,
        }
        self.reverse_unit_types = {v: k for k, v in self.unit_types.items()}
        
        # Action and observation spaces (depends on unit_types)
        self._setup_spaces()
        
        # ID mappings between action indices and actual ActorId
        self._my_unit_ids = []
        self._enemy_unit_ids = []
        self._queue_actor_ids = []
        # Recent action deduplication (avoid re-sending identical actions while animations complete)
        self._recent_actions = deque(maxlen=256)
        self._action_ttl_steps = 8  # skip identical actions for this many env steps
        
        # Connect to game
        self._connect()
    
    def _setup_spaces(self):
        """Setup action and observation spaces"""
        
        # Action space: [action_type, unit_id, target_x, target_y, target_id, unit_type]
        # We'll use discrete actions for simplicity
        max_units = 100
        max_coord = 128
        max_unit_types = len(self.unit_types)
        
        self.action_space = spaces.MultiDiscrete([
            len(self.action_types),  # action_type: 0=move, 1=attack, 2=produce, 3=build, 4=deploy
            max_units,  # unit_id
            max_coord,  # target_x
            max_coord,  # target_y  
            max_units,  # target_id (for attacks)
            max_unit_types  # unit_type (for production)
        ])
        
        # Observation space depends on observation type
        if self.observation_type == "vector":
            # Vector observation: [my_units, enemy_units, resources, map_info]
            obs_dim = (
                max_units * 6 +  # my_units (id, type, x, y, health, is_idle)
                max_units * 5 +  # enemy_units (id, type, x, y, health)
                7 +              # resources and power
                2                # map dimensions
            )
            self.observation_space = spaces.Box(
                low=-1, high=max_coord, shape=(obs_dim,), dtype=np.float32
            )
        
        elif self.observation_type == "image":
            # Image-based observation: multi-channel feature map for model input
            # Channels:
            # 0: my infantry, 1: my vehicles/others, 2: allies (all types),
            # 3: enemy infantry, 4: enemy vehicles/others,
            # 5: resource density (grayscale),
            # 6: power surplus (global, repeated),
            # 7: cash (global, repeated),
            # 8: my low-health mask (<50%), 9: enemy low-health mask (<50%)
            self.observation_space = spaces.Box(
                low=0, high=255,
                shape=(128, 128, 10),
                dtype=np.uint8
            )
    
    def _connect(self):
        """Connect to OpenRA API and WebSocket"""
        try:
            # Test API connection
            response = requests.get(f"{self.api_base}/gamestate", timeout=5)
            if response.status_code == 200:
                print("✅ Connected to OpenRA API")
            else:
                raise ConnectionError("Failed to connect to OpenRA API")
                
            # Start long-poll stream for real-time updates
            self._start_stream_thread()
            
        except Exception as e:
            raise ConnectionError(f"Failed to connect to OpenRA: {e}")
    
    def _setup_websocket(self):
        """Setup WebSocket connection for real-time updates"""
        def on_message(ws, message):
            try:
                data = json.loads(message)
                self.ws_messages.append(data)
            except json.JSONDecodeError:
                pass
        
        def on_open(ws):
            self.ws_connected = True
            print("✅ WebSocket connected")
        
        def on_close(ws, close_status_code, close_msg):
            self.ws_connected = False
            print("❌ WebSocket disconnected")
        
        def on_error(ws, error):
            print(f"WebSocket error: {error}")
        
        # Create WebSocket connection in separate thread
        def run_websocket():
            self.ws = websocket.WebSocketApp(
                self.ws_url,
                on_message=on_message,
                on_open=on_open,
                on_close=on_close,
                on_error=on_error
            )
            self.ws.run_forever()
        
        ws_thread = threading.Thread(target=run_websocket, daemon=True)
        ws_thread.start()
        
        # Wait for connection
        for _ in range(50):  # 5 seconds timeout
            if self.ws_connected:
                break
            time.sleep(0.1)

    def _start_stream_thread(self):
        """Long-poll /api/gamestate/stream for real-time updates when WS is unavailable."""
        def run_stream():
            while True:
                try:
                    r = requests.get(f"{self.api_base}/gamestate/stream", timeout=60)
                    if r.status_code == 200 and r.text:
                        try:
                            data = json.loads(r.text)
                            self.ws_messages.append(data)
                        except json.JSONDecodeError:
                            pass
                except Exception:
                    time.sleep(0.5)
        threading.Thread(target=run_stream, daemon=True).start()
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        try:
            # Reset game via API
            response = requests.post(f"{self.api_base}/reset", timeout=10)
            if response.status_code != 200:
                raise RuntimeError("Failed to reset game")
            
            # Wait a moment for reset to complete
            time.sleep(1)
            
            # Get initial state
            self.current_state = self._get_game_state()
            self.step_count = 0
            self.episode_reward = 0
            self.previous_enemy_count = len(self.current_state.enemy_units)
            self.previous_my_count = len(self.current_state.my_units)
            
            observation = self._state_to_observation(self.current_state)
            info = self._get_info()
            
            return observation, info
            
        except Exception as e:
            raise RuntimeError(f"Failed to reset environment: {e}") from e
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute action and return next state"""
        
        # Execute action
        self._execute_action(action)
        
        # Get new state
        self.current_state = self._get_game_state()
        self.step_count += 1
        
        # Calculate reward
        reward = self._calculate_reward()
        self.episode_reward += reward
        
        # Check if episode is done
        terminated = self._is_terminated()
        truncated = self.step_count >= self.max_episode_steps
        
        # Prepare observation and info
        observation = self._state_to_observation(self.current_state)
        info = self._get_info()
        # Provide action mask for downstream policy
        info['action_mask'] = self._get_action_mask()
        
        return observation, reward, terminated, truncated, info
    
    def _get_game_state(self) -> GameState:
        """Fetch current game state from API"""
        try:
            response = requests.get(f"{self.api_base}/gamestate", timeout=5)
            if response.status_code != 200:
                raise RuntimeError("Failed to get game state")
            
            data = response.json()
            
            # Parse units
            my_units = [
                ActorInfo(
                    id=unit['Id'],
                    type=unit['Type'],
                    x=unit['Position']['X'],
                    y=unit['Position']['Y'],
                    health=unit['Health'],
                    max_health=unit['MaxHealth'],
                    is_idle=unit['IsIdle']
                )
                for unit in data['MyUnits']
            ]
            self._my_unit_ids = [u.id for u in my_units]
            
            enemy_units = [
                ActorInfo(
                    id=unit['Id'],
                    type=unit['Type'],
                    x=unit['Position']['X'],
                    y=unit['Position']['Y'],
                    health=unit['Health'],
                    max_health=unit['MaxHealth']
                )
                for unit in data['EnemyUnits']
            ]
            self._enemy_unit_ids = [u.id for u in enemy_units]

            ally_units = [
                ActorInfo(
                    id=unit['Id'],
                    type=unit['Type'],
                    x=unit['Position']['X'],
                    y=unit['Position']['Y'],
                    health=unit['Health'],
                    max_health=unit['MaxHealth'],
                    is_idle=True,
                )
                for unit in data.get('AllyUnits', [])
            ]

            # Map resource cells (optional)
            resource_cells = []
            if 'Map' in data and isinstance(data['Map'], dict):
                for rc in data['Map'].get('ResourceCells', []) or []:
                    try:
                        resource_cells.append(ResourceCell(
                            x=rc['X'], y=rc['Y'], type=rc.get('Type', ''), density=rc.get('Density', 0)
                        ))
                    except Exception:
                        continue
            
            # Parse placeable areas (optional)
            placeable_areas: Dict[str, List[Tuple[int, int]]] = {}
            for entry in data.get('PlaceableAreas', []) or []:
                try:
                    unit_type = entry.get('UnitType', '')
                    cells = entry.get('Cells', []) or []
                    coords = [(c['X'], c['Y']) for c in cells]
                    if unit_type:
                        if unit_type not in placeable_areas:
                            placeable_areas[unit_type] = []
                        placeable_areas[unit_type].extend(coords)
                except Exception:
                    continue

            # Cache production queue actor ids for action resolution
            self._queue_actor_ids = []
            try:
                for q in (data.get('Production', {}) or {}).get('Queues', []) or []:
                    aid = q.get('ActorId')
                    if isinstance(aid, int):
                        self._queue_actor_ids.append(aid)
            except Exception:
                self._queue_actor_ids = []

            return GameState(
                tick=data['Tick'],
                my_units=my_units,
                enemy_units=enemy_units,
                ally_units=ally_units,
                cash=data['Resources']['Cash'],
                resources=data['Resources']['Resources'],
                resource_capacity=data['Resources']['ResourceCapacity'],
                power_provided=data['Power']['Provided'],
                power_drained=data['Power']['Drained'],
                power_state=data['Power']['State'],
                map_width=data['MapSize']['X'],
                map_height=data['MapSize']['Y'],
                resource_cells=resource_cells,
                production=data.get('Production', {}),
                placeable_areas=placeable_areas
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to get game state: {e}") from e
    
    def _execute_action(self, action: np.ndarray):
        """Execute the given action"""
        action_type, unit_id, target_x, target_y, target_id, unit_type = action
        atype = self.action_types[int(action_type)] if self.action_types else 'move'
        
        actions = []
        
        if atype == 'move':  # Move
            actions.append({
                "Type": "move",
                "ActorId": int(self._resolve_my_unit_id(unit_id)),
                "TargetX": int(target_x),
                "TargetY": int(target_y)
            })
        
        elif atype == 'attack':  # Attack
            actions.append({
                "Type": "attack",
                "ActorId": int(self._resolve_my_unit_id(unit_id)),
                "TargetId": int(self._resolve_enemy_unit_id(target_id))
            })
        
        elif atype == 'noop':  # No operation
            # Intentionally do nothing
            pass
        
        elif atype == 'produce':  # Produce (queue actor index)
            unit_type_name = self.reverse_unit_types.get(unit_type, 'infantry')
            actions.append({
                "Type": "produce",
                "ActorId": int(self._resolve_queue_actor_id(unit_id)),
                "UnitType": unit_type_name
            })
        
        elif atype == 'build':  # Build (queue actor index)
            unit_type_name = self.reverse_unit_types.get(unit_type, 'barracks')
            actions.append({
                "Type": "build",
                "ActorId": int(self._resolve_queue_actor_id(unit_id)),
                "UnitType": unit_type_name,
                "TargetX": int(target_x),
                "TargetY": int(target_y)
            })

        elif atype == 'deploy':  # Deploy (e.g., MCV)
            actions.append({
                "Type": "deploy",
                "ActorId": int(self._resolve_my_unit_id(unit_id))
            })

        if actions:
            # Deduplicate actions: skip any that match a recent signature within TTL
            deduped = []
            for a in actions:
                sig = self._action_signature(a)
                if not self._is_duplicate_action(sig):
                    deduped.append(a)
            actions = deduped

        if actions:
            try:
                response = requests.post(
                    f"{self.api_base}/actions",
                    json=actions,
                    timeout=5
                )
                if response.status_code != 200:
                    print(f"Action execution failed: {response.text}")
                else:
                    # Record successfully sent actions
                    for a in actions:
                        self._record_action(self._action_signature(a))
            except Exception as e:
                print(f"Failed to execute action: {e}")

    def _action_signature(self, a: Dict) -> Tuple:
        """Create a normalized signature for an action dict to support deduplication."""
        return (
            a.get('Type', ''),
            int(a.get('ActorId', -1)),
            int(a.get('TargetX', -1)),
            int(a.get('TargetY', -1)),
            int(a.get('TargetId', -1)),
            str(a.get('UnitType', '')),
        )

    def _is_duplicate_action(self, sig: Tuple) -> bool:
        """Return True if an equivalent action was sent within the TTL steps."""
        now = self.step_count
        # Purge expired
        self._recent_actions = deque([(s, t) for (s, t) in self._recent_actions if now - t <= self._action_ttl_steps], maxlen=256)
        for s, t in self._recent_actions:
            if s == sig:
                return True
        return False

    def _record_action(self, sig: Tuple):
        now = self.step_count
        self._recent_actions.append((sig, now))

    def _get_action_mask(self) -> Dict[str, np.ndarray]:
        """Compute a simple action mask over the configured action set.
        Returns a dict of boolean arrays usable by common RL libraries.
        - move_mask[unit] = 1 if unit exists
        - attack_mask[unit, target] = 1 if both exist and enemy exists
        - deploy_mask[unit] = 1 if unit name suggests deploy capability (heuristic)
        Note: This is a lightweight mask; engine-side validity still applies.
        """
        max_units = 100
        my_units = self.current_state.my_units if self.current_state else []
        enemy_units = self.current_state.enemy_units if self.current_state else []
        queue_ids = self._queue_actor_ids if hasattr(self, '_queue_actor_ids') else []

        move_mask = np.zeros((max_units,), dtype=np.uint8)
        deploy_mask = np.zeros((max_units,), dtype=np.uint8)
        attack_mask = np.zeros((max_units, max_units), dtype=np.uint8)

        for i, u in enumerate(my_units[:max_units]):
            move_mask[i] = 1
            t = u.type.lower()
            if 'mcv' in t or 'deploy' in t:
                deploy_mask[i] = 1

        if enemy_units:
            for i, _ in enumerate(my_units[:max_units]):
                for j, _ in enumerate(enemy_units[:max_units]):
                    attack_mask[i, j] = 1

        mask = {
            'action_type': np.ones((len(self.action_types),), dtype=np.uint8),
            'move_mask': move_mask,
            'attack_mask': attack_mask,
            'deploy_mask': deploy_mask,
        }
        if 'produce' in self.action_types:
            # Enable indices that correspond to existing production queues
            produce_mask = np.zeros((max_units,), dtype=np.uint8)
            n = min(len(queue_ids), max_units)
            if n > 0:
                produce_mask[:n] = 1
            mask['produce_mask'] = produce_mask
        if 'build' in self.action_types:
            # Enable queue indices if any placeable items exist (heuristic)
            build_mask = np.zeros((max_units,), dtype=np.uint8)
            has_any_placeable = bool(self.current_state and self.current_state.placeable_areas)
            if has_any_placeable:
                n = min(len(queue_ids), max_units)
                if n > 0:
                    build_mask[:n] = 1
            mask['build_mask'] = build_mask

        return mask

    def _resolve_my_unit_id(self, index: int) -> int:
        """Map unit index to actual my unit ActorId; clamp if out of range."""
        if not self._my_unit_ids:
            return int(index)
        i = int(max(0, min(index, len(self._my_unit_ids) - 1)))
        return self._my_unit_ids[i]

    def _resolve_enemy_unit_id(self, index: int) -> int:
        """Map target index to actual enemy unit ActorId; clamp if out of range."""
        if not self._enemy_unit_ids:
            return int(index)
        i = int(max(0, min(index, len(self._enemy_unit_ids) - 1)))
        return self._enemy_unit_ids[i]

    def _resolve_queue_actor_id(self, index: int) -> int:
        """Map index to a production queue actor ActorId; clamp if out of range."""
        if not getattr(self, '_queue_actor_ids', None):
            return self._resolve_my_unit_id(index)
        i = int(max(0, min(index, len(self._queue_actor_ids) - 1)))
        return self._queue_actor_ids[i]
    
    def _state_to_observation(self, state: GameState) -> np.ndarray:
        """Convert game state to observation vector/image"""
        
        if self.observation_type == "vector":
            return self._state_to_vector(state)
        elif self.observation_type == "image":
            return self._state_to_image(state)
    
    def _state_to_vector(self, state: GameState) -> np.ndarray:
        """Convert state to vector observation"""
        max_units = 100
        
        # Initialize observation vector
        obs = np.zeros(max_units * 6 + max_units * 5 + 7 + 2, dtype=np.float32)
        
        # My units (normalized)
        for i, unit in enumerate(state.my_units[:max_units]):
            idx = i * 6
            obs[idx:idx+6] = [
                unit.id / 1000.0,  # Normalized ID
                self.unit_types.get(unit.type, 0) / len(self.unit_types),
                unit.x / state.map_width,
                unit.y / state.map_height,
                unit.health / max(unit.max_health, 1),
                1.0 if unit.is_idle else 0.0
            ]
        
        # Enemy units (normalized)
        start_idx = max_units * 6
        for i, unit in enumerate(state.enemy_units[:max_units]):
            idx = start_idx + i * 5
            obs[idx:idx+5] = [
                unit.id / 1000.0,
                self.unit_types.get(unit.type, 0) / len(self.unit_types),
                unit.x / state.map_width,
                unit.y / state.map_height,
                unit.health / max(unit.max_health, 1)
            ]
        
        # Resources and other info
        resource_idx = max_units * 6 + max_units * 5
        obs[resource_idx:resource_idx+7] = [
            state.cash / 10000.0,  # Normalized cash
            state.resources / max(state.resource_capacity, 1),
            state.power_provided / 1000.0,
            state.power_drained / 1000.0,
            1.0 if state.power_state == "Normal" else 0.0,
            1.0 if state.power_state == "Low" else 0.0,
            1.0 if state.power_state == "Critical" else 0.0
        ]
        
        # Map dimensions
        map_idx = resource_idx + 7
        obs[map_idx:map_idx+2] = [
            state.map_width / 128.0,
            state.map_height / 128.0
        ]
        
        return obs
    
    def _state_to_image(self, state: GameState) -> np.ndarray:
        """Convert state to image observation (128x128x10)."""
        img = np.zeros((128, 128, 10), dtype=np.uint8)
        
        # Scale coordinates to image size
        scale_x = 128.0 / state.map_width
        scale_y = 128.0 / state.map_height
        
        # Channel 0-1: My units (infantry, vehicles/others)
        for unit in state.my_units:
            x = int(unit.x * scale_x)
            y = int(unit.y * scale_y)
            if 0 <= x < 128 and 0 <= y < 128:
                if 'infantry' in unit.type.lower():
                    img[y, x, 0] = 255
                else:
                    img[y, x, 1] = 255
        
        # Channel 2: Allies (all types)
        for unit in state.ally_units:
            x = int(unit.x * scale_x)
            y = int(unit.y * scale_y)
            if 0 <= x < 128 and 0 <= y < 128:
                img[y, x, 2] = 255

        # Channel 3-4: Enemy units (infantry, vehicles/others)
        for unit in state.enemy_units:
            x = int(unit.x * scale_x)
            y = int(unit.y * scale_y)
            if 0 <= x < 128 and 0 <= y < 128:
                if 'infantry' in unit.type.lower():
                    img[y, x, 3] = 255
                else:
                    img[y, x, 4] = 255
        
        # Channel 5: Resource density map (from resource_cells)
        if state.resource_cells:
            max_density = max((rc.density for rc in state.resource_cells), default=1)
            max_density = max(1, max_density)
            for rc in state.resource_cells:
                x = int(rc.x * scale_x)
                y = int(rc.y * scale_y)
                if 0 <= x < 128 and 0 <= y < 128:
                    val = int(min(255, (rc.density / max_density) * 255))
                    img[y, x, 5] = max(img[y, x, 5], val)
        
        # Channel 6: Power surplus (global constant map)
        power_level = int((state.power_provided - state.power_drained))
        img[:, :, 6] = max(0, min(255, power_level))

        # Channel 7: Cash (global constant map)
        img[:, :, 7] = max(0, min(255, int(state.cash / 100)))

        # Channel 8: My low-health mask (<50%)
        for unit in state.my_units:
            if unit.max_health > 0 and unit.health / unit.max_health < 0.5:
                x = int(unit.x * scale_x)
                y = int(unit.y * scale_y)
                if 0 <= x < 128 and 0 <= y < 128:
                    img[y, x, 8] = 255

        # Channel 9: Enemy low-health mask (<50%)
        for unit in state.enemy_units:
            if unit.max_health > 0 and unit.health / unit.max_health < 0.5:
                x = int(unit.x * scale_x)
                y = int(unit.y * scale_y)
                if 0 <= x < 128 and 0 <= y < 128:
                    img[y, x, 9] = 255
        
        # Channels 6-7: Reserved for additional features
        
        return img
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on game state changes"""
        if not self.current_state:
            return 0.0
        
        reward = 0.0
        
        # Unit count changes
        current_my_count = len(self.current_state.my_units)
        current_enemy_count = len(self.current_state.enemy_units)
        
        # Reward for destroying enemy units
        enemy_destroyed = self.previous_enemy_count - current_enemy_count
        reward += enemy_destroyed * 10.0
        
        # Penalty for losing own units
        my_lost = self.previous_my_count - current_my_count
        reward -= my_lost * 5.0
        
        # Small reward for resource accumulation
        reward += (self.current_state.cash / 1000.0) * 0.1
        
        # Penalty for power shortage
        if self.current_state.power_state == "Critical":
            reward -= 1.0
        elif self.current_state.power_state == "Low":
            reward -= 0.5
        
        # Update counters
        self.previous_enemy_count = current_enemy_count
        self.previous_my_count = current_my_count
        
        return reward
    
    def _is_terminated(self) -> bool:
        """Check if episode should terminate"""
        if not self.current_state:
            return False
        
        # Game ends if no units left or all enemies destroyed
        return (len(self.current_state.my_units) == 0 or 
                len(self.current_state.enemy_units) == 0)
    
    def _get_info(self) -> Dict:
        """Get additional information about current state"""
        if not self.current_state:
            return {}
        
        return {
            'tick': self.current_state.tick,
            'step_count': self.step_count,
            'episode_reward': self.episode_reward,
            'my_unit_count': len(self.current_state.my_units),
            'ally_unit_count': len(self.current_state.ally_units),
            'enemy_unit_count': len(self.current_state.enemy_units),
            'cash': self.current_state.cash,
            'power_state': self.current_state.power_state,
            'production': self.current_state.production,
            # Expose additional details to enable rule-based logic
            'placeable_areas': self.current_state.placeable_areas,
            'my_units': [
                {
                    'id': u.id,
                    'type': u.type,
                    'x': u.x,
                    'y': u.y,
                    'is_idle': u.is_idle,
                }
                for u in self.current_state.my_units
            ],
            # Mapping from unit index -> ActorId used by the environment
            'my_unit_ids': self._my_unit_ids[:],
            # Mapping from queue index -> ActorId for production/build actions
            'queue_actor_ids': self._queue_actor_ids[:],
        }
    
    def close(self):
        """Close environment and cleanup connections"""
        if self.ws:
            self.ws.close()
        print("🔌 OpenRA environment closed")


# Utility functions for creating specific environment configurations
def create_simple_combat_env(**kwargs):
    """Create environment optimized for simple combat scenarios"""
    return OpenRAEnvironment(observation_type="vector", **kwargs)


def create_visual_env(**kwargs):
    """Create environment with image-based observations"""
    return OpenRAEnvironment(observation_type="image", **kwargs)


def create_resource_management_env(**kwargs):
    """Create environment focused on resource management"""
    env = OpenRAEnvironment(observation_type="vector", **kwargs)
    # Could add custom reward shaping here
    return env