# OpenRA Reinforcement Learning Environment (OpenRA.Bot)

A Gymnasium-compatible Python interface and an in-engine C# bridge for training RL agents on [OpenRA](https://github.com/OpenRA/OpenRA).

## Features

- Gymnasium-compatible API (reset/step/close with `observation, reward, terminated, truncated, info`)
- Vector and image observations
- Action space covering move, attack, and produce (build is stubbed for now)
- Real-time updates via HTTP long-polling (`/api/gamestate/stream`)
- Examples for random play, a simple rule-based agent, and Stable-Baselines3

## Repository Layout

- `openra_env.py`: Python environment implementation
- `example_usage.py`: End-to-end examples (random, agent, SB3, vision)
- `requirements.txt`: Python dependencies
- `OpenRAApiBridge.cs`: C# API bridge (HTTP + long-polling)

## Quick Start (Windows)

### Prerequisites

- Windows 10/11
- Python 3.8+
- Clone [OpenRA](http://github.com/OpenRA/OpenRA) repo
- Copy `OpenRAApiBridge.cs` to your OpenRA Project\OpenRA.Mods.Common\OpenRAApiBridge.cs Built OpenRA solution. Use Visual Studio 2022 or `dotnet` to build, then run `OpenRA.WindowsLauncher` and select the RA mod.

### Enable the Python RL bot in RA

The bridge is already wired into the RA mod:

```yaml
# mods/ra/rules/ai.yaml (Player: section)
PythonApiBridge@PythonRL:
  ApiPort: 8081
  EnableRealTimeUpdates: true
```

Start a Skirmish game, add an AI slot, and choose bot type "Python RL Agent". When the match starts the HTTP API listens on `http://localhost:8081`.

### Python setup

```bash
cd OpenRA.Bot
python -m venv .venv  # optional
.venv\Scripts\activate # PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Run examples
python example_usage.py
```

If you use Stable-Baselines3, install extras from `requirements.txt` (sb3/torch/tensorboard are listed under optional dependencies).

## Basic Usage

```python
from openra_env import OpenRAEnvironment, create_simple_combat_env

env = create_simple_combat_env(api_port=8081)  # must match RA bridge port
obs, info = env.reset()
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
env.close()
```

## SB3 Example

```python
from stable_baselines3 import PPO
from openra_env import create_simple_combat_env

env = create_simple_combat_env(api_port=8081)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000)
model.save("openra_ppo_agent")
```

## API Endpoints (bridge)

- `GET  /api/gamestate` – current state (units/resources/power/map)
- `GET  /api/gamestate/stream` – long-polling for real-time updates/events
- `POST /api/actions` – execute actions (JSON array). Supported: `move`, `attack`, `produce`
- `POST /api/reset` – reset episode (clears queued orders; map reload not guaranteed)

Example curl:

```bash
# Inspect state
curl http://localhost:8081/api/gamestate | cat

# Move a unit (replace ActorId with your unit ID)
curl -X POST http://localhost:8081/api/actions \
  -H "Content-Type: application/json" \
  -d "[{\"Type\":\"move\",\"ActorId\":1,\"TargetX\":10,\"TargetY\":10}]"
```

## Observations and Actions

### Observations
- Vector: concatenated stats for my units, enemy units, resources/power, and map size
- Image: multi-channel semantic feature map (CNN-friendly)

#### Vector observation format
- Shape: `(obs_dim,)`
- Constants inside the environment: `max_units = 100`
- Layout (in order):
  - My units: up to 100 units, 6 values each
    - `[id_norm, type_norm, x_norm, y_norm, health_norm, is_idle]`
  - Enemy units: up to 100 units, 5 values each
    - `[id_norm, type_norm, x_norm, y_norm, health_norm]`
  - Resources & power: 7 values
    - `[cash_norm, resource_fill, power_provided_norm, power_drained_norm, power_is_normal, power_is_low, power_is_critical]`
  - Map size: 2 values
    - `[map_width/128.0, map_height/128.0]`

#### Image observation format
- Shape: `(128, 128, 10)`
- Channels:
  - `0`: my infantry
  - `1`: my vehicles/others
  - `2`: allies (all types)
  - `3`: enemy infantry
  - `4`: enemy vehicles/others
  - `5`: resource density (grayscale, from visible resource cells)
  - `6`: power surplus (global scalar repeated)
  - `7`: cash (global scalar repeated)
  - `8`: my low-health mask (<50%)
  - `9`: enemy low-health mask (<50%)

#### Additional state (metadata)
- Returned via `/api/gamestate` and available in Python through `info` / `state`:
  - `Map`: `{ Tileset, Bounds{X,Y,Width,Height}, ResourceCells[{X,Y,Type,Density}] }`
  - `AllyUnits[]`: allies visible under fog-of-war
  - `Production`: per-building queues owned by player
    - `Production.Queues[]` with:
      - `ActorId`, `Type` (e.g., Building/Infantry/Vehicle), `Group`, `Enabled`
      - `Current`: `{ Item, Cost, Progress(0-100), Paused, Done }`
      - `Items[]`: same shape as `Current`
      - `Buildable[]`: `{ Name, Cost }`

### Actions (MultiDiscrete)
- Layout: `[action_type, unit_idx, target_x, target_y, target_idx, unit_type_idx]`
- `action_type` indexes into a configurable list: `env.action_types`
  - Default reduced set: `['move', 'attack', 'deploy']`
  - Full set (optional): `['move','attack','produce','build','deploy']`
- Semantics:
  - `move`: uses `unit_idx`, `target_x`, `target_y`
  - `attack`: uses `unit_idx`, `target_idx`
  - `produce`: uses `unit_idx` (producer building), `unit_type_idx` (maps to unit type name)
  - `build`: uses `unit_idx` (builder), `unit_type_idx`, `target_x`, `target_y` (note: server-side build handler is a stub)
  - `deploy`: uses `unit_idx`（e.g., MCV deploy）

#### Action mask (provided in `info['action_mask']`)
- Keys / shapes:
  - `action_type`: `(len(env.action_types),)`
  - `move_mask`: `(100,)`
  - `attack_mask`: `(100, 100)`
  - `deploy_mask`: `(100,)`
  - If enabled: `produce_mask`, `build_mask` (currently alias `move_mask`)
- Note: This is a lightweight, heuristic mask to reduce invalid samples. Engine-side checks still apply.

## Configuration

- Bridge port: edit `mods/ra/rules/ai.yaml` (`ApiPort`)
- Python env: pass `api_port=...` to `OpenRAEnvironment` or `create_*_env`

## Reward (default)

- +10 per enemy unit destroyed
- -5 per own unit lost
- +0.1 × (cash / 1000)
- Power penalties when low/critical

You can wrap the env to shape rewards (see `example_usage.py`).

## Troubleshooting

- Connection errors: ensure the RA match with the "Python RL Agent" bot is running; check port 8081 and firewall
- Empty/invalid unit IDs: the unit may have died or be out of vision; refresh state and reselect
- Build actions do nothing: build is a stub in `openra_api_bridge.cs`; extend `ProcessBuildAction` to enable

## ID handling (important)

- Each actor has a unique `ActorID` generated by the engine.
- The environment maps `unit_idx`/`target_idx` to real IDs using the latest `/api/gamestate`:
  - `self._my_unit_ids`, `self._enemy_unit_ids` are refreshed every step.
  - Out-of-range indices are clamped; prefer using the `action_mask` to avoid invalid picks.
- For precise targeting, always base your actions on the latest state.

## Contributing

PRs are welcome. Please keep code readable and update examples as needed.

## License

MIT (see LICENSE)
