# OpenRA Reinforcement Learning Environment (OpenRA.Bot)

A Gymnasium-compatible Python interface and an in-engine C# bridge for training RL agents on [OpenRA](https://github.com/OpenRA/OpenRA).

## Features

- Gymnasium-compatible API (reset/step/close with `observation, reward, terminated, truncated, info`)
- Vector and image observations
- Action space covering move, attack, produce, build, deploy
- Direct in-engine control via `PythonAPI.cs`
- Examples for random play, a simple rule-based agent, and Stable-Baselines3

## Repository Layout

- `openra_env.py`: Python environment implementation
- `example_usage.py`: End-to-end examples (random, agent, SB3, vision)
- `requirements.txt`: Python dependencies
- `PythonAPI.cs`: In-engine C# bridge (local game/lobby control from Python via pythonnet)

## Quick Start (Windows)

### Prerequisites

- Windows 10/11
- Python 3.8+
- Clone [OpenRA](http://github.com/OpenRA/OpenRA) repo
- Add `OpenRA.Bot/utils/PythonAPI.cs` to your OpenRA solution (recommended: `OpenRA.Game` project), then build with Visual Studio 2022 or `dotnet`. This exposes a stable API (StartLocalGame/Step/GetState/SendActions) for Python.

The Python environment can start a local game automatically through `PythonAPI.StartLocalGame(...)` when you `reset()` the env (configure `bin_dir`, `mod_id`, and `map_uid`).

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
from envs.openra_env import make_env

env = make_env(
    bin_dir="OpenRA/bin",
    mod_id="ra",
    map_uid="b53e25e007666442dbf62b87eec7bfbe8160ef3f",
    ticks_per_step=10,
    observation_type="feature",
    enable_actions=['noop','move','attack','produce','build','deploy'],
)
obs, info = env.reset()     # internally calls PythonAPI.StartLocalGame(...)
for _ in range(1000):
    actions = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(actions)
    if terminated or truncated:
        break
env.close()
```

## SB3 Example

```python
from stable_baselines3 import PPO
from envs.openra_env import make_env

env = make_env(
    bin_dir="OpenRA/bin",
    mod_id="ra",
    map_uid="b53e25e007666442dbf62b87eec7bfbe8160ef3f",
    ticks_per_step=10,
    observation_type="feature",
)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000)
model.save("openra_ppo_agent")
```

## PythonAPI (C#) surface

- `StartLocalGame(modId, mapUid[, binDir][, addBotOpponent, botType, botSlotId, explored, fog])`
- `Step()` – advances simulation; returns whether a tick occurred
- `GetState()` – returns `RLState` with `Actors`, `Resources`, `Production`, `PlaceableAreas`, `BuildableCatalog`
- `SendActions(IEnumerable<RLAction>)` – queue orders; supports standard orders and `StartProduction`
- Multiplayer helpers: `JoinServer`, `WaitForConnection`, `GetLobbyInfo`, `ClaimSlot`, `AddBotToFreeSlot`, `StartGameFromLobby`, etc.

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
- Returned via `PythonAPI.GetState()` and available in Python through `info` / `state`:
  - `Map`: `{ Tileset, Bounds{X,Y,Width,Height}, ResourceCells[{X,Y,Type,Density}] }`
  - `AllyUnits[]`: allies visible under fog-of-war
  - `Production`: per-building queues owned by player
    - `Production.Queues[]` with:
      - `ActorId`, `Type` (e.g., Building/Infantry/Vehicle), `Group`, `Enabled`
      - `Items[]`: `{ Item, Cost, Progress(0-100), Paused, Done }`
      - `Buildable[]`: `{ Name, Cost }`
  - `PlaceableAreas[]`: legal placement cells for finished building items
    - Per entry: `{ ActorId, UnitType, Cells: [{X,Y}, ...] }`

### Actions (MultiDiscrete)
- Layout: `[action_type, unit_idx, target_x, target_y, target_idx, unit_type_idx]`
- `action_type` indexes into a configurable list: `env.action_types`
  - Default reduced set: `['move', 'attack', 'deploy']`
  - Full set (optional): `['move','attack','produce','build','deploy']`
- Semantics:
  - `move`: uses `unit_idx`, `target_x`, `target_y`
  - `attack`: uses `unit_idx`, `target_idx`
  - `produce`: uses `unit_idx` (producer building), `unit_type_idx` (maps to unit type name)
  - `build`: uses `unit_idx` (queue actor idx), `unit_type_idx`, `target_x`, `target_y`
  - `deploy`: uses `unit_idx`（e.g., MCV deploy）

#### Action mask (provided in `info['action_mask']`)
- Keys / shapes:
  - `action_type`: `(len(env.action_types),)`
  - `move_mask`: `(100,)`
  - `attack_mask`: `(100, 100)`
  - `deploy_mask`: `(100,)`
  - If enabled: `produce_mask`
  - `build_mask`: `(100,)` enabled when there exists any legal placement derived from `PlaceableAreas`
- Note: This is a lightweight, heuristic mask to reduce invalid samples. Engine-side checks still apply.

## Configuration

- Python env (local game): set `bin_dir`, `mod_id`, `map_uid` in `make_env(...)`
- Host a lobby from Python: use `env.configure_host(options=[...])`
- Join a remote lobby: `env.configure_remote(host, port, password="", spectator=False)`

## Reward (default)

- +10 per enemy unit destroyed
- -5 per own unit lost
- +0.1 × (cash / 1000)
- Power penalties when low/critical

You can wrap the env to shape rewards (see `example_usage.py`).

## Troubleshooting

- Local start fails: ensure `bin_dir` points to your built OpenRA `bin` directory and `map_uid` exists in the mod’s map cache
- Empty/invalid unit IDs: the unit may have died or be out of vision; refresh state and reselect
- Production/build actions do nothing: ensure your building has a production queue and use `StartProduction` via `SendActions`

## ID handling (important)

- Each actor has a unique `ActorID` generated by the engine.
- The environment maps `unit_idx`/`target_idx` to real IDs using the latest `PythonAPI.GetState()` snapshot:
  - `self._my_unit_ids`, `self._enemy_unit_ids` are refreshed every step.
  - Out-of-range indices are clamped; prefer using the `action_mask` to avoid invalid picks.
- For precise targeting, always base your actions on the latest state.

## Contributing

PRs are welcome. Please keep code readable and update examples as needed.

## License

MIT (see LICENSE)
