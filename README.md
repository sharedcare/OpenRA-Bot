# OpenRA Reinforcement Learning Environment (OpenRA.Bot)

A Gymnasium-compatible Python interface and an in-engine C# bridge for training RL agents on OpenRA.

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
- Image: `(128, 128, 8)` channels of semantic layers (demo-oriented)

### Actions (MultiDiscrete)
- `[action_type, unit_idx, target_x, target_y, target_idx, unit_type_idx]`
- `0=move`, `1=attack`, `2=produce`, `3=build` (build is not yet implemented in the bridge)

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

## Contributing

PRs are welcome. Please keep code readable and update examples as needed.

## License

MIT (see LICENSE)