# OpenRA.Bot

`OpenRA.Bot` is a Python-side RL/control package for [OpenRA](https://github.com/OpenRA/OpenRA). It uses `pythonnet` to load the built game assemblies, calls the engine-side `PythonAPI`, and exposes a Gym-style environment for random agents, rule-based agents, and a baseline PPO training loop.

This repository currently contains a usable end-to-end baseline, but it is still closer to a research scaffold than a polished RL framework. The recent work has made the PPO stack, remote-lobby control path, and action masking substantially more reliable, but observation design and trainer quality are still evolving.

## What Is In Scope

- Python environment wrapper around the in-engine API
- Engine-side `PythonAPI` bridge for local game start, stepping, state extraction, and action dispatch
- Rule-based and random agents for smoke testing
- A custom `ActorCritic` + `PPOAgent` baseline
- Local game, local hosted lobby, and remote lobby connection helpers

## Current Layout

- `envs/openra_env.py`: main Gym environment
- `utils/engine.py`: loads `OpenRA.Game.dll` and `PythonAPI` through `pythonnet`
- `utils/obs.py`: converts `PythonAPI.GetState()` output into Python dictionaries
- `utils/actions.py`: encodes Python action dicts into `RLAction`
- `utils/net.py`: local host / remote join / lobby helpers
- `utils/PythonAPI.cs`: engine bridge source used by the Python side
- `agent/agent.py`: `RandomMoveAgent`, `RuleBasedAgent`, `PPOAgent`
- `models/actor.py`: encoders and `ActorCritic`
- `models/buffer.py`: rollout buffer for PPO
- `scripts/example_usage.py`: rule-based / random control example
- `scripts/train_rl.py`: baseline PPO training entry
- `scripts/rl_smoke_test.py`: quick RL smoke test
- `scripts/remote_rule_based.py`: join a remote lobby and run `RuleBasedAgent`
- `scripts/remote_ppo.py`: join a remote lobby and inspect `PPOAgent` actions, masks, and queue state

## Architecture Overview

The current execution path is:

1. `envs/openra_env.py` calls `utils/engine.py` to load the OpenRA assemblies.
2. `PythonAPI.StartLocalGame(...)` or the lobby helpers initialize a match.
3. `PythonAPI.GetState()` returns a simplified `RLState`.
4. `utils/obs.py` converts that state into Python dicts.
5. `OpenRAEnv` converts the raw dict into `feature`, `vector`, or `image` observations.
6. An agent chooses either a legacy dict action list or a `MultiDiscrete` action.
7. `utils/actions.py` and `PythonAPI.SendActions(...)` translate that into OpenRA orders.
8. `PythonAPI.Step()` advances the simulation.

## Prerequisites

- A platform supported by your OpenRA build and `pythonnet`
- Python 3.8+
- A built OpenRA tree with `OpenRA.Game.dll` and `OpenRA.runtimeconfig.json`
- A mod and map that can be started from code, for example `ra`

## Engine Bridge Setup

The Python package expects the compiled `PythonAPI` type to be available from `OpenRA.Game.dll`.

Recommended workflow:

1. Keep the bridge source in `OpenRA.Bot/utils/PythonAPI.cs`.
2. Add or sync that file into the `OpenRA.Game` project in your OpenRA solution.
3. Build OpenRA so that the Python side can load the resulting assemblies from `bin_dir`.

`OpenRAApiBridge.cs` is deprecated and should not be used by new code.

## Python Setup

```powershell
cd F:\Projects\OpenRA\OpenRA.Bot
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Running Existing Scripts

Rule-based / manual smoke test:

```bash
python scripts/example_usage.py
```

Baseline PPO training:

```bash
python scripts/train_rl.py
```

Remote rule-based control:

```bash
python scripts/remote_rule_based.py --host 127.0.0.1 --port 1234 --slot Multi0
```

Remote PPO action debugging:

```bash
python scripts/remote_ppo.py --host 127.0.0.1 --port 1234 --slot Multi0
```

Remote PPO training:

```bash
python scripts/train_rl.py --remote-host 127.0.0.1 --remote-port 1234 --remote-slot Multi0
```

## Minimal Usage

```python
from envs.openra_env import make_env

env = make_env(
    bin_dir="F:/Projects/OpenRA/bin",
    mod_id="ra",
    map_uid="b53e25e007666442dbf62b87eec7bfbe8160ef3f",
    ticks_per_step=10,
    observation_type="vector",
    enable_actions=["noop", "move", "attack", "produce", "build", "deploy"],
)

obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

env.close()
```

## Observation Modes

`OpenRAEnv` currently supports three observation modes:

- `feature`: returns the raw Python dict built from `PythonAPI.GetState()`
- `vector`: flattened numeric observation for MLP policies
- `image`: `128 x 128 x 10` semantic map for CNN-style policies

### `feature`

This is the most complete mode and is the best option for debugging. It includes:

- `actors`
- `resources`
- `production`
- `producible_catalog`
- `placeable_areas`
- `cash`
- `resources_total`
- `resource_capacity`
- `power`
- `my_owner`

Each actor now exposes two order-related fields:

- `available_orders`: a filtered list intended for bot/RL logic
- `available_order_ids`: the raw order ids exposed by engine traits

Important nuance: `available_order_ids` is closer to "what traits are present on this actor", while `available_orders` is the safer field to use for decision logic. For example, transformable buildings may expose a raw `Move` order id even when they should not be treated as currently mobile.

### `vector`

Current vector layout:

- Up to 100 friendly units, 6 features each
- Up to 100 enemy units, 5 features each
- 7 resource/power slots
- 2 map-size slots

Important caveat: the resource/power section is currently placeholder-filled in `envs/openra_env.py`, so the vector observation does not yet fully use the economy state already available from `PythonAPI.GetState()`.

### `image`

Current image layout:

- Shape: `(128, 128, 10)`
- Channels currently used reliably:
  - friendly infantry / non-infantry
  - enemy infantry / non-infantry
- Channels for resources, cash, and power are not fully populated yet

## Action Space

The RL action space is currently:

```text
[action_type, unit_idx, target_x, target_y, target_idx, unit_type_idx]
```

Supported action types depend on `enable_actions`, but the usual set is:

- `noop`
- `move`
- `attack`
- `produce`
- `build`
- `deploy`

Semantics:

- `move`: uses `unit_idx`, `target_x`, `target_y`
- `attack`: uses `unit_idx`, `target_idx`
- `produce`: uses queue actor index + `unit_type_idx`
- `build`: uses queue actor index + `unit_type_idx` + target cell
- `deploy`: uses `unit_idx`

The environment also accepts legacy Python dict actions, which is what the rule-based agent uses.

## Action Masks

`info["action_mask"]` currently includes some of the following fields:

- `action_type`
- `move_mask`
- `attack_mask`
- `deploy_mask`
- `produce_queue_mask`
- `produce_unit_type_mask`
- `build_mask`
- `build_unit_type_mask`
- `unit_idx`
- `target_idx`
- `target_x`
- `target_y`
- `unit_type`

These masks are no longer only heuristic action-type hints. The current implementation mixes:

- engine-side feasibility checks for `move`, `attack`, and `deploy`
- queue-state- and placement-driven checks for `produce` and `build`
- per-head masks consumed by `PPOAgent` during both sampling and training

Current behavior:

- `move_mask`: only set when the actor has a feasible move in a nearby neighborhood
- `attack_mask`: per-attacker / per-target feasibility matrix
- `deploy_mask`: checked through engine feasibility
- `produce_queue_mask`: only queues that are enabled, empty, and can actually produce something in the current catalog
- `build_mask`: only queues with a completed item and a currently available placement area
- `target_x` / `target_y`: conditioned on the selected actor or queue
  - move targets are restricted to a local neighborhood around the selected actor
  - build targets are restricted to coordinates present in `placeable_areas`

Remaining limitation: `target_x` and `target_y` are still masked independently rather than as a joint `(x, y)` cell distribution, so some invalid coordinate pairs can still be sampled.

## Reward Shaping

The default reward in `envs/openra_env.py` is development-oriented, not combat-oriented. It currently rewards and penalizes:

- increase in owned unit count
- increase in owned building count
- starting new production items
- canceling in-progress production
- staying below a minimum cash reserve

This is useful for bootstrapping a macro baseline, but it is not enough on its own for strong tactical play.

## Connection Modes

`OpenRAEnv.reset()` supports three startup modes:

- local single-player start through `PythonAPI.StartLocalGame(...)`
- host-local lobby flow through `env.configure_host(...)`
- remote server join through `env.configure_remote(...)`

See `utils/net.py` for the exact lobby helper flow.

For remote control, the current flow is:

1. `env.configure_remote(...)`
2. `reset()` joins the server
3. the client claims a slot, acknowledges the selected map, and marks itself ready
4. lobby/network state is pumped until the host starts the game
5. once the world exists, normal observation / action stepping begins

Recent bridge changes were specifically made to keep network traffic progressing while still in the lobby, so remote clients can stay synchronized through the lobby-to-game transition.

## Known Limitations

- The PPO baseline is still a baseline, not a polished trainer.
- The current training loop is effectively single-environment, even though some APIs are written as if vectorized training were supported.
- `PythonAPI.GetState()` is expensive because it scans a lot of world state, especially for production and build placement data.
- Observation building and action-mask generation are more consistent than before, but still relatively expensive.
- `target_x` / `target_y` masking is improved but still factorized rather than fully cell-joint.
- Several scripts still assume a local development workflow and should be treated as baseline utilities rather than final UX.

## Practical Recommendations

- Use `feature` observations first when debugging action execution.
- For remote debugging, start with `scripts/remote_rule_based.py` or `scripts/remote_ppo_debug.py` before running long PPO training jobs.
- Treat the current PPO stack as a baseline to iterate on, not a final trainer.
- If you are improving sample efficiency, first optimize state extraction and observation consistency before making the policy larger.
- If you are improving policy quality, prioritize better state encoding and stricter action masking before switching to a more complex network.

## Troubleshooting

- Local start fails: check `bin_dir`, `mod_id`, and `map_uid`, and confirm the OpenRA build artifacts exist.
- Python cannot load the engine: verify `OpenRA.runtimeconfig.json` and the required assemblies are present in `bin_dir`.
- Remote join enters the lobby but does not stay synchronized: rebuild OpenRA after syncing `utils/PythonAPI.cs`, because remote-lobby behavior depends on the latest bridge code.
- Production/build actions appear invalid: inspect `production`, `placeable_areas`, `available_orders`, and `available_order_ids` from `feature` observations first.
- Actor indices behave strangely: remember that `unit_idx` and `target_idx` are mapped through the latest cached unit-id lists, not raw actor IDs.
- A transformable building appears to have `Move`: check `available_order_ids` vs `available_orders`. The raw field may still include transform-related move orders, while the filtered field is the one intended for control logic.

## License

MIT
