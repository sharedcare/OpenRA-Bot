# OpenRA.Bot

`OpenRA.Bot` is a Python-side RL/control package for [OpenRA](https://github.com/OpenRA/OpenRA). It uses `pythonnet` to load the built game assemblies, calls the engine-side `PythonAPI`, and exposes a Gym-style environment for random agents, rule-based agents, and a baseline PPO training loop.

**Status (2026-07-01)**: Goal conditioning achieves 5x baseline improvement (0.08→0.40 reward). Economic reward with state-aware RuleBasedAgent produces diverse actions including infantry (e1/e2/e3), harvesters, and heavy tanks. Two critical ARM64 bugs discovered and fixed: `IsBuildingQueueOccupied` reflection bug (silently dropped all building production orders) and scalar dimension mismatch (first obs missing goal vector). Current bottleneck: BC-frozen encoder is state-unaware (can't distinguish 1 powr from 10 powr), causing repeated overbuilding. Full architecture documented in [ARCHITECTURE.md](ARCHITECTURE.md).

## What Is In Scope

- Python environment wrapper around the in-engine API
- Engine-side `PythonAPI` bridge for local game start, stepping, state extraction, and action dispatch
- Rule-based and random agents for smoke testing
- A custom `ActorCritic` + `PPOAgent` baseline with BC warm-start
- Local game, local hosted lobby, and remote lobby connection helpers
- Build-order distance reward (DI-star / AlphaStar inspired)
- Asset-value reward with production-start / active-production credit
- Macro production action space for development-only training
- Decision-step policy gradient masking
- Action masking with `-inf` penalty (prevents policy collapse)
- Entity-based observations (Phase 1 — `observation_type="entity"`)
- Headless local rollout and `SubprocVecEnv` multi-process training support

## Current Layout

- `envs/openra_env.py`: main Gym environment (MCV type detection fix, BO / asset reward, macro actions, decision mask support)
- `envs/vector_env.py`: spawned `SubprocVecEnv` wrapper for multi-process headless rollout
- `envs/wrappers.py`: `ShapedRewardWrapper` (passthrough + diagnostics), `AugmentedStateWrapper` (frame stacking)
- `utils/engine.py`: loads `OpenRA.Game.dll` and `PythonAPI` through `pythonnet`
- `utils/obs.py`: converts `PythonAPI.GetState()` output into Python dictionaries
- `utils/actions.py`: encodes Python action dicts into `RLAction`
- `utils/net.py`: local host / remote join / lobby helpers
- `utils/PythonAPI.cs`: engine bridge source (C#, ARM64 reflection bugs fixed — `IsBuildingQueueOccupied` guard removed)
- `utils/entity_obs.py`: entity observation builder (14-dim per actor + 16-dim scalar)
- `utils/goal_library.py`: 4 build-order goals (economy/infantry/vehicle/balanced)
- `agent/agent.py`: `RandomMoveAgent`, state-aware `RuleBasedAgent` (cash/power/building checks), `PPOAgent`
- `models/actor.py`: `VectorEncoder`, `SimpleEntityEncoder`, `MultiDiscretePolicy`, `ActorCritic` with GLU gating + multi-value-heads
- `models/entity_encoder.py`: MLP per-entity encoder + masked mean-pool (25K params)
- `models/buffer.py`: rollout buffer for PPO (GAE + dict obs support)
- `scripts/train_rl.py`: PPO training entry (entity/macro/headless/opponent/goal/teacher flags)
- `scripts/warmstart.py`: BC data collection + pre-training from RuleBasedAgent
- `scripts/remote_ppo.py`: join a remote lobby and run trained model (with --stochastic --temperature)
- `scripts/view_best.py`: view best checkpoint in local game window
- `scripts/verify_asset_reward.py`: reward headroom verification
- `ARCHITECTURE.md`: complete obs/action/reward/model specification
- `PLAN.md`: long-term roadmap (AlphaStar-style architecture)
- `REPORT.md`: detailed experiment log and findings

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

Current recommended development-only PPO launcher on Windows:

```powershell
.\scripts\train_best.ps1 -Updates 100 -RunName stronger_teacher_no_head_repeat
```

Useful overrides:

```powershell
.\scripts\train_best.ps1 `
  -Updates 150 `
  -NumSteps 128 `
  -MaxEpisodeTicks 1800 `
  -LearningRate 7e-5 `
  -TargetKl 0.03 `
  -UpdateEpochs 4 `
  -RunName ppo_asset_macro_repeat
```

The launcher uses `observation_type="entity"`, `action_space_mode="macro"`, `reward_mode="asset"` through the default environment, headless mode, BC warm-start from `RuleBasedAgent`, and does **not** load the BC action head unless `-LoadBcActionHead` is specified. Current experiments show that hard-loading the BC action head anchors PPO too strongly to the teacher's action mix and usually hurts training.

Parallel rollout can be enabled from `train_rl.py`:

```bash
python scripts/train_rl.py --observation-type entity --action-space-mode macro --headless --num-envs 4
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

`OpenRAEnv` currently supports four observation modes:

- `feature`: returns the raw Python dict built from `PythonAPI.GetState()`
- `vector`: flattened numeric observation for MLP policies
- `image`: `128 x 128 x 10` semantic map for CNN-style policies
- `entity`: fixed-cap entity tensor plus scalar features for the current lightweight entity encoder

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

### `entity`

This is the current default for new PPO experiments. It uses `utils/entity_obs.py` to build:

- `entities`: up to `MAX_ENTITIES` actor rows with per-actor features
- `entity_mask`: valid-row mask
- `scalar`: compact economy / power / game-state features

The lightweight entity encoder is enough to clone the current `RuleBasedAgent`, but PPO still plateaus near the teacher without a stronger training signal.

## Action Space

The default multidiscrete RL action space is:

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

For development-only training, `action_space_mode="macro"` collapses the meaningful decision to the `action_type` head:

```text
noop, produce:powr, produce:proc, produce:barr, produce:weap, produce:e1, ...
```

The remaining argument heads are masked to index 0. MCV deployment and finished-building placement are handled by environment automation. This was added because the original 6-head action space made successful production a low-probability joint event (`produce + correct queue + correct unit_type`), which was too hard to reinforce reliably in single-env PPO.

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

The default reward in `envs/openra_env.py` is development-oriented, not combat-oriented. The current default `reward_mode="asset"` rewards:

- cost-weighted growth in owned actors (`AssetValueTracker`)
- starting new production items
- keeping production active
- canceling in-progress production, as a penalty
- optional idle-cash and power-deficit penalties, currently disabled by default because per-step penalties can swamp one-time asset gains

`reward_mode="legacy"` keeps the older build-order / unit-count shaping path. The asset reward fixes the capped `[powr, proc, barr]` build-order reward and makes army production visible, but it is still not enough on its own for strong tactical play or decisive improvement over a scripted macro teacher.

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

### Fixed (2026-06)
- ~~Policy collapse (KL explosion to 15+)~~ Fixed with `-inf` mask penalty
- ~~MCV deploy reward=0~~ Fixed with type-based `_is_building` detection (ARM64 reflection bug workaround)
- ~~Reward signal too sparse (0.7% non-zero)~~ Fixed with BO distance reward + producing_per_step (now 98%+ non-zero)
- ~~Action mask penalty insufficient~~ Fixed: `log(clamp(1e-6))` → `-inf`
- ~~Short rollout with `num_steps < seq_len` produced zero PPO batches~~ Fixed in `models/buffer.py`
- ~~Macro action auto-placement could fail silently~~ Fixed: done buildings are placed by queue actor id
- ~~Single-process-only rollout~~ Improved with headless `SubprocVecEnv` support

### Remaining
- PPO can match the upgraded `RuleBasedAgent` and occasionally reaches higher asset reward, but does not yet decisively retain that improvement over long runs
- The final checkpoint is often not the best checkpoint; best-model saving is now required for meaningful comparisons
- Loading the BC action head directly is too strong an anchor and caused high-KL early stopping in recent runs
- The current expert prior is still weak: it is a hand-written macro script, not a full-game expert with scouting, combat, tech transitions, or opponent adaptation
- There is no soft teacher-KL regularizer yet; PPO either drifts unstably or is over-anchored by hard-loaded BC heads
- There is no goal conditioning yet: build-order / strategy targets are not fed into the policy as an explicit `z`
- Rewards are still scalar and development-heavy; there are no separate value heads for economy, army, combat, and win/loss
- Combat and terminal win/loss training are still missing from the main training loop
- `PythonAPI.GetState()` is expensive (scans all world state every call, heavy reflection usage)
- `target_x` / `target_y` masking is still factorized rather than fully cell-joint
- `PythonAPI.cs` uses reflection to access `OpenRA.Mods.Common` types — fragile across OpenRA versions and platforms
- Multi-environment rollout works in headless mode, but Windows multiprocessing / CLR process management remains operationally fragile and needs more soak testing

## Known Hacks / TODOs

- **Deploy mask blocks building undeploy** (`openra_env.py`): The deploy action mask excludes known building types (`fact`, `afld`, `weap`, etc.) from undeploying via DeployTransform. This prevents the agent from constantly undeploying the Construction Yard and canceling in-progress production. In a real game, undeploying to relocate the base is a valid strategy. **TODO**: Remove the building-type blocklist and let the agent learn the cost of interrupting production via reward penalties (e.g. production-cancel penalty, time-waste penalty).
- **Building queue single-item guard** (`PythonAPI.cs`): `SendActions` suppresses `StartProduction` for building-type items when the target queue already has an item (in-progress or Done). This prevents the agent from accidentally overwriting completed items before they can be placed. Unit-type queues (infantry, vehicle) are unaffected. **TODO**: Consider whether this should eventually be relaxed for advanced queue management strategies.
- **Per-category produce mask** (`openra_env.py`): The `produce_unit_type_mask` blocks unit types whose production queue category (e.g. "building") is already occupied. This is the soft counterpart of the C#-side guard above. **TODO**: Re-evaluate once the agent can reliably complete the produce→build cycle.

## Practical Recommendations

- Use `feature` observations first when debugging action execution.
- For remote debugging, start with `scripts/remote_rule_based.py` or `scripts/remote_ppo_debug.py` before running long PPO training jobs.
- Treat the current PPO stack as a baseline to iterate on, not a final trainer.
- For current PPO experiments, prefer `scripts/train_best.ps1` defaults: entity obs, macro actions, asset reward, headless, no BC action head.
- Track `mean_reward`, `last20`, `KL`, `entropy`, `batches`, `decision_steps`, `atype_dist`, and `reward_comp`; reward alone hides collapse.
- If policy quality is the goal, the next highest-leverage work is a stronger teacher / goal-conditioned strategy prior / soft teacher-KL, not a larger network.
- If sample efficiency is the goal, optimize state extraction and keep hardening multi-env headless rollout.

## Troubleshooting

- Local start fails: check `bin_dir`, `mod_id`, and `map_uid`, and confirm the OpenRA build artifacts exist.
- Python cannot load the engine: verify `OpenRA.runtimeconfig.json` and the required assemblies are present in `bin_dir`.
- Remote join enters the lobby but does not stay synchronized: rebuild OpenRA after syncing `utils/PythonAPI.cs`, because remote-lobby behavior depends on the latest bridge code.
- Production/build actions appear invalid: inspect `production`, `placeable_areas`, `available_orders`, and `available_order_ids` from `feature` observations first.
- Actor indices behave strangely: remember that `unit_idx` and `target_idx` are mapped through the latest cached unit-id lists, not raw actor IDs.
- A transformable building appears to have `Move`: check `available_order_ids` vs `available_orders`. The raw field may still include transform-related move orders, while the filtered field is the one intended for control logic.

## Architecture Analysis (2026/06)

This section documents a deep analysis of the current architecture, focusing on environment interface issues and the gaps between the current baseline and an AlphaStar-style RTS RL agent.

### Current Strengths

- **Complete closed loop**: pythonnet bridge → state extraction → Gym env → PPO training all wired up end-to-end
- **Rich action masking**: engine-side feasibility checks for move/attack/deploy, queue-state-aware produce/build masks
- **Multiple connection modes**: local game, host-local lobby, remote server join
- **Auto-placement**: MCV auto-deploy and Done-item auto-place reduce action complexity

### Environment Interface Issues

#### 1. Observation: Fixed-Size Flat Vector is the Wrong Representation

The current `observation_type="vector"` uses a fixed 100-slot bin for both friendly and enemy units ([openra_env.py:1276-1304](envs/openra_env.py#L1276-L1304)). This has several critical flaws:

- **Fixed capacity truncation**: RTS unit counts vary from 1 (start) to 50+ (late game). Fixed 100 slots waste space early and may overflow late.
- **No relational information**: Units are encoded independently — the network cannot learn which units are fighting which.
- **No terrain awareness**: The vector obs lacks terrain type, passability, and fog-of-war boundary information.
- **Order-dependent encoding**: Unit order in the fixed slots changes across ticks, making it hard for the network to track identities.

**AlphaStar's approach**: Entity-based attention (Transformer over variable-length entity list) + spatial grid encoder (ResNet over minimap), which naturally handles variable entity counts and preserves spatial/relational structure.

#### 2. Action Space: Factorized Categorical Heads are Insufficient

The current action space is `MultiDiscrete([action_type, unit_idx, target_x, target_y, target_idx, unit_type])` — 6 independently-sampled categorical heads.

- **`target_x` / `target_y` are independently factorized** ([openra_env.py:1167-1173](envs/openra_env.py#L1167-L1173)): Valid x-coordinates can be paired with invalid y-coordinates, producing impossible targets. This should be a joint 2D spatial distribution.
- **No hierarchical conditioning**: All 6 heads are predicted from the same shared feature vector without autoregressive conditioning. The network cannot learn that `target` depends on `unit_selection` and `action_type`.
- **`unit_idx` is a fixed categorical**: 100-way classification over slots. Unit ordering changes across ticks, so index 5 means different units at different times.
- **Missing action types**: No `harvest`, `repair`, `sell`, `guard`, `stop`, or `cancel_production` actions.

**AlphaStar's approach**: Hierarchical autoregressive action space (`action_type → unit_selection → target`) with pointer networks for entity selection and 2D spatial logits maps for spatial targets.

#### 3. State Extraction is a Performance Bottleneck

`PythonAPI.GetState()` ([PythonAPI.cs:459-651](../OpenRA.Game/PythonAPI.cs#L459-L651)) rebuilds the entire `RLState` every call:

- **Heavy reflection usage**: Economy traits (`PlayerResources`, `PowerManager`), production queues, and building placement are all accessed via reflection. Each `GetState()` call repeats `Assembly.GetType()` scans and `MethodInfo` lookups.
- **Full map scan for placeable areas**: `CollectPlaceableAreas()` iterates every cell on the map (`map.AllCells`) and checks `CanPlaceBuilding` + `IsCloseEnoughToBase` for each Done item type. This is O(cells × building_types).
- **No incremental updates**: Even if only one tick passed and nothing changed, the entire state is rebuild from scratch.

**Target**: Cache reflection calls in static fields, use incremental state updates, and optimize spatial queries.

#### 4. Rollout Throughput and Sample Efficiency

`SubprocVecEnv` now supports spawned headless workers, so the old single-environment hard limit is gone. This improves wall-clock sample throughput, but recent 4-env and 8-env experiments still did not produce a decisive improvement over the scripted teacher. The remaining issue is algorithmic and task-level: on-policy PPO still has high variance on long-horizon RTS development, especially when the reward is development-only and the teacher already covers the easy macro path.

**Target**: Keep hardening 8-64 parallel environments, but pair scale with stronger priors, teacher-KL, goal conditioning, and combat / win-loss rewards.

#### 5. Reward Design is Development-Only

The current reward ([openra_env.py:407-467](envs/openra_env.py#L407-L467)) only incentivizes:
- Unit/building count increase
- Starting new production
- MCV deployment

Critically missing:
- **Combat rewards**: Damage dealt, enemy units killed
- **Win/loss signal**: The most important reward in any competitive game
- **Resource efficiency**: Ore collection rate, not just total
- **Map control / exploration**: Reconnaissance value

### Engine-Side Issues

| Issue | Location | Impact |
|-------|----------|--------|
| Reflection not cached | `CollectProductionInfo()`, `CollectPlaceableAreas()` | ~50-200ms per `GetState()` call |
| Full-cell-scan for build areas | `CollectPlaceableAreas()` map.AllCells loop | O(cells × types) per call |
| Building queue single-item guard | `SendActions()` `IsBuildingQueueOccupied` | Prevents advanced queue management |
| Deploy mask blocklist | `openra_env.py` `_building_types` hardcoded set | Prevents agent from learning base relocation |

### Gaps vs. AlphaStar

| Dimension | Current OpenRA-Bot | AlphaStar |
|-----------|-------------------|-----------|
| **Entity encoding** | Fixed 100-slot bin, order-dependent | Transformer over variable-length entity list |
| **Spatial encoding** | 128×128×10 semantic map (channels mostly empty) | ResNet over rich minimap + camera view |
| **Action structure** | Independent 6-way MultiDiscrete categoricals | Hierarchical autoregressive with pointer networks + 2D spatial heads |
| **Unit selection** | Fixed categorical over 100 slots | Attention-based pointer network over entity embeddings |
| **Spatial target** | Independent x, y categoricals (validity broken) | 2D logits map with spatial masking |
| **Network core** | MLP-based encoder + optional LSTM | Deep LSTM + Transformer + ResNet |
| **Training parallelism** | Headless `SubprocVecEnv` works at small scale; needs hardening and better sample efficiency | Thousands of parallel environments (TPU pods) |
| **Opponents** | Single built-in bot | Self-play league with PFSP, historical agents, exploiter agents |
| **Reward** | Development shaping only | Win/loss + game statistics |
| **Pre-training** | None (random init) | Supervised pre-training on human/expert replays |
| **Curriculum** | Fixed map | Progressive difficulty + map diversity |

---

## Roadmap to AlphaStar-Style Agent

See [PLAN.md](PLAN.md) for the full implementation plan. Below is a high-level summary:

### Phase 0: Baseline Hardening (1-2 weeks)
- Fix `target_x`/`target_y` independent factorization → joint 2D mask
- Complete `unit_types` mapping from CSV or game data
- Add combat + win/loss rewards
- Establish performance baselines (random, rule-based, PPO, built-in bot)

### Phase 1: Observation Upgrade (3-4 weeks)
- Entity-based observation builder (variable-length, feature-rich)
- Extended spatial observation with terrain, fog, threat channels
- Scalar observation (economy, power, game time)
- Dict-based `gym.spaces.Dict` observation space
- Backward compatible with `observation_type="vector"`

### Phase 2: Action Space Upgrade (3-4 weeks)
- Hierarchical autoregressive action structure
- Spatial action head (2D logits map, not factorized x/y)
- Pointer network for unit selection (attention-based)
- Joint 2D spatial action masks
- Expanded action types (harvest, repair, sell, guard, stop)

### Phase 3: Network Architecture (3-4 weeks)
- Entity Transformer encoder (self-attention over variable-length entities)
- Spatial ResNet encoder (deeper, richer than current CNN)
- AlphaStarActorCritic: integrated encoder + LSTM + hierarchical head
- Maintain backward compatibility with old `ActorCritic`

### Phase 4: Training Infrastructure (4-6 weeks)
- `SubprocVecEnv` for parallel environments (target: 8-64 envs)
- Self-play environment (two Python-controlled players)
- Elo rating evaluation framework
- 5x+ throughput improvement over single-env

### Phase 5: Training Strategy (4-6 weeks)
- League training with PFSP (Prioritized Fictitious Self-Play)
- Modular reward system (win/loss, combat, economy, exploration)
- Curriculum learning (economy-only → static opponent → full combat → map generalization)
- Supervised pre-training via behavior cloning from built-in bots

### Phase 6: Engine Optimization (2-3 weeks)
- Cache reflection calls in static fields → ~10x speedup for GetState()
- Incremental state updates (`GetStateDelta()`)
- Batch order feasibility checks
- Target: GetState() < 10ms (currently 50-200ms)

### Key Architecture Decision Points

1. **Parallelism strategy**: Python multiprocessing with `spawn` start method (each subprocess loads its own CLR) vs. single-process multi-instance (requires engine-side changes)

2. **Transformer scale**: Start with 3 layers / 4 heads / 256-dim model. Scale up only after training stability is proven.

3. **Action space granularity**: Start with 8-10 action types. Add more only when the agent masters the basics.

4. **Curriculum vs. end-to-end**: Curriculum (Phase 5) is recommended for training stability, but the architecture should support end-to-end from day one.

5. **Self-play vs. fixed opponents**: Start with built-in bots for baseline, then add self-play gradually. Full league training is the last piece.

---

## License

MIT
