# CLAUDE.md — OpenRA-Bot Project Guide

## Next Session Handoff

2026-06-27: All 4 experiments (100 updates each) completed. **Goal conditioning
is the clear breakthrough** — 2.5x improvement over baseline.

### Results Summary

| Experiment    | best      | final     | last20    |
|--------------|-----------|-----------|-----------|
| Baseline     | 0.0805@57 | 0.0386    | 0.0471    |
| Teacher-KL   | 0.0824@46 | 0.0578    | 0.0578    |
| **Goal**     | **0.2046@92** | **0.1507** | **0.1474** |
| Goal+Teacher | 0.1935@61 | 0.0717    | 0.1085    |

Teacher-KL alone: +50% final reward, prevents collapse, but doesn't increase peak.
Goal conditioning alone: **2.5x peak**, stable last20, the clear winner.
Goal+Teacher: slightly lower peak, teacher constrains when goal signal exists.

### Recommended launcher (best known config)

```bash
python scripts/train_rl.py --num-steps 256 --total-updates 100 \
    --observation-type entity --action-space-mode macro --headless \
    --warmstart-episodes 10 --warmstart-epochs 15 \
    --goal-conditioning --goal-aligned-weight 0.4 \
    --log-dir checkpoints_goal_w04
```

### Do next

1. Try `--goal-aligned-weight 0.6` or `0.8` — trend suggests higher=better (0.2→0.31, 0.4→?)
2. Try goal-only (no BC action head): `--no-load-bc-action-head --goal-conditioning --goal-aligned-weight 0.4`
3. If engine bot works on Windows: `--add-opponent --goal-conditioning` (code ready, macOS blocked)
4. Per PLAN.md priority 5: split reward logging + multi-value-head for goal components

### Already implemented

- ✅ Best checkpoint + enhanced CSV (early_stop, last20, best_reward, etc.)
- ✅ Soft teacher-KL on action_type head (--teacher-kl-coef)
- ✅ Goal conditioning with 4 build-order goals (--goal-conditioning)
- ✅ Opponent + kill reward code (--add-opponent, blocked by macOS bot hang)
- ✅ 6 unit tests pass for goal_library.py

## Quick Start

```bash
cd /Users/sharedcare/Projects/OpenRA/OpenRA-Bot

# Flat vector observation (original)
python scripts/train_rl.py --num-steps 512 --total-updates 80 \
    --max-episode-ticks 1500 --warmstart-episodes 10 --warmstart-epochs 15

# Entity observation (new, 25K params vs 580K, better BC accuracy)
python scripts/train_rl.py --num-steps 256 --total-updates 50 \
    --max-episode-ticks 1500 --warmstart-episodes 10 --warmstart-epochs 15 \
    --observation-type entity

# Macro action space (collapses 6-head -> single produce:<type> categorical, dev-only)
# Default reward is now "asset" (open-ended net-worth growth, no build-order cap).
python scripts/train_rl.py --observation-type entity --num-steps 256 \
    --total-updates 50 --max-episode-ticks 1500 --warmstart-episodes 8 \
    --warmstart-epochs 15 --action-space-mode macro --no-freeze-encoder --ent-coef 0.05
```

## Project Overview

RL agent for OpenRA RTS game using PPO + pythonnet bridge. Training stable with both flat-vector and entity-based observations. BC warm-start achieves 100% accuracy with entity encoder. PPO matches RuleBasedAgent baseline but cannot yet exceed it — credit assignment bottleneck (15 decision points per 300-step episode).

## Key Files

| File | Purpose | Key Functions |
|------|---------|---------------|
| `envs/openra_env.py` | Gym environment | `_compute_reward`, `_get_action_mask`, `_is_building`, `BuildOrderTracker` |
| `agent/agent.py` | PPO training loop | `train()`, `_sample_action`, `_build_effective_masks` |
| `models/actor.py` | Network architecture | `masked_logits` (must use `-inf`), `ActorCritic`, `MultiDiscretePolicy` |
| `models/buffer.py` | Rollout buffer | `compute_advantages`, `recurrent_mini_batch_generator` |
| `scripts/train_rl.py` | Training entry point | Hyperparameters, warm-start integration |
| `scripts/warmstart.py` | BC pre-training | `collect_demonstrations`, `pretrain_policy` |
| `utils/PythonAPI.cs` | C# bridge | State extraction, action dispatch (heavy reflection — fragile) |

## Critical Rules

1. **Mask penalty must be `-inf`**, never `log(clamp(1e-6))`. See models/actor.py `masked_logits` and agent/agent.py `_masked_logits`.
2. **Validate reward changes with RuleBasedAgent** — if cumulative reward isn't increasing, the reward function is broken.
3. **Use feed-forward model + frozen encoder with BC warmstart** — LSTM random weights destroy BC features.
4. **Monitor KL, entropy, and atype_dist** — KL > 1.0 means collapse, any single action > 300/512 means spam.
5. **ARM64 specific**: `OpenRA.runtimeconfig.json` needs `"rollForward": "LatestMajor"`. C# reflection for IMove is broken on ARM64 — use type whitelist instead.

## Related Documents

- `REPORT.md` — detailed experiment log and findings
- `PLAN.md` — long-term AlphaStar-style architecture roadmap
- `README.md` — project overview and usage

## Common Issues

**Training doesn't start**: Check `bin/OpenRA.runtimeconfig.json` has rollForward. Check `bin_dir` path in train_rl.py.

**KL explodes**: Reduce LR, ensure `-inf` mask penalty, check vf_coef ≤ 0.01.

**Reward always zero**: Check MCV type detection — verify `_is_building` returns False for 'mcv' type.

**Warmstart fails**: Ensure `collect_demonstrations` uses `build_observation()` for fresh state, not `_last_raw_state`.
