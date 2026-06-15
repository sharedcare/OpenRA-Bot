# CLAUDE.md — OpenRA-Bot Project Guide

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
