"""View best checkpoint playing in OpenRA with rendering enabled.

Usage:
    python scripts/view_best.py [--bin-dir /path/to/OpenRA/bin] [--steps 500]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.openra_env import make_env  # noqa: E402
from models.actor import ActorCritic  # noqa: E402
from agent.agent import RuleBasedAgent  # noqa: E402


def make_model_from_env(env, device: str = "cpu") -> ActorCritic:
    """Recreate the same ActorCritic that was used during training."""
    if hasattr(env.observation_space, 'spaces') and 'entities' in env.observation_space.spaces:
        obs_space = {
            "entity_dim": int(env.observation_space['entities'].shape[1]),
            "scalar_dim": int(env.observation_space['scalar'].shape[0]),
        }
        obs_type = "entity"
    elif hasattr(env.observation_space, "shape"):
        obs_space = {"vector": env.observation_space.shape[0]}
        obs_type = "vector"
    else:
        raise RuntimeError("Unknown observation space")

    model = ActorCritic(
        obs_space=obs_space,
        action_dims=tuple(env.action_space.nvec),
        observation_type=obs_type,
        recurrent_type=None,
    )
    model.to(device)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="View best checkpoint playing")
    parser.add_argument("--bin-dir", default="/Users/sharedcare/Projects/OpenRA/bin")
    parser.add_argument("--mod-id", default="ra")
    parser.add_argument("--map-uid", default="b53e25e007666442dbf62b87eec7bfbe8160ef3f")
    parser.add_argument("--checkpoint",
                        default="/Users/sharedcare/Projects/OpenRA/checkpoints_goal_w06/model_best.pth",
                        help="Path to model .pth file")
    parser.add_argument("--steps", type=int, default=500,
                        help="Max environment steps (10 ticks each)")
    parser.add_argument("--ticks-per-step", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--delay", type=float, default=0.05,
                        help="Seconds to sleep between steps (0=fast, 0.5=watchable)")
    parser.add_argument("--stochastic", action="store_true",
                        help="Sample from distribution instead of greedy argmax")
    args = parser.parse_args()

    env = make_env(
        bin_dir=args.bin_dir,
        mod_id=args.mod_id,
        map_uid=args.map_uid,
        ticks_per_step=args.ticks_per_step,
        observation_type="entity",
        action_space_mode="macro",
        headless=False,  # show the game window!
        goal_conditioning=True,
        goal_aligned_weight=0.6,
        max_episode_ticks=args.steps * args.ticks_per_step,
    )

    print(f"Loading checkpoint: {args.checkpoint}")
    model = make_model_from_env(env, args.device)
    state = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    # Handle architecture drift: checkpoint may lack newer components (GLU gate, multi-value-heads)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  (skipped {len(missing)} new keys not in checkpoint, e.g. {missing[:3]})")
    if unexpected:
        print(f"  (skipped {len(unexpected)} old keys not in model)")
    print("Model loaded.")

    obs, info = env.reset()
    total_reward = 0.0
    action_counts: Dict[int, int] = {}

    print(f"\n=== Playing {args.steps} steps ===")
    print(f"Actions: {env.action_types[:10]}...")
    print()

    for step in range(args.steps):
        # Build observation tensor
        x = {k: torch.from_numpy(v).unsqueeze(0).to(args.device) for k, v in obs.items()}
        with torch.no_grad():
            logits, values, _, _ = model(x, seq_len=1)

        # Apply action masks (convert numpy -> torch)
        masks = info.get('action_mask', {})
        if masks:
            torch_masks = {k: torch.from_numpy(v).to(args.device) if isinstance(v, np.ndarray) else v
                          for k, v in masks.items()}
            logits = model.policy_head.masked_logits(logits, torch_masks)

        # Action probabilities
        atype_logits = logits['action_type'][0]
        probs = torch.softmax(atype_logits, dim=-1)

        # Action selection: greedy or stochastic
        if args.stochastic:
            action_type = int(torch.multinomial(probs, 1).item())
        else:
            action_type = int(torch.argmax(atype_logits).item())
        action_counts[action_type] = action_counts.get(action_type, 0) + 1

        # Build full action (macro: only action_type matters)
        action = np.zeros(len(env.action_space.nvec), dtype=np.int64)
        action[0] = action_type

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)

        if step % 25 == 0 or step < 5:
            comps = info.get('reward_components', {})
            # Top-5 actions
            top5 = torch.topk(probs, min(5, len(probs)))
            top5_str = " | ".join(f"{env.action_types[i]:<20s} {p:.2f}"
                                  for i, p in zip(top5.indices.tolist(), top5.values.tolist()))
            goal_name = env._active_goal.name if env._active_goal else '?'
            print(f"\n--- step {step:3d} | goal={goal_name} ---")
            print(f"  top-5: {top5_str}")
            print(f"  picked: {env.action_types[action_type]:<20s} reward={reward:.4f} total={total_reward:.2f}")
            print(f"  goal: bld={comps.get('goal_building_progress',0):.2f} "
                  f"unit={comps.get('goal_unit_progress',0):.2f} "
                  f"phase2={comps.get('goal_phase2',0):.1f} "
                  f"cash={comps.get('cash',0):.0f}")

        if args.delay > 0:
            time.sleep(args.delay)

        if terminated or truncated:
            print(f"Episode ended at step {step}")
            break

    print(f"\n=== Done ===")
    print(f"Total reward: {total_reward:.3f}")
    comps = info.get('reward_components', {})
    print(f"Goal: {env._active_goal.name if env._active_goal else '?'}")
    print(f"  building_progress: {comps.get('goal_building_progress',0):.2%}")
    print(f"  unit_progress:     {comps.get('goal_unit_progress',0):.2%}")
    print(f"  phase2_frac:       {comps.get('goal_phase2',0):.1%}")
    print(f"\nAction distribution:")
    for k in sorted(action_counts.keys(), key=lambda x: action_counts[x], reverse=True):
        pct = action_counts[k] / max(1, sum(action_counts.values())) * 100
        print(f"  {env.action_types[k]:<25s} x{action_counts[k]:>3d} ({pct:5.1f}%)")

    print("\nPress Enter to close the game window...")
    input()
    env.close()


if __name__ == "__main__":
    main()
