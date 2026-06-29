"""View best checkpoint playing by joining a local multiplayer lobby.

Workflow:
  1. Start OpenRA manually: open the game → Multiplayer → Create → pick map/RA
  2. Run this script: it joins your lobby as a bot
  3. Add an AI opponent or another player
  4. Start the game — watch the model play

Usage:
  python scripts/view_remote.py --host 127.0.0.1 --port 1234

If no checkpoint is specified, uses the goal w=0.6 best model.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.openra_env import make_env  # noqa: E402
from models.actor import ActorCritic  # noqa: E402


def make_model_from_env(env, device: str = "cpu") -> ActorCritic:
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

    return ActorCritic(
        obs_space=obs_space,
        action_dims=tuple(env.action_space.nvec),
        observation_type=obs_type,
        recurrent_type=None,
    ).to(device).eval()


def print_state(obs: dict, info: dict, step: int, action_types: List[str],
                probs: torch.Tensor, action_type: int, reward: float,
                total_reward: float, goal_name: str) -> None:
    """Compact per-step status line."""
    atype_name = action_types[action_type] if action_type < len(action_types) else "?"
    top3 = torch.topk(probs, min(3, len(probs)))
    top3_str = " | ".join(f"{action_types[i]}({p:.2f})"
                          for i, p in zip(top3.indices.tolist(), top3.values.tolist()))
    comps = info.get('reward_components', {})
    bld = comps.get('goal_building_progress', 0)
    unit = comps.get('goal_unit_progress', 0)
    print(f"[{step:4d}] goal={goal_name:<10s} action={atype_name:<20s} r={reward:.4f} "
          f"Σr={total_reward:.2f} bld={bld:.2f} unit={unit:.2f}  top3=[{top3_str}]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View best model in multiplayer lobby")
    parser.add_argument("--bin-dir", default="/Users/sharedcare/Projects/OpenRA/bin")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1234)
    parser.add_argument("--password", default="")
    parser.add_argument("--slot", default=None, help="e.g., Multi1, Multi2")
    parser.add_argument("--checkpoint",
                        default="/Users/sharedcare/Projects/OpenRA/checkpoints_goal_w06/model_best.pth")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--ticks-per-step", type=int, default=10)
    parser.add_argument("--delay", type=float, default=0.0,
                        help="Sleep seconds between steps (0=fast, 0.5=watchable)")
    parser.add_argument("--stochastic", action="store_true",
                        help="Sample from distribution instead of greedy")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--verbose", action="store_true",
                        help="Print full observation dump each step")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    env = make_env(
        bin_dir=args.bin_dir,
        mod_id="ra",
        map_uid="b53e25e007666442dbf62b87eec7bfbe8160ef3f",
        ticks_per_step=args.ticks_per_step,
        observation_type="entity",
        action_space_mode="macro",
        goal_conditioning=True,
        goal_aligned_weight=0.6,
    )
    env.configure_remote(
        host=args.host,
        port=args.port,
        password=args.password,
        slot=args.slot,
        spectator=False,
    )

    print(f"Loading checkpoint: {args.checkpoint}")
    model = make_model_from_env(env, str(device))
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  (skipped {len(missing)} new keys, e.g. {missing[:2]})")
    print("Model loaded. Waiting for game to start...")

    obs, info = env.reset()
    total_reward = 0.0
    action_counts: Dict[int, int] = {}
    print(f"Goal: {env._active_goal.name if env._active_goal else '?'}")
    print(f"Actions: {env.action_types[:8]}...")
    print()

    for step in range(args.max_steps):
        x = {k: torch.from_numpy(v).unsqueeze(0).to(device) for k, v in obs.items()}
        with torch.no_grad():
            logits, values, _, _ = model(x, seq_len=1)

        masks = info.get('action_mask', {})
        if masks:
            torch_masks = {k: torch.from_numpy(v).to(device) if isinstance(v, np.ndarray) else v
                          for k, v in masks.items()}
            logits = model.policy_head.masked_logits(logits, torch_masks)

        atype_logits = logits['action_type'][0]
        probs = torch.softmax(atype_logits, dim=-1)

        if args.stochastic:
            action_type = int(torch.multinomial(probs, 1).item())
        else:
            action_type = int(torch.argmax(atype_logits).item())
        action_counts[action_type] = action_counts.get(action_type, 0) + 1

        action = np.zeros(len(env.action_space.nvec), dtype=np.int64)
        action[0] = action_type

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)

        goal_name = env._active_goal.name if env._active_goal else '?'
        print_state(obs, info, step, env.action_types, probs, action_type,
                    float(reward), total_reward, goal_name)

        if args.verbose:
            actors = obs.get('actors', [])
            my_owner = obs.get('my_owner', -1)
            my = [a for a in actors if a.get('owner', -1) == my_owner and not a.get('dead')]
            print(f"       my_units={len(my)} cash={obs.get('cash','?')} "
                  f"tick={obs.get('world_tick','?')}")

        if terminated or truncated:
            print(f"Game ended: terminated={terminated} truncated={truncated}")
            break

        if args.delay > 0:
            time.sleep(args.delay)

    print(f"\n=== Done: {step+1} steps, total_reward={total_reward:.3f} ===")
    if env._active_goal:
        print(f"Goal: {env._active_goal.name}")
    print("Action distribution:")
    for k in sorted(action_counts, key=lambda x: action_counts[x], reverse=True):
        pct = action_counts[k] / max(1, sum(action_counts.values())) * 100
        print(f"  {env.action_types[k]:<25s} x{action_counts[k]:>3d} ({pct:5.1f}%)")

    env.close()


if __name__ == "__main__":
    main()
