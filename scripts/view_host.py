"""Host a local game and let the best model play. User connects as spectator.

Workflow:
  1. Run this script: it hosts a game and waits for players
  2. Open your OpenRA client → Multiplayer → Direct Connect → 127.0.0.1:1234
  3. Join as Spectator (or player), then the game auto-starts
  4. Watch the model play in your OpenRA client window

Usage:
  python scripts/view_host.py [--steps 500] [--checkpoint ...]
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
    else:
        raise RuntimeError("Unsupported observation space")
    return ActorCritic(
        obs_space=obs_space,
        action_dims=tuple(env.action_space.nvec),
        observation_type=obs_type,
        recurrent_type=None,
    ).to(device).eval()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Host game with model agent, user spectates")
    parser.add_argument("--bin-dir", default="/Users/sharedcare/Projects/OpenRA/bin")
    parser.add_argument("--map-uid", default="b53e25e007666442dbf62b87eec7bfbe8160ef3f")
    parser.add_argument("--checkpoint",
                        default="/Users/sharedcare/Projects/OpenRA/checkpoints_goal_w06/model_best.pth")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--delay", type=float, default=0.05)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    # Host a local game — the engine creates a listen server on port 1234.
    # The user's OpenRA client connects as spectator.
    env = make_env(
        bin_dir=args.bin_dir,
        mod_id="ra",
        map_uid=args.map_uid,
        ticks_per_step=10,
        observation_type="entity",
        action_space_mode="macro",
        goal_conditioning=True,
        goal_aligned_weight=0.6,
        headless=True,  # python side headless; user watches via client
        max_episode_ticks=args.steps * 10,
    )
    # Host locally (creates a server the client can join)
    env.host_local = True
    env.remote_host = ""
    env.remote_port = 0

    print(f"Loading: {args.checkpoint}")
    model = make_model_from_env(env, str(device))
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=False)
    print("Model loaded.")

    print("\n>>> Open your OpenRA client → Multiplayer → Direct Connect → 127.0.0.1:1234")
    print(">>> Join as Spectator and start the game.\n")

    obs, info = env.reset()
    total_reward = 0.0
    action_counts: Dict[int, int] = {}
    goal_name = env._active_goal.name if env._active_goal else '?'

    for step in range(args.steps):
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

        at_name = env.action_types[action_type] if action_type < len(env.action_types) else "?"
        comps = info.get('reward_components', {})
        if step % 25 == 0 or step < 3:
            top3 = torch.topk(probs, min(3, len(probs)))
            top3_s = " | ".join(f"{env.action_types[i]}({p:.2f})"
                               for i, p in zip(top3.indices.tolist(), top3.values.tolist()))
            print(f"[{step:4d}] goal={goal_name}  action={at_name}  r={reward:.4f}  "
                  f"Σ={total_reward:.2f}  bld={comps.get('goal_building_progress',0):.2f}  "
                  f"unit={comps.get('goal_unit_progress',0):.2f}  top3=[{top3_s}]")

        if args.delay > 0:
            time.sleep(args.delay)
        if terminated or truncated:
            print(f"Game ended at step {step}")
            break

    print(f"\nTotal: {total_reward:.3f}  Goal: {goal_name}")
    print("Actions:")
    for k in sorted(action_counts, key=lambda x: action_counts[x], reverse=True):
        pct = action_counts[k] / max(1, sum(action_counts.values())) * 100
        print(f"  {env.action_types[k]:<25s} x{action_counts[k]:>3d} ({pct:5.1f}%)")
    env.close()


if __name__ == "__main__":
    main()
