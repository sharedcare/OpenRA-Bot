import argparse
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.openra_env import make_env
from models import ActorCritic
from agent import PPOAgent


def print_obs_summary(obs, info, step: int, action_types) -> None:
    if not isinstance(obs, dict):
        print(f"[obs] step={step} type={type(obs).__name__}")
        return

    actors = obs.get("actors") or []
    resources = obs.get("resources") or []
    production = (obs.get("production") or {}).get("Queues") or []
    power = obs.get("power") or {}
    my_owner = int(obs.get("my_owner", -1))
    my_units = [a for a in actors if int(a.get("owner", -1)) == my_owner and not bool(a.get("dead", False))]
    enemy_units = [a for a in actors if int(a.get("owner", -1)) != my_owner and not bool(a.get("dead", False))]

    print(
        f"[obs] step={step} tick={obs.get('world_tick', 0)} "
        f"my_units={len(my_units)} enemy_units={len(enemy_units)} resources={len(resources)} "
        f"cash={obs.get('cash', 0)} stored={obs.get('resources_total', 0)} "
        f"power={power.get('provided', 0)}/{power.get('drained', 0)} state={power.get('state', '')}"
    )
    if production:
        queue_sizes = [len(q.get("Items") or []) for q in production[:3]]
        print(f"[obs] production_queues={len(production)} queue_sizes={queue_sizes}")

    action_mask = (info or {}).get("action_mask", {})
    if action_mask:
        enabled_actions = [
            name for idx, name in enumerate(action_types)
            if idx < len(action_mask.get("action_type", [])) and action_mask["action_type"][idx]
        ]
        print(f"[obs] enabled_actions={enabled_actions}")


def print_production_debug(obs, info, action: np.ndarray, decoded: Dict[str, Any], step: int) -> None:
    production = (obs.get("production") or {}).get("Queues") or []
    placeable_areas = obs.get("placeable_areas") or {}
    catalog = obs.get("producible_catalog") or []
    print(f"[debug] step={step} raw_action={action.tolist() if hasattr(action, 'tolist') else action}")
    print(f"[debug] decoded_action={decoded}")
    print(f"[debug] catalog={[str(x.get('Name', '')) for x in catalog]}")

    if not production:
        print("[debug] production_queues=[]")
    else:
        for idx, queue in enumerate(production):
            q_actor = queue.get("ActorId")
            q_type = queue.get("Type")
            q_group = queue.get("Group")
            q_enabled = queue.get("Enabled")
            items = queue.get("Items") or []
            producible = [str(x.get("Name", "")) for x in (queue.get("Producible") or [])]
            print(
                f"[debug] queue[{idx}] actor={q_actor} type={q_type} group={q_group} "
                f"enabled={q_enabled} items={items} producible={producible}"
            )

    if placeable_areas:
        summary = {k: len(v or []) for k, v in placeable_areas.items()}
        print(f"[debug] placeable_areas={summary}")
    else:
        print("[debug] placeable_areas={}")

    action_mask = (info or {}).get("action_mask", {})
    if action_mask:
        for key in (
            "action_type",
            "produce_queue_mask",
            "produce_unit_type_mask",
            "build_unit_type_mask",
            "build_mask",
            "move_mask",
            "deploy_mask",
        ):
            if key in action_mask:
                values = action_mask[key]
                try:
                    enabled = [i for i, x in enumerate(values) if x]
                except Exception:
                    enabled = []
                print(f"[debug] {key} enabled_indices={enabled}")


def make_model(env, observation_type: str = "vector", recurrent_type: str = "lstm") -> ActorCritic:
    action_dims = (
        len(env.action_types),
        int(env.action_space.nvec[1]),
        int(env.action_space.nvec[2]),
        int(env.action_space.nvec[3]),
        int(env.action_space.nvec[4]),
        int(env.action_space.nvec[5]),
    )
    if observation_type == "vector":
        obs_space = {"vector": int(env.observation_space.shape[0])}
    else:
        obs_space = {"channels": int(env.observation_space.shape[-1])}
    return ActorCritic(
        obs_space=obs_space,
        action_dims=action_dims,
        observation_type=observation_type,
        recurrent_type=recurrent_type,
    )


def decode_action(env, action: np.ndarray) -> Dict[str, Any]:
    raw = [int(x) for x in action.tolist()]
    atype_idx, unit_idx, tx, ty, target_idx, unit_type_idx = raw
    action_type = env.action_types[atype_idx] if 0 <= atype_idx < len(env.action_types) else "noop"
    decoded: Dict[str, Any] = {
        "action_type": action_type,
        "unit_idx": unit_idx,
        "target_x": tx,
        "target_y": ty,
        "target_idx": target_idx,
        "unit_type_idx": unit_type_idx,
    }

    if action_type in ("move", "deploy"):
        decoded["resolved_actor_id"] = env._resolve_my_unit_id(unit_idx)
    elif action_type == "attack":
        decoded["resolved_actor_id"] = env._resolve_my_unit_id(unit_idx)
        decoded["resolved_target_id"] = env._resolve_enemy_unit_id(target_idx)
    elif action_type in ("produce", "build"):
        decoded["resolved_queue_actor_id"] = env._resolve_queue_actor_id(unit_idx)
        decoded["resolved_unit_type"] = env.reverse_unit_types.get(unit_type_idx, None)
    return decoded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Join a remote OpenRA lobby and debug PPOAgent actions.")
    parser.add_argument("--bin-dir", default="/Users/sharedcare/Projects/OpenRA/bin")
    parser.add_argument("--mod-id", default="ra")
    parser.add_argument("--map-uid", default="b53e25e007666442dbf62b87eec7bfbe8160ef3f")
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--password", default="")
    parser.add_argument("--slot", default=None)
    parser.add_argument("--ticks-per-step", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--sleep-ms", type=int, default=0)
    parser.add_argument("--checkpoint", default=None, help="Optional .pth checkpoint to load.")
    parser.add_argument("--observation-type", default="vector", choices=["vector", "image"])
    parser.add_argument("--recurrent-type", default="lstm", choices=["lstm", "gru", "none"])
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    env = make_env(
        bin_dir=args.bin_dir,
        mod_id=args.mod_id,
        map_uid=args.map_uid,
        ticks_per_step=args.ticks_per_step,
        observation_type=args.observation_type,
        enable_actions=["noop", "move", "attack", "produce", "build", "deploy"],
    )
    env.configure_remote(
        host=args.host,
        port=args.port,
        password=args.password,
        slot=args.slot,
        spectator=False,
    )

    recurrent_type = None if args.recurrent_type == "none" else args.recurrent_type
    model = make_model(env, observation_type=args.observation_type, recurrent_type=recurrent_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if args.checkpoint:
        state_dict = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print(f"[ppo] loaded checkpoint={args.checkpoint}")
    else:
        print("[ppo] no checkpoint provided, using randomly initialized weights")

    agent = PPOAgent(model=model, device=str(device))
    obs, info = env.reset()
    print("[remote] connected and in game")
    print_obs_summary(obs, info, step=0, action_types=env.action_types)

    total_reward = 0.0
    for step in range(1, args.max_steps + 1):
        action = agent.act(obs, info)
        decoded = decode_action(env, action)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        print(f"[step] step={step} reward={reward:.4f}")
        print_obs_summary(obs, info, step=step, action_types=env.action_types)
        print_production_debug(obs, info, action=action, decoded=decoded, step=step)

        if terminated or truncated:
            print(f"[remote] game finished terminated={terminated} truncated={truncated}")
            break

        if args.sleep_ms > 0:
            time.sleep(args.sleep_ms / 1000.0)

    print(f"[remote] total_reward={total_reward:.4f}")
    env.close()


if __name__ == "__main__":
    main()
