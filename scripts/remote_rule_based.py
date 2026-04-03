import argparse
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.openra_env import make_env
from agent.agent import RuleBasedAgent


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Join a remote OpenRA lobby and run the rule-based agent.")
    parser.add_argument("--bin-dir", default="/Users/sharedcare/Projects/OpenRA/bin")
    parser.add_argument("--mod-id", default="ra")
    parser.add_argument("--map-uid", default="b53e25e007666442dbf62b87eec7bfbe8160ef3f")
    parser.add_argument("--host", required=True, help="Remote lobby host or IP.")
    parser.add_argument("--port", required=True, type=int, help="Remote lobby port.")
    parser.add_argument("--password", default="", help="Lobby password if required.")
    parser.add_argument("--slot", default=None, help="Preferred slot, for example Multi0.")
    parser.add_argument("--spectator", action="store_true", help="Join as spectator instead of player.")
    parser.add_argument("--ticks-per-step", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--sleep-ms", type=int, default=0, help="Optional sleep between steps for easier log reading.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = make_env(
        bin_dir=args.bin_dir,
        mod_id=args.mod_id,
        map_uid=args.map_uid,
        ticks_per_step=args.ticks_per_step,
        observation_type="feature",
        enable_actions=["noop", "move", "attack", "produce", "build", "deploy"],
    )
    env.configure_remote(
        host=args.host,
        port=args.port,
        password=args.password,
        slot=args.slot,
        spectator=args.spectator,
    )

    agent = RuleBasedAgent()
    obs, info = env.reset()
    print("[remote] connected and in game")
    print_obs_summary(obs, info, step=0, action_types=env.action_types)

    total_reward = 0.0
    step = 0
    while True:
        actions = [] if args.spectator else agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(actions)
        total_reward += reward
        print(f"[step] step={step} reward={reward:.4f} actions={actions}")
        print_obs_summary(obs, info, step=step, action_types=env.action_types)
        step += 1

        if terminated or truncated:
            print(f"[remote] game finished terminated={terminated} truncated={truncated}")
            break

        if args.sleep_ms > 0:
            time.sleep(args.sleep_ms / 1000.0)

    print(f"[remote] total_reward={total_reward:.4f}")
    env.close()


if __name__ == "__main__":
    main()
