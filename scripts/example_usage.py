import sys
import os
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
    sample_types = [str(a.get("type", "")) for a in my_units[:5]]

    print(
        f"[obs] step={step} tick={obs.get('world_tick', 0)} "
        f"my_units={len(my_units)} enemy_units={len(enemy_units)} resources={len(resources)} "
        f"cash={obs.get('cash', 0)} stored={obs.get('resources_total', 0)} "
        f"power={power.get('provided', 0)}/{power.get('drained', 0)} state={power.get('state', '')}"
    )
    if sample_types:
        print(f"[obs] sample_my_types={sample_types}")
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


if __name__ == "__main__":
    # Example: Start a local game via PythonAPI.StartLocalGame and drive it with a simple agent
    env = make_env(
        bin_dir="F:/Projects/OpenRA/bin",
        mod_id="ra",
        map_uid="b53e25e007666442dbf62b87eec7bfbe8160ef3f",
        ticks_per_step=10,
        observation_type="feature",
        enable_actions=['noop','move','attack','produce','build','deploy'],
    )

    # Use StartLocalGame by default (do not configure remote/host unless desired)
    # To host a lobby instead, uncomment:
    # env.configure_host(options=["option gamespeed default", "name PythonAgent", "slot Multi0", "state 1"]) 
    # To join a remote lobby instead, uncomment:
    # env.configure_remote(host="10.10.10.120", port=1234, password="1234", spectator=False)

    rule_based_agent = RuleBasedAgent()
    obs, info = env.reset()
    print_obs_summary(obs, info, step=0, action_types=env.action_types)
    total_reward = 0
    step = 0
    while True:
        step += 1
        actions = rule_based_agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(actions)
        total_reward += reward
        print(f"[step] step={step} reward={reward:.4f} actions={actions}")
        print_obs_summary(obs, info, step=step, action_types=env.action_types)
        if terminated or truncated:
            break
    print(f"Total reward: {total_reward}")
    env.close()
