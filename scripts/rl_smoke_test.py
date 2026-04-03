import os
import sys
import torch

# Make package imports work when running from source
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from OpenRA.Bot.envs.openra_env import make_env
    from OpenRA.Bot.models import ActorCritic
    from OpenRA.Bot.agent import PPOAgent
except Exception:  # noqa: BLE001
    from envs.openra_env import make_env  # type: ignore
    from models import ActorCritic  # type: ignore
    from agent import PPOAgent  # type: ignore


def main():
    env = make_env(
        bin_dir="/Users/sharedcare/Projects/OpenRA/bin",
        mod_id="ra",
        map_uid="b53e25e007666442dbf62b87eec7bfbe8160ef3f",
        ticks_per_step=5,
        observation_type="vector",
        enable_actions=["noop", "move", "attack", "produce", "build", "deploy"],
    )
    observation_type = "vector"
    # Build model
    a_dims = (
        len(env.action_types),
        env.action_space.nvec[1],
        env.action_space.nvec[2],
        env.action_space.nvec[3],
        env.action_space.nvec[4],
        env.action_space.nvec[5],
    )
    obs_space = {"vector": int(env.observation_space.shape[0])}
    model = ActorCritic(obs_space=obs_space, action_dims=a_dims, observation_type=observation_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    agent = PPOAgent(model=model, device=str(device))

    obs, info = env.reset()
    total_reward = 0.0
    for step in range(50):
        action = agent.act(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"Step {step} reward={reward:.2f}")
        if terminated or truncated:
            break
    print(f"Total reward: {total_reward:.2f}")
    env.close()


if __name__ == "__main__":
    main()
