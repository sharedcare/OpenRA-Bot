import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.openra_env import make_env
from agent.agent import RandomMoveAgent


if __name__ == "__main__":
    # Example: Start a local game via PythonAPI.StartLocalGame and drive it with a simple agent
    env = make_env(
        bin_dir="F:/Projects/OpenRA/bin",
        mod_id="ra",
        map_uid="b53e25e007666442dbf62b87eec7bfbe8160ef3f",
        ticks_per_step=5,
    )

    # Use StartLocalGame by default (do not configure remote/host unless desired)
    # To host a lobby instead, uncomment:
    # env.configure_host(options=["option gamespeed default", "name PythonAgent", "slot Multi0", "state 1"]) 
    # To join a remote lobby instead, uncomment:
    # env.configure_remote(host="10.10.10.120", port=1234, password="1234", spectator=False)

    random_move_agent = RandomMoveAgent()
    obs, info = env.reset()
    total_reward = 0
    for _ in range(1000):
        actions = random_move_agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(actions)
        if terminated or truncated:
            break
        total_reward += reward
    print(f"Total reward: {total_reward}")
    env.close()
