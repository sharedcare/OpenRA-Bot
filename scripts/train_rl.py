import os
import sys
from typing import Tuple

import numpy as np
import torch
import torch.optim as optim

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from OpenRA.Bot.envs.openra_env import OpenRAEnv, make_env
    from OpenRA.Bot.models import ActorCritic
except Exception:  # noqa: BLE001
    # Fallback when running directly from source without installed package
    from envs.openra_env import OpenRAEnv, make_env  # type: ignore
    from models import ActorCritic  # type: ignore


def make_model(env: OpenRAEnv, observation_type: str = "vector") -> Tuple[ActorCritic, Tuple[int, int, int, int, int, int]]:
    # Infer action dims from env
    a_dims = (
        len(env.action_types),
        env.action_space.nvec[1],
        env.action_space.nvec[2],
        env.action_space.nvec[3],
        env.action_space.nvec[4],
        env.action_space.nvec[5],
    )
    if observation_type == "vector":
        obs_space = {"vector": int(env.observation_space.shape[0])}
    else:
        obs_space = {"channels": int(env.observation_space.shape[-1])}
    model = ActorCritic(obs_space=obs_space, action_dims=a_dims, observation_type=observation_type)
    return model, a_dims


def compute_returns(rewards, dones, values, gamma=0.99, lam=0.95):
    # Simple GAE-lambda for on-policy updates
    adv = np.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(len(rewards))):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * nonterminal - values[t]
        adv[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    ret = adv + values[:-1]
    return adv, ret


def train(num_steps: int = 2048, total_updates: int = 10, observation_type: str = "vector"):
    env = make_env(
        bin_dir="F:/Projects/OpenRA/bin",
        mod_id="ra",
        map_uid="b53e25e007666442dbf62b87eec7bfbe8160ef3f",
        ticks_per_step=5,
        observation_type=observation_type,
        enable_actions=['noop','move','attack','produce','build','deploy'],
    )
    model, _ = make_model(env, observation_type=observation_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    obs, info = env.reset()

    for update in range(total_updates):
        batch_obs = []
        batch_actions = []
        batch_logps = []
        batch_rewards = []
        batch_dones = []
        batch_values = []

        for _ in range(num_steps):
            x = torch.as_tensor(obs, device=device).unsqueeze(0)
            logits, value = model(x)
            # Masks
            masks = {}
            for k, v in (info.get("action_mask", {}) if isinstance(info, dict) else {}).items():
                t = torch.as_tensor(v, device=device)
                if t.dim() == 1:
                    t = t.unsqueeze(0)
                masks[k] = t
            action, logps = model.policy.sample(logits, masks=masks)
            action_np = action.squeeze(0).cpu().numpy()

            next_obs, reward, terminated, truncated, next_info = env.step(action_np)

            batch_obs.append(obs)
            batch_actions.append(action_np)
            batch_logps.append(sum(lp.item() for lp in logps.values()))
            batch_rewards.append(reward)
            batch_dones.append(float(terminated or truncated))
            batch_values.append(value.item())

            obs, info = next_obs, next_info
            if terminated or truncated:
                obs, info = env.reset()

        # Bootstrap value
        with torch.no_grad():
            x = torch.as_tensor(obs, device=device).unsqueeze(0)
            _, last_v = model(x)
            v_last = float(last_v.item())
        values = np.array(batch_values + [v_last], dtype=np.float32)
        rewards = np.array(batch_rewards, dtype=np.float32)
        dones = np.array(batch_dones, dtype=np.float32)

        adv, ret = compute_returns(rewards, dones, values)
        adv_t = torch.as_tensor(adv, device=device)
        ret_t = torch.as_tensor(ret, device=device)

        # Policy loss and value loss (very small PPO-like without clip for brevity)
        obs_t = torch.as_tensor(np.array(batch_obs), device=device)
        logits, values_pred = model(obs_t)

        # Log prob recompute
        heads = ["action_type", "unit_idx", "target_x", "target_y", "target_idx", "unit_type"]
        total_logp = 0.0
        for i, h in enumerate(heads):
            dist = torch.distributions.Categorical(logits=logits[h])
            a = torch.as_tensor(np.array(batch_actions))[:, i].to(device)
            total_logp = total_logp + dist.log_prob(a)

        loss_policy = -(total_logp * adv_t).mean()
        loss_value = torch.nn.functional.mse_loss(values_pred.squeeze(-1), ret_t)
        loss = loss_policy + 0.5 * loss_value

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        print(f"Update {update+1}/{total_updates}  loss={loss.item():.4f}  return={ret.mean():.2f}")

    env.close()


if __name__ == "__main__":
    train()
