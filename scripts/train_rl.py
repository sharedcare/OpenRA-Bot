import os
import sys
from typing import Tuple, Dict, Any, List

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
    model = ActorCritic(obs_space=obs_space, action_dims=a_dims, observation_type=observation_type)
    return model, a_dims


def gae_returns(rewards: np.ndarray, dones: np.ndarray, values: np.ndarray, gamma: float, lam: float) -> Tuple[np.ndarray, np.ndarray]:
    adv = np.zeros_like(rewards, dtype=np.float32)
    lastgaelam: float = 0.0
    for t in reversed(range(len(rewards))):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * nonterminal - values[t]
        lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        adv[t] = lastgaelam
    ret = adv + values[:-1]
    return adv, ret


def heads_order() -> List[str]:
    return ["action_type", "unit_idx", "target_x", "target_y", "target_idx", "unit_type"]


def masked_logits(logits: Dict[str, torch.Tensor], masks: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # Apply only action_type mask if provided; other heads currently unmasked
    out: Dict[str, torch.Tensor] = {}
    for k, lg in logits.items():
        if k == "action_type" and masks and (k in masks) and masks[k] is not None:
            m = masks[k].to(lg.dtype)
            add = torch.log(m.clamp(min=1e-6))
            out[k] = lg + add
        else:
            out[k] = lg
    return out


def logprob_and_entropy(logits: Dict[str, torch.Tensor], actions: torch.Tensor, masks: Dict[str, torch.Tensor] | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
    if masks is None:
        masks = {}
    total_logp = torch.zeros(actions.shape[0], device=actions.device)
    total_entropy = torch.zeros(actions.shape[0], device=actions.device)
    lg = masked_logits(logits, masks)
    for i, h in enumerate(heads_order()):
        dist = torch.distributions.Categorical(logits=lg[h])
        a = actions[:, i]
        total_logp = total_logp + dist.log_prob(a)
        total_entropy = total_entropy + dist.entropy()
    return total_logp, total_entropy


def train(
    num_steps: int = 2048,
    total_updates: int = 100,
    observation_type: str = "vector",
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_coef: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 1.0,
    learning_rate: float = 3e-4,
    update_epochs: int = 4,
    minibatch_size: int = 256,
    target_kl: float = 0.03,
):
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
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    obs, info = env.reset()

    for update in range(total_updates):
        obs_buf: List[np.ndarray] = []
        actions_buf: List[np.ndarray] = []
        logp_buf: List[float] = []
        val_buf: List[float] = []
        rew_buf: List[float] = []
        done_buf: List[float] = []
        mask_buf: List[Dict[str, np.ndarray]] = []

        for _ in range(num_steps):
            x = torch.as_tensor(obs, device=device).unsqueeze(0)
            logits, value = model(x)
            # Build masks from info if available
            masks_t: Dict[str, torch.Tensor] = {}
            if isinstance(info, dict):
                am = info.get("action_mask", {}) or {}
                if 'action_type' in am:
                    t = torch.as_tensor(am['action_type'], device=device)
                    if t.dim() == 1:
                        t = t.unsqueeze(0)
                    masks_t['action_type'] = t
            action_t, per_head_logps = model.policy.sample(logits, masks=masks_t)
            action_np = action_t.squeeze(0).cpu().numpy()

            # Store rollout
            obs_buf.append(obs)
            actions_buf.append(action_np)
            # Sum of per-head logps
            logp_buf.append(sum(lp.item() for lp in per_head_logps.values()))
            val_buf.append(float(value.item()))
            mask_buf.append({k: v.squeeze(0).detach().cpu().numpy() for k, v in masks_t.items()})

            # Step env
            next_obs, reward, terminated, truncated, next_info = env.step(action_np)
            rew_buf.append(float(reward))
            done_buf.append(float(terminated or truncated))

            obs, info = next_obs, next_info
            if terminated or truncated:
                obs, info = env.reset()

        # Bootstrap value
        with torch.no_grad():
            x = torch.as_tensor(obs, device=device).unsqueeze(0)
            _, last_v = model(x)
            v_last = float(last_v.item())

        values = np.array(val_buf + [v_last], dtype=np.float32)
        rewards = np.array(rew_buf, dtype=np.float32)
        dones = np.array(done_buf, dtype=np.float32)
        advantages, returns = gae_returns(rewards, dones, values, gamma=gamma, lam=gae_lambda)

        # Normalize advantages
        adv_mean = advantages.mean() if advantages.size > 0 else 0.0
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        # Flatten buffers into tensors
        b_obs = torch.as_tensor(np.array(obs_buf), device=device)
        b_actions = torch.as_tensor(np.array(actions_buf), device=device)
        b_logp = torch.as_tensor(np.array(logp_buf), device=device)
        b_returns = torch.as_tensor(returns, device=device)
        b_adv = torch.as_tensor(advantages, device=device)
        # Only mask we use during recompute is action_type
        b_masks: Dict[str, torch.Tensor] = {}
        if mask_buf and ('action_type' in mask_buf[0]):
            b_masks['action_type'] = torch.as_tensor(np.stack([m.get('action_type') for m in mask_buf], axis=0), device=device)

        # PPO epochs
        num_samples = b_obs.shape[0]
        batch_idx = np.arange(num_samples)
        for _ in range(update_epochs):
            np.random.shuffle(batch_idx)
            for start in range(0, num_samples, minibatch_size):
                end = start + minibatch_size
                mb_idx = batch_idx[start:end]

                mb_obs = b_obs[mb_idx]
                mb_actions = b_actions[mb_idx]
                mb_old_logp = b_logp[mb_idx]
                mb_adv = b_adv[mb_idx]
                mb_returns = b_returns[mb_idx]
                mb_masks: Dict[str, torch.Tensor] = {}
                if 'action_type' in b_masks:
                    mb_masks['action_type'] = b_masks['action_type'][mb_idx]

                logits, values_pred = model(mb_obs)
                new_logp, entropy = logprob_and_entropy(logits, mb_actions, masks=mb_masks)
                ratio = (new_logp - mb_old_logp).exp()

                # Policy loss with clipping
                unclipped = ratio * mb_adv
                clipped = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * mb_adv
                policy_loss = -torch.min(unclipped, clipped).mean()

                # Value loss
                v_pred = values_pred.squeeze(-1)
                value_loss = torch.nn.functional.mse_loss(v_pred, mb_returns)

                entropy_loss = -entropy.mean()
                loss = policy_loss + vf_coef * value_loss + ent_coef * (-entropy_loss)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            # Early stop by KL
            with torch.no_grad():
                logits_full, _ = model(b_obs)
                new_logp_full, _ = logprob_and_entropy(logits_full, b_actions, masks=b_masks)
                approx_kl = (b_logp - new_logp_full).mean().item()
                if approx_kl > target_kl:
                    break

        print(f"Update {update+1}/{total_updates}  return_mean={returns.mean():.2f}")

    env.close()


if __name__ == "__main__":
    train()
