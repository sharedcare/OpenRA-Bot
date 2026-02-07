import os
import sys
from typing import Tuple, Any, List, Dict

import numpy as np
import torch

try:
    import matplotlib  # type: ignore

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore

    _HAS_MPL = True
except Exception:  # noqa: BLE001
    _HAS_MPL = False

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from OpenRA.Bot.envs.openra_env import OpenRAEnv, make_env
    from OpenRA.Bot.models import ActorCritic
    from OpenRA.Bot.agent import PPOAgent
except Exception:  # noqa: BLE001
    from envs.openra_env import OpenRAEnv, make_env  # type: ignore
    from models import ActorCritic  # type: ignore
    from agent import PPOAgent  # type: ignore


def make_model(
    env: OpenRAEnv, observation_type: str = "vector", recurrent_type: str = "lstm"
) -> Tuple[ActorCritic, Tuple[int, int, int, int, int, int]]:
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
    model = ActorCritic(
        obs_space=obs_space, action_dims=a_dims, observation_type=observation_type, recurrent_type=recurrent_type
    )
    return model, a_dims


def _ensure_plot_dir(plot_dir: str) -> None:
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)


def _plot_hist(name: str, arr: np.ndarray, plot_dir: str, update_idx: int, bins: int = 50) -> None:
    if not _HAS_MPL or arr.size == 0:
        return
    _ensure_plot_dir(plot_dir)
    try:
        plt.figure(figsize=(6, 4))
        plt.hist(arr, bins=bins, color="#3b82f6", edgecolor="#1e3a8a")
        plt.title(f"{name} (update {update_idx})")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{update_idx:04d}_{name}.png"))
        plt.close()
    except Exception:
        pass


def _plot_bar(name: str, keys: List[int], counts: List[int], plot_dir: str, update_idx: int) -> None:
    if not _HAS_MPL or len(keys) == 0:
        return
    _ensure_plot_dir(plot_dir)
    try:
        plt.figure(figsize=(6, 4))
        plt.bar([str(k) for k in keys], counts, color="#10b981")
        plt.title(f"{name} (update {update_idx})")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{update_idx:04d}_{name}.png"))
        plt.close()
    except Exception:
        pass


def _plot_series(name: str, arr: np.ndarray, plot_dir: str, update_idx: int) -> None:
    if not _HAS_MPL or arr.size == 0:
        return
    _ensure_plot_dir(plot_dir)
    try:
        plt.figure(figsize=(6, 4))
        plt.plot(arr, color="#ef4444")
        plt.title(f"{name} (update {update_idx})")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{update_idx:04d}_{name}.png"))
        plt.close()
    except Exception:
        pass


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
    learning_rate: float = 1e-2,
    update_epochs: int = 4,
    minibatch_size: int = 256,
    target_kl: float = 0.03,
    log_dir: str = "checkpoints",
):
    env = make_env(
        bin_dir="F:/Projects/OpenRA/bin",
        mod_id="ra",
        map_uid="b53e25e007666442dbf62b87eec7bfbe8160ef3f",
        ticks_per_step=10,
        observation_type=observation_type,
        enable_actions=["noop", "move", "attack", "produce", "build", "deploy"],
    )
    model, _ = make_model(env, observation_type=observation_type, recurrent_type="lstm")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = PPOAgent(model=model, device=str(device))

    def _checkpoint(u: int, mdl: Any) -> None:
        try:
            base = log_dir
            if not os.path.exists(base):
                os.makedirs(base)
            path = os.path.join(base, "model_{}.pth".format(u))
            torch.save(mdl.state_dict(), path)
            print(f"Model saved to {path}")
        except Exception as e:
            print(f"Failed to save model: {e}")

    agent.train(
        env=env,
        total_updates=total_updates,
        num_steps=num_steps,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_coef=clip_coef,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        learning_rate=learning_rate,
        update_epochs=update_epochs,
        minibatch_size=minibatch_size,
        target_kl=target_kl,
        checkpoint_fn=_checkpoint,
        log_path=os.path.join(log_dir, "training.csv"),
    )

    env.close()


if __name__ == "__main__":
    train(log_dir=r"F:\Projects\OpenRA\OpenRA.Bot\checkpoints")
