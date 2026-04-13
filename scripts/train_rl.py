import os
import sys
import argparse
from typing import Tuple, Any, List, Dict, Optional

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
    from OpenRA.Bot.envs.wrappers import AugmentedStateWrapper, ShapedRewardWrapper
    from OpenRA.Bot.models import ActorCritic
    from OpenRA.Bot.agent import PPOAgent
except Exception:  # noqa: BLE001
    from envs.openra_env import OpenRAEnv, make_env  # type: ignore
    from envs.wrappers import AugmentedStateWrapper, ShapedRewardWrapper  # type: ignore
    from models import ActorCritic  # type: ignore
    from agent import PPOAgent  # type: ignore


def make_model(
    env,
    observation_type: str = "vector",
    recurrent_type: str = "lstm",
    augmented_config: Optional[Dict[str, int]] = None,
) -> Tuple[ActorCritic, Tuple[int, int, int, int, int, int]]:
    # Resolve action_types from possibly-wrapped env.
    inner = env
    while hasattr(inner, "env"):
        if hasattr(inner, "action_types"):
            break
        inner = inner.env
    a_dims = (
        len(inner.action_types),
        int(inner.action_space.nvec[1]),
        int(inner.action_space.nvec[2]),
        int(inner.action_space.nvec[3]),
        int(inner.action_space.nvec[4]),
        int(inner.action_space.nvec[5]),
    )
    if observation_type == "vector":
        obs_space = {"vector": int(env.observation_space.shape[0])}
    else:
        obs_space = {"channels": int(env.observation_space.shape[-1])}
    model = ActorCritic(
        obs_space=obs_space,
        action_dims=a_dims,
        observation_type=observation_type,
        recurrent_type=recurrent_type,
        augmented_config=augmented_config,
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
    bin_dir: str = "/Users/sharedcare/Projects/OpenRA/bin",
    mod_id: str = "ra",
    map_uid: str = "b53e25e007666442dbf62b87eec7bfbe8160ef3f",
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
    ticks_per_step: int = 10,
    max_episode_ticks: int = 3000,
    remote_host: str = "",
    remote_port: int = 0,
    remote_password: str = "",
    remote_slot: str = "",
    # Augmented state options
    augmented: bool = False,
    frame_stack_k: int = 8,
    verbose_reward: bool = False,
):
    env = make_env(
        bin_dir=bin_dir,
        mod_id=mod_id,
        map_uid=map_uid,
        ticks_per_step=ticks_per_step,
        observation_type=observation_type,
        enable_actions=["noop", "move", "attack", "produce", "build", "deploy"],
        max_episode_ticks=max_episode_ticks,
    )
    if remote_host and remote_port:
        env.configure_remote(
            host=remote_host,
            port=int(remote_port),
            password=remote_password,
            slot=(remote_slot or None),
            spectator=False,
        )

    # Optional augmented-state wrappers for delayed-reward credit assignment.
    augmented_config = None
    if augmented:
        env = ShapedRewardWrapper(env, verbose=verbose_reward)
        env = AugmentedStateWrapper(env, frame_stack_k=frame_stack_k)
        augmented_config = env.augmentation_config
        print(f"[augmented] obs_dim={env.aug_obs_dim} "
              f"(base={env.base_obs_dim}, k={frame_stack_k}, actions={env.num_action_types})")
        print(f"[augmented] reward=shaped (building+unit+production+deploy+idle)")
        print(f"[augmented] max_episode_ticks={max_episode_ticks}")

    model, _ = make_model(
        env,
        observation_type=observation_type,
        recurrent_type="lstm",
        augmented_config=augmented_config,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = PPOAgent(model=model, device=str(device))

    def _checkpoint(u: int, mdl: Any) -> None:
        try:
            base = log_dir
            if not os.path.exists(base):
                os.makedirs(base)
            path = os.path.join(base, f"model_{u:04d}.pth")
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
    parser = argparse.ArgumentParser(description="Train PPO on OpenRA locally or by joining a remote lobby.")
    parser.add_argument("--bin-dir", default="/Users/sharedcare/Projects/OpenRA/bin")
    parser.add_argument("--mod-id", default="ra")
    parser.add_argument("--map-uid", default="b53e25e007666442dbf62b87eec7bfbe8160ef3f")
    parser.add_argument("--num-steps", type=int, default=2048)
    parser.add_argument("--total-updates", type=int, default=100)
    parser.add_argument("--observation-type", choices=["vector", "image"], default="vector")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--log-dir", default="checkpoints")
    parser.add_argument("--ticks-per-step", type=int, default=10)
    parser.add_argument("--max-episode-ticks", type=int, default=3000,
                        help="Truncate episode after this many game ticks (forces reset). 3000 ticks ≈ 300 steps @ ticks_per_step=10.")
    parser.add_argument("--remote-host", default="")
    parser.add_argument("--remote-port", type=int, default=0)
    parser.add_argument("--remote-password", default="")
    parser.add_argument("--remote-slot", default="")
    # Augmented state options
    parser.add_argument(
        "--augmented", action="store_true",
        help="Wrap env with StateDiffRewardWrapper + AugmentedStateWrapper for delayed-reward credit assignment.",
    )
    parser.add_argument(
        "--frame-stack-k", type=int, default=8,
        help="Number of past frames to stack (only used with --augmented).",
    )
    parser.add_argument(
        "--verbose-reward", action="store_true",
        help="Print non-zero reward events to stderr for debugging.",
    )
    args = parser.parse_args()

    train(
        bin_dir=args.bin_dir,
        mod_id=args.mod_id,
        map_uid=args.map_uid,
        num_steps=args.num_steps,
        total_updates=args.total_updates,
        observation_type=args.observation_type,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        learning_rate=args.learning_rate,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        target_kl=args.target_kl,
        log_dir=args.log_dir,
        ticks_per_step=args.ticks_per_step,
        max_episode_ticks=args.max_episode_ticks,
        remote_host=args.remote_host,
        remote_port=args.remote_port,
        remote_password=args.remote_password,
        remote_slot=args.remote_slot,
        augmented=args.augmented,
        frame_stack_k=args.frame_stack_k,
        verbose_reward=args.verbose_reward,
    )
