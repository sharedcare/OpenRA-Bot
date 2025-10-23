from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


def _mlp(sizes: Sequence[int], activation=nn.ReLU, out_act: Optional[nn.Module] = None) -> nn.Sequential:
    layers: List[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(activation())
        elif out_act is not None:
            layers.append(out_act)
    return nn.Sequential(*layers)


class VisionEncoder(nn.Module):
    """
    CNN encoder for 10-channel 128x128 observations used by OpenRA visual env.
    Produces a compact feature vector representation.
    """

    def __init__(self, in_channels: int = 10, feature_dim: int = 256) -> None:
        super().__init__()
        # Simple but effective CNN
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),  # 64x64
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 32x32
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),  # 8x8
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, feature_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect N,H,W,C or N,C,H,W; normalize to N,C,H,W
        if x.dim() == 4 and x.shape[1] != 10:
            # assume NHWC uint8 -> NCHW float
            x = x.permute(0, 3, 1, 2)
        x = x.float() / 255.0
        y = self.conv(x)
        return self.head(y)


class VectorEncoder(nn.Module):
    """
    MLP encoder for vector observations used by OpenRA vector env.
    """

    def __init__(self, obs_dim: int, hidden_sizes: Sequence[int] = (512, 256), feature_dim: int = 256) -> None:
        super().__init__()
        self.net = _mlp([obs_dim, *hidden_sizes, feature_dim])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


class MixedEncoder(nn.Module):
    """
    Combine vector and vision encoders for hybrid observations.
    """

    def __init__(self, obs_dim: int, in_channels: int = 10, feature_dim: int = 256) -> None:
        super().__init__()
        self.vec = VectorEncoder(obs_dim=obs_dim, hidden_sizes=(512, 256), feature_dim=feature_dim)
        self.vision = VisionEncoder(in_channels=in_channels, feature_dim=feature_dim)
        self.proj = _mlp([feature_dim * 2, feature_dim])

    def forward(self, vec: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        v = self.vec(vec)
        i = self.vision(img)
        h = torch.cat([v, i], dim=-1)
        return self.proj(h)


class MultiDiscretePolicy(nn.Module):
    """
    Multi-head policy for OpenRA MultiDiscrete action space:
    [action_type, unit_idx, target_x, target_y, target_idx, unit_type_idx]

    - Each head outputs logits for a categorical distribution.
    - Action masks can be applied per-head to invalidate options.
    """

    def __init__(
        self,
        feature_dim: int,
        action_dims: Tuple[int, int, int, int, int, int],
        hidden_sizes: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        self.trunk = _mlp([feature_dim, *hidden_sizes, feature_dim])
        a0, a1, a2, a3, a4, a5 = action_dims
        self.head_action_type = nn.Linear(feature_dim, a0)
        self.head_unit_idx = nn.Linear(feature_dim, a1)
        self.head_target_x = nn.Linear(feature_dim, a2)
        self.head_target_y = nn.Linear(feature_dim, a3)
        self.head_target_idx = nn.Linear(feature_dim, a4)
        self.head_unit_type = nn.Linear(feature_dim, a5)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.trunk(features)
        return {
            "action_type": self.head_action_type(h),
            "unit_idx": self.head_unit_idx(h),
            "target_x": self.head_target_x(h),
            "target_y": self.head_target_y(h),
            "target_idx": self.head_target_idx(h),
            "unit_type": self.head_unit_type(h),
        }

    @torch.no_grad()
    def sample(
        self,
        logits: Dict[str, torch.Tensor],
        masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Sample from categorical distributions with optional masks.
        masks: dict of boolean tensors broadcastable to logits.
        Returns (action_tensor, logprobs_per_head)
        """
        actions: List[torch.Tensor] = []
        logps: Dict[str, torch.Tensor] = {}
        out_logits: Dict[str, torch.Tensor] = {}
        for key, lg in logits.items():
            if masks and key in masks and masks[key] is not None:
                # very negative for invalid actions
                mask = masks[key].to(lg.dtype)
                # Ensure mask is 1 for valid, 0 for invalid
                # Convert to additive mask in logit space
                add = torch.log(mask.clamp(min=1e-6))
                lg = lg + add
            dist = torch.distributions.Categorical(logits=lg)
            a = dist.sample()
            actions.append(a)
            logps[key] = dist.log_prob(a)
            out_logits[key] = lg
        # Order: action_type, unit_idx, target_x, target_y, target_idx, unit_type
        act = torch.stack(actions, dim=-1)
        return act, logps


class ActorCritic(nn.Module):
    """
    Combined policy + value network. Flexible encoders for vector/image inputs.
    """

    def __init__(
        self,
        obs_space: Dict[str, int],
        action_dims: Tuple[int, int, int, int, int, int],
        observation_type: str = "vector",
        feature_dim: int = 256,
    ) -> None:
        super().__init__()
        self.observation_type = observation_type
        if observation_type == "vector":
            obs_dim = int(obs_space.get("vector", 0))
            self.encoder = VectorEncoder(obs_dim=obs_dim, hidden_sizes=(512, 256), feature_dim=feature_dim)
        elif observation_type == "image":
            in_ch = int(obs_space.get("channels", 10))
            self.encoder = VisionEncoder(in_channels=in_ch, feature_dim=feature_dim)
        else:
            raise ValueError(f"Unsupported observation_type: {observation_type}")

        self.policy = MultiDiscretePolicy(feature_dim=feature_dim, action_dims=action_dims)
        self.value_head = _mlp([feature_dim, 256, 1])

    def forward(self, obs: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        features = self.encoder(obs)
        logits = self.policy(features)
        value = self.value_head(features).squeeze(-1)
        return logits, value
