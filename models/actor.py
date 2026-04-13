from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

import numpy as np


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

    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: Sequence[int] = (512, 256),
        feature_dim: int = 256,
    ) -> None:
        super().__init__()
        self.net = _mlp([obs_dim, *hidden_sizes, feature_dim])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


class AugmentedVectorEncoder(nn.Module):
    """Encoder for augmented observations produced by AugmentedStateWrapper.

    The flat input is split back into structured components and processed by
    specialised sub-networks:

        frames (k * base_dim) -> shared frame encoder -> mean pool  -> 256
        delta  (base_dim)     -> same shared encoder                -> 256
        action_hist + time    -> small context MLP                  -> 64
                                                     concat (576) -> fusion -> feature_dim

    This is ~7x more parameter-efficient than a single MLP over the full
    augmented vector and preserves the per-frame spatial structure that a
    flat MLP would lose.
    """

    def __init__(
        self,
        base_obs_dim: int,
        frame_stack_k: int,
        num_action_types: int,
        feature_dim: int = 256,
    ) -> None:
        super().__init__()
        self.base_obs_dim = base_obs_dim
        self.k = frame_stack_k
        self.num_action_types = num_action_types

        # Shared encoder applied to each frame AND to the state delta.
        self.frame_enc = _mlp([base_obs_dim, 512, 256])

        # Small encoder for the discrete action context.
        action_ctx_dim = frame_stack_k * num_action_types + num_action_types
        self.action_enc = _mlp([action_ctx_dim, 64, 64])

        # Fusion: frame_pool(256) + delta(256) + action_ctx(64) -> feature_dim
        self.fusion = _mlp([256 + 256 + 64, 256, feature_dim])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, aug_dim) where N = B or B*L (ActorCritic flattens before calling).
        f_end = self.k * self.base_obs_dim
        a_end = f_end + self.k * self.num_action_types
        d_end = a_end + self.base_obs_dim
        # d_end+num_action_types == aug_dim (time features at the tail)

        frames_flat = x[:, :f_end]                     # (N, k*base)
        action_hist = x[:, f_end:a_end]                 # (N, k*nact)
        delta = x[:, a_end:d_end]                       # (N, base)
        time_feat = x[:, d_end:]                        # (N, nact)

        N = x.shape[0]

        # --- per-frame encoding (weight-shared) ---
        frames = frames_flat.reshape(N * self.k, self.base_obs_dim)
        frame_feats = self.frame_enc(frames)                # (N*k, 256)
        frame_feats = frame_feats.reshape(N, self.k, -1)    # (N, k, 256)
        frame_pool = frame_feats.mean(dim=1)                 # (N, 256)

        # --- delta through same shared encoder ---
        delta_feat = self.frame_enc(delta)                   # (N, 256)

        # --- action / time context ---
        ctx = torch.cat([action_hist, time_feat], dim=-1)    # (N, k*nact + nact)
        ctx_feat = self.action_enc(ctx)                      # (N, 64)

        # --- fusion ---
        fused = torch.cat([frame_pool, delta_feat, ctx_feat], dim=-1)  # (N, 576)
        return self.fusion(fused)                                       # (N, feature_dim)


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
    HEADS = ["action_type", "unit_idx", "target_x", "target_y", "target_idx", "unit_type"]
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

    @staticmethod
    def masked_logits(
        logits: Dict[str, torch.Tensor], masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        if not masks:
            return logits
        out: Dict[str, torch.Tensor] = {}
        for k, lg in logits.items():
            m = masks.get(k, None)
            if m is None:
                out[k] = lg
                continue
            # m: 1(valid) / 0(invalid), broadcastable to lg
            m = m.to(dtype=lg.dtype, device=lg.device)
            out[k] = lg + torch.log(m.clamp_min(1e-6))
        return out

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
        lg = self.masked_logits(logits, masks)
        actions: List[torch.Tensor] = []
        logps: Dict[str, torch.Tensor] = {}
        for key in self.HEADS:
            dist = torch.distributions.Categorical(logits=lg[key])
            a = dist.sample()
            actions.append(a)
            logps[key] = dist.log_prob(a)
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
        hidden_size: int = 256,
        recurrent_type: Optional[str] = None,  # [None, "lstm", "gru"]
        recurrent_hidden_size: int = 256,
        augmented_config: Optional[Dict[str, int]] = None,
    ) -> None:
        super().__init__()
        self.observation_type = observation_type
        self.recurrent_type = recurrent_type
        self.recurrent_hidden_size = recurrent_hidden_size
        self.num_recurrent_layers = 2
        # Backward-compatible name used by PPOAgent training code.
        self.num_lstm_layers = self.num_recurrent_layers
        if observation_type == "vector":
            if augmented_config is not None:
                self.encoder = AugmentedVectorEncoder(
                    base_obs_dim=int(augmented_config["base_obs_dim"]),
                    frame_stack_k=int(augmented_config["frame_stack_k"]),
                    num_action_types=int(augmented_config["num_action_types"]),
                    feature_dim=feature_dim,
                )
            else:
                obs_dim = int(obs_space.get("vector", 0))
                self.encoder = VectorEncoder(obs_dim=obs_dim, hidden_sizes=(512, 256), feature_dim=feature_dim)
        elif observation_type == "image":
            in_ch = int(obs_space.get("channels", 10))
            self.encoder = VisionEncoder(in_channels=in_ch, feature_dim=feature_dim)
        else:
            raise ValueError(f"Unsupported observation_type: {observation_type}")

        if self.recurrent_type == "lstm":
            self.core = nn.LSTM(
                input_size=feature_dim,
                hidden_size=recurrent_hidden_size,
                num_layers=self.num_recurrent_layers,
                batch_first=True,
            )
        elif self.recurrent_type == "gru":
            self.core = nn.GRU(
                input_size=feature_dim,
                hidden_size=recurrent_hidden_size,
                num_layers=self.num_recurrent_layers,
                batch_first=True,
            )
        # Init recurrent layer
        if self.recurrent_type:
            for name, param in self.core.named_parameters():
                if "bias" in name:
                    nn.init.constant_(param, 0)
                elif "weight" in name:
                    nn.init.orthogonal_(param, np.sqrt(2))
        else:
            self.core = nn.Linear(feature_dim, recurrent_hidden_size)

        self.hidden_layer = nn.Linear(recurrent_hidden_size, hidden_size)

        self.policy_head = MultiDiscretePolicy(feature_dim=recurrent_hidden_size, action_dims=action_dims)
        self.value_head = _mlp([recurrent_hidden_size, 256, 1])

    def init_hidden(self, batch_size: int, device: str) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.recurrent_type is None:
            return None, None
        h = torch.zeros(self.num_recurrent_layers, batch_size, self.recurrent_hidden_size, device=device)
        c = None
        if self.recurrent_type == "lstm":
            c = torch.zeros(self.num_recurrent_layers, batch_size, self.recurrent_hidden_size, device=device)
        return (h, c)

    def forward(
        self, obs: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, seq_len: int = 1
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # obs shape: (batch_size, seq_len, *obs_dim) or (batch_size, *obs_dim) if seq_len=1
        # Ensure obs is at least 2D
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        # Flatten batch and sequence dimensions for encoder
        if seq_len > 1:
            batch_size = obs.shape[0]
            # obs: (B, L, ...) -> (B*L, ...)
            obs_flat = obs.reshape(-1, *obs.shape[2:])
            features_flat = self.encoder(obs_flat)
            # features: (B*L, D) -> (B, L, D)
            features = features_flat.view(batch_size, seq_len, -1)
        else:
            # obs: (B, ...)
            batch_size = obs.shape[0]
            features = self.encoder(obs)
            # features: (B, D) -> (B, 1, D) for batch_first LSTM
            if features.dim() == 2:
                features = features.unsqueeze(1)

        # Recurrent update
        if self.recurrent_type is not None:
            # features: (B, L, D) where L=1 for single step
            # hidden: (num_layers, B, H) for batch_first=True LSTM
            # Ensure hidden state shape matches batch size
            if hidden is not None and hidden[0] is not None:
                # Check if hidden state batch size matches
                num_layers = hidden[0].shape[0]
                if hidden[0].shape[1] != batch_size:
                    # Reinitialize hidden state with correct batch size
                    device = hidden[0].device
                    h = torch.zeros(num_layers, batch_size, self.recurrent_hidden_size, device=device)
                    c = None
                    if self.recurrent_type == "lstm" and hidden[1] is not None:
                        c = torch.zeros(num_layers, batch_size, self.recurrent_hidden_size, device=device)
                    hidden = (h, c)

            output, hidden = self.core(features, hidden)
            # output: (B, L, H)
            # Flatten for heads
            output_flat = output.reshape(-1, self.recurrent_hidden_size)
        else:
            # Feedforward projection
            # features: (B, L, D) -> flatten to (B*L, D) for Linear
            features_flat = features.reshape(-1, features.shape[-1])
            output_flat = self.core(features_flat)  # (B*L, H)
            hidden = None

        logits = self.policy_head(output_flat)
        value = self.value_head(output_flat).squeeze(-1)

        return logits, value, hidden
