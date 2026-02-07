from typing import Any, Dict, Optional, Tuple, Union, Generator
import torch
import numpy as np

EPSILON = 1.0e-5


class Buffer:

    def __init__(
        self,
        num_envs: int,
        seq_len: int,
        buffer_size: int,  # Total number of steps to store (num_sequences * seq_len)
        observation_space: Any,
        action_space: Any,
        device: torch.device,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        num_lstm_layers: int = 2,
        hidden_size: int = 256,
        has_action_masks: bool = True,
    ):
        """
        Initialize the ppo buffer.

        Args:
            num_envs: Number of parallel environments
            seq_len: Length of sequences for LSTM training
            buffer_size: Total number of steps to store (num_sequences * seq_len)
            observation_space: Observation space to determine tensor shapes
            action_space: Action space to determine tensor shapes
            device: PyTorch device (cpu/cuda)
            gamma: Discount factor for rewards
            gae_lambda: Lambda parameter for GAE
            num_lstm_layers: Number of LSTM layers for hidden state storage
            hidden_size: Hidden size of LSTM
            has_action_masks: Whether to allocate space for action masks
        """
        self.device = device
        self.num_envs = num_envs
        self.buffer_size = buffer_size
        self.num_sequences = buffer_size // seq_len
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.num_lstm_layers = num_lstm_layers
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.has_action_masks = has_action_masks

        # Determine observation shape
        if hasattr(observation_space, "shape"):
            obs_shape = observation_space.shape
        elif isinstance(observation_space, dict) and "vector" in observation_space:
            obs_shape = (observation_space["vector"],)
        else:
            raise ValueError("Unsupported observation space")

        # Determine action dimension
        if hasattr(action_space, "nvec"):  # MultiDiscrete
            action_dim = len(action_space.nvec)
        else:
            raise ValueError("Unsupported action space")

        # Pre-allocate tensors on device
        self.obs = torch.zeros(
            (buffer_size, num_envs, *obs_shape), dtype=torch.float32, device=device
        )
        self.actions = torch.zeros(
            (buffer_size, num_envs, action_dim), dtype=torch.int64, device=device
        )
        self.rewards = torch.zeros(
            (buffer_size, num_envs), dtype=torch.float32, device=device
        )
        self.dones = torch.zeros(
            (buffer_size, num_envs), dtype=torch.float32, device=device
        )
        self.values = torch.zeros(
            (buffer_size, num_envs), dtype=torch.float32, device=device
        )
        self.logprobs = torch.zeros(
            (buffer_size, num_envs), dtype=torch.float32, device=device
        )

        # Hidden states at sequence boundaries
        self.hidden_states_h = torch.zeros(
            (self.num_sequences, num_lstm_layers, num_envs, hidden_size),
            dtype=torch.float32,
            device=device,
        )
        self.hidden_states_c = torch.zeros(
            (self.num_sequences, num_lstm_layers, num_envs, hidden_size),
            dtype=torch.float32,
            device=device,
        )

        # Action masks
        if has_action_masks:
            # Assuming action_type mask is binary with shape (num_actions,)
            # We'll determine num_actions from the first experience added
            self.masks_initialized = False
            self.masks_action_type = None  # Will be initialized on first add
        else:
            self.masks_initialized = True
            self.masks_action_type = None

        # Tracking indices
        self.step_idx = 0  # Current step index in buffer
        self.seq_idx = 0  # Current sequence index
        self.full = False  # Whether buffer is full

        # Episode tracking
        self.episode_rewards = [[] for _ in range(num_envs)]
        self.episode_lengths = [0] * num_envs
        self.total_reward = 0.0
        self.total_episodes = 0

        # After advantage calculation
        self.advantages = None
        self.returns = None

    @property
    def is_full(self) -> bool:
        """Check if buffer is full."""
        return self.full

    def __len__(self) -> int:
        """Get current buffer size in steps."""
        return self.step_idx

    def reset(self) -> None:
        """Reset all buffers for a new rollout collection."""
        self.step_idx = 0
        self.seq_idx = 0
        self.full = False

        # Reset episode tracking
        self.episode_rewards = [[] for _ in range(self.num_envs)]
        self.episode_lengths = [0] * self.num_envs
        self.total_reward = 0.0
        self.total_episodes = 0

        # Reset advantages
        self.advantages = None
        self.returns = None

    def add(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        actions: Union[np.ndarray, torch.Tensor],
        rewards: Union[np.ndarray, torch.Tensor],
        dones: Union[np.ndarray, torch.Tensor],
        values: Union[np.ndarray, torch.Tensor],
        logprobs: Union[np.ndarray, torch.Tensor],
        masks: Optional[Dict[str, Union[np.ndarray, torch.Tensor]]] = None,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> None:
        """
        Add a new transition to the buffer using tensor operations.
        All inputs should have shape (num_envs, ...) or be convertible to it.
        """
        if self.full:
            raise RuntimeError("Buffer is full. Call reset() before adding more data.")

        # Convert inputs to tensors if needed
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).to(self.device)
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).to(self.device)
        if isinstance(rewards, np.ndarray):
            rewards = torch.from_numpy(rewards).to(self.device)
        if isinstance(dones, np.ndarray):
            dones = torch.from_numpy(dones).to(self.device)
        if isinstance(values, np.ndarray):
            values = torch.from_numpy(values).to(self.device)
        if isinstance(logprobs, np.ndarray):
            logprobs = torch.from_numpy(logprobs).to(self.device)

        # Add action masks if available
        if masks and self.has_action_masks:
            if "action_type" in masks:
                action_mask = masks["action_type"]
                if isinstance(action_mask, np.ndarray):
                    action_mask = torch.from_numpy(action_mask).to(self.device)

                # Initialize masks tensor if first time
                if not self.masks_initialized:
                    mask_shape = (
                        action_mask.shape[1:] if action_mask.dim() > 1 else (1,)
                    )
                    self.masks_action_type = torch.zeros(
                        (self.buffer_size, self.num_envs, *mask_shape),
                        dtype=torch.float32,
                        device=self.device,
                    )
                    self.masks_initialized = True

                # Store mask
                self.masks_action_type[self.step_idx] = action_mask

        # Store experience
        self.obs[self.step_idx] = obs
        self.actions[self.step_idx] = actions
        self.rewards[self.step_idx] = rewards
        self.dones[self.step_idx] = dones
        self.values[self.step_idx] = values
        self.logprobs[self.step_idx] = logprobs

        # Store hidden state at sequence boundaries
        if self.step_idx % self.seq_len == 0 and hidden_state is not None:
            # hidden_state shape: (num_layers, num_envs, hidden_size)
            if hidden_state[0] is not None:
                self.hidden_states_h[self.seq_idx] = hidden_state[0]
            if hidden_state[1] is not None:
                self.hidden_states_c[self.seq_idx] = hidden_state[1]
            self.seq_idx += 1

        # Track episode statistics
        dones_cpu = dones.cpu().numpy() if isinstance(dones, torch.Tensor) else dones
        rewards_cpu = (
            rewards.cpu().numpy() if isinstance(rewards, torch.Tensor) else rewards
        )

        for i in range(self.num_envs):
            self.episode_rewards[i].append(rewards_cpu[i])
            self.episode_lengths[i] += 1

            if dones_cpu[i]:
                self.total_reward += sum(self.episode_rewards[i])
                self.total_episodes += 1
                self.episode_rewards[i] = []
                self.episode_lengths[i] = 0

        # Update step index
        self.step_idx += 1
        if self.step_idx >= self.buffer_size:
            self.full = True

    def compute_advantages(
        self, last_values: torch.Tensor, gamma: float, lam: float
    ) -> None:
        """
        Compute GAE advantages and returns for the entire buffer.

        Args:
            last_values: Bootstrap values for the last step (num_envs,)
        """
        if not self.full:
            raise RuntimeError(
                "Buffer not full. Compute advantages only after collection is complete."
            )

        T = self.step_idx
        rewards = self.rewards[:T]          # (T, N)
        dones = self.dones[:T]              # (T, N) float 0/1
        values = self.values[:T]            # (T, N)

        last_values = last_values.view(1, -1)              # (1, N)
        values_boot = torch.cat([values, last_values], 0)  # (T+1, N)

        adv = torch.zeros_like(rewards)
        gae = torch.zeros(self.num_envs, device=self.device)

        with torch.no_grad():
            for t in reversed(range(T)):
                nonterminal = 1.0 - dones[t]  # dones 已是 float
                delta = rewards[t] + gamma * values_boot[t + 1] * nonterminal - values_boot[t]
                gae = gae * gamma * lam + delta
                adv[t] = gae

            ret = adv + values_boot[:-1]
            adv = (adv - adv.mean()) / (adv.std() + EPSILON)

        self.advantages = adv
        self.returns = ret

    def recurrent_mini_batch_generator(self, minibatch_size: int, num_epochs: int) -> Generator[Dict[str, torch.Tensor], None, None]:
        """
        Generate minibatches of sequences for recurrent training using tensor slicing.

        Args:
            minibatch_size: Number of sequences per minibatch
            num_epochs: Number of epochs to iterate over the buffer

        Yields:
            Dictionary containing batch tensors with keys:
                'obs', 'actions', 'old_logprobs', 'advantages', 'returns',
                'hidden_states', 'attention_mask'
        """
        if self.advantages is None or self.returns is None:
            raise RuntimeError("Must call compute_returns_and_advantages before generating minibatches")

        if not self.full:
            raise RuntimeError("Buffer not full. Generate batches only after collection is complete.")

        # Create sequence indices (each sequence is seq_len steps long)
        sequence_indices = torch.arange(self.num_sequences, device=self.device)
        max_seq_len = self.seq_len

        for epoch in range(num_epochs):
            # Shuffle sequence indices
            sequence_indices = sequence_indices[torch.randperm(len(sequence_indices))]

            for start in range(0, len(sequence_indices), minibatch_size):
                end = min(start + minibatch_size, len(sequence_indices))
                seq_idxs = sequence_indices[start:end]
                batch_size = len(seq_idxs)

                # Prepare batch tensors by slicing the pre-allocated buffers
                # Shape: (batch_size, seq_len, num_envs, ...)
                obs_batch = torch.zeros(
                    (batch_size, max_seq_len, self.num_envs, *self.obs.shape[2:]),
                    device=self.device,
                )
                actions_batch = torch.zeros(
                    (batch_size, max_seq_len, self.num_envs, self.actions.shape[2]),
                    device=self.device,
                    dtype=torch.int64,
                )
                old_logprobs_batch = torch.zeros(
                    (batch_size, max_seq_len, self.num_envs), device=self.device
                )
                advantages_batch = torch.zeros(
                    (batch_size, max_seq_len, self.num_envs), device=self.device
                )
                returns_batch = torch.zeros(
                    (batch_size, max_seq_len, self.num_envs), device=self.device
                )

                # Fill the batches by slicing from the main buffers
                for i, seq_idx in enumerate(seq_idxs):
                    start_step = seq_idx * self.seq_len
                    end_step = start_step + self.seq_len

                    # Ensure we don't go out of bounds
                    end_step = min(end_step, self.step_idx)
                    actual_seq_len = end_step - start_step

                    # Slice data for this sequence
                    obs_batch[i, :actual_seq_len] = self.obs[start_step:end_step]
                    actions_batch[i, :actual_seq_len] = self.actions[
                        start_step:end_step
                    ]
                    old_logprobs_batch[i, :actual_seq_len] = self.logprobs[
                        start_step:end_step
                    ]
                    advantages_batch[i, :actual_seq_len] = self.advantages[
                        start_step:end_step
                    ]
                    returns_batch[i, :actual_seq_len] = self.returns[
                        start_step:end_step
                    ]

                # Prepare attention mask (1 for valid steps, 0 for padding)
                attention_mask = torch.zeros(
                    (batch_size, max_seq_len, self.num_envs),
                    device=self.device,
                    dtype=torch.bool,
                )

                # Create attention mask based on actual sequence lengths
                for i, seq_idx in enumerate(seq_idxs):
                    start_step = seq_idx * self.seq_len
                    end_step = min(start_step + self.seq_len, self.step_idx)
                    actual_seq_len = end_step - start_step
                    attention_mask[i, :actual_seq_len] = True

                # Prepare hidden states for this batch
                hidden_states_h = self.hidden_states_h[
                    seq_idxs
                ]  # (batch_size, num_layers, num_envs, hidden_size)
                hidden_states_c = self.hidden_states_c[
                    seq_idxs
                ]  # (batch_size, num_layers, num_envs, hidden_size)

                # Prepare masks if available
                masks_batch = {}
                if self.masks_initialized and self.masks_action_type is not None:
                    masks_action_type = torch.zeros(
                        (
                            batch_size,
                            max_seq_len,
                            self.num_envs,
                            *self.masks_action_type.shape[2:],
                        ),
                        device=self.device,
                    )
                    for i, seq_idx in enumerate(seq_idxs):
                        start_step = seq_idx * self.seq_len
                        end_step = min(start_step + self.seq_len, self.step_idx)
                        actual_seq_len = end_step - start_step
                        masks_action_type[i, :actual_seq_len] = self.masks_action_type[
                            start_step:end_step
                        ]
                    masks_batch["action_type"] = masks_action_type

                # Yield batch
                yield {
                    "obs": obs_batch,
                    "actions": actions_batch,
                    "old_logprobs": old_logprobs_batch,
                    "advantages": advantages_batch,
                    "returns": returns_batch,
                    "masks": masks_batch,
                    "hidden_states": (hidden_states_h, hidden_states_c),
                    "attention_mask": attention_mask,
                }

    def to_flattened_tensors(self) -> Dict[str, torch.Tensor]:
        """
        Convert buffer data to flattened tensors for non-recurrent training or analysis.
        Returns data shaped as (total_steps, ...) where total_steps = buffer_size * num_envs
        """
        total_steps = self.step_idx * self.num_envs

        # Flatten environment dimension into batch dimension
        obs_flat = self.obs[: self.step_idx].view(total_steps, *self.obs.shape[2:])
        actions_flat = self.actions[: self.step_idx].view(
            total_steps, self.actions.shape[-1]
        )
        logprobs_flat = self.logprobs[: self.step_idx].view(total_steps)
        advantages_flat = self.advantages[: self.step_idx].view(total_steps)
        returns_flat = self.returns[: self.step_idx].view(total_steps)

        result = {
            "obs": obs_flat,
            "actions": actions_flat,
            "old_logprobs": logprobs_flat,
            "advantages": advantages_flat,
            "returns": returns_flat,
        }

        # Add masks if available
        if self.masks_initialized and self.masks_action_type is not None:
            masks_flat = self.masks_action_type[: self.step_idx].view(
                total_steps, *self.masks_action_type.shape[2:]
            )
            result["masks"] = masks_flat

        return result

    def get_action_distribution(self) -> Dict[int, int]:
        """Get distribution of action types taken."""
        if self.step_idx == 0:
            return {}

        try:
            # Extract action types (first dimension of actions)
            action_types = self.actions[: self.step_idx, :, 0].cpu().numpy().flatten()
            unique, counts = np.unique(action_types, return_counts=True)
            return {int(u): int(c) for u, c in zip(unique, counts)}
        except Exception:
            return {}

    def get_mask_statistics(self) -> Optional[float]:
        """Get statistics about action masks."""
        if not self.masks_initialized or self.masks_action_type is None:
            return None

        try:
            valid_masks = self.masks_action_type[: self.step_idx]
            if valid_masks.numel() == 0:
                return None
            return float(valid_masks.mean().item())
        except Exception:
            return None


def split_and_pad_trajectories(
    tensors_list, dones, hiddens_list=None, init_hidden_only=True
):
    """Splits trajectories at done indices. Then concatenates them and padds with zeros up to the length og the longest trajectory.
    Returns masks corresponding to valid parts of the trajectories
    Example:
        Input: [ [a1, a2, a3, a4 | a5, a6],
                 [b1, b2 | b3, b4, b5 | b6]
                ]

        Output:[ [a1, a2, a3, a4], | [  [True, True, True, True],
                 [a5, a6, 0, 0],   |    [True, True, False, False],
                 [b1, b2, 0, 0],   |    [True, True, False, False],
                 [b3, b4, b5, 0],  |    [True, True, True, False],
                 [b6, 0, 0, 0]     |    [True, False, False, False],
                ]                  | ]

    Assumes that the inputy has the following dimension order: [time, number of envs, aditional dimensions]
    """
    device = dones.device
    dones = dones.clone()
    dones[-1] = 1
    if not isinstance(tensors_list, list):
        tensors_list = [tensors_list]
    len_tensors_list = len(tensors_list)
    if not isinstance(hiddens_list, list):
        hiddens_list = [hiddens_list]
    hiddens_list = [h for h in hiddens_list if h is not None]
    if bool(hiddens_list) is True:
        tensors_list += [
            hidden_states.transpose(1, 2).reshape(dones.shape[0], dones.shape[1], -1)
            for hidden_states in hiddens_list
        ]

    # Permute the buffers to have order (num_envs, num_transitions_per_env, ...), for correct reshaping
    flat_dones = dones.transpose(1, 0).reshape(-1, 1)

    # Get length of trajectory by counting the number of successive not done elements
    done_indices = torch.cat(
        (flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero()[:, 0])
    )
    trajectory_lengths = done_indices[1:] - done_indices[:-1]
    trajectory_lengths_list = trajectory_lengths.tolist()
    # Extract the individual trajectories
    trajectories = [
        (
            torch.split(tensor.transpose(1, 0).flatten(0, 1), trajectory_lengths_list)
            if tensor is not None
            else None
        )
        for tensor in tensors_list
    ]
    padded_trajectories = [
        torch.nn.utils.rnn.pad_sequence(trajectory) if trajectory is not None else None
        for trajectory in trajectories
    ]

    trajectory_masks = trajectory_lengths > torch.arange(
        0, padded_trajectories[0].shape[0], device=device
    ).unsqueeze(1)
    if bool(hiddens_list) is True:
        padded_hiddens = []
        for hidden_states in hiddens_list:
            traj_hidden = padded_trajectories.pop(len_tensors_list)
            traj_hidden = (
                traj_hidden[0, :, :].view(
                    hidden_states.shape[1], -1, hidden_states.shape[3]
                )
                if init_hidden_only
                else traj_hidden[0, :, :].view(
                    hidden_states.shape[1], -1, hidden_states.shape[3]
                )
            )
            padded_hiddens.append(traj_hidden)
        padded_trajectories.append(padded_hiddens)
    padded_trajectories.append(trajectory_masks)
    return padded_trajectories
