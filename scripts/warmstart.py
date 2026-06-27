"""
Behavior-cloning warm-start: pre-train the action-type head by imitating
the RuleBasedAgent, so the randomly-initialised PPO policy does not collapse
to noop before it discovers productive action sequences.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# Mapping from RuleBasedAgent order strings to env action_type names.
_ORDER_TO_ACTION_TYPE: Dict[str, str] = {
    "move": "move",
    "attack": "attack",
    "startproduction": "produce",
    "placebuilding": "build",
    "deploytransform": "deploy",
    "stop": "noop",
}


def _action_dict_to_type_idx(
    action_dicts: List[Dict[str, Any]],
    action_types: List[str],
) -> int:
    """Convert a list of dict actions to a single action_type index.

    Returns the index of the first non-noop action in ``action_types``,
    or 0 (noop) if the list is empty / only contains unrecognised orders.

    Macro mode: action_types are ['noop', 'produce:<t>', ...]. A
    StartProduction maps to the matching 'produce:<target>' entry; deploy and
    placement are auto-handled by the env in macro mode, so they map to noop.
    """
    macro_mode = any(str(a).startswith("produce:") for a in action_types)
    for a in (action_dicts or []):
        order = str(a.get("order", "")).lower()
        if macro_mode:
            if order == "startproduction":
                tgt = str(a.get("target_string", "")).lower().strip()
                cand = f"produce:{tgt}"
                if cand in action_types:
                    return action_types.index(cand)
            # deploy / placebuilding / move / attack are not macro decisions
            continue
        name = _ORDER_TO_ACTION_TYPE.get(order)
        if name is not None and name in action_types:
            return action_types.index(name)
    return 0  # noop


def collect_demonstrations(
    env,
    num_episodes: int = 20,
    max_steps_per_episode: int = 300,
    verbose: bool = False,
) -> Tuple[List[np.ndarray], List[int]]:
    """Run the RuleBasedAgent and collect (obs, action_type_label) pairs.

    Returns
    -------
    observations : list of np.ndarray
        Raw vector observations (before any augmentation wrappers).
    labels : list of int
        Action-type index that the RuleBasedAgent would have taken.
    """
    try:
        from OpenRA.Bot.agent.agent import RuleBasedAgent  # type: ignore
    except ImportError:
        from agent.agent import RuleBasedAgent  # type: ignore

    # Walk to the innermost OpenRAEnv so we can access raw state and
    # call send_actions with dict format.
    inner = env
    while hasattr(inner, "env"):
        inner = inner.env

    rule_agent = RuleBasedAgent()
    observations: List[np.ndarray] = []
    labels: List[int] = []

    for ep in range(num_episodes):
        obs, info = inner.reset()
        ep_steps = 0
        while ep_steps < max_steps_per_episode:
            # Re-read fresh state from the engine every step — do NOT cache
            # _last_raw_state because we bypass OpenRAEnv.step().
            try:
                from OpenRA.Bot.utils.obs import build_observation
            except ImportError:
                from utils.obs import build_observation  # type: ignore
            raw_state = build_observation(inner._openra)

            # Build the observation in whatever format the model sees
            if inner.observation_type == "entity":
                vec_obs = inner._state_to_entity(raw_state)
            elif inner.observation_type == "vector":
                vec_obs = inner._state_to_vector(raw_state)
            else:
                vec_obs = raw_state

            # Get rule-based action (dict format)
            dict_actions = rule_agent.act(raw_state)

            # Convert to action_type index
            label = _action_dict_to_type_idx(dict_actions, inner.action_types)
            observations.append(vec_obs)
            labels.append(label)

            # Execute the action
            if dict_actions:
                inner.send_actions(dict_actions)

            # Step the environment
            api = inner._openra["PythonAPI"]
            for _ in range(inner.ticks_per_step):
                api.Step()

            ep_steps += 1
            if inner.max_episode_ticks and int(raw_state.get("world_tick", 0)) >= inner.max_episode_ticks:
                break

        if verbose:
            atype_counts: Dict[int, int] = {}
            for l in labels[-ep_steps:]:
                atype_counts[l] = atype_counts.get(l, 0) + 1
            atype_names = {i: n for i, n in enumerate(inner.action_types)}
            dist = {atype_names.get(k, str(k)): v for k, v in atype_counts.items()}
            print(f"[warmstart] episode {ep + 1}/{num_episodes}  steps={ep_steps}  actions={dist}")

    return observations, labels


def pretrain_policy(
    model: nn.Module,
    observations: List[np.ndarray],
    labels: List[int],
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cpu",
) -> None:
    """Pre-train the action-type head (and encoder) via cross-entropy.

    Only the action_type head and the shared encoder are updated; the value
    head and LSTM core are kept frozen so they do not interfere.
    """
    device_t = torch.device(device)
    model.to(device_t)
    model.train()

    # Freeze everything except the stateless policy path used by PPO:
    # encoder -> core projection -> policy trunk -> action_type head.
    # Training only encoder -> head_action_type overstates BC accuracy because
    # the real forward pass also includes core and policy_head.trunk.
    _trainable_prefixes = (
        "encoder.",
        "core.",
        "policy_head.trunk.",
        "policy_head.head_action_type.",
    )
    for name, param in model.named_parameters():
        ok = any(name.startswith(p) for p in _trainable_prefixes)
        param.requires_grad = ok

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    # Compute class weights to handle label imbalance (noop dominates).
    # Size the weight vector to the FULL action_type head (not just the labels
    # seen), since CrossEntropyLoss requires one weight per class. Unseen
    # classes get weight 1.0 (they never appear in the loss anyway).
    num_classes = int(model.policy_head.head_action_type.out_features)
    unique, counts = np.unique(labels, return_counts=True)
    n_total = len(labels)
    weight_dict = {int(u): n_total / (len(unique) * c) for u, c in zip(unique, counts)}
    class_weights = torch.tensor(
        [weight_dict.get(i, 1.0) for i in range(num_classes)],
        dtype=torch.float32, device=device_t,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    if isinstance(observations[0], dict):
        # Entity dict observations
        obs_tensor = {
            k: torch.from_numpy(np.stack([o[k] for o in observations], axis=0)).float().to(device_t)
            for k in observations[0].keys()
        }
    else:
        obs_tensor = torch.from_numpy(np.stack(observations, axis=0)).float().to(device_t)
    labels_tensor = torch.tensor(labels, dtype=torch.long, device=device_t)
    n = len(observations)
    indices = np.arange(n)

    print(f"[warmstart] pretraining action_type head  "
          f"samples={n}  epochs={epochs}  lr={lr}")

    for epoch in range(epochs):
        np.random.shuffle(indices)
        total_loss = 0.0
        total_correct = 0
        batches = 0

        for start in range(0, n, batch_size):
            batch_idx = indices[start : start + batch_size]
            if isinstance(obs_tensor, dict):
                x = {k: v[batch_idx] for k, v in obs_tensor.items()}
            else:
                x = obs_tensor[batch_idx]
            y = labels_tensor[batch_idx]

            # Run the same stateless forward path PPO will use after loading
            # warm-start weights.  recurrent_type=None is required here.
            logits_dict, _, _, _ = model(x, seq_len=1)
            logits = logits_dict["action_type"]
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            total_correct += (preds == y).sum().item()
            batches += 1

        acc = total_correct / n
        avg_loss = total_loss / max(1, batches)
        print(f"  epoch {epoch + 1:3d}/{epochs}  loss={avg_loss:.4f}  acc={acc:.3f}")

    # Unfreeze all parameters for PPO fine-tuning.
    for param in model.parameters():
        param.requires_grad = True

    print(f"[warmstart] pretraining complete — all parameters unfrozen for PPO")
