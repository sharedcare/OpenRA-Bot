from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple, Callable, Set
import os
import csv

import numpy as np
import torch


class BaseAgent:
    """
    Minimal agent interface for OpenRAEnv.

    Implement `act(obs)` to return one or more OpenRA actions in the env's expected
    dict format, e.g.:
      { 'order': 'Move', 'subject': <actor_id>, 'target_cell': (x, y), 'queued': False }
    """

    def act(self, obs: Dict[str, Any]) -> List[Dict[str, Any]]:
        raise NotImplementedError


class RandomMoveAgent(BaseAgent):
    """
    Simple baseline agent: each step, pick one of my alive units and issue a Move
    to a fixed or slightly randomized location.
    """

    def __init__(self, seed: Optional[int] = None, target_cell: Optional[Tuple[int, int]] = None) -> None:
        self._rng = random.Random(seed)
        self._target_cell = target_cell or (10, 44)

    def act(self, obs: Dict[str, Any]) -> List[Dict[str, Any]]:
        actors = obs.get("actors", []) or []
        my_owner = obs.get("my_owner")

        # Fallback to infer local owner from available data when not provided
        if my_owner is None:
            counts: Dict[int, int] = {}
            for a in actors:
                counts[a.get("owner", -1)] = counts.get(a.get("owner", -1), 0) + 1
            if counts:
                my_owner = max(counts.items(), key=lambda kv: kv[1])[0]

        my_units = [a for a in actors if a.get("owner") == my_owner and not a.get("dead", True)]
        if not my_units:
            return []

        subject = self._rng.choice(my_units)
        tx, ty = self._target_cell
        return [{
            "order": "Move",
            "subject": int(subject["id"]),
            "target_cell": (int(tx), int(ty)),
            "queued": False,
        }]


class RuleBasedAgent(BaseAgent):
    """
    A simple rule-based agent that:
    1) If the only owned actor is an MCV, deploy it to a Construction Yard (fact).
    2) Never attempt to deploy a 'fact'.
    3) If a 'fact' exists and the producible catalog is non-empty, issue a StartProduction action.
    4) Production order cycles through: powr -> proc -> tent/barr -> powr -> ...
       Picks 'tent' if available in catalog otherwise 'barr'.
    5) If any production queue item is Done==True, place it using PlaceableAreas.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)
        # Index into production cycle
        self._produce_idx: int = 0
        # Fixed cycle with a placeholder token for barracks choice
        self._produce_cycle: List[str] = ['powr', 'proc', 'barracks', 'powr', 'proc', 'barracks']
        self._is_deployed = False

    def act(self, obs: Dict[str, Any]) -> List[Dict[str, Any]]:
        actors = obs.get("actors", []) or []
        my_owner = obs.get("my_owner")
        if my_owner is None:
            # Fallback owner inference
            counts: Dict[int, int] = {}
            for a in actors:
                counts[a.get("owner", -1)] = counts.get(a.get("owner", -1), 0) + 1
            if counts:
                my_owner = max(counts.items(), key=lambda kv: kv[1])[0]

        my_units = [a for a in actors if a.get("owner") == my_owner and not a.get("dead", True)]
        my_types_lc = [str(a.get("type", "")).lower() for a in my_units]
        has_fact = any(t == 'fact' for t in my_types_lc)

        # 1) Only MCV present -> deploy
        if len(my_units) == 1 and my_types_lc and my_types_lc[0] == 'mcv' and not has_fact and not self._is_deployed:
            mcv = my_units[0]
            # Make sure the unit can deploy now (avoid spamming invalid order)
            orders = set(str(x).lower() for x in (mcv.get('available_orders') or []))
            if 'deploytransform' in orders:
                self._is_deployed = True
                return [{
                    "order": "DeployTransform",
                    "subject": int(mcv["id"]),
                    "queued": False,
                }]
            # If cannot deploy now, do nothing this tick
            return []

        # 5) Place any completed production first
        build_actions = self._maybe_build(obs, my_units)
        if build_actions:
            return build_actions

        # 3) Produce if we have a factory and a non-empty catalog
        if has_fact:
            catalog = obs.get('producible_catalog') or []
            if catalog:
                act = self._maybe_produce(obs, catalog)
                if act is not None:
                    return [act]

        return []

    # ----- Helpers -----

    def _maybe_build(self, obs: Dict[str, Any], my_units: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        production = (obs.get('production') or {})
        queues = (production.get('Queues') or [])
        if not queues:
            return []

        # Build map for selecting a subject that can Build
        def _has_build_order(u: Dict[str, Any]) -> bool:
            return 'build' in set(str(x).lower() for x in (u.get('available_orders') or []))
        build_subject_id = None
        for u in my_units:
            if _has_build_order(u):
                build_subject_id = int(u['id'])
                break
        if build_subject_id is None and my_units:
            # Fallback to any owned unit if none explicitly offers Build
            build_subject_id = int(my_units[0]['id'])

        # Normalize placeable areas to lowercase keys
        pa_raw = obs.get('placeable_areas') or {}
        placeable_areas: Dict[str, List[List[int]]] = {str(k).lower(): v for k, v in pa_raw.items()}

        # Find the first done item and place it
        for q in queues:
            try:
                q_actor = int(q.get('ActorId', -1))
                items = q.get('Items') or []
                for it in items:
                    if bool(it.get('Done', False)):
                        name = str(it.get('Item') or '').lower()
                        cells = placeable_areas.get(name) or []
                        if not cells:
                            # No known placeable cells for this building; skip
                            continue
                        # Pick a deterministic or random cell; choose first for reproducibility
                        cx, cy = int(cells[0][0]), int(cells[0][1])
                        return [{
                            "order": "PlaceBuilding",
                            "subject": int(q_actor),
                            "target_cell": (cx, cy),
                            "target_string": name,
                            "extra_data": int(q_actor),
                            "queued": False,
                        }]
            except Exception:
                continue
        return []

    def _maybe_produce(self, obs: Dict[str, Any], catalog: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        # Allowed names
        allowed = set()
        for b in catalog:
            nm = str(b.get('Name') or '').lower()
            if nm:
                allowed.add(nm)

        # Choose producer: prefer factory queues
        production = (obs.get('production') or {})
        queues = (production.get('Queues') or [])
        if not queues:
            return None

        # Determine next item based on cycle and availability
        choice = self._next_build_choice(allowed)
        if choice is None:
            return None

        # Build set of enabled producers that can produce `choice`
        enabled_producers = set()
        for q in queues:
            try:
                if not bool(q.get('Enabled', False)) or len(q.get('Items')) >= 1:
                    continue
                items = q.get('Producible') or []
                if any(str(it.get('Name', '')).lower() == choice for it in items):
                    enabled_producers.add(int(q.get('ActorId', -1)))
            except Exception:
                continue

        # Prefer my producers; among them prefer 'fact' type
        my_owner = int(obs.get('my_owner', -1))
        my_producers = [a for a in (obs.get('actors') or []) if int(a.get('owner', -1)) == my_owner and int(a.get('id', -1)) in enabled_producers]
        subject_id = None
        if my_producers:
            fact_like = [a for a in my_producers if str(a.get('type', '')).lower() == 'fact']
            subject_id = int((fact_like[0] if fact_like else my_producers[0]).get('id', -1))
        if subject_id is None:
            # Fallback: pick any enabled producer
            subject_id = next(iter(enabled_producers), None)
        if subject_id is None:
            return None

        # Issue StartProduction
        action = {
            "order": "StartProduction",
            "subject": int(subject_id),
            "target_string": choice,
            "queued": True,
        }
        # Advance cycle after successful selection
        self._advance_cycle()
        return action

    def _next_build_choice(self, allowed: Set[str]) -> Optional[str]:
        # Try up to the cycle length to find a valid item
        attempts = len(self._produce_cycle)
        idx = self._produce_idx
        for _ in range(attempts):
            token = self._produce_cycle[idx % len(self._produce_cycle)]
            if token == 'barracks':
                cand = 'tent' if 'tent' in allowed else ('barr' if 'barr' in allowed else None)
                if cand is not None:
                    return cand
            else:
                if token in allowed:
                    return token
            idx += 1
            # Do not advance persisted index yet; only after selection
        return None

    def _advance_cycle(self) -> None:
        self._produce_idx = (self._produce_idx + 1) % len(self._produce_cycle)


class PPOAgent(BaseAgent):
    """
    PPO Agent wrapper with PPO training for MultiDiscrete actions over OpenRAEnvironment.
    Expects `info['action_mask']` with at least 'action_type'.
    """

    def __init__(self, model, device: str = "cpu") -> None:
        super().__init__()
        self.model = model
        self.device = device

    # ---------- Acting ----------
    def _obs_to_tensor(self, obs: Any) -> Any:
        if isinstance(obs, dict):
            if "vector" in obs:
                x = obs["vector"]
            else:
                raise ValueError("Unsupported dict observation for RLAgent")
        else:
            x = obs
        return torch.as_tensor(x, device=self.device).unsqueeze(0)

    def _mask_to_tensors(self, mask: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in (mask or {}).items():
            try:
                t = torch.as_tensor(v, device=self.device)
                if t.dim() == 1:
                    t = t.unsqueeze(0)
                out[k] = t
            except Exception:
                pass
        return out

    def act(self, obs: Any, info: Optional[Dict[str, Any]] = None) -> Any:  # type: ignore[override]
        self.model.eval()
        x = self._obs_to_tensor(obs)
        with torch.no_grad():
            logits, _ = self.model(x)
            masks = self._mask_to_tensors((info or {}).get("action_mask", {}))
            action, _ = self.model.policy.sample(logits, masks=masks)
        return action.squeeze(0).cpu().numpy()

    # ---------- PPO Training ----------
    @staticmethod
    def _heads_order() -> List[str]:
        return ["action_type", "unit_idx", "target_x", "target_y", "target_idx", "unit_type"]

    @staticmethod
    def _masked_logits(logits: Dict[str, torch.Tensor], masks: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for k, lg in logits.items():
            if k == "action_type" and masks and (k in masks) and masks[k] is not None:
                m = masks[k].to(lg.dtype)
                add = torch.log(m.clamp(min=1e-6))
                out[k] = lg + add
            else:
                out[k] = lg
        return out

    def _logprob_and_entropy(self, logits: Dict[str, torch.Tensor], actions: torch.Tensor,
                             masks: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if masks is None:
            masks = {}
        total_logp = torch.zeros(actions.shape[0], device=actions.device)
        total_entropy = torch.zeros(actions.shape[0], device=actions.device)
        lg = self._masked_logits(logits, masks)
        for i, h in enumerate(self._heads_order()):
            dist = torch.distributions.Categorical(logits=lg[h])
            a = actions[:, i]
            total_logp = total_logp + dist.log_prob(a)
            total_entropy = total_entropy + dist.entropy()
        return total_logp, total_entropy

    @staticmethod
    def _gae_returns(rewards: np.ndarray, dones: np.ndarray, values: np.ndarray,
                     gamma: float, lam: float) -> Tuple[np.ndarray, np.ndarray]:
        adv = np.zeros_like(rewards, dtype=np.float32)
        lastgaelam: float = 0.0
        for t in reversed(range(len(rewards))):
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * values[t + 1] * nonterminal - values[t]
            lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
            adv[t] = lastgaelam
        ret = adv + values[:-1]
        return adv, ret

    def train(self,
              env,
              total_updates: int = 100,
              num_steps: int = 2048,
              gamma: float = 0.99,
              gae_lambda: float = 0.95,
              clip_coef: float = 0.2,
              ent_coef: float = 0.01,
              vf_coef: float = 0.5,
              max_grad_norm: float = 1.0,
              learning_rate: float = 1e-3,
              update_epochs: int = 4,
              minibatch_size: int = 256,
              target_kl: float = 0.03,
              checkpoint_fn: Optional[Callable[[int, Any], None]] = None,
              log_path: Optional[str] = None) -> None:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        obs, info = env.reset()
        device = torch.device(self.device)
        self.model.to(device)

        # Prepare logging
        if log_path:
            try:
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                if not os.path.isfile(log_path):
                    with open(log_path, "w", newline="", encoding="utf-8") as f:
                        w = csv.writer(f)
                        w.writerow(["update", "mean_reward", "std_reward", "nz_frac", "approx_kl", "entropy_mean", "policy_loss_full", "value_loss_full"])
            except Exception:
                pass

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
                logits, value = self.model(x)
                # Build masks from info if available (only action_type used during sampling)
                masks_t: Dict[str, torch.Tensor] = {}
                if isinstance(info, dict):
                    am = info.get("action_mask", {}) or {}
                    if 'action_type' in am:
                        t = torch.as_tensor(am['action_type'], device=device)
                        if t.dim() == 1:
                            t = t.unsqueeze(0)
                        masks_t['action_type'] = t
                action_t, per_head_logps = self.model.policy.sample(logits, masks=masks_t)
                action_np = action_t.squeeze(0).cpu().numpy()

                # Store rollout
                obs_buf.append(obs)
                actions_buf.append(action_np)
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
                _, last_v = self.model(x)
                v_last = float(last_v.item())

            values = np.array(val_buf + [v_last], dtype=np.float32)
            rewards = np.array(rew_buf, dtype=np.float32)
            dones = np.array(done_buf, dtype=np.float32)
            advantages, returns = self._gae_returns(rewards, dones, values, gamma=gamma, lam=gae_lambda)

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

                    logits, values_pred = self.model(mb_obs)
                    new_logp, entropy = self._logprob_and_entropy(logits, mb_actions, masks=mb_masks)
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
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    optimizer.step()

                # Early stop by KL
                with torch.no_grad():
                    logits_full, _ = self.model(b_obs)
                    new_logp_full, _ = self._logprob_and_entropy(logits_full, b_actions, masks=b_masks)
                    approx_kl = (b_logp - new_logp_full).mean().item()
                    if approx_kl > target_kl:
                        break

            # Optional checkpoint callback
            if checkpoint_fn is not None:
                checkpoint_fn(update, self.model)

            # ---------- Logging ----------
            try:
                # Recompute full-batch stats for logging
                with torch.no_grad():
                    logits_full, values_full = self.model(b_obs)
                    new_logp_full, entropy_full = self._logprob_and_entropy(logits_full, b_actions, masks=b_masks)
                    ratio_full = (new_logp_full - b_logp).exp()
                    unclipped_full = ratio_full * b_adv
                    clipped_full = torch.clamp(ratio_full, 1.0 - clip_coef, 1.0 + clip_coef) * b_adv
                    policy_loss_full = -torch.min(unclipped_full, clipped_full).mean().item()
                    value_loss_full = torch.nn.functional.mse_loss(values_full.squeeze(-1), b_returns).item()
                    approx_kl = (b_logp - new_logp_full).mean().item()
                    entropy_mean = entropy_full.mean().item()

                rewards_np = np.array(rew_buf, dtype=np.float32)
                mean_reward = float(rewards_np.mean()) if rewards_np.size else 0.0
                std_reward = float(rewards_np.std()) if rewards_np.size else 0.0
                nz_frac = float((np.abs(rewards_np) > 1e-8).sum() / max(1, rewards_np.size))

                # Action-type distribution
                try:
                    a_type = np.array(actions_buf, dtype=np.int64)[:, 0]
                    unique, counts = np.unique(a_type, return_counts=True)
                    dist = {int(u): int(c) for u, c in zip(unique, counts)}
                except Exception:
                    dist = {}

                # Mask coverage (action_type)
                mask_mean = None
                try:
                    if mask_buf and ('action_type' in mask_buf[0]):
                        m = np.stack([m['action_type'] for m in mask_buf], axis=0)
                        mask_mean = float(m.mean())
                except Exception:
                    mask_mean = None

                # Console log
                print(f"[PPO] update={update+1} mean_reward={mean_reward:.4f} std={std_reward:.4f} nz={nz_frac:.3f} "
                      f"KL={approx_kl:.5f} entropy={entropy_mean:.4f} ploss={policy_loss_full:.4f} vloss={value_loss_full:.4f} "
                      f"atype_dist={dist} mask_mean={mask_mean if mask_mean is not None else 'n/a'}")

                # CSV log
                if log_path:
                    with open(log_path, "a", newline="", encoding="utf-8") as f:
                        w = csv.writer(f)
                        w.writerow([update + 1, f"{mean_reward:.6f}", f"{std_reward:.6f}", f"{nz_frac:.6f}",
                                    f"{approx_kl:.6f}", f"{entropy_mean:.6f}", f"{policy_loss_full:.6f}", f"{value_loss_full:.6f}"])
            except Exception:
                pass
