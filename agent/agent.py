from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple, Callable, Set
import os
import csv

import numpy as np
import torch

from models.buffer import Buffer
from models.actor import MultiDiscretePolicy


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
        return [
            {
                "order": "Move",
                "subject": int(subject["id"]),
                "target_cell": (int(tx), int(ty)),
                "queued": False,
            }
        ]


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
        self._produce_cycle: List[str] = ["powr", "proc", "barracks", "powr", "proc", "barracks"]

    @staticmethod
    def _infer_my_owner(actors: List[Dict[str, Any]], hint: Optional[int]) -> Optional[int]:
        if hint is not None:
            return int(hint)
        counts: Dict[int, int] = {}
        for a in actors:
            owner = int(a.get("owner", -1))
            counts[owner] = counts.get(owner, 0) + 1
        if not counts:
            return None
        return max(counts.items(), key=lambda kv: kv[1])[0]

    def act(self, obs: Dict[str, Any]) -> List[Dict[str, Any]]:
        actors = obs.get("actors", []) or []
        my_owner = self._infer_my_owner(actors, obs.get("my_owner"))
        if my_owner is None:
            return []

        my_units = [a for a in actors if int(a.get("owner", -1)) == my_owner and not bool(a.get("dead", True))]
        my_types_lc = [str(a.get("type", "")).lower() for a in my_units]
        has_fact = any(t == "fact" for t in my_types_lc)

        # 1) Only MCV present -> deploy
        if len(my_units) == 1 and my_types_lc and my_types_lc[0] == "mcv" and not has_fact:
            mcv = my_units[0]
            # Make sure the unit can deploy now (avoid spamming invalid order)
            orders = set(str(x).lower() for x in (mcv.get("available_orders") or []))
            if "deploytransform" in orders:
                return [
                    {
                        "order": "DeployTransform",
                        "subject": int(mcv["id"]),
                        "queued": False,
                    }
                ]
            # If cannot deploy now, do nothing this tick
            return []

        # 5) Place any completed production first
        build_actions = self._maybe_build(obs, my_units)
        if build_actions:
            return build_actions

        # 3) Produce if we have a factory and a non-empty catalog
        if has_fact:
            catalog = obs.get("producible_catalog") or []
            if catalog:
                act = self._maybe_produce(obs, catalog)
                if act is not None:
                    return [act]

        return []

    # ----- Helpers -----

    def _maybe_build(self, obs: Dict[str, Any], my_units: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        production = obs.get("production") or {}
        queues = production.get("Queues") or []
        if not queues:
            return []

        # Normalize placeable areas to lowercase keys
        pa_raw = obs.get("placeable_areas") or {}
        placeable_areas: Dict[str, List[List[int]]] = {str(k).lower(): v for k, v in pa_raw.items()}

        # Find the first done item and place it
        for q in queues:
            try:
                q_actor = int(q.get("ActorId", -1))
                items = q.get("Items") or []
                for it in items:
                    if bool(it.get("Done", False)):
                        name = str(it.get("Item") or "").lower()
                        cells = placeable_areas.get(name) or []
                        if not cells:
                            # No known placeable cells for this building; skip
                            continue
                        # Pick a deterministic or random cell; choose first for reproducibility
                        cx, cy = int(cells[0][0]), int(cells[0][1])
                        return [
                            {
                                "order": "PlaceBuilding",
                                "subject": int(q_actor),
                                "target_cell": (cx, cy),
                                "target_string": name,
                                "extra_data": int(q_actor),
                                "queued": False,
                            }
                        ]
            except Exception:
                continue
        return []

    def _maybe_produce(self, obs: Dict[str, Any], catalog: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        # Allowed names
        allowed = set()
        for b in catalog:
            nm = str(b.get("Name") or "").lower()
            if nm:
                allowed.add(nm)

        # Choose producer: prefer factory queues
        production = obs.get("production") or {}
        queues = production.get("Queues") or []
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
                if not bool(q.get("Enabled", False)) or len(q.get("Items") or []) >= 1:
                    continue
                items = q.get("Producible") or []
                if any(str(it.get("Name", "")).lower() == choice for it in items):
                    enabled_producers.add(int(q.get("ActorId", -1)))
            except Exception:
                continue

        # Prefer my producers; among them prefer 'fact' type
        my_owner = int(obs.get("my_owner", -1))
        my_producers = [
            a
            for a in (obs.get("actors") or [])
            if int(a.get("owner", -1)) == my_owner and int(a.get("id", -1)) in enabled_producers
        ]
        subject_id = None
        if my_producers:
            fact_like = [a for a in my_producers if str(a.get("type", "")).lower() == "fact"]
            subject_id = int((fact_like[0] if fact_like else my_producers[0]).get("id", -1))
        if subject_id is None:
            return None
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
            if token == "barracks":
                cand = "tent" if "tent" in allowed else ("barr" if "barr" in allowed else None)
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
        self.use_lstm = getattr(model, "recurrent_type", None) is not None
        self._rnn_state = None
        self.action_types: List[str] = ["noop", "move", "attack", "produce", "build", "deploy"]

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
                # The env emits single-environment masks without an explicit env axis.
                # PPO buffer storage expects shape (num_envs, ...), and the current
                # training loop only supports num_envs=1, so we add that leading axis.
                if t.dim() >= 1 and t.shape[0] != 1:
                    t = t.unsqueeze(0)
                out[k] = t
            except Exception:
                pass
        return out

    @staticmethod
    def _policy_masks(mask: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(mask, dict):
            return {}
        return dict(mask)

    def _action_type_index(self, name: str) -> Optional[int]:
        try:
            return self.action_types.index(name)
        except ValueError:
            return None

    @staticmethod
    def _dummy_mask_like(batch_size: int, dim: int, device: torch.device) -> torch.Tensor:
        mask = torch.zeros((batch_size, dim), dtype=torch.float32, device=device)
        if dim > 0:
            mask[:, 0] = 1.0
        return mask

    @staticmethod
    def _ensure_batch_mask(mask: Optional[torch.Tensor], batch_size: int, dim: int, device: torch.device) -> torch.Tensor:
        if mask is None:
            return PPOAgent._dummy_mask_like(batch_size, dim, device)
        mask = mask.to(device=device, dtype=torch.float32)
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        return mask

    @staticmethod
    def _select_rows(mask_3d: torch.Tensor, row_index: torch.Tensor) -> torch.Tensor:
        row_index = row_index.to(device=mask_3d.device, dtype=torch.long)
        row_index = row_index.clamp(min=0, max=max(0, mask_3d.shape[1] - 1))
        batch_idx = torch.arange(mask_3d.shape[0], device=mask_3d.device)
        return mask_3d[batch_idx, row_index]

    def _build_effective_masks(
        self,
        logits: Dict[str, torch.Tensor],
        raw_masks: Optional[Dict[str, torch.Tensor]],
        actions: Optional[torch.Tensor] = None,
        sampled_action_type: Optional[torch.Tensor] = None,
        sampled_unit_idx: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        raw_masks = raw_masks or {}
        batch_size = next(iter(logits.values())).shape[0]
        device = next(iter(logits.values())).device

        action_type_mask = self._ensure_batch_mask(
            raw_masks.get("action_type"), batch_size, logits["action_type"].shape[-1], device
        )

        action_type = sampled_action_type
        if action_type is None and actions is not None:
            action_type = actions[:, 0]

        unit_idx = sampled_unit_idx
        if unit_idx is None and actions is not None:
            unit_idx = actions[:, 1]

        move_idx = self._action_type_index("move")
        attack_idx = self._action_type_index("attack")
        produce_idx = self._action_type_index("produce")
        build_idx = self._action_type_index("build")
        deploy_idx = self._action_type_index("deploy")

        dummy_unit_idx = self._dummy_mask_like(batch_size, logits["unit_idx"].shape[-1], device)
        dummy_target_idx = self._dummy_mask_like(batch_size, logits["target_idx"].shape[-1], device)
        dummy_target_x = self._dummy_mask_like(batch_size, logits["target_x"].shape[-1], device)
        dummy_target_y = self._dummy_mask_like(batch_size, logits["target_y"].shape[-1], device)
        dummy_unit_type = self._dummy_mask_like(batch_size, logits["unit_type"].shape[-1], device)

        effective = {"action_type": action_type_mask}
        if action_type is None:
            effective["unit_idx"] = self._ensure_batch_mask(
                raw_masks.get("unit_idx"), batch_size, logits["unit_idx"].shape[-1], device
            )
            effective["target_x"] = self._ensure_batch_mask(
                raw_masks.get("target_x"), batch_size, logits["target_x"].shape[-1], device
            )
            effective["target_y"] = self._ensure_batch_mask(
                raw_masks.get("target_y"), batch_size, logits["target_y"].shape[-1], device
            )
            effective["target_idx"] = self._ensure_batch_mask(
                raw_masks.get("target_idx"), batch_size, logits["target_idx"].shape[-1], device
            )
            effective["unit_type"] = self._ensure_batch_mask(
                raw_masks.get("unit_type"), batch_size, logits["unit_type"].shape[-1], device
            )
            return effective

        unit_idx_mask = dummy_unit_idx.clone()
        if move_idx is not None:
            move_mask = self._ensure_batch_mask(raw_masks.get("move_mask"), batch_size, logits["unit_idx"].shape[-1], device)
            unit_idx_mask = torch.where((action_type == move_idx).unsqueeze(-1), move_mask, unit_idx_mask)
        if attack_idx is not None:
            attack_mask = raw_masks.get("attack_mask")
            if attack_mask is not None:
                attack_mask = attack_mask.to(device=device, dtype=torch.float32)
                attack_unit_mask = attack_mask.any(dim=-1).to(dtype=torch.float32)
                unit_idx_mask = torch.where((action_type == attack_idx).unsqueeze(-1), attack_unit_mask, unit_idx_mask)
        if produce_idx is not None:
            produce_queue_mask = self._ensure_batch_mask(
                raw_masks.get("produce_queue_mask"), batch_size, logits["unit_idx"].shape[-1], device
            )
            unit_idx_mask = torch.where((action_type == produce_idx).unsqueeze(-1), produce_queue_mask, unit_idx_mask)
        if build_idx is not None:
            build_mask = self._ensure_batch_mask(raw_masks.get("build_mask"), batch_size, logits["unit_idx"].shape[-1], device)
            unit_idx_mask = torch.where((action_type == build_idx).unsqueeze(-1), build_mask, unit_idx_mask)
        if deploy_idx is not None:
            deploy_mask = self._ensure_batch_mask(raw_masks.get("deploy_mask"), batch_size, logits["unit_idx"].shape[-1], device)
            unit_idx_mask = torch.where((action_type == deploy_idx).unsqueeze(-1), deploy_mask, unit_idx_mask)
        effective["unit_idx"] = unit_idx_mask

        target_x_mask = dummy_target_x.clone()
        target_y_mask = dummy_target_y.clone()
        move_or_build = torch.zeros((batch_size,), dtype=torch.bool, device=device)
        if move_idx is not None:
            move_or_build |= action_type == move_idx
        if build_idx is not None:
            move_or_build |= action_type == build_idx
        if move_or_build.any():
            tx_mask = self._ensure_batch_mask(raw_masks.get("target_x"), batch_size, logits["target_x"].shape[-1], device)
            ty_mask = self._ensure_batch_mask(raw_masks.get("target_y"), batch_size, logits["target_y"].shape[-1], device)
            target_x_mask = torch.where(move_or_build.unsqueeze(-1), tx_mask, target_x_mask)
            target_y_mask = torch.where(move_or_build.unsqueeze(-1), ty_mask, target_y_mask)
        effective["target_x"] = target_x_mask
        effective["target_y"] = target_y_mask

        target_idx_mask = dummy_target_idx.clone()
        if attack_idx is not None and unit_idx is not None:
            attack_mask = raw_masks.get("attack_mask")
            if attack_mask is not None:
                attack_mask = attack_mask.to(device=device, dtype=torch.float32)
                selected_target_mask = self._select_rows(attack_mask, unit_idx)
                target_idx_mask = torch.where((action_type == attack_idx).unsqueeze(-1), selected_target_mask, target_idx_mask)
        effective["target_idx"] = target_idx_mask

        unit_type_mask = dummy_unit_type.clone()
        if produce_idx is not None:
            produce_unit_type_mask = self._ensure_batch_mask(
                raw_masks.get("produce_unit_type_mask"), batch_size, logits["unit_type"].shape[-1], device
            )
            unit_type_mask = torch.where((action_type == produce_idx).unsqueeze(-1), produce_unit_type_mask, unit_type_mask)
        if build_idx is not None:
            build_unit_type_mask = self._ensure_batch_mask(
                raw_masks.get("build_unit_type_mask"), batch_size, logits["unit_type"].shape[-1], device
            )
            unit_type_mask = torch.where((action_type == build_idx).unsqueeze(-1), build_unit_type_mask, unit_type_mask)
        effective["unit_type"] = unit_type_mask

        return effective

    def _sample_action(
        self,
        logits: Dict[str, torch.Tensor],
        raw_masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size = next(iter(logits.values())).shape[0]
        device = next(iter(logits.values())).device
        actions: Dict[str, torch.Tensor] = {}
        logps: Dict[str, torch.Tensor] = {}

        eff = self._build_effective_masks(logits, raw_masks)
        action_type_dist = torch.distributions.Categorical(logits=self._masked_logits({"action_type": logits["action_type"]}, {"action_type": eff["action_type"]})["action_type"])
        actions["action_type"] = action_type_dist.sample()
        logps["action_type"] = action_type_dist.log_prob(actions["action_type"])

        eff = self._build_effective_masks(logits, raw_masks, sampled_action_type=actions["action_type"])
        unit_idx_dist = torch.distributions.Categorical(logits=self._masked_logits({"unit_idx": logits["unit_idx"]}, {"unit_idx": eff["unit_idx"]})["unit_idx"])
        actions["unit_idx"] = unit_idx_dist.sample()
        logps["unit_idx"] = unit_idx_dist.log_prob(actions["unit_idx"])

        eff = self._build_effective_masks(
            logits,
            raw_masks,
            sampled_action_type=actions["action_type"],
            sampled_unit_idx=actions["unit_idx"],
        )
        for head in ("target_x", "target_y", "target_idx", "unit_type"):
            dist = torch.distributions.Categorical(
                logits=self._masked_logits({head: logits[head]}, {head: eff[head]})[head]
            )
            actions[head] = dist.sample()
            logps[head] = dist.log_prob(actions[head])

        ordered = torch.stack([actions[h] for h in self._heads_order()], dim=-1).to(device=device)
        return ordered, logps

    def act(self, obs: Any, info: Optional[Dict[str, Any]] = None) -> Any:  # type: ignore[override]
        self.model.eval()
        x = self._obs_to_tensor(obs)
        with torch.no_grad():
            if self.use_lstm:
                logits, _, self._rnn_state = self.model(x, self._rnn_state, seq_len=1)
            else:
                logits, _, _ = self.model(x, seq_len=1)
            masks = self._mask_to_tensors(self._policy_masks((info or {}).get("action_mask", {})))
            action, _ = self._sample_action(logits, raw_masks=masks)
        return action.squeeze(0).cpu().numpy()

    # ---------- PPO Training ----------
    @staticmethod
    def _heads_order() -> List[str]:
        return MultiDiscretePolicy.HEADS

    @staticmethod
    def _masked_logits(logits: Dict[str, torch.Tensor], masks: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not masks:
            return logits
        out: Dict[str, torch.Tensor] = {}
        for k, lg in logits.items():
            m = masks.get(k, None)
            if m is None:
                out[k] = lg
            else:
                m = m.to(dtype=lg.dtype, device=lg.device)
                out[k] = lg + torch.log(m.clamp_min(1e-6))
        return out

    def _logprob_and_entropy(
        self, logits: Dict[str, torch.Tensor], actions: torch.Tensor, masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if masks is None:
            masks = {}
        total_logp = torch.zeros(actions.shape[0], device=actions.device)
        total_entropy = torch.zeros(actions.shape[0], device=actions.device)
        lg = self._masked_logits(logits, self._build_effective_masks(logits, masks, actions=actions))
        for i, h in enumerate(self._heads_order()):
            dist = torch.distributions.Categorical(logits=lg[h])
            a = actions[:, i]
            total_logp = total_logp + dist.log_prob(a)
            total_entropy = total_entropy + dist.entropy()
        return total_logp, total_entropy

    @staticmethod
    def _gae_returns(
        rewards: np.ndarray, dones: np.ndarray, values: np.ndarray, gamma: float, lam: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        adv = np.zeros_like(rewards, dtype=np.float32)
        lastgaelam: float = 0.0
        for t in reversed(range(len(rewards))):
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * values[t + 1] * nonterminal - values[t]
            lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
            adv[t] = lastgaelam
        ret = adv + values[:-1]
        return adv, ret

    @staticmethod
    def _prepare_log_file(log_path: Optional[str]) -> None:
        if not log_path:
            return
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        if os.path.isfile(log_path):
            return
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "update",
                    "mean_reward",
                    "std_reward",
                    "nz_frac",
                    "approx_kl",
                    "entropy_mean",
                    "policy_loss",
                    "value_loss",
                ]
            )

    @staticmethod
    def _append_log_row(
        log_path: Optional[str],
        update_idx: int,
        mean_reward: float,
        std_reward: float,
        nz_frac: float,
        approx_kl: float,
        entropy_mean: float,
        policy_loss: float,
        value_loss: float,
    ) -> None:
        if not log_path:
            return
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    update_idx,
                    f"{mean_reward:.6f}",
                    f"{std_reward:.6f}",
                    f"{nz_frac:.6f}",
                    f"{approx_kl:.6f}",
                    f"{entropy_mean:.6f}",
                    f"{policy_loss:.6f}",
                    f"{value_loss:.6f}",
                ]
            )

    def train(
        self,
        env,
        total_updates: int = 100,
        num_steps: int = 2048,
        num_envs: int = 1,
        seq_len: int = 128,
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
        log_path: Optional[str] = None,
    ) -> None:
        if num_envs != 1:
            raise NotImplementedError("The current PPO training loop only supports num_envs=1.")
        device = torch.device(self.device)
        self.model.to(device)
        self.action_types = list(getattr(env, "action_types", self.action_types))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        obs, info = env.reset()
        self._rnn_state = None
        self._prepare_log_file(log_path)

        if isinstance(env.observation_space, dict):
            obs_space = env.observation_space
        elif hasattr(env.observation_space, "shape"):
            if len(env.observation_space.shape) == 1:
                obs_space = {"vector": env.observation_space.shape[0]}
            elif len(env.observation_space.shape) == 3:
                obs_space = {"channels": env.observation_space.shape[-1]}
            else:
                obs_space = {"vector": int(np.prod(env.observation_space.shape))}
        else:
            obs_space = {"vector": 1}

        buffer = Buffer(
            num_envs=num_envs,
            seq_len=seq_len,
            buffer_size=num_steps,
            observation_space=obs_space,
            action_space=env.action_space,
            device=device,
            gamma=gamma,
            gae_lambda=gae_lambda,
            num_lstm_layers=getattr(self.model, "num_recurrent_layers", 2),
            hidden_size=getattr(self.model, "recurrent_hidden_size", 256),
            has_action_masks=True,
        )

        for update in range(total_updates):
            self.model.train()
            buffer.reset()
            hidden_states = self.model.init_hidden(num_envs, device)
            rollout_rewards: List[float] = []
            action_type_counts: Dict[int, int] = {}

            while not buffer.is_full:
                x = torch.as_tensor(obs, device=device)
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                rollout_hidden = hidden_states
                if self.use_lstm and hidden_states is not None:
                    rollout_hidden = tuple(h.clone() if h is not None else None for h in hidden_states)

                with torch.no_grad():
                    if self.use_lstm:
                        logits, values, hidden_states = self.model(x, hidden_states, seq_len=1)
                    else:
                        logits, values, _ = self.model(x, seq_len=1)

                raw_mask = self._policy_masks((info.get("action_mask") or {}) if isinstance(info, dict) else {})
                masks_t = self._mask_to_tensors(raw_mask)

                actions_t, per_head_logps = self._sample_action(logits, raw_masks=masks_t)
                actions_np = actions_t.cpu().numpy()
                action_for_env = actions_np[0] if actions_np.ndim > 1 else actions_np

                next_obs, reward, terminated, truncated, next_info = env.step(action_for_env)
                rewards = np.array([reward], dtype=np.float32)
                dones = np.array([terminated or truncated], dtype=bool)
                rollout_rewards.append(float(reward))

                action_type = int(action_for_env[0]) if np.ndim(action_for_env) > 0 else int(action_for_env)
                action_type_counts[action_type] = action_type_counts.get(action_type, 0) + 1

                logprobs_np = np.array([sum(float(lp.item()) for lp in per_head_logps.values())], dtype=np.float32)
                values_np = values.detach().cpu().numpy().reshape(-1)
                if values_np.shape[0] != num_envs:
                    values_np = np.pad(values_np[:num_envs], (0, max(0, num_envs - values_np.shape[0])))

                buffer.add(
                    obs=obs,
                    actions=actions_np,
                    rewards=rewards,
                    dones=dones,
                    values=values_np,
                    logprobs=logprobs_np,
                    masks=masks_t,
                    hidden_state=rollout_hidden,
                )

                if self.use_lstm and hidden_states[0] is not None:
                    done_mask = torch.tensor(dones.astype(np.float32), device=device).view(1, -1, 1)
                    hidden_states = (
                        hidden_states[0] * (1 - done_mask),
                        hidden_states[1] * (1 - done_mask),
                    )

                if dones[0]:
                    obs, info = env.reset()
                    if self.use_lstm:
                        hidden_states = self.model.init_hidden(num_envs, device)
                else:
                    obs = next_obs
                    info = next_info

            with torch.no_grad():
                x = torch.as_tensor(obs, device=device)
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                if self.use_lstm:
                    _, last_v, _ = self.model(x, hidden_states, seq_len=1)
                else:
                    _, last_v, _ = self.model(x, seq_len=1)
                v_last = last_v.reshape(-1)[:num_envs]

            buffer.compute_advantages(v_last, gamma=gamma, lam=gae_lambda)

            approx_kl_values: List[float] = []
            entropy_values: List[float] = []
            policy_loss_values: List[float] = []
            value_loss_values: List[float] = []
            stop_early = False

            for _ in range(update_epochs):
                for batch in buffer.recurrent_mini_batch_generator(minibatch_size, 1):
                    mb_obs = batch["obs"]
                    mb_actions = batch["actions"]
                    mb_old_logprobs = batch["old_logprobs"]
                    mb_advantages = batch["advantages"]
                    mb_returns = batch["returns"]
                    mb_masks = batch["masks"]
                    mb_hidden = batch["hidden_states"]
                    mb_attention_mask = batch["attention_mask"]

                    batch_size, seq_len, num_envs = mb_obs.shape[:3]
                    mb_obs_flat = mb_obs.permute(0, 2, 1, *range(3, mb_obs.dim())).contiguous()
                    mb_obs_flat = mb_obs_flat.view(batch_size * num_envs, seq_len, *mb_obs.shape[3:])

                    if self.use_lstm:
                        h_flat = (
                            mb_hidden[0]
                            .permute(1, 0, 2, 3)
                            .contiguous()
                            .view(self.model.num_lstm_layers, batch_size * num_envs, -1)
                        )
                        c_flat = (
                            mb_hidden[1]
                            .permute(1, 0, 2, 3)
                            .contiguous()
                            .view(self.model.num_lstm_layers, batch_size * num_envs, -1)
                        )
                        mb_hidden_flat = (h_flat, c_flat)
                        logits, values_pred, _ = self.model(mb_obs_flat, mb_hidden_flat, seq_len=seq_len)
                    else:
                        logits, values_pred, _ = self.model(mb_obs_flat, seq_len=seq_len)

                    logits = {k: v.view(batch_size, num_envs, seq_len, -1) for k, v in logits.items()}
                    values_pred = values_pred.view(batch_size, num_envs, seq_len)
                    mb_actions_flat = mb_actions.permute(0, 2, 1, 3).contiguous().view(-1, mb_actions.shape[-1])
                    new_logp, entropy = self._logprob_and_entropy(
                        {k: v.permute(0, 1, 3, 2).contiguous().view(-1, v.shape[-1]) for k, v in logits.items()},
                        mb_actions_flat,
                        masks=(
                            {
                                k: v.permute(0, 2, 1, *range(3, v.dim())).contiguous().view(
                                    batch_size * num_envs * seq_len, *v.shape[3:]
                                )
                                for k, v in mb_masks.items()
                            }
                            if mb_masks
                            else None
                        ),
                    )

                    new_logp = new_logp.view(batch_size, num_envs, seq_len)
                    entropy = entropy.view(batch_size, num_envs, seq_len)
                    mb_old_logprobs = mb_old_logprobs.permute(0, 2, 1)
                    mb_advantages = mb_advantages.permute(0, 2, 1)
                    mb_returns = mb_returns.permute(0, 2, 1)
                    mb_attention_mask = mb_attention_mask.permute(0, 2, 1).float()

                    valid_steps = mb_attention_mask.sum()
                    if valid_steps.item() == 0:
                        continue

                    ratio = torch.exp(new_logp - mb_old_logprobs)
                    approx_kl = ((mb_old_logprobs - new_logp) * mb_attention_mask).sum() / valid_steps
                    unclipped = ratio * mb_advantages
                    clipped = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * mb_advantages
                    policy_loss = -(torch.min(unclipped, clipped) * mb_attention_mask).sum() / valid_steps
                    value_loss = (0.5 * ((values_pred - mb_returns) ** 2) * mb_attention_mask).sum() / valid_steps
                    entropy_loss = -(entropy * mb_attention_mask).sum() / valid_steps
                    loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    optimizer.step()

                    approx_kl_values.append(float(approx_kl.item()))
                    entropy_values.append(float(((entropy * mb_attention_mask).sum() / valid_steps).item()))
                    policy_loss_values.append(float(policy_loss.item()))
                    value_loss_values.append(float(value_loss.item()))

                    if target_kl > 0 and approx_kl.item() > target_kl:
                        stop_early = True
                        break

                if stop_early:
                    break

            mean_reward = float(np.mean(rollout_rewards)) if rollout_rewards else 0.0
            std_reward = float(np.std(rollout_rewards)) if rollout_rewards else 0.0
            nz_frac = float(np.mean(np.abs(np.asarray(rollout_rewards)) > 1e-8)) if rollout_rewards else 0.0
            approx_kl_mean = float(np.mean(approx_kl_values)) if approx_kl_values else 0.0
            entropy_mean = float(np.mean(entropy_values)) if entropy_values else 0.0
            policy_loss_mean = float(np.mean(policy_loss_values)) if policy_loss_values else 0.0
            value_loss_mean = float(np.mean(value_loss_values)) if value_loss_values else 0.0
            mask_mean = buffer.get_mask_statistics()

            print(
                f"[PPO] update={update+1} mean_reward={mean_reward:.4f} std={std_reward:.4f} nz={nz_frac:.3f} "
                f"KL={approx_kl_mean:.5f} entropy={entropy_mean:.4f} ploss={policy_loss_mean:.4f} "
                f"vloss={value_loss_mean:.4f} atype_dist={action_type_counts} "
                f"mask_mean={mask_mean if mask_mean is not None else 'n/a'}"
                f"{' early_stop=1' if stop_early else ''}"
            )
            self._append_log_row(
                log_path,
                update + 1,
                mean_reward,
                std_reward,
                nz_frac,
                approx_kl_mean,
                entropy_mean,
                policy_loss_mean,
                value_loss_mean,
            )

            if checkpoint_fn is not None:
                try:
                    checkpoint_fn(update + 1, self.model)
                except Exception as exc:
                    print(f"[PPO] checkpoint failed at update={update+1}: {exc}")
