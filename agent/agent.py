from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple


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
            # Heuristic: choose the most frequent owner among actors if not present
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


class RLAgent(BaseAgent):
    """
    RL Agent wrapper that uses an actor-critic policy to produce MultiDiscrete actions
    for `OpenRAEnvironment` (HTTP). The agent expects the env to supply `action_mask` in info.
    """

    def __init__(self, model, device: str = "cpu") -> None:
        super().__init__()
        self.model = model
        self.device = device

    def _obs_to_tensor(self, obs: Any) -> Any:
        import torch
        if isinstance(obs, dict):
            # Expect vector-based obs in 'vector' key if dict
            if "vector" in obs:
                x = obs["vector"]
            else:
                raise ValueError("Unsupported dict observation for RLAgent")
        else:
            x = obs
        return torch.as_tensor(x, device=self.device).unsqueeze(0)

    def _mask_to_tensors(self, mask: Dict[str, Any]) -> Dict[str, Any]:
        import torch
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
        import torch
        self.model.eval()
        x = self._obs_to_tensor(obs)
        with torch.no_grad():
            logits, _ = self.model(x)
            masks = self._mask_to_tensors((info or {}).get("action_mask", {}))
            action, _ = self.model.policy.sample(logits, masks=masks)
        return action.squeeze(0).cpu().numpy()
