"""Verify the asset-value reward has headroom above RuleBasedAgent.

Runs two agents under reward_mode="asset" for the same number of steps:

  1. RuleBasedAgent  — builds the fixed [powr, proc, barr] order, then idles.
  2. GreedyDevAgent  — same opening, but keeps every enabled queue busy
                       (extra economy + military units), so it accumulates
                       far more cost-weighted net worth.

If the asset reward is well-designed, GreedyDevAgent's cumulative reward and
final actor count must be strictly higher than RuleBasedAgent's. That proves
RuleBasedAgent is NOT optimal under this reward — i.e. RL has real headroom.

Usage:
    python scripts/verify_asset_reward.py --bin-dir <path> [--steps 200]
"""

import argparse
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.openra_env import make_env  # noqa: E402
from agent.agent import RuleBasedAgent  # noqa: E402

# Building actor types (mirrors OpenRAEnv._BUILDING_TYPES); used to prefer
# producing mobile units over more buildings in unit queues.
_BUILDING_TYPES = {'fact', 'powr', 'apwr', 'proc', 'barr', 'tent', 'weap',
                   'afld', 'spen', 'syrd', 'dome', 'hpad', 'eye', 'atek',
                   'stek', 'fix', 'gap', 'gun', 'iron', 'pbox', 'hbox',
                   'sbiz', 'agun', 'silo'}


class GreedyDevAgent(RuleBasedAgent):
    """Rule-based economy opening + continuous infantry production.

    Builds the same powr/proc/barracks economy as RuleBasedAgent, but the
    moment any queue can produce infantry it keeps that queue busy. The extra
    army value (e1 ~ 100 cost each) is the headroom that a fixed build order
    leaves on the table — so this agent must out-earn RuleBasedAgent under the
    asset reward.
    """

    _INFANTRY = ['e1', 'e3', 'e2', 'e4', 'dog', 'e6', 'e7']

    def act(self, obs: Dict[str, Any]) -> List[Dict[str, Any]]:
        actors = obs.get("actors", []) or []
        my_owner = self._infer_my_owner(actors, obs.get("my_owner"))
        if my_owner is None:
            return []

        my_units = [a for a in actors if int(a.get("owner", -1)) == my_owner and not bool(a.get("dead", True))]
        my_types_lc = [str(a.get("type", "")).lower() for a in my_units]
        has_fact = any(t == "fact" for t in my_types_lc)

        # 1) Deploy the MCV first.
        if len(my_units) == 1 and my_types_lc and my_types_lc[0] == "mcv" and not has_fact:
            mcv = my_units[0]
            orders = set(str(x).lower() for x in (mcv.get("available_orders") or []))
            if "deploytransform" in orders:
                return [{"order": "DeployTransform", "subject": int(mcv["id"]), "queued": False}]
            return []

        # 2) Place any completed building first.
        build_actions = self._maybe_build(obs, my_units)
        if build_actions:
            return build_actions

        # 3) Keep any infantry-capable queue busy (the army-value differentiator).
        infantry_act = self._maybe_infantry(obs)
        if infantry_act is not None:
            return [infantry_act]

        # 4) Otherwise fall back to the rule-based economy opening
        #    (powr -> proc -> barracks cycle).
        catalog = self._get_production_catalog(obs)
        if has_fact or catalog:
            act = self._maybe_produce(obs)
            if act is not None:
                return [act]
        return []

    def _maybe_infantry(self, obs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        production = obs.get("production") or {}
        for q in (production.get("Queues") or []):
            try:
                if not bool(q.get("Enabled", False)) or len(q.get("Items") or []) >= 1:
                    continue
                producible = [str(it.get("Name", "")).lower()
                              for it in (q.get("Producible") or []) if it.get("Name")]
                pick = next((u for u in self._INFANTRY if u in producible), None)
                if pick is None:
                    continue
                return {
                    "order": "StartProduction",
                    "subject": int(q.get("ActorId", -1)),
                    "target_string": pick,
                    "queued": True,
                }
            except Exception:
                continue
        return None


def run_agent(env, agent, steps: int, label: str) -> Dict[str, float]:
    from collections import Counter
    obs, info = env.reset()
    total = 0.0
    picks: Counter = Counter()
    for step in range(steps):
        actions = agent.act(obs)
        for a in (actions or []):
            if isinstance(a, dict) and a.get("order") == "StartProduction":
                picks[str(a.get("target_string"))] += 1
        obs, reward, terminated, truncated, info = env.step(actions)
        total += float(reward)
        if terminated or truncated:
            break

    my_owner = int(obs.get("my_owner", -1))
    my_alive = [a for a in (obs.get("actors") or [])
                if int(a.get("owner", -1)) == my_owner and not bool(a.get("dead", False))]
    types = Counter(str(a.get("type", "")).lower() for a in my_alive)
    buildings = sum(1 for a in my_alive if str(a.get("type", "")).lower() in _BUILDING_TYPES)
    units = len(my_alive) - buildings
    result = {
        "cumulative_reward": total,
        "actors": float(len(my_alive)),
        "buildings": float(buildings),
        "units": float(units),
        "cash": float(obs.get("cash", 0) or 0),
    }
    print(f"[{label}] reward={total:.3f} actors={len(my_alive)} "
          f"buildings={buildings} units={units} cash={result['cash']:.0f}")
    print(f"[{label}]   owned={dict(types)}")
    print(f"[{label}]   start_production_attempts={dict(picks)}")
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify asset reward headroom over RuleBasedAgent.")
    p.add_argument("--bin-dir", default="F:/Projects/OpenRA/bin")
    p.add_argument("--mod-id", default="ra")
    p.add_argument("--map-uid", default="b53e25e007666442dbf62b87eec7bfbe8160ef3f")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--ticks-per-step", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    def new_env():
        return make_env(
            bin_dir=args.bin_dir,
            mod_id=args.mod_id,
            map_uid=args.map_uid,
            ticks_per_step=args.ticks_per_step,
            observation_type="feature",
            enable_actions=["noop", "move", "attack", "produce", "build", "deploy"],
            reward_mode="asset",
        )

    print(f"=== asset-reward headroom check ({args.steps} steps) ===")

    env = new_env()
    rb = run_agent(env, RuleBasedAgent(), args.steps, "RuleBasedAgent")
    env.close()

    env = new_env()
    greedy = run_agent(env, GreedyDevAgent(), args.steps, "GreedyDevAgent ")
    env.close()

    gap = greedy["cumulative_reward"] - rb["cumulative_reward"]
    print("=== result ===")
    print(f"reward gap (greedy - rulebased) = {gap:+.3f}")
    if gap > 0:
        print("PASS: greedy out-earns rule-based -> asset reward has headroom for RL.")
    else:
        print("FAIL: no headroom -> reward still caps at rule-based behavior; revise weights.")


if __name__ == "__main__":
    main()
