"""
Entity-based observation builder.

Converts the flat PythonAPI RLState into structured entity features
where each actor is independently encoded (no fixed-size padding).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

# Number of raw features per entity (see _encode_actor).
ENTITY_FEATURE_DIM: int = 14
MAX_ENTITIES: int = 128


_STABLE_TYPE_IDS: Dict[str, int] = {
    name: idx
    for idx, name in enumerate([
        'mcv', 'fact', 'powr', 'apwr', 'proc', 'harv', 'barr', 'tent',
        'weap', 'dome', 'fix', 'afld', 'spen', 'syrd', 'hpad', 'eye',
        'atek', 'stek', 'gap', 'gun', 'pbox', 'hbox', 'agun', 'sbiz',
        'e1', 'e2', 'e3', 'e4', 'e6', 'e7', 'dog', 'spy', 'thf',
        'medi', 'shok', 'gren', 'flmr', 'jeep', 'apc', '1tnk', '2tnk',
        '3tnk', '4tnk', 'arty', 'mgg', 'mrj', 'mnly', 'lstr', 'c17',
    ], start=1)
}


class EntityObservationBuilder:
    """Build entity-based observations from raw RLState dicts.

    Returns a dict with:
        entities: (max_entities, ENTITY_FEATURE_DIM) float32
        entity_mask: (max_entities,) bool — True for valid entities
        scalar: (SCALAR_DIM,) float32 — global economy/map info
    """

    # Unit types that are buildings (same whitelist as openra_env._BUILDING_TYPES).
    _BUILDING_TYPES = frozenset({
        'fact', 'powr', 'proc', 'barr', 'tent', 'weap',
        'afld', 'spen', 'syrd', 'dome', 'hpad', 'eye',
        'atek', 'stek', 'fix', 'gap', 'gun', 'iron',
        'pbox', 'hbox', 'sbiz', 'agun',
    })

    def __init__(self, max_entities: int = MAX_ENTITIES) -> None:
        self.max_entities = max_entities

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, raw: Dict[str, Any], goal_vec: Optional[List[float]] = None) -> Dict[str, np.ndarray]:
        actors = raw.get('actors') or []
        my_owner = int(raw.get('my_owner', -1))
        n = min(len(actors), self.max_entities)

        entities = np.zeros((self.max_entities, ENTITY_FEATURE_DIM), dtype=np.float32)
        mask = np.zeros(self.max_entities, dtype=bool)

        for i, actor in enumerate(actors[:n]):
            entities[i] = self._encode_actor(actor, my_owner)
            mask[i] = True

        scalar = self._encode_scalar(raw, goal_vec)

        return {
            'entities': entities,
            'entity_mask': mask,
            'scalar': scalar,
        }

    # ------------------------------------------------------------------
    # Entity encoding
    # ------------------------------------------------------------------

    @classmethod
    def _encode_actor(cls, actor: Dict[str, Any], my_owner: int) -> np.ndarray:
        """Encode a single actor into ENTITY_FEATURE_DIM floats.

        Feature layout (14 dims):
            0: owner_self          (1.0 if mine, 0 otherwise)
            1: owner_enemy         (1.0 if enemy, 0 otherwise)
            2: is_building         (1.0 if building, 0 if mobile)
            3: is_infantry         (1.0 if infantry-type)
            4: health_ratio        (hp / max_hp)
            5: pos_x_norm          (cell_x / 128)
            6: pos_y_norm          (cell_y / 128)
            7: can_move            (1.0 if 'move' available)
            8: can_attack          (1.0 if 'attack' available)
            9: can_produce         (1.0 if 'startproduction' available)
           10: can_deploy          (1.0 if 'deploytransform' available)
           11: is_idle             (1.0 if idle order present)
           12: type_embed_idx      (stable type id, 0-1 normalised)
           13: type_is_mcv         (1.0 if type == 'mcv')
        """
        owner = int(actor.get('owner', -1))
        atype = str(actor.get('type', '')).lower()
        orders = {str(x).lower() for x in (actor.get('available_orders') or [])}
        order_str = ' '.join(sorted(orders))

        hp = float(int(actor.get('hp', 0)))
        max_hp = float(max(1, int(actor.get('max_hp', 1))))
        cx = float(int(actor.get('cell_x', 0)))
        cy = float(int(actor.get('cell_y', 0)))

        return np.array([
            1.0 if owner == my_owner else 0.0,                  # 0: self
            1.0 if owner != my_owner else 0.0,                  # 1: enemy
            1.0 if cls._is_building_type(atype) else 0.0,       # 2: building
            1.0 if cls._is_infantry(atype, orders) else 0.0,    # 3: infantry
            hp / max_hp,                                         # 4: health
            cx / 128.0,                                          # 5: pos_x
            cy / 128.0,                                          # 6: pos_y
            1.0 if 'move' in orders else 0.0,                    # 7: mobile
            1.0 if 'attack' in orders else 0.0,                  # 8: attack
            1.0 if 'startproduction' in orders else 0.0,         # 9: produce
            1.0 if 'deploytransform' in orders else 0.0,         # 10: deploy
            1.0 if 'idle' in order_str else 0.0,                 # 11: idle
            cls._type_id_norm(atype),                             # 12: type id
            1.0 if atype == 'mcv' else 0.0,                      # 13: is_mcv
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    # Scalar encoding
    # ------------------------------------------------------------------

    SCALAR_BASE_DIM: int = 28

    @classmethod
    def _encode_scalar(cls, raw: Dict[str, Any], goal_vec: Optional[List[float]] = None) -> np.ndarray:
        cash = max(0.0, float(raw.get('cash', 0) or 0.0))
        res_total = max(0.0, float(raw.get('resources_total', 0) or 0.0))
        res_cap = max(1.0, float(raw.get('resource_capacity', 0) or 0.0))
        power = raw.get('power') or {}
        p_prov = max(0.0, float(power.get('provided', 0) or 0.0))
        p_drain = max(0.0, float(power.get('drained', 0) or 0.0))
        p_state = str(power.get('state', '')).lower()
        power_margin = p_prov - p_drain

        prod = raw.get('production') or {}
        queues = prod.get('Queues', []) or []
        active_items = 0
        enabled_queues = 0
        empty_queues = 0
        busy_queues = 0
        done_items = 0
        has_empty = 0.0
        any_done = 0.0
        for q in queues:
            enabled = bool(q.get('Enabled', False))
            if enabled:
                enabled_queues += 1
            items = q.get('Items', []) or []
            if enabled and len(items) == 0:
                empty_queues += 1
            if enabled and len(items) > 0:
                busy_queues += 1
            for it in items:
                if not bool(it.get('Done', False)):
                    active_items += 1
                else:
                    done_items += 1
                    any_done = 1.0
            if len(items) == 0:
                has_empty = 1.0

        my_owner = int(raw.get('my_owner', -1))
        counts: Dict[str, int] = {}
        total_buildings = 0
        total_units = 0
        for actor in raw.get('actors') or []:
            if bool(actor.get('dead', False)):
                continue
            if int(actor.get('owner', -1)) != my_owner:
                continue
            atype = str(actor.get('type', '')).lower()
            if not atype:
                continue
            counts[atype] = counts.get(atype, 0) + 1
            if cls._is_building_type(atype):
                total_buildings += 1
            else:
                total_units += 1

        n_powr = counts.get('powr', 0) + counts.get('apwr', 0)
        n_proc = counts.get('proc', 0)
        n_barr = counts.get('barr', 0) + counts.get('tent', 0)
        n_weap = counts.get('weap', 0)
        n_dome = counts.get('dome', 0)
        n_fix = counts.get('fix', 0)
        n_harv = counts.get('harv', 0)
        n_e1 = counts.get('e1', 0)
        n_e2 = counts.get('e2', 0)
        n_e3 = counts.get('e3', 0)
        n_infantry = sum(counts.get(t, 0) for t in ('e1', 'e2', 'e3', 'e4', 'e6', 'e7', 'dog'))
        n_vehicles = sum(counts.get(t, 0) for t in ('jeep', 'apc', '1tnk', '2tnk', '3tnk', '4tnk', 'arty'))

        base = np.array([
            min(cash / 10000.0, 1.0),                           # 0: cash
            min(res_total / res_cap, 1.0),                      # 1: resource fill
            min(p_prov / 500.0, 1.0),                           # 2: power provided
            min(p_drain / 500.0, 1.0),                          # 3: power drained
            1.0 if 'critical' in p_state else 0.0,              # 4: power critical
            1.0 if 'low' in p_state else 0.0,                   # 5: power low
            float(min(active_items, 5)) / 5.0,                   # 6: active production
            has_empty,                                           # 7: has empty queue
            any_done,                                            # 8: has done item
            float(raw.get('world_tick', 0)) / 20000.0,          # 9: game time
            min(cash / 3000.0, 1.0),                             # 10: short-horizon cash
            float(np.clip(power_margin / 200.0, -1.0, 1.0)),     # 11: signed power margin
            min(n_powr / 8.0, 1.0),                              # 12: power plants
            min(n_proc / 6.0, 1.0),                              # 13: refineries
            min(n_barr / 4.0, 1.0),                              # 14: barracks
            min(n_weap / 4.0, 1.0),                              # 15: war factories
            min((n_dome + n_fix) / 4.0, 1.0),                    # 16: tech/support buildings
            min(n_harv / 12.0, 1.0),                             # 17: harvesters
            min(n_infantry / 50.0, 1.0),                         # 18: infantry total
            min(n_e1 / 30.0, 1.0),                               # 19: rifle infantry
            min((n_e2 + n_e3) / 24.0, 1.0),                      # 20: anti-armor infantry
            min(n_vehicles / 30.0, 1.0),                         # 21: vehicle total
            min(total_buildings / 24.0, 1.0),                    # 22: all buildings
            min(total_units / 80.0, 1.0),                        # 23: all mobile units
            min(enabled_queues / 8.0, 1.0),                      # 24: enabled queues
            min(empty_queues / 8.0, 1.0),                        # 25: empty queues
            min(busy_queues / 8.0, 1.0),                         # 26: busy queues
            min(done_items / 8.0, 1.0),                          # 27: done production items
        ], dtype=np.float32)

        if goal_vec:
            goal_arr = np.array(goal_vec, dtype=np.float32)
            return np.concatenate([base, goal_arr])
        return base

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @classmethod
    def _is_building_type(cls, atype: str) -> bool:
        return atype in cls._BUILDING_TYPES

    @staticmethod
    def _is_infantry(atype: str, _orders: set) -> bool:
        return atype.startswith('e') or atype in {'dog', 'spy', 'thf', 'medi', 'shok', 'gren', 'flmr'}

    @staticmethod
    def _fallback_type_id(atype: str) -> int:
        # Deterministic small hash.  Do not use Python's hash(), which is salted
        # per process and breaks SubprocVecEnv consistency under spawn.
        acc = 0
        for ch in atype:
            acc = (acc * 131 + ord(ch)) % 997
        return 64 + acc

    @classmethod
    def _type_id_norm(cls, atype: str) -> float:
        tid = _STABLE_TYPE_IDS.get(atype)
        if tid is None:
            tid = cls._fallback_type_id(atype)
        return float(min(tid, 1023)) / 1023.0
