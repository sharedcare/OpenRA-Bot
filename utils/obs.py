from typing import Any, Dict, List


def build_observation(openra: Dict[str, Any]) -> Dict[str, Any]:
    api = openra['PythonAPI']
    state = api.GetState()

    actors: List[Dict[str, Any]] = []
    for a in state.Actors:
        actors.append({
            'id': int(a.ActorId),
            'type': str(a.Type) if a.Type is not None else '',
            'owner': int(a.OwnerIndex),
            'cell_bits': int(a.CellBits),
            'cell_x': int(a.CellX),
            'cell_y': int(a.CellY),
            'hp': int(a.HP),
            'max_hp': int(a.MaxHP),
            'dead': bool(a.IsDead),
            'available_orders': list(a.AvailableOrders) if getattr(a, 'AvailableOrders', None) is not None else [],
        })

    obs = {
        'world_tick': int(state.WorldTick),
        'net_frame': int(state.NetFrame),
        'local_frame': int(state.LocalFrame),
        'actors': actors,
    }

    my_owner = int(openra['Game'].LocalClientId)
    available = set()
    for u in actors:
        if u['owner'] == my_owner and not u['dead']:
            for oid in u.get('available_orders', []):
                available.add(oid)
    obs['valid_action_mask'] = sorted(list(available))
    obs['my_owner'] = my_owner

    return obs
