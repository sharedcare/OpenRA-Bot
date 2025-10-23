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

    resources: List[Dict[str, Any]] = []
    for r in state.Resources:
        resources.append({
            'cell_x': int(r.CellX),
            'cell_y': int(r.CellY),
            'type_index': int(r.TypeIndex),
            'density': int(r.Density),
        })

    obs: Dict[str, Any] = {
        'world_tick': int(state.WorldTick),
        'net_frame': int(state.NetFrame),
        'local_frame': int(state.LocalFrame),
        'actors': actors,
        'resources': resources,
    }

    # Optional production overview and placeable areas if available from engine
    try:
        prod = getattr(state, 'Production', None)
        if prod is not None and getattr(prod, 'Queues', None) is not None:
            queues: List[Dict[str, Any]] = []
            for q in prod.Queues:
                try:
                    items = []
                    for it in (q.Items or []):
                        items.append({
                            'Item': getattr(it, 'Item', ''),
                            'Cost': int(getattr(it, 'Cost', 0)),
                            'Progress': int(getattr(it, 'Progress', 0)),
                            'Paused': bool(getattr(it, 'Paused', False)),
                            'Done': bool(getattr(it, 'Done', False)),
                        })
                    buildable = []
                    for b in (q.Buildable or []):
                        buildable.append({
                            'Name': getattr(b, 'Name', ''),
                            'Cost': int(getattr(b, 'Cost', 0)),
                        })
                    queues.append({
                        'ActorId': int(getattr(q, 'ActorId', 0)),
                        'Type': getattr(q, 'Type', ''),
                        'Group': getattr(q, 'Group', None),
                        'Enabled': bool(getattr(q, 'Enabled', False)),
                        'Items': items,
                        'Buildable': buildable,
                    })
                except Exception:
                    continue
            obs['production'] = { 'Queues': queues }
    except Exception:
        pass

    try:
        placeable = getattr(state, 'PlaceableAreas', None)
        if placeable is not None:
            pa: Dict[str, List[List[int]]] = {}
            for entry in placeable:
                try:
                    unit_type = getattr(entry, 'UnitType', '')
                    cells = getattr(entry, 'Cells', []) or []
                    coords = [[int(c.X), int(c.Y)] for c in cells]
                    if unit_type:
                        pa.setdefault(unit_type, []).extend(coords)
                except Exception:
                    continue
            obs['placeable_areas'] = pa
    except Exception:
        pass

    my_owner = int(openra['Game'].LocalClientId)
    available = set()
    for u in actors:
        if u['owner'] == my_owner and not u['dead']:
            for oid in u.get('available_orders', []):
                available.add(oid)
    obs['valid_action_mask'] = sorted(list(available))
    obs['my_owner'] = my_owner

    return obs
