from typing import Any, Dict, List, Tuple


def _make_cell_target(openra: Dict[str, Any], xy: Tuple[int, int], subcell: int = 0):
    RLTarget = openra['RLTarget']
    CPos = openra['CPos']
    x, y = int(xy[0]), int(xy[1])
    t = RLTarget()
    t.Type = "Cell"
    t.CellBits = CPos(x, y).Bits
    t.SubCell = int(subcell)
    return t


def _make_actor_target(openra: Dict[str, Any], actor_id: int):
    RLTarget = openra['RLTarget']
    t = RLTarget()
    t.Type = "Actor"
    t.ActorId = int(actor_id)
    return t


def encode_actions(openra: Dict[str, Any], action: Any) -> List[Any]:
    if action is None:
        return []

    if isinstance(action, dict):
        actions_list = [action]
    elif isinstance(action, (list, tuple)):
        actions_list = list(action)
    else:
        raise ValueError("Unsupported action type. Expect dict or list of dicts.")

    RLAction = openra['RLAction']

    rl_actions = []
    for a in actions_list:
        if not isinstance(a, dict):
            continue

        order = str(a.get('order', ''))
        subject = a.get('subject')
        if subject is None:
            continue

        queued = bool(a.get('queued', False))
        target = None

        if 'target_cell' in a:
            target = _make_cell_target(openra, a['target_cell'], int(a.get('subcell', 0)))
        elif 'target_actor' in a:
            target = _make_actor_target(openra, a['target_actor'])

        target_string = a.get('target_string')
        extra_data = int(a.get('extra_data', 0))

        ra = RLAction()
        ra.Order = order
        ra.SubjectActorId = int(subject)
        ra.Queued = bool(queued)
        ra.Target = target
        ra.TargetString = target_string
        ra.ExtraData = int(extra_data)
        rl_actions.append(ra)

    return rl_actions


def send_actions(openra: Dict[str, Any], action: Any) -> None:
    api = openra['PythonAPI']
    rl_actions = encode_actions(openra, action)
    if rl_actions:
        api.SendActions(rl_actions)
