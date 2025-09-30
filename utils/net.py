from typing import Any, Dict, List, Optional
import time


def join_remote(openra: Dict[str, Any], mod_id: str, host: str, port: int, password: str, bin_dir: str,
                slot: Optional[str], spectator: bool, connect_timeout_ms: int = 15000) -> None:
    api = openra['PythonAPI']
    api.SetNetworkConnectTimeout(int(connect_timeout_ms))

    success = api.JoinServer(mod_id, host, int(port), password or "", bin_dir)
    if not success:
        raise RuntimeError(f"Failed to initiate connection to {host}:{port}")

    if not api.WaitForConnection(connect_timeout_ms):
        state = api.GetConnectionState()
        raise RuntimeError(f"Failed to connect to server. Connection state: {state}")

    if spectator:
        api.SetSpectator(True)
    else:
        claim_slot_if_available(openra, slot)
        try_acknowledge_map(api)
        api.SetReady(True)

    # Wait for host to start the game
    if not wait_for_game_start(openra, timeout_ms=600000):  # 5 minutes by default
        state = api.GetLobbyInfo()["Slots"].items()
        raise RuntimeError(f"Timeout waiting for game to start; lobby={state}")


def host_local(openra: Dict[str, Any], mod_id: str, map_uid: str, bin_dir: str, setup_orders: Optional[List[str]] = None) -> None:
    api = openra['PythonAPI']
    setup = list(setup_orders or [])
    if not any(s.startswith("state ") for s in setup):
        setup.append("state 1")
    api.CreateAndStartLocalServer(mod_id, map_uid, bin_dir, setup)

    for _ in range(6000):
        if api.IsInGame():
            break
        api.Step()


def claim_slot_if_available(openra: Dict[str, Any], slot: Optional[str]) -> None:
    api = openra['PythonAPI']
    if slot:
        api.ClaimSlot(slot)
        return
    slots = list(api.GetAvailableSlots())
    if slots:
        api.ClaimSlot(slots[0])


def try_acknowledge_map(api: Any) -> bool:
    try:
        return bool(api.TryAcknowledgeMap())
    except Exception:  # noqa: BLE001
        return False


def get_connection_state(openra: Dict[str, Any]) -> str:
    return openra['PythonAPI'].GetConnectionState()


def is_connected_to_lobby(openra: Dict[str, Any]) -> bool:
    return bool(openra['PythonAPI'].IsConnectedToLobby())


def wait_for_connection(openra: Dict[str, Any], timeout_ms: int = 10000) -> bool:
    return bool(openra['PythonAPI'].WaitForConnection(int(timeout_ms)))


def wait_for_game_start(openra: Dict[str, Any], timeout_ms: int = 300000, poll_ms: int = 100) -> bool:
    api = openra['PythonAPI']
    deadline = time.monotonic() + (timeout_ms / 1000.0)
    last_ack = 0.0
    while time.monotonic() < deadline:
        # Connection health
        if not api.IsConnectedToLobby():
            return False

        # Try to keep lobby state fresh (ack map periodically)
        now = time.monotonic()
        if now - last_ack > 2.0:
            try:
                api.TryAcknowledgeMap()
            except Exception:
                pass
            last_ack = now

        # Progress the network/tick
        api.Step()

        # Game started?
        if api.IsInGame():
            return True

        time.sleep(max(0.0, poll_ms / 1000.0))
    return api.IsInGame()
