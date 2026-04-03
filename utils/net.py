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
        _ensure_remote_player_setup(openra, api, slot)

    # Wait for host to start the game
    if not wait_for_game_start(openra, timeout_ms=600000, slot=slot, spectator=spectator):  # 5 minutes by default
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


def _ensure_remote_player_setup(openra: Dict[str, Any], api: Any, slot: Optional[str]) -> None:
    claim_slot_if_available(openra, slot)
    # Let the slot claim propagate before acknowledging the map and setting ready.
    for _ in range(3):
        try:
            api.Step()
        except Exception:
            break
    try_acknowledge_map(api)
    for _ in range(3):
        try:
            api.Step()
        except Exception:
            break
    api.SetReady(True)
    for _ in range(3):
        try:
            api.Step()
        except Exception:
            break


def _get_local_client(api: Any, slot: Optional[str] = None) -> Optional[Dict[str, Any]]:
    try:
        lobby = api.GetLobbyInfo()
        clients = lobby.get("Clients", []) if isinstance(lobby, dict) else []
        if slot:
            for client in clients:
                if isinstance(client, dict) and client.get("Slot") == slot and bool(client.get("IsBot", False)) is False:
                    return client
        for client in clients:
            if isinstance(client, dict) and bool(client.get("IsBot", False)) is False:
                return client
    except Exception:
        return None
    return None


def get_connection_state(openra: Dict[str, Any]) -> str:
    return openra['PythonAPI'].GetConnectionState()


def is_connected_to_lobby(openra: Dict[str, Any]) -> bool:
    return bool(openra['PythonAPI'].IsConnectedToLobby())


def wait_for_connection(openra: Dict[str, Any], timeout_ms: int = 10000) -> bool:
    return bool(openra['PythonAPI'].WaitForConnection(int(timeout_ms)))


def wait_for_game_start(
    openra: Dict[str, Any],
    timeout_ms: int = 300000,
    poll_ms: int = 100,
    slot: Optional[str] = None,
    spectator: bool = False,
) -> bool:
    api = openra['PythonAPI']
    deadline = time.monotonic() + (timeout_ms / 1000.0)
    last_ready = 0.0
    last_slot_claim = 0.0
    disconnected_since: Optional[float] = None
    acknowledged_map = False
    while time.monotonic() < deadline:
        # Always progress the network state first. During the normal transition from
        # lobby to in-game, IsConnectedToLobby() can briefly drop before IsInGame()
        # flips to true, so checking connectivity first causes false negatives.
        api.Step()

        if api.IsInGame():
            return True

        connected_to_lobby = bool(api.IsConnectedToLobby())
        now = time.monotonic()

        if connected_to_lobby:
            disconnected_since = None

            # Keep lobby state fresh and actively maintain our player state until start.
            if not spectator and now - last_slot_claim > 2.0:
                try:
                    claim_slot_if_available(openra, slot)
                except Exception:
                    pass
                last_slot_claim = now

            local_client = _get_local_client(api, slot=slot)
            local_state = str(local_client.get("State", "")) if local_client else ""

            if not acknowledged_map and local_state != "Ready":
                try:
                    acknowledged_map = bool(api.TryAcknowledgeMap())
                except Exception:
                    acknowledged_map = False

            if not spectator and now - last_ready > 2.0:
                try:
                    api.SetReady(True)
                except Exception:
                    pass
                last_ready = now
        else:
            # Tolerate a short lobby disconnect during map launch / world creation.
            if disconnected_since is None:
                disconnected_since = now
            elif now - disconnected_since > 10.0:
                # One last chance: the world may have been created just after the timeout window.
                for _ in range(5):
                    api.Step()
                    if api.IsInGame():
                        return True
                    time.sleep(0.05)
                return False

        time.sleep(max(0.0, poll_ms / 1000.0))
    return api.IsInGame()
