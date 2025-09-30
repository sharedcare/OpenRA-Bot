import os
import sys
from typing import Dict, Any


def ensure_engine(bin_dir: str) -> Dict[str, Any]:
    """
    Initialize pythonnet and load OpenRA assemblies. Returns a dict containing
    PythonAPI and other interop types for convenient access.
    """
    # Load .NET Core runtime
    # import locally to avoid linter import errors when pythonnet is not present
    import pythonnet  # type: ignore
    runtime_config = os.path.join(bin_dir, "OpenRA.runtimeconfig.json")
    pythonnet.load("coreclr", runtime_config=runtime_config)

    os.makedirs(bin_dir, exist_ok=True)
    os.chdir(bin_dir)
    if bin_dir not in sys.path:
        sys.path.append(bin_dir)

    import clr  # type: ignore
    try:
        clr.AddReference('OpenRA.Game')  # type: ignore[attr-defined]
        clr.AddReference('OpenRA.Utility')  # type: ignore[attr-defined]
        clr.AddReference('OpenRA.Platforms.Default')  # type: ignore[attr-defined]
    except Exception:
        # In editor or lints without runtime, these may fail; defer real failures to runtime usage
        pass

    from OpenRA import PythonAPI, CPos, RLAction, RLTarget, Game  # type: ignore

    return {
        'PythonAPI': PythonAPI,
        'CPos': CPos,
        'RLAction': RLAction,
        'RLTarget': RLTarget,
        'Game': Game,
    }
