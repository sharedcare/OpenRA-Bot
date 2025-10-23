try:
    from OpenRA.Bot.models.actor import (
        VisionEncoder,
        VectorEncoder,
        MixedEncoder,
        MultiDiscretePolicy,
        ActorCritic,
    )
except Exception:  # noqa: BLE001
    from .actor import (
    VisionEncoder,
    VectorEncoder,
    MixedEncoder,
    MultiDiscretePolicy,
    ActorCritic,
    )

__all__ = [
    "VisionEncoder",
    "VectorEncoder",
    "MixedEncoder",
    "MultiDiscretePolicy",
    "ActorCritic",
]


