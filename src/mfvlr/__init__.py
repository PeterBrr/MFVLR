"""MVE-only Phase 2A; engineering assumptions are recorded in docs/."""

from .mve import (
    AppearanceEncoding,
    GlobalEncoding,
    LocalEncoding,
    MVE,
    MVEOutput,
    SharedVisualEncoder,
    VisualTransformerBlock,
)

__all__ = [
    "AppearanceEncoding", "GlobalEncoding", "LocalEncoding", "MVE", "MVEOutput",
    "SharedVisualEncoder", "VisualTransformerBlock",
]
