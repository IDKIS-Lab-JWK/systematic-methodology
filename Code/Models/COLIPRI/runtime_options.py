"""COLIPRI launch-time batch validation, independent of training identity."""

from __future__ import annotations


def resolve_micro_batch_size(value: int | None, effective_batch_size: int) -> int:
    """Default to a full batch and validate explicit accumulation sizes."""
    batch = int(effective_batch_size)
    if batch <= 0:
        raise ValueError("Effective batch size must be positive.")
    micro = batch if value is None else int(value)
    if micro <= 0:
        raise ValueError("Microbatch size must be positive.")
    if micro > batch:
        raise ValueError(
            f"Selected microbatch size {micro} exceeds the effective batch size ({batch}). "
            f"Select a microbatch size equal to or lower than {batch}."
        )
    if batch % micro:
        allowed = ", ".join(str(size) for size in range(1, batch + 1) if batch % size == 0)
        raise ValueError(
            f"Selected microbatch size {micro} must divide the effective batch size ({batch}) "
            f"evenly. Select one of: {allowed}."
        )
    return micro
