"""Pillar-0 CT window construction.

The released abdomen encoder consumes eleven channels: the ten anatomical
windows defined by RAVE followed by a per-volume min/max channel.  Keeping the
operation here makes training differentiable and makes the exact channel order
part of this repository's auditable contract.  The values match RAVE commit
20adeb873021c864e9410fae8898b0c8da309769.
"""

from __future__ import annotations

from collections import OrderedDict


CT_WINDOWS = OrderedDict(
    (
        ("lung", (-600.0, 1500.0)),
        ("mediastinum", (50.0, 400.0)),
        ("abdomen", (40.0, 400.0)),
        ("liver", (80.0, 150.0)),
        ("bone", (400.0, 1800.0)),
        ("brain", (40.0, 80.0)),
        ("subdural", (75.0, 215.0)),
        ("stroke", (40.0, 40.0)),
        ("temporal_bone", (600.0, 2800.0)),
        ("soft_tissue", (50.0, 350.0)),
    )
)
WINDOW_ORDER = tuple(CT_WINDOWS) + ("minmax",)


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("Pillar-0 windowing requires PyTorch.") from exc
    return torch


def _canonical_raw_volume(volume):
    """Return raw HU as ``[B, 1, D, H, W]`` without changing intensities."""

    torch = _require_torch()
    if not isinstance(volume, torch.Tensor):
        volume = torch.as_tensor(volume)
    if not volume.is_floating_point() or volume.dtype == torch.float64:
        volume = volume.to(dtype=torch.float32)
    if volume.ndim == 3:
        volume = volume.unsqueeze(0).unsqueeze(0)
    elif volume.ndim == 4:
        # Cache samples are [1,D,H,W]; batched samples may be [B,D,H,W].
        if volume.shape[0] == 1:
            volume = volume.unsqueeze(0)
        else:
            volume = volume.unsqueeze(1)
    if volume.ndim != 5 or volume.shape[1] != 1:
        raise ValueError(
            "Expected raw HU volume with shape [D,H,W], [1,D,H,W], or "
            f"[B,1,D,H,W]; received {tuple(volume.shape)}."
        )
    return volume


def apply_pillar_ct_windows(volume):
    """Map raw HU volumes to Pillar-0's ordered eleven-channel ``[0,1]`` view.

    The ten fixed windows use the standard center/width transform.  The final
    min/max channel is computed independently for every input sample, matching
    RATE-Evals' ``per_sample_windowing: true`` setting.
    """

    torch = _require_torch()
    raw = _canonical_raw_volume(volume)
    channels = []
    source = raw[:, 0]
    for center, width in CT_WINDOWS.values():
        lower = center - width / 2.0
        upper = center + width / 2.0
        channels.append(((source - lower) / (upper - lower)).clamp_(0.0, 1.0))

    flat = source.flatten(start_dim=1)
    minima = flat.amin(dim=1).view(-1, 1, 1, 1)
    maxima = flat.amax(dim=1).view(-1, 1, 1, 1)
    denominator = (maxima - minima).clamp_min(torch.finfo(source.dtype).eps)
    channels.append(((source - minima) / denominator).clamp_(0.0, 1.0))
    return torch.stack(channels, dim=1)


def window_contract() -> dict:
    """Return a JSON-serializable description for provenance manifests."""

    return {
        "channel_count": len(WINDOW_ORDER),
        "channel_order": list(WINDOW_ORDER),
        "anatomical_windows": {
            name: {"center": center, "width": width}
            for name, (center, width) in CT_WINDOWS.items()
        },
        "minmax_scope": "per_sample_raw_hu_volume",
        "output_range": [0.0, 1.0],
        "apply_normalization_after_windowing": False,
    }
