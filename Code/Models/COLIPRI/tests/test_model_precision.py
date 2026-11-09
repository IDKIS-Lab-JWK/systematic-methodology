from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from COLIPRI.model import resolve_precision  # noqa: E402


def test_feature_precision_auto_cpu_resolves_fp32() -> None:
    assert resolve_precision("cpu", "auto") == ("fp32", None)
    assert resolve_precision("cpu", "fp32") == ("fp32", None)


@pytest.mark.parametrize("precision", ["bf16", "fp16"])
def test_explicit_low_precision_on_cpu_is_rejected(precision: str) -> None:
    with pytest.raises(RuntimeError, match="requires a CUDA"):
        resolve_precision("cpu", precision)


def test_invalid_precision_is_rejected_before_device_fallback() -> None:
    with pytest.raises(ValueError, match="Unsupported precision"):
        resolve_precision("cpu", "tf32")


def test_cuda_precision_rejects_unavailable_cuda(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA feature extraction"):
        resolve_precision("cuda", "auto")


def test_cuda_auto_prefers_fp16_when_bf16_is_not_supported(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)

    assert resolve_precision("cuda:0", "auto") == ("fp16", torch.float16)
