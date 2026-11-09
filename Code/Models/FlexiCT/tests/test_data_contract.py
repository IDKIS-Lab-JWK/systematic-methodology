from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as functional

from FlexiCT import INPUT_SHAPE_DHW
from FlexiCT.data import _center_pad_crop, preprocess_nifti, preprocessing_contract


def test_preprocessing_contract_is_explicit_and_code_backed() -> None:
    contract = preprocessing_contract()
    assert contract["orientation"] == "LPS"
    assert contract["target_spacing_xyz_mm"] == [2.0, 2.0, 2.0]
    assert "default background 0 HU" in contract["resampling"]
    assert "output pixel type retained" in contract["resampling"]
    assert contract["intensity"] == "clip [-1000,1000], then whole-volume z-score"
    assert contract["std_definition"].startswith("torch.std correction=1")
    assert contract["padding"] == "symmetric constant padding with normalized volume minimum"
    assert contract["crop"] == "center crop"
    assert contract["output_shape_cdhw"] == [1, 160, 160, 160]
    assert "task_adaptation" in contract["source"]


def test_center_pad_crop_is_symmetric_and_uses_volume_minimum() -> None:
    source = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4)
    source[0, 0, 0, 0] = -7.0
    output = _center_pad_crop(source, (4, 5, 6), functional)
    assert output.shape == (1, 4, 5, 6)
    assert output.min().item() == -7.0
    torch.testing.assert_close(output[:, 1:3, 1:4, 1:5], source)
    assert output[0, 0, 0, 0].item() == -7.0


def test_center_crop_is_deterministic() -> None:
    source = torch.arange(6**3, dtype=torch.float32).reshape(1, 6, 6, 6)
    output = _center_pad_crop(source, (4, 4, 4), functional)
    torch.testing.assert_close(output, source[:, 1:5, 1:5, 1:5])


@pytest.mark.parametrize("target", [(0, 4, 4), (4, -1, 4), (4, 4)])
def test_center_pad_crop_rejects_invalid_targets(target) -> None:
    with pytest.raises(ValueError, match="three positive"):
        _center_pad_crop(torch.zeros(1, 2, 2, 2), target, functional)


def test_synthetic_nifti_preprocessing_contract(tmp_path: Path) -> None:
    sitk = pytest.importorskip("SimpleITK")
    array = np.full((4, 5, 6), 250.0, dtype=np.float32)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((2.0, 2.0, 2.0))
    path = tmp_path / "constant.nii.gz"
    sitk.WriteImage(image, str(path))
    tensor, metadata = preprocess_nifti(path)
    assert tuple(tensor.shape) == (1, *INPUT_SHAPE_DHW)
    assert tensor.dtype == torch.float32
    assert torch.isfinite(tensor).all()
    assert torch.count_nonzero(tensor) == 0
    assert metadata["orientation"] == "LPS"
    assert metadata["resampled_spacing_xyz_mm"] == [2.0, 2.0, 2.0]
    assert metadata["clipped_mean"] == pytest.approx(250.0)
    assert metadata["clipped_std_correction_1"] == 0.0
