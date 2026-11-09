"""In-memory adapters for the exact Merlin and Pillar preprocessing paths."""
from functools import lru_cache
import importlib
import importlib.metadata
from pathlib import Path
import sys
from types import SimpleNamespace
from .direct import InferenceError, sha256, activate_models

RAVE_REVISION = '20adeb873021c864e9410fae8898b0c8da309769'
RAVE_PROCESSOR_HASHES = {
    'ct_processor.py': 'c1fa47919495bf5ab6e1164dc1479073c020b41a3c05c131648b9ade1df3604c',
    'base_processor.py': '44f0b2bb7c7ea06d747dc26007295109731f568c37db8ee2b79761c97572b61f',
}

@lru_cache(maxsize=2)
def _merlin_transform():
    version = importlib.metadata.version('merlin-vlm')
    if version != '0.0.7':
        raise InferenceError(f'Merlin study preprocessing requires merlin-vlm==0.0.7, found {version}')
    from merlin.data.monai_transforms import ImageTransforms
    return ImageTransforms

@lru_cache(maxsize=4)
def _pillar_processor(rave_root):
    root = Path(rave_root).resolve()
    for filename, expected in RAVE_PROCESSOR_HASHES.items():
        path = root / 'vision_engine' / 'processing' / filename
        if not path.is_file() or sha256(path) != expected:
            raise InferenceError(f'Pillar0 requires unchanged RAVE {RAVE_REVISION}: {path}')
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    module = importlib.import_module('vision_engine.processing.ct_processor')
    if Path(module.__file__).resolve() != root / 'vision_engine' / 'processing' / 'ct_processor.py':
        raise InferenceError('Another vision_engine shadows the pinned RAVE source')
    config = SimpleNamespace(anatomy='abdomen', processing={
        'resampling': {'target_spacing': [1.25, 1.25, 1.25]},
        'crop_pad': {'size': [384, 384]},
        'slice_selection': {'enabled': True, 'slices': 384}})
    return module.CTProcessor(config)


def pillar_model_input(array, target=(384, 384, 384)):
    """RATE-Evals final crop/pad and depth flip, after raw RAVE processing."""
    import numpy as np
    import torch
    from torch.nn import functional as F
    volume = torch.from_numpy(np.ascontiguousarray(array)).float().unsqueeze(0)
    if volume.ndim != 4 or not bool(torch.isfinite(volume).all()):
        raise InferenceError('RAVE must produce a finite [D,H,W] raw-HU volume')
    for axis, desired in enumerate(target, start=1):
        current = volume.shape[axis]
        if current > desired:
            start = (current - desired) // 2
            slices = [slice(None)] * 4
            slices[axis] = slice(start, start + desired)
            volume = volume[tuple(slices)]
    differences = [desired - current for desired, current in zip(target, volume.shape[1:])]
    padding = []
    for difference in reversed(differences):
        padding.extend([difference // 2, difference - difference // 2])
    volume = F.pad(volume, padding, value=-1.0)
    return torch.flip(volume, dims=[1]).contiguous()


def preprocess_direct(model_family, path, config):
    """Return the published model input without persistent or temporary tensors.

    Pillar uses the pinned RAVE numerical processor directly, never its pipeline
    or exporter. Its int16 HU truncation, native orientation, resampling default
    zero, per-volume minimum in-plane padding, -1 depth padding and depth flip
    are intentionally retained. Merlin calls its official transform directly.
    """
    activate_models()
    if model_family == 'Merlin':
        return _merlin_transform()({'image': str(path)})['image']
    if model_family != 'Pillar0':
        raise InferenceError(f'No preprocessing adapter for {model_family}')
    import numpy as np
    import SimpleITK as sitk
    assets = config.get('assets', config)
    root = assets.get('rave_root')
    if not root:
        raise InferenceError(f'Pillar0 raw inference requires assets.rave_root pointing to RAVE {RAVE_REVISION}')
    image = sitk.ReadImage(str(path))
    if image.GetDimension() != 3:
        raise InferenceError('Pillar0 requires a three-dimensional source NIfTI')
    processor = _pillar_processor(str(root))
    array = processor._process_volume(image, np.asarray(image.GetSpacing()))
    return pillar_model_input(array)
