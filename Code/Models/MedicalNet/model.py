"""Official dilated ResNet-18 encoder with a five-class global-average head."""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .vendor.resnet import resnet18


def shortcut_a(x, channels, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zeros = out.new_zeros(out.shape[0], channels - out.shape[1], *out.shape[2:])
    # Upstream used out.data/Variable: same forward values but disconnected the
    # shortcut gradient and assumed float32 CUDA. Preserve autograd and dtype.
    return torch.cat((out, zeros), dim=1)


class MedicalNetClassifier(nn.Module):
    def __init__(self, classes=5):
        super().__init__()
        from functools import partial
        released = resnet18(sample_input_D=32, sample_input_H=32,
                            sample_input_W=32, num_seg_classes=2,
                            shortcut_type="A", no_cuda=True)
        for name in ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4"):
            setattr(self, name, getattr(released, name))
        for layer in (self.layer2, self.layer3, self.layer4):
            first = layer[0]
            first.downsample = partial(shortcut_a, channels=first.conv1.out_channels, stride=first.stride)
        self.fc = nn.Linear(512, classes)

    def features(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        return self.layer4(self.layer3(self.layer2(self.layer1(x))))

    def forward(self, x):
        return self.fc(F.adaptive_avg_pool3d(self.features(x), 1).flatten(1))

    def load_pretrained(self, path: str | Path):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        raw = payload.get("state_dict", payload)
        weights = {k.removeprefix("module."): v for k, v in raw.items()}
        # Only the segmentation decoder is legitimately unused.
        weights = {k: v for k, v in weights.items() if not k.startswith("conv_seg.")}
        expected = {k: v for k, v in self.state_dict().items() if not k.startswith("fc.")}
        if set(weights) != set(expected):
            raise RuntimeError(f"MedicalNet backbone mismatch: missing={sorted(set(expected)-set(weights))}; unexpected={sorted(set(weights)-set(expected))}")
        for key, value in weights.items():
            if value.shape != expected[key].shape or not torch.isfinite(value).all():
                raise RuntimeError(f"Invalid MedicalNet tensor: {key}")
        self.load_state_dict({**self.state_dict(), **weights}, strict=True)
        return {"loaded_backbone_tensors": len(weights), "new_head": "fc", "backbone_coverage": 1.0}
