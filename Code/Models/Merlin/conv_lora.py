"""ConvLoRA adapters for 3D convolutional layers."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as functional


class ConvLoRA3d(nn.Module):
    """Frozen Conv3d plus a trainable low-rank residual kernel."""

    def __init__(self, convolution: nn.Conv3d, rank: int = 2, alpha: float = 2.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("ConvLoRA rank must be positive")
        self.convolution = convolution
        for parameter in self.convolution.parameters():
            parameter.requires_grad = False

        output_channels, input_channels_per_group, kd, kh, kw = (
            self.convolution.weight.shape
        )
        flattened_input = input_channels_per_group * kd * kh * kw
        self.lora_a = nn.Parameter(torch.empty(rank, flattened_input))
        self.lora_b = nn.Parameter(torch.zeros(output_channels, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank

    def adapted_weight(self) -> torch.Tensor:
        update = (self.lora_b @ self.lora_a).view_as(self.convolution.weight)
        return self.convolution.weight + self.scaling * update

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return functional.conv3d(
            inputs,
            self.adapted_weight(),
            self.convolution.bias,
            self.convolution.stride,
            self.convolution.padding,
            self.convolution.dilation,
            self.convolution.groups,
        )


def _parent_and_child(root: nn.Module, qualified_name: str) -> tuple[nn.Module, str]:
    parts = qualified_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def inject_conv_lora(
    encoder: nn.Module,
    rank: int = 2,
    alpha: float = 2.0,
    target_stages: tuple[str, ...] = ("layer1", "layer2", "layer3", "layer4"),
) -> list[str]:
    """Replace Conv3d modules whose qualified names contain a target stage."""
    replacements = []
    for name, module in list(encoder.named_modules()):
        lowered_parts = {part.lower() for part in name.split(".")}
        if isinstance(module, nn.Conv3d) and lowered_parts.intersection(target_stages):
            parent, child_name = _parent_and_child(encoder, name)
            setattr(parent, child_name, ConvLoRA3d(module, rank=rank, alpha=alpha))
            replacements.append(name)
    if not replacements:
        available = [
            name for name, module in encoder.named_modules() if isinstance(module, nn.Conv3d)
        ]
        raise RuntimeError(
            "No Merlin Conv3d modules matched layer1-layer4. "
            f"Available Conv3d names include: {available[:20]}"
        )
    return replacements

