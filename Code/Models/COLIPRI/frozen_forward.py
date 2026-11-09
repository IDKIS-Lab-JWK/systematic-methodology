"""Split the pinned Primus forward at the inputs to blocks 12 and/or 15."""

import torch
from einops import rearrange

from .partial_finetune import PrimusMPartialClassifier, _primus_components
from _linear_probe.common import ContractError

BOUNDARIES = {"final_stage": 12, "final_block": 15}


def validate_backbone(encoder):
    _primus_components(encoder)
    backbone = encoder.image_encoder.backbone
    if backbone.register_tokens is not None or not isinstance(backbone.up_projection, torch.nn.Identity):
        raise ContractError("Frozen COLIPRI caching requires the pinned no-register, no-decoder backbone")
    if backbone.eva.dynamic_img_size:
        raise ContractError("Frozen COLIPRI caching does not support dynamic patch grids")
    return backbone


def capture_prefixes(encoder, images, scopes):
    """One traversal, stopping before the last requested trainable boundary.

    Caller owns no_grad/autocast. Preserve actual activation dtype losslessly;
    BF16 execution does not imply every residual tensor is BF16.
    """
    backbone = validate_backbone(encoder)
    if encoder.training:
        raise ContractError("Frozen-output extraction requires evaluation mode")
    patches = backbone.down_projection(images)
    grid = tuple(patches.shape[2:])
    x = rearrange(patches, "b c w h d -> b (w h d) c")
    x, rope, keep = backbone.eva._pos_embed(x)
    if keep is not None:
        raise ContractError("Stochastic patch selection cannot be cached")
    wanted = {BOUNDARIES[scope]: scope for scope in scopes}
    result = {}
    for index in range(max(wanted) + 1):
        if index in wanted:
            result[wanted[index]] = x.detach().clone()
        if index == max(wanted):
            break
        x = backbone.eva.blocks[index](x, rope=rope)
    return result, grid


def pooled_from_prefix(encoder, tokens, scope, grid):
    eva = encoder.image_encoder.backbone.eva
    rope = eva.rope.get_embed() if eva.rope is not None else None
    x = tokens
    for block in eva.blocks[BOUNDARIES[scope]:]:
        x = block(x, rope=rope)
    x = eva.norm(x)
    # Match Primus.forward exactly, including singleton-batch strides.
    # transpose(...).reshape(...) preserves values but can give batch stride C
    # instead of N*C. cuDNN then chooses a different output memory layout, which
    # changes BF16 attention-pooler numerics on the released 192-cube model.
    dense = rearrange(x, "b (w h d) c -> b c w h d", w=grid[0], h=grid[1], d=grid[2])
    projected = encoder.image_encoder.projector(dense)
    return encoder.image_encoder.pooler(rearrange(projected, "b c w h d -> b (w h d) c"))


class FrozenPrimusMPartialClassifier(PrimusMPartialClassifier):
    def __init__(self, encoder, scope, dropout=0.0, grid=(24, 24, 24)):
        super().__init__(encoder, scope, dropout)
        validate_backbone(encoder)
        self.grid = tuple(grid)

    def forward(self, tokens):
        pooled = pooled_from_prefix(self.encoder, tokens, self.scope, self.grid)
        if torch.is_grad_enabled() and not pooled.requires_grad:
            raise ContractError("Frozen-cache fine-tuning detached its trainable encoder suffix")
        return self.classifier(pooled)
