"""Split the pinned Atlas path without dropping its earlier-scale readouts."""

import math

import torch
import torch.nn.functional as F
from einops import rearrange

from .partial_finetune import AtlasPartialClassifier, _atlas_stages
from .windows import apply_pillar_ct_windows
from _linear_probe.common import ContractError

SCOPES = ("final_block", "final_stage")


def atlas_layout(encoder):
    _atlas_stages(encoder)
    visual = encoder.model.visual
    if not visual.multiscale_feats:
        raise ContractError("Frozen Atlas input requires the released multiscale readout")
    cfg = visual.model_config["modalities"]["abdomen_ct"]
    layout = visual.prepare_multiscale_layout(
        img_size=cfg["image_size"], merge_ratio=cfg["merge_ratio"],
        local2global=cfg["local2global"], patch_size=cfg["patch_size"])
    if len(layout) != 3 or math.prod(layout[-1]["grid_size"]) != 1:
        raise ContractError("Frozen Atlas input requires exactly three scales and one final window")
    return visual, layout, cfg


def stage_arguments(layout, cfg, level):
    return {"grid_sizes": [item["grid_size"] for item in layout[level:]],
            "multiscale_layout": layout[level:], "merge_ratio": cfg["merge_ratio"],
            "local2global": cfg["local2global"], "modality": "abdomen_ct"}


def _pool(visual, tokens, batch_size):
    tokens = rearrange(tokens, "(b nw) k c -> b (nw k) c", b=batch_size)
    return visual.maxpool(tokens.transpose(1, 2)).squeeze(2)


def capture_prefixes(encoder, raw_hu, scopes):
    if encoder.training or not scopes or set(scopes) - set(SCOPES):
        raise ContractError("Frozen Atlas extraction requires eval mode and a supported scope")
    visual, layout, cfg = atlas_layout(encoder)
    x, _ = visual.build_scales({"abdomen_ct": apply_pillar_ct_windows(raw_hu)})
    if len(x) != 3:
        raise ContractError("Atlas produced an unexpected number of image scales")
    early = []
    for level in range(2):
        x = visual.atlas_models[level](x, **stage_arguments(layout, cfg, level))
        early.append(_pool(visual, x[0], raw_hu.shape[0]))
        x = x[1:]
    if len(x) != 1:
        raise ContractError("Atlas final stage must receive one scale")
    pooled_early = torch.cat(early, dim=1).detach()
    result = {}
    if "final_stage" in scopes:
        result["final_stage"] = {"tokens": x[0].detach().clone(), "early": pooled_early.clone()}
    if "final_block" in scopes:
        for block in visual.atlas_models[2].blocks[:-1]:
            x = block(x, **stage_arguments(layout, cfg, 2))
        result["final_block"] = {"tokens": x[0].detach().clone(), "early": pooled_early.clone()}
    return result


def features_from_prefix(encoder, inputs, scope):
    visual, layout, cfg = atlas_layout(encoder)
    # Released blocks replace elements of their input list. Always give them
    # a fresh list so repeated cache reads cannot inherit a previous forward.
    x = [inputs["tokens"]]
    if scope == "final_stage":
        x = visual.atlas_models[2](x, **stage_arguments(layout, cfg, 2))
    elif scope == "final_block":
        x = visual.atlas_models[2].blocks[-1](x, **stage_arguments(layout, cfg, 2))
    else:
        raise ContractError(f"Unsupported frozen Atlas scope: {scope}")
    final = _pool(visual, x[0], inputs["early"].shape[0])
    # The released extract_vision_feats calls encode_image(normalize=True),
    # including when this project's wrapper is passed normalize=False.
    return F.normalize(torch.cat([inputs["early"], final], dim=1), dim=-1)


class FrozenAtlasClassifier(AtlasPartialClassifier):
    def forward(self, inputs):
        features = features_from_prefix(self.encoder, inputs, self.scope)
        if torch.is_grad_enabled() and not features.requires_grad:
            raise ContractError("Frozen Atlas execution detached the trainable suffix")
        return self.classifier(features)
