from types import SimpleNamespace

import pytest
import torch.nn as nn

from FlexiCT.final_quarter_finetune import FlexiCTFinalQuarterAdapter, FlexiCTFinalQuarterClassifier
from _linear_probe.common import ContractError, read_json
from _supervised_adaptation.profiles import build_candidates


def encoder():
    model = nn.Module()
    model.vision = nn.Module()
    model.vision.blocks = nn.ModuleList([nn.Linear(2, 2) for _ in range(16)])
    model.vision.norm = nn.LayerNorm(2)
    model.vision.patch_embed = nn.Linear(2, 2)
    model.projection = nn.Linear(2, 2)
    return model


def test_quarter_scope_modes_and_optimizer_coverage():
    adapter = FlexiCTFinalQuarterAdapter()
    parameters = build_candidates(read_json(adapter.publication_profile_path))[0]
    model = FlexiCTFinalQuarterClassifier(encoder())
    adapter.validate_trainable_scope(model, parameters)
    model.train()
    for index, block in enumerate(model.encoder.vision.blocks):
        assert block.training == (index >= 12)
        assert all(p.requires_grad == (index >= 12) for p in block.parameters())
    assert not model.encoder.projection.training
    assert not model.encoder.vision.patch_embed.training
    groups = adapter.optimizer_groups(model, parameters)
    observed = [id(p) for g in groups for p in g["params"]]
    assert len(observed) == len(set(observed))
    assert set(observed) == {id(p) for p in model.parameters() if p.requires_grad}
    model.eval()
    assert all(not module.training for module in model.modules())
    model.encoder.vision.blocks[11].requires_grad_(True)
    with pytest.raises(ContractError, match="exactly blocks"):
        adapter.validate_trainable_scope(model, parameters)


def test_quarter_profiles_are_scope_locked_and_keep_fixed_publication():
    adapter = FlexiCTFinalQuarterAdapter()
    for path, count in ((adapter.publication_profile_path, 1),
                        (adapter.default_profile_path, 1), (adapter.search_profile_path, 3)):
        profile = read_json(path)
        assert profile["method"] == "FinalQuarterFineTune"
        candidates = build_candidates(profile)
        assert len(candidates) == count
        for parameters in candidates:
            adapter.validate_parameters(parameters)
            with pytest.raises(ContractError, match="final_quarter"):
                adapter.validate_parameters({**parameters, "adaptation_scope": "final_block"})
    assert read_json(adapter.publication_profile_path)["parameter_selection"] == "fixed"
