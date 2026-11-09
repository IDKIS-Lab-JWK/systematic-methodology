from __future__ import annotations

import json
from pathlib import Path

import torch

from FlexiCT.attention_cache import _source_identity
from FlexiCT.attention_readout import SingleHeadAttentionReadout
from FlexiCT.run_zero_shot import build_parser as build_zero_shot_parser
from FlexiCT.zero_shot import load_prompts, pairwise_five_class_probabilities
from FlexiCT.zero_shot_model import (
    OFFICIAL_CHECKPOINT_SHA256,
    TEXT_MODEL_REVISION,
)


PACKAGE = Path(__file__).resolve().parents[1]


def test_single_head_attention_shape_and_trainable_contract():
    model = SingleHeadAttentionReadout()
    logits = model(torch.randn(3, 20, 864))
    assert tuple(logits.shape) == (3, 5)
    assert model.attention.num_heads == 1
    assert tuple(model.query.shape) == (1, 1, 864)
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_new_flexict_profiles_are_valid_json_with_inert_source_notes():
    for name in (
        "publication_final_block_finetune.json",
        "default_final_block_finetune.json",
        "grid_final_block_finetune.json",
        "publication_attention_readout.json",
        "default_attention_readout.json",
        "grid_attention_readout.json",
    ):
        document = json.loads((PACKAGE / name).read_text(encoding="utf-8"))
        assert document["parameter_source"] in {"publication", "default", "grid"}
        assert any(key.startswith("#") for key in document)


def test_attention_cache_source_identity_is_resume_stable_and_change_sensitive(
    tmp_path: Path,
):
    source = tmp_path / "volume.nii.gz"
    source.write_bytes(b"first-volume")
    first = _source_identity(source)
    assert _source_identity(source, prior=first) == first
    source.write_bytes(b"changed-volume-content")
    assert _source_identity(source, prior=first) != first


def test_zero_shot_prompt_profile_and_cli_are_pinned():
    profile, raw = load_prompts(PACKAGE / "zero_shot_prompts.json")
    assert profile["parameter_source"] == "publication"
    assert profile["text_model"]["revision"] == TEXT_MODEL_REVISION
    assert profile["scoring"]["logit_scale"] == "checkpoint_learned"
    assert list(profile["classes"]) == list(profile["class_order"])
    assert any(key.startswith("#") for key in raw)

    args = build_zero_shot_parser().parse_args([])
    assert args.checkpoint_sha256 == OFFICIAL_CHECKPOINT_SHA256
    assert args.text_model_revision == TEXT_MODEL_REVISION
    assert args.resume is True


def test_zero_shot_pairwise_and_five_class_normalization():
    similarities = torch.tensor(
        [
            [0.8, 0.2],
            [0.5, 0.4],
            [0.1, 0.6],
            [0.7, 0.3],
            [0.4, 0.4],
        ],
        dtype=torch.float64,
    ).numpy()
    five_class, pairwise = pairwise_five_class_probabilities(similarities, 2.0)
    assert five_class.shape == (5,)
    assert pairwise.shape == (5,)
    assert abs(float(five_class.sum()) - 1.0) < 1e-12
    assert pairwise[0] > pairwise[1] > pairwise[2]
