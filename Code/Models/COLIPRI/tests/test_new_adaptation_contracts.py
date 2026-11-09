from __future__ import annotations

import json
from pathlib import Path

from COLIPRI.zero_shot import load_prompts


PACKAGE = Path(__file__).resolve().parents[1]


def test_zero_shot_prompt_profile_is_sealed_five_class_publication_source():
    profile, raw = load_prompts(PACKAGE / "zero_shot_prompts.json")
    assert profile["parameter_source"] == "publication"
    assert profile["temperature"] == 0.07
    assert len(profile["classes"]) == 5
    assert any(key.startswith("#") for key in raw)


def test_new_colipri_profiles_are_valid_json_with_source_notes():
    strategies = {
        "final_block": "PrimusMFinalBlockFineTune",
        "final_stage": "PrimusMFinalStageFineTune",
    }
    for scope, method in strategies.items():
        stem = f"primus_m_{scope}_finetune"
        for source in ("publication", "default", "grid"):
            document = json.loads(
                (PACKAGE / f"{source}_{stem}.json").read_text(encoding="utf-8")
            )
            assert any(key.startswith("#") for key in document)
            assert document["method"] == method
            assert document["parameter_source"] == source
            executable = document.get("parameters", document.get("fixed_parameters"))
            assert executable["adaptation_scope"] == scope
            if source == "publication":
                assert document["parameter_selection"] == "fixed"
                assert "candidates" not in document and "search_space" not in document
            if source == "grid":
                assert document["candidate_count"] == 3
