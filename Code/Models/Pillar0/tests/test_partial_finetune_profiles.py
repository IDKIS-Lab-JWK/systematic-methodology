from __future__ import annotations

import json
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1]


def test_atlas_partial_finetune_profiles_have_inert_source_notes():
    strategies = {
        "final_block": "AtlasFinalBlockFineTune",
        "final_stage": "AtlasFinalStageFineTune",
    }
    for scope, method in strategies.items():
        stem = f"atlas_{scope}_finetune"
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
