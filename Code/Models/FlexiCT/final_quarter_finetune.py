"""Final-quarter adaptation: blocks 12-15 of the flat FlexiCT vision encoder."""

from pathlib import Path
from typing import Any, Mapping

from _linear_probe.common import ContractError
from .final_block_finetune import (
    FlexiCTFinalBlockAdapter, FlexiCTFinalBlockClassifier, _vision_components,
)
from .model import build_vision_only_encoder


class FlexiCTFinalQuarterClassifier(FlexiCTFinalBlockClassifier):
    def __init__(self, encoder, dropout: float = 0.0):
        super().__init__(encoder, dropout)
        _vision, blocks, norm = _vision_components(encoder)
        selected = (*blocks[12:16], norm)
        for module in selected:
            module.requires_grad_(True)
        object.__setattr__(self, "_selected_modules", selected)


class FlexiCTFinalQuarterAdapter(FlexiCTFinalBlockAdapter):
    method_label = "FinalQuarterFineTune"
    method_id = "final_quarter_finetune"

    def __init__(self):
        super().__init__()
        package = Path(__file__).resolve().parent
        self.publication_profile_path = package / "publication_final_quarter_finetune.json"
        self.default_profile_path = package / "default_final_quarter_finetune.json"
        self.search_profile_path = package / "grid_final_quarter_finetune.json"

    def implementation_paths(self):
        return [*super().implementation_paths(), Path(__file__)]

    def planned_identity(self, args):
        claim = super().planned_identity(args)
        claim["adaptation_scope"] = "vision.blocks[12:16] + vision.norm + new five-class head"
        claim["scope_evidence"] = "publication_adapted; final quarter of flat 16-block encoder"
        return claim

    def representation_contract(self, args):
        contract = super().representation_contract(args)
        contract.pop("final_block_path")
        contract["final_quarter_path"] = "vision.blocks[12:16] + vision.norm"
        return contract

    def validate_parameters(self, parameters: Mapping[str, Any]):
        if parameters.get("adaptation_scope") != "final_quarter":
            raise ContractError("FlexiCT final-quarter method requires adaptation_scope=final_quarter")
        super().validate_parameters({**parameters, "adaptation_scope": "final_block"})

    def build_model(self, args, parameters, device):
        if self._checkpoint_identity is None:
            self.preflight(args, [])
        encoder, _details = build_vision_only_encoder(
            upstream_root=args.upstream_root,
            checkpoint_path=self._checkpoint_identity["path"],
            device=device,
            allow_trusted_legacy_checkpoint=bool(args.allow_trusted_legacy_checkpoint),
            trusted_checkpoint_sha256=args.checkpoint_sha256,
        )
        return FlexiCTFinalQuarterClassifier(encoder, float(parameters.get("dropout", 0.0)))

    def validate_trainable_scope(self, model, parameters):
        self.validate_parameters(parameters)
        _vision, blocks, norm = _vision_components(model.encoder)
        expected = {
            id(p) for module in (*blocks[12:16], norm, model.classifier)
            for p in module.parameters()
        }
        actual = {id(p) for p in model.parameters() if p.requires_grad}
        if not expected or actual != expected:
            raise ContractError("FlexiCT trainable tensors must be exactly blocks 12-15, norm, and head")

    def optimizer_groups(self, model, parameters):
        groups = super().optimizer_groups(model, parameters)
        groups[0]["group_name"] = "flexict_final_quarter_and_norm"
        return groups
