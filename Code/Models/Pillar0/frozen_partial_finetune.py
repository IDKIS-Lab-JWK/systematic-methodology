"""Atlas adapter selecting cached inputs with the original augmentation RNG."""

from pathlib import Path

import torch

from .augmentations import augmentation_contract, draw_orientations
from .frozen_cache import FrozenStore, execution_contract, weights_digest
from .frozen_forward import FrozenAtlasClassifier
from .model import load_official_model
from .partial_finetune import PillarAtlasAdapter
from _linear_probe.common import ContractError, fingerprint


class FrozenPillarAtlasAdapter(PillarAtlasAdapter):
    def implementation_paths(self):
        return [*super().implementation_paths(), Path(__file__),
                *[Path(__file__).with_name(name) for name in
                  ("augmentations.py", "frozen_forward.py", "frozen_cache.py", "prefetch.py")]]

    def planned_identity(self, args):
        # 64 x 384 final-scale tokens plus two 384-D earlier-scale readouts.
        return {**super().planned_identity(args), "input_source": "frozen",
                "augmentation": augmentation_contract(),
                "frozen_boundaries": list(getattr(args, "frozen_scopes", [self.adaptation_scope])),
                "frozen_cache_root": str(args.frozen_cache_root),
                "estimated_raw_fp32_bytes_per_patient_per_boundary": (64 * 384 + 768) * 4 * 10,
                "cache_lifetime": "remove after successful launch; retain after interruption"}

    def _encoder(self, args):
        encoder = load_official_model(repo_id=str(args.model_repo_id), revision=str(args.model_revision),
                                      device=str(args.device), dtype=None,
                                      local_files_only=bool(args.local_files_only))
        if hasattr(self, "_frozen_store") and weights_digest(encoder) != self._frozen_store.contract["vision_weights_sha256"]:
            raise ContractError("Atlas training weights differ from frozen extraction weights")
        return encoder

    def preflight(self, args, rows):
        super().preflight(args, rows)
        if not hasattr(self, "_frozen_store"):
            encoder = self._encoder(args)
            try:
                contract = execution_contract(args, super().resolve_identity(args), encoder,
                                              getattr(args, "frozen_scopes", [self.adaptation_scope]))
            finally:
                del encoder
            digest = fingerprint(contract)
            if digest not in args.frozen_stores:
                args.frozen_stores[digest] = FrozenStore(args, contract)
            self._frozen_store = args.frozen_stores[digest]
        self._frozen_store.prepare(super().prepare_rows(args, rows), lambda: self._encoder(args))

    def resolve_identity(self, args):
        if not hasattr(self, "_frozen_store"):
            raise ContractError("Frozen Atlas inputs must be preflighted before identity resolution")
        return {**super().resolve_identity(args), "frozen_cache": self._frozen_store.identity_for(self.adaptation_scope)}

    def representation_contract(self, args):
        return {**super().representation_contract(args),
                "training_input": "rotation-specific final-scale tokens and frozen earlier-scale readouts",
                "augmentation": augmentation_contract(), "native_feature_normalization": "L2"}

    def build_model(self, args, parameters, device):
        return FrozenAtlasClassifier(self._encoder(args), self.adaptation_scope,
                                     float(parameters.get("dropout", 0.0)))

    def frozen_rows_identity(self, args, rows):
        return self._frozen_store.rows_digest(super().prepare_rows(args, rows), self.adaptation_scope)

    @staticmethod
    def move_inputs(inputs, device):
        return {name: value.to(device) for name, value in inputs.items()}

    def load_batch(self, args, rows, indices, *, training, generator):
        orientations = draw_orientations(len(indices), generator, training)
        items = [self._frozen_store.load(rows[index], orientation, self.adaptation_scope)
                 for index, orientation in zip(indices, orientations)]
        inputs = {name: torch.stack([item[name] for item in items]) for name in ("tokens", "early")}
        labels = torch.as_tensor([int(rows[index]["label"]) for index in indices], dtype=torch.long)
        return inputs, labels
