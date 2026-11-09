"""Fine-tuning adapter using disposable, deterministic encoder prefixes."""

from pathlib import Path

import torch

from .cached_partial_finetune import CachedCOLIPRIPartialAdapter
from .cache import _entry_key
from .frozen_cache import FrozenStore, execution_contract
from .frozen_forward import FrozenPrimusMPartialClassifier
from .model import build_image_encoder


class FrozenCOLIPRIPartialAdapter(CachedCOLIPRIPartialAdapter):
    def implementation_paths(self):
        return [*super().implementation_paths(), Path(__file__),
                *[Path(__file__).with_name(name) for name in
                  ("frozen_cache.py", "frozen_forward.py", "prefetch.py", "prefetched_backends.py")]]

    def planned_identity(self, args):
        return {**super().planned_identity(args), "input_source": "frozen",
                "frozen_boundaries": list(getattr(args, "frozen_scopes", [self.adaptation_scope])),
                "frozen_cache_lifetime": "delete after successful launch; retain on interruption",
                "frozen_cache_directory": str(Path(args.cache_root) / "FrozenRuns"),
                "augmentation": "none"}

    def _encoder(self, args):
        encoder, processor, _ = build_image_encoder(
            upstream_root=args.upstream_root, checkpoint_path=self._checkpoint_identity["path"],
            device=str(args.device))
        self._processor_instance = processor
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
            # The CLI owns stores until every requested scope/stage succeeds.
            registry = args.frozen_stores
            from _linear_probe.common import fingerprint
            digest = fingerprint(contract)
            if digest not in registry:
                registry[digest] = FrozenStore(args, contract)
            self._frozen_store = registry[digest]
        self._frozen_store.prepare(rows, self._store(args), lambda: self._encoder(args))

    def resolve_identity(self, args):
        if not hasattr(self, "_frozen_store"):
            self.preflight(args, [])
        return {**super().resolve_identity(args), "frozen_cache": self._frozen_store.identity_for(self.adaptation_scope)}

    def representation_contract(self, args):
        return {**super().representation_contract(args),
                "training_input": "cached tokens immediately before the selected trainable scope",
                "input_shape_bnc": ["B", 13824, 864], "random_image_augmentation": False}

    def build_model(self, args, parameters, device):
        return FrozenPrimusMPartialClassifier(self._encoder(args), self.adaptation_scope,
                                              float(parameters.get("dropout", 0.0)))

    def frozen_rows_identity(self, rows):
        from _linear_probe.common import fingerprint
        return fingerprint([
            [str(row["patient_id"]), self._frozen_store.manifest["entries"][_entry_key(row)]
             ["tensors"][self.adaptation_scope]["tensor_sha256"]]
            for row in rows
        ])

    def load_batch(self, args, rows, indices, *, training, generator):
        volumes = [self._frozen_store.load(rows[index], self.adaptation_scope) for index in indices]
        labels = torch.as_tensor([int(rows[index]["label"]) for index in indices], dtype=torch.long)
        return torch.stack(volumes), labels
