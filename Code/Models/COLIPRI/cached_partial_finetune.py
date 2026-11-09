"""Cached-CT adapter; the original NIfTI adapter remains available for resume."""

from pathlib import Path

import torch

from .partial_finetune import COLIPRIPartialAdapter
from .volume_cache import VolumeStore
from _linear_probe.common import ContractError
from _supervised_adaptation.compatibility import read_migration


class CachedCOLIPRIPartialAdapter(COLIPRIPartialAdapter):
    def _store(self, args):
        if not hasattr(self, "_volume_store"):
            migration = read_migration(Path(args.results_root) / self.method_label)
            if migration is not None:
                for entry in migration["entries"]:
                    if (entry["percentage"] == int(args.percentage)
                            and entry["seed"] == int(args.seed)
                            and entry["parameter_source"] == args.parameter_source
                            and "volume_cache" not in entry["legacy_model_identity"]["pretrained_model"]):
                        raise ContractError(
                            "This result root contains a migrated original-NIfTI run. "
                            "Use -InputSource nifti to resume it, or a fresh -ResultsRoot "
                            "with the same -CacheRoot for a new cached-input experiment."
                        )
            self._volume_store = VolumeStore(args)
        return self._volume_store

    def implementation_paths(self):
        return [*super().implementation_paths(), Path(__file__), Path(__file__).with_name("volume_cache.py")]

    def planned_identity(self, args):
        return {**super().planned_identity(args), "input_source": "cache",
                "volume_cache_root": str(args.cache_root),
                "cache_shared_by": ["final_block", "final_stage"]}

    def preflight(self, args, rows):
        self._store(args).prepare_rows(rows)
        super().preflight(args, rows)

    def resolve_identity(self, args):
        return {**super().resolve_identity(args), "volume_cache": self._store(args).identity}

    def prepare_rows(self, args, rows):
        return self._store(args).prepare_rows(rows)

    def load_batch(self, args, rows, indices, *, training, generator):
        volumes = [self._store(args).load(rows[index]) for index in indices]
        labels = torch.as_tensor([int(rows[index]["label"]) for index in indices], dtype=torch.long)
        return torch.stack(volumes), labels
