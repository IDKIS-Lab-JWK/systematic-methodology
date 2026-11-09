"""COLIPRI-only I/O scheduling around the unchanged numerical trainers."""

from pathlib import Path

from .backend import COLIPRIBackend, DeterministicBatchStream as ProbeStream
from .prefetch import MicrobatchLookahead, prefetched_loader
from _linear_probe.common import candidate_id, sha256_file
from _supervised_adaptation.compatibility import CompatibleAdaptationBackend
from _supervised_adaptation.runtime import DeterministicBatchStream, torch_load
from _supervised_adaptation.trainer import _seed


def _state(path):
    return torch_load(path)["batch_stream"] if Path(path).is_file() else None


def _log_prefetch(logger):
    logger.info("COLIPRI input prefetch: one CPU microbatch ahead; original sampler/checkpoint state retained; implementation=%s",
                {name: sha256_file(Path(__file__).with_name(name))
                 for name in ("prefetch.py", "prefetched_backends.py")})


class PrefetchedAdaptationBackend(CompatibleAdaptationBackend):
    def dry_run_model_claim(self, args):
        return {**super().dry_run_model_claim(args), "prefetch_microbatches": int(args.prefetch)}

    def _train(self, parameters, train_rows, validation_rows, output_dir, args, context, logger, **kwargs):
        frozen_identity = getattr(self.adapter, "frozen_rows_identity", None)
        if frozen_identity is not None:
            self._frozen_input_identity = {
                "training": frozen_identity(train_rows),
                "validation": frozen_identity(validation_rows) if validation_rows is not None else None,
            }
        if not getattr(args, "prefetch", True):
            return super()._train(parameters, train_rows, validation_rows, output_dir, args, context, logger, **kwargs)

        identifier = str(parameters.get("candidate_id") or candidate_id(parameters))
        seed = _seed(int(args.seed), int(context.get("fold", 0)), identifier) + 17
        stream = DeterministicBatchStream(len(train_rows), seed,
                                          _state(Path(output_dir) / "last_checkpoint.pt"))
        ahead = MicrobatchLookahead(stream, int(parameters["effective_batch_size"]), int(args.micro_batch_size))
        _log_prefetch(logger)
        with prefetched_loader(self.adapter, "load_batch", ahead):
            return super()._train(parameters, train_rows, validation_rows, output_dir, args, context, logger, **kwargs)

    def _training_identity(self, **kwargs):
        identity = super()._training_identity(**kwargs)
        if hasattr(self, "_frozen_input_identity"):
            identity["frozen_input_values"] = self._frozen_input_identity
        return identity

    def _probabilities(self, model, rows, args, parameters, precision_dtype):
        # During training use its single queue, including validation switches.
        if not getattr(args, "prefetch", True) or "load_batch" in self.adapter.__dict__:
            return super()._probabilities(model, rows, args, parameters, precision_dtype)
        with prefetched_loader(self.adapter, "load_batch"):
            return super()._probabilities(model, rows, args, parameters, precision_dtype)


class PrefetchedCOLIPRIBackend(COLIPRIBackend):
    def dry_run_model_claim(self, args):
        return {**super().dry_run_model_claim(args), "prefetch_microbatches": int(args.prefetch)}

    def _train_candidates(self, candidates, train_rows, checkpoint_path, args, context, logger):
        if not getattr(args, "prefetch", True):
            return super()._train_candidates(candidates, train_rows, checkpoint_path, args, context, logger)
        fixed = self._validate_shared_recipe(candidates)
        state = _state(checkpoint_path)
        stream = ProbeStream(len(train_rows), int(args.seed) + int(context.get("fold", 0)) * 1_000_003, state)
        ahead = MicrobatchLookahead(stream, int(fixed.get("effective_batch_size", 16)), int(args.micro_batch_size))
        _log_prefetch(logger)
        with prefetched_loader(self, "_load_batch", ahead):
            return super()._train_candidates(candidates, train_rows, checkpoint_path, args, context, logger)

    def _evaluate_models(self, *args, **kwargs):
        # The probe evaluator runs outside its training loop.
        run_args = args[2] if len(args) > 2 else kwargs["args"]
        if not getattr(run_args, "prefetch", True):
            return super()._evaluate_models(*args, **kwargs)
        with prefetched_loader(self, "_load_batch"):
            return super()._evaluate_models(*args, **kwargs)
