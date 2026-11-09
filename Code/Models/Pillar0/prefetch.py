"""One CPU batch ahead; commit augmentation RNG only when the batch is consumed."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import torch

from _linear_probe.common import ContractError, candidate_id, sha256_file
from _supervised_adaptation.compatibility import CompatibleAdaptationBackend
from _supervised_adaptation.runtime import DeterministicBatchStream, torch_load
from _supervised_adaptation.trainer import _seed


class OneAhead:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pillar0-input")
        self.future = self.key = None

    def discard(self):
        if self.future is not None:
            future, self.future, self.key = self.future, None, None
            if not future.cancel():
                future.result()

    def get(self, key, load):
        if self.future is not None and self.key == key:
            future, self.future, self.key = self.future, None, None
            return future.result()
        self.discard()
        return load()

    def submit(self, key, load):
        if self.future is not None:
            raise ContractError("Pillar0 prefetch exceeded one waiting microbatch")
        self.key, self.future = key, self.executor.submit(load)

    def close(self):
        try:
            self.discard()
        finally:
            self.executor.shutdown(wait=True, cancel_futures=True)


class MicrobatchLookahead:
    def __init__(self, stream, effective_batch, micro_batch):
        self.stream, self.effective_batch, self.micro_batch = stream, effective_batch, micro_batch
        self.pending = []
        self.expected = self._next()

    def _next(self):
        if not self.pending:
            batch = self.stream.next(self.effective_batch)
            self.pending = [batch[i:i + self.micro_batch] for i in range(0, len(batch), self.micro_batch)]
        return self.pending.pop(0)

    def following(self, indices):
        if list(indices) != self.expected:
            raise ContractError("Pillar0 prefetch sampler disagrees with training order")
        self.expected = self._next()
        return list(self.expected)


@contextmanager
def prefetched_loader(adapter, lookahead=None):
    original = adapter.load_batch
    had_override = "load_batch" in adapter.__dict__
    saved = adapter.__dict__.get("load_batch")
    queue = OneAhead()

    def load(args, rows, indices, *, training, generator):
        before = generator.get_state().clone() if generator is not None else None

        def key_for(selected, state):
            return (id(rows), bool(training), tuple(selected),
                    state.numpy().tobytes() if state is not None else None)

        def prepare(selected, state):
            isolated = None
            if state is not None:
                isolated = torch.Generator(device="cpu")
                isolated.set_state(state)
            value = original(args, rows, selected, training=training, generator=isolated)
            after = isolated.get_state().clone() if isolated is not None else None
            return value, after

        if training and lookahead is not None:
            following = lookahead.following(indices)
        elif not training:
            start = indices[-1] + 1
            following = list(range(start, min(start + len(indices), len(rows))))
        else:
            following = []
        value, after = queue.get(key_for(indices, before), lambda: prepare(indices, before))
        if generator is not None:
            # Workers never advance the checkpointed generator. Only consuming
            # this batch commits its exact post-augmentation state.
            if not torch.equal(generator.get_state(), before):
                raise ContractError("Pillar0 augmentation generator changed while loading")
            generator.set_state(after)
        if following:
            queue.submit(key_for(following, after), lambda: prepare(following, after))
        return value

    adapter.load_batch = load
    try:
        yield
    finally:
        try:
            queue.close()
        finally:
            if had_override:
                adapter.load_batch = saved
            else:
                delattr(adapter, "load_batch")


class PrefetchedAtlasBackend(CompatibleAdaptationBackend):
    def dry_run_model_claim(self, args):
        return {**super().dry_run_model_claim(args), "input_source": args.input_source,
                "prefetch_microbatches": int(args.prefetch)}

    def _train(self, parameters, train_rows, validation_rows, output_dir, args, context, logger, **kwargs):
        digest = getattr(self.adapter, "frozen_rows_identity", None)
        if digest is not None:
            self._frozen_inputs = {"training": digest(args, train_rows),
                                   "validation": digest(args, validation_rows) if validation_rows is not None else None}
        if not getattr(args, "prefetch", True):
            return super()._train(parameters, train_rows, validation_rows, output_dir, args, context, logger, **kwargs)
        path = Path(output_dir) / "last_checkpoint.pt"
        state = torch_load(path)["batch_stream"] if path.is_file() else None
        identifier = str(parameters.get("candidate_id") or candidate_id(parameters))
        seed = _seed(int(args.seed), int(context.get("fold", 0)), identifier) + 17
        stream = DeterministicBatchStream(len(train_rows), seed, state)
        ahead = MicrobatchLookahead(stream, int(parameters["effective_batch_size"]), int(args.micro_batch_size))
        logger.info("Pillar0 prefetch: one CPU microbatch; augmentation RNG committed on consumption; implementation=%s", sha256_file(__file__))
        with prefetched_loader(self.adapter, ahead):
            return super()._train(parameters, train_rows, validation_rows, output_dir, args, context, logger, **kwargs)

    def _training_identity(self, **kwargs):
        identity = super()._training_identity(**kwargs)
        if hasattr(self, "_frozen_inputs"):
            identity["frozen_input_values"] = self._frozen_inputs
        return identity

    def _probabilities(self, model, rows, args, parameters, precision_dtype):
        if not getattr(args, "prefetch", True) or "load_batch" in self.adapter.__dict__:
            return super()._probabilities(model, rows, args, parameters, precision_dtype)
        with prefetched_loader(self.adapter):
            return super()._probabilities(model, rows, args, parameters, precision_dtype)
