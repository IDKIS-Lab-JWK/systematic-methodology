"""One CPU microbatch ahead without advancing the trainer's sampler or RNG.

Only deterministic COLIPRI input loaders use this module. A private sampler
predicts the next indices; every consumed batch is checked against the actual
trainer. Checkpoints continue to record only the original, consumed stream.
"""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

from _linear_probe.common import ContractError


class OneAhead:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="colipri-input")
        self.future = None
        self.key = None

    def discard(self):
        if self.future is not None:
            # Join even on validation switches/failures: never leave a loader
            # reading a temporary cache after cleanup starts.
            self.future.cancel()
            if not self.future.cancelled():
                self.future.result()
            self.future = None
            self.key = None

    def get(self, key, load, next_key=None, next_load=None):
        if self.future is not None and self.key == key:
            future, self.future, self.key = self.future, None, None
            value = future.result()
        else:
            self.discard()
            value = load()
        if next_load is not None:
            self.key = next_key
            self.future = self.executor.submit(next_load)
        return value

    def close(self):
        try:
            self.discard()
        finally:
            self.executor.shutdown(wait=True, cancel_futures=True)


class MicrobatchLookahead:
    def __init__(self, stream, effective_batch, micro_batch):
        self.stream = stream
        self.effective_batch = effective_batch
        self.micro_batch = micro_batch
        self.pending = []
        self.expected = self._next()

    def _next(self):
        if not self.pending:
            batch = self.stream.next(self.effective_batch)
            self.pending = [batch[i:i + self.micro_batch]
                            for i in range(0, len(batch), self.micro_batch)]
        return self.pending.pop(0)

    def following(self, indices):
        if list(indices) != self.expected:
            raise ContractError("COLIPRI prefetch sampler disagrees with training order")
        self.expected = self._next()
        return list(self.expected)


@contextmanager
def prefetched_loader(owner, attribute, lookahead=None):
    """Temporarily wrap one instance; never patch shared trainer/module globals."""
    original = getattr(owner, attribute)
    had_override = attribute in owner.__dict__
    saved_override = owner.__dict__.get(attribute)
    queue = OneAhead()

    def load(args, rows, indices, **kwargs):
        training = kwargs.get("training", lookahead is not None)
        if training and lookahead is not None:
            following = lookahead.following(indices)
        elif not training:
            start = indices[-1] + 1
            following = list(range(start, min(start + len(indices), len(rows))))
        else:
            following = []
        # COLIPRI currently ignores the augmentation generator. Do not extend
        # this wrapper to stochastic loaders without explicit RNG accounting.
        key = (id(rows), bool(training), tuple(indices))
        next_key = (id(rows), bool(training), tuple(following))
        return queue.get(
            key, lambda: original(args, rows, indices, **kwargs),
            next_key, (lambda: original(args, rows, following, **kwargs)) if following else None,
        )

    setattr(owner, attribute, load)
    try:
        yield
    finally:
        try:
            queue.close()
        finally:
            if had_override:
                setattr(owner, attribute, saved_override)
            else:
                delattr(owner, attribute)
