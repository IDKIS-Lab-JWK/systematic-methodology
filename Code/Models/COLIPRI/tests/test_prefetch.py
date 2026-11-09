import json
import threading
from types import SimpleNamespace

import pytest
import torch

from COLIPRI.prefetch import OneAhead, MicrobatchLookahead
from COLIPRI.prefetched_backends import PrefetchedAdaptationBackend, PrefetchedCOLIPRIBackend
from COLIPRI.backend import COLIPRIBackend
from _supervised_adaptation.runtime import DeterministicBatchStream, torch_load
from _supervised_adaptation.trainer import AdaptationBackend
from _supervised_adaptation.tests.test_runtime_and_resume import TinyAdapter, args, parameters, rows, Logger
from _linear_probe.common import ContractError, candidate_id


def test_one_cpu_batch_is_prepared_while_consumer_is_busy():
    started = threading.Event()
    release = threading.Event()
    queue = OneAhead()
    def next_load():
        started.set()
        assert release.wait(3)
        return "next"
    try:
        assert queue.get(1, lambda: "current", 2, next_load) == "current"
        assert started.wait(3)
        assert not queue.future.done()
        release.set()
        assert queue.get(2, lambda: pytest.fail("Loaded twice")) == "next"
        assert queue.future is None
    finally:
        release.set()
        queue.close()


def test_prefetch_errors_propagate_and_worker_joins():
    queue = OneAhead()
    def fail():
        raise RuntimeError("bad input")
    queue.get(1, lambda: 1, 2, fail)
    with pytest.raises(RuntimeError, match="bad input"):
        queue.get(2, lambda: 2)
    queue.close()


@pytest.mark.parametrize("micro", [1, 5])
def test_lookahead_crosses_batch_and_epoch_boundaries_without_touching_sampler(micro):
    actual = DeterministicBatchStream(7, 17)
    ahead = MicrobatchLookahead(DeterministicBatchStream(7, 17), 5, micro)
    for step in range(12):
        indices = actual.next(5)
        before = actual.state_dict()
        for offset in range(0, 5, micro):
            ahead.following(indices[offset:offset + micro])
        assert actual.position == before["position"]
        assert torch.equal(actual.generator.get_state(), before["generator_state"])
    with pytest.raises(ContractError, match="disagrees"):
        ahead.following([-1])


@pytest.mark.parametrize("micro", [1, 5])
def test_prefetched_training_and_interrupted_resume_match_original(tmp_path, micro):
    run_args = args()
    run_args.micro_batch_size = micro
    run_args.prefetch = True
    context = {"fold": 1}
    def backend(prefetch, fail=None):
        instance = PrefetchedAdaptationBackend(TinyAdapter(fail)) if prefetch else AdaptationBackend(TinyAdapter())
        # Isolate execution equivalence from migration bookkeeping.
        instance.feature_identity = lambda _args: {"synthetic": True}
        return instance
    def fit(instance, folder):
        instance.fit_one(parameters(), rows("train"), rows("validation"), folder,
                         run_args, context, Logger())
    fit(backend(False), tmp_path / "original")
    fit(backend(True), tmp_path / "prefetched")
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        fit(backend(True, 5 // micro + 1), tmp_path / "resumed")
    fit(backend(True), tmp_path / "resumed")
    reference = torch_load(tmp_path / "original" / "model.pt")["trainable_state_dict"]
    for name in ("prefetched", "resumed"):
        observed = torch_load(tmp_path / name / "model.pt")["trainable_state_dict"]
        assert all(torch.equal(value, observed[key]) for key, value in reference.items())
        a = json.loads((tmp_path / "original" / "training_state.json").read_text())
        b = json.loads((tmp_path / name / "training_state.json").read_text())
        assert a["history"] == b["history"]


def test_checkpoint_refuses_changed_regenerated_prefix_values(tmp_path):
    def backend(value_digest):
        adapter = TinyAdapter()
        adapter.frozen_rows_identity = lambda rows: value_digest
        instance = PrefetchedAdaptationBackend(adapter)
        instance.feature_identity = lambda args: {"synthetic": True}
        return instance
    run_args = args()
    run_args.prefetch = True
    backend("original-values").fit_one(parameters(), rows("train"), rows("validation"),
                                      tmp_path, run_args, {"fold": 1}, Logger())
    with pytest.raises(ContractError):
        backend("changed-values").fit_one(parameters(), rows("train"), rows("validation"),
                                         tmp_path, run_args, {"fold": 1}, Logger())


@pytest.mark.parametrize("micro", [1, 16])
def test_probe_prefetch_preserves_all_twenty_candidates_and_resume(tmp_path, micro):
    from pathlib import Path
    profile = json.loads((Path(__file__).parents[1] / "publication_pooling_optimised_linear_probe.json").read_text())
    candidates = []
    for p in COLIPRIBackend().build_candidates(profile):
        p = {**p, "max_steps": 3}
        p.pop("candidate_id", None)
        candidates.append({**p, "candidate_id": candidate_id(p)})
    class Probe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 5)
        def forward_tokens(self, inputs):
            return self.linear(inputs)
    train = [{"patient_id": str(i), "label": i % 5} for i in range(17)]
    run_args = SimpleNamespace(seed=42, device="cpu", training_precision="fp32", micro_batch_size=micro,
                              checkpoint_interval=1, resume=True, prefetch=True)
    def run(prefetch, path, fail=None):
        backend = PrefetchedCOLIPRIBackend() if prefetch else COLIPRIBackend()
        backend.feature_identity = lambda a: {"synthetic": True}
        backend._feature_store = lambda a: SimpleNamespace(identity={"synthetic": True})
        def build(p, seed, device):
            torch.manual_seed(seed)
            return Probe()
        backend._build_probe = build
        calls = 0
        def load(a, r, indices):
            nonlocal calls
            calls += 1
            if calls == fail:
                raise RuntimeError("synthetic interruption")
            return torch.tensor([[float(i % 5), 1.] for i in indices]), torch.tensor([r[i]["label"] for i in indices])
        backend._load_batch = load
        backend._dense_to_tokens = lambda x: x
        return backend._train_candidates(candidates, train, path, run_args, {"fold": 1}, Logger())
    expected, _ = run(False, tmp_path / "original.pt")
    actual, _ = run(True, tmp_path / "prefetch.pt")
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        run(True, tmp_path / "resume.pt", 16 // micro + 1)
    resumed, _ = run(True, tmp_path / "resume.pt")
    assert len(expected) == len(actual) == len(resumed) == 20
    for key in expected:
        for name, value in expected[key].state_dict().items():
            assert torch.equal(value, actual[key].state_dict()[name])
            assert torch.equal(value, resumed[key].state_dict()[name])
