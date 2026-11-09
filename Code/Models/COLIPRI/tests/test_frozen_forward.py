"""Real installed Primus/EVA/rotary/projector/pooler, with small synthetic CTs."""

import copy
import os
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch

from COLIPRI.frozen_forward import capture_prefixes, FrozenPrimusMPartialClassifier, pooled_from_prefix
from COLIPRI.partial_finetune import PrimusMPartialClassifier
from COLIPRI.model import _activate_source


def tiny_encoder():
    _activate_source(Path(__file__).resolve().parents[4] / "tmp" / "upstream_colipri")
    from colipri.model.image import ImageEncoder
    from colipri.model.multimodal import Model
    from colipri.pooling import AttentionPool1D
    from dynamic_network_architectures.architectures.primus import Primus
    backbone = Primus(1, 72, (8, 8, 8), 1, eva_depth=16, eva_numheads=3,
                      input_shape=(16, 16, 16), use_rot_pos_emb=True, use_abs_pos_embed=False,
                      drop_path_rate=0.2, init_values=0.1, scale_attn_inner=True)
    return Model(ImageEncoder(backbone, torch.nn.Conv3d(72, 768, 1), AttentionPool1D(768, 12)), None).eval()


@pytest.mark.parametrize("scope", ["final_block", "final_stage"])
@pytest.mark.parametrize("device, precision", [("cpu", "fp32"), ("cuda", "fp32"), ("cuda", "bf16")])
def test_real_primus_cached_logits_gradients_and_optimizer_step_match(scope, tmp_path, device, precision):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 unavailable")
    def autocast():
        return torch.autocast("cuda", dtype=torch.bfloat16) if precision == "bf16" else nullcontext()
    torch.manual_seed(37)
    original = PrimusMPartialClassifier(tiny_encoder(), scope).to(device)
    cached = FrozenPrimusMPartialClassifier(copy.deepcopy(original.encoder), scope, grid=(2, 2, 2))
    cached.load_state_dict(original.state_dict())
    cached.to(device)
    images = torch.randn(3, 1, 16, 16, 16, device=device)
    labels = torch.tensor([0, 2, 4], device=device)
    original.eval()
    # First isolate the split itself from batch-dependent CUDA rounding:
    # extract at the same batch shape, serialize per patient, then stack.
    with torch.no_grad(), autocast():
        tokens = []
        outputs, grid = capture_prefixes(original.encoder, images, [scope])
        assert grid == (2, 2, 2)
        for index in range(len(images)):
            path = tmp_path / f"patient_{index}.pt"
            torch.save(outputs[scope][index].cpu(), path)
            tokens.append(torch.load(path, weights_only=True))
    tokens = torch.stack(tokens).to(device)
    original.train()
    cached.train()
    # Selected blocks use stochastic depth during training. Preserve its RNG.
    state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state() if device == "cuda" else None
    with autocast():
        logits = original(images)
        loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    torch.set_rng_state(state)
    if cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state)
    with autocast():
        observed = cached(tokens)
        loss = torch.nn.functional.cross_entropy(observed, labels)
    loss.backward()
    torch.testing.assert_close(logits, observed, rtol=1e-5, atol=1e-6)
    for (name, value), (other_name, other) in zip(original.named_parameters(), cached.named_parameters()):
        assert name == other_name
        if value.requires_grad:
            assert value.grad is not None and other.grad is not None
            torch.testing.assert_close(value.grad, other.grad, rtol=1e-4, atol=2e-6)
        else:
            assert value.grad is None and other.grad is None
    for model in (original, cached):
        torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4).step()
    for a, b in zip(original.parameters(), cached.parameters()):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=2e-6)


@pytest.mark.parametrize("scope", ["final_block", "final_stage"])
def test_bf16_per_patient_extraction_has_bounded_batch_rounding_on_synthetic_ct(scope):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA BF16 unavailable")
    torch.manual_seed(37)
    original = PrimusMPartialClassifier(tiny_encoder(), scope).cuda().eval()
    cached = FrozenPrimusMPartialClassifier(copy.deepcopy(original.encoder), scope, grid=(2, 2, 2)).cuda().eval()
    cached.load_state_dict(original.state_dict())
    images = torch.randn(3, 1, 16, 16, 16, device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        tokens = torch.cat([capture_prefixes(original.encoder, images[i:i + 1], [scope])[0][scope]
                            for i in range(len(images))])
        logits = original(images)
        observed = cached(tokens)
    # Per-patient extraction and batched execution can select different CUDA
    # kernels. These are explicit BF16-scale tolerances, not bit-exact claims
    # or evidence of unchanged cohort-level accuracy after retraining.
    torch.testing.assert_close(logits.float(), observed.float(), rtol=0, atol=2**-6)
    torch.testing.assert_close(logits.float().softmax(-1), observed.float().softmax(-1),
                               rtol=0, atol=0.002)


def test_two_boundaries_captured_in_one_traversal_and_suffix_skips_frozen_blocks():
    encoder = tiny_encoder()
    calls = [0] * 16
    handles = []
    for index, block in enumerate(encoder.image_encoder.backbone.eva.blocks):
        def count(module, inputs, output, index=index):
            calls[index] += 1
        handles.append(block.register_forward_hook(count))
    with torch.no_grad():
        outputs, grid = capture_prefixes(encoder, torch.randn(1, 1, 16, 16, 16), ["final_stage", "final_block"])
    assert calls == [1] * 15 + [0]
    assert set(outputs) == {"final_stage", "final_block"}
    model = FrozenPrimusMPartialClassifier(encoder, "final_block", grid=grid).eval()
    with torch.no_grad():
        model(outputs["final_block"])
    assert calls == [1] * 16
    for handle in handles:
        handle.remove()


@pytest.mark.parametrize("batch", [1, 3])
def test_projector_and_pooler_layout_matches_released_forward(batch):
    """Singleton batch stride affects cuDNN layout even with identical values."""
    encoder = tiny_encoder()
    images = torch.randn(batch, 1, 16, 16, 16)
    layouts = {}
    phase = "reference"
    handles = []
    for name, module in [("projector", encoder.image_encoder.projector), ("pooler", encoder.image_encoder.pooler)]:
        def capture_layout(module, inputs, name=name):
            layouts[(phase, name)] = inputs[0].stride()
        handles.append(module.register_forward_pre_hook(capture_layout))
    try:
        with torch.no_grad():
            prefixes, grid = capture_prefixes(encoder, images, ["final_block", "final_stage"])
            reference = encoder.encode_image(images, pool=True, project=True, normalize=False)
            for scope, prefix in prefixes.items():
                phase = scope
                # Match per-patient shard storage and microbatch assembly.
                loaded = torch.stack([tensor.contiguous() for tensor in prefix])
                observed = pooled_from_prefix(encoder, loaded, scope, grid)
                for name in ("projector", "pooler"):
                    assert layouts[(scope, name)] == layouts[("reference", name)]
                torch.testing.assert_close(reference, observed)
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.skipif(os.environ.get("COLIPRI_TEST_RELEASED_FROZEN") != "1",
                    reason="Opt-in full-size CUDA check with the locally pinned released checkpoint")
@pytest.mark.parametrize("precision", ["bf16", "fp32"])
def test_released_checkpoint_full_size_cache_round_trip(tmp_path, precision):
    """One synthetic 192-cube, both boundaries, real shard I/O and verification."""
    from types import SimpleNamespace
    from COLIPRI import frozen_cache as fc
    from COLIPRI.model import build_image_encoder
    from COLIPRI.provisioning import resolve_checkpoint
    if not torch.cuda.is_available() or (precision == "bf16" and not torch.cuda.is_bf16_supported()):
        pytest.skip("Required CUDA precision unavailable")
    root = Path(__file__).resolve().parents[4]
    upstream = root / "tmp" / "upstream_colipri"
    checkpoint = resolve_checkpoint(None, upstream_root=upstream, local_files_only=True, provision_log=None)
    encoder, _, _ = build_image_encoder(upstream_root=upstream, checkpoint_path=checkpoint["path"], device="cuda")
    volume = torch.linspace(-1, 1, 192**3, dtype=torch.float32).reshape(1, 192, 192, 192)
    row = {"patient_id": "synthetic-full-size", "image_path": str(tmp_path / "synthetic.pt"), "label": 0}
    volume_sha = fc.tensor_digest(volume)
    volumes = SimpleNamespace(manifest={"entries": {fc._entry_key(row): {"sha256": volume_sha}}},
                              load=lambda row: volume)
    args = SimpleNamespace(cache_root=tmp_path / "cache", results_root=tmp_path / "results",
                           percentage=100, seed=42, parameter_source="publication", device="cuda",
                           training_precision=precision)
    contract = fc.execution_contract(args, {"checkpoint": checkpoint, "synthetic_volume_sha256": volume_sha},
                                     encoder, ["final_stage", "final_block"])
    store = fc.FrozenStore(args, contract)
    try:
        store.prepare([row], volumes, lambda: encoder)
        assert set(store.manifest["forward_verification"]) == {"final_stage", "final_block"}
        with torch.no_grad(), (torch.autocast("cuda", dtype=torch.bfloat16) if precision == "bf16" else nullcontext()):
            reference = encoder.encode_image(volume.unsqueeze(0).cuda(), pool=True, project=True, normalize=False)
            for scope in ("final_stage", "final_block"):
                tokens = torch.stack([store.load(row, scope)]).cuda()
                observed = pooled_from_prefix(encoder, tokens, scope, (24, 24, 24))
                torch.testing.assert_close(reference.float(), observed.float(), rtol=1e-5, atol=1e-6)
    finally:
        store.finish(True)
    assert not store.root.exists()
