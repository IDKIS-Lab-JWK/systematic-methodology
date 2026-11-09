import json
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest
import torch
from torch import nn

from _cnn.adapter import CNNAdapter
from _cnn.geometry import fit_plan, image_geometry, grid_for_image
from _cnn.data import preprocess, VolumeCache
from _cnn.trainer import CNNBackend
from _linear_probe.common import (CLASS_NAMES, ContractError, candidate_id, atomic_write_json,
    load_manifest, development_mapping, rows_from_mapping, validate_completion, read_json)
from _supervised_adaptation.profiles import load_profile
from _supervised_adaptation.runtime import torch_load
from _supervised_adaptation.workflow import run_pipeline


def volume(path, shape=(10, 12, 14), spacing=(1, 1, 2), affine=None, hu=50):
    path.parent.mkdir(parents=True, exist_ok=True)
    a = np.diag([*spacing, 1.]) if affine is None else affine
    image = nib.Nifti1Image(np.full(shape, hu, np.float32), a)
    image.header.set_xyzt_units("mm")
    nib.save(image, path)
    return path


def row(path, label=0):
    return {"patient_id": path.name.split(".")[0], "class_name": CLASS_NAMES[label],
            "label": label, "image_path": str(path.resolve())}


def test_plan_fits_training_only_and_keeps_spacing_for_large_validation(tmp_path):
    train = [row(volume(tmp_path / "a.nii.gz"))]
    plan = fit_plan(train, [32]*3)
    large = volume(tmp_path / "validation.nii.gz", shape=(80, 90, 40), spacing=(2, 2, 3))
    before = json.dumps(plan, sort_keys=True)
    grid = grid_for_image(image_geometry(large), plan)
    assert grid["expanded_canvas"]
    assert grid["actual_spacing_xyz_mm"] == plan["numeric"]["working_spacing_xyz_mm"]
    assert json.dumps(plan, sort_keys=True) == before
    assert plan["training_patient_count"] == 1
    assert plan["reference_spacing_xyz_mm"] == [1, 1, 2]
    g = image_geometry(large)
    low = np.array(grid["affine"])[:3, 3]
    high = low + (np.array(grid["shape_xyz"])-1) * grid["actual_spacing_xyz_mm"]
    assert np.all(low <= np.array(g["world_low_mm"]) + 1e-6)
    assert np.all(high >= np.array(g["world_high_mm"]) - 1e-6)


def test_orientation_obliquity_hu_padding_and_physical_coverage(tmp_path):
    angle = .2
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    affine = np.eye(4)
    affine[:3, :3] = rotation @ np.diag([-1., -1.3, 2.5])
    affine[:3, 3] = [45, 30, -10]
    path = volume(tmp_path / "oblique.nii.gz", affine=affine)
    plan = fit_plan([row(path)], [32]*3)
    array, metadata = preprocess(path, plan)
    assert array.shape == (1, 32, 32, 32)
    assert array.dtype == np.float32
    assert array.min() == 0
    assert array[0, 16, 16, 16] == pytest.approx(250/450)
    assert metadata["grid"]["actual_spacing_xyz_mm"] == plan["numeric"]["working_spacing_xyz_mm"]
    assert metadata["grid"]["geometric_coverage"] == "full_acquired_world_bounding_box"
    bad = nib.load(path)
    bad.header.set_xyzt_units("unknown")
    nib.save(bad, tmp_path / "unknown.nii.gz")
    with pytest.raises(ContractError, match="units"):
        image_geometry(tmp_path / "unknown.nii.gz")


def test_cache_integrity_orphans_and_owned_cleanup(tmp_path):
    path = volume(tmp_path / "input.nii.gz")
    plan = fit_plan([row(path)], [32]*3)
    root = tmp_path / "cache"
    cache = VolumeCache(root, {"version": "test"})
    expected, _ = cache.get(path, plan)
    assert cache.misses == 1
    actual, _ = cache.get(path, plan)
    assert torch.equal(expected, actual) and cache.hits == 1
    receipt = next(p for p in root.glob("*/*.json") if p.name != "cnn_cache_owner.json")
    receipt.unlink()
    cache.get(path, plan)
    assert cache.misses == 2
    artifact = next(root.glob("*/*.pt"))
    artifact.write_bytes(b"corruption")
    with pytest.raises(ContractError, match="Corrupt"):
        VolumeCache(root, {"version": "test"}).get(path, plan)
    unowned = root / "keep.txt"
    unowned.write_text("unrelated")
    cache.release()
    assert unowned.read_text() == "unrelated"
    assert not list(root.glob("*/*.pt"))


@pytest.mark.parametrize("name", ["MedicalNet", "DenseNet"])
@pytest.mark.parametrize("profile_name,source,count", [("publication", "publication", 1), ("default", "default", 1), ("grid", "grid", 3), ("resolution_grid", "grid", 2)])
def test_packaged_profiles(name, profile_name, source, count):
    adapter = CNNAdapter(name)
    profile, _ = load_profile(adapter.package / f"{profile_name}.json", expected_model=name,
                              expected_method=adapter.method_label, expected_source=source)
    assert len(CNNBackend(adapter).build_candidates(profile)) == count


@pytest.mark.parametrize("override", [{"warmup_steps": 10}, {"dropout": .2}, {"canvas_xyz": [32.5, 32, 32]}, {"weight_decay": float("inf")}])
def test_unsupported_or_nonfinite_settings_are_rejected(override):
    p = parameters()
    p.pop("candidate_id")
    p.update(override)
    with pytest.raises(ContractError):
        CNNBackend(CNNAdapter("MedicalNet"))._validate_parameters(p)


class TinyAdapter(CNNAdapter):
    def __init__(self, fail_at=None):
        super().__init__("MedicalNet")
        self.fail_at, self.calls = fail_at, 0

    def resolve_identity(self, args):
        return {"test_model": True}

    def build_model(self, args, parameters, device):
        return nn.Sequential(nn.Conv3d(1, 2, 3, padding=1), nn.BatchNorm3d(2), nn.ReLU(),
                             nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Dropout(.2), nn.Linear(2, 5)).to(device)

    def load_batch(self, args, rows, indices, training, generator):
        if training:
            self.calls += 1
            if self.calls == self.fail_at:
                raise RuntimeError("synthetic interruption")
        return super().load_batch(args, rows, indices, training, generator)


def parameters(optimizer="adam"):
    adapter = CNNAdapter("MedicalNet")
    p = read_json(adapter.publication_profile_path)["parameters"]
    p.update(epochs=3, effective_batch_size=5, canvas_xyz=[32]*3,
             early_stopping_patience_epochs=0, optimizer=optimizer)
    if optimizer == "sgd":
        p.update(momentum=.9, nesterov=True, scheduler="multistep", milestone_epochs=[1, 2], lr_gamma=.1)
    return {**p, "candidate_id": candidate_id(p)}


def args(tmp_path):
    return SimpleNamespace(seed=42, device="cpu", training_precision="fp32", micro_batch_size=5,
                           checkpoint_interval=1, resume=True, cache_root=tmp_path / "cache",
                           results_root=tmp_path / "results", cache_policy="temporary")


class Logger:
    def info(self, *args, **kwargs):
        pass


@pytest.mark.parametrize("optimizer", ["adam", "sgd"])
def test_exact_resume_including_batchnorm_and_schedule(tmp_path, optimizer):
    train = [row(volume(tmp_path / f"train{i}.nii.gz", hu=i*20), i) for i in range(5)]
    val = [row(volume(tmp_path / f"val{i}.nii.gz", hu=i*20), i) for i in range(5)]
    interrupted, straight = tmp_path / "interrupted", tmp_path / "straight"
    run_args, p, context = args(tmp_path), parameters(optimizer), {"fold": 1}
    with pytest.raises(RuntimeError, match="interruption"):
        CNNBackend(TinyAdapter(2)).fit_one(p, train, val, interrupted, run_args, context, Logger())
    assert torch_load(interrupted / "last_checkpoint.pt")["completed_optimizer_steps"] == 1
    CNNBackend(TinyAdapter()).fit_one(p, train, val, interrupted, run_args, context, Logger())
    CNNBackend(TinyAdapter()).fit_one(p, train, val, straight, run_args, context, Logger())
    a, b = [torch_load(r / "model.pt")["trainable_state_dict"] for r in (interrupted, straight)]
    assert any("running_mean" in k for k in a)
    assert all(torch.equal(a[k], b[k]) for k in a)
    sa, sb = [torch_load(r / "last_checkpoint.pt") for r in (interrupted, straight)]
    assert sa["scheduler"] == sb["scheduler"]
    assert all(torch.equal(sa["model_trainable_state"][k], sb["model_trainable_state"][k]) for k in a)
    validate_completion(interrupted, "completion.json")
    assert (interrupted / "preprocessing_observed.json").is_file()
    # Reload artifact plan without calling training; expanded inference keeps spacing.
    backend = CNNBackend(TinyAdapter())
    backend.ensure_feature_rows(run_args, val)
    probabilities = backend.predict(interrupted / "model.pt", val, run_args, {})
    assert probabilities.shape == (5, 5)
    np.testing.assert_allclose(probabilities.sum(1), 1, atol=1e-6)


def test_medicalnet_official_forward_checkpoint_and_shortcut_gradients():
    from MedicalNet.model import MedicalNetClassifier, shortcut_a
    from MedicalNet.vendor.resnet import resnet18, downsample_basic_block
    model = MedicalNetClassifier().eval()
    checkpoint = Path(__file__).resolve().parents[5] / "resnet_18_23dataset.pth"
    if checkpoint.exists():
        assert model.load_pretrained(checkpoint)["backbone_coverage"] == 1
    vendor = resnet18(sample_input_D=32, sample_input_H=32, sample_input_W=32,
                       num_seg_classes=2, shortcut_type="A", no_cuda=True).eval()
    vendor.load_state_dict({**vendor.state_dict(), **{k: v for k, v in model.state_dict().items() if not k.startswith("fc.")}})
    x = torch.randn(1, 1, 32, 32, 32)
    with torch.no_grad():
        expected = vendor.maxpool(vendor.relu(vendor.bn1(vendor.conv1(x))))
        expected = vendor.layer4(vendor.layer3(vendor.layer2(vendor.layer1(expected))))
        assert torch.equal(expected, model.features(x))
    z = torch.randn(2, 3, 8, 8, 8, requires_grad=True)
    fixed = shortcut_a(z, 6, 2)
    assert torch.equal(fixed, downsample_basic_block(z, 6, 2, no_cuda=True))
    fixed.sum().backward()
    assert z.grad.abs().sum() > 0


def test_densenet_is_3d():
    from DenseNet.model import build_model
    model = build_model().eval()
    assert any(isinstance(m, nn.Conv3d) for m in model.modules())
    assert not any(isinstance(m, nn.Conv2d) for m in model.modules())
    with torch.no_grad():
        assert model(torch.zeros(1, 1, 32, 32, 32)).shape == (1, 5)


def test_fivefold_refit_test_reuse_and_grid_materialization(tmp_path):
    dataset = tmp_path / "Dataset"
    folds = []
    for fold in range(1, 7):
        mapping = {}
        for label, name in enumerate(CLASS_NAMES):
            patient = f"patient_{fold}_{label}"
            volume(dataset / "NIFTI" / name / f"{patient}.nii.gz", hu=label*20)
            mapping[name] = [patient]
        if fold < 6:
            folds.append({"fold": fold, "patients_by_class": mapping})
        else:
            heldout = mapping
    atomic_write_json(dataset / "Splits" / "20_percent.json", {
        "heldout_test": {"patients_by_class": heldout}, "cross_validation": {"folds": folds}})
    p = parameters()
    p.pop("candidate_id")
    p.update(epochs=1)
    profile = tmp_path / "profile.json"
    atomic_write_json(profile, {"model": "MedicalNet", "method": "FullFineTune", "parameter_source": "grid",
                               "fixed_parameters": p, "search_space": {"learning_rate": [.001, .002]}})
    run_args = args(tmp_path)
    run_args.__dict__.update(stage="all", percentage=20, parameter_source="grid", grid_config=profile,
                             parameter_config=None, max_grid_candidates=None, dry_run=False,
                             model_source="refit_full_development", dataset_root=dataset)
    first = run_pipeline(run_args, CNNBackend(TinyAdapter()))
    assert first
    assert len(list(run_args.results_root.glob("FullFineTune/CV/CV_*/20_percent/fold_*/completion.json"))) == 5
    # Completed reruns perform no new optimization.
    run_pipeline(run_args, CNNBackend(TinyAdapter(fail_at=1)))
    run_args.stage = "test"
    for source in ("best_cv_fold", "cv_ensemble"):
        run_args.model_source = source
        run_pipeline(run_args, CNNBackend(TinyAdapter(fail_at=1)))
    assert len(list(run_args.results_root.glob("FullFineTune/Test/CV_*/*/20_percent/test_completion.json"))) == 3
