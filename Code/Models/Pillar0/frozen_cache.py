"""Temporary, lossless Atlas inputs for every existing rotation choice."""

import hashlib
import inspect
import json
import math
import shutil
import sys
from pathlib import Path

import torch
from filelock import FileLock

from .augmentations import ORIENTATIONS, augmentation_contract, orientation_key, rotate_volume
from .cache import load_pillar_volume
from .frozen_forward import atlas_layout, capture_prefixes, features_from_prefix
from .model import extract_vision_features
from _linear_probe.common import ContractError, atomic_write_json, fingerprint, read_json, sha256_file, utc_now
from _supervised_adaptation.runtime import atomic_torch_save, autocast_context, resolve_precision
from _supervised_adaptation.trainer import _runtime_identity

MANIFEST = "frozen_cache_manifest.json"
TENSOR_KEYS = ("tokens", "early")


def tensor_digest(tensor):
    tensor = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(f"{tensor.dtype}:{list(tensor.shape)}".encode())
    digest.update(memoryview(tensor.reshape(-1).view(torch.uint8).numpy()))
    return digest.hexdigest()


def weights_digest(encoder):
    return fingerprint([[name, tensor_digest(value)]
                        for name, value in encoder.model.visual.state_dict().items()])


def execution_contract(args, source_identity, encoder, scopes):
    visual, layout, cfg = atlas_layout(encoder)
    width = int(visual.atlas_models[2].blocks[0].dim)
    precision, _ = resolve_precision(str(args.device), str(args.training_precision))
    sources = {}
    for module in encoder.modules():
        try:
            path = Path(inspect.getfile(type(module))).resolve()
            if path.is_file() and str(path) not in sources:
                sources[str(path)] = sha256_file(path)
        except (TypeError, OSError):
            pass
    package = Path(__file__).parent
    return {"schema_version": 1, "representation": "atlas_final_scale_tokens_and_early_pools",
            "source": source_identity, "boundaries": sorted(scopes),
            "augmentation": augmentation_contract(), "extraction_batch_size": 1,
            "storage": "lossless_native_activation_dtype", "vision_weights_sha256": weights_digest(encoder),
            "layout": layout, "modality_config": cfg,
            "tensor_shapes": {"tokens": [math.prod(layout[-1]["window_dims"]), width],
                              "early": [width * 2]},
            "runtime": _runtime_identity(str(args.device), precision),
            "torch_math": {"matmul_precision": torch.get_float32_matmul_precision(),
                           "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                           "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                           "cudnn_benchmark": torch.backends.cudnn.benchmark,
                           "deterministic": torch.are_deterministic_algorithms_enabled()},
            "implementation": {name: sha256_file(package / name) for name in
                               ("frozen_cache.py", "frozen_forward.py", "augmentations.py")},
            "encoder_sources": sources}


def _key(row):
    return fingerprint({"patient_id": str(row["patient_id"]),
                        "cache_path": str(Path(row["cache_path"]).resolve())})


def _reject_link(path):
    if path.exists() and (path.is_symlink() or getattr(path.lstat(), "st_file_attributes", 0) & 0x400):
        raise ContractError(f"Refusing frozen-cache access through a link: {path}")


def _write_manifest(path, document):
    atomic_write_json(path, {**document, "seal_sha256": fingerprint(document)})


class FrozenStore:
    def __init__(self, args, contract):
        self.args, self.contract = args, contract
        self.digest = fingerprint(contract)
        self.base = Path(args.frozen_cache_root).resolve()
        owner = fingerprint({"results_root": str(Path(args.results_root).resolve()),
                             "percentage": int(args.percentage), "seed": int(args.seed),
                             "parameter_source": args.parameter_source})[:16]
        self.root = self.base / f"{owner}_{self.digest[:16]}"
        self.log = Path(args.results_root) / "Setup" / "frozen_cache_events.jsonl"
        self.base.mkdir(parents=True, exist_ok=True)
        self.lock = FileLock(str(self.base / f".{self.root.name}.lock"))
        self._event("waiting_for_exclusive_cache_lease")
        self.lock.acquire()
        try:
            _reject_link(self.root)
            _reject_link(self.root / MANIFEST)
            if (self.root / MANIFEST).is_file():
                self.manifest = read_json(self.root / MANIFEST)
                seal = self.manifest.pop("seal_sha256", None)
                if seal != fingerprint(self.manifest) or self.manifest.get("contract") != contract:
                    raise ContractError("Invalid frozen Atlas manifest seal or contract")
            else:
                self.manifest = {"contract": contract, "entries": {}, "status": "building"}
            self.checked, self.sources = set(), {}
            self._event("lease_acquired", cached_patients=len(self.manifest["entries"]))
        except BaseException:
            self.lock.release()
            raise

    def identity_for(self, scope):
        contract = {**self.contract, "boundaries": [scope]}
        return {"contract": contract, "contract_sha256": fingerprint(contract)}

    def _event(self, event, **details):
        self.log.parent.mkdir(parents=True, exist_ok=True)
        with self.log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"time": utc_now(), "event": event,
                                     "cache_directory": str(self.root), **details}) + "\n")
        print(f"Pillar0 frozen cache: {event}", file=sys.stderr, flush=True)

    def _source(self, row):
        path = Path(row["cache_path"]).resolve()
        stat = path.stat()
        stamp = (str(path), stat.st_size, stat.st_mtime_ns)
        if stamp not in self.sources:
            digest = sha256_file(path)
            after = path.stat()
            if (after.st_size, after.st_mtime_ns) != stamp[1:]:
                raise ContractError("RAVE input changed while hashing")
            self.sources[stamp] = {"path": str(path), "bytes": stat.st_size, "sha256": digest}
        return self.sources[stamp]

    def _path(self, key, orientation, scope):
        if orientation not in self.contract["augmentation"]["orientations"] or scope not in self.contract["boundaries"]:
            raise ContractError("Unknown frozen Atlas orientation or boundary")
        # Compact disk names leave room for atomic-save suffixes on Windows.
        # Payloads and manifests still bind the complete SHA-256 patient key.
        view = self.contract["augmentation"]["orientations"].index(orientation)
        boundary = "b" if scope == "final_block" else "s"
        path = self.root / "artifacts" / f"{key[:32]}_{view}{boundary}.pt"
        for parent in (self.root, path.parent, path):
            _reject_link(parent)
        if not path.resolve().is_relative_to(self.root.resolve()):
            raise ContractError("Frozen artifact escaped its owned directory")
        return path

    def load(self, row, orientation, scope):
        key = _key(row)
        entry = self.manifest["entries"].get(key)
        if entry is None:
            raise ContractError("Missing frozen Atlas patient")
        try:
            spec = entry["orientations"][orientation][scope]
        except KeyError as exc:
            raise ContractError("Missing frozen Atlas orientation/boundary") from exc
        path = self._path(key, orientation, scope)
        if spec["artifact"] != path.relative_to(self.root).as_posix():
            raise ContractError("Frozen artifact declaration changed")
        stat = path.stat()
        stamp = (key, orientation, scope, stat.st_size, stat.st_mtime_ns)
        if stamp not in self.checked and sha256_file(path) != spec["sha256"]:
            raise ContractError("Frozen Atlas artifact hash changed")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (payload.get("contract_sha256") != self.digest or payload.get("source") != entry["source"]
                or payload.get("patient_key") != key or payload.get("orientation") != orientation
                or payload.get("scope") != scope):
            raise ContractError("Frozen Atlas payload identity changed")
        tensors = payload.get("inputs", {})
        if set(tensors) != set(TENSOR_KEYS):
            raise ContractError("Frozen Atlas inputs are incomplete")
        for name, tensor in tensors.items():
            meta = spec["tensors"][name]
            if (not isinstance(tensor, torch.Tensor) or list(tensor.shape) != self.contract["tensor_shapes"][name]
                    or list(tensor.shape) != meta["shape"] or str(tensor.dtype) != meta["dtype"]):
                raise ContractError("Frozen Atlas tensor shape/dtype changed")
            if stamp not in self.checked and (not bool(torch.isfinite(tensor).all()) or tensor_digest(tensor) != meta["value_sha256"]):
                raise ContractError("Frozen Atlas tensor values changed")
        self.checked.add(stamp)
        return tensors

    def prepare(self, rows, encoder_factory, volume_loader=None):
        if volume_loader is None:
            volume_loader = load_pillar_volume
        pending = []
        for row in rows:
            entry = self.manifest["entries"].get(_key(row))
            source = self._source(row)
            if entry is None:
                pending.append(row)
                continue
            if source != entry["source"]:
                raise ContractError("Frozen Atlas RAVE input changed")
            if set(entry["orientations"]) != set(self.contract["augmentation"]["orientations"]):
                raise ContractError("Incomplete frozen Atlas augmentation inventory")
            for orientation, scopes in entry["orientations"].items():
                if set(scopes) != set(self.contract["boundaries"]):
                    raise ContractError("Incomplete frozen Atlas boundaries")
                for scope in scopes:
                    path = self._path(_key(row), orientation, scope)
                    stat = path.stat()
                    if (_key(row), orientation, scope, stat.st_size, stat.st_mtime_ns) not in self.checked:
                        self.load(row, orientation, scope)
        if not pending:
            self._event("reused", requested_patients=len(rows))
            return
        # All ten views are materialized before training. Bound storage using
        # FP32 even when autocast yields a smaller native dtype.
        per_view = sum(math.prod(shape) * 4 for shape in self.contract["tensor_shapes"].values()) + 8192
        required = int(len(pending) * len(ORIENTATIONS) * len(self.contract["boundaries"]) * per_view * 1.08) + 2 * 1024**3
        if shutil.disk_usage(self.base).free < required:
            raise ContractError(f"Frozen Atlas cache needs approximately {required / 1024**3:.2f} GiB free; select a larger -FrozenCacheRoot")
        self._event("preparing", missing_patients=len(pending), orientations_per_patient=10,
                    required_free_gib=required / 1024**3)
        encoder = encoder_factory()
        encoder.eval().requires_grad_(False)
        if weights_digest(encoder) != self.contract["vision_weights_sha256"]:
            raise ContractError("Atlas extraction weights changed")
        _, dtype = resolve_precision(str(self.args.device), str(self.args.training_precision))
        try:
            for index, row in enumerate(pending, 1):
                key, source = _key(row), self._source(row)
                volume = volume_loader(row["cache_path"])
                orientations = {}
                verify = not self.manifest.get("forward_verification")
                verification = {}
                for plane, turns in ORIENTATIONS:
                    orientation = orientation_key(plane, turns)
                    with torch.no_grad(), autocast_context(str(self.args.device), dtype):
                        raw = rotate_volume(volume, plane, turns).unsqueeze(0).to(str(self.args.device))
                        captured = capture_prefixes(encoder, raw, self.contract["boundaries"])
                        cpu = {scope: {name: value[0].detach().cpu().contiguous() for name, value in inputs.items()}
                               for scope, inputs in captured.items()}
                        if verify:
                            reference = extract_vision_features(encoder, raw, normalize=False)
                            verification[orientation] = {}
                            for scope, inputs in cpu.items():
                                stacked = {name: torch.stack([value]).to(str(self.args.device)) for name, value in inputs.items()}
                                observed = features_from_prefix(encoder, stacked, scope)
                                error = float((reference.float() - observed.float()).abs().max())
                                if not torch.allclose(reference.float(), observed.float(), rtol=1e-5, atol=1e-6):
                                    self._event("forward_verification_failed", orientation=orientation, scope=scope,
                                                maximum_absolute_error=error, rtol=1e-5, atol=1e-6)
                                    raise ContractError(f"Frozen Atlas verification failed: {orientation}/{scope}; maximum error={error:.9g}")
                                verification[orientation][scope] = error
                                del stacked, observed
                            del reference
                    specs = {}
                    for scope, inputs in cpu.items():
                        if any(not bool(torch.isfinite(value).all()) or list(value.shape) != self.contract["tensor_shapes"][name]
                               for name, value in inputs.items()):
                            raise ContractError("Atlas extraction produced invalid tensor values or shapes")
                        path = self._path(key, orientation, scope)
                        atomic_torch_save({"contract_sha256": self.digest, "patient_key": key,
                                           "source": source, "orientation": orientation, "scope": scope,
                                           "inputs": inputs}, path)
                        specs[scope] = {"artifact": path.relative_to(self.root).as_posix(), "sha256": sha256_file(path),
                                        "tensors": {name: {"shape": list(value.shape), "dtype": str(value.dtype),
                                                           "value_sha256": tensor_digest(value)} for name, value in inputs.items()}}
                    orientations[orientation] = specs
                    del raw, captured, cpu
                if self._source(row) != source:
                    raise ContractError("RAVE input changed during Atlas extraction")
                self.manifest["entries"][key] = {"source": source, "orientations": orientations}
                if verify:
                    self.manifest["forward_verification"] = verification
                    self._event("all_orientations_forward_verified", maximum_absolute_errors=verification)
                self.manifest["updated_at"] = utc_now()
                # Commit a complete patient. An interrupted partial patient's
                # predictable files are safely overwritten on the next prepare.
                _write_manifest(self.root / MANIFEST, self.manifest)
                for orientation in orientations:
                    for scope in self.contract["boundaries"]:
                        self.load(row, orientation, scope)
                self._event("progress", completed_patients=index, requested_patients=len(pending))
                del volume
            self.manifest["status"] = "ready"
            _write_manifest(self.root / MANIFEST, self.manifest)
            self._event("ready", cached_patients=len(self.manifest["entries"]))
        finally:
            del encoder

    def rows_digest(self, rows, scope):
        records = []
        for row in rows:
            orientations = self.manifest["entries"][_key(row)]["orientations"]
            values = [[orientation, orientations[orientation][scope]["tensors"]]
                      for orientation in self.contract["augmentation"]["orientations"]]
            records.append([_key(row), values])
        return fingerprint(records)

    def finish(self, success):
        try:
            if success:
                self._cleanup()
                self._event("removed_after_success")
            else:
                self._event("retained_for_resume")
        finally:
            self.lock.release()

    def _cleanup(self):
        _reject_link(self.root)
        if self.root.resolve().parent != self.base or self.root.resolve() != self.root:
            raise ContractError("Refusing Atlas cleanup outside its owned directory")
        if not self.root.exists():
            return
        expected = {self.root / MANIFEST}
        for key, entry in self.manifest["entries"].items():
            for orientation, scopes in entry["orientations"].items():
                for scope in scopes:
                    expected.add(self._path(key, orientation, scope))
        observed = set(self.root.iterdir())
        _reject_link(self.root / "artifacts")
        if (self.root / "artifacts").is_dir():
            observed.update((self.root / "artifacts").iterdir())
        for path in observed:
            _reject_link(path)
        if ({p for p in observed if p.is_file()} != expected
                or {p for p in observed if p.is_dir()} - {self.root / "artifacts"}):
            raise ContractError("Unexpected Atlas cache files; retaining them for inspection")
        for path in sorted(expected):
            path.unlink()
        if (self.root / "artifacts").exists():
            (self.root / "artifacts").rmdir()
        self.root.rmdir()
