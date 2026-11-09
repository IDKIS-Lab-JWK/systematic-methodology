"""Disposable, resumable Primus prefix tensors, locked for the whole launch."""

import inspect
import hashlib
import json
import shutil
import sys
from pathlib import Path

import torch
from filelock import FileLock

from .cache import _atomic_torch_save, _entry_key, _safe_artifact_path
from .frozen_forward import capture_prefixes, pooled_from_prefix
from .volume_cache import _read_sealed, _write_sealed
from _linear_probe.common import ContractError, fingerprint, sha256_file, utc_now
from _supervised_adaptation.runtime import autocast_context, resolve_precision
from _supervised_adaptation.trainer import _runtime_identity


MANIFEST = "frozen_cache_manifest.json"
GRID_SHAPE = (24, 24, 24)


def tensor_digest(tensor):
    """Stable value digest independent of torch.save container names/timestamps."""
    tensor = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(f"{tensor.dtype}:{list(tensor.shape)}".encode())
    digest.update(memoryview(tensor.view(torch.uint8).numpy()))
    return digest.hexdigest()


def execution_contract(args, source_identity, encoder, scopes):
    precision, _dtype = resolve_precision(str(args.device), str(args.training_precision))
    package = Path(__file__).resolve().parent
    # Bind the actual installed prefix implementation, including attention,
    # rotary embeddings, projection and normalization dependencies.
    sources = {}
    for module in encoder.image_encoder.modules():
        try:
            path = Path(inspect.getfile(type(module))).resolve()
            if str(path) not in sources and path.is_file():
                sources[str(path)] = sha256_file(path)
        except (TypeError, OSError):
            pass
    return {
        "schema_version": 1, "representation": "primus_pre_block_tokens",
        "boundaries": sorted(scopes), "source": source_identity,
        "extraction_batch_size": 1, "storage": "lossless_native_activation_dtype",
        "runtime": _runtime_identity(str(args.device), precision),
        "torch_math": {
            "matmul_precision": torch.get_float32_matmul_precision(),
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "deterministic": torch.are_deterministic_algorithms_enabled(),
        },
        "implementation": {name: sha256_file(package / name) for name in
                           ("frozen_cache.py", "frozen_forward.py")},
        "encoder_sources": sources,
    }


class FrozenStore:
    def __init__(self, args, contract):
        self.args, self.contract = args, contract
        self.digest = fingerprint(contract)
        self.base = (Path(args.cache_root) / "FrozenRuns").resolve()
        # Scope-independent result namespace allows an explicit 'both' launch
        # to use one traversal; distinct result roots never own one another's cache.
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
            self.manifest = _read_sealed(self.root / MANIFEST) if (self.root / MANIFEST).is_file() else {
                "contract": contract, "entries": {}, "status": "building",
            }
            if self.manifest.get("contract") != contract:
                raise ContractError("Frozen cache contract changed; refusing reuse")
            self.checked = set()
            self._event("lease_acquired", cached_patients=len(self.manifest["entries"]))
        except BaseException:
            self.lock.release()
            raise

    @property
    def identity(self):
        # Durable experiment identity intentionally excludes disposable file
        # timestamps/paths. Input rows still bind their permanent CT artifacts.
        return {"contract_sha256": self.digest, "contract": self.contract}

    def identity_for(self, scope):
        # A model's identity depends on its boundary, not on whether the same
        # traversal also saved the other scope. Later single-scope evaluation
        # can regenerate its inputs after a combined launch was cleaned up.
        contract = {**self.contract, "boundaries": [scope]}
        return {"contract_sha256": fingerprint(contract), "contract": contract}

    def _event(self, event, **details):
        self.log.parent.mkdir(parents=True, exist_ok=True)
        with self.log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"time": utc_now(), "event": event,
                                     "cache_directory": str(self.root), **details}) + "\n")
        print(f"COLIPRI frozen cache: {event}", file=sys.stderr, flush=True)

    def _load(self, row, scope=None):
        key = _entry_key(row)
        entry = self.manifest["entries"].get(key)
        if entry is None or set(entry.get("tensors", {})) != set(self.contract["boundaries"]):
            raise ContractError("Missing or invalid frozen cache entry")
        tokens = {}
        for boundary in ([scope] if scope else self.contract["boundaries"]):
            specification = entry["tensors"][boundary]
            if specification.get("artifact") != f"artifacts/{key}_{boundary}.pt":
                raise ContractError("Invalid frozen cache artifact declaration")
            path = _safe_artifact_path(self.root, specification["artifact"])
            stat = path.stat()
            stamp = (key, boundary, stat.st_size, stat.st_mtime_ns)
            if stamp not in self.checked and sha256_file(path) != specification["sha256"]:
                raise ContractError("Frozen cache artifact hash changed; refusing checkpoint resume")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if (payload.get("contract_sha256") != self.digest
                    or payload.get("patient_id") != str(row["patient_id"])
                    or payload.get("ct_sha256") != entry["ct_sha256"] or payload.get("boundary") != boundary):
                raise ContractError("Frozen cache payload identity changed")
            tensor = payload.get("tokens")
            if (not isinstance(tensor, torch.Tensor) or list(tensor.shape) != specification["shape"]
                    or str(tensor.dtype) != specification["dtype"]):
                raise ContractError("Frozen cache shape/dtype changed")
            if stamp not in self.checked and not bool(torch.isfinite(tensor).all()):
                raise ContractError("Non-finite frozen cache tensor")
            if stamp not in self.checked and tensor_digest(tensor) != specification.get("tensor_sha256"):
                raise ContractError("Frozen cache tensor value digest changed")
            self.checked.add(stamp)
            tokens[boundary] = tensor
        return tokens

    def prepare(self, rows, volume_store, encoder_factory):
        entries = self.manifest["entries"]
        pending = []
        for row in rows:
            key = _entry_key(row)
            ct_sha = volume_store.manifest["entries"][key]["sha256"]
            if key in entries:
                if entries[key]["ct_sha256"] != ct_sha:
                    raise ContractError("Frozen cache CT input changed")
                # Already verified within this launch: only inspect stats.
                # A fresh/resumed process verifies payload hashes once again.
                for scope, item in entries[key]["tensors"].items():
                    path = _safe_artifact_path(self.root, item["artifact"])
                    stat = path.stat()
                    if (key, scope, stat.st_size, stat.st_mtime_ns) not in self.checked:
                        self._load(row, scope)
            else:
                pending.append(row)
        if not pending:
            self._event("reused", requested_patients=len(rows))
            return
        # Conservative FP32 upper estimate: 13,824 x 864 tokens per boundary.
        required = int(len(pending) * 13824 * 864 * 4 * len(self.contract["boundaries"]) * 1.08) + 2 * 1024**3
        if shutil.disk_usage(self.base).free < required:
            raise ContractError(f"Temporary frozen cache needs approximately {required / 1024**3:.1f} GiB free")
        self._event("preparing", missing_patients=len(pending), required_free_gib=required / 1024**3)
        encoder = encoder_factory()
        encoder.eval().requires_grad_(False)
        _, dtype = resolve_precision(str(self.args.device), str(self.args.training_precision))
        try:
            for index, row in enumerate(pending, 1):
                key = _entry_key(row)
                ct_sha = volume_store.manifest["entries"][key]["sha256"]
                with torch.no_grad(), autocast_context(str(self.args.device), dtype):
                    images = volume_store.load(row).unsqueeze(0).to(str(self.args.device))
                    tensors, grid = capture_prefixes(encoder, images, self.contract["boundaries"])
                    if not self.manifest.get("forward_verification"):
                        reference = encoder.encode_image(images, pool=True, project=True, normalize=False)
                        errors = {}
                        for scope, tensor in tensors.items():
                            # Verify the contiguous CPU shard / stacked training
                            # layout, not only the resident extraction tensor.
                            loaded = torch.stack([tensor[0].detach().cpu().contiguous()]).to(images.device)
                            observed = pooled_from_prefix(encoder, loaded, scope, grid)
                            errors[scope] = float((reference.float() - observed.float()).abs().max())
                            # Same patient, batch and precision: splitting must
                            # reproduce the released pooled forward closely.
                            if not torch.allclose(reference.float(), observed.float(), rtol=1e-5, atol=1e-6):
                                self._event("forward_verification_failed", scope=scope,
                                            maximum_absolute_error=errors[scope],
                                            output_dtype=str(reference.dtype), rtol=1e-5, atol=1e-6)
                                raise ContractError(
                                    f"Frozen prefix forward verification failed: {scope}; "
                                    f"maximum absolute error={errors[scope]:.9g}, "
                                    f"output dtype={reference.dtype}, rtol=1e-5, atol=1e-6. "
                                    "See Setup/frozen_cache_events.jsonl for diagnostics."
                                )
                            del loaded
                        self.manifest["forward_verification"] = errors
                        self._event("forward_verified", maximum_absolute_errors=errors)
                tokens = {scope: tensor[0].detach().cpu().contiguous() for scope, tensor in tensors.items()}
                if any(not bool(torch.isfinite(value).all()) for value in tokens.values()):
                    raise ContractError("Frozen extraction produced non-finite tokens")
                if grid != GRID_SHAPE:
                    raise ContractError(f"Unexpected frozen cache patch grid: {grid}")
                specifications = {}
                for scope, tensor in tokens.items():
                    artifact = f"artifacts/{key}_{scope}.pt"
                    path = _safe_artifact_path(self.root, artifact)
                    _atomic_torch_save({"contract_sha256": self.digest, "patient_id": str(row["patient_id"]),
                                        "ct_sha256": ct_sha, "boundary": scope, "tokens": tensor}, path)
                    specifications[scope] = {"artifact": artifact, "sha256": sha256_file(path),
                                             "tensor_sha256": tensor_digest(tensor),
                                             "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
                entries[key] = {"ct_sha256": ct_sha, "tensors": specifications}
                self.manifest["updated_at"] = utc_now()
                _write_sealed(self.root / MANIFEST, self.manifest)
                self._load(row)
                if index % 25 == 0 or index == len(pending):
                    self._event("progress", completed_patients=index, requested_patients=len(pending))
                del images, tensors, tokens
            self.manifest["status"] = "ready"
            _write_sealed(self.root / MANIFEST, self.manifest)
            self._event("ready", cached_patients=len(entries))
        finally:
            del encoder

    def load(self, row, scope):
        return self._load(row, scope)[scope]

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
        # No recursive deletion. Validate the absolute owned directory, reject
        # links/junctions and unexpected contents, then unlink only our files.
        root = self.root.resolve()
        if root.parent != self.base or root != self.root or self.root.is_symlink():
            raise ContractError("Refusing frozen cache cleanup outside its owned directory")
        if not root.exists():
            return
        expected = {root / MANIFEST}
        expected.update(root / item["artifact"] for entry in self.manifest["entries"].values()
                        for item in entry["tensors"].values())
        observed = set(root.iterdir())
        for path in observed:
            if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()) or not path.resolve().is_relative_to(root):
                raise ContractError("Refusing frozen cache cleanup through a link")
        if (root / "artifacts").is_dir():
            observed.update((root / "artifacts").iterdir())
        for path in observed:
            if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()) or not path.resolve().is_relative_to(root):
                raise ContractError("Refusing frozen cache cleanup through a link")
        files = {path for path in observed if path.is_file()}
        directories = {path for path in observed if path.is_dir()}
        if files != expected or directories - {root / "artifacts"}:
            raise ContractError("Unexpected frozen cache files; retaining cache for inspection")
        for path in sorted(files):
            path.unlink()
        if (root / "artifacts").exists():
            (root / "artifacts").rmdir()
        root.rmdir()
