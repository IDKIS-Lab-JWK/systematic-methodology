"""Training and evaluation engines for the isolated 3DINO experiments.

The publication-backed path keeps the encoder frozen and trains all requested
linear heads from the same stochastic volume batch and the same encoder pass.
The end-to-end path is deliberately labelled project-adapted and requires an
explicit authorization switch.  Upstream loading stays lazy and injectable so
unit tests never need the gated checkpoint or upstream repository.
"""

from __future__ import annotations

import contextlib
import csv
import functools
import hashlib
import inspect
import importlib.metadata
import json
import logging
import math
import os
import random
import shutil
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler

try:
    from .data import (
        AUGMENTATION_SEED_KEY,
        DRAW_INDEX_KEY,
        DRAW_SEED_PROTOCOL,
        make_draw_request,
        preprocessing_contract,
    )
    from .metrics import compute_metrics, fold_selection_score, format_predictions
    from .search import build_probe_candidates, candidate_id, parameter_fingerprint
except ImportError:  # pragma: no cover - direct script/test imports
    from data import (
        AUGMENTATION_SEED_KEY,
        DRAW_INDEX_KEY,
        DRAW_SEED_PROTOCOL,
        make_draw_request,
        preprocessing_contract,
    )
    from metrics import compute_metrics, fold_selection_score, format_predictions
    from search import build_probe_candidates, candidate_id, parameter_fingerprint


SCHEMA_VERSION = 2
MODEL_FAMILY = "3DINO"
DEFAULT_CLASSES = [
    "Appendicitis",
    "Cholezystitis",
    "Normal",
    "Perforation",
    "Sigmoid",
]

DATA_SEQUENCE_SCHEMA = "threedino-draw-sequence-v2"
AUGMENTATION_SEED_SCHEMA = "threedino-uint32-feistel-v1"
PROBE_HEAD_INITIALIZATION_SCHEMA = "threedino-candidate-sha256-seed-v1"
TRAINING_DATA_STATE_SCHEMA_VERSION = 2
_MAX_DRAW_COUNT = 1 << 32


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
    os.replace(temporary, path)


def save_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fieldnames} for row in rows)
    os.replace(temporary, path)


def atomic_torch_save(value: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


class _ReleaseAfterEmitFileHandler(logging.FileHandler):
    """Append a record, then release the file so Windows never pins run folders."""

    def __init__(self, filename, encoding: str = "utf-8"):
        super().__init__(filename, mode="a", encoding=encoding, delay=True)

    def emit(self, record) -> None:
        try:
            super().emit(record)
        finally:
            self.acquire()
            try:
                if self.stream is not None:
                    self.flush()
                    self.stream.close()
                    self.stream = None
            finally:
                self.release()


def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"threedino_{abs(hash(str(output_dir.resolve())))}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    log_path = output_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    for handler in (
        _ReleaseAfterEmitFileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def close_logger(logger: logging.Logger) -> None:
    """Release Windows file handles once a run has written its final artifact."""

    for handler in list(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
        state["cuda_device_count"] = torch.cuda.device_count()
    return state


def _restore_rng_state(state: Mapping | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        if int(state.get("cuda_device_count", -1)) != torch.cuda.device_count():
            raise RuntimeError(
                "Visible CUDA device count changed; exact RNG resume is impossible."
            )
        torch.cuda.set_rng_state_all(state["cuda"])


def _device_identity(device: torch.device) -> dict:
    identity = {"device": str(device), "device_type": device.type}
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        identity.update(
            {
                "cuda_index": int(index),
                "cuda_name": torch.cuda.get_device_name(index),
                "cuda_capability": list(torch.cuda.get_device_capability(index)),
            }
        )
    return identity


@functools.lru_cache(maxsize=16)
def _sha256_file_cached(path_string: str, size: int, modified_ns: int) -> str:
    del size, modified_ns  # included in the cache key to invalidate changed files
    digest = hashlib.sha256()
    with Path(path_string).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    path = Path(path)
    stat = path.stat()
    return _sha256_file_cached(
        str(path.resolve()), int(stat.st_size), int(stat.st_mtime_ns)
    )


def _source_provenance(
    upstream_root: str | Path | None,
    upstream_revision: str | None,
    checkpoint_path: str | Path | None,
    expected_checkpoint_sha256: str | None,
    *,
    injected: bool,
) -> dict:
    if injected and checkpoint_path is None:
        return {
            "loader": "injected_factory",
            "upstream_root": str(upstream_root) if upstream_root else None,
            "upstream_revision": upstream_revision,
            "checkpoint_path": None,
            "checkpoint_sha256": None,
        }
    if upstream_root is None or not str(upstream_revision or "").strip():
        raise ValueError(
            "Default 3DINO loading requires upstream_root and a pinned upstream_revision."
        )
    root = Path(upstream_root)
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    if not root.is_dir():
        raise FileNotFoundError(f"3DINO upstream root not found: {root}")
    if checkpoint is None or not checkpoint.is_file():
        raise FileNotFoundError(f"3DINO checkpoint not found: {checkpoint}")
    actual_hash = _sha256_file(checkpoint)
    if expected_checkpoint_sha256 and actual_hash.lower() != str(
        expected_checkpoint_sha256
    ).lower():
        raise RuntimeError(
            "3DINO checkpoint SHA-256 mismatch; refusing to train with unknown weights."
        )
    return {
        "loader": "official_pinned_3dino_assets",
        "upstream_root": str(root.resolve()),
        "upstream_revision": str(upstream_revision),
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": actual_hash,
    }


def _log_asset_provisioning(
    logger: logging.Logger, asset_provisioning: Mapping | None
) -> None:
    """Mirror token-safe acquisition status into each training run log."""

    if not isinstance(asset_provisioning, Mapping):
        return
    upstream = asset_provisioning.get("upstream")
    checkpoint = asset_provisioning.get("checkpoint")
    upstream = upstream if isinstance(upstream, Mapping) else {}
    checkpoint = checkpoint if isinstance(checkpoint, Mapping) else {}
    logger.info(
        "3DINO assets | source_action=%s source_revision=%s "
        "checkpoint_action=%s model_revision=%s checkpoint_sha256=%s "
        "downloads_performed=%s git_operations_performed=%s",
        upstream.get("acquisition", upstream.get("action", "unknown")),
        upstream.get("revision", "unknown"),
        checkpoint.get("acquisition", checkpoint.get("action", "unknown")),
        checkpoint.get("revision", "unknown"),
        checkpoint.get("sha256", "unknown"),
        bool(asset_provisioning.get("downloads_performed")),
        bool(asset_provisioning.get("git_operations_performed")),
    )


def _call_supported(function: Callable, /, *args, **kwargs):
    """Call an injectable factory without forcing it to accept every option."""

    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return function(*args, **kwargs)
    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if not accepts_var_kwargs:
        kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return function(*args, **kwargs)


def _default_encoder_factory(**kwargs):
    try:
        from .model import load_official_model
    except ImportError:  # pragma: no cover
        from model import load_official_model
    return load_official_model(**kwargs)


def _default_feature_bank_extractor(model, images, representations):
    try:
        from .model import extract_representation_bank
    except ImportError:  # pragma: no cover
        from model import extract_representation_bank
    return extract_representation_bank(model, images, representations=representations)


def _default_feature_dimension(n_last_blocks: int, use_mean_patch_token: bool) -> int:
    try:
        from .model import feature_dimension
    except ImportError:  # pragma: no cover
        from model import feature_dimension
    return int(feature_dimension(n_last_blocks, use_mean_patch_token))


def _default_model_factory(
    *,
    num_classes: int,
    parameters: dict,
    device: torch.device,
    encoder_factory: Callable | None,
    upstream_root,
    upstream_revision,
    checkpoint_path,
    expected_checkpoint_sha256,
):
    factory = encoder_factory or _default_encoder_factory
    encoder = _call_supported(
        factory,
        upstream_root=upstream_root,
        upstream_revision=upstream_revision,
        checkpoint_path=checkpoint_path,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        device=device,
    )
    try:
        from .model import ThreeDINOClassifier
    except ImportError:  # pragma: no cover
        from model import ThreeDINOClassifier
    model = ThreeDINOClassifier(
        encoder,
        num_classes=num_classes,
        n_last_blocks=int(parameters.get("n_last_blocks", 1)),
        use_mean_patch_token=bool(parameters.get("use_mean_patch_token", False)),
        freeze_encoder=False,
        dropout=float(parameters.get("dropout", 0.0)),
    )
    return model.to(device)


def _canonical_sha256(value: Mapping) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _training_transform_contract_fingerprint(dataset: Any) -> str:
    """Bind draw seeding to the stable transform and injected-dataset contract."""

    dataset_type = f"{type(dataset).__module__}.{type(dataset).__qualname__}"
    topology = [
        {
            "path": path,
            "type": f"{type(node).__module__}.{type(node).__qualname__}",
        }
        for path, node in sorted(_random_state_nodes(dataset).items())
    ]
    try:
        monai_version = importlib.metadata.version("monai")
    except importlib.metadata.PackageNotFoundError:
        monai_version = None
    return _canonical_sha256(
        {
            "sequence_schema": DATA_SEQUENCE_SCHEMA,
            "augmentation_seed_schema": AUGMENTATION_SEED_SCHEMA,
            "preprocessing_contract": preprocessing_contract(),
            "dataset_type": dataset_type,
            "randomizable_topology": topology,
            "torch_version": str(torch.__version__),
            "numpy_version": str(np.__version__),
            "monai_version": monai_version,
        }
    )


def _seed_dataset_for_draw(dataset: Any, seed: int) -> None:
    """Reset dataset-local Randomizables for one draw.

    The default post-cache dataset implements ``set_random_state``.  The
    fallback supports small injected datasets used by non-production tests and
    fails closed if a discovered Randomizable cannot be reset.
    """

    setter = getattr(dataset, "set_random_state", None)
    if callable(setter):
        setter(seed=int(seed))
        return
    nodes = _random_state_nodes(dataset)
    for offset, (path, node) in enumerate(sorted(nodes.items())):
        node_seed = (int(seed) + 0x9E3779B9 * offset) % _MAX_DRAW_COUNT
        node_setter = getattr(node, "set_random_state", None)
        if callable(node_setter):
            node_setter(seed=node_seed)
            continue
        random_state = getattr(node, "R", None)
        random_seed = getattr(random_state, "seed", None)
        if callable(random_seed):
            random_seed(node_seed)
            continue
        raise RuntimeError(
            f"Dataset Randomizable at {path} cannot be reseeded per draw."
        )


class DrawSeededDataset:
    """Picklable adapter that executes one deterministic structured draw."""

    def __init__(self, dataset: Any):
        self.dataset = dataset
        protocol = getattr(dataset, "_threedino_draw_seed_protocol", None)
        if protocol is None and isinstance(dataset, (list, tuple)):
            # A materialized built-in sequence has no executable sample-time
            # transform or private RNG; copying a mapping is deterministic.
            protocol = DRAW_SEED_PROTOCOL
        if protocol != DRAW_SEED_PROTOCOL:
            raise RuntimeError(
                "Injected training datasets must explicitly implement the "
                f"versioned {DRAW_SEED_PROTOCOL!r} contract (or return a "
                "plain materialized sequence). Supply a dataset whose "
                "set_random_state/draw-request implementation controls all "
                "sample-time randomness."
            )
        self._threedino_draw_seed_protocol = protocol

    def __len__(self) -> int:
        return len(self.dataset)

    def set_random_state(self, seed: int | None = None, state: Any = None):
        setter = getattr(self.dataset, "set_random_state", None)
        if callable(setter):
            _call_supported(setter, seed=seed, state=state)
        return self

    def __getitem__(self, request: tuple[int, int, int]) -> dict:
        if not isinstance(request, tuple) or len(request) != 3:
            raise TypeError(
                "Training datasets require a structured 3DINO draw request."
            )
        sample_index, draw_index, augmentation_seed = make_draw_request(*request)

        # Preserve caller RNG when num_workers=0: stochastic data must not
        # perturb model/dropout RNG in the main training process.
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        try:
            random.seed(augmentation_seed)
            np.random.seed(augmentation_seed)
            # Seed only the CPU default generator used inside DataLoader
            # samples. ``torch.manual_seed`` also mutates CUDA generators,
            # which would couple num_workers=0 augmentation to model RNG.
            torch.random.default_generator.manual_seed(augmentation_seed)
            accepts_request = bool(
                getattr(self.dataset, "_threedino_accepts_draw_request", False)
            )
            if accepts_request:
                value = self.dataset[request]
            else:
                _seed_dataset_for_draw(self.dataset, augmentation_seed)
                value = self.dataset[sample_index]
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)

        if not isinstance(value, Mapping):
            raise TypeError("3DINO training datasets must return mapping samples.")
        result = dict(value)
        image = result.get("image")
        if isinstance(image, torch.Tensor) and image.device.type != "cpu":
            raise RuntimeError(
                "3DINO DataLoader samples must remain on CPU until the main "
                "training process transfers a committed batch to the device."
            )
        result[DRAW_INDEX_KEY] = int(draw_index)
        result[AUGMENTATION_SEED_KEY] = int(augmentation_seed)
        return result


class DrawIndexedSampler(Sampler[tuple[int, int, int]]):
    """Deterministic sampler whose persisted cursor advances only on commit.

    DataLoader prefetch advances ``issued_draw_count`` for diagnostics, but that
    value is deliberately excluded from checkpoints.  Restarting regenerates
    every uncommitted request from ``next_committed_draw``.
    """

    def __init__(
        self,
        data_source: Sequence,
        *,
        seed: int,
        infinite: bool = False,
        batch_size: int = 1,
        drop_last: bool = False,
        training_rows_fingerprint: str,
        transform_contract_fingerprint: str,
    ):
        self.data_source = data_source
        self.seed = int(seed)
        self.infinite = bool(infinite)
        self.batch_size = max(1, int(batch_size))
        self.drop_last = bool(drop_last)
        self.training_rows_fingerprint = str(training_rows_fingerprint)
        self.transform_contract_fingerprint = str(transform_contract_fingerprint)
        self.next_committed_draw = 0
        self.issued_draw_count = 0
        key_payload = {
            "schema": AUGMENTATION_SEED_SCHEMA,
            "seed": self.seed,
            "training_rows_fingerprint": self.training_rows_fingerprint,
            "transform_contract_fingerprint": self.transform_contract_fingerprint,
        }
        self._augmentation_key = bytes.fromhex(_canonical_sha256(key_payload))

    @property
    def sampler_kind(self) -> str:
        return "sharded_infinite" if self.infinite else "finite_shuffle"

    @property
    def usable_epoch_samples(self) -> int:
        size = len(self.data_source)
        if self.infinite or not self.drop_last:
            return size
        return (size // self.batch_size) * self.batch_size

    def __len__(self) -> int:
        return len(self.data_source) if self.infinite else self.usable_epoch_samples

    def _epoch_permutation(self, epoch: int) -> torch.Tensor:
        payload = {
            "schema": DATA_SEQUENCE_SCHEMA,
            "seed": self.seed,
            "training_rows_fingerprint": self.training_rows_fingerprint,
            "sampler": self.sampler_kind,
            "epoch": int(epoch),
        }
        epoch_seed = int.from_bytes(
            bytes.fromhex(_canonical_sha256(payload))[:8], "big"
        ) & ((1 << 63) - 1)
        generator = torch.Generator().manual_seed(epoch_seed)
        return torch.randperm(len(self.data_source), generator=generator)

    def _augmentation_seed(self, draw_index: int) -> int:
        if not 0 <= int(draw_index) < _MAX_DRAW_COUNT:
            raise RuntimeError(
                "3DINO exact-resume draw horizon (2^32 samples) was exceeded."
            )
        # Four-round keyed Feistel permutation: collision-free over uint32 while
        # still thoroughly mixing adjacent absolute draw counters.
        left = (int(draw_index) >> 16) & 0xFFFF
        right = int(draw_index) & 0xFFFF
        for round_index in range(4):
            digest = hashlib.sha256(
                self._augmentation_key
                + bytes([round_index])
                + right.to_bytes(2, "big")
            ).digest()
            left, right = right, left ^ int.from_bytes(digest[:2], "big")
        return ((left & 0xFFFF) << 16) | (right & 0xFFFF)

    def __iter__(self):
        epoch_length = self.usable_epoch_samples
        if epoch_length <= 0:
            return
        draw_index = int(self.next_committed_draw)
        while True:
            epoch = draw_index // epoch_length
            offset = draw_index % epoch_length
            order = self._epoch_permutation(epoch)
            while offset < epoch_length:
                sample_index = int(order[offset])
                request = make_draw_request(
                    sample_index,
                    draw_index,
                    self._augmentation_seed(draw_index),
                )
                self.issued_draw_count = max(
                    int(self.issued_draw_count), draw_index + 1
                )
                yield request
                draw_index += 1
                offset += 1
            if not self.infinite:
                return

    @property
    def mid_epoch(self) -> bool:
        if self.infinite:
            return self.next_committed_draw > 0
        epoch_length = self.usable_epoch_samples
        return bool(
            epoch_length > 0
            and self.next_committed_draw > 0
            and self.next_committed_draw % epoch_length != 0
        )

    def commit(self, draw_indices: Sequence[int]) -> None:
        values = [int(value) for value in draw_indices]
        if not values:
            raise RuntimeError("Cannot commit an empty 3DINO training draw batch.")
        expected = list(
            range(self.next_committed_draw, self.next_committed_draw + len(values))
        )
        if values != expected:
            raise RuntimeError(
                "Training draws are missing, duplicated, or out of order; "
                f"expected {expected[0]}..{expected[-1]}, got {values}."
            )
        if values[-1] >= self.issued_draw_count:
            raise RuntimeError("Cannot commit a draw that the DataLoader did not issue.")
        self.next_committed_draw += len(values)

    def state_dict(self) -> dict:
        epoch_samples = self.usable_epoch_samples
        next_epoch_index = self.next_committed_draw // epoch_samples
        next_offset = self.next_committed_draw % epoch_samples
        last_committed_epoch = (
            0
            if self.next_committed_draw == 0
            else int(math.ceil(self.next_committed_draw / epoch_samples))
        )
        return {
            "schema_version": TRAINING_DATA_STATE_SCHEMA_VERSION,
            "sequence_schema": DATA_SEQUENCE_SCHEMA,
            "augmentation_seed_schema": AUGMENTATION_SEED_SCHEMA,
            "dataset_size": len(self.data_source),
            "sampler": self.sampler_kind,
            "seed": self.seed,
            "batch_size": self.batch_size,
            "drop_last": self.drop_last,
            "training_rows_fingerprint": self.training_rows_fingerprint,
            "transform_contract_fingerprint": self.transform_contract_fingerprint,
            "next_committed_draw": int(self.next_committed_draw),
            "usable_epoch_samples": int(epoch_samples),
            "next_data_epoch_index": int(next_epoch_index),
            "next_sample_offset_in_epoch": int(next_offset),
            "last_committed_data_epoch": int(last_committed_epoch),
        }

    def load_state_dict(self, state: Mapping) -> None:
        if int(state.get("schema_version", -1)) != TRAINING_DATA_STATE_SCHEMA_VERSION:
            raise RuntimeError(
                "Legacy schema-v1 DataLoader state cannot be migrated to the "
                "draw-indexed exact-resume-v2 stream; start a separate new run."
            )
        expected = self.state_dict()
        for key in (
            "sequence_schema",
            "augmentation_seed_schema",
            "dataset_size",
            "sampler",
            "seed",
            "batch_size",
            "drop_last",
            "training_rows_fingerprint",
            "transform_contract_fingerprint",
            "usable_epoch_samples",
        ):
            if state.get(key) != expected[key]:
                raise RuntimeError(
                    f"Training draw contract changed at {key}; refusing exact resume."
                )
        position = int(state.get("next_committed_draw", -1))
        if position < 0 or position >= _MAX_DRAW_COUNT:
            raise RuntimeError("Saved committed draw position is invalid.")
        if not self.infinite and self.usable_epoch_samples <= 0:
            raise RuntimeError("Saved finite sampler has no usable training samples.")
        epoch_samples = self.usable_epoch_samples
        expected_epoch = position // epoch_samples
        expected_offset = position % epoch_samples
        expected_last_epoch = (
            0 if position == 0 else int(math.ceil(position / epoch_samples))
        )
        if int(state.get("next_data_epoch_index", -1)) != expected_epoch or int(
            state.get("next_sample_offset_in_epoch", -1)
        ) != expected_offset or int(
            state.get("last_committed_data_epoch", -1)
        ) != expected_last_epoch:
            raise RuntimeError(
                "Saved draw cursor disagrees with its epoch/offset provenance."
            )
        legal_batch_boundary = (
            position % self.batch_size == 0
            if self.infinite
            else expected_offset == 0 or expected_offset % self.batch_size == 0
        )
        if not legal_batch_boundary:
            raise RuntimeError(
                "Saved committed draw position is not a legal batch boundary."
            )
        self.next_committed_draw = position
        self.issued_draw_count = position


# Backward import compatibility for callers that referenced the old internal
# name. Its persisted schema is intentionally not backward compatible.
StatefulRandomSampler = DrawIndexedSampler


def _random_state_nodes(root: Any) -> dict[str, Any]:
    """Find MONAI-style Randomizable objects without traversing tensors/models."""

    nodes: dict[str, Any] = {}
    visited: set[int] = set()

    def visit(value: Any, path: str) -> None:
        if value is None or id(value) in visited:
            return
        visited.add(id(value))
        random_state = getattr(value, "R", None)
        if random_state is not None and (
            callable(getattr(random_state, "get_state", None))
            or hasattr(random_state, "bit_generator")
        ):
            nodes[path] = value
        for attribute in (
            "dataset",
            "base_dataset",
            "transform",
            "post_transform",
        ):
            child = getattr(value, attribute, None)
            if child is not None and child is not value:
                visit(child, f"{path}.{attribute}")
        transforms = getattr(value, "transforms", None)
        if isinstance(transforms, (list, tuple)):
            for index, child in enumerate(transforms):
                visit(child, f"{path}.transforms[{index}]")
        # MONAI random transforms sometimes delegate sampling to nested helper
        # objects (for example an affine grid). Traverse only transform-like
        # children, never arbitrary row dictionaries or tensor payloads.
        for attribute, child in getattr(value, "__dict__", {}).items():
            if attribute in {
                "dataset",
                "base_dataset",
                "transform",
                "post_transform",
                "transforms",
                "R",
            }:
                continue
            child_module = type(child).__module__
            if hasattr(child, "R") or child_module.startswith("monai."):
                visit(child, f"{path}.{attribute}")
            elif isinstance(child, (list, tuple)):
                for index, item in enumerate(child):
                    if hasattr(item, "R") or type(item).__module__.startswith("monai."):
                        visit(item, f"{path}.{attribute}[{index}]")

    visit(root, "dataset")
    return nodes


def _capture_dataset_random_state(dataset: Any) -> dict[str, dict]:
    states: dict[str, dict] = {}
    for path, node in _random_state_nodes(dataset).items():
        random_state = node.R
        if callable(getattr(random_state, "get_state", None)):
            states[path] = {
                "kind": "RandomState",
                "state": random_state.get_state(),
                "type": f"{type(node).__module__}.{type(node).__qualname__}",
            }
        elif hasattr(random_state, "bit_generator"):
            states[path] = {
                "kind": "Generator",
                "state": random_state.bit_generator.state,
                "type": f"{type(node).__module__}.{type(node).__qualname__}",
            }
    return states


def _restore_dataset_random_state(dataset: Any, states: Mapping) -> None:
    nodes = _random_state_nodes(dataset)
    if set(nodes) != set(states):
        raise RuntimeError(
            "Dataset random-transform topology changed; exact resume is impossible."
        )
    for path, saved in states.items():
        node = nodes[path]
        node_type = f"{type(node).__module__}.{type(node).__qualname__}"
        if saved.get("type") != node_type:
            raise RuntimeError(
                f"Dataset random transform changed at {path}; exact resume refused."
            )
        if saved.get("kind") == "RandomState":
            node.R.set_state(saved["state"])
        elif saved.get("kind") == "Generator":
            node.R.bit_generator.state = saved["state"]
        else:
            raise RuntimeError(f"Unknown dataset RNG state kind at {path}.")


def _make_loader(
    rows: Sequence[Mapping],
    *,
    split: str,
    batch_size: int,
    cache_dir: str | Path | None,
    seed: int,
    num_workers: int,
    dataset_factory: Callable | None,
    sampler_mode: str = "finite_shuffle",
    drop_last: bool = False,
) -> DataLoader:
    if not rows:
        raise ValueError(f"Cannot create an empty {split} loader.")
    if dataset_factory is None:
        if cache_dir is not None:
            try:
                from .cache import make_cached_dataset
            except ImportError:  # pragma: no cover
                from cache import make_cached_dataset

            def dataset_factory(values, *, split, cache_dir, seed):
                # Resolve the sealed manifest's data_directory and reject rows
                # outside its declared membership before MONAI is constructed.
                return make_cached_dataset(
                    cache_dir, values, split=split, seed=seed
                )
        else:
            try:
                from .data import make_dataset as dataset_factory
            except ImportError:  # pragma: no cover
                from data import make_dataset as dataset_factory
    dataset = _call_supported(
        dataset_factory,
        rows,
        split=split,
        cache_dir=cache_dir,
        seed=seed,
    )
    sampler_mode = str(sampler_mode).lower()
    if sampler_mode not in {"finite_shuffle", "sharded_infinite"}:
        raise ValueError(f"Unsupported 3DINO sampler mode: {sampler_mode}")
    if split != "train" and sampler_mode != "finite_shuffle":
        raise ValueError("Infinite sampling is valid only for the training partition.")
    if isinstance(dataset, DataLoader):
        if sampler_mode != "finite_shuffle" or bool(drop_last):
            raise RuntimeError(
                "An injected DataLoader cannot prove the requested publication "
                "sampler/drop_last contract. Supply a dataset instead."
            )
        if split != "train" and int(dataset.num_workers) != 0:
            raise RuntimeError(
                "Injected validation/test DataLoaders must use num_workers=0."
            )
        setattr(dataset, "_threedino_exact_resume_supported", False)
        setattr(
            dataset,
            "_threedino_exact_resume_reason",
            "injected DataLoader does not use the internal stateful sampler",
        )
        return dataset
    requested_num_workers = int(num_workers)
    if requested_num_workers < 0:
        raise ValueError("3DINO training num_workers must be non-negative.")
    effective_num_workers = requested_num_workers if split == "train" else 0
    rows_fingerprint = _training_rows_fingerprint(rows)
    transform_contract_fingerprint = _training_transform_contract_fingerprint(
        dataset
    )
    if split == "train":
        dataset = DrawSeededDataset(dataset)
    loader_generator = torch.Generator().manual_seed(int(seed) + 1_000_003)
    sampler = (
        DrawIndexedSampler(
            dataset,
            seed=int(seed),
            infinite=sampler_mode == "sharded_infinite",
            batch_size=max(1, int(batch_size)),
            drop_last=bool(drop_last),
            training_rows_fingerprint=rows_fingerprint,
            transform_contract_fingerprint=transform_contract_fingerprint,
        )
        if split == "train"
        else None
    )
    worker_init_fn = None
    if effective_num_workers > 0:
        try:
            from monai.data.utils import worker_init_fn as monai_worker_init_fn
        except ImportError:
            # An injected non-MONAI dataset remains valid without this hook.
            # The default dataset factory already raises a targeted MONAI error.
            monai_worker_init_fn = None
        worker_init_fn = monai_worker_init_fn
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": max(1, int(batch_size)),
        "shuffle": False,
        "sampler": sampler,
        "num_workers": effective_num_workers,
        "pin_memory": torch.cuda.is_available(),
        "generator": loader_generator,
        "worker_init_fn": worker_init_fn,
        "drop_last": bool(drop_last) if split == "train" else False,
        "persistent_workers": bool(split == "train" and effective_num_workers > 0),
    }
    if effective_num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(
        **loader_kwargs,
    )
    loader._threedino_stateful_sampler = sampler
    loader._threedino_loader_generator = loader_generator
    loader._threedino_exact_resume_supported = split == "train"
    loader._threedino_exact_resume_reason = (
        None if split == "train" else "evaluation loaders do not carry training state"
    )
    loader._threedino_sampler_mode = sampler_mode
    loader._threedino_drop_last = bool(drop_last) if split == "train" else False
    loader._threedino_requested_num_workers = requested_num_workers
    loader._threedino_transform_contract_fingerprint = (
        transform_contract_fingerprint
    )
    loader._threedino_training_rows_fingerprint = rows_fingerprint
    loader._threedino_worker_history = [effective_num_workers]
    loader._threedino_resume_events = []
    if split == "train" and sampler_mode == "finite_shuffle" and len(loader) == 0:
        raise ValueError(
            "drop_last removed every training batch; reduce batch_size or use "
            "the publication sharded_infinite sampler."
        )
    return loader


def _capture_training_data_state(loader: DataLoader) -> dict:
    supported = bool(
        getattr(loader, "_threedino_exact_resume_supported", False)
    )
    reason = getattr(loader, "_threedino_exact_resume_reason", None)
    state = {
        "schema_version": TRAINING_DATA_STATE_SCHEMA_VERSION,
        "exact_resume_supported": supported,
        "unsupported_reason": reason,
        "dataset_size": len(loader.dataset),
        "batch_size": loader.batch_size,
        "training_num_workers": int(loader.num_workers),
        "persistent_training_workers": bool(loader.persistent_workers),
        "prefetch_factor": (
            int(loader.prefetch_factor) if loader.num_workers > 0 else None
        ),
        "validation_num_workers": 0,
        "test_num_workers": 0,
        "worker_count_change_supported": True,
        "worker_history": list(
            getattr(loader, "_threedino_worker_history", [int(loader.num_workers)])
        ),
        "resume_events": list(getattr(loader, "_threedino_resume_events", [])),
        "sampler": getattr(loader, "_threedino_sampler_mode", None),
        "drop_last": bool(getattr(loader, "_threedino_drop_last", False)),
        "sequence_schema": DATA_SEQUENCE_SCHEMA,
        "augmentation_seed_schema": AUGMENTATION_SEED_SCHEMA,
        "training_rows_fingerprint": getattr(
            loader, "_threedino_training_rows_fingerprint", None
        ),
        "transform_contract_fingerprint": getattr(
            loader, "_threedino_transform_contract_fingerprint", None
        ),
    }
    if not supported:
        return state
    sampler = getattr(loader, "_threedino_stateful_sampler", None)
    if not isinstance(sampler, DrawIndexedSampler):
        raise RuntimeError(
            "Loader claimed exact resume support without a draw-indexed sampler."
        )
    sampler_state = sampler.state_dict()
    state.update(sampler_state)
    state["sampler_state"] = sampler_state
    return state


def _restore_training_data_state(loader: DataLoader, state: Mapping) -> bool:
    """Restore loader state; return whether the checkpoint was mid-epoch."""

    if int(state.get("schema_version", -1)) != TRAINING_DATA_STATE_SCHEMA_VERSION:
        raise RuntimeError(
            "Legacy schema-v1 DataLoader state cannot be resumed under the "
            "draw-indexed exact-resume-v2 contract; start a separate new run."
        )
    if not bool(state.get("exact_resume_supported", False)):
        reason = state.get("unsupported_reason") or "unknown loader configuration"
        raise RuntimeError(
            f"Checkpoint cannot be resumed exactly ({reason}); refusing approximate resume."
        )
    if not bool(getattr(loader, "_threedino_exact_resume_supported", False)):
        reason = getattr(loader, "_threedino_exact_resume_reason", None)
        raise RuntimeError(
            f"Current loader cannot resume exactly ({reason}); refusing resume."
        )
    if int(state.get("dataset_size", -1)) != len(loader.dataset):
        raise RuntimeError("Training dataset size changed; refusing resume.")
    if state.get("batch_size") != loader.batch_size:
        raise RuntimeError("Training batch size changed; refusing resume.")
    if state.get("sampler") != getattr(loader, "_threedino_sampler_mode", None):
        raise RuntimeError("Training sampler mode changed; refusing resume.")
    if bool(state.get("drop_last")) != bool(
        getattr(loader, "_threedino_drop_last", False)
    ):
        raise RuntimeError("Training drop_last policy changed; refusing resume.")
    sampler = getattr(loader, "_threedino_stateful_sampler", None)
    if not isinstance(sampler, DrawIndexedSampler):
        raise RuntimeError("Current loader is missing exact-resume components.")
    for key, expected in (
        ("sequence_schema", DATA_SEQUENCE_SCHEMA),
        ("augmentation_seed_schema", AUGMENTATION_SEED_SCHEMA),
        (
            "training_rows_fingerprint",
            getattr(loader, "_threedino_training_rows_fingerprint", None),
        ),
        (
            "transform_contract_fingerprint",
            getattr(loader, "_threedino_transform_contract_fingerprint", None),
        ),
    ):
        if state.get(key) != expected:
            raise RuntimeError(
                f"Training draw contract changed at {key}; refusing exact resume."
            )
    sampler_state = state.get("sampler_state")
    if not isinstance(sampler_state, Mapping):
        raise RuntimeError("Checkpoint lacks its sealed draw-sampler state.")
    for key, value in sampler_state.items():
        if key == "schema_version":
            continue
        if key not in state or state.get(key) != value:
            raise RuntimeError(
                f"Checkpoint has conflicting top-level/nested draw state at {key}."
            )
    sampler.load_state_dict(sampler_state)
    previous_workers = int(state.get("training_num_workers", 0))
    current_workers = int(loader.num_workers)
    history = [int(value) for value in state.get("worker_history", [previous_workers])]
    if not history or history[-1] != current_workers:
        history.append(current_workers)
    loader._threedino_worker_history = history
    events = list(state.get("resume_events", []))
    events.append(
        {
            "resumed_at_utc": utc_now(),
            "previous_training_num_workers": previous_workers,
            "current_training_num_workers": current_workers,
            "persistent_training_workers": bool(loader.persistent_workers),
            "validation_num_workers": 0,
            "test_num_workers": 0,
            "next_committed_draw": int(sampler.next_committed_draw),
            "data_sequence_preserved": True,
        }
    )
    loader._threedino_resume_events = events
    return sampler.mid_epoch


def _iterator_after_data_restore(
    loader: DataLoader, *, restored_mid_epoch: bool
):
    """Create a new iterator at the sampler's committed draw position."""

    del restored_mid_epoch  # Encoded by DrawIndexedSampler.next_committed_draw.
    return iter(loader)


def _batch_draw_indices(batch: Any) -> list[int]:
    if not isinstance(batch, Mapping) or DRAW_INDEX_KEY not in batch:
        raise KeyError(
            f"Training batch lacks required internal draw identity {DRAW_INDEX_KEY!r}."
        )
    values = torch.as_tensor(batch[DRAW_INDEX_KEY], dtype=torch.long).reshape(-1)
    return [int(value) for value in values.tolist()]


def _commit_training_batch(loader: DataLoader, batch: Any) -> None:
    sampler = getattr(loader, "_threedino_stateful_sampler", None)
    if not isinstance(sampler, DrawIndexedSampler):
        raise RuntimeError("Training loader has no draw-indexed commit protocol.")
    sampler.commit(_batch_draw_indices(batch))


def _commit_training_batches(loader: DataLoader, batches: Sequence[Any]) -> None:
    draw_indices: list[int] = []
    for batch in batches:
        draw_indices.extend(_batch_draw_indices(batch))
    sampler = getattr(loader, "_threedino_stateful_sampler", None)
    if not isinstance(sampler, DrawIndexedSampler):
        raise RuntimeError("Training loader has no draw-indexed commit protocol.")
    sampler.commit(draw_indices)


def _remaining_finite_batches(data_state: Mapping) -> int | None:
    sampler_state = data_state.get("sampler_state", {})
    if sampler_state.get("sampler") != "finite_shuffle":
        return None
    dataset_size = int(data_state["dataset_size"])
    batch_size = int(data_state["batch_size"])
    epoch_samples = (
        (dataset_size // batch_size) * batch_size
        if bool(data_state.get("drop_last", False))
        else dataset_size
    )
    position = int(sampler_state.get("next_committed_draw", 0)) % epoch_samples
    if position == 0 and int(sampler_state.get("next_committed_draw", 0)) > 0:
        return 0
    remaining_samples = epoch_samples - position
    return int(math.ceil(remaining_samples / batch_size))


def _consumed_finite_batches(data_state: Mapping) -> int | None:
    sampler_state = data_state.get("sampler_state", {})
    if sampler_state.get("sampler") != "finite_shuffle":
        return None
    dataset_size = int(data_state["dataset_size"])
    batch_size = int(data_state["batch_size"])
    epoch_samples = (
        (dataset_size // batch_size) * batch_size
        if bool(data_state.get("drop_last", False))
        else dataset_size
    )
    absolute_position = int(sampler_state.get("next_committed_draw", 0))
    position = absolute_position % epoch_samples
    if position == 0 and absolute_position > 0:
        position = epoch_samples
    return int(math.ceil(position / batch_size))


def _training_loader_operational_state(loader: DataLoader) -> dict:
    sampler = getattr(loader, "_threedino_stateful_sampler", None)
    return {
        "training_num_workers": int(loader.num_workers),
        "persistent_training_workers": bool(loader.persistent_workers),
        "prefetch_factor": (
            int(loader.prefetch_factor) if loader.num_workers > 0 else None
        ),
        "validation_num_workers": 0,
        "test_num_workers": 0,
        "worker_count_semantic": False,
        "worker_count_change_supported_at_checkpoint_restart": True,
        "worker_history": list(
            getattr(loader, "_threedino_worker_history", [int(loader.num_workers)])
        ),
        "resume_events": list(getattr(loader, "_threedino_resume_events", [])),
        "next_committed_draw": (
            int(sampler.next_committed_draw)
            if isinstance(sampler, DrawIndexedSampler)
            else None
        ),
        "data_sequence_preserved": True,
    }


def _training_data_position(loader: DataLoader) -> dict:
    sampler = getattr(loader, "_threedino_stateful_sampler", None)
    if not isinstance(sampler, DrawIndexedSampler):
        raise RuntimeError("Training loader has no draw-indexed data position.")
    epoch_samples = int(sampler.usable_epoch_samples)
    committed = int(sampler.next_committed_draw)
    return {
        "next_committed_draw": committed,
        "usable_epoch_samples": epoch_samples,
        "next_data_epoch_index": committed // epoch_samples,
        "next_sample_offset_in_epoch": committed % epoch_samples,
        "last_committed_data_epoch": (
            0 if committed == 0 else int(math.ceil(committed / epoch_samples))
        ),
    }


def _batch_values(batch) -> tuple[torch.Tensor, torch.Tensor, list[str], list[str]]:
    if isinstance(batch, Mapping):
        images = batch.get("image")
        labels = batch.get("label")
        patient_ids = batch.get("patient_id")
        label_strings = batch.get("label_str")
    elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
        images, labels = batch[:2]
        patient_ids = batch[2] if len(batch) > 2 else None
        label_strings = batch[3] if len(batch) > 3 else None
    else:
        raise TypeError("3DINO batches must be mappings or image/label tuples.")
    if images is None or labels is None:
        raise KeyError("3DINO batches require image and label values.")
    labels = torch.as_tensor(labels, dtype=torch.long).reshape(-1)
    batch_size = int(labels.numel())

    def strings(value, fallback):
        if value is None:
            return [fallback(index) for index in range(batch_size)]
        if isinstance(value, str):
            return [value]
        if torch.is_tensor(value):
            value = value.detach().cpu().tolist()
        return [str(item) for item in value]

    return (
        torch.as_tensor(images),
        labels,
        strings(patient_ids, lambda index: str(index)),
        strings(label_strings, lambda index: str(int(labels[index]))),
    )


def _autocast(
    device: torch.device, enabled: bool, dtype_name: str = "auto"
):
    if not enabled or device.type != "cuda":
        return contextlib.nullcontext()
    dtype_name = str(dtype_name).lower()
    if dtype_name == "float16":
        dtype = torch.float16
    elif dtype_name == "bfloat16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("bfloat16 autocast was requested but is unsupported.")
        dtype = torch.bfloat16
    elif dtype_name == "auto":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        raise ValueError(
            "autocast_dtype must be one of auto, float16, or bfloat16."
        )
    return torch.autocast(device_type="cuda", dtype=dtype)


def _grad_scaler(
    device: torch.device, enabled: bool, dtype_name: str = "auto"
):
    dtype_name = str(dtype_name).lower()
    uses_float16 = dtype_name == "float16" or (
        dtype_name == "auto"
        and device.type == "cuda"
        and not torch.cuda.is_bf16_supported()
    )
    active = bool(enabled and device.type == "cuda" and uses_float16)
    modern_scaler = getattr(getattr(torch, "amp", None), "GradScaler", None)
    if callable(modern_scaler):
        try:
            return modern_scaler("cuda", enabled=active)
        except TypeError:  # pragma: no cover - intermediate torch signature
            return modern_scaler(enabled=active)
    # PyTorch 2.0, pinned by the official 3DINO runtime, exposes GradScaler
    # only through torch.cuda.amp even when it is constructed disabled on CPU.
    return torch.cuda.amp.GradScaler(enabled=active)


def class_weights(
    rows: Sequence[Mapping],
    num_classes: int,
    scheme: str = "none",
    beta: float = 0.999,
) -> torch.Tensor:
    counts = Counter(int(row["label"]) for row in rows)
    values = np.asarray(
        [counts.get(index, 0) for index in range(num_classes)], dtype=np.float64
    )
    if np.any(values == 0):
        raise ValueError(
            f"Every training partition must contain all classes: {values.tolist()}"
        )
    scheme = str(scheme).lower()
    if scheme == "none":
        weights = np.ones_like(values)
    elif scheme == "inverse_frequency":
        weights = values.sum() / (len(values) * values)
    elif scheme == "effective_number":
        if not 0.0 < float(beta) < 1.0:
            raise ValueError("effective_number_beta must be in (0, 1).")
        weights = (1.0 - float(beta)) / (1.0 - np.power(float(beta), values))
    else:
        raise ValueError(f"Unknown class weighting scheme: {scheme}")
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def _max_iterations(parameters: Mapping, budget: int | None = None) -> int:
    if budget is not None:
        value = int(budget)
    else:
        value = int(
            parameters.get(
                "max_iterations",
                parameters.get("max_optimizer_steps", parameters.get("iterations", 0)),
            )
        )
    if value <= 0:
        raise ValueError("A positive max_iterations/max_optimizer_steps is required.")
    return value


def _runtime_plan_fingerprint(parameters: Mapping, runtime_plan: Mapping) -> str:
    return parameter_fingerprint(
        {"parameters": dict(parameters), "runtime_plan": dict(runtime_plan)}
    )


def _training_rows_fingerprint(rows: Sequence[Mapping]) -> str:
    identity = [
        {
            "position": index,
            "patient_id": str(row.get("patient_id", "")),
            "label": int(row["label"]),
            "label_str": str(row.get("label_str", "")),
            "source_fingerprint": row.get("source_fingerprint"),
            "source_size_bytes": row.get("source_size_bytes"),
            "source_mtime_ns": row.get("source_mtime_ns"),
            "source": str(
                row.get("source_path", row.get("cached_path", row.get("image", "")))
                if isinstance(
                    row.get("source_path", row.get("cached_path", row.get("image", ""))),
                    (str, Path),
                )
                else "in_memory"
            ),
        }
        for index, row in enumerate(rows)
    ]
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def _validate_resume_runtime(
    checkpoint: Mapping,
    *,
    parameters: Mapping,
    runtime_plan: Mapping,
) -> int:
    training_data_state = checkpoint.get("training_data_state")
    if not isinstance(training_data_state, Mapping) or int(
        training_data_state.get("schema_version", -1)
    ) != TRAINING_DATA_STATE_SCHEMA_VERSION:
        raise RuntimeError(
            "Legacy schema-v1 checkpoint state cannot be migrated to the "
            "draw-indexed exact-resume-v2 stream; start a separate new run."
        )
    expected_fingerprint = _runtime_plan_fingerprint(parameters, runtime_plan)
    if checkpoint.get("runtime_plan_fingerprint") != expected_fingerprint:
        raise RuntimeError(
            "Refusing resume because the effective runtime budget, scheduler "
            "horizon, loader plan, or parameters changed."
        )
    if checkpoint.get("runtime_plan") != dict(runtime_plan):
        raise RuntimeError("Runtime plan differs despite its fingerprint; refusing resume.")
    start_step = int(checkpoint.get("optimizer_step", -1))
    horizon = int(runtime_plan["scheduler_total_updates"])
    if start_step < 0 or start_step > horizon:
        raise RuntimeError(
            f"Checkpoint optimizer_step={start_step} is outside scheduler horizon {horizon}."
        )
    scheduler_state = checkpoint.get("scheduler_state_dict", {})
    scheduler_step = int(scheduler_state.get("last_epoch", -1))
    if scheduler_step != start_step:
        raise RuntimeError(
            "Checkpoint scheduler step does not match optimizer progress; refusing resume."
        )
    return start_step


def _cosine_scheduler(optimizer, total_updates: int, warmup_fraction: float = 0.0):
    if total_updates <= 0:
        raise ValueError("total_updates must be positive.")
    warmup_updates = int(total_updates * float(warmup_fraction))

    def factor(update_index: int) -> float:
        if warmup_updates and update_index < warmup_updates:
            return max(1e-8, (update_index + 1) / warmup_updates)
        progress = (update_index - warmup_updates) / max(
            1, total_updates - warmup_updates
        )
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _representation(candidate: Mapping) -> tuple[int, bool]:
    return (
        int(candidate.get("n_last_blocks", 1)),
        bool(candidate.get("use_mean_patch_token", False)),
    )


def _normalize_feature_bank(
    bank: Mapping, representations: Sequence[tuple[int, bool]]
) -> dict[tuple[int, bool], torch.Tensor]:
    result: dict[tuple[int, bool], torch.Tensor] = {}
    for representation in representations:
        possible_keys = (
            representation,
            str(representation),
            f"last{representation[0]}_mean{int(representation[1])}",
        )
        value = next((bank[key] for key in possible_keys if key in bank), None)
        if value is None:
            raise KeyError(f"Feature bank omitted representation {representation}.")
        if not torch.is_tensor(value) or value.ndim != 2:
            raise ValueError(
                f"Representation {representation} must be a [batch, feature] tensor."
            )
        result[representation] = value.detach()
    return result


def _extract_bank(
    encoder,
    images: torch.Tensor,
    representations: Sequence[tuple[int, bool]],
    extractor: Callable | None,
) -> dict[tuple[int, bool], torch.Tensor]:
    extractor = extractor or _default_feature_bank_extractor
    with torch.no_grad():
        bank = _call_supported(
            extractor,
            encoder,
            images,
            representations=tuple(representations),
        )
    if not isinstance(bank, Mapping):
        if len(representations) != 1 or not torch.is_tensor(bank):
            raise TypeError("Feature-bank extractor must return a mapping.")
        bank = {representations[0]: bank}
    return _normalize_feature_bank(bank, representations)


class ProbeHeadBank(nn.Module):
    """Independent linear classifiers keyed by deterministic candidate id."""

    def __init__(
        self,
        candidates: Sequence[Mapping],
        num_classes: int,
        feature_dimension_fn: Callable[[int, bool], int] | None = None,
        seed: int = 42,
    ):
        super().__init__()
        feature_dimension_fn = feature_dimension_fn or _default_feature_dimension
        self.candidates = {str(value["candidate_id"]): dict(value) for value in candidates}
        self.heads = nn.ModuleDict()
        for identity, value in self.candidates.items():
            n_last_blocks, use_mean = _representation(value)
            expected_dimension = int(feature_dimension_fn(n_last_blocks, use_mean))
            supplied_dimension = value.get("feature_dim")
            if supplied_dimension is not None and int(supplied_dimension) != expected_dimension:
                raise ValueError(
                    f"Candidate {identity} feature_dim={supplied_dimension} disagrees "
                    f"with the model dimension {expected_dimension}."
                )
            init_payload = (
                f"3dino-probe-head-init-v1|{int(seed)}|{identity}"
            ).encode("utf-8")
            init_seed = int.from_bytes(
                hashlib.sha256(init_payload).digest()[:8], "big"
            ) % (2**63 - 1)
            # Make one candidate's initialization independent of candidate-bank
            # order/size while leaving the caller's global RNG stream untouched.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(init_seed)
                head = nn.Linear(expected_dimension, num_classes)
                nn.init.normal_(head.weight, mean=0.0, std=0.01)
                nn.init.zeros_(head.bias)
            self.heads[identity] = head

    def forward_bank(self, features: Mapping[tuple[int, bool], torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            identity: self.heads[identity](features[_representation(candidate)])
            for identity, candidate in self.candidates.items()
        }


def _probe_optimizer(heads: ProbeHeadBank, candidates: Sequence[Mapping], parameters: Mapping):
    if str(parameters.get("optimizer", "sgd")).lower() != "sgd":
        raise ValueError("Publication-faithful 3DINO linear probing requires SGD.")
    if str(parameters.get("scheduler", "cosine")).lower() != "cosine":
        raise ValueError("Publication-faithful 3DINO linear probing requires cosine decay.")
    if str(parameters.get("loss", "cross_entropy")).lower() not in {
        "cross_entropy",
        "ce",
    }:
        raise ValueError("Publication-faithful 3DINO linear probing requires cross entropy.")
    momentum = float(parameters.get("momentum", 0.9))
    weight_decay = float(parameters.get("weight_decay", 0.0))
    if not math.isclose(momentum, 0.9, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("The publication linear-probe momentum is fixed at 0.9.")
    if not math.isclose(weight_decay, 0.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("The publication linear-probe weight decay is fixed at zero.")
    if not math.isclose(
        float(parameters.get("scheduler_eta_min", 0.0)),
        0.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("The publication cosine schedule anneals to zero.")
    groups = []
    for candidate in candidates:
        identity = str(candidate["candidate_id"])
        groups.append(
            {
                "params": heads.heads[identity].parameters(),
                "lr": float(candidate["lr"]),
                "initial_lr": float(candidate["lr"]),
                "candidate_id": identity,
                "name": identity,
            }
        )
    return torch.optim.SGD(groups, momentum=momentum, weight_decay=weight_decay)


def _candidate_evaluation(
    logits: torch.Tensor,
    labels: np.ndarray,
    patient_ids: list[str],
    label_strings: list[str],
    classes: list[str],
    mean_loss: float,
) -> dict:
    probabilities = torch.softmax(logits.float(), dim=1).cpu().numpy()
    predicted = probabilities.argmax(axis=1)
    metrics = compute_metrics(labels, predicted, probabilities, classes)
    metrics["loss"] = float(mean_loss)
    metrics["selection_score"] = fold_selection_score(metrics)
    prediction_rows, frame = format_predictions(
        patient_ids, label_strings, probabilities, classes
    )
    return {
        "metrics": metrics,
        "predictions": prediction_rows,
        "probabilities": frame,
        "logits": logits.float().cpu(),
        "labels": labels,
    }


@torch.no_grad()
def evaluate_probe_heads(
    encoder,
    heads: ProbeHeadBank,
    loader: DataLoader,
    classes: list[str],
    criterion,
    device: torch.device,
    *,
    feature_bank_extractor: Callable | None = None,
    mixed_precision: bool = False,
    autocast_dtype: str = "auto",
) -> dict[str, dict]:
    """Evaluate every head while sharing one frozen-encoder pass per batch."""

    encoder.eval()
    heads.eval()
    representations = sorted(
        {_representation(value) for value in heads.candidates.values()}
    )
    logits_by_candidate: dict[str, list[torch.Tensor]] = {
        identity: [] for identity in heads.candidates
    }
    loss_sums = {identity: 0.0 for identity in heads.candidates}
    labels_all: list[torch.Tensor] = []
    patient_ids: list[str] = []
    label_strings: list[str] = []
    sample_count = 0
    for batch in loader:
        images, labels, batch_ids, batch_labels = _batch_values(batch)
        images = images.to(device, non_blocking=True, dtype=torch.float32)
        labels = labels.to(device, non_blocking=True)
        with _autocast(device, mixed_precision, autocast_dtype):
            features = _extract_bank(
                encoder, images, representations, feature_bank_extractor
            )
            logits_bank = heads.forward_bank(features)
        for identity, logits in logits_bank.items():
            loss_sums[identity] += float(criterion(logits.float(), labels).item()) * int(
                labels.numel()
            )
            logits_by_candidate[identity].append(logits.detach().float().cpu())
        labels_all.append(labels.detach().cpu())
        patient_ids.extend(batch_ids)
        label_strings.extend(batch_labels)
        sample_count += int(labels.numel())
    if sample_count == 0:
        raise ValueError("Evaluation loader yielded no samples.")
    labels_array = torch.cat(labels_all).numpy()
    return {
        identity: _candidate_evaluation(
            torch.cat(logits_parts),
            labels_array,
            patient_ids,
            label_strings,
            classes,
            loss_sums[identity] / sample_count,
        )
        for identity, logits_parts in logits_by_candidate.items()
    }


def _save_evaluation(output_dir: Path, prefix: str, evaluation: Mapping) -> None:
    output_dir = Path(output_dir)
    save_json(output_dir / "metrics_summary.json", evaluation["metrics"])
    save_json(
        output_dir / f"{prefix}_patient_predictions.json",
        evaluation["predictions"],
    )
    evaluation["probabilities"].to_csv(
        output_dir / f"{prefix}_probabilities.csv", index=False
    )


def _promote_candidate_evaluation(candidate_dir: Path, output_dir: Path) -> None:
    """Copy the selected head's best predictions into the fold-level contract."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "validation_patient_predictions.json",
        "validation_probabilities.csv",
    ):
        source = Path(candidate_dir) / name
        if not source.is_file():
            raise FileNotFoundError(f"Candidate evaluation artifact missing: {source}")
        shutil.copy2(source, output_dir / name)


def _head_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _probe_checkpoint(
    *,
    heads: ProbeHeadBank,
    optimizer,
    scheduler,
    candidates: list[dict],
    base_parameters: dict,
    classes: list[str],
    source_provenance: dict,
    runtime_plan: dict,
    training_data_state: dict,
    optimizer_step: int,
    epoch: int,
    batch_in_epoch: int,
    history: list[dict],
    best_records: dict[str, dict],
    best_head_states: dict[str, dict[str, torch.Tensor]],
    selected_candidate_id: str | None,
) -> dict:
    selected_state = (
        best_head_states.get(selected_candidate_id) if selected_candidate_id else None
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "model_name": "3dino_linear_probe",
        "model_family": MODEL_FAMILY,
        "method": "linear_probe",
        "variant": "linear_probe",
        "checkpoint_scope": "linear_heads_only_frozen_encoder_reconstructed",
        "encoder_frozen": True,
        "base_parameters": dict(base_parameters),
        "parameters": dict(base_parameters),
        "parameters_fingerprint": parameter_fingerprint(base_parameters),
        "runtime_plan": runtime_plan,
        "runtime_plan_fingerprint": _runtime_plan_fingerprint(
            base_parameters, runtime_plan
        ),
        "candidates": candidates,
        "classes": classes,
        "source_provenance": source_provenance,
        "optimizer_step": int(optimizer_step),
        "epoch": int(epoch),
        "batch_in_epoch": int(batch_in_epoch),
        "heads_state_dict": heads.state_dict(),
        "candidate_head_state_dicts": best_head_states,
        "head_state_dict": selected_state,
        "selected_candidate_id": selected_candidate_id,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "history": history,
        "best_records": best_records,
        "rng_state": _rng_state(),
        "training_data_state": training_data_state,
        "resume_semantics": "exact_or_fail_closed",
        "saved_at_utc": utc_now(),
    }


def _candidate_runtime_plan(
    runtime_plan: Mapping,
    candidate: Mapping,
    *,
    candidate_count: int,
) -> dict:
    """Bind one independently optimized head to its shared-grid execution plan."""

    parameters = dict(candidate)
    identity = str(parameters["candidate_id"])
    return {
        **dict(runtime_plan),
        "candidate_id": identity,
        "candidate_parameters_fingerprint": parameter_fingerprint(parameters),
        "joint_candidate_count": int(candidate_count),
        "candidate_execution": (
            "independent_linear_head_with_shared_frozen_encoder_pass"
        ),
        "head_initialization_schema": PROBE_HEAD_INITIALIZATION_SCHEMA,
    }


def _candidate_checkpoint(
    *,
    candidate: Mapping,
    candidate_count: int,
    classes: list[str],
    source_provenance: Mapping,
    runtime_plan: Mapping,
    seed: int,
    best_record: Mapping,
    head_state_dict: Mapping[str, torch.Tensor],
) -> dict:
    """Build the canonical, self-contained checkpoint for one grid head."""

    parameters = dict(candidate)
    identity = str(parameters["candidate_id"])
    candidate_plan = _candidate_runtime_plan(
        runtime_plan,
        parameters,
        candidate_count=candidate_count,
    )
    best_optimizer_step = int(best_record["optimizer_step"])
    return {
        "schema_version": SCHEMA_VERSION,
        "model_name": "3dino_linear_probe",
        "model_family": MODEL_FAMILY,
        "method": "linear_probe",
        "variant": "linear_probe",
        "checkpoint_scope": "single_linear_head",
        "encoder_frozen": True,
        "shared_encoder_pass_per_candidate_bank": True,
        "candidate": parameters,
        "candidate_id": identity,
        "parameters": parameters,
        "parameters_fingerprint": parameter_fingerprint(parameters),
        "runtime_plan": candidate_plan,
        "runtime_plan_fingerprint": _runtime_plan_fingerprint(
            parameters, candidate_plan
        ),
        "classes": list(classes),
        "source_provenance": dict(source_provenance),
        "seed": int(seed),
        "optimizer_step": best_optimizer_step,
        "best_optimizer_step": best_optimizer_step,
        "epoch": int(best_record["epoch"]),
        "head_state_dict": dict(head_state_dict),
        "validation_metrics": dict(best_record["metrics"]),
        "resume_checkpoint_scope": "shared_training_checkpoint_only",
        "saved_at_utc": utc_now(),
    }


def _finalize_candidate_artifacts(
    *,
    output_dir: Path,
    candidates: Sequence[Mapping],
    classes: list[str],
    source_provenance: Mapping,
    runtime_plan: Mapping,
    seed: int,
    optimizer_steps_completed: int,
    best_records: Mapping[str, Mapping],
    best_head_states: Mapping[str, Mapping[str, torch.Tensor]],
    class_weight_values: Sequence[float],
    total_runtime_seconds: float,
) -> None:
    """Seal every jointly trained head as a normal reusable fold artifact."""

    expected = {str(value["candidate_id"]) for value in candidates}
    if set(best_records) != expected or set(best_head_states) != expected:
        raise RuntimeError(
            "Cannot complete a linear-probe grid with missing candidate best states."
        )
    candidate_count = len(candidates)
    for candidate in candidates:
        parameters = dict(candidate)
        identity = str(parameters["candidate_id"])
        record = dict(best_records[identity])
        candidate_dir = Path(output_dir) / "candidates" / identity
        required_evaluation_artifacts = (
            candidate_dir / "metrics_summary.json",
            candidate_dir / "validation_patient_predictions.json",
            candidate_dir / "validation_probabilities.csv",
        )
        missing = [
            str(path) for path in required_evaluation_artifacts if not path.is_file()
        ]
        if missing:
            raise RuntimeError(
                "Cannot complete a linear-probe candidate whose best validation "
                f"artifacts are missing: {missing}"
            )
        candidate_plan = _candidate_runtime_plan(
            runtime_plan,
            parameters,
            candidate_count=candidate_count,
        )
        atomic_torch_save(
            _candidate_checkpoint(
                candidate=parameters,
                candidate_count=candidate_count,
                classes=classes,
                source_provenance=source_provenance,
                runtime_plan=runtime_plan,
                seed=seed,
                best_record=record,
                head_state_dict=best_head_states[identity],
            ),
            candidate_dir / "best_model.pt",
        )
        save_json(
            candidate_dir / "training_state.json",
            {
                "schema_version": SCHEMA_VERSION,
                "model_name": "3dino_linear_probe",
                "model_family": MODEL_FAMILY,
                "method": "linear_probe",
                "completed": True,
                "encoder_frozen": True,
                "stochastic_batch_encoder_passes": True,
                "shared_encoder_pass_per_candidate_bank": True,
                "candidate_id": identity,
                "candidate_count_in_shared_training": candidate_count,
                "parameters": parameters,
                "parameters_fingerprint": parameter_fingerprint(parameters),
                "runtime_plan": candidate_plan,
                "runtime_plan_fingerprint": _runtime_plan_fingerprint(
                    parameters, candidate_plan
                ),
                "resume_checkpoint_scope": "shared_training_checkpoint_only",
                "optimizer_steps_completed": int(optimizer_steps_completed),
                "best_optimizer_step": int(record["optimizer_step"]),
                "best_epoch": int(record["epoch"]),
                "best_selection_score": float(
                    record["metrics"]["selection_score"]
                ),
                "validation_metrics": dict(record["metrics"]),
                "classes": list(classes),
                "class_weights": [float(value) for value in class_weight_values],
                "source_provenance": dict(source_provenance),
                "seed": int(seed),
                "total_runtime_seconds": float(total_runtime_seconds),
                "completed_at_utc": utc_now(),
            },
        )


def _candidate_sort_key(record: Mapping) -> tuple:
    metrics = record["metrics"]
    perforation = metrics.get("per_class", {}).get("Perforation", {})
    recall = perforation.get("recall", perforation.get("sensitivity", -1.0))
    return (
        -float(metrics["selection_score"]),
        -float(recall if recall is not None else -1.0),
        float(record.get("nominal_lr", math.inf)),
        str(record["candidate_id"]),
    )


def train_linear_probe_grid(
    train_rows: list[dict],
    validation_rows: list[dict],
    base_parameters: dict,
    candidates: list[dict] | None,
    output_dir: Path,
    *,
    device: torch.device,
    upstream_root: str | Path | None = None,
    upstream_revision: str | None = None,
    checkpoint_path: str | Path | None = None,
    expected_checkpoint_sha256: str | None = None,
    cache_dir: str | Path | None = None,
    classes: list[str] | None = None,
    seed: int = 42,
    budget: int | None = None,
    num_workers: int = 0,
    resume: bool = False,
    encoder_factory: Callable | None = None,
    feature_bank_extractor: Callable | None = None,
    feature_dimension_fn: Callable[[int, bool], int] | None = None,
    dataset_factory: Callable | None = None,
    asset_provisioning: Mapping | None = None,
) -> dict:
    """Train independent publication heads from shared stochastic encoder passes.

    ``candidates`` may contain the full 52-head publication product or a curated
    subset.  Validation and checkpointing happen at optimizer-update boundaries;
    the held-out test is never accepted by this API.
    """

    if not validation_rows:
        raise ValueError("Linear-probe candidate selection requires validation rows.")
    classes = list(classes or DEFAULT_CLASSES)
    if len(classes) != 5:
        raise ValueError("3DINO experiments require the canonical five classes.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir)
    _log_asset_provisioning(logger, asset_provisioning)
    seed_everything(seed)
    parameters = dict(base_parameters)
    definition = {"fixed_parameters": parameters}
    if candidates is None:
        definition.update(parameters)
    else:
        definition["candidates"] = candidates
    resolved_candidates = build_probe_candidates(
        definition, world_size=int(parameters.get("world_size", 1))
    )
    if not resolved_candidates:
        raise ValueError("At least one linear-probe candidate is required.")
    max_iterations = _max_iterations(parameters, budget)
    batch_size = int(parameters.get("batch_size", 32))
    evaluation_interval = int(
        parameters.get("evaluation_interval_iterations", max_iterations)
    )
    if evaluation_interval <= 0:
        evaluation_interval = max_iterations
    checkpoint_interval = int(
        parameters.get("checkpoint_interval_iterations", evaluation_interval)
    )
    if checkpoint_interval <= 0:
        checkpoint_interval = evaluation_interval
    sampler_mode = str(parameters.get("sampler", "finite_shuffle")).lower()
    drop_last = bool(parameters.get("drop_last", False))
    autocast_dtype = str(parameters.get("autocast_dtype", "auto")).lower()
    mixed_precision = bool(parameters.get("mixed_precision", False))
    if sampler_mode == "sharded_infinite":
        if int(parameters.get("world_size", 1)) != 1:
            raise RuntimeError(
                "This training engine supports publication sharded_infinite "
                "sampling only for world_size=1."
            )
        if not drop_last:
            raise ValueError(
                "Publication sharded_infinite sampling requires drop_last=true."
            )
        if not mixed_precision or autocast_dtype != "float16":
            raise ValueError(
                "Publication probing requires mixed_precision=true and "
                "autocast_dtype=float16."
            )
        if device.type != "cuda":
            raise RuntimeError(
                "Publication sharded_infinite/fp16 probing requires a CUDA device."
            )
    runtime_plan = {
        "method": "linear_probe",
        "seed": int(seed),
        "max_iterations": max_iterations,
        "scheduler_total_updates": max_iterations,
        "scheduler_warmup_updates": 0,
        "evaluation_interval_iterations": evaluation_interval,
        "checkpoint_interval_iterations": checkpoint_interval,
        "batch_size": batch_size,
        "data_sequence_schema": DATA_SEQUENCE_SCHEMA,
        "augmentation_seed_schema": AUGMENTATION_SEED_SCHEMA,
        "preprocessing_contract_fingerprint": _canonical_sha256(
            preprocessing_contract()
        ),
        "validation_num_workers": 0,
        "test_num_workers": 0,
        "training_patient_count": len(train_rows),
        "training_rows_fingerprint": _training_rows_fingerprint(train_rows),
        "sampler": sampler_mode,
        "drop_last": drop_last,
        "mixed_precision": mixed_precision,
        "autocast_dtype": autocast_dtype,
        "device_type": device.type,
        "device_identity": _device_identity(device),
        "head_initialization_schema": PROBE_HEAD_INITIALIZATION_SCHEMA,
    }
    source = _source_provenance(
        upstream_root,
        upstream_revision,
        checkpoint_path,
        expected_checkpoint_sha256,
        injected=encoder_factory is not None,
    )
    factory = encoder_factory or _default_encoder_factory
    encoder = _call_supported(
        factory,
        upstream_root=upstream_root,
        upstream_revision=upstream_revision,
        checkpoint_path=checkpoint_path,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        device=device,
    )
    if hasattr(encoder, "to"):
        encoder = encoder.to(device)
    if hasattr(encoder, "eval"):
        encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None

    heads = ProbeHeadBank(
        resolved_candidates,
        len(classes),
        feature_dimension_fn=feature_dimension_fn,
        seed=seed,
    ).to(device)
    optimizer = _probe_optimizer(heads, resolved_candidates, parameters)
    scheduler = _cosine_scheduler(optimizer, max_iterations, warmup_fraction=0.0)
    weights = class_weights(
        train_rows,
        len(classes),
        str(parameters.get("class_weighting", "none")),
        float(parameters.get("effective_number_beta", 0.999)),
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    train_loader = _make_loader(
        train_rows,
        split="train",
        batch_size=batch_size,
        cache_dir=cache_dir,
        seed=seed,
        num_workers=num_workers,
        dataset_factory=dataset_factory,
        sampler_mode=sampler_mode,
        drop_last=drop_last,
    )
    validation_loader = _make_loader(
        validation_rows,
        split="validation",
        batch_size=batch_size,
        cache_dir=cache_dir,
        seed=seed,
        num_workers=0,
        dataset_factory=dataset_factory,
    )

    history: list[dict] = []
    best_records: dict[str, dict] = {}
    best_head_states: dict[str, dict[str, torch.Tensor]] = {}
    start_step = 0
    epoch = 1
    batch_in_epoch = 0
    last_path = output_dir / "last_checkpoint.pt"
    best_path = output_dir / "best_model.pt"
    restored_mid_epoch = False
    if resume and last_path.is_file():
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        if state.get("method") != "linear_probe":
            raise RuntimeError("Refusing to resume from a non-linear-probe checkpoint.")
        start_step = _validate_resume_runtime(
            state, parameters=parameters, runtime_plan=runtime_plan
        )
        if state.get("candidates") != resolved_candidates:
            raise RuntimeError("Refusing to resume with a changed candidate set.")
        if state.get("source_provenance") != source:
            raise RuntimeError("Refusing to resume with changed upstream/checkpoint provenance.")
        heads.load_state_dict(state["heads_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        epoch = int(state.get("epoch", 1))
        batch_in_epoch = int(state.get("batch_in_epoch", 0))
        history = list(state.get("history", []))
        best_records = dict(state.get("best_records", {}))
        best_head_states = dict(state.get("candidate_head_state_dicts", {}))
        _restore_rng_state(state.get("rng_state"))
        restored_mid_epoch = _restore_training_data_state(
            train_loader, state.get("training_data_state", {})
        )
        consumed_batches = _consumed_finite_batches(
            state.get("training_data_state", {})
        )
        if consumed_batches is not None and consumed_batches != batch_in_epoch:
            raise RuntimeError(
                "Sampler position and saved in-epoch batch count disagree; "
                "refusing resume."
            )
        resume_event = train_loader._threedino_resume_events[-1]
        logger.info(
            "Resuming linear-probe grid at optimizer step %d | "
            "training_workers=%d->%d next_committed_draw=%d "
            "data_sequence_preserved=True",
            start_step,
            resume_event["previous_training_num_workers"],
            resume_event["current_training_num_workers"],
            resume_event["next_committed_draw"],
        )

    representations = sorted({_representation(value) for value in resolved_candidates})
    iterator = _iterator_after_data_restore(
        train_loader, restored_mid_epoch=restored_mid_epoch
    )
    if resume and last_path.is_file() and not restored_mid_epoch:
        epoch += 1
        batch_in_epoch = 0
    start_time = time.time()
    for optimizer_step in range(start_step + 1, max_iterations + 1):
        try:
            batch = next(iterator)
            batch_in_epoch += 1
        except StopIteration:
            epoch += 1
            batch_in_epoch = 1
            iterator = iter(train_loader)
            batch = next(iterator)
        images, labels, _, _ = _batch_values(batch)
        images = images.to(device, non_blocking=True, dtype=torch.float32)
        labels = labels.to(device, non_blocking=True)
        encoder.eval()
        heads.train()
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, mixed_precision, autocast_dtype):
            features = _extract_bank(
                encoder, images, representations, feature_bank_extractor
            )
            logits_bank = heads.forward_bank(features)
            candidate_losses = {
                identity: criterion(logits.float(), labels)
                for identity, logits in logits_bank.items()
            }
            total_loss = torch.stack(list(candidate_losses.values())).sum()
        total_loss.backward()
        if any(parameter.grad is not None for parameter in encoder.parameters()):
            raise RuntimeError("Frozen 3DINO encoder unexpectedly received gradients.")
        optimizer.step()
        scheduler.step()
        _commit_training_batch(train_loader, batch)

        should_evaluate = (
            optimizer_step % evaluation_interval == 0
            or optimizer_step == max_iterations
        )
        if should_evaluate:
            evaluations = evaluate_probe_heads(
                encoder,
                heads,
                validation_loader,
                classes,
                criterion,
                device,
                feature_bank_extractor=feature_bank_extractor,
                mixed_precision=mixed_precision,
                autocast_dtype=autocast_dtype,
            )
            for candidate in resolved_candidates:
                identity = str(candidate["candidate_id"])
                evaluation = evaluations[identity]
                metrics = evaluation["metrics"]
                history.append(
                    {
                        "optimizer_step": optimizer_step,
                        "epoch": epoch,
                        "candidate_id": identity,
                        "nominal_lr": candidate["nominal_lr"],
                        "effective_lr": candidate["lr"],
                        "current_lr": next(
                            float(group["lr"])
                            for group in optimizer.param_groups
                            if group["candidate_id"] == identity
                        ),
                        "n_last_blocks": candidate["n_last_blocks"],
                        "use_mean_patch_token": candidate["use_mean_patch_token"],
                        "train_loss": float(candidate_losses[identity].detach().cpu()),
                        "validation_loss": metrics["loss"],
                        "validation_f1_macro": metrics["f1_macro"],
                        "validation_balanced_accuracy": metrics["balanced_accuracy"],
                        "validation_average_precision_macro": metrics[
                            "average_precision_macro"
                        ],
                        "validation_selection_score": metrics["selection_score"],
                    }
                )
                previous = best_records.get(identity)
                if previous is None or float(metrics["selection_score"]) > float(
                    previous["metrics"]["selection_score"]
                ):
                    candidate_dir = output_dir / "candidates" / identity
                    record = {
                        "candidate_id": identity,
                        "candidate": candidate,
                        "optimizer_step": optimizer_step,
                        "epoch": epoch,
                        "metrics": metrics,
                        "checkpoint": str(candidate_dir / "best_model.pt"),
                    }
                    best_records[identity] = record
                    best_head_states[identity] = _head_state(heads.heads[identity])
                    _save_evaluation(candidate_dir, "validation", evaluation)
                    atomic_torch_save(
                        _candidate_checkpoint(
                            candidate=candidate,
                            candidate_count=len(resolved_candidates),
                            classes=classes,
                            source_provenance=source,
                            runtime_plan=runtime_plan,
                            seed=seed,
                            best_record=record,
                            head_state_dict=best_head_states[identity],
                        ),
                        candidate_dir / "best_model.pt",
                    )

            selected = sorted(best_records.values(), key=_candidate_sort_key)[0]
            selected_id = str(selected["candidate_id"])
            checkpoint = _probe_checkpoint(
                heads=heads,
                optimizer=optimizer,
                scheduler=scheduler,
                candidates=resolved_candidates,
                base_parameters=parameters,
                classes=classes,
                source_provenance=source,
                runtime_plan=runtime_plan,
                training_data_state=_capture_training_data_state(train_loader),
                optimizer_step=optimizer_step,
                epoch=epoch,
                batch_in_epoch=batch_in_epoch,
                history=history,
                best_records=best_records,
                best_head_states=best_head_states,
                selected_candidate_id=selected_id,
            )
            atomic_torch_save(checkpoint, best_path)
            _promote_candidate_evaluation(
                output_dir / "candidates" / selected_id, output_dir
            )
            save_json(
                output_dir / "candidate_ranking.json",
                sorted(best_records.values(), key=_candidate_sort_key),
            )
            logger.info(
                "Linear probe step %d/%d | selected=%s | score=%.4f",
                optimizer_step,
                max_iterations,
                selected_id,
                selected["metrics"]["selection_score"],
            )

        if optimizer_step % checkpoint_interval == 0 or optimizer_step == max_iterations:
            selected_id = (
                str(sorted(best_records.values(), key=_candidate_sort_key)[0]["candidate_id"])
                if best_records
                else None
            )
            atomic_torch_save(
                _probe_checkpoint(
                    heads=heads,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    candidates=resolved_candidates,
                    base_parameters=parameters,
                    classes=classes,
                    source_provenance=source,
                    runtime_plan=runtime_plan,
                    training_data_state=_capture_training_data_state(train_loader),
                    optimizer_step=optimizer_step,
                    epoch=epoch,
                    batch_in_epoch=batch_in_epoch,
                    history=history,
                    best_records=best_records,
                    best_head_states=best_head_states,
                    selected_candidate_id=selected_id,
                ),
                last_path,
            )
            save_csv(output_dir / "training_history.csv", history)

    selected = sorted(best_records.values(), key=_candidate_sort_key)[0]
    selected_id = str(selected["candidate_id"])
    total_runtime_seconds = time.time() - start_time
    _finalize_candidate_artifacts(
        output_dir=output_dir,
        candidates=resolved_candidates,
        classes=classes,
        source_provenance=source,
        runtime_plan=runtime_plan,
        seed=seed,
        optimizer_steps_completed=max_iterations,
        best_records=best_records,
        best_head_states=best_head_states,
        class_weight_values=weights.detach().cpu().tolist(),
        total_runtime_seconds=total_runtime_seconds,
    )
    state = {
        "schema_version": SCHEMA_VERSION,
        "model_name": "3dino_linear_probe",
        "model_family": MODEL_FAMILY,
        "method": "linear_probe",
        "completed": True,
        "encoder_frozen": True,
        "stochastic_batch_encoder_passes": True,
        "shared_encoder_pass_per_candidate_bank": True,
        "optimizer_steps_completed": max_iterations,
        "selected_candidate_id": selected_id,
        "selected_candidate": selected["candidate"],
        "best_selection_score": selected["metrics"]["selection_score"],
        "candidate_count": len(resolved_candidates),
        "candidate_best_records": best_records,
        "parameters": parameters,
        "parameters_fingerprint": parameter_fingerprint(parameters),
        "runtime_plan": runtime_plan,
        "runtime_plan_fingerprint": _runtime_plan_fingerprint(
            parameters, runtime_plan
        ),
        "resume_semantics": "exact_or_fail_closed",
        "exact_resume_supported": bool(
            getattr(train_loader, "_threedino_exact_resume_supported", False)
        ),
        "operational_loader": _training_loader_operational_state(train_loader),
        "classes": classes,
        "class_weights": weights.detach().cpu().tolist(),
        "source_provenance": source,
        "seed": int(seed),
        "total_runtime_seconds": total_runtime_seconds,
        "completed_at_utc": utc_now(),
    }
    save_json(output_dir / "training_state.json", state)
    summary = {
        "model_name": "3dino_linear_probe",
        "model_family": MODEL_FAMILY,
        "method": "linear_probe",
        "selected_candidate_id": selected_id,
        "selected_candidate": selected["candidate"],
        "validation_metrics": selected["metrics"],
        "candidate_best_records": best_records,
        "source_provenance": source,
    }
    save_json(output_dir / "metrics_summary.json", summary)
    result = {
        "state": state,
        "candidates": best_records,
        "best_candidate_id": selected_id,
        "metrics": selected["metrics"],
        "checkpoint": str(best_path),
    }
    close_logger(logger)
    return result


def train_linear_probe(
    train_rows: list[dict],
    validation_rows: list[dict],
    parameters: dict,
    output_dir: Path,
    **kwargs,
) -> dict:
    """Train one resolved linear-probe candidate using the shared grid engine."""

    candidate = {
        key: parameters[key]
        for key in (
            "name",
            "nominal_lr",
            "lr",
            "learning_rate_scale_rule",
            "n_last_blocks",
            "use_mean_patch_token",
            "feature_dim",
        )
        if key in parameters
    }
    if "lr" not in candidate and "nominal_lr" not in candidate:
        raise ValueError("Resolved linear-probe parameters require lr or nominal_lr.")
    return train_linear_probe_grid(
        train_rows,
        validation_rows,
        parameters,
        [candidate],
        output_dir,
        **kwargs,
    )


def _set_gradient_checkpointing(model: nn.Module, enabled: bool) -> bool:
    if not enabled:
        return False
    try:
        from .model import set_gradient_checkpointing
    except ImportError:  # pragma: no cover
        try:
            from model import set_gradient_checkpointing
        except ImportError:
            set_gradient_checkpointing = None
    if set_gradient_checkpointing is not None:
        activated = set_gradient_checkpointing(model, True)
        if activated:
            return True
    for target in (model, getattr(model, "encoder", None)):
        if target is None:
            continue
        method = getattr(target, "gradient_checkpointing_enable", None)
        if callable(method):
            method()
            return True
        method = getattr(target, "set_grad_checkpointing", None)
        if callable(method):
            _call_supported(method, enable=True, enabled=True)
            return True
    raise RuntimeError(
        "gradient_checkpointing=true but the loaded 3DINO model exposes no "
        "supported activation-checkpointing API."
    )


def _build_full_model(
    *,
    parameters: dict,
    classes: list[str],
    device: torch.device,
    upstream_root,
    upstream_revision,
    checkpoint_path,
    expected_checkpoint_sha256,
    encoder_factory: Callable | None,
    model_factory: Callable | None,
) -> nn.Module:
    factory = model_factory or _default_model_factory
    factory_parameters = {
        **parameters,
        "upstream_root": upstream_root,
        "upstream_revision": upstream_revision,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": expected_checkpoint_sha256,
    }
    result = _call_supported(
        factory,
        variant="full_finetune",
        num_classes=len(classes),
        parameters=factory_parameters,
        device=device,
        encoder_factory=encoder_factory,
        upstream_root=upstream_root,
        upstream_revision=upstream_revision,
        checkpoint_path=checkpoint_path,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
    )
    model = result[0] if isinstance(result, tuple) else result
    if not isinstance(model, nn.Module):
        raise TypeError("model_factory must return an nn.Module or (nn.Module, details).")
    model = model.to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    encoder = getattr(model, "encoder", None)
    if encoder is None or not any(parameter.requires_grad for parameter in encoder.parameters()):
        raise RuntimeError("Full fine-tuning requires trainable encoder parameters.")
    return model


def full_finetune_optimizer(model: nn.Module, parameters: Mapping):
    if str(parameters.get("optimizer", "adamw")).lower() != "adamw":
        raise ValueError("Project-adapted 3DINO full fine-tuning requires AdamW.")
    encoder = getattr(model, "encoder", None)
    classifier = getattr(model, "classifier", None)
    if encoder is None or classifier is None:
        raise AttributeError("3DINO classifier must expose encoder and classifier modules.")
    encoder_parameters = [value for value in encoder.parameters() if value.requires_grad]
    head_parameters = [value for value in classifier.parameters() if value.requires_grad]
    if not encoder_parameters or not head_parameters:
        raise RuntimeError("Both encoder and classifier require trainable parameters.")
    return torch.optim.AdamW(
        [
            {
                "params": encoder_parameters,
                "lr": float(parameters.get("encoder_lr", 1e-6)),
                "name": "encoder",
            },
            {
                "params": head_parameters,
                "lr": float(parameters.get("head_lr", 1e-4)),
                "name": "classifier",
            },
        ],
        weight_decay=float(parameters.get("weight_decay", 0.01)),
        betas=(0.9, 0.999),
    )


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    classes: list[str],
    criterion,
    device: torch.device,
    *,
    mixed_precision: bool = False,
    autocast_dtype: str = "auto",
) -> dict:
    model.eval()
    logits_values: list[torch.Tensor] = []
    labels_values: list[torch.Tensor] = []
    patient_ids: list[str] = []
    label_strings: list[str] = []
    loss_sum = 0.0
    sample_count = 0
    for batch in loader:
        images, labels, batch_ids, batch_labels = _batch_values(batch)
        images = images.to(device, non_blocking=True, dtype=torch.float32)
        labels = labels.to(device, non_blocking=True)
        with _autocast(device, mixed_precision, autocast_dtype):
            logits = model(images)
        loss_sum += float(criterion(logits.float(), labels).item()) * int(labels.numel())
        sample_count += int(labels.numel())
        logits_values.append(logits.detach().float().cpu())
        labels_values.append(labels.detach().cpu())
        patient_ids.extend(batch_ids)
        label_strings.extend(batch_labels)
    if sample_count == 0:
        raise ValueError("Evaluation loader yielded no samples.")
    return _candidate_evaluation(
        torch.cat(logits_values),
        torch.cat(labels_values).numpy(),
        patient_ids,
        label_strings,
        classes,
        loss_sum / sample_count,
    )


def _build_lora_model(**kwargs):
    factory = kwargs.get("model_factory")
    if factory is None:
        # The legacy default factory constructs a fully unfrozen classifier
        # and has a different signature. Use the production LoRA-aware factory.
        try:
            from .model import build_model as factory
        except ImportError:
            from model import build_model as factory
    parameters = {
        **kwargs["parameters"],
        "upstream_root": kwargs.get("upstream_root"),
        "upstream_revision": kwargs.get("upstream_revision"),
        "checkpoint_path": kwargs.get("checkpoint_path"),
        "checkpoint_sha256": kwargs.get("expected_checkpoint_sha256"),
    }
    result = _call_supported(factory, variant="lora", num_classes=len(kwargs["classes"]),
                             parameters=parameters, device=kwargs["device"])
    model = result[0] if isinstance(result, tuple) else result
    try:
        from .lora import ThreeDINOLoRAClassifier
    except ImportError:
        from lora import ThreeDINOLoRAClassifier
    if not isinstance(model, ThreeDINOLoRAClassifier):
        raise TypeError("LoRA factory must return ThreeDINOLoRAClassifier.")
    model.to(kwargs["device"])
    model.assert_trainable_scope()
    return model


def _load_adapted_weights(model, payload, method):
    if method == "lora":
        if payload.get("checkpoint_scope") != "lora_and_classifier":
            raise RuntimeError("LoRA checkpoint has an incompatible scope.")
        if payload.get("lora_target_modules") != list(model.target_modules):
            raise RuntimeError("LoRA checkpoint targets differ from the loaded encoder.")
        model.load_adaptation_state_dict(payload["model_state_dict"])
    else:
        model.load_state_dict(payload["model_state_dict"], strict=True)


def _full_checkpoint(
    *,
    model: nn.Module,
    optimizer,
    scheduler,
    scaler,
    parameters: dict,
    classes: list[str],
    source_provenance: dict,
    runtime_plan: dict,
    training_data_state: dict,
    batches_remaining_in_epoch: int,
    optimizer_step: int,
    epoch: int,
    history: list[dict],
    best_step: int | None,
    best_score: float,
    stale_evaluations: int,
    validation_metrics: dict | None,
) -> dict:
    method = runtime_plan["method"]
    return {
        "schema_version": SCHEMA_VERSION,
        "model_name": f"3dino_{method}",
        "model_family": MODEL_FAMILY,
        "method": method,
        "variant": method,
        "checkpoint_scope": "lora_and_classifier" if method == "lora" else "complete_adapted_model",
        "pretrained_encoder_weights_frozen": method == "lora",
        "lora_target_modules": list(model.target_modules) if method == "lora" else None,
        "encoder_frozen": False,
        "parameters": parameters,
        "parameters_fingerprint": parameter_fingerprint(parameters),
        "runtime_plan": runtime_plan,
        "runtime_plan_fingerprint": _runtime_plan_fingerprint(
            parameters, runtime_plan
        ),
        "classes": classes,
        "source_provenance": source_provenance,
        "model_state_dict": model.adaptation_state_dict() if method == "lora" else model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "optimizer_step": int(optimizer_step),
        "epoch": int(epoch),
        "best_optimizer_step": best_step,
        "best_selection_score": float(best_score),
        "stale_evaluations": int(stale_evaluations),
        "validation_metrics": validation_metrics,
        "history": history,
        "rng_state": _rng_state(),
        "training_data_state": training_data_state,
        "batches_remaining_in_epoch": int(batches_remaining_in_epoch),
        "resume_semantics": "exact_or_fail_closed",
        "saved_at_utc": utc_now(),
    }


def train_full_finetune(*args, **kwargs):
    """Run the existing explicitly authorized full fine-tuning workflow."""
    return _train_encoder_adaptation(*args, method="full_finetune", **kwargs)


def train_lora(*args, **kwargs):
    """Run attention adapters plus the study MLP, with pretrained weights frozen."""
    return _train_encoder_adaptation(*args, method="lora", **kwargs)


def _train_encoder_adaptation(
    train_rows: list[dict],
    validation_rows: list[dict],
    parameters: dict,
    output_dir: Path,
    *,
    method: str,
    device: torch.device,
    upstream_root: str | Path | None = None,
    upstream_revision: str | None = None,
    checkpoint_path: str | Path | None = None,
    expected_checkpoint_sha256: str | None = None,
    cache_dir: str | Path | None = None,
    classes: list[str] | None = None,
    seed: int = 42,
    budget: int | None = None,
    num_workers: int = 0,
    resume: bool = False,
    encoder_factory: Callable | None = None,
    model_factory: Callable | None = None,
    dataset_factory: Callable | None = None,
    asset_provisioning: Mapping | None = None,
) -> dict:
    """Shared data/resume/evaluation engine with separate model/optimizer policies."""

    parameters = dict(parameters)
    if method not in {"lora", "full_finetune"}:
        raise ValueError(f"Unsupported encoder adaptation: {method}")
    is_lora = method == "lora"
    if is_lora:
        try:
            from .lora import lora_optimizer, validate_parameters
        except ImportError:
            from lora import lora_optimizer, validate_parameters
        validate_parameters(parameters)
    if not is_lora and not bool(
        parameters.get("allow_experimental_full_finetune", False)
        or parameters.get("allow_full_finetune", False)
    ):
        raise PermissionError(
            "Full fine-tuning is project-adapted, not publication-backed, and is "
            "disabled by default. Set allow_experimental_full_finetune=true only "
            "after the documented licence and memory preflight."
        )
    if not validation_rows:
        raise ValueError("Encoder adaptation requires development validation rows.")
    classes = list(classes or DEFAULT_CLASSES)
    if len(classes) != 5:
        raise ValueError("3DINO experiments require the canonical five classes.")
    output_dir = Path(output_dir)
    if is_lora and not resume and any((output_dir / name).exists() for name in
                                    ("last_checkpoint.pt", "best_model.pt", "training_state.json")):
        raise FileExistsError("LoRA output already contains a run; use resume or a new directory.")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir)
    _log_asset_provisioning(logger, asset_provisioning)
    seed_everything(seed)
    source = _source_provenance(
        upstream_root,
        upstream_revision,
        checkpoint_path,
        expected_checkpoint_sha256,
        injected=model_factory is not None or encoder_factory is not None,
    )
    builder = _build_lora_model if is_lora else _build_full_model
    model = builder(
        parameters=parameters,
        classes=classes,
        device=device,
        upstream_root=upstream_root,
        upstream_revision=upstream_revision,
        checkpoint_path=checkpoint_path,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        encoder_factory=encoder_factory,
        model_factory=model_factory,
    )
    checkpointing_active = _set_gradient_checkpointing(
        model, bool(parameters.get("gradient_checkpointing", False))
    )
    batch_size = int(parameters.get("batch_size", 4 if is_lora else 1))
    accumulation = max(1, int(parameters.get("grad_accum_steps", 1)))
    train_loader = _make_loader(
        train_rows,
        split="train",
        batch_size=batch_size,
        cache_dir=cache_dir,
        seed=seed,
        num_workers=num_workers,
        dataset_factory=dataset_factory,
        drop_last=True if is_lora else False,
    )
    validation_loader = _make_loader(
        validation_rows,
        split="validation",
        batch_size=batch_size,
        cache_dir=cache_dir,
        seed=seed,
        num_workers=0,
        dataset_factory=dataset_factory,
    )
    updates_per_epoch = max(1, math.ceil(len(train_loader) / accumulation))
    configured_max = int(
        parameters.get("max_iterations", parameters.get("max_optimizer_steps", 0))
    )
    max_iterations = int(budget) if budget is not None else configured_max
    if max_iterations <= 0:
        max_iterations = int(parameters.get("epochs", 50 if is_lora else 20)) * updates_per_epoch
    evaluation_interval = int(
        parameters.get("evaluation_interval_iterations", updates_per_epoch)
    )
    if evaluation_interval <= 0:
        evaluation_interval = updates_per_epoch
    warmup_fraction = float(parameters.get("warmup_fraction", 0.0 if is_lora else 0.05))
    mixed_precision = bool(parameters.get("mixed_precision", True))
    autocast_dtype = str(parameters.get("autocast_dtype", "float16" if is_lora else "auto")).lower()
    runtime_plan = {
        "method": method,
        "seed": int(seed),
        "max_iterations": max_iterations,
        "scheduler_total_updates": max_iterations,
        "scheduler_warmup_updates": int(max_iterations * warmup_fraction),
        "evaluation_interval_iterations": evaluation_interval,
        "checkpoint_interval_iterations": evaluation_interval,
        "batch_size": batch_size,
        "grad_accum_steps": accumulation,
        "updates_per_epoch": updates_per_epoch,
        "data_sequence_schema": DATA_SEQUENCE_SCHEMA,
        "augmentation_seed_schema": AUGMENTATION_SEED_SCHEMA,
        "preprocessing_contract_fingerprint": _canonical_sha256(
            preprocessing_contract()
        ),
        "validation_num_workers": 0,
        "test_num_workers": 0,
        "training_patient_count": len(train_rows),
        "training_rows_fingerprint": _training_rows_fingerprint(train_rows),
        "mixed_precision": mixed_precision,
        "autocast_dtype": autocast_dtype,
        "device_type": device.type,
        "device_identity": _device_identity(device),
    }
    weights = class_weights(
        train_rows,
        len(classes),
        str(parameters.get("class_weighting", "none" if is_lora else "effective_number")),
        float(parameters.get("effective_number_beta", 0.999)),
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = lora_optimizer(model, parameters) if is_lora else full_finetune_optimizer(model, parameters)
    # Constant LR preserves the study recipe while retaining the shared exact
    # resume contract's step counter. It performs no warm-up or decay.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0) if is_lora else _cosine_scheduler(
        optimizer,
        max_iterations,
        warmup_fraction=warmup_fraction,
    )
    scaler = _grad_scaler(device, mixed_precision, autocast_dtype)
    best_score = -math.inf
    best_step: int | None = None
    stale_evaluations = 0
    history: list[dict] = []
    start_step = 0
    epoch = 0
    last_path = output_dir / "last_checkpoint.pt"
    best_path = output_dir / "best_model.pt"
    restored_mid_epoch = False
    batches_remaining_in_epoch = len(train_loader)
    if resume and last_path.is_file():
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        if state.get("method") != method:
            raise RuntimeError("Refusing to resume from another training method.")
        start_step = _validate_resume_runtime(
            state, parameters=parameters, runtime_plan=runtime_plan
        )
        if state.get("source_provenance") != source:
            raise RuntimeError("Refusing to resume with changed checkpoint provenance.")
        _load_adapted_weights(model, state, method)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        scaler.load_state_dict(state.get("scaler_state_dict", {}))
        epoch = int(state.get("epoch", 0))
        best_step = state.get("best_optimizer_step")
        best_score = float(state.get("best_selection_score", -math.inf))
        stale_evaluations = int(state.get("stale_evaluations", 0))
        history = list(state.get("history", []))
        _restore_rng_state(state.get("rng_state"))
        restored_mid_epoch = _restore_training_data_state(
            train_loader, state.get("training_data_state", {})
        )
        saved_data_epoch = int(
            state.get("training_data_state", {}).get(
                "last_committed_data_epoch", -1
            )
        )
        if epoch != saved_data_epoch:
            raise RuntimeError(
                "Checkpoint epoch provenance disagrees with its committed "
                "training-data cursor; refusing resume."
            )
        batches_remaining_in_epoch = int(
            state.get("batches_remaining_in_epoch", -1)
        )
        if not 0 <= batches_remaining_in_epoch <= len(train_loader):
            raise RuntimeError(
                "Checkpoint has an invalid in-epoch batch cursor; refusing resume."
            )
        expected_remaining = _remaining_finite_batches(
            state.get("training_data_state", {})
        )
        if expected_remaining != batches_remaining_in_epoch:
            raise RuntimeError(
                "Sampler position and saved remaining-batch count disagree; "
                "refusing resume."
            )
        if restored_mid_epoch != (batches_remaining_in_epoch > 0):
            raise RuntimeError(
                "Sampler cursor and in-epoch batch cursor disagree; refusing resume."
            )
        resume_event = train_loader._threedino_resume_events[-1]
        logger.info(
            "Resuming encoder adaptation at optimizer step %d | "
            "training_workers=%d->%d next_committed_draw=%d "
            "data_sequence_preserved=True",
            start_step,
            resume_event["previous_training_num_workers"],
            resume_event["current_training_num_workers"],
            resume_event["next_committed_draw"],
        )

    iterator = _iterator_after_data_restore(
        train_loader, restored_mid_epoch=restored_mid_epoch
    )
    if resume and last_path.is_file() and not restored_mid_epoch:
        batches_remaining_in_epoch = len(train_loader)
    optimizer.zero_grad(set_to_none=True)
    start_time = time.time()
    optimizer_step = start_step
    stop = False
    while optimizer_step < max_iterations and not stop:
        if batches_remaining_in_epoch <= 0:
            iterator = iter(train_loader)
            batches_remaining_in_epoch = len(train_loader)
        accumulation_divisor = min(accumulation, batches_remaining_in_epoch)
        micro_batches = 0
        consumed_batches: list[Any] = []
        accumulated_samples = 0
        accumulated_loss = 0.0
        model.train()
        while micro_batches < accumulation_divisor:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            images, labels, _, _ = _batch_values(batch)
            images = images.to(device, non_blocking=True, dtype=torch.float32)
            labels = labels.to(device, non_blocking=True)
            with _autocast(device, mixed_precision, autocast_dtype):
                logits = model(images)
                raw_loss = criterion(logits.float(), labels)
                loss = raw_loss / accumulation_divisor
            scaler.scale(loss).backward()
            accumulated_loss += float(raw_loss.detach().cpu()) * int(labels.numel())
            accumulated_samples += int(labels.numel())
            micro_batches += 1
            consumed_batches.append(batch)
        batches_remaining_in_epoch -= accumulation_divisor
        scaler.unscale_(optimizer)
        max_grad_norm = float(parameters.get("max_grad_norm", 0.0 if is_lora else 1.0))
        if max_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scale_before_update = float(scaler.get_scale())
        scaler.step(optimizer)
        scaler.update()
        scale_after_update = float(scaler.get_scale())
        optimizer.zero_grad(set_to_none=True)
        # The data/scaler transaction consumed these draws even if AMP skipped
        # the parameter update. Commit in the main process only after that
        # transaction completes; uncommitted prefetched work is replayable.
        _commit_training_batches(train_loader, consumed_batches)
        epoch = int(_training_data_position(train_loader)["last_committed_data_epoch"])
        update_skipped = (
            mixed_precision
            and device.type == "cuda"
            and scale_after_update < scale_before_update
        )
        if update_skipped:
            logger.warning("AMP overflow skipped an optimizer update.")
            continue
        scheduler.step()
        optimizer_step += 1

        should_evaluate = (
            optimizer_step % evaluation_interval == 0
            or optimizer_step == max_iterations
        )
        if not should_evaluate:
            continue
        evaluation = evaluate_model(
            model,
            validation_loader,
            classes,
            criterion,
            device,
            mixed_precision=mixed_precision,
            autocast_dtype=autocast_dtype,
        )
        score = float(evaluation["metrics"]["selection_score"])
        row = {
            "epoch": epoch,
            "optimizer_step": optimizer_step,
            "train_loss": accumulated_loss / max(1, accumulated_samples),
            "encoder_lr": float(optimizer.param_groups[0]["lr"]),
            "head_lr": float(optimizer.param_groups[1]["lr"]),
            "validation_loss": evaluation["metrics"]["loss"],
            "validation_f1_macro": evaluation["metrics"]["f1_macro"],
            "validation_balanced_accuracy": evaluation["metrics"][
                "balanced_accuracy"
            ],
            "validation_average_precision_macro": evaluation["metrics"][
                "average_precision_macro"
            ],
            "validation_selection_score": score,
        }
        # The LoRA study evaluates the final epoch, with no best-epoch search.
        # Intermediate diagnostics/checkpoints are retained for resumption only.
        improved = is_lora or score > best_score + float(parameters.get("min_delta", 0.0))
        row["is_best_step"] = improved
        if improved:
            best_score = score
            best_step = optimizer_step
            stale_evaluations = 0
        else:
            stale_evaluations += 1
        row["stale_evaluations"] = stale_evaluations
        history.append(row)
        checkpoint = _full_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            parameters=parameters,
            classes=classes,
            source_provenance=source,
            runtime_plan=runtime_plan,
            training_data_state=_capture_training_data_state(train_loader),
            batches_remaining_in_epoch=batches_remaining_in_epoch,
            optimizer_step=optimizer_step,
            epoch=epoch,
            history=history,
            best_step=best_step,
            best_score=best_score,
            stale_evaluations=stale_evaluations,
            validation_metrics=evaluation["metrics"],
        )
        if improved:
            atomic_torch_save(checkpoint, best_path)
            _save_evaluation(output_dir, "validation", evaluation)
        atomic_torch_save(checkpoint, last_path)
        save_csv(output_dir / "training_history.csv", history)
        logger.info(
            "Encoder adaptation step %d/%d | validation score=%.4f",
            optimizer_step,
            max_iterations,
            score,
        )
        patience = int(parameters.get("early_stopping_patience", 0))
        if patience > 0 and stale_evaluations >= patience:
            stop = True

    if not best_path.is_file():
        raise RuntimeError("Encoder adaptation completed without a validation checkpoint.")
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    _load_adapted_weights(model, best, method)
    final_evaluation = evaluate_model(
        model,
        validation_loader,
        classes,
        criterion,
        device,
        mixed_precision=mixed_precision,
        autocast_dtype=autocast_dtype,
    )
    _save_evaluation(output_dir, "validation", final_evaluation)
    state = {
        "schema_version": SCHEMA_VERSION,
        "model_name": f"3dino_{method}",
        "model_family": MODEL_FAMILY,
        "method": method,
        "completed": True,
        "publication_backed_classification_method": False,
        "project_adapted": True,
        "authorization_acknowledged": not is_lora,
        "pretrained_encoder_weights_frozen": is_lora,
        "checkpoint_selection": "final_budget" if is_lora else "best_validation_score",
        "encoder_frozen": False,
        "gradient_checkpointing_active": checkpointing_active,
        "optimizer_steps_completed": optimizer_step,
        "best_optimizer_step": best["best_optimizer_step"],
        "best_selection_score": best["best_selection_score"],
        "parameters": parameters,
        "parameters_fingerprint": parameter_fingerprint(parameters),
        "runtime_plan": runtime_plan,
        "runtime_plan_fingerprint": _runtime_plan_fingerprint(
            parameters, runtime_plan
        ),
        "resume_semantics": "exact_or_fail_closed",
        "exact_resume_supported": bool(
            getattr(train_loader, "_threedino_exact_resume_supported", False)
        ),
        "operational_loader": _training_loader_operational_state(train_loader),
        "classes": classes,
        "class_weights": weights.detach().cpu().tolist(),
        "source_provenance": source,
        "seed": int(seed),
        "total_runtime_seconds": time.time() - start_time,
        "completed_at_utc": utc_now(),
    }
    save_json(output_dir / "training_state.json", state)
    summary = {
        "model_name": f"3dino_{method}",
        "model_family": MODEL_FAMILY,
        "method": method,
        "project_adapted": True,
        "parameters": parameters,
        "best_optimizer_step": best["best_optimizer_step"],
        "best_selection_score": best["best_selection_score"],
        "validation_metrics": final_evaluation["metrics"],
        "source_provenance": source,
    }
    save_json(output_dir / "metrics_summary.json", summary)
    result = {
        "state": state,
        "metrics": final_evaluation["metrics"],
        "checkpoint": str(best_path),
    }
    close_logger(logger)
    return result


@torch.no_grad()
def _predict_probe_head(
    encoder,
    head: nn.Module,
    representation: tuple[int, bool],
    loader: DataLoader,
    device: torch.device,
    feature_bank_extractor: Callable | None,
    mixed_precision: bool,
    autocast_dtype: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    encoder.eval()
    head.eval()
    logits_values: list[torch.Tensor] = []
    label_values: list[torch.Tensor] = []
    for batch in loader:
        images, labels, _, _ = _batch_values(batch)
        images = images.to(device, non_blocking=True, dtype=torch.float32)
        with _autocast(device, mixed_precision, autocast_dtype):
            features = _extract_bank(
                encoder, images, [representation], feature_bank_extractor
            )[representation]
            logits = head(features)
        logits_values.append(logits.detach().float().cpu())
        label_values.append(labels.detach().cpu())
    if not logits_values:
        raise ValueError("Prediction loader yielded no samples.")
    return torch.cat(logits_values), torch.cat(label_values)


def predict_checkpoint(
    rows: list[dict],
    classifier_checkpoint_path: Path,
    *,
    method: str,
    device: torch.device,
    upstream_root: str | Path | None = None,
    upstream_revision: str | None = None,
    checkpoint_path: str | Path | None = None,
    expected_checkpoint_sha256: str | None = None,
    cache_dir: str | Path | None = None,
    classes: list[str] | None = None,
    batch_size: int | None = None,
    num_workers: int = 0,
    encoder_factory: Callable | None = None,
    model_factory: Callable | None = None,
    feature_bank_extractor: Callable | None = None,
    feature_dimension_fn: Callable[[int, bool], int] | None = None,
    dataset_factory: Callable | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return logits and labels without writing artifacts (for CV ensembles)."""

    payload = torch.load(
        Path(classifier_checkpoint_path), map_location="cpu", weights_only=False
    )
    method = str(method).lower()
    if payload.get("method") != method:
        raise RuntimeError(
            f"Checkpoint method={payload.get('method')!r} differs from requested {method!r}."
        )
    checkpoint_classes = list(payload["classes"])
    classes = list(classes or checkpoint_classes)
    if classes != checkpoint_classes:
        raise RuntimeError("Checkpoint class order differs from the requested class order.")
    parameters = dict(payload.get("parameters", payload.get("base_parameters", {})))
    source = _source_provenance(
        upstream_root,
        upstream_revision,
        checkpoint_path,
        expected_checkpoint_sha256,
        injected=encoder_factory is not None or model_factory is not None,
    )
    if payload.get("source_provenance") != source:
        raise RuntimeError("Evaluation source/checkpoint provenance differs from training.")
    effective_batch_size = int(batch_size or parameters.get("batch_size", 1))
    loader = _make_loader(
        rows,
        split="test",
        batch_size=effective_batch_size,
        cache_dir=cache_dir,
        seed=int(parameters.get("seed", 42)),
        num_workers=0,
        dataset_factory=dataset_factory,
    )
    if method == "linear_probe":
        candidate = payload.get("candidate")
        if candidate is None:
            selected_id = payload.get("selected_candidate_id")
            candidate = next(
                (
                    value
                    for value in payload.get("candidates", [])
                    if str(value.get("candidate_id")) == str(selected_id)
                ),
                None,
            )
        if candidate is None:
            raise RuntimeError("Linear-probe checkpoint lacks its selected candidate.")
        state_dict = payload.get("head_state_dict")
        if state_dict is None:
            state_dict = payload.get("candidate_head_state_dicts", {}).get(
                str(payload.get("selected_candidate_id"))
            )
        if state_dict is None:
            raise RuntimeError("Linear-probe checkpoint lacks selected head weights.")
        representation = _representation(candidate)
        dimension_fn = feature_dimension_fn or _default_feature_dimension
        head = nn.Linear(
            int(dimension_fn(*representation)), len(classes)
        ).to(device)
        head.load_state_dict(state_dict, strict=True)
        factory = encoder_factory or _default_encoder_factory
        encoder = _call_supported(
            factory,
            upstream_root=upstream_root,
            upstream_revision=upstream_revision,
            checkpoint_path=checkpoint_path,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
            device=device,
        )
        if hasattr(encoder, "to"):
            encoder = encoder.to(device)
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
        return _predict_probe_head(
            encoder,
            head,
            representation,
            loader,
            device,
            feature_bank_extractor,
            bool(parameters.get("mixed_precision", False)),
            str(parameters.get("autocast_dtype", "auto")),
        )
    if method in {"full_finetune", "lora"}:
        builder = _build_lora_model if method == "lora" else _build_full_model
        model = builder(
            parameters=parameters,
            classes=classes,
            device=device,
            upstream_root=upstream_root,
            upstream_revision=upstream_revision,
            checkpoint_path=checkpoint_path,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
            encoder_factory=encoder_factory,
            model_factory=model_factory,
        )
        _load_adapted_weights(model, payload, method)
        model.eval()
        logits_values: list[torch.Tensor] = []
        label_values: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in loader:
                images, labels, _, _ = _batch_values(batch)
                images = images.to(device, non_blocking=True, dtype=torch.float32)
                with _autocast(
                    device,
                    bool(parameters.get("mixed_precision", True)),
                    str(parameters.get("autocast_dtype", "auto")),
                ):
                    logits = model(images)
                logits_values.append(logits.detach().float().cpu())
                label_values.append(labels.detach().cpu())
        if not logits_values:
            raise ValueError("Prediction loader yielded no samples.")
        return torch.cat(logits_values), torch.cat(label_values)
    raise ValueError(f"Unknown 3DINO method: {method}")


def evaluate_checkpoint(
    rows: list[dict],
    classifier_checkpoint_path: Path,
    output_dir: Path,
    *,
    method: str,
    device: torch.device,
    prefix: str = "test",
    **kwargs,
) -> dict:
    logits, labels = predict_checkpoint(
        rows,
        classifier_checkpoint_path,
        method=method,
        device=device,
        **kwargs,
    )
    payload = torch.load(
        Path(classifier_checkpoint_path), map_location="cpu", weights_only=False
    )
    classes = list(payload["classes"])
    probabilities = torch.softmax(logits.float(), dim=1).numpy()
    truth = labels.numpy()
    metrics = compute_metrics(truth, probabilities.argmax(axis=1), probabilities, classes)
    metrics["selection_score"] = fold_selection_score(metrics)
    prediction_rows, frame = format_predictions(
        [str(value["patient_id"]) for value in rows],
        [str(value["label_str"]) for value in rows],
        probabilities,
        classes,
    )
    evaluation = {
        "metrics": metrics,
        "predictions": prediction_rows,
        "probabilities": frame,
    }
    _save_evaluation(Path(output_dir), prefix, evaluation)
    return metrics


def refit_model(
    rows: list[dict],
    parameters: dict,
    output_dir: Path,
    *,
    method: str,
    device: torch.device,
    refit_budget: int,
    **kwargs,
) -> dict:
    """Refit a fixed CV-selected configuration on all development patients.

    The training rows are supplied as the diagnostic evaluation partition only;
    there is no early stopping or further configuration selection.
    """

    if int(refit_budget) <= 0:
        raise ValueError("refit_budget must be positive.")
    refit_parameters = dict(parameters)
    refit_parameters["evaluation_interval_iterations"] = int(refit_budget)
    refit_parameters["early_stopping_patience"] = 0
    if method == "linear_probe":
        result = train_linear_probe(
            rows,
            rows,
            refit_parameters,
            output_dir,
            device=device,
            budget=int(refit_budget),
            **kwargs,
        )
    elif method in {"full_finetune", "lora"}:
        trainer = train_lora if method == "lora" else train_full_finetune
        result = trainer(
            rows,
            rows,
            refit_parameters,
            output_dir,
            device=device,
            budget=int(refit_budget),
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown 3DINO method: {method}")
    diagnostics = dict(result["metrics"])
    diagnostics.update(
        {
            "evaluation_partition": "full_development_training_set",
            "selection_use": "none_training_diagnostics_only",
        }
    )
    state = dict(result["state"])
    state.update(
        {
            "refit_full_development": True,
            "refit_budget_source": "median_best_budget_from_five_cv_folds",
            "refit_budget": int(refit_budget),
            "heldout_test_used": False,
        }
    )
    save_json(Path(output_dir) / "training_state.json", state)
    save_json(Path(output_dir) / "metrics_summary.json", diagnostics)
    return {
        "state": state,
        "metrics": diagnostics,
        "checkpoint": result["checkpoint"],
    }


def train_run(
    variant: str,
    train_rows: list[dict],
    validation_rows: list[dict] | None,
    classes: list[str],
    parameters: dict,
    runtime: dict,
    output_dir: Path,
    cache_dir: Path | None,
    parameter_provenance: dict | None = None,
    fixed_iterations: int | None = None,
) -> dict:
    """Merlin-style dispatcher retained for orchestration symmetry."""

    del parameter_provenance  # parameters already carry a deterministic fingerprint
    device = torch.device(runtime.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    common = {
        "device": device,
        "upstream_root": runtime.get("upstream_root"),
        "upstream_revision": runtime.get("upstream_revision"),
        "checkpoint_path": runtime.get("checkpoint_path"),
        "expected_checkpoint_sha256": runtime.get("expected_checkpoint_sha256"),
        "cache_dir": cache_dir,
        "classes": classes,
        "seed": int(runtime.get("seed", 42)),
        "budget": fixed_iterations,
        "num_workers": int(runtime.get("num_workers", 0)),
        "resume": bool(runtime.get("resume", False)),
    }
    if variant == "linear_probe":
        if not validation_rows:
            raise ValueError("train_run linear_probe requires validation rows.")
        candidates = parameters.get("candidates")
        if candidates:
            return train_linear_probe_grid(
                train_rows,
                validation_rows,
                parameters,
                list(candidates),
                output_dir,
                **common,
            )
        return train_linear_probe(
            train_rows, validation_rows, parameters, output_dir, **common
        )
    if variant in {"full_finetune", "lora"}:
        if not validation_rows:
            raise ValueError("train_run full_finetune requires validation rows.")
        trainer = train_lora if variant == "lora" else train_full_finetune
        return trainer(
            train_rows, validation_rows, parameters, output_dir, **common
        )
    raise ValueError(f"Unknown 3DINO variant: {variant}")
