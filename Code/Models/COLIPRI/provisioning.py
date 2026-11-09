"""Pinned official COLIPRI source and checkpoint provisioning."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

try:
    from . import (
        MODEL_FILENAME,
        MODEL_REPO_ID,
        MODEL_REVISION,
        MODEL_SHA256,
        MODEL_SIZE_BYTES,
    )
except ImportError:  # pragma: no cover
    from __init__ import (
        MODEL_FILENAME,
        MODEL_REPO_ID,
        MODEL_REVISION,
        MODEL_SHA256,
        MODEL_SIZE_BYTES,
    )

try:
    from _linear_probe.common import ContractError, append_jsonl, sha256_file, utc_now
except ImportError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from _linear_probe.common import ContractError, append_jsonl, sha256_file, utc_now


SOURCE_URL = f"https://huggingface.co/{MODEL_REPO_ID}"


def _git(*arguments: str, cwd: Path | None = None, skip_lfs: bool = False) -> str:
    environment = os.environ.copy()
    if skip_lfs:
        environment["GIT_LFS_SKIP_SMUDGE"] = "1"
    process = subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode != 0:
        raise ContractError(process.stderr.strip() or "Git command failed")
    return process.stdout.strip()


def validate_upstream(root: str | Path, revision: str = MODEL_REVISION) -> dict:
    path = Path(root).resolve()
    required = [
        path / "src" / "colipri" / "model" / "multimodal.py",
        path / "src" / "colipri" / "processor.py",
        path / "src" / "colipri" / "configs" / "config.yaml",
    ]
    if not (path / ".git").exists() or not all(item.is_file() for item in required):
        raise ContractError(f"Not a complete official COLIPRI checkout: {path}")
    observed = _git("rev-parse", "HEAD", cwd=path)
    if observed.lower() != revision.lower():
        raise ContractError(
            f"COLIPRI source revision mismatch: expected {revision}, observed {observed}"
        )
    if _git("status", "--porcelain", cwd=path):
        raise ContractError(f"COLIPRI checkout contains local modifications: {path}")
    return {
        "repository": SOURCE_URL,
        "revision": observed,
        "path": str(path),
        "acquisition": "existing_local_checkout",
    }


def ensure_upstream(
    root: str | Path,
    *,
    revision: str = MODEL_REVISION,
    local_files_only: bool = False,
    provision_log: str | Path | None = None,
) -> dict:
    destination = Path(root).resolve()
    if destination.exists():
        return validate_upstream(destination, revision)
    if local_files_only:
        raise ContractError(f"COLIPRI source is absent in local-files-only mode: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="upstream_colipri_", dir=destination.parent))
    try:
        _git("clone", "--filter=blob:none", SOURCE_URL, str(temporary), skip_lfs=True)
        _git("checkout", "--detach", revision, cwd=temporary, skip_lfs=True)
        os.replace(temporary, destination)
        identity = validate_upstream(destination, revision)
        identity["acquisition"] = "git_clone_without_lfs_weights"
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    if provision_log is not None:
        append_jsonl(provision_log, {"timestamp": utc_now(), "asset": "source", **identity})
    return identity


def _validate_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if size != MODEL_SIZE_BYTES:
        raise ContractError(
            f"COLIPRI checkpoint size mismatch: expected {MODEL_SIZE_BYTES}, observed {size}"
        )
    digest = sha256_file(path)
    if digest.lower() != MODEL_SHA256.lower():
        raise ContractError(
            f"COLIPRI checkpoint SHA-256 mismatch: expected {MODEL_SHA256}, observed {digest}"
        )
    return {
        "path": str(path.resolve()),
        "bytes": size,
        "sha256": digest,
        "repo_id": MODEL_REPO_ID,
        "revision": MODEL_REVISION,
        "filename": MODEL_FILENAME,
    }


def resolve_checkpoint(
    explicit: str | Path | None,
    *,
    upstream_root: str | Path | None = None,
    local_files_only: bool = False,
    provision_log: str | Path | None = None,
) -> dict[str, Any]:
    if explicit is not None:
        explicit_path = Path(explicit).expanduser()
        if not explicit_path.is_file():
            raise FileNotFoundError(explicit_path)
        identity = _validate_checkpoint(explicit_path.resolve())
        identity["acquisition"] = "explicit_local_checkpoint"
        return identity
    if upstream_root is not None:
        candidate = Path(upstream_root) / MODEL_FILENAME
        if candidate.is_file():
            try:
                identity = _validate_checkpoint(candidate.resolve())
            except ContractError:
                # ``ensure_upstream`` deliberately clones with LFS smudging
                # disabled.  The resulting small Git-LFS pointer is source
                # provenance, not a usable checkpoint, so continue to the
                # pinned Hub cache/download below.
                with candidate.open("rb") as handle:
                    pointer_prefix = handle.read(128)
                if not pointer_prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
                    raise
            else:
                identity["acquisition"] = "existing_local_checkpoint"
                return identity
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("huggingface-hub is required to provision COLIPRI weights") from exc
    try:
        path = Path(
            hf_hub_download(
                repo_id=MODEL_REPO_ID,
                filename=MODEL_FILENAME,
                revision=MODEL_REVISION,
                local_files_only=local_files_only,
            )
        )
    except Exception as exc:
        mode = "local cache" if local_files_only else "Hugging Face Hub"
        raise ContractError(f"Could not resolve pinned COLIPRI checkpoint from {mode}") from exc
    identity = _validate_checkpoint(path)
    identity["acquisition"] = "huggingface_cache_or_download"
    if provision_log is not None:
        append_jsonl(
            provision_log,
            {
                "timestamp": utc_now(),
                "asset": "checkpoint",
                **{key: value for key, value in identity.items() if key != "path"},
                "path": str(path.resolve()),
            },
        )
    return identity


__all__ = ["ensure_upstream", "resolve_checkpoint", "validate_upstream"]
