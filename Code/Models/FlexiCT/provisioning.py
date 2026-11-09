"""Pinned FlexiCT source resolution and user-supplied checkpoint validation."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

try:
    from . import UPSTREAM_REPOSITORY, UPSTREAM_REVISION
except ImportError:  # pragma: no cover
    from __init__ import UPSTREAM_REPOSITORY, UPSTREAM_REVISION

try:
    from _linear_probe.common import ContractError, append_jsonl, sha256_file, utc_now
except ImportError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from _linear_probe.common import ContractError, append_jsonl, sha256_file, utc_now


def _git(*arguments: str, cwd: Path | None = None) -> str:
    process = subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode != 0:
        raise ContractError(process.stderr.strip() or "Git command failed")
    return process.stdout.strip()


def validate_upstream(root: str | Path, revision: str = UPSTREAM_REVISION) -> dict:
    path = Path(root).resolve()
    required = [path / "flexi_ct" / "models.py", path / "flexi_ct" / "flexi_ct_2d.py"]
    if not all(item.is_file() for item in required) or not (path / ".git").exists():
        raise ContractError(f"Not a complete official FlexiCT checkout: {path}")
    observed = _git("rev-parse", "HEAD", cwd=path)
    if observed.lower() != revision.lower():
        raise ContractError(
            f"FlexiCT source revision mismatch: expected {revision}, observed {observed}"
        )
    if _git("status", "--porcelain", cwd=path):
        raise ContractError(f"FlexiCT checkout contains local modifications: {path}")
    return {
        "repository": UPSTREAM_REPOSITORY,
        "revision": observed,
        "path": str(path),
        "acquisition": "existing_local_checkout",
    }


def ensure_upstream(
    root: str | Path,
    *,
    revision: str = UPSTREAM_REVISION,
    local_files_only: bool = False,
    provision_log: str | Path | None = None,
) -> dict:
    destination = Path(root).resolve()
    if destination.exists():
        return validate_upstream(destination, revision)
    if local_files_only:
        raise ContractError(f"FlexiCT source is absent in local-files-only mode: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="upstream_flexict_", dir=destination.parent))
    try:
        _git("clone", "--filter=blob:none", UPSTREAM_REPOSITORY, str(temporary))
        _git("checkout", "--detach", revision, cwd=temporary)
        os.replace(temporary, destination)
        identity = validate_upstream(destination, revision)
        identity["acquisition"] = "git_clone"
    except Exception:
        # The temporary directory is implementation-owned and contains no user data.
        import shutil

        shutil.rmtree(temporary, ignore_errors=True)
        raise
    if provision_log is not None:
        append_jsonl(
            provision_log,
            {"timestamp": utc_now(), "asset": "source", **identity},
        )
    return identity


def resolve_checkpoint(
    explicit: str | Path | None,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    value = explicit or os.environ.get("FLEXICT_CHECKPOINT") or os.environ.get(
        "FLEXICT_VLM_CHECKPOINT"
    )
    if not value:
        raise ContractError(
            "FlexiCT-3D-VLM checkpoint is required. Pass --checkpoint or set "
            "FLEXICT_VLM_CHECKPOINT after obtaining the official Google Drive release."
        )
    path = Path(value).expanduser().resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(path)
    digest = sha256_file(path)
    if expected_sha256 and digest.lower() != expected_sha256.lower():
        raise ContractError(
            f"FlexiCT checkpoint SHA-256 mismatch: expected {expected_sha256}, observed {digest}"
        )
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": digest,
        "source": "official FlexiCT Google Drive release supplied by user",
    }


__all__ = ["ensure_upstream", "resolve_checkpoint", "validate_upstream"]
