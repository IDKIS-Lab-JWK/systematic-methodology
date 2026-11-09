"""Automatic, pinned provisioning of the external 3DINO source and weights.

The source checkout and Hugging Face checkpoint remain outside this package.
This module acquires only immutable revisions, validates their identities, and
records token-free JSONL setup events.  It never prompts for or reads a token;
``huggingface_hub`` uses its normal credential store.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

try:
    from . import (
        MODEL_ACCESS_URL,
        MODEL_FILENAME,
        MODEL_REPO_ID,
        MODEL_REVISION,
        MODEL_SHA256,
        MODEL_SIZE_BYTES,
        UPSTREAM_REPOSITORY,
        UPSTREAM_REVISION,
    )
    from .model import resolve_checkpoint, validate_upstream_checkout
except ImportError:  # pragma: no cover - direct invocation from this folder.
    from __init__ import (
        MODEL_ACCESS_URL,
        MODEL_FILENAME,
        MODEL_REPO_ID,
        MODEL_REVISION,
        MODEL_SHA256,
        MODEL_SIZE_BYTES,
        UPSTREAM_REPOSITORY,
        UPSTREAM_REVISION,
    )
    from model import resolve_checkpoint, validate_upstream_checkout


_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ProvisioningError(RuntimeError):
    """A pinned source or checkpoint could not be provisioned safely."""


class AssetAccessError(ProvisioningError):
    """The Hugging Face gate or authentication prevents checkpoint access."""


@dataclass(frozen=True)
class ProvisionedAssets:
    upstream_root: Path
    checkpoint_path: Path
    upstream: dict
    checkpoint: dict
    downloads_performed: bool
    git_operations_performed: bool
    log_path: Path

    def to_dict(self) -> dict:
        return {
            "upstream_root": str(self.upstream_root),
            "checkpoint_path": str(self.checkpoint_path),
            "upstream": dict(self.upstream),
            "checkpoint": dict(self.checkpoint),
            "downloads_performed": bool(self.downloads_performed),
            "git_operations_performed": bool(self.git_operations_performed),
            "log_path": str(self.log_path),
        }


def default_provision_log(results_root: str | Path) -> Path:
    """Return the append-only runtime setup log below a model results root."""

    return Path(results_root).expanduser().resolve(strict=False) / "Setup" / "asset_provisioning.jsonl"


def provision_assets(
    *,
    upstream_root: str | Path,
    log_path: str | Path,
    upstream_revision: str = UPSTREAM_REVISION,
    checkpoint_path: str | Path | None = None,
    repo_id: str = MODEL_REPO_ID,
    model_revision: str = MODEL_REVISION,
    expected_sha256: str = MODEL_SHA256,
    expected_size_bytes: int = MODEL_SIZE_BYTES,
    local_files_only: bool = False,
) -> ProvisionedAssets:
    """Resolve or acquire both immutable assets and return sealed provenance.

    An explicit ``checkpoint_path`` is a strict local override: if absent or
    invalid it fails and never falls back to Hugging Face.  With no override,
    the pinned file is resolved through the normal Hugging Face cache and is
    downloaded only when needed.  ``local_files_only`` disables both network
    cloning and Hub downloads.
    """

    destination = Path(upstream_root).expanduser().resolve(strict=False)
    event_log = Path(log_path).expanduser().resolve(strict=False)
    revision = _immutable_revision(upstream_revision, "upstream_revision")
    hub_revision = _immutable_revision(model_revision, "model_revision")
    expected_digest = str(expected_sha256).strip().lower()
    if not _SHA256.fullmatch(expected_digest):
        raise ValueError("expected_sha256 must be a 64-character lowercase hex digest")
    expected_size = int(expected_size_bytes)
    if expected_size <= 0:
        raise ValueError("expected_size_bytes must be positive")

    _append_event(
        event_log,
        "provision_start",
        upstream_root=str(destination),
        upstream_revision=revision,
        repo_id=str(repo_id),
        model_revision=hub_revision,
        checkpoint_override=checkpoint_path is not None,
        local_files_only=bool(local_files_only),
    )
    cloned = False
    git_used = False
    downloaded = False

    def mark_git_activity() -> None:
        nonlocal git_used
        git_used = True

    try:
        # A user-supplied path is a strict override. Reject a typo before a
        # missing source checkout can trigger any network or filesystem
        # acquisition work.
        if checkpoint_path is not None:
            explicit = Path(checkpoint_path).expanduser().resolve(strict=False)
            if not explicit.is_file():
                raise ProvisioningError(
                    "Explicit 3DINO checkpoint override does not exist; automatic "
                    f"fallback is disabled for explicit paths: {explicit}"
                )
        upstream, cloned = _resolve_upstream(
            destination,
            revision,
            local_files_only=bool(local_files_only),
            log_path=event_log,
            on_git_activity=mark_git_activity,
        )
        checkpoint, downloaded = _resolve_weights(
            checkpoint_path=checkpoint_path,
            repo_id=str(repo_id),
            revision=hub_revision,
            expected_sha256=expected_digest,
            expected_size_bytes=expected_size,
            local_files_only=bool(local_files_only),
            log_path=event_log,
        )
        result = ProvisionedAssets(
            upstream_root=destination,
            checkpoint_path=Path(checkpoint["path"]),
            upstream=upstream,
            checkpoint=checkpoint,
            downloads_performed=downloaded,
            git_operations_performed=git_used,
            log_path=event_log,
        )
        _append_event(
            event_log,
            "provision_success",
            upstream_acquisition=upstream["acquisition"],
            checkpoint_acquisition=checkpoint["acquisition"],
            upstream_root=str(result.upstream_root),
            checkpoint_path=str(result.checkpoint_path),
            checkpoint_sha256=checkpoint["sha256"],
            downloads_performed=downloaded,
            git_operations_performed=git_used,
        )
        return result
    except Exception as exc:
        safe_message = _sanitize(str(exc))
        _append_event(
            event_log,
            "provision_failure",
            error_type=type(exc).__name__,
            error=safe_message,
            downloads_performed=downloaded,
            git_operations_performed=git_used,
        )
        if isinstance(exc, (ProvisioningError, ValueError)):
            raise
        raise ProvisioningError(safe_message) from None


def _resolve_upstream(
    destination: Path,
    revision: str,
    *,
    local_files_only: bool,
    log_path: Path,
    on_git_activity: Callable[[], None],
) -> tuple[dict, bool]:
    if destination.exists():
        if not destination.is_dir():
            raise ProvisioningError(
                f"3DINO source destination exists but is not a directory: {destination}"
            )
        upstream = _attest_checkout(
            destination, revision, on_git_activity=on_git_activity
        )
        upstream["acquisition"] = "existing_local_checkout"
        _append_event(
            log_path,
            "source_reuse",
            path=str(destination),
            revision=revision,
            scm_attested=True,
        )
        return upstream, False
    if local_files_only:
        raise ProvisioningError(
            "3DINO source checkout is missing and offline/local-only mode forbids "
            f"Git acquisition: {destination}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / (
        f".{destination.name}.provision-{os.getpid()}-{uuid.uuid4().hex}"
    )
    _append_event(
        log_path,
        "source_clone_started",
        repository=UPSTREAM_REPOSITORY,
        revision=revision,
        destination=str(destination),
    )
    try:
        on_git_activity()
        _git(["clone", "--no-checkout", "--no-tags", UPSTREAM_REPOSITORY, str(temporary)])
        _git(["-C", str(temporary), "checkout", "--detach", revision])
        upstream = _attest_checkout(
            temporary, revision, on_git_activity=on_git_activity
        )
        try:
            os.rename(temporary, destination)
        except OSError as exc:
            # Another process may have atomically installed the same pinned
            # checkout while this clone was in progress. Never overwrite it.
            if not destination.is_dir():
                raise ProvisioningError(
                    f"Could not atomically install 3DINO source at {destination}: {_sanitize(str(exc))}"
                ) from None
            _safe_remove_temporary(temporary, destination.parent)
            upstream = _attest_checkout(
                destination, revision, on_git_activity=on_git_activity
            )
            upstream["acquisition"] = "concurrent_checkout_reused"
            _append_event(
                log_path,
                "source_reuse_after_race",
                path=str(destination),
                revision=revision,
                scm_attested=True,
            )
            return upstream, True
        upstream["root"] = str(destination)
        upstream["acquisition"] = "git_clone"
        _append_event(
            log_path,
            "source_clone_completed",
            path=str(destination),
            revision=revision,
            scm_attested=True,
        )
        return upstream, True
    except Exception:
        _safe_remove_temporary(temporary, destination.parent)
        raise


def _attest_checkout(
    root: Path, revision: str, *, on_git_activity: Callable[[], None]
) -> dict:
    structural = validate_upstream_checkout(root, revision)
    on_git_activity()
    origin = _git(["-C", str(root), "remote", "get-url", "origin"]).strip()
    if _normalise_repository(origin) != _normalise_repository(UPSTREAM_REPOSITORY):
        raise ProvisioningError(
            "Existing 3DINO checkout has an unexpected Git origin: "
            f"{_sanitize(origin)}"
        )
    head = _git(["-C", str(root), "rev-parse", "HEAD"]).strip().lower()
    if head != revision:
        raise ProvisioningError(
            f"Existing 3DINO checkout HEAD is {head}, expected pinned {revision}"
        )
    status_lines = _git(
        ["-C", str(root), "status", "--porcelain", "--untracked-files=all"]
    ).splitlines()
    unsafe_status: list[str] = []
    ignored_runtime_artifacts: list[str] = []
    for line in status_lines:
        if line.startswith("?? "):
            relative = line[3:].strip()
            normalized = relative.replace("\\", "/").strip('"')
            parts = [part for part in normalized.split("/") if part]
            if "__pycache__" in parts and normalized.lower().endswith(".pyc"):
                ignored_runtime_artifacts.append(normalized)
                continue
        unsafe_status.append(line)
    if unsafe_status:
        raise ProvisioningError(
            "Existing 3DINO checkout has modified tracked files or untracked "
            "non-bytecode content; refusing to use or overwrite it."
        )
    structural.update(
        {
            "root": str(root.resolve(strict=False)),
            "origin": UPSTREAM_REPOSITORY,
            "head_revision": head,
            "revision_claim_verified_from_scm": True,
            "tracked_tree_clean": True,
            "tracked_modifications_present": False,
            "modification_status": "clean_tracked_tree",
            "untracked_runtime_artifacts_ignored": len(ignored_runtime_artifacts),
            "untracked_content_policy": "only __pycache__/*.pyc allowed",
            "git_invoked": True,
            "validation_scope": (
                "required-file presence plus Git origin, exact detached/checked-out "
                "HEAD revision, clean tracked tree, and no untracked content "
                "except generated __pycache__ bytecode"
            ),
        }
    )
    return structural


def _resolve_weights(
    *,
    checkpoint_path: str | Path | None,
    repo_id: str,
    revision: str,
    expected_sha256: str,
    expected_size_bytes: int,
    local_files_only: bool,
    log_path: Path,
) -> tuple[dict, bool]:
    if checkpoint_path is not None:
        explicit = Path(checkpoint_path).expanduser().resolve(strict=False)
        if not explicit.is_file():
            raise ProvisioningError(
                "Explicit 3DINO checkpoint override does not exist; automatic "
                f"fallback is disabled for explicit paths: {explicit}"
            )
        info = _verify_checkpoint(explicit, expected_sha256, expected_size_bytes)
        checkpoint = _checkpoint_document(
            info, repo_id, revision, "explicit_local_override", False
        )
        _append_event(
            log_path,
            "checkpoint_reuse",
            acquisition=checkpoint["acquisition"],
            path=str(info.path),
            sha256=info.sha256,
            size_bytes=info.size_bytes,
        )
        return checkpoint, False

    hf_hub_download, try_to_load_from_cache, cached_no_exist = _load_hf()
    cached_before: str | None = None
    try:
        candidate = try_to_load_from_cache(
            repo_id=repo_id,
            filename=MODEL_FILENAME,
            revision=revision,
        )
        if candidate is not None and candidate is not cached_no_exist:
            candidate_path = Path(str(candidate)).resolve(strict=False)
            if candidate_path.is_file():
                cached_before = str(candidate_path)
    except Exception:
        # Cache discovery is an optimization only. hf_hub_download remains the
        # authoritative, lock-protected resolver.
        cached_before = None
    try:
        if cached_before is not None:
            # A full immutable revision plus a discovered local cache path is
            # sufficient. Verify its bytes directly and make cache reuse a
            # genuinely zero-network operation.
            resolved = cached_before
        else:
            resolved = hf_hub_download(
                repo_id=repo_id,
                filename=MODEL_FILENAME,
                revision=revision,
                local_files_only=local_files_only,
                token=None if local_files_only else True,
            )
    except Exception as exc:
        safe = _sanitize(str(exc))
        if _looks_like_access_error(exc):
            raise AssetAccessError(
                f"Access to gated {repo_id} is unavailable. Sign in at "
                f"{MODEL_ACCESS_URL}, accept its conditions with the account used "
                "by this environment, run `hf auth login`, and retry. "
                f"Hub response: {safe}"
            ) from None
        if local_files_only or _looks_like_offline_error(exc):
            raise ProvisioningError(
                "Pinned 3DINO weights are not available in the local Hugging "
                f"Face cache and offline/local-only resolution cannot fetch them: {safe}"
            ) from None
        raise ProvisioningError(
            f"Could not resolve {repo_id}@{revision}/{MODEL_FILENAME}: {safe}"
        ) from None

    resolved_path = Path(resolved).resolve(strict=False)
    info = _verify_checkpoint(resolved_path, expected_sha256, expected_size_bytes)
    downloaded = cached_before is None and not local_files_only
    acquisition = "huggingface_download" if downloaded else "huggingface_cache"
    checkpoint = _checkpoint_document(
        info, repo_id, revision, acquisition, downloaded
    )
    _append_event(
        log_path,
        "checkpoint_download" if downloaded else "checkpoint_cache_reuse",
        acquisition=acquisition,
        repo_id=repo_id,
        revision=revision,
        filename=MODEL_FILENAME,
        path=str(info.path),
        sha256=info.sha256,
        size_bytes=info.size_bytes,
    )
    return checkpoint, downloaded


def _verify_checkpoint(path: Path, expected_sha256: str, expected_size: int) -> Any:
    try:
        info = resolve_checkpoint(path, expected_sha256=expected_sha256)
    except Exception as exc:
        raise ProvisioningError(
            f"3DINO checkpoint verification failed for {path}: {_sanitize(str(exc))}"
        ) from None
    if int(info.size_bytes) != int(expected_size):
        raise ProvisioningError(
            f"3DINO checkpoint size mismatch for {path}: expected {expected_size}, "
            f"found {info.size_bytes}"
        )
    return info


def _checkpoint_document(
    info: Any,
    repo_id: str,
    revision: str,
    acquisition: str,
    downloaded: bool,
) -> dict:
    return {
        "path": str(info.path),
        "size_bytes": int(info.size_bytes),
        "sha256": str(info.sha256),
        "repo_id": repo_id,
        "revision": revision,
        "filename": MODEL_FILENAME,
        "acquisition": acquisition,
        "download_performed": bool(downloaded),
        "token_logged": False,
    }


def _load_hf() -> tuple[Callable[..., str], Callable[..., Any], Any]:
    try:
        from huggingface_hub import (
            _CACHED_NO_EXIST,
            hf_hub_download,
            try_to_load_from_cache,
        )
    except ImportError as exc:
        raise ProvisioningError(
            "Automatic 3DINO checkpoint provisioning requires huggingface-hub."
        ) from exc
    return hf_hub_download, try_to_load_from_cache, _CACHED_NO_EXIST


def _git(arguments: Sequence[str]) -> str:
    command = ["git", *map(str, arguments)]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError as exc:
        raise ProvisioningError(
            "Git is required to acquire or attest the pinned 3DINO source, but "
            "the `git` executable was not found."
        ) from exc
    if completed.returncode != 0:
        detail = _sanitize(completed.stderr.strip() or completed.stdout.strip())
        raise ProvisioningError(
            f"Git command failed with exit code {completed.returncode}: {detail}"
        )
    return completed.stdout


def _immutable_revision(value: str, name: str) -> str:
    revision = str(value).strip().lower()
    if not _COMMIT_SHA.fullmatch(revision):
        raise ValueError(f"{name} must be a full immutable 40-character commit SHA")
    return revision


def _normalise_repository(value: str) -> str:
    text = str(value).strip().lower().rstrip("/")
    if text.startswith("git@github.com:"):
        text = "https://github.com/" + text[len("git@github.com:") :]
    if text.endswith(".git"):
        text = text[:-4]
    return text


def _looks_like_access_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(
        marker in text
        for marker in (
            "gatedrepo",
            "gated repo",
            "cannot access gated",
            "access to model",
            "401 unauthorized",
            "403 forbidden",
            "localtokennotfound",
            "token is required",
        )
    )


def _looks_like_offline_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(
        marker in text
        for marker in (
            "localentrynotfound",
            "offline",
            "connection",
            "connecterror",
            "name resolution",
        )
    )


def _sanitize(value: str) -> str:
    text = str(value)
    text = re.sub(r"(?i)bearer\s+\S+", "Bearer [REDACTED]", text)
    text = re.sub(r"\bhf_[A-Za-z0-9_-]{6,}\b", "[REDACTED_HF_TOKEN]", text)
    text = re.sub(
        r"(?i)(token|authorization|password|secret)(\s*[=:]\s*)([^\s,;]+)",
        r"\1\2[REDACTED]",
        text,
    )
    text = re.sub(r"(?i)(https?://)[^/@\s]+:[^/@\s]+@", r"\1[REDACTED]@", text)
    return text


def _append_event(log_file: Path, event: str, **fields: Any) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    record = _safe_log_value({
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **fields,
    })
    # One O_APPEND write keeps each compact JSON record intact when independent
    # launchers provision concurrently; no secret-bearing environment is read.
    payload = (json.dumps(record, sort_keys=True, default=str) + "\n").encode("utf-8")
    descriptor = os.open(log_file, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, payload)
    finally:
        os.close(descriptor)


def _safe_log_value(value: Any) -> Any:
    """Recursively redact strings before any value reaches persistent logs."""

    if isinstance(value, str):
        return _sanitize(value)
    if isinstance(value, dict):
        return {str(key): _safe_log_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_log_value(item) for item in value]
    return value


def _safe_remove_temporary(path: Path, expected_parent: Path) -> None:
    if not path.exists():
        return
    resolved = path.resolve(strict=False)
    parent = expected_parent.resolve(strict=False)
    if resolved.parent != parent or ".provision-" not in resolved.name:
        raise ProvisioningError(
            f"Refusing to clean an unexpected provisioning path: {resolved}"
        )
    shutil.rmtree(resolved)


__all__ = [
    "AssetAccessError",
    "ProvisionedAssets",
    "ProvisioningError",
    "default_provision_log",
    "provision_assets",
]
