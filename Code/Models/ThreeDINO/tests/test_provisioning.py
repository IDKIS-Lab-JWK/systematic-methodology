"""Mocked automatic asset provisioning tests; no network or real Git is used."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SYSTEMATIC_ROOT = Path(__file__).resolve().parents[4]
if str(SYSTEMATIC_ROOT) not in sys.path:
    sys.path.insert(0, str(SYSTEMATIC_ROOT))

from Code.Models.ThreeDINO import (  # noqa: E402
    MODEL_REVISION,
    MODEL_SHA256,
    MODEL_SIZE_BYTES,
    UPSTREAM_REPOSITORY,
    UPSTREAM_REVISION,
)
from Code.Models.ThreeDINO import provisioning  # noqa: E402


REQUIRED_SOURCE_FILES = (
    "LICENSE",
    "dinov2/__init__.py",
    "dinov2/configs/__init__.py",
    "dinov2/configs/train/vit3d_highres.yaml",
    "dinov2/models/__init__.py",
    "dinov2/models/vision_transformer.py",
)


def make_source(root: Path) -> None:
    for relative in REQUIRED_SOURCE_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic\n", encoding="utf-8")


def make_checkpoint(root: Path, payload: bytes = b"synthetic-weights") -> tuple[Path, str, int]:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "3dino_vit_weights.pth"
    path.write_bytes(payload)
    return path, hashlib.sha256(payload).hexdigest(), len(payload)


def attested_git(*, head: str = UPSTREAM_REVISION, origin: str = UPSTREAM_REPOSITORY, status: str = ""):
    def run(arguments):
        command = list(arguments)
        if "remote" in command:
            return origin + "\n"
        if "rev-parse" in command:
            return head + "\n"
        if "status" in command:
            return status
        raise AssertionError(f"Unexpected mocked Git command: {command}")

    return run


class ProvisioningTests(unittest.TestCase):
    def _explicit_kwargs(self, root: Path) -> dict:
        source = root / "upstream"
        make_source(source)
        checkpoint, digest, size = make_checkpoint(root / "weights")
        return {
            "upstream_root": source,
            "checkpoint_path": checkpoint,
            "expected_sha256": digest,
            "expected_size_bytes": size,
            "log_path": root / "Results" / "3DINO" / "Setup" / "provision.jsonl",
        }

    def test_official_source_and_hub_revisions_are_independent_and_pinned(self):
        self.assertEqual(UPSTREAM_REVISION, "85bd4435c1b2ada41cd34cd15cad17c4d3c88d89")
        self.assertEqual(MODEL_REVISION, "8a00a2bb14becb3fbe955064837322a3217f7ac0")
        self.assertNotEqual(MODEL_REVISION, UPSTREAM_REVISION)
        self.assertEqual(MODEL_SHA256, "b68dde2e435648bdf72edae70f970b07e4a903a8122c5be9e360bdb076cf695c")
        self.assertEqual(MODEL_SIZE_BYTES, 1_417_113_974)

    def test_existing_attested_source_and_explicit_checkpoint_are_reused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            kwargs = self._explicit_kwargs(root)
            with (
                mock.patch.object(provisioning, "_git", side_effect=attested_git()) as git,
                mock.patch.object(provisioning, "_load_hf") as hub,
            ):
                assets = provisioning.provision_assets(**kwargs)

            self.assertFalse(assets.downloads_performed)
            self.assertTrue(assets.git_operations_performed)
            self.assertEqual(assets.upstream["acquisition"], "existing_local_checkout")
            self.assertTrue(assets.upstream["revision_claim_verified_from_scm"])
            self.assertEqual(assets.checkpoint["acquisition"], "explicit_local_override")
            self.assertEqual(assets.to_dict()["checkpoint_path"], str(assets.checkpoint_path))
            self.assertEqual(git.call_count, 3)
            hub.assert_not_called()
            events = [json.loads(line)["event"] for line in kwargs["log_path"].read_text(encoding="utf-8").splitlines()]
            self.assertEqual(events, ["provision_start", "source_reuse", "checkpoint_reuse", "provision_success"])

    def test_missing_source_is_cloned_at_exact_detached_commit_then_atomically_installed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "upstream"
            checkpoint, digest, size = make_checkpoint(root / "weights")
            commands: list[list[str]] = []

            def fake_git(arguments):
                command = list(arguments)
                commands.append(command)
                if command[0] == "clone":
                    make_source(Path(command[-1]))
                    return ""
                if "remote" in command:
                    return UPSTREAM_REPOSITORY + "\n"
                if "rev-parse" in command:
                    return UPSTREAM_REVISION + "\n"
                if "status" in command:
                    return ""
                if "checkout" in command:
                    return ""
                raise AssertionError(command)

            log = root / "setup.jsonl"
            with mock.patch.object(provisioning, "_git", side_effect=fake_git):
                assets = provisioning.provision_assets(
                    upstream_root=source,
                    checkpoint_path=checkpoint,
                    expected_sha256=digest,
                    expected_size_bytes=size,
                    log_path=log,
                )

            self.assertTrue(source.is_dir())
            self.assertTrue(assets.git_operations_performed)
            self.assertEqual(assets.upstream["acquisition"], "git_clone")
            self.assertEqual(
                commands[0][:5],
                ["clone", "--no-checkout", "--no-tags", UPSTREAM_REPOSITORY, commands[0][4]],
            )
            self.assertEqual(commands[1][-3:], ["checkout", "--detach", UPSTREAM_REVISION])
            self.assertFalse(any(".provision-" in path.name for path in root.iterdir()))

    def test_existing_incompatible_checkout_fails_without_clone_or_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            kwargs = self._explicit_kwargs(root)
            marker = kwargs["upstream_root"] / "LICENSE"
            before = marker.read_bytes()
            with mock.patch.object(
                provisioning,
                "_git",
                side_effect=attested_git(head="0" * 40),
            ) as git:
                with self.assertRaisesRegex(provisioning.ProvisioningError, "HEAD"):
                    provisioning.provision_assets(**kwargs)
            self.assertEqual(marker.read_bytes(), before)
            self.assertFalse(any(call.args[0][0] == "clone" for call in git.call_args_list))
            failure = json.loads(
                kwargs["log_path"].read_text(encoding="utf-8").splitlines()[-1]
            )
            self.assertEqual(failure["event"], "provision_failure")
            self.assertTrue(failure["git_operations_performed"])

    def test_untracked_source_is_rejected_but_generated_bytecode_is_allowed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            kwargs = self._explicit_kwargs(root)
            with mock.patch.object(
                provisioning,
                "_git",
                side_effect=attested_git(status="?? dinov2/evil.py\n"),
            ):
                with self.assertRaisesRegex(
                    provisioning.ProvisioningError, "untracked non-bytecode"
                ):
                    provisioning.provision_assets(**kwargs)

            with mock.patch.object(
                provisioning,
                "_git",
                side_effect=attested_git(
                    status="?? dinov2/__pycache__/runtime.cpython-310.pyc\n"
                ),
            ):
                assets = provisioning.provision_assets(**kwargs)
            self.assertEqual(
                assets.upstream["untracked_runtime_artifacts_ignored"], 1
            )

    def test_concurrent_atomic_install_reuses_attested_winner_and_cleans_temp(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "upstream"
            checkpoint, digest, size = make_checkpoint(root / "weights")

            def fake_git(arguments):
                command = list(arguments)
                if command[0] == "clone":
                    make_source(Path(command[-1]))
                    return ""
                if "checkout" in command or "status" in command:
                    return ""
                if "remote" in command:
                    return UPSTREAM_REPOSITORY + "\n"
                if "rev-parse" in command:
                    return UPSTREAM_REVISION + "\n"
                raise AssertionError(command)

            def lose_race(_source, target):
                make_source(Path(target))
                raise FileExistsError("synthetic concurrent winner")

            with (
                mock.patch.object(provisioning, "_git", side_effect=fake_git),
                mock.patch.object(provisioning.os, "rename", side_effect=lose_race),
            ):
                assets = provisioning.provision_assets(
                    upstream_root=destination,
                    checkpoint_path=checkpoint,
                    expected_sha256=digest,
                    expected_size_bytes=size,
                    log_path=root / "setup.jsonl",
                )

            self.assertEqual(assets.upstream["acquisition"], "concurrent_checkout_reused")
            self.assertTrue(assets.git_operations_performed)
            self.assertTrue(destination.is_dir())
            self.assertFalse(any(".provision-" in item.name for item in root.iterdir()))

    def test_explicit_missing_checkpoint_never_falls_back_to_hub(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "upstream"
            with (
                mock.patch.object(provisioning, "_git") as git,
                mock.patch.object(provisioning, "_load_hf") as hub,
            ):
                with self.assertRaisesRegex(provisioning.ProvisioningError, "fallback is disabled"):
                    provisioning.provision_assets(
                        upstream_root=source,
                        checkpoint_path=root / "missing.pth",
                        log_path=root / "setup.jsonl",
                    )
            git.assert_not_called()
            hub.assert_not_called()

    def test_hub_cache_hit_uses_exact_pinned_arguments_and_stored_credentials(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "upstream"
            make_source(source)
            checkpoint, digest, size = make_checkpoint(root / "hf-cache")
            download = mock.Mock(return_value=str(checkpoint))
            cached = mock.Mock(return_value=str(checkpoint))
            with (
                mock.patch.object(provisioning, "_git", side_effect=attested_git()),
                mock.patch.object(provisioning, "_load_hf", return_value=(download, cached, object())),
            ):
                assets = provisioning.provision_assets(
                    upstream_root=source,
                    checkpoint_path=None,
                    model_revision=MODEL_REVISION,
                    expected_sha256=digest,
                    expected_size_bytes=size,
                    log_path=root / "setup.jsonl",
                )
            download.assert_not_called()
            self.assertFalse(assets.downloads_performed)
            self.assertEqual(assets.checkpoint["acquisition"], "huggingface_cache")

    def test_hub_cache_miss_is_recorded_as_download(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "upstream"
            make_source(source)
            checkpoint, digest, size = make_checkpoint(root / "hf-cache")
            download = mock.Mock(return_value=str(checkpoint))
            with (
                mock.patch.object(provisioning, "_git", side_effect=attested_git()),
                mock.patch.object(
                    provisioning,
                    "_load_hf",
                    return_value=(download, mock.Mock(return_value=None), object()),
                ),
            ):
                assets = provisioning.provision_assets(
                    upstream_root=source,
                    expected_sha256=digest,
                    expected_size_bytes=size,
                    log_path=root / "setup.jsonl",
                )
            self.assertTrue(assets.downloads_performed)
            self.assertEqual(assets.checkpoint["acquisition"], "huggingface_download")
            self.assertIn("checkpoint_download", (root / "setup.jsonl").read_text(encoding="utf-8"))

    def test_offline_mode_never_supplies_or_reads_a_token(self):
        class LocalEntryNotFoundError(Exception):
            pass

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "upstream"
            make_source(source)
            download = mock.Mock(side_effect=LocalEntryNotFoundError("offline cache miss"))
            with (
                mock.patch.object(provisioning, "_git", side_effect=attested_git()),
                mock.patch.object(
                    provisioning,
                    "_load_hf",
                    return_value=(download, mock.Mock(return_value=None), object()),
                ),
            ):
                with self.assertRaisesRegex(provisioning.ProvisioningError, "offline/local-only"):
                    provisioning.provision_assets(
                        upstream_root=source,
                        local_files_only=True,
                        log_path=root / "setup.jsonl",
                    )
            self.assertIsNone(download.call_args.kwargs["token"])
            self.assertTrue(download.call_args.kwargs["local_files_only"])

    def test_missing_source_fails_before_git_in_offline_mode(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch.object(provisioning, "_git") as git:
                with self.assertRaisesRegex(provisioning.ProvisioningError, "offline/local-only"):
                    provisioning.provision_assets(
                        upstream_root=root / "missing-source",
                        local_files_only=True,
                        log_path=root / "setup.jsonl",
                    )
            git.assert_not_called()

    def test_checksum_and_size_mismatches_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            kwargs = self._explicit_kwargs(root)
            with mock.patch.object(provisioning, "_git", side_effect=attested_git()):
                with self.assertRaisesRegex(provisioning.ProvisioningError, "verification failed"):
                    provisioning.provision_assets(**{**kwargs, "expected_sha256": "0" * 64})
                with self.assertRaisesRegex(provisioning.ProvisioningError, "size mismatch"):
                    provisioning.provision_assets(**{**kwargs, "expected_size_bytes": 999})

    def test_gate_errors_and_jsonl_logs_redact_credentials(self):
        class GatedRepoError(Exception):
            pass

        secret = "hf_thismustneverappear12345"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "upstream"
            make_source(source)
            download = mock.Mock(
                side_effect=GatedRepoError(
                    f"403 Forbidden Authorization: Bearer {secret} token={secret}"
                )
            )
            log = root / "setup.jsonl"
            with (
                mock.patch.object(provisioning, "_git", side_effect=attested_git()),
                mock.patch.object(
                    provisioning,
                    "_load_hf",
                    return_value=(download, mock.Mock(return_value=None), object()),
                ),
            ):
                with self.assertRaises(provisioning.AssetAccessError) as caught:
                    provisioning.provision_assets(
                        upstream_root=source,
                        log_path=log,
                    )
            self.assertNotIn(secret, str(caught.exception))
            self.assertIsNone(caught.exception.__cause__)
            text = log.read_text(encoding="utf-8")
            self.assertNotIn(secret, text)
            self.assertIn("REDACTED", text)
            for line in text.splitlines():
                json.loads(line)

    def test_default_log_is_below_results_setup(self):
        with tempfile.TemporaryDirectory() as temporary:
            results = Path(temporary) / "Results" / "3DINO"
            self.assertEqual(
                provisioning.default_provision_log(results),
                results.resolve() / "Setup" / "asset_provisioning.jsonl",
            )


if __name__ == "__main__":
    unittest.main()
