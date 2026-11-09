"""Readiness-contract checks for the read-only 3DINO setup inspector."""

from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

SYSTEMATIC_ROOT = Path(__file__).resolve().parents[4]
if str(SYSTEMATIC_ROOT) not in sys.path:
    sys.path.insert(0, str(SYSTEMATIC_ROOT))

from Code.Models.ThreeDINO import check_setup  # noqa: E402


class SetupReadinessTests(unittest.TestCase):
    @staticmethod
    def _runtime_report(compatible: bool = True) -> dict:
        packages = {
            name: {
                "importable": compatible,
                "installed_version": "synthetic" if compatible else None,
                "version_compatible": compatible,
                "native_extension_loaded": compatible,
                "compatible": compatible,
            }
            for name in (
                set(check_setup.CORE_PACKAGE_POLICY)
                | set(check_setup.REQUIRED_PACKAGE_POLICY)
            )
        }
        return {"compatible": compatible, "packages": packages}

    def _inspect(
        self,
        cache_root: Path,
        *,
        source_scm_attested: bool = True,
        request_cache: bool = True,
    ) -> dict:
        checkpoint = mock.Mock()
        checkpoint.to_dict.return_value = {
            "path": "synthetic-checkpoint",
            "sha256": "0" * 64,
            "download_performed": False,
        }
        with (
            mock.patch.object(check_setup, "load_manifest", return_value={}),
            mock.patch.object(
                check_setup,
                "collect_required_rows",
                return_value=[{"patient_id": "synthetic"}],
            ),
            mock.patch.object(
                check_setup,
                "validate_upstream_checkout",
                return_value={
                    "status": "valid",
                    "git_invoked": source_scm_attested,
                    "revision_claim_verified_from_scm": source_scm_attested,
                },
            ),
            mock.patch.object(
                check_setup, "resolve_checkpoint", return_value=checkpoint
            ),
            mock.patch.object(
                check_setup,
                "inspect_runtime_environment",
                return_value=self._runtime_report(),
            ),
        ):
            return check_setup.inspect_setup(
                dataset_root=cache_root / "Dataset",
                percentage=100,
                upstream_root=cache_root / "upstream",
                upstream_revision="synthetic-revision",
                checkpoint_path=cache_root / "weights.pth",
                cache_root=cache_root if request_cache else None,
            )

    def test_missing_requested_cache_is_build_ready_but_not_model_ready(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = self._inspect(Path(temporary) / "Cache")
        self.assertEqual(report["cache"]["status"], "missing")
        self.assertEqual(report["status"], "failed_cache_missing")
        self.assertTrue(report["ready_for_cache_build"])
        self.assertFalse(report["ready_for_model_execution"])

    def test_invalid_requested_cache_is_never_model_ready(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache_root = Path(temporary) / "Cache"
            cache_dir = check_setup.cache_directory(
                cache_root, check_setup.DEFAULT_CACHE_NAMESPACE
            )
            cache_dir.mkdir(parents=True)
            (cache_dir / "cache_manifest.json").write_text("{}", encoding="utf-8")
            with mock.patch.object(
                check_setup,
                "validate_cache",
                side_effect=ValueError("synthetic invalid cache"),
            ):
                report = self._inspect(cache_root)
        self.assertEqual(report["cache"]["status"], "invalid")
        self.assertEqual(report["status"], "failed_cache_invalid")
        self.assertFalse(report["ready_for_model_execution"])

    def test_valid_requested_cache_allows_model_readiness(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache_root = Path(temporary) / "Cache"
            cache_dir = check_setup.cache_directory(
                cache_root, check_setup.DEFAULT_CACHE_NAMESPACE
            )
            cache_dir.mkdir(parents=True)
            (cache_dir / "cache_manifest.json").write_text("{}", encoding="utf-8")
            with mock.patch.object(
                check_setup,
                "validate_cache",
                return_value={"status": "valid", "artifacts_verified": True},
            ):
                report = self._inspect(cache_root)
        self.assertEqual(report["cache"]["status"], "valid")
        self.assertEqual(report["status"], "passed")
        self.assertTrue(report["ready_for_cache_build"])
        self.assertTrue(report["ready_for_model_execution"])

    def test_structural_only_cache_check_is_not_model_readiness(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache_root = Path(temporary) / "Cache"
            cache_dir = check_setup.cache_directory(
                cache_root, check_setup.DEFAULT_CACHE_NAMESPACE
            )
            cache_dir.mkdir(parents=True)
            (cache_dir / "cache_manifest.json").write_text("{}", encoding="utf-8")
            with mock.patch.object(
                check_setup,
                "validate_cache",
                return_value={"status": "valid", "artifacts_verified": False},
            ):
                report = self._inspect(cache_root)
        self.assertEqual(
            report["cache"]["status"],
            "structurally_valid_artifacts_unverified",
        )
        self.assertEqual(report["status"], "failed_cache_artifacts_unverified")
        self.assertFalse(report["ready_for_model_execution"])

    def test_structural_source_without_scm_attestation_is_not_model_readiness(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = self._inspect(
                Path(temporary),
                source_scm_attested=False,
                request_cache=False,
            )
        self.assertEqual(report["status"], "failed_source_scm_unattested")
        self.assertFalse(report["source_scm_attested"])
        self.assertTrue(report["ready_for_cache_build"])
        self.assertFalse(report["ready_for_model_execution"])

    def test_incompatible_runtime_skips_torch_bearing_model_adapter(self):
        upstream = mock.Mock()
        checkpoint = mock.Mock()
        with tempfile.TemporaryDirectory() as temporary:
            with (
                mock.patch.object(check_setup, "load_manifest", return_value={}),
                mock.patch.object(
                    check_setup, "collect_required_rows", return_value=[]
                ),
                mock.patch.object(
                    check_setup,
                    "inspect_runtime_environment",
                    return_value=self._runtime_report(False),
                ),
                mock.patch.object(
                    check_setup, "validate_upstream_checkout", upstream
                ),
                mock.patch.object(check_setup, "resolve_checkpoint", checkpoint),
            ):
                report = check_setup.inspect_setup(
                    dataset_root=Path(temporary) / "Dataset",
                    percentage=100,
                    upstream_root=Path(temporary) / "upstream",
                    upstream_revision="synthetic",
                    checkpoint_path=Path(temporary) / "weights.pth",
                    cache_root=None,
                )
        self.assertEqual(report["status"], "failed_runtime_compatibility")
        self.assertFalse(report["ready_for_cache_build"])
        self.assertFalse(report["ready_for_model_execution"])
        upstream.assert_not_called()
        checkpoint.assert_not_called()

    def test_auto_provisioning_is_reflected_in_token_safe_setup_provenance(self):
        checkpoint = mock.Mock()
        checkpoint.to_dict.return_value = {
            "path": "synthetic-checkpoint",
            "sha256": "f" * 64,
            "download_performed": True,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            assets = types.SimpleNamespace(
                upstream_root=root / "upstream",
                checkpoint_path=root / "hf-cache" / "3dino_vit_weights.pth",
                upstream={
                    "revision": check_setup.UPSTREAM_REVISION,
                    "revision_claim_verified_from_scm": True,
                    "modification_status": "clean_tracked_tree",
                },
                checkpoint={
                    "path": str(root / "hf-cache" / "3dino_vit_weights.pth"),
                    "repo_id": "AICONSlab/3DINO-ViT",
                    "revision": check_setup.MODEL_REVISION,
                    "sha256": "f" * 64,
                    "token_logged": False,
                },
                downloads_performed=True,
                git_operations_performed=True,
                to_dict=lambda: {
                    "upstream": {
                        "revision": check_setup.UPSTREAM_REVISION,
                        "revision_claim_verified_from_scm": True,
                    },
                    "checkpoint": {
                        "sha256": "f" * 64,
                        "token_logged": False,
                    },
                    "downloads_performed": True,
                    "git_operations_performed": True,
                },
            )
            with (
                mock.patch.object(check_setup, "load_manifest", return_value={}),
                mock.patch.object(
                    check_setup,
                    "collect_required_rows",
                    return_value=[{"patient_id": "synthetic"}],
                ),
                mock.patch.object(
                    check_setup,
                    "inspect_runtime_environment",
                    return_value=self._runtime_report(),
                ),
                mock.patch.object(
                    check_setup, "provision_assets", return_value=assets
                ) as provision,
                mock.patch.object(
                    check_setup, "validate_upstream_checkout"
                ) as structural,
                mock.patch.object(
                    check_setup, "resolve_checkpoint", return_value=checkpoint
                ),
            ):
                report = check_setup.inspect_setup(
                    dataset_root=root / "Dataset",
                    percentage=100,
                    upstream_root=root / "upstream",
                    upstream_revision=check_setup.UPSTREAM_REVISION,
                    checkpoint_path=None,
                    auto_provision=True,
                    provision_log=root / "Results" / "3DINO" / "Setup" / "assets.jsonl",
                    cache_root=None,
                )
        provision.assert_called_once()
        structural.assert_not_called()
        self.assertTrue(report["downloads_performed"])
        self.assertTrue(report["git_operations_performed"])
        self.assertTrue(report["upstream"]["revision_claim_verified_from_scm"])
        self.assertEqual(report["checkpoint"]["revision"], check_setup.MODEL_REVISION)
        self.assertFalse(
            report["asset_provisioning"]["checkpoint"]["token_logged"]
        )


class RuntimePolicyTests(unittest.TestCase):
    @staticmethod
    def _versions() -> dict[str, str]:
        return {
            "torch": "2.0.0+cu117",
            "torchvision": "0.15.0+cu117",
            "xformers": "0.0.18",
            "torchmetrics": "0.10.3",
            "monai": "1.3.0",
            "nibabel": "5.1.0",
            "omegaconf": "2.3.0",
            "fvcore": "0.1.5.post20221221",
            "iopath": "0.1.10",
            "submitit": "1.4.6",
            "torchio": "0.19.3",
            "numpy": "1.24.4",
            "pandas": "2.0.3",
            "scikit-learn": "1.3.2",
            "scipy": "1.10.1",
            "PyYAML": "6.0.1",
            "huggingface-hub": "0.30.2",
            "tqdm": "4.66.1",
        }

    def _inspect_runtime(
        self,
        *,
        python=(3, 10, 19),
        versions: dict[str, str] | None = None,
        xformers_native: bool = True,
        torch_native: bool = True,
        cuda_available: bool = False,
        device: str = "cpu",
        import_failure: str | None = None,
    ) -> dict:
        torch_module = types.SimpleNamespace(
            _C=object() if torch_native else None,
            cuda=types.SimpleNamespace(is_available=lambda: cuda_available),
            version=types.SimpleNamespace(cuda="11.7"),
        )
        modules = {
            name: types.SimpleNamespace()
            for name in (
                set(check_setup.CORE_PACKAGE_POLICY)
                | set(check_setup.REQUIRED_PACKAGE_POLICY)
            )
        }
        modules["torch"] = torch_module
        modules["xformers"] = types.SimpleNamespace(
            _has_cpp_library=xformers_native
        )

        def fake_import(name: str):
            if name == import_failure:
                raise OSError(f"synthetic native import failure: {name}")
            return modules[name]

        installed = versions or self._versions()
        return check_setup.inspect_runtime_environment(
            device=device,
            version_info=python,
            implementation="CPython",
            platform_name="win32",
            pointer_bits=64,
            import_module=fake_import,
            distribution_version=lambda name: installed[name],
        )

    def test_supported_cp310_windows_stack_accepts_cuda_local_suffixes(self):
        report = self._inspect_runtime()
        self.assertTrue(report["compatible"])
        self.assertTrue(report["packages"]["torch"]["native_extension_loaded"])
        self.assertTrue(
            report["packages"]["xformers"]["native_extension_loaded"]
        )

    def test_python_and_package_versions_are_fail_closed(self):
        self.assertFalse(self._inspect_runtime(python=(3, 9, 19))["compatible"])
        versions = self._versions()
        versions["torch"] = "2.1.0+cu121"
        report = self._inspect_runtime(versions=versions)
        self.assertFalse(report["compatible"])
        self.assertFalse(report["packages"]["torch"]["version_compatible"])

    def test_native_torch_and_xformers_failures_are_fatal(self):
        self.assertFalse(self._inspect_runtime(torch_native=False)["compatible"])
        self.assertFalse(self._inspect_runtime(xformers_native=False)["compatible"])
        failed_import = self._inspect_runtime(import_failure="xformers")
        self.assertFalse(failed_import["compatible"])
        self.assertEqual(
            failed_import["packages"]["xformers"]["import_error_type"], "OSError"
        )

    def test_cuda_request_requires_available_cuda_runtime(self):
        self.assertFalse(self._inspect_runtime(device="cuda")["device_compatible"])
        self.assertTrue(
            self._inspect_runtime(device="cuda", cuda_available=True)[
                "device_compatible"
            ]
        )

    def test_documentation_selects_explicit_cp310_compatibility_runtime(self):
        model_root = Path(check_setup.__file__).resolve().parent
        readme = (model_root / "README.md").read_text(encoding="utf-8")
        requirements = (model_root / "requirements.txt").read_text(
            encoding="utf-8"
        )
        self.assertIn("py -3.10 -m venv .venv_3dino", readme)
        self.assertIn("64-bit CPython 3.10.x on Windows", readme)
        self.assertIn("compatibility", readme.lower())
        self.assertNotIn("\npython -m venv .venv_3dino", readme)
        self.assertIn("64-bit CPython 3.10 only", requirements)


if __name__ == "__main__":
    unittest.main()
