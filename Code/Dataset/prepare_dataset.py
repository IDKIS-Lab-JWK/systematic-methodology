#!/usr/bin/env python
"""Prepare the unsplit, class-wise medical imaging dataset.

Two workflows are provided:

1. ``merge-existing`` (the workflow used now)
   Merge the existing 100% Training/Validation/Test DICOM, NIfTI, and PNG
   folders into ``Dataset/{DICOM,NIFTI,PNG}/<class>/...``.

2. ``build-from-source`` (the reproducible workflow for future use)
   Find every patient CT folder in an original class-wise dataset, copy all
   DICOM data without sampling or splitting, and create matching NIfTI and PNG
   outputs.

Dependencies for conversion:
    pip install SimpleITK pydicom numpy Pillow
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


SPLITS = ("Training", "Validation", "Test")
FORMATS = {
    "DICOM": "100_percent",
    "NIFTI": "100_percent_nifti",
    "PNG": "100_percent_png",
}


def copy_directory_contents(source: Path, destination: Path, overwrite: bool) -> None:
    """Copy a directory recursively while retaining metadata."""
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        target = destination / item.name
        if target.exists() and not overwrite:
            raise FileExistsError(
                f"Destination already exists: {target}. "
                "This may be a duplicate patient ID; use --overwrite only if intended."
            )
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=overwrite, copy_function=shutil.copy2)
        else:
            shutil.copy2(item, target)


def merge_existing(workspace: Path, dataset: Path, overwrite: bool = False) -> None:
    """Merge only Training/Validation/Test from the three existing 100% sets."""
    for format_name, source_name in FORMATS.items():
        source_root = workspace / source_name
        if not source_root.is_dir():
            raise FileNotFoundError(f"Missing source dataset: {source_root}")

        for split in SPLITS:
            split_dir = source_root / split
            if not split_dir.is_dir():
                raise FileNotFoundError(f"Missing split: {split_dir}")

            for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
                destination = dataset / format_name / class_dir.name
                print(f"{format_name}: {split}/{class_dir.name} -> {destination}", flush=True)
                copy_directory_contents(class_dir, destination, overwrite=overwrite)


def find_ct_folders(class_dir: Path) -> list[Path]:
    """Find populated CT folders using the layout/filter from the earlier copier."""
    found: dict[str, Path] = {}
    for image_group in class_dir.iterdir():
        if not image_group.is_dir():
            continue
        lowered = image_group.name.lower()
        if "imag" not in lowered and "dicom" not in lowered:
            continue

        for patient_dir in image_group.iterdir():
            if not patient_dir.is_dir():
                continue
            ct_dir = next(
                (child for child in patient_dir.iterdir()
                 if child.is_dir() and child.name.lower() == "ct"),
                None,
            )
            if ct_dir and any(path.is_file() for path in ct_dir.rglob("*")):
                if patient_dir.name in found:
                    raise RuntimeError(
                        f"Duplicate patient ID {patient_dir.name!r}: "
                        f"{found[patient_dir.name]} and {ct_dir}"
                    )
                found[patient_dir.name] = ct_dir
    return sorted(found.values(), key=lambda path: path.parent.name)


def dicom_series_files(patient_dir: Path) -> list[str]:
    import SimpleITK as sitk

    reader = sitk.ImageSeriesReader()
    candidates = [patient_dir]
    candidates.extend(sorted({path.parent for path in patient_dir.rglob("*") if path.is_file()}))
    best: list[str] = []
    for candidate in candidates:
        for series_id in reader.GetGDCMSeriesIDs(str(candidate)) or []:
            files = list(reader.GetGDCMSeriesFileNames(str(candidate), series_id))
            if len(files) > len(best):
                best = files
    return best


def convert_to_nifti(patient_dir: Path, output_file: Path, overwrite: bool) -> None:
    import SimpleITK as sitk

    if output_file.exists() and not overwrite:
        raise FileExistsError(f"Destination already exists: {output_file}")
    files = dicom_series_files(patient_dir)
    if not files:
        raise RuntimeError(f"No readable DICOM series found in {patient_dir}")
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(files)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(reader.Execute(), str(output_file), useCompression=True)


def sorted_dicom_files(patient_dir: Path) -> list[Path]:
    import pydicom

    candidates = [path for path in patient_dir.rglob("*") if path.is_file()]

    def sort_key(path: Path) -> tuple:
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
            position = getattr(ds, "ImagePositionPatient", None)
            if position is not None and len(position) >= 3:
                return (0, float(position[2]), path.name)
            instance = getattr(ds, "InstanceNumber", None)
            if instance is not None:
                return (1, int(instance), path.name)
            location = getattr(ds, "SliceLocation", None)
            if location is not None:
                return (2, float(location), path.name)
        except Exception:
            pass
        return (3, path.name)

    return sorted(candidates, key=sort_key)


def convert_to_png(
    patient_dir: Path,
    output_dir: Path,
    overwrite: bool,
    window_level: float,
    window_width: float,
) -> None:
    import numpy as np
    import pydicom
    from PIL import Image

    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Destination already contains files: {output_dir}")
    files = sorted_dicom_files(patient_dir)
    if not files:
        raise RuntimeError(f"No files found in {patient_dir}")

    low = window_level - window_width / 2
    high = window_level + window_width / 2
    slices = []
    for path in files:
        ds = pydicom.dcmread(str(path), force=True)
        hu = (
            ds.pixel_array.astype(np.float32) * float(getattr(ds, "RescaleSlope", 1.0))
            + float(getattr(ds, "RescaleIntercept", 0.0))
        )
        windowed = np.clip(hu, low, high)
        slices.append(((windowed - low) / (high - low + 1e-6) * 255).astype(np.uint8))

    output_dir.mkdir(parents=True, exist_ok=True)
    for index, center in enumerate(slices):
        rgb = np.stack(
            [slices[max(0, index - 1)], center, slices[min(len(slices) - 1, index + 1)]],
            axis=-1,
        )
        Image.fromarray(rgb).save(output_dir / f"{index:04d}.png")


def build_from_source(
    source: Path,
    dataset: Path,
    overwrite: bool = False,
    window_level: float = 40,
    window_width: float = 400,
) -> None:
    """Copy all source patients and convert each one; no sampling or splitting."""
    if not source.is_dir():
        raise FileNotFoundError(f"Source folder not found: {source}")

    for class_dir in sorted(path for path in source.iterdir() if path.is_dir()):
        patients = find_ct_folders(class_dir)
        print(f"{class_dir.name}: found {len(patients)} patients", flush=True)
        for index, ct_dir in enumerate(patients, 1):
            patient_id = ct_dir.parent.name
            dicom_destination = dataset / "DICOM" / class_dir.name / patient_id
            print(f"  [{index}/{len(patients)}] {patient_id}", flush=True)
            copy_directory_contents(ct_dir, dicom_destination, overwrite=overwrite)
            convert_to_nifti(
                dicom_destination,
                dataset / "NIFTI" / class_dir.name / f"{patient_id}.nii.gz",
                overwrite,
            )
            convert_to_png(
                dicom_destination,
                dataset / "PNG" / class_dir.name / patient_id,
                overwrite,
                window_level,
                window_width,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    merge = subparsers.add_parser("merge-existing")
    repository_root = Path(__file__).resolve().parents[2]
    dataset_root = repository_root / "Dataset"
    workspace_root = repository_root.parent

    merge.add_argument("--workspace", type=Path, default=workspace_root)
    merge.add_argument("--dataset", type=Path, default=dataset_root)
    merge.add_argument("--overwrite", action="store_true")

    build = subparsers.add_parser("build-from-source")
    build.add_argument("source", type=Path, help="Original class-wise DICOM data root")
    build.add_argument("--dataset", type=Path, default=dataset_root)
    build.add_argument("--overwrite", action="store_true")
    build.add_argument("--window-level", type=float, default=40)
    build.add_argument("--window-width", type=float, default=400)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "merge-existing":
        merge_existing(args.workspace.resolve(), args.dataset.resolve(), args.overwrite)
    else:
        build_from_source(
            args.source.resolve(),
            args.dataset.resolve(),
            args.overwrite,
            args.window_level,
            args.window_width,
        )


if __name__ == "__main__":
    main()
