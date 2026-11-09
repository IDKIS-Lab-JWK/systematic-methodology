"""No-overwrite final-output writers for resumable experiment completion."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from _linear_probe.common import ContractError, atomic_write_csv, atomic_write_json, canonical_json


def same_or_write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    if destination.is_file():
        with destination.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if canonical_json(existing) != canonical_json(value):
            raise ContractError(f"Existing JSON differs; refusing overwrite: {destination}")
        return
    if destination.exists():
        raise ContractError(f"Expected a file path, found another object: {destination}")
    atomic_write_json(destination, value)


def _csv_bytes(rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fieldnames), extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def same_or_write_csv(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    destination = Path(path)
    expected = _csv_bytes(rows, fieldnames)
    if destination.is_file():
        if destination.read_bytes() != expected:
            raise ContractError(f"Existing CSV differs; refusing overwrite: {destination}")
        return
    if destination.exists():
        raise ContractError(f"Expected a file path, found another object: {destination}")
    atomic_write_csv(destination, rows, fieldnames)


__all__ = ["same_or_write_csv", "same_or_write_json"]
