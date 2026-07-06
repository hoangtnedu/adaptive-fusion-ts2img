"""Download utilities for UCR/UEA time-series datasets.

The project expects local files in this layout:

    data/UCR/<Dataset>/<Dataset>_TRAIN.tsv
    data/UCR/<Dataset>/<Dataset>_TEST.tsv

This module keeps dataset-download logic inside the source code so notebooks can
remain lightweight runners that only call command-line modules.
"""

from __future__ import annotations

import csv
import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal


SplitName = Literal["TRAIN", "TEST"]


@dataclass(frozen=True)
class UCRDownloadResult:
    """Result for one dataset download operation."""

    dataset: str
    train_file: Path
    test_file: Path
    status: str


def download_ucr_datasets(
    datasets: Iterable[str],
    data_root: str | Path = "data/UCR",
    base_url: str = "https://www.timeseriesclassification.com/aeon-toolkit",
    overwrite: bool = False,
    timeout: int = 60,
) -> list[UCRDownloadResult]:
    """Download multiple UCR datasets into the project data directory."""

    dataset_names = _clean_dataset_names(datasets)
    if not dataset_names:
        raise ValueError("No dataset names were provided.")

    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)

    results: list[UCRDownloadResult] = []
    for dataset_name in dataset_names:
        results.append(
            download_one_ucr_dataset(
                dataset_name=dataset_name,
                data_root=root,
                base_url=base_url,
                overwrite=overwrite,
                timeout=timeout,
            )
        )
    return results


def download_one_ucr_dataset(
    dataset_name: str,
    data_root: str | Path = "data/UCR",
    base_url: str = "https://www.timeseriesclassification.com/aeon-toolkit",
    overwrite: bool = False,
    timeout: int = 60,
) -> UCRDownloadResult:
    """Download one dataset and create TRAIN/TEST TSV files."""

    dataset_name = str(dataset_name).strip()
    if not dataset_name:
        raise ValueError("dataset_name must not be empty.")

    target_dir = Path(data_root) / dataset_name
    target_dir.mkdir(parents=True, exist_ok=True)

    train_target = target_dir / f"{dataset_name}_TRAIN.tsv"
    test_target = target_dir / f"{dataset_name}_TEST.tsv"

    if train_target.exists() and test_target.exists() and not overwrite:
        return UCRDownloadResult(
            dataset=dataset_name,
            train_file=train_target,
            test_file=test_target,
            status="skipped_existing",
        )

    url = f"{base_url.rstrip('/')}/{dataset_name}.zip"

    with tempfile.TemporaryDirectory(prefix=f"ucr_{dataset_name}_") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        zip_path = tmp_dir / f"{dataset_name}.zip"
        extract_dir = tmp_dir / "extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)

        _download_file(url=url, output_path=zip_path, timeout=timeout)
        _extract_zip(zip_path=zip_path, output_dir=extract_dir)

        train_source = _find_split_file(extract_dir, dataset_name, "TRAIN")
        test_source = _find_split_file(extract_dir, dataset_name, "TEST")

        _copy_or_convert_to_tsv(train_source, train_target)
        _copy_or_convert_to_tsv(test_source, test_target)

    _validate_output_file(train_target, dataset_name, "TRAIN")
    _validate_output_file(test_target, dataset_name, "TEST")

    return UCRDownloadResult(
        dataset=dataset_name,
        train_file=train_target,
        test_file=test_target,
        status="downloaded",
    )


def _clean_dataset_names(datasets: Iterable[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in datasets:
        name = str(item).strip()
        if not name or name in seen:
            continue
        cleaned.append(name)
        seen.add(name)
    return cleaned


def _download_file(url: str, output_path: Path, timeout: int) -> None:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (compatible; adaptive-fusion-ts2img; "
                "+https://github.com/)"
            )
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            output_path.write_bytes(response.read())
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download dataset archive from {url}. "
            "Check the dataset name, network connection, or base_url."
        ) from exc


def _extract_zip(zip_path: Path, output_dir: Path) -> None:
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(output_dir)
    except zipfile.BadZipFile as exc:
        raise RuntimeError(f"Downloaded file is not a valid ZIP archive: {zip_path}") from exc


def _find_split_file(extract_dir: Path, dataset_name: str, split: SplitName) -> Path:
    """Find TRAIN/TEST file in common UCR archive formats."""

    suffixes = [".tsv", ".txt", ".ts"]

    # Exact root-level names first.
    for suffix in suffixes:
        candidate = extract_dir / f"{dataset_name}_{split}{suffix}"
        if candidate.exists():
            return candidate

    lower_dataset = dataset_name.lower()
    lower_split = f"_{split.lower()}"
    matches: list[Path] = []

    for path in extract_dir.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        if lower_dataset in name and lower_split in name and path.suffix.lower() in suffixes:
            matches.append(path)

    if matches:
        suffix_priority = {".tsv": 0, ".txt": 1, ".ts": 2}
        matches.sort(key=lambda p: suffix_priority.get(p.suffix.lower(), 99))
        return matches[0]

    available = "\n".join(
        str(p.relative_to(extract_dir)) for p in extract_dir.rglob("*") if p.is_file()
    )
    raise FileNotFoundError(
        f"Could not find {dataset_name}_{split}.tsv/.txt/.ts in the downloaded archive.\n"
        f"Available files:\n{available}"
    )


def _copy_or_convert_to_tsv(source_path: Path, target_path: Path) -> None:
    suffix = source_path.suffix.lower()

    if suffix == ".tsv":
        shutil.copyfile(source_path, target_path)
        return

    if suffix == ".txt":
        _normalize_delimited_text_to_tsv(source_path, target_path)
        return

    if suffix == ".ts":
        _convert_ts_file_to_tsv(source_path, target_path)
        return

    raise ValueError(f"Unsupported dataset file format: {source_path}")


def _normalize_delimited_text_to_tsv(source_path: Path, target_path: Path) -> None:
    """Convert classic UCR TXT/CSV/whitespace format to TSV."""

    with source_path.open("r", encoding="utf-8", errors="ignore") as src, target_path.open(
        "w", encoding="utf-8", newline=""
    ) as dst:
        writer = csv.writer(dst, delimiter="\t")
        for raw_line in src:
            line = raw_line.strip()
            if not line:
                continue
            if "\t" in line:
                row = line.split("\t")
            elif "," in line:
                row = [item.strip() for item in line.split(",")]
            else:
                row = line.split()
            writer.writerow(row)


def _convert_ts_file_to_tsv(source_path: Path, target_path: Path) -> None:
    """Convert aeon/sktime .ts format to label-first TSV.

    This converter supports the common equal-length classification case used by
    the project. For multivariate .ts files, dimensions are flattened in order.
    """

    data_started = False
    rows_written = 0

    with source_path.open("r", encoding="utf-8", errors="ignore") as src, target_path.open(
        "w", encoding="utf-8", newline=""
    ) as dst:
        writer = csv.writer(dst, delimiter="\t")

        for raw_line in src:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            lower_line = line.lower()
            if lower_line.startswith("@data"):
                data_started = True
                continue
            if lower_line.startswith("@"):
                continue
            if not data_started:
                continue

            parts = line.split(":")
            if len(parts) < 2:
                raise ValueError(f"Invalid .ts data line in {source_path}: {line[:120]}")

            label = parts[-1].strip()
            dimensions = parts[:-1]
            values: list[str] = []

            for dimension in dimensions:
                values.extend(
                    item.strip()
                    for item in dimension.strip().split(",")
                    if item.strip() and item.strip() != "?"
                )

            if not values:
                raise ValueError(f"No time-series values found in .ts line: {line[:120]}")

            writer.writerow([label, *values])
            rows_written += 1

    if rows_written == 0:
        raise ValueError(f"No data rows were converted from {source_path}")


def _validate_output_file(path: Path, dataset_name: str, split: SplitName) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Expected output file was not created for {dataset_name} {split}: {path}"
        )
    if path.stat().st_size == 0:
        raise ValueError(f"Output file is empty for {dataset_name} {split}: {path}")
