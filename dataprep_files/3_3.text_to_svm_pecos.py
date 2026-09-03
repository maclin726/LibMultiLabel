#!/usr/bin/env python3
"""Create PECOS TF-IDF features and SVM files for EURLEX-4K.

The EURLEX input files are expected to contain one document per line in this
format::

    document_id<TAB>space-separated-labels<TAB>document text

PECOS fits its TF-IDF preprocessor on the training documents plus the label
descriptions.  The same preprocessor is then used for the train, test, and
label-description matrices before they are exported in SVM text format.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import scipy.sparse as sp


@dataclass(frozen=True)
class DatasetPaths:
    dataset_dir: Path
    train: Path
    test: Path
    label_descriptions: Path


@dataclass(frozen=True)
class GeneratedPaths:
    combined_text: Path
    train_text: Path
    test_text: Path
    label_text: Path
    train_matrix: Path
    test_matrix: Path
    label_matrix: Path
    train_svm: Path
    test_svm: Path
    label_svm: Path
    label_mapping: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create PECOS TF-IDF SVM files for EURLEX-4K."
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        type=Path,
        help=(
            "Dataset config containing dataset_path, train_file, test_file, "
            "and label_description_file."
        ),
    )
    parser.add_argument(
        "--vectorizer-config",
        type=Path,
        help="PECOS vectorizer JSON (default: <dataset_path>/config.json).",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="PECOS preprocessor directory (default: <dataset_path>/tfidf-model).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: dataset_path).",
    )
    parser.add_argument(
        "--output-prefix",
        help="Output filename prefix (default: normalized dataset directory name).",
    )
    parser.add_argument(
        "--reuse-model",
        action="store_true",
        help="Use an existing PECOS model instead of fitting it again.",
    )
    parser.add_argument(
        "--reuse-tfidf",
        action="store_true",
        help="Skip PECOS and use existing *.tfidf.npz matrices.",
    )
    return parser.parse_args()


def resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def load_dataset_paths(config_path: Path) -> DatasetPaths:
    config_path = config_path.expanduser().resolve()
    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)

    missing = {
        key
        for key in (
            "dataset_path",
            "train_file",
            "test_file",
            "label_description_file",
        )
        if key not in config
    }
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Missing required config field(s): {names}")

    dataset_dir = resolve_path(config["dataset_path"], config_path.parent)
    paths = DatasetPaths(
        dataset_dir=dataset_dir,
        train=resolve_path(config["train_file"], dataset_dir),
        test=resolve_path(config["test_file"], dataset_dir),
        label_descriptions=resolve_path(config["label_description_file"], dataset_dir),
    )
    for path in (paths.train, paths.test, paths.label_descriptions):
        if not path.is_file():
            raise FileNotFoundError(path)
    return paths


def normalized_prefix(dataset_name: str) -> str:
    prefix = re.sub(r"[^a-z0-9]+", "", dataset_name.lower())
    if not prefix:
        raise ValueError(f"Cannot create an output prefix from {dataset_name!r}")
    return prefix


def generated_paths(output_dir: Path, prefix: str) -> GeneratedPaths:
    return GeneratedPaths(
        combined_text=output_dir / f"{prefix}_tfidf_input_pecos.txt",
        train_text=output_dir / f"{prefix}_tfidf_input_train_pecos.txt",
        test_text=output_dir / f"{prefix}_tfidf_input_test_pecos.txt",
        label_text=output_dir / f"{prefix}_tfidf_label_desc_pecos.txt",
        train_matrix=output_dir / f"{prefix}_tfidf_input_train_pecos.tfidf.npz",
        test_matrix=output_dir / f"{prefix}_tfidf_input_test_pecos.tfidf.npz",
        label_matrix=output_dir / f"{prefix}_tfidf_label_desc_pecos.tfidf.npz",
        train_svm=output_dir / f"{prefix}_tfidf_train_ext_pecos.svm",
        test_svm=output_dir / f"{prefix}_tfidf_test_ext_pecos.svm",
        label_svm=output_dir / f"{prefix}_tfidf_lf_pecos.svm",
        label_mapping=output_dir / "label_mapping.txt",
    )


def parse_dataset_line(
    line: str, path: Path, line_number: int
) -> tuple[list[str], str]:
    parts = line.rstrip("\r\n").split("\t", 2)
    if len(parts) != 3:
        raise ValueError(
            f"{path}:{line_number}: expected document_id, labels, and text "
            "separated by tabs"
        )
    labels = parts[1].split()
    if not labels:
        raise ValueError(f"{path}:{line_number}: document has no labels")
    return labels, parts[2]


def prepare_document_text(
    source_path: Path, output_path: Path
) -> tuple[list[list[str]], set[str]]:
    labels_by_row: list[list[str]] = []
    unique_labels: set[str] = set()

    with source_path.open(encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as output:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            labels, text = parse_dataset_line(line, source_path, line_number)
            labels_by_row.append(labels)
            unique_labels.update(labels)
            output.write(text)
            output.write("\n")

    return labels_by_row, unique_labels


def read_label_descriptions(path: Path) -> list[str]:
    with path.open(encoding="latin-1") as label_file:
        descriptions = [line.rstrip("\r\n") for line in label_file]
    if not descriptions:
        raise ValueError(f"No label descriptions found in {path}")
    return descriptions


def descriptions_in_mapping_order(
    label_mapping: Sequence[str], descriptions: Sequence[str], source_path: Path
) -> list[str]:
    ordered: list[str] = []
    for label in label_mapping:
        try:
            description_index = int(label)
        except ValueError as error:
            raise ValueError(
                f"EURLEX label {label!r} is not an integer description index"
            ) from error
        if not 0 <= description_index < len(descriptions):
            raise ValueError(
                f"Label {label!r} has no corresponding line in {source_path}"
            )
        ordered.append(descriptions[description_index])
    return ordered


def write_lines(path: Path, lines: Iterable[str]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for line in lines:
            output.write(line.rstrip("\r\n"))
            output.write("\n")


def concatenate_text_files(paths: Sequence[Path], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        for path in paths:
            with path.open(encoding="utf-8") as source:
                for line in source:
                    output.write(line.rstrip("\r\n"))
                    output.write("\n")


def run_command(command: Sequence[str]) -> None:
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def ensure_pecos_is_available() -> None:
    if importlib.util.find_spec("pecos") is None:
        raise ModuleNotFoundError(
            "PECOS is not installed in this Python environment. Run the script "
            "with the PECOS environment, for example: "
            "conda run -n zeroshot python tfidf_lf_EURLex-4K.py ..."
        )


def run_pecos(
    paths: GeneratedPaths,
    vectorizer_config: Path,
    model_dir: Path,
    reuse_model: bool,
) -> None:
    ensure_pecos_is_available()
    module = "pecos.utils.featurization.text.preprocess"

    if reuse_model:
        if not model_dir.is_dir():
            raise FileNotFoundError(
                f"Cannot reuse missing PECOS model directory: {model_dir}"
            )
    else:
        run_command(
            [
                sys.executable,
                "-m",
                module,
                "build",
                "--text-pos",
                "0",
                "--from-file",
                "true",
                "--input-text-path",
                str(paths.combined_text),
                "--vectorizer-config-path",
                str(vectorizer_config),
                "--output-model-folder",
                str(model_dir),
            ]
        )

    for text_path, matrix_path in (
        (paths.train_text, paths.train_matrix),
        (paths.test_text, paths.test_matrix),
        (paths.label_text, paths.label_matrix),
    ):
        run_command(
            [
                sys.executable,
                "-m",
                module,
                "run",
                "--text-pos",
                "0",
                "--from-file",
                "true",
                "--input-preprocessor-folder",
                str(model_dir),
                "--input-text-path",
                str(text_path),
                "--output-inst-path",
                str(matrix_path),
            ]
        )


def require_matrices(paths: GeneratedPaths) -> None:
    for path in (paths.train_matrix, paths.test_matrix, paths.label_matrix):
        if not path.is_file():
            raise FileNotFoundError(f"Missing TF-IDF matrix: {path}")


def format_features(indices: Sequence[int], values: Sequence[float]) -> str:
    return " ".join(
        f"{feature_index + 1}:{feature_value}"
        for feature_index, feature_value in zip(indices, values)
    )


def write_dataset_svm(
    path: Path,
    matrix: sp.spmatrix,
    labels_by_row: Sequence[Sequence[str]],
    label_to_id: dict[str, int],
) -> None:
    matrix = matrix.tocsr()
    if matrix.shape[0] != len(labels_by_row):
        raise ValueError(
            f"{path}: matrix has {matrix.shape[0]} rows but the dataset has "
            f"{len(labels_by_row)} rows"
        )

    with path.open("w", encoding="utf-8", newline="\n") as output:
        for row_number, row_labels in enumerate(labels_by_row):
            try:
                label_text = ",".join(str(label_to_id[label]) for label in row_labels)
            except KeyError as error:
                raise ValueError(
                    f"Unknown label while writing {path}: {error}"
                ) from error

            start, end = matrix.indptr[row_number : row_number + 2]
            feature_text = format_features(
                matrix.indices[start:end], matrix.data[start:end]
            )
            output.write(label_text)
            if feature_text:
                output.write(" ")
                output.write(feature_text)
            output.write("\n")


def write_label_svm(path: Path, matrix: sp.spmatrix) -> int:
    matrix = matrix.tocsr()
    zero_rows = 0
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for row_number in range(matrix.shape[0]):
            start, end = matrix.indptr[row_number : row_number + 2]
            if start == end:
                zero_rows += 1
            output.write("\t")
            output.write(
                format_features(matrix.indices[start:end], matrix.data[start:end])
            )
            output.write("\n")
    return zero_rows


def average_label_count(labels_by_row: Sequence[Sequence[str]]) -> float:
    if not labels_by_row:
        return 0.0
    return sum(map(len, labels_by_row)) / len(labels_by_row)


def main() -> None:
    args = parse_args()
    dataset = load_dataset_paths(args.config)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else dataset.dataset_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.output_prefix or normalized_prefix(dataset.dataset_dir.name)
    paths = generated_paths(output_dir, prefix)
    vectorizer_config = (
        args.vectorizer_config.expanduser().resolve()
        if args.vectorizer_config
        else dataset.dataset_dir / "config.json"
    )
    model_dir = (
        args.model_dir.expanduser().resolve()
        if args.model_dir
        else output_dir / "tfidf-model"
    )
    if not vectorizer_config.is_file():
        raise FileNotFoundError(f"Missing PECOS vectorizer config: {vectorizer_config}")

    print(f"Dataset directory: {dataset.dataset_dir}")
    train_labels, train_label_set = prepare_document_text(
        dataset.train, paths.train_text
    )
    test_labels, test_label_set = prepare_document_text(dataset.test, paths.test_text)

    label_mapping = sorted(train_label_set | test_label_set)
    label_to_id = {label: index for index, label in enumerate(label_mapping)}
    descriptions = read_label_descriptions(dataset.label_descriptions)
    ordered_descriptions = descriptions_in_mapping_order(
        label_mapping, descriptions, dataset.label_descriptions
    )

    write_lines(paths.label_mapping, label_mapping)
    write_lines(paths.label_text, ordered_descriptions)
    concatenate_text_files((paths.train_text, paths.label_text), paths.combined_text)

    unseen_labels = test_label_set - train_label_set
    print(f"Training samples: {len(train_labels)}")
    print(f"Test samples: {len(test_labels)}")
    print(f"Labels: {len(label_mapping)}")
    print(f"Unseen test labels: {len(unseen_labels)}")
    print(f"Average train labels/sample: {average_label_count(train_labels):.6f}")
    print(f"Average test labels/sample: {average_label_count(test_labels):.6f}")

    if args.reuse_tfidf:
        require_matrices(paths)
    else:
        run_pecos(paths, vectorizer_config, model_dir, args.reuse_model)

    train_matrix = sp.load_npz(paths.train_matrix).tocsr()
    test_matrix = sp.load_npz(paths.test_matrix).tocsr()
    label_matrix = sp.load_npz(paths.label_matrix).tocsr()
    feature_counts = {
        train_matrix.shape[1],
        test_matrix.shape[1],
        label_matrix.shape[1],
    }
    if len(feature_counts) != 1:
        raise ValueError(
            "Train, test, and label matrices do not share one feature dimension: "
            f"{train_matrix.shape}, {test_matrix.shape}, {label_matrix.shape}"
        )
    if label_matrix.shape[0] != len(label_mapping):
        raise ValueError(
            f"Label matrix has {label_matrix.shape[0]} rows but label_mapping.txt "
            f"has {len(label_mapping)} labels"
        )

    print(f"Train TF-IDF shape: {train_matrix.shape}")
    print(f"Test TF-IDF shape: {test_matrix.shape}")
    print(f"Label TF-IDF shape: {label_matrix.shape}")

    write_dataset_svm(paths.train_svm, train_matrix, train_labels, label_to_id)
    write_dataset_svm(paths.test_svm, test_matrix, test_labels, label_to_id)
    zero_label_vectors = write_label_svm(paths.label_svm, label_matrix)

    print(f"Zero label-description vectors: {zero_label_vectors}")
    print("Created:")
    for path in (
        paths.train_svm,
        paths.test_svm,
        paths.label_svm,
        paths.label_mapping,
    ):
        print(f"  {path}")


if __name__ == "__main__":
    main()
