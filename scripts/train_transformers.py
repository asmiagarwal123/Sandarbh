#!/usr/bin/env python3
"""Train the fixed SANDARBH Phase 4B DistilRoBERTa experiments."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import shutil
import statistics
import sys
import tempfile
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_PARTITIONS = frozenset({"train", "dev_tune"})
EXPECTED_SOURCE_COLUMNS = ("preceding", "target", "following", "A1_Score", "A2_Score", "A3_Score")
EXPERIMENTS = {
    "E5": {"input_mode": "target", "loss": "weighted_hard_cross_entropy"},
    "E6": {"input_mode": "context", "loss": "weighted_hard_cross_entropy"},
    "E7": {"input_mode": "context", "loss": "weighted_soft_cross_entropy"},
}
TOKEN_MANIFEST_FIELDS = (
    "example_id", "partition", "input_mode", "sequence_length", "target_original_tokens",
    "target_retained_tokens", "target_beginning_tokens", "target_ellipsis_tokens", "target_ending_tokens",
    "target_truncated", "preceding_original_tokens", "preceding_retained_tokens",
    "following_original_tokens", "following_retained_tokens", "input_ids_sha256",
    "attention_mask_sha256", "target_original_sha256", "target_retained_sha256",
)
EPOCH_FIELDS = (
    "experiment_id", "seed", "epoch", "selected", "train_loss", "epoch_seconds", "optimizer_updates",
    "microbatches_processed", "learning_rate_after_epoch", "macro_f1", "positive_precision",
    "positive_recall", "positive_f1", "accuracy", "balanced_accuracy", "confusion_matrix_0_1",
    "support_0_1", "average_precision", "roc_auc", "predicted_positive_count",
    "no_positive_predictions",
)
SELECTED_FIELDS = (
    "experiment_id", "seed", "primary_seed", "selected_epoch", "loss_definition", "macro_f1",
    "positive_precision", "positive_recall", "positive_f1", "accuracy", "balanced_accuracy",
    "confusion_matrix_0_1", "support_0_1", "average_precision", "roc_auc",
    "predicted_positive_count", "no_positive_predictions", "negative_log_likelihood", "brier_score",
    "expected_calibration_error", "ece_bin_count", "ece_binning", "soft_cross_entropy",
    "soft_brier_score", "soft_positive_mean_absolute_difference", "run_seconds",
    "peak_process_rss_bytes", "optimizer_updates", "microbatches_processed", "actual_device",
)
PREDICTION_FIELDS = (
    "experiment_id", "seed", "selected_epoch", "example_id", "partition", "true_hard_label",
    "soft_other", "soft_positive", "predicted_label", "probability_0", "probability_1",
)
PACKAGE_NAMES = ("torch", "transformers", "tokenizers", "safetensors", "huggingface_hub", "numpy", "psutil")


class TrainingError(Exception):
    """Blocking Phase 4B configuration, provenance, training, or publication error."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def hash_ids(ids: Iterable[int]) -> str:
    return sha256_json([int(value) for value in ids])


def record_digest(values: Iterable[str]) -> str:
    payload = json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def collapse_whitespace(text: str) -> str:
    return " ".join(text.split())


def safe_path(relative: str, label: str) -> Path:
    path = (PROJECT_ROOT / relative).resolve()
    try:
        path.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise TrainingError(f"{label} path escapes the project root") from exc
    return path


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TrainingError(f"Cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise TrainingError(f"{label} must contain a JSON object")
    return value


def read_csv(path: Path, label: str) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise TrainingError(f"{label} has a missing or duplicate header")
            return list(reader.fieldnames), list(reader)
    except TrainingError:
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        raise TrainingError(f"Cannot read {label}: {exc}") from exc


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path: Path, fields: Iterable[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    write_json(temp, value)
    os.replace(temp, path)


def atomic_write_csv(path: Path, fields: Iterable[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    write_csv(temp, fields, rows)
    os.replace(temp, path)


def installed_versions() -> dict[str, str]:
    versions = {name: importlib.metadata.version(name) for name in PACKAGE_NAMES}
    versions["python"] = platform.python_version()
    return versions


def load_config(config_path: Path) -> dict[str, Any]:
    config = load_json(config_path, "transformer training configuration")
    required = {
        "training_schema_version", "input_csv", "partition_artifacts", "baseline_artifacts",
        "preflight_artifacts", "reference", "model", "access", "input_construction", "experiments",
        "training", "class_weights", "selection", "calibration_description", "outputs",
    }
    missing = sorted(required - set(config))
    if missing:
        raise TrainingError(f"Configuration keys missing: {missing}")
    fixed_training = {
        "seeds": [42, 43, 44], "primary_seed": 42, "epochs": 3, "microbatch_size": 2,
        "gradient_accumulation_steps": 8, "effective_batch_size": 16, "learning_rate": 0.00002,
        "weight_decay": 0.01, "gradient_clipping_norm": 1.0, "warmup_ratio": 0.10,
        "scheduler": "linear decay after warmup", "precision": "float32", "device": "cpu",
        "dynamic_padding": True, "gradient_checkpointing": False, "parameter_freezing": False,
        "early_stopping": False, "deterministic_algorithms_warn_only": True,
    }
    if config["training"] != fixed_training:
        raise TrainingError("Training settings differ from the approved fixed configuration")
    observed_experiments = {item["experiment_id"]: {"input_mode": item["input_mode"], "loss": item["loss"]} for item in config["experiments"]}
    if observed_experiments != EXPERIMENTS or [item["experiment_id"] for item in config["experiments"]] != ["E5", "E6", "E7"]:
        raise TrainingError("Experiment definitions must be exactly E5, E6, and E7")
    if set(config["access"]["allowed_partitions"]) != ALLOWED_PARTITIONS:
        raise TrainingError("Allowed partitions must be exactly train and dev_tune")
    construction = config["input_construction"]
    if construction["maximum_sequence_length"] != 256 or construction["expected_target_content_budget"] != 241:
        raise TrainingError("Approved length or target budget changed")
    model = config["model"]
    if model != {"repository": "distilbert/distilroberta-base", "revision": "fb53ab8802853c8e4fbdbcd0529f21fc6f459b2b", "weight_format": "safetensors", "trust_remote_code": False, "num_labels": 2}:
        raise TrainingError("Approved model definition changed")
    return config


def verify_upstream(config: dict[str, Any]) -> tuple[dict[str, Path], dict[str, str], dict[str, Any]]:
    paths: dict[str, Path] = {"source": safe_path(config["input_csv"], "source")}
    paths.update({f"partition_{key}": safe_path(value, f"partition {key}") for key, value in config["partition_artifacts"].items()})
    paths.update({f"baseline_{key}": safe_path(value, f"baseline {key}") for key, value in config["baseline_artifacts"].items()})
    # Phase 4A generated outputs have a different Windows creator ACL. The required
    # Phase 4A validator is run separately; training directly binds its readable
    # configuration plus the approved hashes recorded in this Phase 4B config.
    paths["preflight_config"] = safe_path(config["preflight_artifacts"]["config"], "preflight config")
    for name, path in paths.items():
        if not path.is_file():
            raise TrainingError(f"Missing upstream artifact {name}: {path}")
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    reference = config["reference"]
    comparisons = {
        "source": hashes["source"] == reference["source_sha256"],
        "targets": hashes["partition_targets"] == reference["targets_sha256"],
        "split_manifest": hashes["partition_split_manifest"] == reference["split_manifest_sha256"],
        "preflight_config": hashes["preflight_config"] == reference["preflight_config_sha256"],
    }
    if not all(comparisons.values()):
        raise TrainingError(f"Upstream reference hash mismatch: {comparisons}")
    partition_status = load_json(paths["partition_run_status"], "partition status")
    partition_summary = load_json(paths["partition_summary"], "partition summary")
    baseline_status = load_json(paths["baseline_run_status"], "baseline status")
    baseline_manifest = load_json(paths["baseline_run_manifest"], "baseline manifest")
    preflight_config = load_json(paths["preflight_config"], "preflight config")
    if partition_status.get("status") != "COMPLETED" or partition_summary.get("status") != "FROZEN":
        raise TrainingError("Frozen partitions are not completed")
    if baseline_status.get("status") != "COMPLETED" or baseline_manifest.get("status") != "COMPLETED":
        raise TrainingError("Phase 3 baseline run is not completed")
    if partition_summary.get("provenance_fingerprint") != reference["partition_fingerprint"]:
        raise TrainingError("Partition fingerprint mismatch")
    if preflight_config.get("model", {}).get("revision") != config["model"]["revision"]:
        raise TrainingError("Preflight configuration model revision mismatch")
    return paths, hashes, {"partition_summary": partition_summary, "baseline_manifest": baseline_manifest, "preflight_config": preflight_config}


def load_source_rows(path: Path, desired_ids: set[str]) -> dict[str, dict[str, str]]:
    selected: dict[str, dict[str, str]] = {}
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle, strict=True)
            header = next(reader)
            if tuple(header) != EXPECTED_SOURCE_COLUMNS:
                raise TrainingError("Source header differs from the audited schema")
            for source_row, fields in enumerate(reader, 1):
                if len(fields) != len(header):
                    raise TrainingError(f"Malformed source record at source_row={source_row}")
                example_id = f"row-{source_row:06d}-{record_digest(fields)[:12]}"
                if example_id in desired_ids:
                    raw = dict(zip(header, fields))
                    selected[example_id] = {field: collapse_whitespace(raw[field]) for field in ("preceding", "target", "following")}
    except (OSError, UnicodeError, csv.Error, StopIteration) as exc:
        raise TrainingError(f"Cannot resolve source examples: {exc}") from exc
    missing = sorted(desired_ids - set(selected))
    if missing:
        raise TrainingError(f"Source is missing requested stable IDs: {missing[:5]}")
    return selected


def load_partitions(requested: Iterable[str], paths: dict[str, Path]) -> dict[str, list[dict[str, Any]]]:
    requested_list = list(requested)
    forbidden = sorted(set(requested_list) - ALLOWED_PARTITIONS)
    if forbidden:
        raise TrainingError(f"Phase 4B data access denied for {forbidden}; allowed: train, dev_tune")
    _, manifest = read_csv(paths["partition_split_manifest"], "split manifest")
    _, targets = read_csv(paths["partition_targets"], "targets")
    target_by_id = {row["example_id"]: row for row in targets}
    selected_manifest = {partition: [row for row in manifest if row.get("partition") == partition] for partition in requested_list}
    desired_ids = {row["example_id"] for rows in selected_manifest.values() for row in rows}
    source = load_source_rows(paths["source"], desired_ids)
    output: dict[str, list[dict[str, Any]]] = {}
    for partition, rows in selected_manifest.items():
        rows.sort(key=lambda row: int(row["source_row"]))
        ids = [row["example_id"] for row in rows]
        if not ids or len(ids) != len(set(ids)):
            raise TrainingError(f"Empty or duplicate IDs in {partition}")
        partition_rows = []
        for row in rows:
            target = target_by_id.get(row["example_id"])
            if target is None or target.get("source_row") != row.get("source_row") or target.get("rule_b_group_id") != row.get("group_id"):
                raise TrainingError(f"Target/manifest mismatch for {row['example_id']}")
            soft_positive = float(target["soft_positive"])
            soft_other = float(target["soft_other"])
            if not math.isclose(soft_positive + soft_other, 1.0, abs_tol=1e-12):
                raise TrainingError(f"Invalid soft target for {row['example_id']}")
            partition_rows.append({
                "example_id": row["example_id"], "source_row": int(row["source_row"]), "group_id": row["group_id"],
                "partition": partition, "hard_label": int(target["hard_label"]), "soft_positive": soft_positive,
                "soft_other": soft_other, "annotations": [int(target[f"A{i}_Score"]) for i in (1, 2, 3)],
                **source[row["example_id"]],
            })
        output[partition] = partition_rows
    return output


def truncate_target(target_ids: list[int], budget: int, ellipsis_ids: list[int]) -> tuple[list[int], dict[str, int | bool]]:
    if budget <= 0:
        raise TrainingError("Target budget must be positive")
    if len(target_ids) <= budget:
        return list(target_ids), {"original": len(target_ids), "retained": len(target_ids), "beginning": len(target_ids), "ellipsis": 0, "ending": 0, "truncated": False}
    if not ellipsis_ids or len(ellipsis_ids) >= budget:
        raise TrainingError("Ellipsis tokenization does not fit inside the target budget")
    capacity = budget - len(ellipsis_ids)
    beginning = (capacity + 1) // 2
    ending = capacity // 2
    retained = list(target_ids[:beginning]) + list(ellipsis_ids) + list(target_ids[-ending:] if ending else [])
    if len(retained) != budget:
        raise TrainingError("Overlong target retention failed to fill its budget")
    return retained, {"original": len(target_ids), "retained": len(retained), "beginning": beginning, "ellipsis": len(ellipsis_ids), "ending": ending, "truncated": True}


def allocate_context(preceding_ids: list[int], following_ids: list[int], capacity: int) -> tuple[list[int], list[int]]:
    if capacity < 0:
        raise TrainingError("Negative context capacity")
    preceding_allocation = (capacity + 1) // 2
    following_allocation = capacity // 2
    if len(preceding_ids) < preceding_allocation:
        following_allocation += preceding_allocation - len(preceding_ids)
        preceding_allocation = len(preceding_ids)
    if len(following_ids) < following_allocation:
        preceding_allocation += following_allocation - len(following_ids)
        following_allocation = len(following_ids)
    preceding_take = min(len(preceding_ids), preceding_allocation)
    following_take = min(len(following_ids), following_allocation)
    return list(preceding_ids[-preceding_take:] if preceding_take else []), list(following_ids[:following_take])


def construct_encodings(tokenizer: Any, partitions: dict[str, list[dict[str, Any]]], config: dict[str, Any]) -> tuple[dict[str, dict[str, dict[str, Any]]], list[dict[str, Any]], dict[str, Any]]:
    construction = config["input_construction"]
    markers = {name: tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"] for name, text in construction["markers"].items()}
    ellipsis_ids = tokenizer(construction["overlong_target_ellipsis"], add_special_tokens=False, truncation=False)["input_ids"]
    bos, eos = tokenizer.bos_token_id, tokenizer.eos_token_id
    if bos is None or eos is None:
        raise TrainingError("Tokenizer is missing BOS or EOS token")
    outer_count = 2
    marker_total = sum(len(values) for values in markers.values())
    target_budget = construction["maximum_sequence_length"] - outer_count - marker_total
    if outer_count != construction["expected_outer_special_tokens"] or target_budget != construction["expected_target_content_budget"]:
        raise TrainingError(f"Recalculated special/target budget disagrees: outer={outer_count}, target_budget={target_budget}")
    encoded: dict[str, dict[str, dict[str, Any]]] = {"target": {}, "context": {}}
    manifest_rows: list[dict[str, Any]] = []
    overlong_ids: list[str] = []
    for partition, rows in partitions.items():
        for row in rows:
            target_original = tokenizer(row["target"], add_special_tokens=False, truncation=False)["input_ids"]
            preceding_original = tokenizer(row["preceding"], add_special_tokens=False, truncation=False)["input_ids"]
            following_original = tokenizer(row["following"], add_special_tokens=False, truncation=False)["input_ids"]
            target_retained, target_info = truncate_target(target_original, target_budget, ellipsis_ids)
            if target_info["truncated"]:
                overlong_ids.append(row["example_id"])
            target_sequence = [bos] + list(markers["target"]) + target_retained + [eos]
            context_capacity = construction["maximum_sequence_length"] - outer_count - marker_total - len(target_retained)
            preceding_retained, following_retained = allocate_context(preceding_original, following_original, context_capacity)
            context_sequence = [bos] + list(markers["preceding"]) + preceding_retained + list(markers["target"]) + target_retained + list(markers["following"]) + following_retained + [eos]
            if len(target_sequence) > 256 or len(context_sequence) > 256:
                raise TrainingError(f"Constructed sequence exceeds 256 for {row['example_id']}")
            common = {key: row[key] for key in ("example_id", "partition", "hard_label", "soft_other", "soft_positive", "annotations")}
            for mode, sequence in (("target", target_sequence), ("context", context_sequence)):
                record = {**common, "input_ids": sequence, "attention_mask": [1] * len(sequence)}
                encoded[mode][row["example_id"]] = record
                manifest_rows.append({
                    "example_id": row["example_id"], "partition": partition, "input_mode": mode,
                    "sequence_length": len(sequence), "target_original_tokens": len(target_original),
                    "target_retained_tokens": target_info["retained"], "target_beginning_tokens": target_info["beginning"],
                    "target_ellipsis_tokens": target_info["ellipsis"], "target_ending_tokens": target_info["ending"],
                    "target_truncated": str(bool(target_info["truncated"])),
                    "preceding_original_tokens": len(preceding_original), "preceding_retained_tokens": len(preceding_retained) if mode == "context" else 0,
                    "following_original_tokens": len(following_original), "following_retained_tokens": len(following_retained) if mode == "context" else 0,
                    "input_ids_sha256": hash_ids(sequence), "attention_mask_sha256": hash_ids([1] * len(sequence)),
                    "target_original_sha256": hash_ids(target_original), "target_retained_sha256": hash_ids(target_retained),
                })
    manifest_rows.sort(key=lambda item: (item["partition"], item["example_id"], item["input_mode"]))
    context_signature = [{key: row[key] for key in ("example_id", "partition", "input_ids_sha256", "attention_mask_sha256")} for row in manifest_rows if row["input_mode"] == "context"]
    metadata = {
        "bos_token_id": bos, "eos_token_id": eos, "outer_special_token_count": outer_count,
        "marker_token_counts": {name: len(values) for name, values in markers.items()},
        "marker_token_hashes": {name: hash_ids(values) for name, values in markers.items()},
        "ellipsis_token_count": len(ellipsis_ids), "ellipsis_token_hash": hash_ids(ellipsis_ids),
        "target_content_budget": target_budget, "overlong_target_example_ids": sorted(set(overlong_ids)),
        "context_encoded_manifest_sha256": sha256_json(context_signature),
        "e6_encoded_manifest_sha256": sha256_json(context_signature),
        "e7_encoded_manifest_sha256": sha256_json(context_signature),
    }
    if metadata["e6_encoded_manifest_sha256"] != metadata["e7_encoded_manifest_sha256"]:
        raise TrainingError("E6/E7 encoded manifests differ")
    return encoded, manifest_rows, metadata


def calculate_class_weights(train_rows: list[dict[str, Any]], config: dict[str, Any]) -> tuple[list[float], dict[int, int]]:
    counts = Counter(int(row["hard_label"]) for row in train_rows)
    expected = config["class_weights"]
    if len(train_rows) != expected["expected_train_examples"] or [counts[0], counts[1]] != expected["expected_class_counts_0_1"]:
        raise TrainingError(f"Training class counts differ: n={len(train_rows)}, counts={dict(counts)}")
    weights = [len(train_rows) / (2.0 * counts[label]) for label in (0, 1)]
    return weights, dict(counts)


def weighted_hard_loss(log_probabilities: Any, labels: Any, class_weights: Any) -> Any:
    import torch
    selected = log_probabilities[torch.arange(labels.shape[0], device=labels.device), labels]
    selected_weights = class_weights[labels]
    return -(selected_weights * selected).mean()


def weighted_soft_loss(log_probabilities: Any, soft_targets: Any, class_weights: Any) -> Any:
    return -(soft_targets * class_weights.unsqueeze(0) * log_probabilities).sum(dim=1).mean()


def accumulation_group_size(batch_index: int, total_microbatches: int, accumulation_steps: int) -> int:
    if not 0 <= batch_index < total_microbatches or accumulation_steps < 1:
        raise TrainingError("Invalid accumulation indices")
    group_start = (batch_index // accumulation_steps) * accumulation_steps
    return min(accumulation_steps, total_microbatches - group_start)


def selection_key(epoch_record: dict[str, Any]) -> tuple[float, int]:
    return (-float(epoch_record["macro_f1"]), int(epoch_record["epoch"]))


def select_epoch(epoch_records: list[dict[str, Any]]) -> dict[str, Any]:
    if not epoch_records:
        raise TrainingError("Cannot select from no epoch records")
    return min(epoch_records, key=selection_key)


def average_precision(y_true: list[int], scores: list[float]) -> float:
    positives = sum(y_true)
    if positives == 0:
        return float("nan")
    ordered = sorted(zip(scores, y_true), key=lambda pair: pair[0], reverse=True)
    tp = fp = 0
    previous_recall = 0.0
    result = 0.0
    index = 0
    while index < len(ordered):
        score = ordered[index][0]
        group = []
        while index < len(ordered) and ordered[index][0] == score:
            group.append(ordered[index][1])
            index += 1
        tp += sum(group)
        fp += len(group) - sum(group)
        recall = tp / positives
        precision = tp / (tp + fp)
        result += (recall - previous_recall) * precision
        previous_recall = recall
    return result


def roc_auc(y_true: list[int], scores: list[float]) -> float:
    positives = sum(y_true)
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda index: scores[index])
    ranks = [0.0] * len(scores)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and scores[order[end]] == scores[order[index]]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        for position in range(index, end):
            ranks[order[position]] = average_rank
        index = end
    positive_rank_sum = sum(rank for rank, label in zip(ranks, y_true) if label == 1)
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def probability_metrics(y_true: list[int], soft_positive: list[float], probabilities: list[float], ece_bins: int) -> dict[str, float]:
    epsilon = 1e-15
    clipped = [min(max(value, epsilon), 1.0 - epsilon) for value in probabilities]
    nll = -sum(label * math.log(probability) + (1 - label) * math.log(1 - probability) for label, probability in zip(y_true, clipped)) / len(y_true)
    brier = sum((probability - label) ** 2 for label, probability in zip(y_true, probabilities)) / len(y_true)
    ece = 0.0
    for bin_index in range(ece_bins):
        low, high = bin_index / ece_bins, (bin_index + 1) / ece_bins
        members = [index for index, value in enumerate(probabilities) if value >= low and (value < high or (bin_index == ece_bins - 1 and value <= high))]
        if members:
            confidence = sum(probabilities[index] for index in members) / len(members)
            accuracy = sum(y_true[index] for index in members) / len(members)
            ece += len(members) / len(y_true) * abs(accuracy - confidence)
    soft_ce = -sum((1.0 - target) * math.log(1.0 - probability) + target * math.log(probability) for target, probability in zip(soft_positive, clipped)) / len(soft_positive)
    soft_brier = sum((probability - target) ** 2 for target, probability in zip(soft_positive, probabilities)) / len(soft_positive)
    soft_mad = sum(abs(probability - target) for target, probability in zip(soft_positive, probabilities)) / len(soft_positive)
    return {"negative_log_likelihood": nll, "brier_score": brier, "expected_calibration_error": ece, "soft_cross_entropy": soft_ce, "soft_brier_score": soft_brier, "soft_positive_mean_absolute_difference": soft_mad}


def calculate_metrics(y_true: list[int], y_pred: list[int], probabilities: list[float], soft_positive: list[float], ece_bins: int = 10) -> dict[str, Any]:
    if not (len(y_true) == len(y_pred) == len(probabilities) == len(soft_positive)) or not y_true:
        raise TrainingError("Metric arrays are empty or have unequal lengths")
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities):
        raise TrainingError("Probabilities are nonfinite or outside [0,1]")
    tn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 0 and pred == 0)
    fp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 0 and pred == 1)
    fn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 1 and pred == 0)
    tp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 1 and pred == 1)
    support = [tn + fp, fn + tp]
    precision_1 = tp / (tp + fp) if tp + fp else 0.0
    recall_1 = tp / support[1] if support[1] else 0.0
    f1_1 = 2 * precision_1 * recall_1 / (precision_1 + recall_1) if precision_1 + recall_1 else 0.0
    precision_0 = tn / (tn + fn) if tn + fn else 0.0
    recall_0 = tn / support[0] if support[0] else 0.0
    f1_0 = 2 * precision_0 * recall_0 / (precision_0 + recall_0) if precision_0 + recall_0 else 0.0
    result: dict[str, Any] = {
        "macro_f1": (f1_0 + f1_1) / 2.0, "positive_precision": precision_1,
        "positive_recall": recall_1, "positive_f1": f1_1, "accuracy": (tn + tp) / len(y_true),
        "balanced_accuracy": (recall_0 + recall_1) / 2.0, "confusion_matrix_0_1": [[tn, fp], [fn, tp]],
        "support_0_1": support, "average_precision": average_precision(y_true, probabilities),
        "roc_auc": roc_auc(y_true, probabilities), "predicted_positive_count": sum(y_pred),
        "no_positive_predictions": sum(y_pred) == 0,
    }
    result.update(probability_metrics(y_true, soft_positive, probabilities, ece_bins))
    return result


def flatten_epoch(record: dict[str, Any], selected_epoch: int) -> dict[str, Any]:
    result = {field: record.get(field, "") for field in EPOCH_FIELDS}
    result["selected"] = str(int(record["epoch"]) == selected_epoch)
    result["confusion_matrix_0_1"] = json.dumps(record["confusion_matrix_0_1"], separators=(",", ":"))
    result["support_0_1"] = json.dumps(record["support_0_1"], separators=(",", ":"))
    return result


def flatten_selected(record: dict[str, Any]) -> dict[str, Any]:
    result = {field: record.get(field, "") for field in SELECTED_FIELDS}
    result["confusion_matrix_0_1"] = json.dumps(record["confusion_matrix_0_1"], separators=(",", ":"))
    result["support_0_1"] = json.dumps(record["support_0_1"], separators=(",", ":"))
    return result


def set_reproducible_seed(seed: int) -> dict[str, Any]:
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    return {
        "python_seed": seed, "numpy_seed": seed, "torch_seed": seed,
        "deterministic_algorithms_enabled": bool(torch.are_deterministic_algorithms_enabled()),
        "warn_only_for_unsupported_deterministic_operations": True,
        "cpu_thread_count": int(torch.get_num_threads()),
    }


def collate(records: list[dict[str, Any]], pad_token_id: int, device: Any) -> dict[str, Any]:
    import torch
    maximum = max(len(record["input_ids"]) for record in records)
    if maximum > 256:
        raise TrainingError("Dynamic padding would exceed 256")
    input_ids = [record["input_ids"] + [pad_token_id] * (maximum - len(record["input_ids"])) for record in records]
    masks = [record["attention_mask"] + [0] * (maximum - len(record["attention_mask"])) for record in records]
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(masks, dtype=torch.long, device=device),
        "hard_labels": torch.tensor([record["hard_label"] for record in records], dtype=torch.long, device=device),
        "soft_targets": torch.tensor([[record["soft_other"], record["soft_positive"]] for record in records], dtype=torch.float32, device=device),
    }


def evaluate_model(model: Any, records: list[dict[str, Any]], pad_token_id: int, device: Any, batch_size: int, experiment_id: str, seed: int, epoch: int, config: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    probabilities: list[float] = []
    soft_positive: list[float] = []
    prediction_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch_records = records[start:start + batch_size]
            batch = collate(batch_records, pad_token_id, device)
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            if not bool(torch.isfinite(logits).all().item()):
                raise TrainingError(f"Nonfinite logits during {experiment_id} seed {seed} epoch {epoch}")
            batch_probabilities = torch.softmax(logits, dim=1)
            batch_predictions = torch.argmax(logits, dim=1)
            for index, source in enumerate(batch_records):
                probability_0 = float(batch_probabilities[index, 0].item())
                probability_1 = float(batch_probabilities[index, 1].item())
                predicted = int(batch_predictions[index].item())
                y_true.append(int(source["hard_label"]))
                y_pred.append(predicted)
                probabilities.append(probability_1)
                soft_positive.append(float(source["soft_positive"]))
                prediction_rows.append({
                    "experiment_id": experiment_id, "seed": seed, "selected_epoch": epoch,
                    "example_id": source["example_id"], "partition": "dev_tune",
                    "true_hard_label": source["hard_label"], "soft_other": source["soft_other"],
                    "soft_positive": source["soft_positive"], "predicted_label": predicted,
                    "probability_0": probability_0, "probability_1": probability_1,
                })
    metrics = calculate_metrics(y_true, y_pred, probabilities, soft_positive, config["calibration_description"]["ece_bin_count"])
    return metrics, prediction_rows


def checkpoint_hashes(directory: Path) -> dict[str, str]:
    if not directory.is_dir():
        raise TrainingError(f"Checkpoint directory missing: {directory}")
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    if not files or not any(path.name == "model.safetensors" for path in files):
        raise TrainingError(f"Checkpoint lacks safetensors weights: {directory}")
    if any(path.suffix in {".bin", ".pt"} for path in files):
        raise TrainingError("Checkpoint contains a disallowed duplicate weight format")
    return {str(path.relative_to(directory)).replace("\\", "/"): sha256_file(path) for path in files}


def unit_fingerprint(experiment_id: str, seed: int, code_hashes: dict[str, str], upstream_hashes: dict[str, str], token_manifest_hash: str, config: dict[str, Any]) -> str:
    return sha256_json({
        "experiment_id": experiment_id, "seed": seed, "code_hashes": code_hashes,
        "upstream_hashes": upstream_hashes, "token_manifest_hash": token_manifest_hash,
        "model_revision": config["model"]["revision"], "training": config["training"],
        "experiment": EXPERIMENTS[experiment_id], "input_construction": config["input_construction"],
    })


def resolve_model_snapshot(config: dict[str, Any]) -> Path:
    from huggingface_hub import snapshot_download
    path = Path(snapshot_download(
        repo_id=config["model"]["repository"], revision=config["model"]["revision"],
        allow_patterns=["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json"],
        local_files_only=True,
    ))
    if not (path / "model.safetensors").is_file() or any(path.glob("*.bin")):
        raise TrainingError("Pinned local snapshot lacks the single safetensors weight format")
    return path


def verify_completed_unit(unit_dir: Path, expected_fingerprint: str, config: dict[str, Any]) -> dict[str, Any]:
    unit_path = unit_dir / "unit.json"
    if not unit_path.is_file():
        raise TrainingError(f"Incomplete/corrupt unit directory exists: {unit_dir}")
    unit = load_json(unit_path, f"unit {unit_dir.name}")
    if unit.get("status") != "COMPLETED" or unit.get("provenance_fingerprint") != expected_fingerprint:
        raise TrainingError(f"Completed unit provenance/status mismatch: {unit_dir.name}")
    required = {"epoch_metrics": unit_dir / "epoch_metrics.csv", "predictions": unit_dir / "predictions.csv", "selected_metrics": unit_dir / "selected_metrics.json"}
    if any(not path.is_file() for path in required.values()):
        raise TrainingError(f"Completed unit files missing: {unit_dir.name}")
    if any(unit.get("output_hashes", {}).get(name) != sha256_file(path) for name, path in required.items()):
        raise TrainingError(f"Completed unit output hash mismatch: {unit_dir.name}")
    _, epochs = read_csv(required["epoch_metrics"], f"{unit_dir.name} epochs")
    _, predictions = read_csv(required["predictions"], f"{unit_dir.name} predictions")
    if len(epochs) != 3 or len(predictions) != 171:
        raise TrainingError(f"Completed unit row counts invalid: {unit_dir.name}")
    seed = int(unit["seed"])
    checkpoint = unit.get("checkpoint")
    if seed == config["training"]["primary_seed"]:
        if not checkpoint:
            raise TrainingError(f"Primary unit lacks checkpoint record: {unit_dir.name}")
        checkpoint_dir = safe_path(checkpoint["path"], "checkpoint")
        current = checkpoint_hashes(checkpoint_dir)
        if current != checkpoint.get("files"):
            raise TrainingError(f"Primary checkpoint hash mismatch: {unit_dir.name}")
    elif checkpoint is not None:
        raise TrainingError(f"Robustness unit unexpectedly retains weights: {unit_dir.name}")
    return unit


def train_unit(experiment_id: str, seed: int, encoded: dict[str, dict[str, dict[str, Any]]], partitions: dict[str, list[dict[str, Any]]], tokenizer: Any, snapshot: Path, config: dict[str, Any], fingerprint: str, class_weights: list[float], unit_dir: Path) -> dict[str, Any]:
    import psutil
    import torch
    from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

    if unit_dir.exists():
        return verify_completed_unit(unit_dir, fingerprint, config)
    experiment = EXPERIMENTS[experiment_id]
    mode = experiment["input_mode"]
    train_records = [encoded[mode][row["example_id"]] for row in partitions["train"]]
    dev_records = [encoded[mode][row["example_id"]] for row in partitions["dev_tune"]]
    seed_record = set_reproducible_seed(seed)
    device = torch.device("cpu")
    model = AutoModelForSequenceClassification.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False, use_safetensors=True,
        num_labels=config["model"]["num_labels"],
    ).to(device)
    if any(not parameter.requires_grad for parameter in model.parameters()):
        raise TrainingError("At least one model parameter is frozen")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"])
    microbatch_size = config["training"]["microbatch_size"]
    accumulation_steps = config["training"]["gradient_accumulation_steps"]
    total_microbatches = math.ceil(len(train_records) / microbatch_size)
    updates_per_epoch = math.ceil(total_microbatches / accumulation_steps)
    total_updates = updates_per_epoch * config["training"]["epochs"]
    warmup_updates = int(total_updates * config["training"]["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_updates, num_training_steps=total_updates)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    run_start = time.perf_counter()
    epoch_records: list[dict[str, Any]] = []
    selected_epoch = 0
    selected_metrics: dict[str, Any] | None = None
    selected_predictions: list[dict[str, Any]] | None = None
    best_macro_f1 = -math.inf
    optimizer_update_count = 0
    microbatch_count = 0
    checkpoint_work: Path | None = None
    checkpoint_parent: Path | None = None
    if seed == config["training"]["primary_seed"]:
        checkpoint_root = safe_path(config["outputs"]["checkpoint_root"], "checkpoint root")
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        checkpoint_parent = checkpoint_root / f".{experiment_id}_seed{seed}.{uuid.uuid4().hex}.work"
        checkpoint_work = checkpoint_parent / "checkpoint"
        checkpoint_work.mkdir(parents=True)
    optimizer.zero_grad(set_to_none=True)
    try:
        for epoch in range(1, config["training"]["epochs"] + 1):
            epoch_start = time.perf_counter()
            order = list(range(len(train_records)))
            random.Random(seed * 1_000_003 + epoch).shuffle(order)
            epoch_loss_sum = 0.0
            epoch_examples = 0
            epoch_updates_start = optimizer_update_count
            epoch_microbatches_start = microbatch_count
            model.train()
            for batch_index, start in enumerate(range(0, len(order), microbatch_size)):
                indices = order[start:start + microbatch_size]
                records = [train_records[index] for index in indices]
                batch = collate(records, tokenizer.pad_token_id, device)
                logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
                if not bool(torch.isfinite(logits).all().item()):
                    raise TrainingError(f"Nonfinite training logits in {experiment_id} seed {seed}")
                log_probabilities = torch.log_softmax(logits, dim=1)
                if experiment["loss"] == "weighted_hard_cross_entropy":
                    loss = weighted_hard_loss(log_probabilities, batch["hard_labels"], weight_tensor)
                else:
                    loss = weighted_soft_loss(log_probabilities, batch["soft_targets"], weight_tensor)
                if not bool(torch.isfinite(loss).item()):
                    raise TrainingError(f"Nonfinite training loss in {experiment_id} seed {seed}")
                group_size = accumulation_group_size(batch_index, total_microbatches, accumulation_steps)
                (loss / group_size).backward()
                epoch_loss_sum += float(loss.item()) * len(records)
                epoch_examples += len(records)
                microbatch_count += 1
                if (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == total_microbatches:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clipping_norm"])
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_update_count += 1
                peak_rss = max(peak_rss, process.memory_info().rss)
            metrics, predictions = evaluate_model(model, dev_records, tokenizer.pad_token_id, device, microbatch_size, experiment_id, seed, epoch, config)
            epoch_record = {
                "experiment_id": experiment_id, "seed": seed, "epoch": epoch,
                "train_loss": epoch_loss_sum / epoch_examples, "epoch_seconds": time.perf_counter() - epoch_start,
                "optimizer_updates": optimizer_update_count - epoch_updates_start,
                "microbatches_processed": microbatch_count - epoch_microbatches_start,
                "learning_rate_after_epoch": float(scheduler.get_last_lr()[0]),
                **{key: metrics[key] for key in ("macro_f1", "positive_precision", "positive_recall", "positive_f1", "accuracy", "balanced_accuracy", "confusion_matrix_0_1", "support_0_1", "average_precision", "roc_auc", "predicted_positive_count", "no_positive_predictions")},
            }
            epoch_records.append(epoch_record)
            if metrics["macro_f1"] > best_macro_f1:
                best_macro_f1 = metrics["macro_f1"]
                selected_epoch = epoch
                selected_metrics = metrics
                selected_predictions = predictions
                if checkpoint_work is not None:
                    model.save_pretrained(checkpoint_work, safe_serialization=True)
                    tokenizer.save_pretrained(checkpoint_work)
            print(f"{experiment_id} seed={seed} epoch={epoch}/3 macro_f1={metrics['macro_f1']:.6f} epoch_seconds={epoch_record['epoch_seconds']:.1f}", flush=True)
        if selected_metrics is None or selected_predictions is None:
            raise TrainingError("No checkpoint was selected")
        if select_epoch(epoch_records)["epoch"] != selected_epoch:
            raise TrainingError("Incremental checkpoint selection differs from exact selection rule")
        run_seconds = time.perf_counter() - run_start
        selected_record = {
            "experiment_id": experiment_id, "seed": seed,
            "primary_seed": seed == config["training"]["primary_seed"], "selected_epoch": selected_epoch,
            "loss_definition": experiment["loss"], **selected_metrics,
            "ece_bin_count": config["calibration_description"]["ece_bin_count"],
            "ece_binning": config["calibration_description"]["ece_binning"],
            "run_seconds": run_seconds, "peak_process_rss_bytes": int(peak_rss),
            "optimizer_updates": optimizer_update_count, "microbatches_processed": microbatch_count,
            "actual_device": "cpu",
        }
        for row in selected_predictions:
            row["selected_epoch"] = selected_epoch
        checkpoint_record = None
        if checkpoint_work is not None and checkpoint_parent is not None:
            final_checkpoint = safe_path(f"{config['outputs']['checkpoint_root']}/{experiment_id}_seed{seed}", "final checkpoint")
            if final_checkpoint.exists():
                raise TrainingError(f"Orphan or pre-existing checkpoint blocks publication: {final_checkpoint}")
            files = checkpoint_hashes(checkpoint_work)
            os.replace(checkpoint_work, final_checkpoint)
            checkpoint_record = {"path": str(final_checkpoint.relative_to(PROJECT_ROOT)).replace("\\", "/"), "files": files}
        unit_dir.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".{unit_dir.name}.", dir=unit_dir.parent) as temp_name:
            staged = Path(temp_name) / unit_dir.name
            staged.mkdir()
            epoch_path = staged / "epoch_metrics.csv"
            prediction_path = staged / "predictions.csv"
            selected_path = staged / "selected_metrics.json"
            write_csv(epoch_path, EPOCH_FIELDS, [flatten_epoch(record, selected_epoch) for record in epoch_records])
            write_csv(prediction_path, PREDICTION_FIELDS, selected_predictions)
            write_json(selected_path, selected_record)
            output_hashes = {"epoch_metrics": sha256_file(epoch_path), "predictions": sha256_file(prediction_path), "selected_metrics": sha256_file(selected_path)}
            unit = {
                "status": "COMPLETED", "timestamp_utc": utc_now(), "experiment_id": experiment_id,
                "seed": seed, "primary_seed": seed == config["training"]["primary_seed"],
                "provenance_fingerprint": fingerprint, "input_mode": mode,
                "loss_definition": experiment["loss"], "selected_epoch": selected_epoch,
                "selected_checkpoint_metrics": selected_record, "epoch_count": len(epoch_records),
                "train_count": len(train_records), "dev_tune_count": len(dev_records),
                "partitions_accessed": ["train", "dev_tune"], "class_weights_0_1": class_weights,
                "scheduler": {"total_optimizer_updates": total_updates, "warmup_updates": warmup_updates, "updates_per_epoch": updates_per_epoch},
                "runtime": {"run_seconds": run_seconds, "epoch_seconds": [record["epoch_seconds"] for record in epoch_records], "optimizer_updates": optimizer_update_count, "microbatches_processed": microbatch_count, "peak_process_rss_bytes": int(peak_rss)},
                "reproducibility": seed_record, "output_hashes": output_hashes,
                "checkpoint": checkpoint_record,
            }
            write_json(staged / "unit.json", unit)
            os.replace(staged, unit_dir)
        if checkpoint_parent is not None and checkpoint_parent.exists():
            checkpoint_parent.rmdir()
        del model, optimizer, scheduler
        gc.collect()
        return unit
    except Exception:
        if checkpoint_parent is not None and checkpoint_parent.exists():
            shutil.rmtree(checkpoint_parent)
        raise


SUMMARY_METRICS = (
    "macro_f1", "positive_precision", "positive_recall", "positive_f1", "accuracy",
    "balanced_accuracy", "average_precision", "roc_auc", "negative_log_likelihood", "brier_score",
    "expected_calibration_error", "soft_cross_entropy", "soft_brier_score",
    "soft_positive_mean_absolute_difference",
)


def summarize_seeds(selected_records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"description": "Dev_tune selected-checkpoint descriptive summary; sample standard deviation uses n-1.", "experiments": {}}
    for experiment_id in ("E5", "E6", "E7"):
        records = sorted((record for record in selected_records if record["experiment_id"] == experiment_id), key=lambda record: int(record["seed"]))
        if len(records) != 3:
            continue
        metrics = {}
        for name in SUMMARY_METRICS:
            values = [float(record[name]) for record in records]
            metrics[name] = {"mean": statistics.mean(values), "sample_standard_deviation": statistics.stdev(values), "minimum": min(values), "maximum": max(values)}
        result["experiments"][experiment_id] = {"seeds": [int(record["seed"]) for record in records], "metrics": metrics}
    return result


def paired_comparisons(selected_records: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {(record["experiment_id"], int(record["seed"])): record for record in selected_records}
    comparisons: dict[str, Any] = {
        "description": "Paired dev_tune differences are descriptive only; no significance tests or winner selection.",
        "difference_direction": {"E6_minus_E5": "context hard minus target-only hard", "E7_minus_E6": "context soft minus context hard"},
        "comparisons": {},
    }
    for name, left, right in (("E6_minus_E5", "E6", "E5"), ("E7_minus_E6", "E7", "E6")):
        rows = []
        for seed in (42, 43, 44):
            if (left, seed) not in by_key or (right, seed) not in by_key:
                continue
            rows.append({"seed": seed, "differences": {metric: float(by_key[(left, seed)][metric]) - float(by_key[(right, seed)][metric]) for metric in SUMMARY_METRICS}})
        comparisons["comparisons"][name] = rows
    return comparisons


def load_all_units(units_directory: Path) -> list[dict[str, Any]]:
    units = []
    for experiment_id in ("E5", "E6", "E7"):
        for seed in (42, 43, 44):
            path = units_directory / f"{experiment_id}_seed{seed}" / "unit.json"
            if path.is_file():
                units.append(load_json(path, f"unit {experiment_id} seed {seed}"))
    return units


def aggregate_units(config: dict[str, Any], units: list[dict[str, Any]], token_manifest_hash: str, token_metadata: dict[str, Any], upstream_hashes: dict[str, str], code_hashes: dict[str, str], class_weights: list[float], run_started: str, protected_before: dict[str, str]) -> dict[str, Any]:
    outputs = {key: safe_path(value, f"output {key}") for key, value in config["outputs"].items() if key not in {"directory", "units_directory", "checkpoint_root"}}
    units_directory = safe_path(config["outputs"]["units_directory"], "units directory")
    epoch_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    selected_records: list[dict[str, Any]] = []
    for unit in sorted(units, key=lambda item: (item["experiment_id"], int(item["seed"]))):
        unit_dir = units_directory / f"{unit['experiment_id']}_seed{unit['seed']}"
        _, unit_epochs = read_csv(unit_dir / "epoch_metrics.csv", "unit epoch metrics")
        _, unit_predictions = read_csv(unit_dir / "predictions.csv", "unit predictions")
        epoch_rows.extend(unit_epochs)
        prediction_rows.extend(unit_predictions)
        selected_records.append(unit["selected_checkpoint_metrics"])
    atomic_write_csv(outputs["epoch_metrics"], EPOCH_FIELDS, epoch_rows)
    atomic_write_csv(outputs["selected_checkpoint_metrics"], SELECTED_FIELDS, [flatten_selected(record) for record in selected_records])
    atomic_write_csv(outputs["dev_tune_predictions"], PREDICTION_FIELDS, prediction_rows)
    seed_summary = summarize_seeds(selected_records)
    comparisons = paired_comparisons(selected_records)
    atomic_write_json(outputs["seed_summary"], seed_summary)
    atomic_write_json(outputs["comparisons"], comparisons)
    completed = len(units)
    status = "COMPLETED" if completed == 9 else "RUNNING"
    output_hashes = {}
    if completed == 9:
        output_hashes = {
            "tokenization_manifest": token_manifest_hash,
            "epoch_metrics": sha256_file(outputs["epoch_metrics"]),
            "selected_checkpoint_metrics": sha256_file(outputs["selected_checkpoint_metrics"]),
            "dev_tune_predictions": sha256_file(outputs["dev_tune_predictions"]),
            "seed_summary": sha256_file(outputs["seed_summary"]),
            "comparisons": sha256_file(outputs["comparisons"]),
        }
    manifest = {
        "status": status, "timestamp_utc": utc_now(), "run_started_utc": run_started,
        "training_schema_version": config["training_schema_version"], "completed_unit_count": completed,
        "completed_units": [f"{unit['experiment_id']}_seed{unit['seed']}" for unit in sorted(units, key=lambda item: (item["experiment_id"], int(item["seed"])))],
        "model": config["model"], "training_configuration": config["training"],
        "input_construction": {**config["input_construction"], **token_metadata},
        "partition_fingerprint": config["reference"]["partition_fingerprint"],
        "upstream_hashes": upstream_hashes, "protected_upstream_hashes_before": protected_before,
        "phase4a_external_validation_receipts": {key: value for key, value in config["reference"].items() if key.startswith("preflight_")},
        "code_and_config_hashes": code_hashes, "tokenization_manifest_sha256": token_manifest_hash,
        "e6_e7_encoded_inputs_identical": token_metadata["e6_encoded_manifest_sha256"] == token_metadata["e7_encoded_manifest_sha256"],
        "class_weights_0_1": class_weights,
        "class_weight_counts_0_1": config["class_weights"]["expected_class_counts_0_1"],
        "unit_summaries": [{"experiment_id": unit["experiment_id"], "seed": unit["seed"], "selected_epoch": unit["selected_epoch"], "provenance_fingerprint": unit["provenance_fingerprint"], "runtime": unit["runtime"], "checkpoint": unit["checkpoint"]} for unit in sorted(units, key=lambda item: (item["experiment_id"], int(item["seed"])))],
        "output_hashes": output_hashes,
        "partitions_accessed": ["train", "dev_tune"],
        "primary_seed": 42, "robustness_seeds": [43, 44],
        "not_performed": ["dev_calibration inference", "test inference", "probability calibration", "threshold tuning", "selective prediction", "final evaluation", "ensembling"],
        "development_results_only": True,
        "python_and_dependencies": installed_versions(),
    }
    atomic_write_json(outputs["run_manifest"], manifest)
    atomic_write_json(outputs["run_status"], {"status": status, "timestamp_utc": manifest["timestamp_utc"], "completed_unit_count": completed, "total_unit_count": 9, "message": "Phase 4B training completed." if status == "COMPLETED" else "Phase 4B training in progress; completed units are resumable."})
    return manifest


def markdown_report(manifest: dict[str, Any], selected_records: list[dict[str, Any]], seed_summary: dict[str, Any], comparisons: dict[str, Any]) -> str:
    lines = [
        "# Phase 04B DistilRoBERTa Development Results", "", "**Status:** `COMPLETED`  ",
        f"**Generated (UTC):** {manifest['timestamp_utc']}  ", "",
        "These are `dev_tune` development results, not final test performance. Seed 42 was predesignated for later calibration/final stages and was not selected by performance. No `dev_calibration` or test inference, calibration, threshold tuning, or selective prediction occurred.", "",
        "## Selected checkpoints", "",
        "| Experiment | Seed | Epoch | Macro-F1 | Positive P/R/F1 | Accuracy | Balanced accuracy | AP | ROC-AUC | NLL | Brier | ECE |", "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in sorted(selected_records, key=lambda item: (item["experiment_id"], int(item["seed"]))):
        lines.append(f"| {record['experiment_id']} | {record['seed']} | {record['selected_epoch']} | {record['macro_f1']:.6f} | {record['positive_precision']:.6f}/{record['positive_recall']:.6f}/{record['positive_f1']:.6f} | {record['accuracy']:.6f} | {record['balanced_accuracy']:.6f} | {record['average_precision']:.6f} | {record['roc_auc']:.6f} | {record['negative_log_likelihood']:.6f} | {record['brier_score']:.6f} | {record['expected_calibration_error']:.6f} |")
    lines.extend(["", "## Across-seed macro-F1", "", "| Experiment | Mean | Sample SD | Min | Max |", "|---|---:|---:|---:|---:|"])
    for experiment_id in ("E5", "E6", "E7"):
        summary = seed_summary["experiments"][experiment_id]["metrics"]["macro_f1"]
        lines.append(f"| {experiment_id} | {summary['mean']:.6f} | {summary['sample_standard_deviation']:.6f} | {summary['minimum']:.6f} | {summary['maximum']:.6f} |")
    lines.extend(["", "## Interpretation", "", "E5-versus-E6 and E6-versus-E7 paired differences are stored in `comparisons.json`. They are descriptive across the three fixed seeds; no significance test or overall winner is declared. ECE and all development differences are unstable with only 19 hard-positive dev_tune examples. Soft-target agreement is not described as ground-truth probability calibration.", "", "Only selected seed-42 checkpoints are retained locally. Seeds 43 and 44 retain metrics and predictions but no weights. No final generalization claim is permitted.", ""])
    return "\n".join(lines)


def run_training(config_path: Path) -> int:
    from transformers import AutoTokenizer
    config = load_config(config_path)
    paths, upstream_hashes, upstream = verify_upstream(config)
    protected_before = dict(upstream_hashes)
    outputs_root = safe_path(config["outputs"]["directory"], "output directory")
    units_directory = safe_path(config["outputs"]["units_directory"], "units directory")
    outputs_root.mkdir(parents=True, exist_ok=True)
    units_directory.mkdir(parents=True, exist_ok=True)
    run_started = utc_now()
    atomic_write_json(safe_path(config["outputs"]["run_status"], "run status"), {"status": "RUNNING", "timestamp_utc": run_started, "completed_unit_count": len(load_all_units(units_directory)), "total_unit_count": 9, "message": "Phase 4B training started or resumed."})
    snapshot = resolve_model_snapshot(config)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False, use_fast=True)
    if not tokenizer.is_fast:
        raise TrainingError("Fast tokenizer is required")
    partitions = load_partitions(["train", "dev_tune"], paths)
    if len(partitions["train"]) != 1595 or len(partitions["dev_tune"]) != 171:
        raise TrainingError("Train/dev_tune counts differ from frozen references")
    class_weights, class_counts = calculate_class_weights(partitions["train"], config)
    encoded, token_rows, token_metadata = construct_encodings(tokenizer, partitions, config)
    token_manifest_path = safe_path(config["outputs"]["tokenization_manifest"], "tokenization manifest")
    with tempfile.NamedTemporaryFile(prefix="phase4b-tokenization-", suffix=".csv", dir=outputs_root, delete=False) as handle:
        temp_token_path = Path(handle.name)
    try:
        write_csv(temp_token_path, TOKEN_MANIFEST_FIELDS, token_rows)
        new_hash = sha256_file(temp_token_path)
        if token_manifest_path.exists():
            if sha256_file(token_manifest_path) != new_hash:
                raise TrainingError("Existing tokenization manifest differs; refusing overwrite")
            temp_token_path.unlink()
        else:
            os.replace(temp_token_path, token_manifest_path)
        token_manifest_hash = sha256_file(token_manifest_path)
    finally:
        if temp_token_path.exists():
            temp_token_path.unlink()
    code_hashes = {
        "training_config": sha256_file(config_path),
        "training_script": sha256_file(Path(__file__)),
        "validation_script": sha256_file(PROJECT_ROOT / "scripts" / "validate_transformer_run.py"),
        "tests": sha256_file(PROJECT_ROOT / "tests" / "test_transformer_training.py"),
        "protocol": sha256_file(PROJECT_ROOT / "reports" / "PHASE_04B_TRANSFORMER_PROTOCOL.md"),
        "requirements": sha256_file(PROJECT_ROOT / "requirements-transformers.txt"),
    }
    units: list[dict[str, Any]] = []
    for experiment_id in ("E5", "E6", "E7"):
        for seed in (42, 43, 44):
            fingerprint = unit_fingerprint(experiment_id, seed, code_hashes, upstream_hashes, token_manifest_hash, config)
            unit_dir = units_directory / f"{experiment_id}_seed{seed}"
            atomic_write_json(safe_path(config["outputs"]["run_status"], "run status"), {"status": "RUNNING", "timestamp_utc": utc_now(), "active_unit": f"{experiment_id}_seed{seed}", "completed_unit_count": len(units), "total_unit_count": 9})
            unit = train_unit(experiment_id, seed, encoded, partitions, tokenizer, snapshot, config, fingerprint, class_weights, unit_dir)
            units = load_all_units(units_directory)
            aggregate_units(config, units, token_manifest_hash, token_metadata, upstream_hashes, code_hashes, class_weights, run_started, protected_before)
            print(f"Published {experiment_id} seed={seed}; completed_units={len(units)}/9", flush=True)
    if len(units) != 9:
        raise TrainingError("Not all nine units completed")
    current_upstream = {name: sha256_file(path) for name, path in paths.items()}
    if current_upstream != protected_before:
        raise TrainingError("An upstream source/partition/baseline/preflight artifact changed during training")
    manifest = aggregate_units(config, units, token_manifest_hash, token_metadata, upstream_hashes, code_hashes, class_weights, run_started, protected_before)
    selected_records = [unit["selected_checkpoint_metrics"] for unit in units]
    seed_summary = summarize_seeds(selected_records)
    comparisons = paired_comparisons(selected_records)
    report_path = safe_path(config["outputs"]["report"], "report")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(markdown_report(manifest, selected_records, seed_summary, comparisons))
    print("Phase 4B transformer training completed: COMPLETED")
    print(f"Manifest: {safe_path(config['outputs']['run_manifest'], 'manifest').relative_to(PROJECT_ROOT)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/transformer_training.json")
    args = parser.parse_args(argv)
    try:
        return run_training(safe_path(args.config, "configuration"))
    except TrainingError as exc:
        print(f"Phase 4B training failed: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Phase 4B training interrupted; completed units remain resumable.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Phase 4B training failed unexpectedly: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
