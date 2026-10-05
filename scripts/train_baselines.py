#!/usr/bin/env python3
"""Train the four frozen SANDARBH Phase 3 classical baseline experiments."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import sys
import tempfile
import time
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_MODELING_PARTITIONS = frozenset({"train", "dev_tune"})
EXPECTED_COLUMNS = ("preceding", "target", "following", "A1_Score", "A2_Score", "A3_Score")
CANDIDATE_FIELDS = (
    "experiment_id", "candidate_id", "classifier", "input_mode", "C", "class_weight", "macro_f1",
    "positive_precision", "positive_recall", "positive_f1", "accuracy", "balanced_accuracy",
    "confusion_matrix_0_1", "support_0_1", "average_precision", "roc_auc", "predicted_positive_count",
    "no_positive_predictions", "class_order", "vocabulary_size", "converged", "warning_messages",
    "effective_tfidf_params", "effective_classifier_params", "selected",
)
PREDICTION_FIELDS = (
    "experiment_id", "example_id", "group_id", "partition", "true_hard_label", "predicted_label",
    "decision_score", "positive_probability",
)
PACKAGE_NAMES = ("cloudpickle", "joblib", "narwhals", "numpy", "scikit-learn", "scipy", "threadpoolctl")


class BaselineError(Exception):
    """Blocking Phase 3 input, training, or publication error."""


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise BaselineError(f"Cannot import {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = load_module("sandarbh_phase1_for_baselines", PROJECT_ROOT / "scripts" / "audit_dataset.py")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_path(relative: str, label: str) -> Path:
    path = (PROJECT_ROOT / relative).resolve()
    try:
        path.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise BaselineError(f"{label} path escapes the project root") from exc
    return path


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BaselineError(f"Cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise BaselineError(f"{label} must contain a JSON object")
    return value


def read_csv(path: Path, label: str) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise BaselineError(f"{label} has a missing or duplicate header")
            return list(reader.fieldnames), list(reader)
    except BaselineError:
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        raise BaselineError(f"Cannot read {label}: {exc}") from exc


def installed_versions() -> dict[str, str]:
    versions = {name: importlib.metadata.version(name) for name in PACKAGE_NAMES}
    versions["python"] = platform.python_version()
    return versions


def load_config(path: Path) -> dict[str, Any]:
    config = load_json(path, "baseline configuration")
    required = {"baseline_schema_version", "input_csv", "partition_artifacts", "required_partition_fingerprint",
                "reference_hashes", "modeling_partition_allowlist", "experiments", "tfidf", "candidate_grid",
                "logistic_regression", "linear_svc", "selection", "expected_versions", "outputs"}
    missing = sorted(required - config.keys())
    if missing:
        raise BaselineError(f"Configuration keys missing: {missing}")
    if set(config["modeling_partition_allowlist"]) != ALLOWED_MODELING_PARTITIONS:
        raise BaselineError("Modeling allowlist must contain exactly train and dev_tune")
    expected_experiments = [("E1", "logistic_regression", "target"), ("E2", "logistic_regression", "context"),
                            ("E3", "linear_svc", "target"), ("E4", "linear_svc", "context")]
    observed = [(x.get("experiment_id"), x.get("classifier"), x.get("input_mode")) for x in config["experiments"]]
    if observed != expected_experiments:
        raise BaselineError("Experiments must be exactly the approved E1-E4 definitions")
    if config["candidate_grid"] != {"C": [0.1, 1.0, 10.0], "class_weight": [None, "balanced"]}:
        raise BaselineError("Candidate grid differs from the approved six-candidate grid")
    versions = installed_versions()
    if versions != config["expected_versions"]:
        raise BaselineError(f"Dependency versions differ from configuration: observed={versions}")
    return config


def verify_provenance(config: dict[str, Any]) -> tuple[dict[str, Path], dict[str, str], dict[str, Any]]:
    paths = {"source": safe_path(config["input_csv"], "source")}
    paths.update({key: safe_path(value, f"partition artifact {key}") for key, value in config["partition_artifacts"].items()})
    for name, path in paths.items():
        if not path.is_file():
            raise BaselineError(f"Missing {name}: {path.relative_to(PROJECT_ROOT)}")
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    references = config["reference_hashes"]
    comparisons = {
        "source": hashes["source"] == references["source_sha256"],
        "targets": hashes["targets"] == references["targets_sha256"],
        "split_manifest": hashes["split_manifest"] == references["split_manifest_sha256"],
    }
    if not all(comparisons.values()):
        raise BaselineError(f"Source or frozen partition reference hash mismatch: {comparisons}")
    status = load_json(paths["run_status"], "partition run status")
    summary = load_json(paths["summary"], "partition summary")
    if status.get("status") != "COMPLETED" or summary.get("status") != "FROZEN":
        raise BaselineError("Phase 2B is not completed and frozen")
    if summary.get("validation", {}).get("status") != "PASSED":
        raise BaselineError("Phase 2B prepublication validation did not pass")
    if summary.get("provenance_fingerprint") != config["required_partition_fingerprint"]:
        raise BaselineError("Frozen partition fingerprint differs from the required fingerprint")
    if summary.get("generated_artifact_hashes", {}).get("targets") != hashes["targets"] or summary.get("generated_artifact_hashes", {}).get("split_manifest") != hashes["split_manifest"]:
        raise BaselineError("Frozen summary does not reconcile with target/manifest hashes")
    return paths, hashes, summary


def collapse_whitespace(text: str) -> str:
    return " ".join(text.split())


def construct_input(texts: dict[str, str], mode: str) -> str:
    target = collapse_whitespace(texts.get("target", ""))
    if mode == "target":
        return target
    if mode != "context":
        raise BaselineError(f"Unknown input mode: {mode}")
    parts = [collapse_whitespace(texts.get(field, "")) for field in ("preceding", "target", "following")]
    return "\n".join(part for part in parts if part)


def read_selected_source(path: Path, desired_ids: set[str]) -> dict[str, dict[str, str]]:
    selected = {}
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle, strict=True)
            header = next(reader)
            if tuple(header) != EXPECTED_COLUMNS:
                raise BaselineError("Source header differs from the audited schema")
            for source_row, fields in enumerate(reader, 1):
                if len(fields) != len(header):
                    raise BaselineError(f"Malformed source record at source_row={source_row}")
                digest = audit.record_digest(fields)
                example_id = f"row-{source_row:06d}-{digest[:12]}"
                if example_id in desired_ids:
                    raw = dict(zip(header, fields))
                    selected[example_id] = {field: raw[field] for field in ("preceding", "target", "following")}
    except (OSError, UnicodeError, csv.Error, StopIteration) as exc:
        raise BaselineError(f"Cannot resolve source examples: {exc}") from exc
    missing = sorted(desired_ids - set(selected))
    if missing:
        raise BaselineError(f"Source is missing requested stable IDs: {missing[:5]}")
    return selected


def load_modeling_partition(partition: str, paths: dict[str, Path], input_mode: str) -> list[dict[str, Any]]:
    if partition not in ALLOWED_MODELING_PARTITIONS:
        raise BaselineError(f"Phase 3 modeling access denied for partition {partition!r}; allowed: train, dev_tune")
    _, manifest = read_csv(paths["split_manifest"], "split manifest")
    _, targets = read_csv(paths["targets"], "targets")
    target_by_id = {row["example_id"]: row for row in targets}
    chosen = [row for row in manifest if row["partition"] == partition]
    chosen.sort(key=lambda row: int(row["source_row"]))
    ids = [row["example_id"] for row in chosen]
    if len(ids) != len(set(ids)):
        raise BaselineError(f"Duplicate IDs in {partition} manifest")
    source = read_selected_source(paths["source"], set(ids))
    output = []
    for item in chosen:
        example_id = item["example_id"]
        target = target_by_id.get(example_id)
        if target is None or target["source_row"] != item["source_row"] or target["rule_b_group_id"] != item["group_id"]:
            raise BaselineError(f"Target/manifest mismatch for {example_id}")
        output.append({"example_id": example_id, "source_row": int(item["source_row"]), "group_id": item["group_id"],
                       "partition": partition, "hard_label": int(target["hard_label"]),
                       "text": construct_input(source[example_id], input_mode)})
    if not output or {row["hard_label"] for row in output} != {0, 1}:
        raise BaselineError(f"Modeling partition {partition} is empty or lacks a hard class")
    return output


def vectorizer_from_config(config: dict[str, Any]) -> TfidfVectorizer:
    params = config["tfidf"]
    return TfidfVectorizer(analyzer=params["analyzer"], lowercase=params["lowercase"],
                           ngram_range=tuple(params["ngram_range"]), min_df=params["min_df"], max_df=params["max_df"],
                           max_features=params["max_features"], sublinear_tf=params["sublinear_tf"], norm=params["norm"],
                           use_idf=params["use_idf"], smooth_idf=params["smooth_idf"], stop_words=params["stop_words"],
                           dtype=np.float64, token_pattern=params["token_pattern"])


def classifier_from_config(family: str, candidate: dict[str, Any], config: dict[str, Any]):
    if family == "logistic_regression":
        params = config["logistic_regression"]
        return LogisticRegression(C=candidate["C"], class_weight=candidate["class_weight"], l1_ratio=params["l1_ratio"],
                                  solver=params["solver"], max_iter=params["max_iter"], tol=params["tol"],
                                  random_state=params["random_state"])
    if family == "linear_svc":
        params = config["linear_svc"]
        return LinearSVC(C=candidate["C"], class_weight=candidate["class_weight"], penalty=params["penalty"],
                         loss=params["loss"], dual=params["dual"], max_iter=params["max_iter"], tol=params["tol"],
                         random_state=params["random_state"])
    raise BaselineError(f"Unsupported classifier family: {family}")


def candidate_grid(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"candidate_id": f"C={C:g}|class_weight={'null' if weight is None else weight}", "C": C, "class_weight": weight}
            for C in config["candidate_grid"]["C"] for weight in config["candidate_grid"]["class_weight"]]


def calculate_metrics(y_true: Iterable[int], y_pred: Iterable[int], ranking_scores: Iterable[float]) -> dict[str, Any]:
    truth = np.asarray(list(y_true), dtype=int)
    pred = np.asarray(list(y_pred), dtype=int)
    scores = np.asarray(list(ranking_scores), dtype=float)
    if truth.shape != pred.shape or truth.shape != scores.shape or not np.isfinite(scores).all():
        raise BaselineError("Metric inputs have inconsistent shapes or non-finite scores")
    precision, recall, f1, support = precision_recall_fscore_support(truth, pred, labels=[0, 1], zero_division=0)
    matrix = confusion_matrix(truth, pred, labels=[0, 1])
    roc = float(roc_auc_score(truth, scores)) if len(set(truth.tolist())) == 2 else None
    return {
        "macro_f1": float(f1_score(truth, pred, labels=[0, 1], average="macro", zero_division=0)),
        "positive_precision": float(precision[1]), "positive_recall": float(recall[1]), "positive_f1": float(f1[1]),
        "accuracy": float(accuracy_score(truth, pred)), "balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
        "confusion_matrix_0_1": matrix.astype(int).tolist(), "support_0_1": support.astype(int).tolist(),
        "average_precision": float(average_precision_score(truth, scores)), "roc_auc": roc,
        "predicted_positive_count": int(np.sum(pred == 1)), "no_positive_predictions": bool(np.sum(pred == 1) == 0),
    }


def score_pipeline(pipeline: Pipeline, family: str, texts: list[str], labels: list[int]) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray | None]:
    classifier = pipeline.named_steps["classifier"]
    classes = classifier.classes_.astype(int).tolist()
    if classes != [0, 1]:
        raise BaselineError(f"Classifier class order is {classes}, expected [0, 1]")
    predicted = pipeline.predict(texts).astype(int)
    decision = np.asarray(pipeline.decision_function(texts), dtype=float)
    if decision.ndim != 1:
        raise BaselineError("Expected a one-dimensional binary decision function")
    probability = None
    ranking = decision
    if family == "logistic_regression":
        probabilities = np.asarray(pipeline.predict_proba(texts), dtype=float)
        positive_index = classes.index(1)
        probability = probabilities[:, positive_index]
        ranking = probability
    metrics = calculate_metrics(labels, predicted, ranking)
    return metrics, predicted, decision, probability


def selection_key(row: dict[str, Any]) -> tuple[float, float, int]:
    weight_rank = 0 if row["class_weight"] is None else 1
    return (-row["metrics"]["macro_f1"], float(row["C"]), weight_rank)


def select_candidate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row["converged"]]
    if not valid:
        raise BaselineError("No converged candidate is available for selection")
    return min(valid, key=selection_key)


def contains_convergence_warning(caught_warnings: Iterable[warnings.WarningMessage]) -> bool:
    return any(issubclass(item.category, ConvergenceWarning) for item in caught_warnings)


def jsonable_params(params: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for key, value in params.items():
        if isinstance(value, type):
            output[key] = f"{value.__module__}.{value.__name__}"
        elif isinstance(value, tuple):
            output[key] = list(value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            output[key] = value
        else:
            output[key] = repr(value)
    return output


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_csv(path: Path, fields: Iterable[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def write_status(path: Path, status: str, message: str, timestamp: str, extra: dict[str, Any] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=".phase3-status-", suffix=".json", dir=path.parent)
    os.close(fd)
    temp = Path(temp_name)
    try:
        write_json(temp, {"status": status, "timestamp_utc": timestamp, "message": message, **(extra or {})})
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def provenance_fingerprint(input_hashes: dict[str, str], config_path: Path) -> tuple[str, dict[str, str], dict[str, str]]:
    code_hashes = {"baseline_config": sha256_file(config_path), "training_script": sha256_file(Path(__file__)),
                   "validation_script": sha256_file(PROJECT_ROOT / "scripts" / "validate_baseline_run.py"),
                   "requirements": sha256_file(PROJECT_ROOT / "requirements-baselines.txt")}
    versions = installed_versions()
    payload = json.dumps({"inputs": input_hashes, "code": code_hashes, "versions": versions}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest(), code_hashes, versions


def flatten_candidate(row: dict[str, Any], selected: bool) -> dict[str, Any]:
    metrics = row["metrics"]
    return {
        "experiment_id": row["experiment_id"], "candidate_id": row["candidate_id"], "classifier": row["classifier"],
        "input_mode": row["input_mode"], "C": format(row["C"], ".12g"),
        "class_weight": "null" if row["class_weight"] is None else row["class_weight"],
        "macro_f1": format(metrics["macro_f1"], ".17g"), "positive_precision": format(metrics["positive_precision"], ".17g"),
        "positive_recall": format(metrics["positive_recall"], ".17g"), "positive_f1": format(metrics["positive_f1"], ".17g"),
        "accuracy": format(metrics["accuracy"], ".17g"), "balanced_accuracy": format(metrics["balanced_accuracy"], ".17g"),
        "confusion_matrix_0_1": json.dumps(metrics["confusion_matrix_0_1"], separators=(",", ":")),
        "support_0_1": json.dumps(metrics["support_0_1"], separators=(",", ":")),
        "average_precision": format(metrics["average_precision"], ".17g"),
        "roc_auc": "" if metrics["roc_auc"] is None else format(metrics["roc_auc"], ".17g"),
        "predicted_positive_count": metrics["predicted_positive_count"], "no_positive_predictions": metrics["no_positive_predictions"],
        "class_order": "[0,1]", "vocabulary_size": row["vocabulary_size"], "converged": row["converged"],
        "warning_messages": json.dumps(row["warning_messages"], ensure_ascii=False, separators=(",", ":")),
        "effective_tfidf_params": json.dumps(row["effective_tfidf_params"], sort_keys=True, separators=(",", ":")),
        "effective_classifier_params": json.dumps(row["effective_classifier_params"], sort_keys=True, separators=(",", ":")),
        "selected": selected,
    }


def dummy_benchmark(train: list[dict[str, Any]], dev: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["hard_label"] for row in train)
    majority = min((-count, label) for label, count in counts.items())[1]
    truth = [row["hard_label"] for row in dev]
    predicted = [majority] * len(dev)
    ranking = [float(majority)] * len(dev)
    return {"benchmark": "train_majority_class", "majority_class": majority,
            "train_class_counts": {str(label): counts[label] for label in (0, 1)},
            "dev_tune_count": len(dev), "competitive_model": False, "metrics": calculate_metrics(truth, predicted, ranking)}


def markdown_report(manifest: dict[str, Any], selected: dict[str, Any], dummy: dict[str, Any]) -> str:
    lines = [
        "# Phase 03 Classical Baselines", "", f"**Status:** `{manifest['status']}`  ",
        f"**Generated (UTC):** {manifest['timestamp_utc']}  ", "",
        "Four TF-IDF baselines were fitted on `train` only and selected independently on `dev_tune`. No model-facing access to `dev_calibration` or `test` occurred. These are development results, not final generalization estimates.", "",
        "## Selected candidates", "",
        "| Experiment | Input | Classifier | C | Class weight | Macro-F1 | Positive F1 | Average precision | ROC-AUC | Vocabulary |",
        "|---|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for experiment_id in ("E1", "E2", "E3", "E4"):
        item = selected[experiment_id]
        metrics = item["metrics"]
        roc = "undefined" if metrics["roc_auc"] is None else f"{metrics['roc_auc']:.6f}"
        lines.append(f"| {experiment_id} | {item['input_mode']} | {item['classifier']} | {item['C']} | {item['class_weight']} | {metrics['macro_f1']:.6f} | {metrics['positive_f1']:.6f} | {metrics['average_precision']:.6f} | {roc} | {item['vocabulary_size']} |")
    lines.extend([
        "", "Selection used exact unrounded macro-F1 over labels `[0, 1]`, then smaller C, then null class weight before `balanced`. No decision threshold was tuned. Logistic Regression probabilities are native but uncalibrated. LinearSVC scores are margins, not probabilities.", "",
        "## Dummy benchmark", "",
        f"The train-only majority class was `{dummy['majority_class']}`. Its dev_tune macro-F1 was {dummy['metrics']['macro_f1']:.6f}. This is a descriptive benchmark, not a competitive model selected for deployment.", "",
        "## Limitations", "",
        "- `dev_tune` contains only 19 hard-positive examples; differences may be unstable.",
        "- Concatenated context does not explicitly encode preceding/target/following roles.",
        "- Serialized TF-IDF pipelines contain learned vocabulary terms and are kept local; reports and tabular outputs contain no raw sentences or feature lists.",
        "- No overall project winner was selected and no experiment was dropped.",
        "- Calibration, selective prediction, transformers, threshold selection, and final test evaluation were not performed.", "",
    ])
    return "\n".join(lines)


def existing_run_state(outputs: dict[str, Path], fingerprint: str, config_path: Path) -> str:
    required = [outputs[key] for key in ("candidate_metrics", "selected_models", "dev_predictions", "dummy_metrics", "run_manifest", "report")]
    existing = [path.exists() for path in required]
    if not any(existing):
        return "absent"
    if not all(existing):
        raise BaselineError("Partial baseline outputs already exist; refusing to overwrite them")
    manifest = load_json(outputs["run_manifest"], "existing baseline run manifest")
    if manifest.get("status") != "COMPLETED" or manifest.get("provenance_fingerprint") != fingerprint:
        raise BaselineError("Completed baseline outputs exist with different provenance; use a separately named run directory")
    validator = load_module("sandarbh_baseline_validator_for_reuse", PROJECT_ROOT / "scripts" / "validate_baseline_run.py")
    validation = validator.validate_run(config_path, quiet=True)
    if not validation["passed"]:
        raise BaselineError("Existing identical-provenance baseline outputs fail validation")
    return "reused"


def run_training(config_path: Path) -> int:
    started = time.perf_counter()
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    config = load_config(config_path)
    outputs = {key: safe_path(value, f"output {key}") for key, value in config["outputs"].items()}
    paths, input_hashes, partition_summary = verify_provenance(config)
    fingerprint, code_hashes, versions = provenance_fingerprint(input_hashes, config_path)
    state = existing_run_state(outputs, fingerprint, config_path)
    if state == "reused":
        write_status(outputs["run_status"], "COMPLETED", "Existing identical-provenance baseline run validated and reused unchanged.", timestamp,
                     {"provenance_fingerprint": fingerprint, "artifacts_reused": True})
        print("Existing Phase 3 baseline outputs validated and reused unchanged.")
        return 0
    write_status(outputs["run_status"], "RUNNING", "Phase 3 baseline training started; outputs are not current until completion.", timestamp)
    protected_before = {name: sha256_file(path) for name, path in paths.items()}

    all_candidates: list[dict[str, Any]] = []
    selected_rows: dict[str, dict[str, Any]] = {}
    selected_pipelines: dict[str, Pipeline] = {}
    selected_predictions: list[dict[str, Any]] = []
    cache: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for experiment in config["experiments"]:
        experiment_id, family, mode = experiment["experiment_id"], experiment["classifier"], experiment["input_mode"]
        if ("train", mode) not in cache:
            cache[("train", mode)] = load_modeling_partition("train", paths, mode)
        if ("dev_tune", mode) not in cache:
            cache[("dev_tune", mode)] = load_modeling_partition("dev_tune", paths, mode)
        train = cache[("train", mode)]
        dev = cache[("dev_tune", mode)]
        train_texts, train_labels = [row["text"] for row in train], [row["hard_label"] for row in train]
        dev_texts, dev_labels = [row["text"] for row in dev], [row["hard_label"] for row in dev]
        experiment_candidates = []
        for candidate in candidate_grid(config):
            pipeline = Pipeline([("tfidf", vectorizer_from_config(config)),
                                 ("classifier", classifier_from_config(family, candidate, config))])
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                pipeline.fit(train_texts, train_labels)
            warning_messages = [f"{type(item.message).__name__}: {item.message}" for item in caught]
            convergence = contains_convergence_warning(caught)
            metrics, predicted, decision, probability = score_pipeline(pipeline, family, dev_texts, dev_labels)
            row = {**candidate, "experiment_id": experiment_id, "classifier": family, "input_mode": mode,
                   "metrics": metrics, "converged": not convergence, "warning_messages": warning_messages,
                   "vocabulary_size": len(pipeline.named_steps["tfidf"].vocabulary_),
                   "effective_tfidf_params": jsonable_params(pipeline.named_steps["tfidf"].get_params(deep=False)),
                   "effective_classifier_params": jsonable_params(pipeline.named_steps["classifier"].get_params(deep=False)),
                   "pipeline": pipeline, "predicted": predicted, "decision": decision, "probability": probability}
            experiment_candidates.append(row)
            all_candidates.append(row)
        selected = select_candidate(experiment_candidates)
        selected_rows[experiment_id] = selected
        selected_pipelines[experiment_id] = selected["pipeline"]
        for index, item in enumerate(dev):
            selected_predictions.append({"experiment_id": experiment_id, "example_id": item["example_id"],
                                         "group_id": item["group_id"], "partition": "dev_tune",
                                         "true_hard_label": item["hard_label"], "predicted_label": int(selected["predicted"][index]),
                                         "decision_score": format(float(selected["decision"][index]), ".17g"),
                                         "positive_probability": "" if selected["probability"] is None else format(float(selected["probability"][index]), ".17g")})

    convergence_failures = [f"{row['experiment_id']}:{row['candidate_id']}" for row in all_candidates if not row["converged"]]
    if convergence_failures:
        write_status(outputs["run_status"], "NEEDS_REVIEW", "At least one candidate emitted a convergence warning; no winner or completed run was published.", timestamp,
                     {"affected_candidates": convergence_failures})
        return 1
    dummy = dummy_benchmark(cache[("train", "target")], cache[("dev_tune", "target")])
    candidate_rows = [flatten_candidate(row, selected_rows[row["experiment_id"]]["candidate_id"] == row["candidate_id"]) for row in all_candidates]
    selected_public = {experiment_id: {key: value for key, value in row.items() if key in {
        "experiment_id", "candidate_id", "classifier", "input_mode", "C", "class_weight", "metrics", "converged",
        "warning_messages", "vocabulary_size", "effective_tfidf_params", "effective_classifier_params"}}
                       for experiment_id, row in selected_rows.items()}

    with tempfile.TemporaryDirectory(prefix="phase3-baselines-stage-", dir=PROJECT_ROOT) as temp_dir:
        stage = Path(temp_dir)
        staged = {key: stage / outputs[key].relative_to(PROJECT_ROOT) for key in ("candidate_metrics", "selected_models", "dev_predictions", "dummy_metrics", "run_manifest", "report")}
        staged_model_dir = stage / outputs["model_directory"].relative_to(PROJECT_ROOT)
        write_csv(staged["candidate_metrics"], CANDIDATE_FIELDS, candidate_rows)
        write_csv(staged["dev_predictions"], PREDICTION_FIELDS, selected_predictions)
        write_json(staged["dummy_metrics"], dummy)
        model_records = {}
        for experiment_id, pipeline in selected_pipelines.items():
            model_path = staged_model_dir / f"{experiment_id}_pipeline.joblib"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(pipeline, model_path)
            final_relative = (outputs["model_directory"] / model_path.name).relative_to(PROJECT_ROOT)
            model_records[experiment_id] = {"path": str(final_relative).replace("\\", "/"), "sha256": sha256_file(model_path),
                                              "candidate_id": selected_rows[experiment_id]["candidate_id"],
                                              "classifier": selected_rows[experiment_id]["classifier"],
                                              "input_mode": selected_rows[experiment_id]["input_mode"],
                                              "vocabulary_size": selected_rows[experiment_id]["vocabulary_size"], "class_order": [0, 1]}
        selected_doc = {"selection_rule": config["selection"], "models": model_records, "selected_candidates": selected_public}
        write_json(staged["selected_models"], selected_doc)
        output_hashes = {"candidate_metrics": sha256_file(staged["candidate_metrics"]),
                         "selected_models": sha256_file(staged["selected_models"]),
                         "dev_predictions": sha256_file(staged["dev_predictions"]),
                         "dummy_metrics": sha256_file(staged["dummy_metrics"])}
        duration = time.perf_counter() - started
        manifest = {
            "baseline_schema_version": config["baseline_schema_version"], "timestamp_utc": timestamp, "status": "COMPLETED",
            "provenance_fingerprint": fingerprint, "input_hashes": protected_before, "code_and_config_hashes": code_hashes,
            "partition_fingerprint": partition_summary["provenance_fingerprint"], "python_and_dependencies": versions,
            "experiments": selected_public, "model_artifacts": model_records, "output_hashes": output_hashes,
            "training_counts": {"train": len(cache[("train", "target")]), "dev_tune": len(cache[("dev_tune", "target")])},
            "modeling_partitions_accessed": ["train", "dev_tune"], "candidate_fits": len(all_candidates),
            "selection_rule": config["selection"], "convergence_failures": [], "duration_seconds": duration,
            "validation": {"status": "PENDING_PREPUBLICATION_VALIDATION"},
            "not_run": ["dev_calibration vectorization or prediction", "test vectorization or prediction", "calibration",
                        "threshold tuning", "selective prediction", "transformer training", "final evaluation"],
        }
        write_json(staged["run_manifest"], manifest)
        staged["report"].parent.mkdir(parents=True, exist_ok=True)
        staged["report"].write_text(markdown_report(manifest, selected_public, dummy), encoding="utf-8", newline="\n")
        validator = load_module("sandarbh_baseline_validator_prepublication", PROJECT_ROOT / "scripts" / "validate_baseline_run.py")
        overrides = {key: staged[key] for key in ("candidate_metrics", "selected_models", "dev_predictions", "dummy_metrics", "run_manifest")}
        overrides["model_directory"] = staged_model_dir
        validation = validator.validate_run(config_path, quiet=True, output_overrides=overrides)
        if not validation["passed"]:
            raise BaselineError("Independent prepublication validation failed: " + "; ".join(validation["failures"]))
        manifest["validation"] = {"status": "PASSED", "checks": validation["checks"]}
        write_json(staged["run_manifest"], manifest)
        staged["report"].write_text(markdown_report(manifest, selected_public, dummy), encoding="utf-8", newline="\n")
        if {name: sha256_file(path) for name, path in paths.items()} != protected_before:
            raise BaselineError("Source or frozen partition artifacts changed during training")
        for key in ("candidate_metrics", "selected_models", "dev_predictions", "dummy_metrics", "run_manifest", "report"):
            outputs[key].parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged[key], outputs[key])
        outputs["model_directory"].mkdir(parents=True, exist_ok=True)
        for model_path in sorted(staged_model_dir.glob("*.joblib")):
            os.replace(model_path, outputs["model_directory"] / model_path.name)
    write_status(outputs["run_status"], "COMPLETED", "Phase 3 baselines completed and passed independent validation.", timestamp,
                 {"provenance_fingerprint": fingerprint, "duration_seconds": time.perf_counter() - started})
    print("Phase 3 baseline training completed: COMPLETED")
    print(f"Manifest: {outputs['run_manifest'].relative_to(PROJECT_ROOT)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train SANDARBH Phase 3 classical baselines.")
    parser.add_argument("--config", default="configs/baselines.json", help="Configuration path relative to project root")
    args = parser.parse_args(argv)
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    status = PROJECT_ROOT / "results" / "baselines" / "run_status.json"
    try:
        return run_training(safe_path(args.config, "configuration"))
    except (BaselineError, audit.AuditInputError) as exc:
        try:
            write_status(status, "FAILED", f"Baseline run failed without replacing completed outputs: {exc}", timestamp)
        except OSError:
            pass
        print(f"Baseline training failed: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        try:
            write_status(status, "FAILED", f"Unexpected failure ({type(exc).__name__}); outputs are not current.", timestamp)
        except OSError:
            pass
        print(f"Baseline training failed unexpectedly: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
