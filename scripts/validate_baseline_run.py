#!/usr/bin/env python3
"""Read-only independent validation for a saved SANDARBH Phase 3 baseline run."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score)
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ("E1", "E2", "E3", "E4")
PACKAGE_NAMES = ("cloudpickle", "joblib", "narwhals", "numpy", "scikit-learn", "scipy", "threadpoolctl")


class ValidationError(Exception):
    pass


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
        raise ValidationError(f"{label} path escapes the project root") from exc
    return path


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"Cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValidationError(f"{label} must be a JSON object")
    return value


def read_csv(path: Path, label: str) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise ValidationError(f"{label} has a missing or duplicate header")
            return list(reader.fieldnames), list(reader)
    except ValidationError:
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValidationError(f"Cannot read {label}: {exc}") from exc


def installed_versions() -> dict[str, str]:
    versions = {name: importlib.metadata.version(name) for name in PACKAGE_NAMES}
    versions["python"] = sys.version.split()[0]
    return versions


def metrics(y_true: list[int], y_pred: list[int], scores: list[float]) -> dict[str, Any]:
    truth, pred, rank = np.asarray(y_true), np.asarray(y_pred), np.asarray(scores, dtype=float)
    if truth.shape != pred.shape or truth.shape != rank.shape or not np.isfinite(rank).all():
        raise ValidationError("Prediction arrays have inconsistent shapes or non-finite scores")
    precision, recall, f1, support = precision_recall_fscore_support(truth, pred, labels=[0, 1], zero_division=0)
    return {"macro_f1": float(f1_score(truth, pred, labels=[0, 1], average="macro", zero_division=0)),
            "positive_precision": float(precision[1]), "positive_recall": float(recall[1]), "positive_f1": float(f1[1]),
            "accuracy": float(accuracy_score(truth, pred)), "balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
            "confusion_matrix_0_1": confusion_matrix(truth, pred, labels=[0, 1]).astype(int).tolist(),
            "support_0_1": support.astype(int).tolist(), "average_precision": float(average_precision_score(truth, rank)),
            "roc_auc": float(roc_auc_score(truth, rank)) if len(set(truth.tolist())) == 2 else None,
            "predicted_positive_count": int(np.sum(pred == 1)), "no_positive_predictions": bool(np.sum(pred == 1) == 0)}


def parse_bool(value: str) -> bool:
    if value == "True": return True
    if value == "False": return False
    raise ValidationError(f"Invalid boolean value {value!r}")


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)


def selection_key(row: dict[str, str]) -> tuple[float, float, int]:
    return (-float(row["macro_f1"]), float(row["C"]), 0 if row["class_weight"] == "null" else 1)


def validate_run(config_path: Path, quiet: bool = False, output_overrides: dict[str, Path] | None = None) -> dict[str, Any]:
    failures: list[str] = []
    checks: dict[str, bool] = {}
    try:
        config = load_json(config_path, "baseline configuration")
        partition_paths = {key: safe_path(value, f"partition artifact {key}") for key, value in config["partition_artifacts"].items()}
        source = safe_path(config["input_csv"], "source")
        outputs = {key: safe_path(value, f"output {key}") for key, value in config["outputs"].items()}
        outputs.update(output_overrides or {})
        required_outputs = ("candidate_metrics", "selected_models", "dev_predictions", "dummy_metrics", "run_manifest", "model_directory")
        for name, path in {"source": source, **partition_paths, **{key: outputs[key] for key in required_outputs}}.items():
            if not path.exists(): failures.append(f"Missing required artifact: {name}")
        if failures:
            return {"passed": False, "checks": checks, "failures": failures}
        manifest = load_json(outputs["run_manifest"], "baseline run manifest")
        selected = load_json(outputs["selected_models"], "selected models")
        if output_overrides is None:
            status = load_json(outputs["run_status"], "baseline run status")
            checks["run_status_completed"] = status.get("status") == "COMPLETED"
            if not checks["run_status_completed"]: failures.append("Baseline run status is not COMPLETED")
        checks["manifest_completed"] = manifest.get("status") == "COMPLETED"
        if not checks["manifest_completed"]: failures.append("Run manifest is not COMPLETED")

        input_paths = {"source": source, **partition_paths}
        input_hashes = {name: sha256_file(path) for name, path in input_paths.items()}
        checks["input_provenance_matches"] = manifest.get("input_hashes") == input_hashes
        if not checks["input_provenance_matches"]: failures.append("Current source/partition hashes differ from the run manifest")
        code_hashes = {"baseline_config": sha256_file(config_path),
                       "training_script": sha256_file(PROJECT_ROOT / "scripts" / "train_baselines.py"),
                       "validation_script": sha256_file(Path(__file__)),
                       "requirements": sha256_file(PROJECT_ROOT / "requirements-baselines.txt")}
        checks["code_config_hashes_match"] = manifest.get("code_and_config_hashes") == code_hashes
        if not checks["code_config_hashes_match"]: failures.append("Current code/config hashes differ from the run manifest")
        versions = installed_versions()
        checks["dependency_versions_match"] = manifest.get("python_and_dependencies") == versions == config.get("expected_versions")
        if not checks["dependency_versions_match"]: failures.append("Python/dependency versions differ from the frozen run")
        partition_summary = load_json(partition_paths["summary"], "partition summary")
        checks["partition_fingerprint_matches"] = manifest.get("partition_fingerprint") == partition_summary.get("provenance_fingerprint") == config.get("required_partition_fingerprint")
        if not checks["partition_fingerprint_matches"]: failures.append("Frozen partition fingerprint mismatch")
        checks["only_allowed_modeling_partitions_accessed"] = manifest.get("modeling_partitions_accessed") == ["train", "dev_tune"]
        if not checks["only_allowed_modeling_partitions_accessed"]: failures.append("Run manifest records forbidden modeling partition access")

        _, candidates = read_csv(outputs["candidate_metrics"], "candidate metrics")
        by_experiment: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in candidates: by_experiment[row.get("experiment_id", "")].append(row)
        checks["six_candidates_per_experiment"] = set(by_experiment) == set(EXPERIMENTS) and all(len(by_experiment[e]) == 6 for e in EXPERIMENTS) and len(candidates) == 24
        if not checks["six_candidates_per_experiment"]: failures.append("Expected exactly six candidate rows for each E1-E4")
        selection_ok = True
        selected_rows = {}
        for experiment_id in EXPERIMENTS:
            rows = by_experiment.get(experiment_id, [])
            if rows:
                valid = [row for row in rows if parse_bool(row["converged"])]
                expected = min(valid, key=selection_key) if valid else None
                marked = [row for row in rows if parse_bool(row["selected"])]
                if expected is None or len(marked) != 1 or marked[0]["candidate_id"] != expected["candidate_id"]:
                    selection_ok = False
                else: selected_rows[experiment_id] = marked[0]
        checks["candidate_selection_rule_matches"] = selection_ok
        if not selection_ok: failures.append("Saved candidate selection does not match macro-F1/C/class-weight rule")

        models = selected.get("models", {})
        selected_candidates = selected.get("selected_candidates", {})
        checks["exactly_four_selected_models"] = set(models) == set(EXPERIMENTS) == set(selected_candidates) and len(models) == 4
        if not checks["exactly_four_selected_models"]: failures.append("Selected-model metadata does not contain exactly E1-E4")
        for experiment_id in EXPERIMENTS:
            if experiment_id in models and experiment_id in selected_rows and models[experiment_id].get("candidate_id") != selected_rows[experiment_id]["candidate_id"]:
                failures.append(f"Selected model/candidate mismatch for {experiment_id}")

        _, split_rows = read_csv(partition_paths["split_manifest"], "split manifest")
        dev_rows = [row for row in split_rows if row["partition"] == "dev_tune"]
        dev_by_id = {row["example_id"]: row for row in dev_rows}
        _, target_rows = read_csv(partition_paths["targets"], "targets")
        labels = {row["example_id"]: int(row["hard_label"]) for row in target_rows}
        _, predictions = read_csv(outputs["dev_predictions"], "development predictions")
        prediction_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in predictions: prediction_groups[row.get("experiment_id", "")].append(row)
        ids_ok = set(prediction_groups) == set(EXPERIMENTS)
        output_values_ok = True
        metrics_ok = True
        for experiment_id in EXPERIMENTS:
            rows = prediction_groups.get(experiment_id, [])
            row_ids = [row.get("example_id", "") for row in rows]
            ids_ok &= len(row_ids) == len(set(row_ids)) == len(dev_by_id) and set(row_ids) == set(dev_by_id)
            ids_ok &= all(row.get("partition") == "dev_tune" and row.get("group_id") == dev_by_id.get(row.get("example_id", ""), {}).get("group_id") for row in rows)
            if not rows: continue
            rows.sort(key=lambda row: row["example_id"])
            try:
                truth = [int(row["true_hard_label"]) for row in rows]
                pred = [int(row["predicted_label"]) for row in rows]
                decision = [float(row["decision_score"]) for row in rows]
                output_values_ok &= all(labels[row["example_id"]] == y for row, y in zip(rows, truth))
                family = selected_candidates.get(experiment_id, {}).get("classifier")
                if family == "logistic_regression":
                    probability = [float(row["positive_probability"]) for row in rows]
                    output_values_ok &= all(math.isfinite(value) and 0 <= value <= 1 for value in probability)
                    ranking = probability
                else:
                    output_values_ok &= all(row["positive_probability"] == "" for row in rows)
                    ranking = decision
                output_values_ok &= all(math.isfinite(value) for value in decision) and set(pred).issubset({0, 1})
                calculated = metrics(truth, pred, ranking)
                saved = selected_rows.get(experiment_id, {})
                for key in ("macro_f1", "positive_precision", "positive_recall", "positive_f1", "accuracy", "balanced_accuracy", "average_precision"):
                    metrics_ok &= key in saved and close(calculated[key], float(saved[key]))
                saved_roc = None if saved.get("roc_auc", "") == "" else float(saved["roc_auc"])
                metrics_ok &= (calculated["roc_auc"] is None and saved_roc is None) or (calculated["roc_auc"] is not None and saved_roc is not None and close(calculated["roc_auc"], saved_roc))
                metrics_ok &= calculated["confusion_matrix_0_1"] == json.loads(saved.get("confusion_matrix_0_1", "null"))
                metrics_ok &= calculated["support_0_1"] == json.loads(saved.get("support_0_1", "null"))
                metrics_ok &= calculated["predicted_positive_count"] == int(saved.get("predicted_positive_count", -1))
                metrics_ok &= calculated["no_positive_predictions"] == parse_bool(saved.get("no_positive_predictions", ""))
            except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                output_values_ok = False; metrics_ok = False
        checks["dev_prediction_ids_exactly_once"] = ids_ok
        checks["no_calibration_or_test_predictions"] = all(row.get("partition") == "dev_tune" for row in predictions)
        checks["prediction_values_valid"] = output_values_ok
        checks["selected_metrics_reconcile"] = metrics_ok
        if not ids_ok: failures.append("Saved prediction IDs do not match dev_tune exactly once per experiment")
        if not checks["no_calibration_or_test_predictions"]: failures.append("Predictions include a forbidden partition")
        if not output_values_ok: failures.append("Prediction scores, probabilities, labels, or score semantics are invalid")
        if not metrics_ok: failures.append("Selected candidate metrics do not reconcile with saved predictions")

        output_hashes = manifest.get("output_hashes", {})
        named_outputs = {"candidate_metrics": outputs["candidate_metrics"], "selected_models": outputs["selected_models"],
                         "dev_predictions": outputs["dev_predictions"], "dummy_metrics": outputs["dummy_metrics"]}
        checks["tabular_output_hashes_match"] = all(output_hashes.get(key) == sha256_file(path) for key, path in named_outputs.items())
        if not checks["tabular_output_hashes_match"]: failures.append("A saved tabular output hash differs from the run manifest")
        model_hashes_ok = True
        pipelines_load = True
        for experiment_id, record in models.items():
            model_path = outputs["model_directory"] / Path(record["path"]).name
            if not model_path.is_file() or sha256_file(model_path) != record.get("sha256"):
                model_hashes_ok = False
                continue
            try:
                pipeline = joblib.load(model_path)
                pipelines_load &= isinstance(pipeline, Pipeline) and set(pipeline.named_steps) == {"tfidf", "classifier"}
                pipelines_load &= pipeline.named_steps["classifier"].classes_.astype(int).tolist() == [0, 1]
                pipelines_load &= len(pipeline.named_steps["tfidf"].vocabulary_) == record.get("vocabulary_size")
            except Exception:
                pipelines_load = False
        checks["model_artifact_hashes_match"] = model_hashes_ok and len(models) == 4
        checks["saved_pipelines_load_successfully"] = pipelines_load and len(models) == 4
        if not checks["model_artifact_hashes_match"]: failures.append("Model artifact missing or hash mismatch")
        if not checks["saved_pipelines_load_successfully"]: failures.append("A hash-verified selected pipeline does not load or reconcile")
    except (ValidationError, KeyError, ValueError, TypeError, OSError) as exc:
        failures.append(str(exc))
    passed = not failures and all(checks.values())
    result = {"passed": passed, "checks": checks, "failures": failures}
    if not quiet: print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a completed SANDARBH Phase 3 baseline run without fitting models.")
    parser.add_argument("--config", default="configs/baselines.json", help="Configuration path relative to project root")
    args = parser.parse_args(argv)
    try:
        result = validate_run(safe_path(args.config, "configuration"))
        return 0 if result["passed"] else 1
    except Exception as exc:
        print(f"Baseline validation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
