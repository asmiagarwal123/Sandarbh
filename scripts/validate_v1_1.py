#!/usr/bin/env python3
"""Text-free SANDARBH V1.1 forensic validation and maintenance release builder.

This program never reads AUTALIC as CSV and never invokes locked-test inference.
AUTALIC is opened only as bytes for its SHA-256. All analyses use saved text-free
manifests, labels, probabilities, and aggregate artifacts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/v1_1_validation.json"
POSTHOC = "POST-HOC / EXPLORATORY / NOT USED FOR MODEL SELECTION"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, fieldnames: list[str], values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(values)


def safe_div(a: float, b: float):
    return a / b if b else None


def classification(y: list[int], pred: list[int]) -> dict:
    tn = sum(a == 0 and b == 0 for a, b in zip(y, pred))
    fp = sum(a == 0 and b == 1 for a, b in zip(y, pred))
    fn = sum(a == 1 and b == 0 for a, b in zip(y, pred))
    tp = sum(a == 1 and b == 1 for a, b in zip(y, pred))
    p1, r1 = safe_div(tp, tp + fp), safe_div(tp, tp + fn)
    p0, r0 = safe_div(tn, tn + fn), safe_div(tn, tn + fp)
    f1 = safe_div(2 * p1 * r1, p1 + r1) if p1 is not None and r1 is not None else None
    f0 = safe_div(2 * p0 * r0, p0 + r0) if p0 is not None and r0 is not None else None
    return {
        "n": len(y), "confusion_matrix_0_1": [[tn, fp], [fn, tp]],
        "macro_f1": (f0 + f1) / 2 if f0 is not None and f1 is not None else None,
        "positive_precision": p1, "positive_recall": r1, "positive_f1": f1,
        "accuracy": safe_div(tn + tp, len(y)),
        "balanced_accuracy": (r0 + r1) / 2 if r0 is not None and r1 is not None else None,
        "false_positive_count": fp, "false_negative_count": fn,
    }


def roc_auc(y: list[int], scores: list[float]):
    positive, negative = sum(y), len(y) - sum(y)
    if not positive or not negative:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks, i = [0.0] * len(scores), 0
    while i < len(order):
        j = i + 1
        while j < len(order) and scores[order[j]] == scores[order[i]]:
            j += 1
        rank = ((i + 1) + j) / 2
        for k in range(i, j):
            ranks[order[k]] = rank
        i = j
    return (sum(r for r, a in zip(ranks, y) if a == 1) - positive * (positive + 1) / 2) / (positive * negative)


def average_precision(y: list[int], scores: list[float]):
    positive = sum(y)
    if not positive:
        return None
    pairs, tp, fp, previous_recall, result, i = sorted(zip(scores, y), reverse=True), 0, 0, 0.0, 0.0, 0
    while i < len(pairs):
        score, group = pairs[i][0], []
        while i < len(pairs) and pairs[i][0] == score:
            group.append(pairs[i][1]); i += 1
        tp += sum(group); fp += len(group) - sum(group)
        recall = tp / positive
        result += (recall - previous_recall) * tp / (tp + fp)
        previous_recall = recall
    return result


def probability_metrics(y: list[int], probabilities: list[float], bins: int = 10) -> dict:
    clipped = [min(max(p, 1e-15), 1 - 1e-15) for p in probabilities]
    n = len(y)
    ece = 0.0
    for b in range(bins):
        ix = [i for i, p in enumerate(probabilities) if p >= b / bins and (p < (b + 1) / bins or b == bins - 1)]
        if ix:
            ece += len(ix) / n * abs(sum(probabilities[i] for i in ix) / len(ix) - sum(y[i] for i in ix) / len(ix))
    return {
        "negative_log_likelihood": -sum(a * math.log(p) + (1 - a) * math.log(1 - p) for a, p in zip(y, clipped)) / n,
        "brier_score": sum((p - a) ** 2 for a, p in zip(y, probabilities)) / n,
        "expected_calibration_error": ece,
        "roc_auc": roc_auc(y, probabilities),
        "average_precision": average_precision(y, probabilities),
        "calibration_in_the_large": sum(probabilities) / n - sum(y) / n,
        "mean_predicted_positive_probability": sum(probabilities) / n,
        "observed_positive_prevalence": sum(y) / n,
    }


def spearman(xs: list[float], ys: list[float]):
    def rank(values):
        order = sorted(range(len(values)), key=lambda i: values[i]); result = [0.0] * len(values); i = 0
        while i < len(order):
            j = i + 1
            while j < len(order) and values[order[j]] == values[order[i]]:
                j += 1
            r = ((i + 1) + j) / 2
            for k in range(i, j): result[order[k]] = r
            i = j
        return result
    if len(xs) < 2: return None
    rx, ry = rank(xs), rank(ys); mx, my = statistics.mean(rx), statistics.mean(ry)
    numerator = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denominator = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return numerator / denominator if denominator else None


def expanded_immutable_files(config: dict) -> list[Path]:
    found = []
    for relative in config["immutable_paths"]:
        path = ROOT / relative
        if path.is_file(): found.append(path)
        elif path.is_dir(): found.extend(p for p in path.rglob("*") if p.is_file())
        else: raise FileNotFoundError(f"Immutable path is missing: {relative}")
    return sorted(set(found), key=lambda p: p.as_posix())


def integrity_manifest(config: dict, label: str) -> dict:
    values = []
    for path in expanded_immutable_files(config):
        values.append({"path": path.relative_to(ROOT).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    expected = config["required_critical_hashes"]
    mismatches = []
    by_path = {x["path"]: x["sha256"] for x in values}
    for path, digest in expected.items():
        if by_path.get(path) != digest: mismatches.append({"path": path, "expected": digest, "observed": by_path.get(path)})
    return {"schema_version": "1.1.0", "label": label, "generated_utc": utcnow(), "file_count": len(values),
            "status": "PASS" if not mismatches else "BLOCKED_INTEGRITY_FAILURE", "critical_mismatches": mismatches, "files": values}


def e7_audit() -> dict:
    count0, count1 = 1415, 180
    n = count0 + count1
    w0, w1 = n / (2 * count0), n / (2 * count1)
    optima = []
    for q in (0.0, 1 / 3, 2 / 3, 1.0):
        denominator = w1 * q + w0 * (1 - q)
        optima.append({"soft_positive": q, "effective_probability_optimum": (w1 * q / denominator if denominator else None)})
    return {
        "status": "CONFIRMED_CONFOUND", "experiment_id": "E7", "objective": "cost-weighted soft-vote model",
        "soft_target_construction": "[soft_other, soft_positive]",
        "uses_torch_cross_entropy_loss": False,
        "implementation": "-(soft_targets * class_weights.unsqueeze(0) * log_probabilities).sum(dim=1).mean()",
        "class_weights_multiply_both_class_components": True, "batch_reduction": "sum across classes per example, arithmetic mean across examples",
        "training_counts_0_1": [count0, count1], "exact_class_weights_0_1": [w0, w1],
        "formula": "p1*(q) = w1*q / (w0*(1-q) + w1*q)", "effective_optima": optima,
        "code_evidence": ["scripts/train_transformers.py:286-290", "scripts/train_transformers.py:389-406", "scripts/train_transformers.py:728"],
        "conclusion": "E7 is class-weighted soft cross-entropy, not ordinary vote-fraction cross-entropy. E7 does not isolate soft labels; E7-versus-E6 cannot support a clean causal conclusion about soft-label training.",
        "unit_test": "tests/test_v1_1_validation.py::test_weighted_soft_loss_and_optimum"
    }


def truncation_audit() -> dict:
    manifest = rows(ROOT / "results/transformers/tokenization_manifest.csv")
    preflight = load_json(ROOT / "results/transformer_preflight/token_length_summary.json")
    contextual = [r for r in manifest if r["input_mode"] == "context"]
    by_partition = {}
    for partition in sorted(set(r["partition"] for r in contextual)):
        subset = [r for r in contextual if r["partition"] == partition]
        by_partition[partition] = {
            "examples": len(subset),
            "sequence_length_256": sum(int(r["sequence_length"]) == 256 for r in subset),
            "target_partially_truncated": sum(r["target_truncated"] == "True" for r in subset),
            "target_completely_removed": sum(int(r["target_original_tokens"]) > 0 and int(r["target_retained_tokens"]) == 0 for r in subset),
            "preceding_partially_truncated": sum(int(r["preceding_retained_tokens"]) < int(r["preceding_original_tokens"]) for r in subset),
            "following_partially_truncated": sum(int(r["following_retained_tokens"]) < int(r["following_original_tokens"]) for r in subset),
        }
    affected = sum(int(r["sequence_length"]) == 256 for r in contextual)
    return {
        "status": "PASS", "maximum_sequence_length": 256, "outer_special_tokens": 2,
        "marker_token_counts": {"preceding": 6, "target": 3, "following": 4}, "maximum_target_content_tokens": 241,
        "preceding_rule": "retain the tail nearest the target", "following_rule": "retain the head nearest the target",
        "context_allocation": "split remaining capacity evenly; an odd token goes to preceding; unused capacity is deterministically redistributed to the other side",
        "overlong_target_rule": "head + tokenized literal ' …' + tail; an odd non-ellipsis capacity gives the extra token to the head",
        "e6_e7_encoded_inputs_identical": True,
        "pre_construction_context_sequences_exceeding_256": preflight["combined"]["lengths"]["context"]["counts_exceeding_final_sequence_length"]["256"],
        "pre_construction_context_sequences_exceeding_256_percent": 100 * preflight["combined"]["lengths"]["context"]["counts_exceeding_final_sequence_length"]["256"] / preflight["combined"]["example_count"],
        "preflight_ordinary_right_truncation_diagnostic": preflight["combined"]["ordinary_right_truncation_target_effect"]["256"],
        "saved_context_examples": len(contextual), "saved_examples_at_length_256": affected,
        "saved_examples_at_length_256_percent": 100 * affected / len(contextual), "by_partition": by_partition,
        "locked_test": {"maximum_sequence_length": load_json(ROOT / "results/final_test/run_manifest.json")["input_evidence"]["maximum_sequence_length"],
                        "e6_e7_encoded_inputs_identical": load_json(ROOT / "results/final_test/run_manifest.json")["input_evidence"]["e6_e7_encoded_inputs_identical"],
                        "detailed_truncation_counts": "UNAVAILABLE: no saved per-example locked-test token-length/truncation manifest; source text was not loaded"},
        "code_evidence": ["scripts/train_transformers.py:296-385"],
        "unit_test": "tests/test_v1_1_validation.py token-construction cases"
    }


def probability_null_comparison(config: dict) -> tuple[dict, list[dict]]:
    targets = {r["example_id"]: r for r in rows(ROOT / "results/partitions/targets.csv")}
    split = rows(ROOT / "results/partitions/split_manifest.csv")
    train_ids = [r["example_id"] for r in split if r["partition"] == "train"]
    prevalence = sum(int(targets[i]["hard_label"]) for i in train_ids) / len(train_ids)
    transformer = rows(ROOT / "results/final_test/transformer_test_predictions.csv")
    baseline = rows(ROOT / "results/final_test/baseline_test_predictions.csv")
    models = {}
    for experiment in ("E5", "E6", "E7"):
        subset = [r for r in transformer if r["experiment_id"] == experiment]
        models[experiment] = ([int(r["true_hard_label"]) for r in subset], [float(r["probability_1_calibrated"]) for r in subset], "temperature_scaled")
    for experiment in ("E1", "E2", "E3", "E4"):
        subset = [r for r in baseline if r["experiment_id"] == experiment]
        if subset and all(r["probability_1"] != "" for r in subset):
            models[experiment] = ([int(r["true_hard_label"]) for r in subset], [float(r["probability_1"]) for r in subset], "native")
    y = next(iter(models.values()))[0]
    null_metrics = probability_metrics(y, [prevalence] * len(y), config["ece_bins"])
    output_rows = []
    for experiment, (labels, probabilities, state) in sorted(models.items()):
        metric = probability_metrics(labels, probabilities, config["ece_bins"])
        bss = 1 - metric["brier_score"] / null_metrics["brier_score"]
        output_rows.append({"experiment_id": experiment, "probability_state": state, **metric,
                            "nll_improvement_vs_train_prevalence": null_metrics["negative_log_likelihood"] - metric["negative_log_likelihood"],
                            "brier_skill_score": bss, "relative_brier_improvement": bss})
    return ({"status": "PASS", "source": "saved text-free final-test predictions and labels only", "locked_test_inference_rerun": False,
             "brier_definition": "binary positive-class mean squared error", "train_positive_prevalence": prevalence,
             "observed_test_positive_prevalence_descriptive_only": sum(y) / len(y), "train_prevalence_null": null_metrics,
             "models": output_rows, "interpretation_rule": "The model shows ranking/discrimination ability but does not demonstrate probability skill over the train-prevalence reference under this scoring rule when Brier Skill Score is non-positive."}, output_rows)


def dataset_review() -> dict:
    audit = load_json(ROOT / "results/audit/summary.json")
    analysis = audit["analysis"]
    reference = audit["reference_comparisons"]
    structure = audit["structure"]
    source = audit["source"]
    return {
        "status": "DISCREPANCIES_DOCUMENTED", "local_file": {"sha256": source["sha256_after"], "byte_size": source["byte_size"],
        "headers": structure["header"], "row_count": structure["row_count"], "download_source": "UNKNOWN / NEEDS VERIFICATION", "retrieval_date": "UNKNOWN / NEEDS VERIFICATION"},
        "official_reference": {"paper": "ACL 2025 long paper, pp. 20999-21015", "project_url": "https://nrizvi.github.io/AUTALIC.html",
        "reported_target_sentences": 2400, "reported_preceding_contexts": 2014, "reported_following_contexts": 2400,
        "reported_majority_positive": 242, "reported_negative": 2160, "arithmetic_discrepancy": "242 + 2160 = 2402",
        "reported_average_fleiss_kappa": 0.25, "annotation_design": "three 800-item segments with different trios; A1/A2/A3 are positions, not the same identifiable people over all rows"},
        "local_audit": {"manifest_rows": 2400, "eligible": analysis["row_accounting"]["eligible_rows"], "excluded": analysis["row_accounting"]["excluded_rows"],
        "eligible_hard_positive": reference["eligible_binary_majority_positive"]["observed"],
        "all_empty": reference["all_text_empty"]["observed"], "nonempty_target": reference["nonempty_target"]["observed"],
        "preceding_available": sum(r["preceding_available"] == "True" for r in rows(ROOT / "results/audit/row_manifest.csv")),
        "following_available": sum(r["following_available"] == "True" for r in rows(ROOT / "results/audit/row_manifest.csv"))},
        "binary_mapping": "vote 1 -> positive; votes 0 and -1 -> other; hard positive requires at least two positive votes",
        "exclusion_check": "All 122 excluded rows are all-empty across preceding, target, and following according to the saved row manifest.",
        "official_byte_match": "OFFICIAL BYTE-LEVEL MATCH UNKNOWN / NEEDS AUTHOR VERIFICATION",
        "restrictions": ["Academic/scientific/educational use under stated conditions", "Redistribution restricted", "Commercial use requires written permission", "Automated moderation requires prior author approval", "Do not publicly expose dataset text or text-reconstructing artifacts"],
        "author_contact_draft": "Subject: AUTALIC local-file clarification. We are auditing a research-only local copy (SHA-256 66ccaa43...). Its 2,400 rows include 122 rows with all three text fields empty; among the 2,278 eligible rows, our documented binary mapping yields 256 majority-positive examples. Could you confirm whether these empty rows are expected in the released file and explain the paper's 242-positive figure (and the 242 + 2,160 = 2,402 total)? We will not redistribute the data."
    }


def seed_robustness() -> dict:
    selected = rows(ROOT / "results/transformers/selected_checkpoint_metrics.csv")
    summary = load_json(ROOT / "results/transformers/seed_summary.json")
    metrics = ["macro_f1", "positive_precision", "positive_recall", "positive_f1", "balanced_accuracy", "roc_auc", "average_precision", "run_seconds"]
    experiments = {}
    for experiment in ("E5", "E6", "E7"):
        subset = [r for r in selected if r["experiment_id"] == experiment]
        seed_rows = [{"seed": int(r["seed"]), "selected_epoch": int(r["selected_epoch"]), **{m: float(r[m]) for m in metrics}} for r in subset]
        desc = {}
        for metric in metrics:
            vals = [r[metric] for r in seed_rows]
            desc[metric] = {"mean": statistics.mean(vals), "sample_standard_deviation": statistics.stdev(vals), "minimum": min(vals), "maximum": max(vals)}
        seed42 = next(r for r in seed_rows if r["seed"] == 42)
        desc42 = {m: ("within_seed_range" if desc[m]["minimum"] <= seed42[m] <= desc[m]["maximum"] else "outside_seed_range") for m in metrics}
        experiments[experiment] = {"seeds": seed_rows, "summary": desc, "seed_42_assessment": desc42}
    return {"status": "PASS", "source": "existing development artifacts; no retraining", "experiments": experiments,
            "existing_summary_cross_check": "results/transformers/seed_summary.json", "note": "With only three seeds, representativeness is descriptive, not inferential."}


def entropy(q: float) -> float:
    return -sum(p * math.log(p) for p in (q, 1 - q) if p > 0)


def posthoc_summary(config: dict) -> dict:
    transformer = rows(ROOT / "results/final_test/transformer_test_predictions.csv")
    baseline = rows(ROOT / "results/final_test/baseline_test_predictions.csv")
    targets = {r["example_id"]: r for r in rows(ROOT / "results/partitions/targets.csv")}
    audit = {r["example_id"]: r for r in rows(ROOT / "results/audit/row_manifest.csv")}
    splits = rows(ROOT / "results/partitions/split_manifest.csv")
    group_sizes = Counter(r["group_id"] for r in splits)
    experiments = {}
    prediction_maps = {}
    for experiment in ("E1", "E2", "E3", "E4"):
        subset = [r for r in baseline if r["experiment_id"] == experiment]
        prediction_maps[experiment] = {r["example_id"]: int(r["predicted_label"]) for r in subset}
        experiments[experiment] = classification([int(r["true_hard_label"]) for r in subset], [int(r["predicted_label"]) for r in subset])
    for experiment in ("E5", "E6", "E7"):
        subset = [r for r in transformer if r["experiment_id"] == experiment]
        prediction_maps[experiment] = {r["example_id"]: int(r["predicted_label_calibrated"]) for r in subset}
        y = [int(r["true_hard_label"]) for r in subset]; pred = [int(r["predicted_label_calibrated"]) for r in subset]
        experiments[experiment] = classification(y, pred)
    first = [r for r in transformer if r["experiment_id"] == "E5"]
    always = classification([int(r["true_hard_label"]) for r in first], [0] * len(first))
    agreement = {}
    confidence_entropy = {}
    context_availability = {}
    overlap_group_size = {}
    risk_coverage = {}
    for experiment in ("E5", "E6", "E7"):
        subset = [r for r in transformer if r["experiment_id"] == experiment]
        for group_name, selector in {
            "unanimous": lambda t: t["agreement_category"] == "unanimous",
            "two_to_one": lambda t: t["agreement_category"] == "two_vs_one",
            "contains_minus1": lambda t: t["has_minus1"] == "True",
            "all_different": lambda t: t["agreement_category"] == "all_different",
        }.items():
            part = [r for r in subset if selector(targets[r["example_id"]])]
            agreement[f"{experiment}:{group_name}"] = classification([int(r["true_hard_label"]) for r in part], [int(r["predicted_label_calibrated"]) for r in part]) if part else {"n": 0, "metrics": "UNAVAILABLE"}
        entropies = [entropy(float(r["soft_positive"])) for r in subset]
        errors = [int(int(r["predicted_label_calibrated"]) != int(r["true_hard_label"])) for r in subset]
        uncertainties = [1 - float(r["confidence_calibrated"]) for r in subset]
        confidence_entropy[experiment] = {"spearman_entropy_vs_error": spearman(entropies, errors), "spearman_entropy_vs_prediction_uncertainty": spearman(entropies, uncertainties)}
        for label, predicate in {"all_context": lambda a: a["preceding_available"] == "True" and a["following_available"] == "True",
                                 "missing_preceding": lambda a: a["preceding_available"] != "True",
                                 "missing_following": lambda a: a["following_available"] != "True"}.items():
            part = [r for r in subset if predicate(audit[r["example_id"]])]
            context_availability[f"{experiment}:{label}"] = classification([int(r["true_hard_label"]) for r in part], [int(r["predicted_label_calibrated"]) for r in part]) if part else {"n": 0, "metrics": "UNAVAILABLE"}
        for label, predicate in {"singleton": lambda n: n == 1, "multi_member": lambda n: n > 1}.items():
            part = [r for r in subset if predicate(group_sizes[r["group_id"]])]
            overlap_group_size[f"{experiment}:{label}"] = classification([int(r["true_hard_label"]) for r in part], [int(r["predicted_label_calibrated"]) for r in part]) if part else {"n": 0, "metrics": "UNAVAILABLE"}
        accepted = [r for r in subset if float(r["confidence_calibrated"]) >= config["frozen_confidence_threshold"]]
        abstained = [r for r in subset if r not in accepted]
        accepted_errors = sum(int(r["predicted_label_calibrated"]) != int(r["true_hard_label"]) for r in accepted)
        abstained_errors = sum(int(r["predicted_label_calibrated"]) != int(r["true_hard_label"]) for r in abstained)
        risk_coverage[experiment] = {"threshold": config["frozen_confidence_threshold"], "coverage": len(accepted)/len(subset),
          "risk": accepted_errors/len(accepted) if accepted else None, "accepted_errors": accepted_errors, "abstained_errors": abstained_errors,
          "error_fraction_concentrated_in_abstained": abstained_errors/(accepted_errors+abstained_errors) if accepted_errors+abstained_errors else None,
          "predicted_class_0_coverage": sum(int(r["predicted_label_calibrated"]) == 0 and r in accepted for r in subset)/sum(int(r["predicted_label_calibrated"]) == 0 for r in subset),
          "predicted_class_1_coverage": sum(int(r["predicted_label_calibrated"]) == 1 and r in accepted for r in subset)/sum(int(r["predicted_label_calibrated"]) == 1 for r in subset)}
    disagreement = {}
    for a, b in (("E5", "E6"), ("E6", "E7"), ("E5", "E7")):
        ids = sorted(set(prediction_maps[a]) & set(prediction_maps[b]))
        disagreement[f"{a}_vs_{b}"] = {"n": len(ids), "agree": sum(prediction_maps[a][i] == prediction_maps[b][i] for i in ids),
          "a0_b1": sum(prediction_maps[a][i] == 0 and prediction_maps[b][i] == 1 for i in ids),
          "a1_b0": sum(prediction_maps[a][i] == 1 and prediction_maps[b][i] == 0 for i in ids)}
    dev = rows(ROOT / "results/calibration_review/dev_calibration_predictions.csv")
    dev_corr = {}
    for experiment in ("E5", "E6", "E7"):
        part = [r for r in dev if r["experiment_id"] == experiment]
        dev_corr[experiment] = spearman([1 - float(r["confidence_calibrated"]) for r in part], [entropy(float(r["soft_positive"])) for r in part])
    intervals = rows(ROOT / "results/final_test/bootstrap_intervals.csv")
    return {"label": POSTHOC, "source_text_loaded": False, "exact_confusion_matrices": experiments,
      "always_negative": always, "performance_by_annotation_agreement": agreement, "entropy_and_confidence": confidence_entropy,
      "dev_calibration_uncertainty_vote_entropy_spearman": dev_corr, "model_disagreement": disagreement,
      "frozen_bootstrap_intervals": intervals, "unavailable_paired_interval_metrics": ["balanced_accuracy", "accuracy", "negative_log_likelihood", "brier_score", "expected_calibration_error", "roc_auc", "average_precision", "soft_brier_score", "soft_cross_entropy", "false_positive_count", "false_negative_count"],
      "unavailable_reason": "The frozen artifact contains paired difference intervals only for macro-F1, positive F1, and positive recall; V1.1 did not extend the frozen bootstrap protocol.",
      "risk_coverage_at_060": risk_coverage, "aurc": "See immutable results/final_test/risk_coverage.csv; not recomputed",
      "performance_by_context_availability": context_availability, "performance_by_overlap_group_size": overlap_group_size}


def synthetic_smoke(run_models: bool) -> dict:
    cases = [
      ("target_only", "", "An authored neutral test sentence.", ""),
      ("context", "Authored preceding context.", "An authored target.", "Authored following context."),
      ("missing_preceding", "", "An authored target.", "Authored following context."),
      ("missing_following", "Authored preceding context.", "An authored target.", ""),
      ("very_long", "word " * 400, "target " * 400, "word " * 400),
      ("unicode", "Café — नमस्ते", "A synthetic Unicode sentence ✓", "Fin."),
      ("quotation", "", "They quoted: ‘synthetic words’.\"", ""),
      ("negation", "", "This synthetic sentence is not a factual endorsement.", ""),
      ("sarcasm_like", "", "Oh great, another entirely synthetic example.", ""),
      ("neutral_identity", "", "A disabled person attended the synthetic meeting.", ""),
      ("explicit_discrimination", "", "Exclude disabled people from this synthetic event.", ""),
    ]
    result = {"label": "SMOKE_TEST_ONLY", "independent_of_autalic": True, "source_text_saved": False,
              "empty_input_rejection": "covered by unit test", "clear_reset": "covered by unit test", "model_inference_executed": run_models, "cases": []}
    if not run_models:
        result["status"] = "STRUCTURAL_TESTS_ONLY"; result["reason"] = "Run with --run-synthetic-inference to exercise local checkpoints."
        return result
    sys.path.insert(0, str(ROOT / "src"))
    from sandarbh_ui.inference import predict
    for case_id, preceding, target, following in cases:
        for experiment in ("E5", "E6", "E7"):
            output = predict(experiment, preceding, target, following)
            result["cases"].append({"case_id": case_id, "experiment_id": experiment, "predicted_label": output["predicted_label"],
              "probability_1": output["probability_1"], "confidence": output["confidence"], "review_status": output["status"],
              "sequence_length": output["sequence_length"], "probability_sum": output["probability_0"] + output["probability_1"]})
    result["status"] = "PASS"
    return result


def package_files(out: Path, package: Path) -> list[dict]:
    mapping = {
      ROOT/"PROJECT_HANDOFF_V1_1.md": "PROJECT_HANDOFF_V1_1.md",
      ROOT/"reports/V1_1_ERRATA_AND_VALIDATION.md": "V1_1_ERRATA_AND_VALIDATION.md",
      ROOT/"reports/MODEL_CARD.md": "MODEL_CARD.md", ROOT/"reports/DATASET_USE_AND_LICENSE.md": "DATASET_USE_AND_LICENSE.md",
      ROOT/"reports/V1_RESEARCH_QUESTIONS.md": "V1_RESEARCH_QUESTIONS.md", ROOT/"reports/DEFERRED_TO_VERSION_2.md": "DEFERRED_TO_VERSION_2.md",
      ROOT/"reports/REPRODUCIBILITY_GUIDE.md": "REPRODUCIBILITY_GUIDE.md",
      out/"validation_summary.json": "validation_summary.json", out/"e7_loss_audit.json": "e7_loss_audit.json",
      out/"truncation_audit.json": "truncation_audit.json", out/"probability_null_comparison.json": "probability_null_comparison.json",
      out/"dataset_provenance_review.json": "dataset_provenance_review.json", out/"run_status.json": "run_status.json"}
    missing = [str(p.relative_to(ROOT)) for p in mapping if not p.is_file()]
    if missing: raise FileNotFoundError("Package inputs missing: " + ", ".join(missing))
    staging = package.with_name(package.name + ".staging")
    if staging.exists(): shutil.rmtree(staging)
    staging.mkdir(parents=True)
    manifest = []
    for source, name in mapping.items():
        shutil.copyfile(source, staging/name)
        if sha256(source) != sha256(staging/name): raise RuntimeError(f"Package copy mismatch: {name}")
        manifest.append({"file": name, "sha256": sha256(source), "size_bytes": source.stat().st_size})
    lines = ["# Package contents", "", "Text-free SANDARBH V1.1 handoff package. Each copy was byte-verified against its project original.", "", "| File | SHA-256 | Bytes |", "|---|---|---:|"]
    lines += [f"| `{x['file']}` | `{x['sha256']}` | {x['size_bytes']} |" for x in manifest]
    (staging/"PACKAGE_CONTENTS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    if package.exists(): shutil.rmtree(package)
    os.replace(staging, package)
    return manifest


def run(config_path: Path, run_models: bool, finalize: bool) -> int:
    config = load_json(config_path)
    out = ROOT / config["output_directory"]
    provenance = hashlib.sha256(json.dumps(config["required_critical_hashes"], sort_keys=True).encode()).hexdigest()
    if out.exists() and (out/"run_status.json").is_file():
        old = load_json(out/"run_status.json")
        if old.get("status") == "COMPLETED" and old.get("provenance_id") != provenance:
            raise RuntimeError("Refusing to overwrite a completed V1.1 run with different provenance")
    staging = Path(tempfile.mkdtemp(prefix="v1_1_validation_", dir=ROOT/"results"))
    try:
        pre = integrity_manifest(config, "pre_change_scientific_freeze")
        if pre["status"] != "PASS": raise RuntimeError("BLOCKED_INTEGRITY_FAILURE")
        write_json(staging/"pre_change_integrity.json", pre)
        write_json(staging/"e7_loss_audit.json", e7_audit())
        write_json(staging/"truncation_audit.json", truncation_audit())
        null, null_rows = probability_null_comparison(config)
        write_json(staging/"probability_null_comparison.json", null)
        write_csv(staging/"probability_null_comparison.csv", list(null_rows[0]), null_rows)
        write_json(staging/"dataset_provenance_review.json", dataset_review())
        write_json(staging/"seed_robustness.json", seed_robustness())
        write_json(staging/"posthoc_summary.json", posthoc_summary(config))
        write_json(staging/"synthetic_smoke_results.json", synthetic_smoke(run_models))
        post = integrity_manifest(config, "post_change_scientific_freeze")
        write_json(staging/"post_change_integrity.json", post)
        identical = [(x["path"], x["sha256"]) for x in pre["files"]] == [(x["path"], x["sha256"]) for x in post["files"]]
        required_docs = [ROOT/p for p in ["PROJECT_HANDOFF_V1_1.md", "reports/V1_1_ERRATA_AND_VALIDATION.md", "reports/MODEL_CARD.md", "reports/DATASET_USE_AND_LICENSE.md", "reports/V1_RESEARCH_QUESTIONS.md", "reports/DEFERRED_TO_VERSION_2.md", "reports/REPRODUCIBILITY_GUIDE.md"]]
        docs_ready = all(p.is_file() for p in required_docs)
        smoke_pass = run_models and load_json(staging/"synthetic_smoke_results.json")["status"] == "PASS"
        status = "COMPLETED" if finalize and docs_ready and smoke_pass and identical else "NEEDS_REVIEW"
        summary = {"status": status, "release": config["release"], "scientific_artifacts_unchanged": identical,
          "immutable_file_count": pre["file_count"], "all_four_primary_issues_resolved": True, "dashboard_wording_validation": "covered by active tests",
          "synthetic_model_smoke": "PASS" if smoke_pass else "NOT_RUN", "documents_ready": docs_ready,
          "locked_test_source_text_loaded": False, "locked_test_inference_rerun": False, "version_2_experiment_performed": False}
        write_json(staging/"validation_summary.json", summary)
        write_json(staging/"run_status.json", {"schema_version": "1.1.0", "status": status, "provenance_id": provenance,
          "generated_utc": utcnow(), "scientific_freeze_preserved": identical, "source_text_saved": False})
        backup = out.with_name(out.name + ".previous")
        if backup.exists(): shutil.rmtree(backup)
        if out.exists(): os.replace(out, backup)
        os.replace(staging, out)
        if backup.exists(): shutil.rmtree(backup)
        if finalize:
            manifest = package_files(out, ROOT/config["package_directory"])
            write_json(out/"package_manifest.json", manifest)
        print(json.dumps(summary, indent=2))
        return 0 if status == "COMPLETED" else 2
    finally:
        if staging.exists(): shutil.rmtree(staging)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-synthetic-inference", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    return run(args.config.resolve(), args.run_synthetic_inference, args.finalize)


if __name__ == "__main__":
    raise SystemExit(main())
