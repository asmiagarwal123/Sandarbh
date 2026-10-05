#!/usr/bin/env python3
"""Prepare frozen, group-constrained SANDARBH Phase 2B partitions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARTITIONS = ("train", "dev_tune", "dev_calibration", "test")
TARGET_FIELDS = (
    "example_id", "source_row", "rule_b_group_id", "A1_Score", "A2_Score", "A3_Score",
    "positive_vote_count", "hard_label", "soft_positive", "soft_other", "fraction_minus1",
    "fraction_zero", "fraction_one", "agreement_category", "has_minus1",
)
MANIFEST_FIELDS = ("example_id", "source_row", "group_id", "outer_split", "partition")


class PartitionError(Exception):
    """Blocking Phase 2B input, provenance, allocation, or publication error."""


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise PartitionError(f"Cannot import {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = load_module("sandarbh_phase1_for_partitions", PROJECT_ROOT / "scripts" / "audit_dataset.py")


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
        raise PartitionError(f"{label} path escapes the project root") from exc
    return path


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PartitionError(f"Cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise PartitionError(f"{label} must contain a JSON object")
    return value


def read_csv(path: Path, label: str) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise PartitionError(f"{label} has a missing or duplicate header")
            return list(reader.fieldnames), list(reader)
    except PartitionError:
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        raise PartitionError(f"Cannot read {label}: {exc}") from exc


def load_config(path: Path) -> dict[str, Any]:
    config = load_json(path, "partition configuration")
    required = {"partition_schema_version", "input_csv", "upstream", "approved_grouping_policy", "label_policy", "partitions", "outer_split_mapping", "allocation", "reference_observations", "outputs"}
    missing = sorted(required - config.keys())
    if missing:
        raise PartitionError(f"Configuration keys missing: {missing}")
    if tuple(config["partitions"].keys()) != PARTITIONS:
        raise PartitionError(f"Partition order must be {PARTITIONS}")
    if not math.isclose(sum(config["partitions"].values()), 1.0, abs_tol=1e-12):
        raise PartitionError("Partition proportions must sum to one")
    if config["allocation"].get("base_seed") != 42:
        raise PartitionError("The approved Phase 2B base seed is 42")
    if not isinstance(config["allocation"].get("candidate_budget"), int) or config["allocation"]["candidate_budget"] < 1:
        raise PartitionError("candidate_budget must be a positive integer")
    return config


def verify_upstream(config: dict[str, Any]) -> tuple[dict[str, Path], dict[str, str], dict[str, Any], list[dict[str, Any]], list[dict[str, str]]]:
    paths = {"source": safe_path(config["input_csv"], "source")}
    paths.update({key: safe_path(value, f"upstream {key}") for key, value in config["upstream"].items()})
    for name, path in paths.items():
        if not path.is_file():
            raise PartitionError(f"Missing {name}: {path.relative_to(PROJECT_ROOT)}")
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    audit_status = load_json(paths["audit_run_status"], "Phase 1 run status")
    audit_summary = load_json(paths["audit_summary"], "Phase 1 summary")
    overlap_status = load_json(paths["overlap_run_status"], "Phase 2A run status")
    overlap_summary = load_json(paths["overlap_policy_comparison"], "Phase 2A policy comparison")
    failures = []
    if audit_status.get("status") != "COMPLETED" or audit_summary.get("failures"):
        failures.append("Phase 1 is not a completed audit without blocking failures")
    if overlap_status.get("status") != "COMPLETED":
        failures.append("Phase 2A run status is not COMPLETED")
    if overlap_summary.get("policy_status") != "PENDING_RESEARCHER_REVIEW" or overlap_summary.get("failures"):
        failures.append("Phase 2A historical policy status or failures are inconsistent")
    if hashes["source"] != audit_summary.get("source", {}).get("sha256_after"):
        failures.append("Source hash differs from Phase 1 provenance")
    if hashes["audit_script"] != audit_summary.get("runtime", {}).get("audit_script_sha256"):
        failures.append("Phase 1 script hash differs from recorded provenance")
    if hashes["audit_config"] != audit_summary.get("configuration", {}).get("sha256"):
        failures.append("Phase 1 config hash differs from recorded provenance")
    if hashes["overlap_script"] != overlap_summary.get("runtime", {}).get("review_script_sha256"):
        failures.append("Phase 2A script hash differs from recorded provenance")
    if hashes["overlap_config"] != overlap_summary.get("configuration", {}).get("sha256"):
        failures.append("Phase 2A config hash differs from recorded provenance")
    if failures:
        raise PartitionError("; ".join(failures))

    source_rows, _ = audit.read_rows(paths["source"], audit_summary["configuration"]["effective"]["expected_columns"])
    eligible = [row for row in source_rows if row["eligible"]]
    excluded = {row["example_id"] for row in source_rows if not row["eligible"]}
    _, manifest = read_csv(paths["audit_manifest"], "audit manifest")
    manifest_ids = [row["example_id"] for row in manifest]
    if len(manifest_ids) != len(set(manifest_ids)):
        raise PartitionError("Audit manifest contains duplicate example IDs")
    manifest_by_id = {row["example_id"]: row for row in manifest}
    if set(manifest_by_id) != {row["example_id"] for row in source_rows}:
        raise PartitionError("Audit manifest IDs do not reconcile with the source")
    _, memberships = read_csv(paths["overlap_group_membership"], "Phase 2A group membership")
    member_ids = [row["example_id"] for row in memberships]
    eligible_ids = {row["example_id"] for row in eligible}
    if len(member_ids) != len(set(member_ids)):
        raise PartitionError("Phase 2A membership has duplicate example IDs")
    if set(member_ids) != eligible_ids:
        missing = sorted(eligible_ids - set(member_ids))
        unknown = sorted(set(member_ids) - eligible_ids)
        raise PartitionError(f"Phase 2A Rule B membership mismatch: missing={missing[:5]}, unknown={unknown[:5]}")
    if set(member_ids) & excluded:
        raise PartitionError("Excluded IDs appear in Phase 2A membership")
    for item in memberships:
        if item["source_row"] != manifest_by_id[item["example_id"]]["source_row"]:
            raise PartitionError(f"Membership source_row mismatch for {item['example_id']}")
        if not item.get("rule_b_group_id"):
            raise PartitionError(f"Missing Rule B group for {item['example_id']}")
    _, group_summary = read_csv(paths["overlap_group_summary"], "Phase 2A group summary")
    reported_b = {row["group_id"]: int(row["size"]) for row in group_summary if row["rule"] == "B"}
    calculated_b = Counter(row["rule_b_group_id"] for row in memberships)
    if dict(calculated_b) != reported_b:
        raise PartitionError("Rule B memberships do not reconcile with the Phase 2A group summary")
    reference = config["reference_observations"]
    observed = {
        "eligible_examples": len(eligible), "excluded_examples": len(excluded), "rule_b_groups": len(calculated_b),
        "rule_b_singletons": sum(size == 1 for size in calculated_b.values()), "rule_b_largest_group": max(calculated_b.values()),
    }
    mismatches = [key for key, value in observed.items() if reference.get(key) != value]
    if mismatches:
        raise PartitionError("Upstream observations differ from configured references: " + ", ".join(mismatches))
    return paths, hashes, {"audit": audit_summary, "overlap": overlap_summary}, eligible, memberships


def make_target(row: dict[str, Any], group_id: str) -> dict[str, Any]:
    annotations = tuple(row["annotations"])
    counts = Counter(annotations)
    positive_votes = counts[1]
    return {
        "example_id": row["example_id"], "source_row": row["source_row"], "rule_b_group_id": group_id,
        "A1_Score": annotations[0], "A2_Score": annotations[1], "A3_Score": annotations[2],
        "positive_vote_count": positive_votes, "hard_label": 1 if positive_votes >= 2 else 0,
        "soft_positive": positive_votes / 3.0, "soft_other": 1.0 - positive_votes / 3.0,
        "fraction_minus1": counts[-1] / 3.0, "fraction_zero": counts[0] / 3.0, "fraction_one": counts[1] / 3.0,
        "agreement_category": row["agreement"], "has_minus1": -1 in annotations,
    }


def build_targets(rows: list[dict[str, Any]], memberships: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups = {row["example_id"]: row["rule_b_group_id"] for row in memberships}
    return [make_target(row, groups[row["example_id"]]) for row in sorted(rows, key=lambda item: item["source_row"])]


def build_group_records(targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in targets:
        grouped[row["rule_b_group_id"]].append(row)
    return [{"group_id": group_id, "size": len(rows), "positive": sum(row["hard_label"] for row in rows),
             "other": sum(1 - row["hard_label"] for row in rows)} for group_id, rows in sorted(grouped.items())]


def objective(counts: dict[str, dict[str, int]], targets: dict[str, dict[str, float]]) -> float:
    return sum((counts[p][metric] - targets[p][metric]) ** 2 / max(targets[p][metric], 1.0)
               for p in PARTITIONS for metric in ("examples", "positive", "other"))


def evaluate_assignment(assignment: dict[str, str], groups: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    totals = {"examples": sum(g["size"] for g in groups), "positive": sum(g["positive"] for g in groups), "other": sum(g["other"] for g in groups)}
    targets = {p: {metric: config["partitions"][p] * total for metric, total in totals.items()} for p in PARTITIONS}
    counts = {p: {"examples": 0, "positive": 0, "other": 0, "groups": 0} for p in PARTITIONS}
    for group in groups:
        partition = assignment[group["group_id"]]
        counts[partition]["examples"] += group["size"]
        counts[partition]["positive"] += group["positive"]
        counts[partition]["other"] += group["other"]
        counts[partition]["groups"] += 1
    overall_rate = totals["positive"] / totals["examples"]
    deviations = {}
    feasible = True
    violation = 0.0
    for p in PARTITIONS:
        example_fraction = counts[p]["examples"] / totals["examples"]
        positive_rate = counts[p]["positive"] / counts[p]["examples"] if counts[p]["examples"] else 0.0
        example_deviation = abs(example_fraction - config["partitions"][p])
        rate_deviation = abs(positive_rate - overall_rate)
        both = counts[p]["positive"] > 0 and counts[p]["other"] > 0
        ex_excess = max(0.0, example_deviation - config["allocation"]["example_fraction_tolerance"])
        rate_excess = max(0.0, rate_deviation - config["allocation"]["positive_rate_tolerance"])
        violation += ex_excess + rate_excess + (0.0 if both else 1.0)
        feasible &= ex_excess == 0 and rate_excess == 0 and both
        deviations[p] = {"example_fraction": example_fraction, "target_fraction": config["partitions"][p],
                         "absolute_example_fraction_deviation": example_deviation, "positive_rate": positive_rate,
                         "overall_positive_rate": overall_rate, "absolute_positive_rate_deviation": rate_deviation,
                         "both_hard_classes_present": both}
    return {"objective": objective(counts, targets), "constraint_violation": violation, "feasible": feasible,
            "totals": totals, "target_counts": targets, "counts": counts, "deviations": deviations}


def allocate_candidate(groups: list[dict[str, Any]], config: dict[str, Any], candidate_index: int) -> tuple[dict[str, str], dict[str, Any]]:
    allocation = config["allocation"]
    seed = allocation["base_seed"] + candidate_index * allocation["candidate_seed_stride"]
    rng = random.Random(seed)
    canonical = sorted(groups, key=lambda group: group["group_id"])
    random_keys = {group["group_id"]: rng.random() for group in canonical}
    ordered = sorted(canonical, key=lambda group: (-group["size"], random_keys[group["group_id"]], group["group_id"]))
    totals = {"examples": sum(g["size"] for g in groups), "positive": sum(g["positive"] for g in groups), "other": sum(g["other"] for g in groups)}
    desired = {p: {metric: config["partitions"][p] * total for metric, total in totals.items()} for p in PARTITIONS}
    counts = {p: {"examples": 0, "positive": 0, "other": 0} for p in PARTITIONS}
    assignment = {}
    for group in ordered:
        candidates = []
        for rank, partition in enumerate(PARTITIONS):
            trial = {p: values.copy() for p, values in counts.items()}
            trial[partition]["examples"] += group["size"]
            trial[partition]["positive"] += group["positive"]
            trial[partition]["other"] += group["other"]
            fill_ratio = trial[partition]["examples"] / max(desired[partition]["examples"], 1.0)
            candidates.append((objective(trial, desired), fill_ratio, rank, partition))
        partition = min(candidates)[3]
        assignment[group["group_id"]] = partition
        counts[partition]["examples"] += group["size"]
        counts[partition]["positive"] += group["positive"]
        counts[partition]["other"] += group["other"]
    return assignment, evaluate_assignment(assignment, groups, config)


def select_assignment(groups: list[dict[str, Any]], config: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any], list[dict[str, Any]]]:
    candidates = []
    assignments = []
    for index in range(config["allocation"]["candidate_budget"]):
        assignment, metrics = allocate_candidate(groups, config, index)
        signature = "|".join(f"{gid}:{assignment[gid]}" for gid in sorted(assignment))
        rank = (0 if metrics["feasible"] else 1, metrics["constraint_violation"], metrics["objective"], signature)
        candidates.append({"candidate_index": index, "derived_seed": config["allocation"]["base_seed"] + index * config["allocation"]["candidate_seed_stride"],
                           "feasible": metrics["feasible"], "constraint_violation": metrics["constraint_violation"], "objective": metrics["objective"]})
        assignments.append((rank, index, assignment, metrics))
    _, selected_index, assignment, metrics = min(assignments, key=lambda item: item[0])
    metrics = {**metrics, "selected_candidate_index": selected_index,
               "selected_candidate_seed": config["allocation"]["base_seed"] + selected_index * config["allocation"]["candidate_seed_stride"]}
    return assignment, metrics, candidates


def build_split_manifest(targets: list[dict[str, Any]], assignment: dict[str, str], outer_mapping: dict[str, str]) -> list[dict[str, Any]]:
    return [{"example_id": row["example_id"], "source_row": row["source_row"], "group_id": row["rule_b_group_id"],
             "outer_split": outer_mapping[assignment[row["rule_b_group_id"]]], "partition": assignment[row["rule_b_group_id"]]}
            for row in targets]


def format_value(value: Any) -> Any:
    return format(value, ".12g") if isinstance(value, float) else value


def write_csv(path: Path, fields: Iterable[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_value(row[key]) for key in fields})


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_status(path: Path, status: str, message: str, timestamp: str, extra: dict[str, Any] | None = None) -> None:
    value = {"status": status, "timestamp_utc": timestamp, "message": message, **(extra or {})}
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=".phase2b-status-", suffix=".json", dir=path.parent)
    os.close(fd)
    temp = Path(temp_name)
    try:
        write_json(temp, value)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def provenance_fingerprint(input_hashes: dict[str, str], config_path: Path) -> tuple[str, dict[str, str]]:
    code_hashes = {"partition_config": sha256_file(config_path), "prepare_script": sha256_file(Path(__file__)),
                   "validate_script": sha256_file(PROJECT_ROOT / "scripts" / "validate_partitions.py")}
    payload = json.dumps({"inputs": input_hashes, "code": code_hashes}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest(), code_hashes


def summarize_diagnostics(targets: list[dict[str, Any]], splits: list[dict[str, Any]], rows_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    target_by_id = {row["example_id"]: row for row in targets}
    result = {}
    for partition in PARTITIONS:
        ids = [row["example_id"] for row in splits if row["partition"] == partition]
        agreements = Counter(target_by_id[item]["agreement_category"] for item in ids)
        availability = Counter("|".join(f"{field}={str(rows_by_id[item]['available'][field]).lower()}" for field in ("preceding", "target", "following")) for item in ids)
        result[partition] = {
            "examples": len(ids), "groups": len({row["group_id"] for row in splits if row["partition"] == partition}),
            "hard_positive": sum(target_by_id[item]["hard_label"] for item in ids),
            "hard_other": sum(1 - target_by_id[item]["hard_label"] for item in ids),
            "examples_with_minus1": sum(bool(target_by_id[item]["has_minus1"]) for item in ids),
            "minus1_votes": sum(sum(label == -1 for label in rows_by_id[item]["annotations"]) for item in ids),
            "agreement": {key: agreements[key] for key in ("unanimous", "two-versus-one", "all-different")},
            "context_availability": dict(sorted(availability.items())),
        }
    return result


def markdown_report(summary: dict[str, Any]) -> str:
    counts = summary["partition_diagnostics"]
    rows = [
        f"| {partition} | {item['examples']} | {item['groups']} | {item['hard_positive']} | {item['hard_other']} | {item['examples_with_minus1']} |"
        for partition, item in counts.items()
    ]
    return "\n".join([
        "# Phase 02B Approved Targets and Partitions", "",
        f"**Status:** `{summary['status']}`  ", f"**Generated (UTC):** {summary['timestamp_utc']}  ",
        "**Approved grouping policy:** Rule B, approved by the user through the Phase 2B prompt after assistant-assisted text inspection. This does not claim that the user personally completed every historical manual-review case.", "",
        "## Grouping policy", "",
        "Rule B connects eligible examples sharing any nonempty exact normalized `preceding`, `target`, or `following` field and takes connected components including singletons. Matching uses `\" \".join(text.split()).casefold()` only. It is a conservative exact-overlap proxy: generic greetings can overgroup unrelated passages, it does not reconstruct original posts, and it cannot remove every possible form of leakage.", "",
        "## Binary benchmark targets", "",
        "All three original annotations are retained. A vote is positive only when its value is `1`. `hard_label` is 1 when at least two votes are positive and 0 otherwise. Hard label 0 means **no majority ableist annotation under this benchmark mapping**; it is not a definitive safe or not-ableist judgment.", "",
        "`soft_positive` is the fraction of annotations equal to 1 and `soft_other` is its complement. Both `-1` and `0` contribute to binary other, while their original columns and separate fractions remain available. A `-1` can reflect insufficient context or lack of relevance, so the binary convention loses a meaningful distinction. Disagreement and `-1` examples remain present without relabeling, dropping, or reweighting. Three-class training is outside this phase.", "",
        "Hard and soft experiments must later use the same rows, partitions, input construction, and binary output space; only their declared loss/training targets may differ.", "",
        "## Partition roles and observed counts", "",
        "| Partition | Examples | Groups | Hard positive | Hard other | Has -1 |", "|---|---:|---:|---:|---:|---:|", *rows, "",
        "- `train`: fit models and every learned text-preprocessing object.",
        "- `dev_tune`: hyperparameters, early stopping, and checkpoint selection.",
        "- `dev_calibration`: fit the predeclared calibration method only after model selection is frozen; never use it for checkpoint selection.",
        "- `test`: one locked final evaluation after methods and evaluation rules are frozen; never use its predictions or performance for development decisions.", "",
        "The dataset is small. In particular, the development and calibration roles may each contain only roughly 19 hard-positive examples; their model-selection and calibration estimates may be unstable. No extra tiny partitions were created.", "",
        "## Allocation and validation", "",
        f"Groups were allocated by a deterministic {summary['allocation']['candidate_budget']}-candidate greedy multi-start search using base seed {summary['allocation']['base_seed']}. Candidate {summary['allocation']['selected_candidate_index']} (derived seed {summary['allocation']['selected_candidate_seed']}) minimized the predeclared count-balance objective among candidates, subject to fixed feasibility checks. Labels were used only for stratification and diagnostics.", "",
        f"Independent prepublication validation: `{summary['validation']['status']}`. Guarantee: **No cross-partition nonempty exact normalized field overlap was found.** This is not an unqualified zero-leakage claim.", "",
        "## Limitations and later decisions", "",
        "- Rule B may overgroup generic short phrases and misses non-exact or semantic reuse.",
        "- The binary target collapses the distinction between `-1` and `0` for training purposes, although the source distinctions remain recorded.",
        "- A later protocol must predeclare selective-prediction thresholds or a selection method before final test evaluation; these small partitions do not support a promised error guarantee.",
        "- Model families, learned preprocessing, hyperparameter spaces, calibration method, and evaluation rules remain for later locked protocols.", "",
        "No model, tokenizer, preprocessing object, calibration model, threshold, prediction, or evaluation was produced in Phase 2B. No raw-text train/dev/test copies were saved.", "",
    ])


def frozen_outputs_state(outputs: dict[str, Path], fingerprint: str, config_path: Path) -> str:
    substantive = [outputs[key] for key in ("targets", "split_manifest", "split_summary", "report")]
    existing = [path.exists() for path in substantive]
    if not any(existing):
        return "absent"
    if not all(existing):
        raise PartitionError("Partial partition outputs already exist; refusing to overwrite them")
    summary = load_json(outputs["split_summary"], "existing split summary")
    if summary.get("status") != "FROZEN" or summary.get("provenance_fingerprint") != fingerprint:
        raise PartitionError("Frozen partition outputs exist with different code/config/input provenance; refusing to overwrite or regenerate")
    validator = load_module("sandarbh_partition_validator_for_reuse", PROJECT_ROOT / "scripts" / "validate_partitions.py")
    validation = validator.validate_saved(config_path, quiet=True)
    if not validation["passed"]:
        raise PartitionError("Existing frozen outputs have identical provenance but fail independent validation")
    return "reused"


def run_prepare(config_path: Path) -> int:
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    config = load_config(config_path)
    outputs = {key: safe_path(value, f"output {key}") for key, value in config["outputs"].items()}
    paths, input_hashes, upstream, eligible, memberships = verify_upstream(config)
    fingerprint, code_hashes = provenance_fingerprint(input_hashes, config_path)
    state = frozen_outputs_state(outputs, fingerprint, config_path)
    if state == "reused":
        write_status(outputs["run_status"], "COMPLETED", "Existing frozen Phase 2B partitions were independently validated and reused unchanged.", timestamp,
                     {"provenance_fingerprint": fingerprint, "membership_reused": True})
        print("Existing frozen Phase 2B partitions have identical provenance and passed independent validation; membership was reused unchanged.")
        return 0
    write_status(outputs["run_status"], "RUNNING", "Partition preparation started; no output is frozen until this becomes COMPLETED.", timestamp)
    protected_before = {name: sha256_file(path) for name, path in paths.items()}
    targets = build_targets(eligible, memberships)
    positives = sum(row["hard_label"] for row in targets)
    reference = config["reference_observations"]
    if positives != reference["hard_positive_examples"] or len(targets) - positives != reference["hard_other_examples"]:
        raise PartitionError("Calculated hard-label counts differ from configured reference observations")
    groups = build_group_records(targets)
    assignment, metrics, candidates = select_assignment(groups, config)
    if not metrics["feasible"]:
        write_status(outputs["run_status"], "NEEDS_REVIEW",
                     "No candidate met the fixed balance and class-presence constraints; no split was frozen.", timestamp,
                     {"best_candidate": metrics})
        return 1
    splits = build_split_manifest(targets, assignment, config["outer_split_mapping"])
    rows_by_id = {row["example_id"]: row for row in eligible}
    diagnostics = summarize_diagnostics(targets, splits, rows_by_id)
    if {name: sha256_file(path) for name, path in paths.items()} != protected_before:
        raise PartitionError("Source or an upstream artifact changed during preparation")

    with tempfile.TemporaryDirectory(prefix="phase2b-partition-stage-", dir=PROJECT_ROOT) as temp_dir:
        stage_root = Path(temp_dir)
        staged = {key: stage_root / outputs[key].relative_to(PROJECT_ROOT) for key in ("targets", "split_manifest", "split_summary", "report")}
        write_csv(staged["targets"], TARGET_FIELDS, targets)
        write_csv(staged["split_manifest"], MANIFEST_FIELDS, splits)
        generated_hashes = {"targets": sha256_file(staged["targets"]), "split_manifest": sha256_file(staged["split_manifest"])}
        summary = {
            "partition_schema_version": config["partition_schema_version"], "timestamp_utc": timestamp, "status": "FROZEN",
            "provenance_fingerprint": fingerprint, "source_and_upstream_hashes": protected_before,
            "code_and_config_hashes": code_hashes, "generated_artifact_hashes": generated_hashes,
            "runtime": {"python_version": platform.python_version(), "platform": platform.platform()},
            "approved_grouping_policy": config["approved_grouping_policy"], "label_policy": config["label_policy"],
            "allocation": {**config["allocation"], "partition_proportions": config["partitions"],
                           "outer_split_mapping": config["outer_split_mapping"],
                           "selected_candidate_index": metrics["selected_candidate_index"],
                           "selected_candidate_seed": metrics["selected_candidate_seed"],
                           "selected_objective": metrics["objective"], "selected_constraint_violation": metrics["constraint_violation"],
                           "candidate_results": candidates},
            "balance": metrics, "partition_diagnostics": diagnostics,
            "exclusion_reconciliation": {"eligible_examples": len(targets), "excluded_examples": reference["excluded_examples"],
                                         "excluded_examples_in_outputs": 0},
            "validation": {"status": "PENDING_PREPUBLICATION_VALIDATION"},
            "limitations": ["Rule B is an exact-overlap proxy and can overgroup generic phrases.",
                            "Small development and calibration positive counts may yield unstable estimates.",
                            "Selective-prediction thresholds remain unspecified for a later protocol."],
            "not_run": ["model training", "learned preprocessing", "tokenization", "calibration", "threshold selection", "evaluation"],
        }
        write_json(staged["split_summary"], summary)
        staged["report"].parent.mkdir(parents=True, exist_ok=True)
        staged["report"].write_text(markdown_report(summary), encoding="utf-8", newline="\n")
        validator = load_module("sandarbh_partition_validator_prepublication", PROJECT_ROOT / "scripts" / "validate_partitions.py")
        validation = validator.validate_saved(config_path, quiet=True, output_overrides={
            "targets": staged["targets"], "split_manifest": staged["split_manifest"], "split_summary": staged["split_summary"]})
        if not validation["passed"]:
            raise PartitionError("Independent prepublication validation failed: " + "; ".join(validation["failures"]))
        summary["validation"] = {"status": "PASSED", "checks": validation["checks"],
                                 "guarantee": "No cross-partition nonempty exact normalized field overlap was found."}
        write_json(staged["split_summary"], summary)
        staged["report"].write_text(markdown_report(summary), encoding="utf-8", newline="\n")
        if {name: sha256_file(path) for name, path in paths.items()} != protected_before:
            raise PartitionError("Source or an upstream artifact changed before publication")
        for key in ("targets", "split_manifest", "split_summary", "report"):
            outputs[key].parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged[key], outputs[key])
    write_status(outputs["run_status"], "COMPLETED", "Phase 2B targets and partitions are frozen and independently validated.", timestamp,
                 {"provenance_fingerprint": fingerprint})
    print("Phase 2B partition preparation completed: FROZEN")
    print(f"Summary: {outputs['split_summary'].relative_to(PROJECT_ROOT)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare approved SANDARBH Phase 2B targets and group-constrained partitions.")
    parser.add_argument("--config", default="configs/partition.json", help="Configuration path relative to project root")
    args = parser.parse_args(argv)
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    default_status = PROJECT_ROOT / "results" / "partitions" / "run_status.json"
    try:
        return run_prepare(safe_path(args.config, "configuration"))
    except (PartitionError, audit.AuditInputError) as exc:
        try:
            write_status(default_status, "FAILED", f"Preparation failed without replacing frozen outputs: {exc}", timestamp)
        except OSError:
            pass
        print(f"Partition preparation failed: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        try:
            write_status(default_status, "FAILED", f"Unexpected failure ({type(exc).__name__}); outputs are not current.", timestamp)
        except OSError:
            pass
        print(f"Partition preparation failed unexpectedly: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
