#!/usr/bin/env python3
"""Independently validate saved SANDARBH Phase 2B targets and partitions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARTITIONS = ("train", "dev_tune", "dev_calibration", "test")


class ValidationError(Exception):
    pass


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValidationError(f"Cannot import {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = load_module("sandarbh_phase1_for_validation", PROJECT_ROOT / "scripts" / "audit_dataset.py")


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


def expected_target(row: dict[str, Any], group_id: str) -> dict[str, Any]:
    labels = tuple(row["annotations"])
    counts = Counter(labels)
    positives = counts[1]
    return {
        "source_row": row["source_row"], "rule_b_group_id": group_id, "annotations": labels,
        "positive_vote_count": positives, "hard_label": 1 if positives >= 2 else 0,
        "soft_positive": positives / 3.0, "soft_other": 1.0 - positives / 3.0,
        "fraction_minus1": counts[-1] / 3.0, "fraction_zero": counts[0] / 3.0, "fraction_one": counts[1] / 3.0,
        "agreement_category": row["agreement"], "has_minus1": -1 in labels,
    }


def _bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"invalid boolean {value!r}")


def validate_saved(config_path: Path, quiet: bool = False, output_overrides: dict[str, Path] | None = None) -> dict[str, Any]:
    failures: list[str] = []
    checks: dict[str, bool] = {}
    try:
        config = load_json(config_path, "partition configuration")
        source = safe_path(config["input_csv"], "source")
        upstream = {key: safe_path(value, f"upstream {key}") for key, value in config["upstream"].items()}
        outputs = {key: safe_path(value, f"output {key}") for key, value in config["outputs"].items()}
        outputs.update(output_overrides or {})
        required = {"targets", "split_manifest", "split_summary"}
        for name, path in {"source": source, **upstream, **{key: outputs[key] for key in required}}.items():
            if not path.is_file():
                failures.append(f"Missing required file: {name}")
        if failures:
            return {"passed": False, "checks": checks, "failures": failures}

        before_hash = sha256_file(source)
        summary = load_json(outputs["split_summary"], "split summary")
        audit_summary = load_json(upstream["audit_summary"], "Phase 1 summary")
        overlap_summary = load_json(upstream["overlap_policy_comparison"], "Phase 2A policy comparison")
        current_inputs = {"source": sha256_file(source), **{key: sha256_file(path) for key, path in upstream.items()}}
        recorded_inputs = summary.get("source_and_upstream_hashes", {})
        checks["summary_is_frozen_rule_b"] = (summary.get("status") == "FROZEN"
                                               and summary.get("approved_grouping_policy") == config.get("approved_grouping_policy"))
        if not checks["summary_is_frozen_rule_b"]:
            failures.append("Summary is not a frozen output of the configured approved Rule B policy")
        checks["provenance_hashes_match"] = current_inputs == recorded_inputs
        if not checks["provenance_hashes_match"]:
            failures.append("Current source/upstream hashes differ from the frozen summary")
        code_hashes = summary.get("code_and_config_hashes", {})
        current_code = {"partition_config": sha256_file(config_path),
                        "prepare_script": sha256_file(PROJECT_ROOT / "scripts" / "prepare_partitions.py"),
                        "validate_script": sha256_file(Path(__file__))}
        checks["code_and_config_hashes_match"] = code_hashes == current_code
        if not checks["code_and_config_hashes_match"]:
            failures.append("Current preparation/validation/config hashes differ from the frozen summary")
        checks["upstream_statuses_valid"] = (
            load_json(upstream["audit_run_status"], "audit status").get("status") == "COMPLETED"
            and not audit_summary.get("failures")
            and load_json(upstream["overlap_run_status"], "overlap status").get("status") == "COMPLETED"
            and overlap_summary.get("policy_status") == "PENDING_RESEARCHER_REVIEW"
            and not overlap_summary.get("failures")
        )
        if not checks["upstream_statuses_valid"]:
            failures.append("Upstream completion/status checks failed")

        source_rows, _ = audit.read_rows(source, audit_summary["configuration"]["effective"]["expected_columns"])
        eligible = {row["example_id"]: row for row in source_rows if row["eligible"]}
        excluded = {row["example_id"] for row in source_rows if not row["eligible"]}
        _, membership_rows = read_csv(upstream["overlap_group_membership"], "Rule B membership")
        member_ids = [row["example_id"] for row in membership_rows]
        groups = {row["example_id"]: row["rule_b_group_id"] for row in membership_rows}
        checks["upstream_membership_exactly_eligible"] = len(member_ids) == len(set(member_ids)) and set(member_ids) == set(eligible)
        if not checks["upstream_membership_exactly_eligible"]:
            failures.append("Upstream Rule B membership is not a one-to-one map of eligible IDs")

        target_header, target_rows = read_csv(outputs["targets"], "targets")
        manifest_header, manifest_rows = read_csv(outputs["split_manifest"], "split manifest")
        required_target = {"example_id", "source_row", "rule_b_group_id", "A1_Score", "A2_Score", "A3_Score", "positive_vote_count", "hard_label", "soft_positive", "soft_other", "fraction_minus1", "fraction_zero", "fraction_one", "agreement_category", "has_minus1"}
        required_manifest = {"example_id", "source_row", "group_id", "outer_split", "partition"}
        if not required_target.issubset(target_header) or not required_manifest.issubset(manifest_header):
            failures.append("Saved CSV schema is incomplete")
            return {"passed": False, "checks": checks, "failures": failures}
        target_ids = [row["example_id"] for row in target_rows]
        manifest_ids = [row["example_id"] for row in manifest_rows]
        checks["eligible_examples_exactly_once"] = (len(target_ids) == len(set(target_ids)) == len(eligible)
                                                     and len(manifest_ids) == len(set(manifest_ids)) == len(eligible)
                                                     and set(target_ids) == set(manifest_ids) == set(eligible))
        if not checks["eligible_examples_exactly_once"]:
            failures.append("Eligible IDs are missing, duplicated, unknown, or inconsistent across saved outputs")
        checks["excluded_examples_absent"] = not ((set(target_ids) | set(manifest_ids)) & excluded)
        if not checks["excluded_examples_absent"]:
            failures.append("Excluded IDs appear in partition outputs")
        target_by_id = {row["example_id"]: row for row in target_rows}
        manifest_by_id = {row["example_id"]: row for row in manifest_rows}

        target_errors = []
        soft_valid = True
        for example_id in sorted(set(eligible) & set(target_by_id) & set(groups)):
            row = target_by_id[example_id]
            expected = expected_target(eligible[example_id], groups[example_id])
            try:
                observed_labels = tuple(int(row[key]) for key in ("A1_Score", "A2_Score", "A3_Score"))
                exact = (int(row["source_row"]) == expected["source_row"] and row["rule_b_group_id"] == expected["rule_b_group_id"]
                         and observed_labels == expected["annotations"] and int(row["positive_vote_count"]) == expected["positive_vote_count"]
                         and int(row["hard_label"]) == expected["hard_label"] and row["agreement_category"] == expected["agreement_category"]
                         and _bool(row["has_minus1"]) == expected["has_minus1"])
                floats = {key: float(row[key]) for key in ("soft_positive", "soft_other", "fraction_minus1", "fraction_zero", "fraction_one")}
                exact &= all(math.isclose(floats[key], expected[key], abs_tol=1e-9, rel_tol=0.0) for key in floats)
                soft_valid &= all(-1e-12 <= floats[key] <= 1 + 1e-12 for key in floats)
                soft_valid &= math.isclose(floats["soft_positive"] + floats["soft_other"], 1.0, abs_tol=1e-9, rel_tol=0.0)
                soft_valid &= math.isclose(floats["fraction_minus1"] + floats["fraction_zero"] + floats["fraction_one"], 1.0, abs_tol=1e-9, rel_tol=0.0)
                if not exact:
                    target_errors.append(example_id)
            except (ValueError, KeyError):
                target_errors.append(example_id)
                soft_valid = False
        checks["targets_match_source_annotations"] = not target_errors and len(target_by_id) == len(eligible)
        checks["soft_targets_valid"] = soft_valid
        if not checks["targets_match_source_annotations"]:
            failures.append(f"Saved targets disagree with source annotations/Rule B groups: {target_errors[:5]}")
        if not soft_valid:
            failures.append("Soft targets or three-way fractions are invalid")

        mapping = config["outer_split_mapping"]
        manifest_errors = []
        partitions_by_group: dict[str, set[str]] = defaultdict(set)
        for example_id, row in manifest_by_id.items():
            if example_id not in eligible or example_id not in groups:
                continue
            try:
                valid = (int(row["source_row"]) == eligible[example_id]["source_row"] and row["group_id"] == groups[example_id]
                         and row["partition"] in PARTITIONS and row["outer_split"] == mapping[row["partition"]])
            except (ValueError, KeyError):
                valid = False
            if not valid:
                manifest_errors.append(example_id)
            partitions_by_group[row.get("group_id", "")].add(row.get("partition", ""))
        checks["ids_source_rows_and_outer_roles_reconcile"] = not manifest_errors and len(manifest_by_id) == len(eligible)
        checks["rule_b_groups_intact"] = all(len(values) == 1 for values in partitions_by_group.values()) and set(partitions_by_group) == set(groups.values())
        if not checks["ids_source_rows_and_outer_roles_reconcile"]:
            failures.append(f"Split manifest mapping errors: {manifest_errors[:5]}")
        if not checks["rule_b_groups_intact"]:
            failures.append("At least one Rule B group crosses partitions or is missing")

        counts = {p: {"examples": 0, "positive": 0, "other": 0, "groups": 0} for p in PARTITIONS}
        group_sets = {p: set() for p in PARTITIONS}
        for example_id, row in manifest_by_id.items():
            if example_id not in target_by_id or row.get("partition") not in counts:
                continue
            label = int(target_by_id[example_id]["hard_label"])
            counts[row["partition"]]["examples"] += 1
            counts[row["partition"]]["positive"] += label
            counts[row["partition"]]["other"] += 1 - label
            group_sets[row["partition"]].add(row["group_id"])
        for p in PARTITIONS:
            counts[p]["groups"] = len(group_sets[p])
        total_examples = sum(value["examples"] for value in counts.values())
        total_positive = sum(value["positive"] for value in counts.values())
        overall_rate = total_positive / total_examples if total_examples else 0.0
        both_classes = all(counts[p]["positive"] > 0 and counts[p]["other"] > 0 for p in PARTITIONS)
        balance = all(abs(counts[p]["examples"] / total_examples - config["partitions"][p]) <= config["allocation"]["example_fraction_tolerance"] + 1e-12
                      and abs(counts[p]["positive"] / counts[p]["examples"] - overall_rate) <= config["allocation"]["positive_rate_tolerance"] + 1e-12
                      for p in PARTITIONS)
        checks["both_hard_classes_in_every_partition"] = both_classes
        checks["balance_tolerances_hold"] = balance
        if not both_classes:
            failures.append("At least one partition lacks a hard class")
        if not balance:
            failures.append("Saved partition balance exceeds configured tolerances")

        diagnostics = summary.get("partition_diagnostics", {})
        expected_diagnostics = {}
        for p in PARTITIONS:
            ids = [example_id for example_id, row in manifest_by_id.items() if row.get("partition") == p and example_id in eligible]
            agreements = Counter(eligible[item]["agreement"] for item in ids)
            availability = Counter("|".join(f"{field}={str(eligible[item]['available'][field]).lower()}"
                                             for field in ("preceding", "target", "following")) for item in ids)
            expected_diagnostics[p] = {
                "examples": counts[p]["examples"], "groups": counts[p]["groups"],
                "hard_positive": counts[p]["positive"], "hard_other": counts[p]["other"],
                "examples_with_minus1": sum(-1 in eligible[item]["annotations"] for item in ids),
                "minus1_votes": sum(sum(label == -1 for label in eligible[item]["annotations"]) for item in ids),
                "agreement": {key: agreements[key] for key in ("unanimous", "two-versus-one", "all-different")},
                "context_availability": dict(sorted(availability.items())),
            }
        reconciles = diagnostics == expected_diagnostics
        checks["saved_summary_reconciles"] = bool(reconciles)
        if not reconciles:
            failures.append("Saved split summary does not reconcile with manifests")
        exclusion_summary = summary.get("exclusion_reconciliation", {})
        checks["exclusion_summary_reconciles"] = (exclusion_summary.get("eligible_examples") == len(eligible)
                                                   and exclusion_summary.get("excluded_examples") == len(excluded)
                                                   and exclusion_summary.get("excluded_examples_in_outputs") == 0)
        if not checks["exclusion_summary_reconciles"]:
            failures.append("Saved exclusion reconciliation is incorrect")
        reference = config["reference_observations"]
        checks["hard_label_reference_totals_match"] = (total_positive == reference["hard_positive_examples"]
                                                        and total_examples - total_positive == reference["hard_other_examples"])
        if not checks["hard_label_reference_totals_match"]:
            failures.append("Calculated hard-label totals differ from configured reference observations")
        generated = summary.get("generated_artifact_hashes", {})
        checks["generated_artifact_hashes_match"] = (generated.get("targets") == sha256_file(outputs["targets"])
                                                       and generated.get("split_manifest") == sha256_file(outputs["split_manifest"]))
        if not checks["generated_artifact_hashes_match"]:
            failures.append("Saved target or manifest hash differs from split summary")

        partition_by_id = {example_id: row["partition"] for example_id, row in manifest_by_id.items() if row.get("partition") in PARTITIONS}
        normalized_index: dict[str, set[str]] = defaultdict(set)
        target_index: dict[str, set[str]] = defaultdict(set)
        for example_id, row in eligible.items():
            for field in ("preceding", "target", "following"):
                value = audit.normalize_text(row["texts"][field])
                if value:
                    normalized_index[value].add(example_id)
                    if field == "target":
                        target_index[value].add(example_id)
        cross = [hashlib.sha256(value.encode("utf-8")).hexdigest() for value, ids in normalized_index.items()
                 if len(ids) > 1 and len({partition_by_id.get(item) for item in ids}) > 1]
        duplicate_target_cross = [hashlib.sha256(value.encode("utf-8")).hexdigest() for value, ids in target_index.items()
                                  if len(ids) > 1 and len({partition_by_id.get(item) for item in ids}) > 1]
        checks["no_cross_partition_nonempty_exact_normalized_field_overlap"] = not cross
        checks["duplicate_normalized_targets_do_not_cross_partitions"] = not duplicate_target_cross
        if cross:
            failures.append(f"Cross-partition exact normalized field overlap detected: {cross[:5]}")
        if duplicate_target_cross:
            failures.append(f"Duplicate normalized targets cross partitions: {duplicate_target_cross[:5]}")
        checks["source_bytes_unchanged"] = sha256_file(source) == before_hash
        if not checks["source_bytes_unchanged"]:
            failures.append("Source bytes changed during validation")
    except (ValidationError, audit.AuditInputError, KeyError, ValueError, TypeError) as exc:
        failures.append(str(exc))
    passed = not failures and all(checks.values())
    result = {"passed": passed, "checks": checks, "failures": failures,
              "guarantee": "No cross-partition nonempty exact normalized field overlap was found." if passed else None}
    if not quiet:
        print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Independently validate frozen SANDARBH Phase 2B outputs.")
    parser.add_argument("--config", default="configs/partition.json", help="Configuration path relative to project root")
    args = parser.parse_args(argv)
    try:
        result = validate_saved(safe_path(args.config, "configuration"))
        return 0 if result["passed"] else 1
    except Exception as exc:
        print(f"Partition validation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
