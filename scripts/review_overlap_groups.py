#!/usr/bin/env python3
"""Prepare deterministic Phase 2A overlap-group policy evidence without splits."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import platform
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEXT_FIELDS = ("preceding", "target", "following")
OUTPUT_KEYS = ("group_membership", "group_summary", "policy_comparison", "manual_review", "report")
MANUAL_FIELDS = (
    "review_case_id", "selection_reason", "rule_a_group_ids", "rule_a_group_sizes", "rule_b_group_ids",
    "rule_b_group_sizes", "source_rows", "example_ids", "matching_fields", "matching_word_count",
    "normalized_text_sha256", "review_question", "reviewer_observation", "reviewer_notes",
)


class ReviewInputError(Exception):
    """Blocking input, provenance, or consistency error."""


def _load_audit_module():
    path = PROJECT_ROOT / "scripts" / "audit_dataset.py"
    spec = importlib.util.spec_from_file_location("sandarbh_phase1_audit", path)
    if spec is None or spec.loader is None:
        raise ReviewInputError("Cannot import the Phase 1 audit module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = _load_audit_module()


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
        raise ReviewInputError(f"{label} path escapes the project root") from exc
    return path


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReviewInputError(f"Cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReviewInputError(f"{label} must contain a JSON object")
    return value


def load_config(path: Path) -> dict[str, Any]:
    config = load_json(path, "configuration")
    required = {"review_schema_version", "input_csv", "audit_script", "audit_config", "audit_artifacts", "minimum_substantial_word_count", "normalization", "reference_observations", "outputs"}
    missing = sorted(required - config.keys())
    if missing:
        raise ReviewInputError(f"Configuration keys missing: {missing}")
    if config["minimum_substantial_word_count"] != 5:
        raise ReviewInputError("Phase 2A requires minimum_substantial_word_count=5")
    for key in OUTPUT_KEYS + ("run_status",):
        if key not in config["outputs"]:
            raise ReviewInputError(f"Configured output missing: {key}")
    return config


def read_csv_rows(path: Path, label: str) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            if reader.fieldnames is None:
                raise ReviewInputError(f"{label} has no header")
            if len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise ReviewInputError(f"{label} has duplicate headers")
            return list(reader.fieldnames), list(reader)
    except ReviewInputError:
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ReviewInputError(f"Cannot read {label}: {exc}") from exc


def verify_provenance(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Path]]:
    source = safe_path(config["input_csv"], "source")
    audit_script = safe_path(config["audit_script"], "audit script")
    audit_config = safe_path(config["audit_config"], "audit config")
    artifacts = {key: safe_path(value, f"audit artifact {key}") for key, value in config["audit_artifacts"].items()}
    required = {"run_status", "summary", "row_manifest", "exclusions", "overlap_links", "report"}
    if set(artifacts) != required:
        raise ReviewInputError(f"audit_artifacts must contain exactly {sorted(required)}")
    for label, path in {"source": source, "audit script": audit_script, "audit config": audit_config, **artifacts}.items():
        if not path.is_file():
            raise ReviewInputError(f"Missing required {label}: {path.relative_to(PROJECT_ROOT)}")
    status = load_json(artifacts["run_status"], "Phase 1 run status")
    summary = load_json(artifacts["summary"], "Phase 1 summary")
    failures = []
    if status.get("status") != "COMPLETED":
        failures.append(f"Phase 1 run status is {status.get('status')!r}, not COMPLETED")
    if summary.get("failures"):
        failures.append("Phase 1 summary contains blocking failures")
    hashes = {
        "source": sha256_file(source), "audit_script": sha256_file(audit_script), "audit_config": sha256_file(audit_config),
        **{f"audit_{key}": sha256_file(path) for key, path in artifacts.items()},
    }
    if hashes["source"] != summary.get("source", {}).get("sha256_after"):
        failures.append("Current source hash differs from Phase 1 summary")
    if hashes["audit_script"] != summary.get("runtime", {}).get("audit_script_sha256"):
        failures.append("Current Phase 1 audit-script hash differs from Phase 1 summary")
    if hashes["audit_config"] != summary.get("configuration", {}).get("sha256"):
        failures.append("Current Phase 1 audit-config hash differs from Phase 1 summary")
    if failures:
        raise ReviewInputError("; ".join(failures))
    return {"phase_1_run_status": status, "phase_1_summary": summary, "hashes": hashes}, {"source": source, "audit_script": audit_script, "audit_config": audit_config, **artifacts}


def validate_inputs(paths: dict[str, Path], summary: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    expected_columns = summary["configuration"]["effective"]["expected_columns"]
    source_rows, _ = audit.read_rows(paths["source"], expected_columns)
    manifest_header, manifest = read_csv_rows(paths["row_manifest"], "row manifest")
    required_manifest = {"source_row", "example_id", "eligible", "exclusion_reason", "A1_Score", "A2_Score", "A3_Score", "diagnostic_binary_majority"}
    if not required_manifest.issubset(manifest_header):
        raise ReviewInputError(f"Row manifest missing fields: {sorted(required_manifest - set(manifest_header))}")
    manifest_ids = [row["example_id"] for row in manifest]
    duplicates = sorted(item for item, count in Counter(manifest_ids).items() if count > 1)
    if duplicates:
        raise ReviewInputError(f"Row manifest has duplicated example IDs: {duplicates[:5]}")
    by_id = {row["example_id"]: row for row in manifest}
    source_by_id = {row["example_id"]: row for row in source_rows}
    missing = sorted(set(source_by_id) - set(by_id))
    unknown = sorted(set(by_id) - set(source_by_id))
    if missing or unknown:
        raise ReviewInputError(f"Row manifest ID mismatch: missing={missing[:5]}, unknown={unknown[:5]}")
    consistency = []
    for example_id, row in source_by_id.items():
        item = by_id[example_id]
        expected = {
            "source_row": str(row["source_row"]), "eligible": str(row["eligible"]), "exclusion_reason": row["exclusion_reason"],
            "A1_Score": str(row["annotations"][0]), "A2_Score": str(row["annotations"][1]), "A3_Score": str(row["annotations"][2]),
            "diagnostic_binary_majority": str(row["binary_majority"]),
        }
        if any(item[key] != value for key, value in expected.items()):
            consistency.append(example_id)
    if consistency:
        raise ReviewInputError(f"Source/manifest values disagree for example IDs: {consistency[:5]}")
    _, exclusions = read_csv_rows(paths["exclusions"], "exclusions")
    exclusion_ids = [row.get("example_id", "") for row in exclusions]
    if len(exclusion_ids) != len(set(exclusion_ids)):
        raise ReviewInputError("Exclusions contain duplicated example IDs")
    expected_excluded = {row["example_id"] for row in source_rows if not row["eligible"]}
    if set(exclusion_ids) != expected_excluded:
        raise ReviewInputError("Exclusion IDs do not exactly match source eligibility")
    link_header, links = read_csv_rows(paths["overlap_links"], "overlap links")
    required_links = {"example_id_1", "example_id_2"}
    if not required_links.issubset(link_header):
        raise ReviewInputError("Overlap links lack example ID fields")
    linked_ids = {row[field] for row in links for field in required_links}
    unknown_links = sorted(linked_ids - set(source_by_id))
    if unknown_links:
        raise ReviewInputError(f"Overlap links contain unknown example IDs: {unknown_links[:5]}")
    eligible_rows = [row for row in source_rows if row["eligible"]]
    if len(eligible_rows) != summary["analysis"]["row_accounting"]["eligible_rows"]:
        raise ReviewInputError("Calculated eligible count differs from Phase 1 summary")
    return eligible_rows, by_id


class UnionFind:
    def __init__(self, items: Iterable[str]):
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union_many(self, items: Iterable[str]) -> None:
        values = sorted(set(items))
        for other in values[1:]:
            left, right = self.find(values[0]), self.find(other)
            if left != right:
                self.parent[max(left, right)] = min(left, right)


def build_index(rows: list[dict[str, Any]]) -> dict[str, list[tuple[str, str]]]:
    index: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in rows:
        for field in TEXT_FIELDS:
            value = audit.normalize_text(row["texts"][field])
            if value:
                index[value].append((row["example_id"], field))
    return {value: sorted(set(occurrences)) for value, occurrences in index.items()}


def materialize_groups(uf: UnionFind, example_ids: Iterable[str], prefix: str) -> tuple[dict[str, str], dict[str, list[str]]]:
    components: dict[str, list[str]] = defaultdict(list)
    for example_id in sorted(example_ids):
        components[uf.find(example_id)].append(example_id)
    groups: dict[str, list[str]] = {}
    memberships: dict[str, str] = {}
    for members in sorted((sorted(values) for values in components.values()), key=lambda values: values[0]):
        group_id = prefix + hashlib.sha256("|".join(members).encode("utf-8")).hexdigest()[:16]
        groups[group_id] = members
        memberships.update({example_id: group_id for example_id in members})
    return memberships, groups


def build_groupings(rows: list[dict[str, Any]], minimum_words: int) -> dict[str, Any]:
    ids = [row["example_id"] for row in rows]
    index = build_index(rows)
    uf_five = UnionFind(ids)
    uf_a = UnionFind(ids)
    uf_b = UnionFind(ids)
    for value, occurrences in index.items():
        occurrence_ids = {item[0] for item in occurrences}
        if len(occurrence_ids) < 2:
            continue
        uf_b.union_many(occurrence_ids)
        if len(value.split()) >= minimum_words:
            uf_five.union_many(occurrence_ids)
            uf_a.union_many(occurrence_ids)
        target_ids = {example_id for example_id, field in occurrences if field == "target"}
        if len(target_ids) >= 2:
            uf_a.union_many(target_ids)
    five_membership, five_groups = materialize_groups(uf_five, ids, "five-")
    a_membership, a_groups = materialize_groups(uf_a, ids, "rule-a-")
    b_membership, b_groups = materialize_groups(uf_b, ids, "rule-b-")
    return {"index": index, "five_membership": five_membership, "five_groups": five_groups,
            "rule_a_membership": a_membership, "rule_a_groups": a_groups,
            "rule_b_membership": b_membership, "rule_b_groups": b_groups}


def group_statistics(groups: dict[str, list[str]], rows_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    sizes = Counter(len(members) for members in groups.values())
    nontrivial = {gid: members for gid, members in groups.items() if len(members) > 1}
    positives = {example_id for example_id, row in rows_by_id.items() if row["binary_majority"] == 1}
    return {
        "eligible_examples": len(rows_by_id), "total_groups": len(groups), "singleton_groups": sizes[1],
        "nontrivial_groups": len(nontrivial), "group_size_distribution": {str(k): sizes[k] for k in sorted(sizes)},
        "largest_group_size": max(sizes, default=0),
        "examples_in_nontrivial_groups": sum(len(members) for members in nontrivial.values()),
        "diagnostic_binary_positive_examples": len(positives),
        "groups_with_at_least_one_diagnostic_binary_positive": sum(bool(set(members) & positives) for members in groups.values()),
    }


def validate_groupings(grouping: dict[str, Any], rows: list[dict[str, Any]], minimum_words: int) -> dict[str, bool]:
    ids = {row["example_id"] for row in rows}
    a = grouping["rule_a_membership"]
    b = grouping["rule_b_membership"]
    duplicate_targets_together = True
    duplicate_targets_together_b = True
    substantial_together = True
    substantial_together_b = True
    for value, occurrences in grouping["index"].items():
        target_ids = {example_id for example_id, field in occurrences if field == "target"}
        if len(target_ids) > 1 and len({a[item] for item in target_ids}) != 1:
            duplicate_targets_together = False
        if len(target_ids) > 1 and len({b[item] for item in target_ids}) != 1:
            duplicate_targets_together_b = False
        occurrence_ids = {item[0] for item in occurrences}
        if len(occurrence_ids) > 1 and len(value.split()) >= minimum_words and len({a[item] for item in occurrence_ids}) != 1:
            substantial_together = False
        if len(occurrence_ids) > 1 and len(value.split()) >= minimum_words and len({b[item] for item in occurrence_ids}) != 1:
            substantial_together_b = False
    a_contained_in_b = all(len({b[item] for item in members}) == 1 for members in grouping["rule_a_groups"].values())
    return {
        "all_eligible_examples_have_rule_a_membership": set(a) == ids,
        "all_eligible_examples_have_rule_b_membership": set(b) == ids,
        "all_normalized_duplicate_targets_grouped_by_rule_a": duplicate_targets_together,
        "all_normalized_duplicate_targets_grouped_by_rule_b": duplicate_targets_together_b,
        "all_substantial_exact_text_links_within_rule_a": substantial_together,
        "all_substantial_exact_text_links_within_rule_b": substantial_together_b,
        "every_rule_a_group_contained_in_one_rule_b_group": a_contained_in_b,
    }


def compare_rules(grouping: dict[str, Any]) -> dict[str, Any]:
    a_groups = grouping["rule_a_groups"]
    b_groups = grouping["rule_b_groups"]
    a_membership = grouping["rule_a_membership"]
    b_membership = grouping["rule_b_membership"]
    five_membership = grouping["five_membership"]
    merged = []
    affected = set()
    for b_group_id, members in sorted(b_groups.items()):
        constituent = sorted({a_membership[item] for item in members})
        if len(constituent) > 1:
            affected.update(members)
            merged.append({
                "rule_b_group_id": b_group_id, "rule_b_size": len(members),
                "merged_rule_a_group_ids": constituent,
                "merged_rule_a_group_sizes": [len(a_groups[group_id]) for group_id in constituent],
            })
    short_strings = []
    for value, occurrences in grouping["index"].items():
        ids = sorted({item[0] for item in occurrences})
        a_ids = sorted({a_membership[item] for item in ids})
        word_count = len(value.split())
        if len(ids) > 1 and word_count < 5 and len(a_ids) > 1:
            short_strings.append({
                "normalized_text_sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
                "word_count": word_count, "affected_examples": len(ids), "rule_a_groups_connected": len(a_ids),
            })
    short_strings.sort(key=lambda item: (item["word_count"], item["normalized_text_sha256"]))
    word_lengths = Counter(item["word_count"] for item in short_strings)
    repeated_exception_values = []
    repeated_affected = set()
    for value, occurrences in grouping["index"].items():
        target_ids = sorted({example_id for example_id, field in occurrences if field == "target"})
        five_groups = sorted({five_membership[item] for item in target_ids})
        if len(target_ids) > 1 and len(five_groups) > 1:
            repeated_affected.update(target_ids)
            repeated_exception_values.append({
                "normalized_text_sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
                "word_count": len(value.split()), "target_examples": len(target_ids), "five_word_groups_merged": len(five_groups),
            })
    repeated_exception_values.sort(key=lambda item: item["normalized_text_sha256"])
    return {
        "rule_a_groups_merged_under_rule_b": len(merged),
        "affected_examples": len(affected),
        "merged_group_size_changes": merged,
        "short_strings_responsible": {
            "distinct_strings": len(short_strings),
            "word_count_distribution": {str(k): word_lengths[k] for k in sorted(word_lengths)},
            "details": short_strings,
        },
        "repeated_target_exception_effect_on_five_word_components": {
            "five_word_groups_merged": sum(item["five_word_groups_merged"] - 1 for item in repeated_exception_values),
            "affected_target_examples": len(repeated_affected),
            "distinct_repeated_target_strings_causing_change": len(repeated_exception_values),
            "details": repeated_exception_values,
        },
    }


def _link_categories(value: str, occurrences: list[tuple[str, str]]) -> set[str]:
    categories = set()
    unique = sorted(set(occurrences))
    for i, (left_id, left_field) in enumerate(unique):
        for right_id, right_field in unique[i + 1:]:
            if left_id == right_id:
                continue
            fields = {left_field, right_field}
            if fields == {"target"}:
                categories.add("target_target")
            elif "target" in fields:
                categories.add("target_context")
            else:
                categories.add("context_context")
    return categories


def select_review_cases(grouping: dict[str, Any], rows_by_id: dict[str, dict[str, Any]], minimum_words: int) -> list[dict[str, str]]:
    index = grouping["index"]
    a_membership = grouping["rule_a_membership"]
    b_membership = grouping["rule_b_membership"]
    a_groups = grouping["rule_a_groups"]
    b_groups = grouping["rule_b_groups"]
    shared = []
    for value, occurrences in index.items():
        ids = sorted({item[0] for item in occurrences})
        if len(ids) > 1:
            shared.append((hashlib.sha256(value.encode("utf-8")).hexdigest(), value, occurrences, ids, _link_categories(value, occurrences)))
    shared.sort(key=lambda item: item[0])
    choices: list[tuple[str, str, str, list[tuple[str, str]], list[str], str]] = []

    def add(code: str, reason: str, question: str, candidate: tuple | None) -> None:
        if candidate is None:
            return
        text_hash, value, occurrences, ids, _ = candidate
        choices.append((code, reason, question, occurrences, ids, text_hash))

    repeated = next((item for item in shared if len({i for i, f in item[2] if f == "target"}) > 1), None)
    add("repeated_target", "Normalized repeated target, included regardless of length by Rule A.",
        "Does this repeated target support keeping these examples in one partition group?", repeated)
    target_context = next((item for item in shared if len(item[1].split()) >= minimum_words and "target_context" in item[4]), None)
    add("target_context", "Substantial target-to-context exact match.",
        "Does the target-to-context reuse indicate meaningful leakage risk?", target_context)
    context_context = next((item for item in shared if len(item[1].split()) >= minimum_words and "context_context" in item[4]), None)
    add("context_context", "Substantial context-to-context exact match.",
        "Does this shared context justify grouping despite different targets?", context_context)

    largest_a_id, largest_a_members = min(grouping["rule_a_groups"].items(), key=lambda item: (-len(item[1]), item[0]))
    largest_candidate = next((item for item in shared if len(set(item[3]) & set(largest_a_members)) >= 2), None)
    add("largest_rule_a", f"Representative link from largest Rule A component ({len(largest_a_members)} examples).",
        "Does this representative link help explain why the largest Rule A component should remain intact?", largest_candidate)

    values_by_a: dict[str, set[str]] = defaultdict(set)
    for text_hash, value, occurrences, ids, categories in shared:
        if len(value.split()) >= minimum_words or len({i for i, f in occurrences if f == "target"}) > 1:
            for group_id in {a_membership[item] for item in ids}:
                values_by_a[group_id].add(text_hash)
    multi_group = next((gid for gid in sorted(a_groups) if len(a_groups[gid]) >= 3 and len(values_by_a[gid]) >= 2), None)
    multi_candidate = next((item for item in shared if multi_group and item[0] in values_by_a[multi_group]), None)
    add("transitive_component", "Component connected through multiple exact-text links.",
        "Do the chained links represent a coherent leakage-control component?", multi_candidate)

    short_merges = [item for item in shared if len(item[1].split()) < minimum_words and len({a_membership[x] for x in item[3]}) > 1]
    short_merges.sort(key=lambda item: (-len(item[1].split()), item[0]))
    for number, candidate in enumerate(short_merges[:4], 1):
        add(f"short_merge_{number}", "Short exact match that merges distinct Rule A groups under Rule B.",
            "Is this short match specific enough to justify the additional Rule B merge?", candidate)

    near_cutoff = sorted(shared, key=lambda item: (abs(len(item[1].split()) - minimum_words), len(item[1].split()) < minimum_words, item[0]))
    add("near_cutoff", "Exact match nearest the five-word cutoff, selected deterministically.",
        "Does the five-word threshold treat this boundary case appropriately?", near_cutoff[0] if near_cutoff else None)

    cases = []
    seen_ids = set()
    for code, reason, question, occurrences, ids, text_hash in choices:
        case_id = "review-" + hashlib.sha256(f"{code}|{text_hash}".encode("utf-8")).hexdigest()[:16]
        if case_id in seen_ids:
            continue
        seen_ids.add(case_id)
        a_ids = sorted({a_membership[item] for item in ids})
        b_ids = sorted({b_membership[item] for item in ids})
        word_count = next(len(value.split()) for digest, value, _, _, _ in shared if digest == text_hash)
        cases.append({
            "review_case_id": case_id, "selection_reason": reason,
            "rule_a_group_ids": "|".join(a_ids), "rule_a_group_sizes": "|".join(str(len(a_groups[item])) for item in a_ids),
            "rule_b_group_ids": "|".join(b_ids), "rule_b_group_sizes": "|".join(str(len(b_groups[item])) for item in b_ids),
            "source_rows": "|".join(str(rows_by_id[item]["source_row"]) for item in ids), "example_ids": "|".join(ids),
            "matching_fields": "|".join(f"{example_id}:{field}" for example_id, field in occurrences if example_id in ids),
            "matching_word_count": str(word_count), "normalized_text_sha256": text_hash,
            "review_question": question, "reviewer_observation": "", "reviewer_notes": "",
        })
    return cases[:12]


def preserve_manual_notes(path: Path, cases: list[dict[str, str]]) -> None:
    if not path.exists():
        return
    header, old_rows = read_csv_rows(path, "existing manual review")
    if not set(MANUAL_FIELDS).issubset(header):
        raise ReviewInputError("Existing manual review has an incompatible schema; choose a new output directory")
    current = {row["review_case_id"]: row for row in cases}
    noted_or_observed = [row for row in old_rows if row.get("reviewer_observation", "").strip() or row.get("reviewer_notes", "").strip()]
    orphaned = sorted(row["review_case_id"] for row in noted_or_observed if row["review_case_id"] not in current)
    if orphaned:
        raise ReviewInputError(f"Existing reviewer notes cannot be mapped to the current shortlist ({orphaned[:5]}); choose a new output directory")
    for row in noted_or_observed:
        current[row["review_case_id"]]["reviewer_observation"] = row["reviewer_observation"]
        current[row["review_case_id"]]["reviewer_notes"] = row["reviewer_notes"]


def group_summary_rows(rule: str, groups: dict[str, list[str]], rows_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for group_id, members in sorted(groups.items()):
        labels = Counter(label for item in members for label in rows_by_id[item]["annotations"])
        agreements = Counter(rows_by_id[item]["agreement"] for item in members)
        positives = sum(rows_by_id[item]["binary_majority"] for item in members)
        output.append({
            "rule": rule, "group_id": group_id, "size": len(members), "diagnostic_binary_positive_examples": positives,
            "diagnostic_binary_other_examples": len(members) - positives, "label_-1_votes": labels[-1], "label_0_votes": labels[0],
            "label_1_votes": labels[1], "unanimous_examples": agreements["unanimous"],
            "two_versus_one_examples": agreements["two-versus-one"], "all_different_examples": agreements["all-different"],
        })
    return output


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


def write_status(path: Path, status: str, message: str, timestamp: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=".phase2a-status-", suffix=".json", dir=path.parent)
    os.close(fd)
    temp = Path(temp_name)
    try:
        write_json(temp, {"status": status, "timestamp_utc": timestamp, "message": message})
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def markdown_report(result: dict[str, Any]) -> str:
    a = result["candidate_rules"]["rule_a"]["statistics"]
    b = result["candidate_rules"]["rule_b"]["statistics"]
    comparison = result["comparison"]
    checks = result["consistency_checks"]
    refs = result["reference_observation_comparisons"]
    ref_differences = [f"- `{key}`: reference {value['reference']}, observed {value['observed']}" for key, value in refs.items() if not value["matches"]]
    return "\n".join([
        "# Phase 02A Overlap-Group Review", "",
        f"**Policy status:** `{result['policy_status']}`  ",
        f"**Generated (UTC):** {result['timestamp_utc']}  ", "",
        "This report compares deterministic proxy grouping rules for leakage control. It does not approve a grouping policy or create a data split.", "",
        "## Candidate rules", "",
        "Rule A connects identical nonempty normalized targets regardless of length and also connects examples sharing any normalized text field of at least five words. Rule B connects examples sharing any normalized nonempty text field regardless of length. Both use connected components and retain every unconnected eligible example as a singleton.", "",
        "Normalization is `\" \".join(text.split()).casefold()` for matching only. It does not establish model-input preprocessing. Labels do not create edges.", "",
        "| Measure | Rule A | Rule B |", "|---|---:|---:|",
        f"| Eligible examples | {a['eligible_examples']} | {b['eligible_examples']} |",
        f"| Total groups | {a['total_groups']} | {b['total_groups']} |",
        f"| Singleton groups | {a['singleton_groups']} | {b['singleton_groups']} |",
        f"| Nontrivial groups | {a['nontrivial_groups']} | {b['nontrivial_groups']} |",
        f"| Examples in nontrivial groups | {a['examples_in_nontrivial_groups']} | {b['examples_in_nontrivial_groups']} |",
        f"| Largest group | {a['largest_group_size']} | {b['largest_group_size']} |",
        f"| Diagnostic binary-positive examples | {a['diagnostic_binary_positive_examples']} | {b['diagnostic_binary_positive_examples']} |",
        f"| Groups containing a diagnostic binary-positive example | {a['groups_with_at_least_one_diagnostic_binary_positive']} | {b['groups_with_at_least_one_diagnostic_binary_positive']} |", "",
        "The binary mapping remains descriptive only: two or more votes of `1` are positive; votes of `0` or `-1` map to the other category. It does not finalize training targets.", "",
        "## What changes under Rule B", "",
        f"Rule B merges {comparison['rule_a_groups_merged_under_rule_b']} Rule A group combinations and affects {comparison['affected_examples']} examples. {comparison['short_strings_responsible']['distinct_strings']} distinct short normalized strings create additional connections. Their word-count distribution is `{json.dumps(comparison['short_strings_responsible']['word_count_distribution'], sort_keys=True)}`.", "",
        f"The repeated-target exception changes the five-word-only components for {comparison['repeated_target_exception_effect_on_five_word_components']['affected_target_examples']} target examples across {comparison['repeated_target_exception_effect_on_five_word_components']['distinct_repeated_target_strings_causing_change']} repeated normalized target strings.", "",
        "These groups are proxies: exact text reuse can indicate leakage risk, but short generic phrases can connect unrelated passages, while exact matching cannot recover source-post identity or paraphrases.", "",
        "## Consistency and reference observations", "",
        *[f"- {key}: {value}" for key, value in checks.items()],
        *(ref_differences if ref_differences else ["- All configured independent reference observations match their calculated definitions."]), "",
        "## Human review", "",
        "Review `results/overlap_review/manual_review.csv`. It contains no raw text. Locate a source record using `source_row`, the one-based logical CSV data record excluding the header. In a correctly imported spreadsheet, the displayed row is normally `source_row + 1`; do not treat it as a physical text-editor line number because CSV fields may contain embedded newlines.", "",
        "Use review only to understand grouping-policy behavior. Do not make ad hoc label changes, delete examples, or add metric-driven exceptions. Record observations and notes in the blank reviewer columns; reruns preserve notes by stable review case ID.", "",
        "## Pending decisions", "",
        "- Choose between Rule A and Rule B, or explicitly authorize a later alternative analysis.",
        "- Decide whether short generic exact matches represent unacceptable leakage risk.",
        "- Confirm how candidate components will constrain a future partitioning procedure.", "",
        "No final split or grouping policy has been approved. No train/validation/test membership, training, calibration, or model evaluation was produced.", "",
        "## Warnings and failures", "",
        *([f"- Warning: {item}" for item in result["warnings"]] or ["- No warnings."]),
        *([f"- Failure: {item}" for item in result["failures"]] or ["- No blocking failures."]), "",
    ])


def reference_comparisons(config: dict[str, Any], five_stats: dict[str, Any], b_stats: dict[str, Any]) -> dict[str, Any]:
    observed = {
        "five_word_components_total_groups": five_stats["total_groups"],
        "five_word_components_singletons": five_stats["singleton_groups"],
        "five_word_components_largest_size": five_stats["largest_group_size"],
        "all_nonempty_components_total_groups": b_stats["total_groups"],
        "all_nonempty_components_singletons": b_stats["singleton_groups"],
        "all_nonempty_components_largest_size": b_stats["largest_group_size"],
    }
    return {key: {"reference": value, "observed": observed.get(key), "matches": observed.get(key) == value}
            for key, value in config["reference_observations"].items()}


def run_review(config_path: Path) -> int:
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    config = load_config(config_path)
    outputs = {key: safe_path(value, f"output {key}") for key, value in config["outputs"].items()}
    write_status(outputs["run_status"], "RUNNING", "Review started; earlier outputs are not current until this becomes COMPLETED.", timestamp)
    provenance, paths = verify_provenance(config)
    protected_before = {name: sha256_file(path) for name, path in paths.items()}
    rows, manifest_by_id = validate_inputs(paths, provenance["phase_1_summary"])
    rows_by_id = {row["example_id"]: row for row in rows}
    grouping = build_groupings(rows, config["minimum_substantial_word_count"])
    checks = validate_groupings(grouping, rows, config["minimum_substantial_word_count"])
    if not all(checks.values()):
        failed = [key for key, value in checks.items() if not value]
        raise ReviewInputError(f"Calculated grouping consistency checks failed: {failed}")
    five_stats = group_statistics(grouping["five_groups"], rows_by_id)
    a_stats = group_statistics(grouping["rule_a_groups"], rows_by_id)
    b_stats = group_statistics(grouping["rule_b_groups"], rows_by_id)
    comparison = compare_rules(grouping)
    shortlist = select_review_cases(grouping, rows_by_id, config["minimum_substantial_word_count"])
    preserve_manual_notes(outputs["manual_review"], shortlist)
    references = reference_comparisons(config, five_stats, b_stats)
    warnings = []
    mismatches = [key for key, value in references.items() if not value["matches"]]
    if mismatches:
        warnings.append("Calculated values differ from independent reference observations: " + ", ".join(mismatches))
    if not 8 <= len(shortlist) <= 12:
        warnings.append(f"Structural criteria produced {len(shortlist)} distinct review cases rather than approximately 8-12.")
    protected_after = {name: sha256_file(path) for name, path in paths.items()}
    if protected_before != protected_after:
        raise ReviewInputError("Source or a Phase 1 audit artifact changed during review")
    result = {
        "review_schema_version": config["review_schema_version"], "timestamp_utc": timestamp,
        "policy_status": "PENDING_RESEARCHER_REVIEW",
        "status_scope": "Candidate overlap-policy evidence only; no grouping policy or data partition is approved.",
        "runtime": {"python_version": platform.python_version(), "platform": platform.platform(),
                    "review_script_sha256": sha256_file(Path(__file__))},
        "configuration": {"filename": str(config_path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(config_path), "effective": config},
        "inputs": {"hashes_before": protected_before, "hashes_after": protected_after,
                   "phase_1_timestamp_utc": provenance["phase_1_summary"]["timestamp_utc"],
                   "phase_1_status": provenance["phase_1_summary"]["status"]},
        "rule_definitions": {
            "normalization": "\" \".join(text.split()).casefold(); matching only",
            "rule_a": "Connect identical nonempty normalized targets at any length; also connect different examples sharing any normalized nonempty text field with at least five words; take connected components including singletons.",
            "rule_b": "Connect different examples sharing any normalized nonempty text field regardless of length; take connected components including singletons.",
            "group_id_method": "Rule prefix plus first 16 hex characters of SHA-256 over sorted member example IDs joined with |.",
            "labels_used_for_edges": False,
        },
        "candidate_rules": {
            "rule_a": {"statistics": a_stats, "all_duplicate_targets_grouped": checks["all_normalized_duplicate_targets_grouped_by_rule_a"],
                       "all_substantial_links_contained": checks["all_substantial_exact_text_links_within_rule_a"]},
            "rule_b": {"statistics": b_stats, "all_duplicate_targets_grouped": checks["all_normalized_duplicate_targets_grouped_by_rule_b"],
                       "all_substantial_links_contained": checks["all_substantial_exact_text_links_within_rule_b"]},
            "five_word_only_diagnostic": {"statistics": five_stats},
        },
        "comparison": comparison, "reference_observation_comparisons": references,
        "consistency_checks": checks,
        "input_consistency": {"eligible_examples": len(rows), "manifest_examples": len(manifest_by_id),
                              "excluded_examples_omitted": len(rows) < len(manifest_by_id)},
        "manual_review": {"case_count": len(shortlist), "selection_uses_labels": False,
                          "raw_text_in_output": False, "notes_preserved_by_review_case_id": True},
        "warnings": warnings, "failures": [],
        "not_run": ["final grouping-policy approval", "train/validation/test splitting", "model training", "calibration", "model evaluation"],
    }
    membership_rows = [{"example_id": item, "source_row": rows_by_id[item]["source_row"],
                        "rule_a_group_id": grouping["rule_a_membership"][item], "rule_b_group_id": grouping["rule_b_membership"][item]}
                       for item in sorted(rows_by_id, key=lambda value: rows_by_id[value]["source_row"])]
    summaries = group_summary_rows("A", grouping["rule_a_groups"], rows_by_id) + group_summary_rows("B", grouping["rule_b_groups"], rows_by_id)
    summary_fields = ("rule", "group_id", "size", "diagnostic_binary_positive_examples", "diagnostic_binary_other_examples",
                      "label_-1_votes", "label_0_votes", "label_1_votes", "unanimous_examples", "two_versus_one_examples", "all_different_examples")
    with tempfile.TemporaryDirectory(prefix="phase2a-review-stage-", dir=PROJECT_ROOT) as temp_dir:
        stage_root = Path(temp_dir)
        staged = {key: stage_root / outputs[key].relative_to(PROJECT_ROOT) for key in OUTPUT_KEYS}
        write_csv(staged["group_membership"], ("example_id", "source_row", "rule_a_group_id", "rule_b_group_id"), membership_rows)
        write_csv(staged["group_summary"], summary_fields, summaries)
        write_json(staged["policy_comparison"], result)
        write_csv(staged["manual_review"], MANUAL_FIELDS, shortlist)
        staged["report"].parent.mkdir(parents=True, exist_ok=True)
        staged["report"].write_text(markdown_report(result), encoding="utf-8", newline="\n")
        if {name: sha256_file(path) for name, path in paths.items()} != protected_before:
            raise ReviewInputError("Source or a Phase 1 artifact changed before output publication")
        for key in ("group_membership", "group_summary", "policy_comparison", "manual_review", "report"):
            outputs[key].parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged[key], outputs[key])
    write_status(outputs["run_status"], "COMPLETED", "Phase 2A evidence completed; grouping policy remains PENDING_RESEARCHER_REVIEW.", timestamp)
    print("Phase 2A overlap review completed: PENDING_RESEARCHER_REVIEW")
    print(f"Comparison: {outputs['policy_comparison'].relative_to(PROJECT_ROOT)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare SANDARBH Phase 2A overlap-group review evidence.")
    parser.add_argument("--config", default="configs/overlap_review.json", help="Configuration path relative to project root")
    args = parser.parse_args(argv)
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    default_status = PROJECT_ROOT / "results" / "overlap_review" / "run_status.json"
    try:
        config_path = safe_path(args.config, "configuration")
        return run_review(config_path)
    except (ReviewInputError, audit.AuditInputError) as exc:
        try:
            write_status(default_status, "FAILED", f"Input/provenance/consistency failure: {exc}", timestamp)
        except OSError:
            pass
        print(f"Overlap review failed: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        try:
            write_status(default_status, "FAILED", f"Unexpected failure ({type(exc).__name__}); previous outputs are not current.", timestamp)
        except OSError:
            pass
        print(f"Overlap review failed unexpectedly: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
