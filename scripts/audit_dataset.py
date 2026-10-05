#!/usr/bin/env python3
"""Read-only, reproducible Phase 1 audit for AUTALIC."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABELS = (-1, 0, 1)
TEXT_COLUMNS = ("preceding", "target", "following")
ANNOTATION_COLUMNS = ("A1_Score", "A2_Score", "A3_Score")
MISSING_PLACEHOLDERS = {"nan", "none", "null", "na", "n/a", "missing"}


class AuditInputError(Exception):
    """Input/configuration error for which normal audit outputs are unsafe."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(text: str) -> str:
    return " ".join(text.split()).casefold()


def record_digest(values: Iterable[str]) -> str:
    payload = json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AuditInputError(f"Cannot read configuration: {exc}") from exc
    required = {"audit_schema_version", "input_csv", "expected_columns", "text_columns", "annotation_columns", "outputs", "reference", "overlap"}
    missing = sorted(required - config.keys())
    if missing:
        raise AuditInputError(f"Configuration keys missing: {missing}")
    if tuple(config["text_columns"]) != TEXT_COLUMNS or tuple(config["annotation_columns"]) != ANNOTATION_COLUMNS:
        raise AuditInputError("Configured text or annotation columns do not match the Phase 1 contract")
    minimum = config["overlap"].get("minimum_substantial_word_count")
    if not isinstance(minimum, int) or minimum < 1:
        raise AuditInputError("minimum_substantial_word_count must be a positive integer")
    return config


def parse_annotation(value: str | None, source_row: int, column: str) -> int:
    if value is None or not value.strip():
        raise AuditInputError(f"Missing annotation at source_row={source_row}, column={column}")
    stripped = value.strip()
    if stripped not in {"-1", "0", "1"}:
        raise AuditInputError(f"Invalid annotation at source_row={source_row}, column={column}")
    return int(stripped)


def read_rows(path: Path, expected_columns: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle, strict=True)
            try:
                header = next(reader)
            except StopIteration as exc:
                raise AuditInputError("CSV is empty") from exc
            duplicates = sorted(name for name, count in Counter(header).items() if count > 1)
            missing = sorted(set(expected_columns) - set(header))
            unexpected = sorted(set(header) - set(expected_columns))
            if duplicates or missing or unexpected or header != expected_columns:
                raise AuditInputError(
                    f"Header validation failed; missing={missing}, duplicate={duplicates}, "
                    f"unexpected={unexpected}, order_matches={header == expected_columns}"
                )
            for source_row, fields in enumerate(reader, 1):
                if len(fields) != len(header):
                    raise AuditInputError(
                        f"Malformed record at source_row={source_row}: expected {len(header)} fields, observed {len(fields)}"
                    )
                raw = dict(zip(header, fields))
                annotations = tuple(parse_annotation(raw[c], source_row, c) for c in ANNOTATION_COLUMNS)
                texts = {c: raw[c] if raw[c] is not None else "" for c in TEXT_COLUMNS}
                available = {c: bool(texts[c].strip()) for c in TEXT_COLUMNS}
                if not available["target"]:
                    reason = "all_text_empty" if not any(available.values()) else "target_empty"
                    eligible = False
                else:
                    reason = ""
                    eligible = True
                digest = record_digest(fields)
                counts = Counter(annotations)
                if len(counts) == 1:
                    agreement = "unanimous"
                elif len(counts) == 2:
                    agreement = "two-versus-one"
                else:
                    agreement = "all-different"
                majority = next((str(label) for label in LABELS if counts[label] >= 2), "tie")
                fractions = {label: counts[label] / 3.0 for label in LABELS}
                if not math.isclose(sum(fractions.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
                    raise AuditInputError(f"Annotation fractions fail validation at source_row={source_row}")
                rows.append({
                    "source_row": source_row,
                    "example_id": f"row-{source_row:06d}-{digest[:12]}",
                    "record_sha256": digest,
                    "texts": texts,
                    "available": available,
                    "eligible": eligible,
                    "exclusion_reason": reason,
                    "annotations": annotations,
                    "agreement": agreement,
                    "raw_majority": majority,
                    "binary_majority": 1 if counts[1] >= 2 else 0,
                    "fractions": fractions,
                    "target_word_count": len(texts["target"].split()),
                    "combined_word_count": sum(len(texts[c].split()) for c in TEXT_COLUMNS),
                })
    except AuditInputError:
        raise
    except UnicodeError as exc:
        raise AuditInputError(f"CSV UTF-8 decoding failed: {exc}") from exc
    except csv.Error as exc:
        raise AuditInputError(f"CSV parsing failed: {exc}") from exc
    except OSError as exc:
        raise AuditInputError(f"Cannot read CSV: {exc}") from exc
    return rows, {"header": header, "column_count": len(header), "row_count": len(rows)}


def annotation_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    overall = Counter()
    per_column = {column: Counter() for column in ANNOTATION_COLUMNS}
    agreements = Counter()
    majority = Counter()
    binary = Counter()
    with_negative_one = 0
    for row in rows:
        agreements[row["agreement"]] += 1
        majority[row["raw_majority"]] += 1
        binary[str(row["binary_majority"])] += 1
        with_negative_one += -1 in row["annotations"]
        for column, label in zip(ANNOTATION_COLUMNS, row["annotations"]):
            overall[label] += 1
            per_column[column][label] += 1
    ordered = lambda counter: {str(label): counter[label] for label in LABELS}
    return {
        "rows": len(rows),
        "label_counts_overall": ordered(overall),
        "label_counts_per_column": {c: ordered(per_column[c]) for c in ANNOTATION_COLUMNS},
        "agreement": {k: agreements[k] for k in ("unanimous", "two-versus-one", "all-different")},
        "rows_with_at_least_one_-1": with_negative_one,
        "raw_majority": {k: majority[k] for k in ("-1", "0", "1", "tie")},
        "diagnostic_binary_majority": {"positive": binary["1"], "other": binary["0"]},
    }


class UnionFind:
    def __init__(self, items: Iterable[str]):
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def analyze_overlaps(rows: list[dict[str, Any]], minimum_words: int) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    eligible = [row for row in rows if row["eligible"]]
    indices: dict[str, dict[str, list[tuple[str, str]]]] = {}
    for setting in ("original", "whitespace_casefold"):
        index: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for row in eligible:
            for field in TEXT_COLUMNS:
                text = row["texts"][field]
                value = text if setting == "original" else normalize_text(text)
                if value:
                    index[value].append((row["example_id"], field))
        indices[setting] = index

    links_set: set[tuple[str, ...]] = set()
    substantial_strings: set[str] = set()
    affected: set[str] = set()
    target_in_context: set[str] = set()
    uf = UnionFind(row["example_id"] for row in eligible)
    normalized_index = indices["whitespace_casefold"]
    for setting, index in indices.items():
        for value, occurrences in index.items():
            unique = sorted(set(occurrences))
            unique_ids = {item[0] for item in unique}
            if len(unique_ids) < 2:
                continue
            word_count = len(value.split())
            substantial = setting == "whitespace_casefold" and word_count >= minimum_words
            if substantial:
                substantial_strings.add(value)
                affected.update(unique_ids)
                ordered_ids = sorted(unique_ids)
                for other in ordered_ids[1:]:
                    uf.union(ordered_ids[0], other)
            for i, (left_id, left_field) in enumerate(unique):
                for right_id, right_field in unique[i + 1:]:
                    if left_id == right_id:
                        continue
                    a = (left_id, left_field)
                    b = (right_id, right_field)
                    if b < a:
                        a, b = b, a
                    fields = {left_field, right_field}
                    if fields == {"target"}:
                        base = "target_target"
                    elif "target" in fields:
                        base = "target_context"
                    else:
                        base = "context_context"
                    link_type = base + ("_substantial" if substantial else "_short_or_unthresholded")
                    links_set.add((setting, a[0], a[1], b[0], b[1], hashlib.sha256(value.encode("utf-8")).hexdigest(), str(word_count), link_type))
                    if substantial and base == "target_context":
                        target_in_context.add(left_id if left_field == "target" else right_id)

    links = [dict(zip(("comparison_setting", "example_id_1", "field_1", "example_id_2", "field_2", "normalized_text_sha256", "word_count", "link_type"), item)) for item in sorted(links_set)]
    groups: dict[str, list[str]] = defaultdict(list)
    for row in eligible:
        groups[uf.find(row["example_id"])].append(row["example_id"])
    nontrivial = [sorted(ids) for ids in groups.values() if len(ids) > 1]
    nontrivial.sort(key=lambda ids: ids[0])
    group_ids: dict[str, str] = {}
    for ids in nontrivial:
        group_id = "overlap-" + hashlib.sha256("|".join(ids).encode()).hexdigest()[:12]
        for example_id in ids:
            group_ids[example_id] = group_id

    target_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        target_groups[normalize_text(row["texts"]["target"])].append(row)
    repeated_targets = {text: rs for text, rs in target_groups.items() if text and len(rs) > 1}
    conflicting = sum(1 for rs in repeated_targets.values() if len({tuple(r["annotations"]) for r in rs}) > 1)
    triplets_original = Counter(tuple(row["texts"][c] for c in TEXT_COLUMNS) for row in eligible)
    triplets_normalized = Counter(tuple(normalize_text(row["texts"][c]) for c in TEXT_COLUMNS) for row in eligible)
    size_distribution = Counter(len(ids) for ids in nontrivial)
    repeated_any = {value for value, occ in normalized_index.items() if len({x[0] for x in occ}) > 1}
    summary = {
        "eligible_rows_analyzed": len(eligible),
        "comparison_settings": {
            "original": "exact decoded field value",
            "whitespace_casefold": "single-space Unicode whitespace normalization followed by Unicode casefold",
        },
        "minimum_substantial_word_count": minimum_words,
        "repeated_targets": {
            "original_distinct_strings": sum(1 for value, occ in indices["original"].items() if value and len({i for i, f in occ if f == "target"}) > 1),
            "normalized_distinct_strings": len(repeated_targets),
            "normalized_affected_examples": len({r["example_id"] for rs in repeated_targets.values() for r in rs}),
            "groups_with_conflicting_annotation_triples": conflicting,
        },
        "repeated_complete_triplets": {
            "original_groups": sum(count > 1 for count in triplets_original.values()),
            "original_affected_examples": sum(count for count in triplets_original.values() if count > 1),
            "normalized_groups": sum(count > 1 for count in triplets_normalized.values()),
            "normalized_affected_examples": sum(count for count in triplets_normalized.values() if count > 1),
        },
        "shared_normalized_text": {
            "distinct_repeated_strings_any_fields": len(repeated_any),
            "substantial_distinct_strings": len(substantial_strings),
            "substantial_affected_examples": len(affected),
            "substantial_target_examples_in_other_context": len(target_in_context),
        },
        "candidate_components": {
            "nontrivial_component_count": len(nontrivial),
            "size_distribution": {str(k): size_distribution[k] for k in sorted(size_distribution)},
            "largest_size": max(size_distribution, default=1),
        },
        "overlap_link_records": len(links),
    }
    return summary, links, group_ids


def percentile_nearest_rank(values: list[int], percentile: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def length_summary(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"minimum": None, "median": None, "p95_nearest_rank": None, "maximum": None, "exceeding_256": 0}
    return {
        "minimum": min(values),
        "median": statistics.median(values),
        "p95_nearest_rank": percentile_nearest_rank(values, 0.95),
        "maximum": max(values),
        "exceeding_256": sum(value > 256 for value in values),
    }


def text_quality(rows: list[dict[str, Any]]) -> dict[str, Any]:
    categories: dict[str, set[str]] = {"literal_missing_value_placeholder": set(), "replacement_character": set(), "unexpected_control_character": set()}
    field_counts = Counter()
    for row in rows:
        for field, text in row["texts"].items():
            found = []
            if text.strip().casefold() in MISSING_PLACEHOLDERS:
                found.append("literal_missing_value_placeholder")
            if "\ufffd" in text:
                found.append("replacement_character")
            if any((ord(ch) < 32 and ch not in "\t\n\r") or ord(ch) == 127 for ch in text):
                found.append("unexpected_control_character")
            for category in found:
                categories[category].add(row["example_id"])
                field_counts[(category, field)] += 1
    return {category: {"affected_examples": len(ids), "example_ids": sorted(ids), "field_occurrences": {field: field_counts[(category, field)] for field in TEXT_COLUMNS}} for category, ids in categories.items()}


def compare_references(observed: dict[str, int], reference: dict[str, int]) -> dict[str, Any]:
    return {key: {"expected": expected, "observed": observed.get(key), "matches": observed.get(key) == expected} for key, expected in reference.items()}


def analyze(rows: list[dict[str, Any]], minimum_words: int) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    eligible = [row for row in rows if row["eligible"]]
    excluded = [row for row in rows if not row["eligible"]]
    availability = Counter("|".join(f"{c}={str(row['available'][c]).lower()}" for c in TEXT_COLUMNS) for row in rows)
    overlap, links, group_ids = analyze_overlaps(rows, minimum_words)
    return ({
        "row_accounting": {
            "raw_rows": len(rows), "eligible_rows": len(eligible), "excluded_rows": len(excluded),
            "included_plus_excluded_equals_raw": len(eligible) + len(excluded) == len(rows),
            "availability_combinations": dict(sorted(availability.items())),
            "exclusion_reasons": dict(sorted(Counter(row["exclusion_reason"] for row in excluded).items())),
            "unexpected_target_empty_with_context": sum(row["exclusion_reason"] == "target_empty" for row in excluded),
        },
        "annotations": {"raw": annotation_summary(rows), "eligible": annotation_summary(eligible), "excluded": annotation_summary(excluded)},
        "overlap": overlap,
        "lengths": {
            "definition": "Whitespace-separated word counts; p95 uses nearest rank ceil(0.95*n). These are not transformer token counts.",
            "target_eligible": length_summary([r["target_word_count"] for r in eligible]),
            "combined_eligible": length_summary([r["combined_word_count"] for r in eligible]),
            "actual_tokenizer_truncation_evaluated": False,
        },
        "text_quality": text_quality(rows),
    }, links, group_ids)


def observed_reference_counts(analysis: dict[str, Any]) -> dict[str, int]:
    account = analysis["row_accounting"]
    raw = analysis["annotations"]["raw"]
    eligible = analysis["annotations"]["eligible"]
    overlap = analysis["overlap"]
    combinations = account["availability_combinations"]
    return {
        "raw_rows": account["raw_rows"],
        "all_text_available": combinations.get("preceding=true|target=true|following=true", 0),
        "target_following_only": combinations.get("preceding=false|target=true|following=true", 0),
        "all_text_empty": combinations.get("preceding=false|target=false|following=false", 0),
        "nonempty_target": account["eligible_rows"],
        "raw_unanimous": raw["agreement"]["unanimous"],
        "raw_two_versus_one": raw["agreement"]["two-versus-one"],
        "raw_all_different": raw["agreement"]["all-different"],
        "raw_binary_majority_positive": raw["diagnostic_binary_majority"]["positive"],
        "raw_binary_majority_other": raw["diagnostic_binary_majority"]["other"],
        "raw_label_-1": raw["label_counts_overall"]["-1"],
        "raw_label_0": raw["label_counts_overall"]["0"],
        "raw_label_1": raw["label_counts_overall"]["1"],
        "eligible_unanimous": eligible["agreement"]["unanimous"],
        "eligible_two_versus_one": eligible["agreement"]["two-versus-one"],
        "eligible_all_different": eligible["agreement"]["all-different"],
        "eligible_binary_majority_positive": eligible["diagnostic_binary_majority"]["positive"],
        "eligible_binary_majority_other": eligible["diagnostic_binary_majority"]["other"],
        "eligible_rows_with_-1": eligible["rows_with_at_least_one_-1"],
        "eligible_label_-1": eligible["label_counts_overall"]["-1"],
        "eligible_label_0": eligible["label_counts_overall"]["0"],
        "eligible_label_1": eligible["label_counts_overall"]["1"],
        "normalized_repeated_target_groups": overlap["repeated_targets"]["normalized_distinct_strings"],
        "substantial_target_in_other_context_examples": overlap["shared_normalized_text"]["substantial_target_examples_in_other_context"],
        "substantial_shared_text_affected_examples": overlap["shared_normalized_text"]["substantial_affected_examples"],
        "exact_duplicate_eligible_triplets": overlap["repeated_complete_triplets"]["original_groups"],
    }


def safe_project_path(relative: str, label: str) -> Path:
    path = (PROJECT_ROOT / relative).resolve()
    try:
        path.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise AuditInputError(f"{label} path escapes the project root") from exc
    return path


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def write_status(path: Path, status: str, message: str, timestamp: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=".run-status-", suffix=".json", dir=path.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        write_json(temp_path, {"status": status, "timestamp_utc": timestamp, "message": message})
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def write_manifest(path: Path, rows: list[dict[str, Any]], group_ids: dict[str, str]) -> None:
    fields = [
        "source_row", "example_id", "record_sha256", "preceding_available", "target_available", "following_available",
        "eligible", "exclusion_reason", *ANNOTATION_COLUMNS, "agreement_category", "raw_majority_or_tie",
        "diagnostic_binary_majority", "fraction_-1", "fraction_0", "fraction_1", "target_word_count",
        "combined_word_count", "candidate_overlap_group_id",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "source_row": row["source_row"], "example_id": row["example_id"], "record_sha256": row["record_sha256"],
                "preceding_available": row["available"]["preceding"], "target_available": row["available"]["target"],
                "following_available": row["available"]["following"], "eligible": row["eligible"],
                "exclusion_reason": row["exclusion_reason"],
                **{column: label for column, label in zip(ANNOTATION_COLUMNS, row["annotations"])},
                "agreement_category": row["agreement"], "raw_majority_or_tie": row["raw_majority"],
                "diagnostic_binary_majority": row["binary_majority"],
                "fraction_-1": format(row["fractions"][-1], ".12g"), "fraction_0": format(row["fractions"][0], ".12g"),
                "fraction_1": format(row["fractions"][1], ".12g"), "target_word_count": row["target_word_count"],
                "combined_word_count": row["combined_word_count"], "candidate_overlap_group_id": group_ids.get(row["example_id"], ""),
            })


def write_exclusions(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["example_id", "source_row", "reason"], lineterminator="\n")
        writer.writeheader()
        for row in rows:
            if not row["eligible"]:
                writer.writerow({"example_id": row["example_id"], "source_row": row["source_row"], "reason": row["exclusion_reason"]})


def write_links(path: Path, links: list[dict[str, Any]]) -> None:
    fields = ["comparison_setting", "example_id_1", "field_1", "example_id_2", "field_2", "normalized_text_sha256", "word_count", "link_type"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(links)


def markdown_report(summary: dict[str, Any]) -> str:
    account = summary["analysis"]["row_accounting"]
    annotations = summary["analysis"]["annotations"]
    overlap = summary["analysis"]["overlap"]
    references = summary["reference_comparisons"]
    differences = [f"- `{key}`: expected {item['expected']}, observed {item['observed']}" for key, item in references.items() if not item["matches"]]
    def annotation_row(name: str) -> str:
        item = annotations[name]
        labels = item["label_counts_overall"]
        agreements = item["agreement"]
        binary = item["diagnostic_binary_majority"]
        return f"| {name.title()} | {item['rows']} | {labels['-1']} | {labels['0']} | {labels['1']} | {agreements['unanimous']} | {agreements['two-versus-one']} | {agreements['all-different']} | {binary['positive']} | {binary['other']} |"
    lines = [
        "# Phase 01 Dataset Audit", "", f"**Status:** `{summary['status']}`  ",
        f"**Generated (UTC):** {summary['timestamp_utc']}  ",
        f"**Source integrity:** before and after SHA-256 {'match' if summary['source']['unchanged_during_audit'] else 'DO NOT MATCH'}; configured reference {'matches' if summary['source']['matches_reference_sha256'] else 'does not match'}. All access was read-only.", "",
        "Audit status describes implemented checks, not scientific validity or model quality.", "", "## Row accounting", "",
        f"The CSV contains {account['raw_rows']} logical data records. `source_row` is one-based after the header; CSV records can span physical lines. {account['eligible_rows']} rows have a nonempty target and are eligible; {account['excluded_rows']} are excluded. Included plus excluded equals raw: {account['included_plus_excluded_equals_raw']}.", "",
        "| Exclusion reason | Rows |", "|---|---:|",
        *[f"| {reason} | {count} |" for reason, count in account["exclusion_reasons"].items()], "",
        f"Target-empty rows with available context (an unexpected pattern): {account['unexpected_target_empty_with_context']}.", "",
        "## Annotation summaries", "",
        "Supplied meanings: `1` = ableist towards autistic people; `0` = not ableist towards autistic people; `-1` = unrelated to autism or needs more context. There are three annotations per example. Column names do not establish that the same three individuals annotated the entire dataset.", "",
        "| Scope | Rows | -1 | 0 | 1 | Unanimous | Two-versus-one | All-different/tie | Binary positive | Binary other |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|", annotation_row("raw"), annotation_row("eligible"), annotation_row("excluded"), "",
        "The binary mapping is a descriptive benchmark only: `1` maps to 1, while `0` and `-1` map to 0, with positive requiring at least two votes. It does not claim unresolved content is substantively non-ableist and does not finalize training targets.", "",
        "## Exact overlap diagnostics", "",
        f"Eligible rows only were compared. Original text and diagnostic whitespace-normalized plus Unicode-casefolded text were indexed; empty strings never formed groups. Substantial matches require at least {overlap['minimum_substantial_word_count']} whitespace-separated words.", "",
        f"- Repeated normalized target strings: {overlap['repeated_targets']['normalized_distinct_strings']} groups affecting {overlap['repeated_targets']['normalized_affected_examples']} examples; {overlap['repeated_targets']['groups_with_conflicting_annotation_triples']} groups have conflicting annotation triples.",
        f"- Exact duplicate usable full triplets: {overlap['repeated_complete_triplets']['original_groups']} groups affecting {overlap['repeated_complete_triplets']['original_affected_examples']} examples.",
        f"- Substantial shared normalized sentences: {overlap['shared_normalized_text']['substantial_distinct_strings']} strings affecting {overlap['shared_normalized_text']['substantial_affected_examples']} examples.",
        f"- Target examples whose substantial target appears in another row's context: {overlap['shared_normalized_text']['substantial_target_examples_in_other_context']}.",
        f"- Candidate nontrivial connected components: {overlap['candidate_components']['nontrivial_component_count']}; size distribution: `{json.dumps(overlap['candidate_components']['size_distribution'], sort_keys=True)}`.", "",
        "These are candidate text-overlap groups, not verified source-post IDs. Short matches remain separately identified in the link evidence and do not by themselves establish a common Reddit post. The diagnostics do not authorize lowercasing model input. Future splitting must resolve grouping policy before partitions are created.", "",
        "## Length and text quality", "",
        "Lengths are whitespace-separated word counts. The 95th percentile uses nearest rank (`ceil(0.95*n)`). They are not tokenizer token counts, and actual tokenizer truncation has not been evaluated.", "",
        f"- Eligible target: `{json.dumps(summary['analysis']['lengths']['target_eligible'], sort_keys=True)}`",
        f"- Eligible combined context/target: `{json.dumps(summary['analysis']['lengths']['combined_eligible'], sort_keys=True)}`", "",
        "Unusual-text diagnostics are reported in `summary.json` by count and example ID without exposing sentences; no unusual text was automatically modified or discarded.", "",
        "## Reference comparison", "",
        *(differences if differences else ["All configured reference counts match the observed definitions."]), "",
        "## Limitations and decisions pending", "",
        "- Candidate overlap components need research review before defining split groups.",
        "- The treatment of `-1`, hard versus soft targets, and any model-input normalization remain unresolved.",
        "- Word-count thresholds do not predict tokenizer truncation.",
        "- Exact text overlap cannot verify source-post identity or semantic paraphrase.", "",
        "No splits, training, calibration, selective prediction, or model evaluation were performed in Phase 1.", "",
        "## Warnings and failures", "",
        *([f"- Warning: {item}" for item in summary["warnings"]] or ["- No warnings."]),
        *([f"- Failure: {item}" for item in summary["failures"]] or ["- No blocking failures."]), "",
    ]
    return "\n".join(lines)


def run_audit(config_path: Path) -> int:
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    config = load_config(config_path)
    output_paths = {key: safe_project_path(value, f"output {key}") for key, value in config["outputs"].items()}
    status_path = output_paths["run_status"]
    write_status(status_path, "RUNNING", "Audit started; previous outputs are not current until this becomes COMPLETED.", timestamp)
    source = safe_project_path(config["input_csv"], "input")
    if not source.is_file():
        raise AuditInputError(f"Input CSV not found: {config['input_csv']}")
    before_hash = sha256_file(source)
    source_size = source.stat().st_size
    rows, structure = read_rows(source, config["expected_columns"])
    analysis, links, group_ids = analyze(rows, config["overlap"]["minimum_substantial_word_count"])
    after_hash = sha256_file(source)
    observed = observed_reference_counts(analysis)
    comparisons = compare_references(observed, config["reference"]["counts"])
    failures = []
    warnings = []
    if before_hash != after_hash:
        failures.append("Source SHA-256 changed while the audit was running.")
    if before_hash != config["reference"]["sha256"]:
        failures.append("Source SHA-256 differs from the configured independent reference; provenance review is required.")
    mismatches = [key for key, item in comparisons.items() if not item["matches"]]
    if mismatches:
        failures.append("Observed reference-count discrepancies require review: " + ", ".join(mismatches))
    excluded_count = analysis["row_accounting"]["excluded_rows"]
    if excluded_count:
        warnings.append(f"{excluded_count} rows lack a usable target and are excluded from audit eligibility.")
    substantial = analysis["overlap"]["shared_normalized_text"]["substantial_affected_examples"]
    if substantial:
        warnings.append(f"{substantial} eligible rows participate in substantial exact-text overlap; split grouping remains pending.")
    raw_labels = analysis["annotations"]["raw"]["label_counts_overall"]
    if len(set(raw_labels.values())) > 1:
        warnings.append("Raw annotation labels are imbalanced; this is descriptive and not an automatic usability failure.")
    status = "FAIL" if failures else ("PASS_WITH_WARNINGS" if warnings else "PASS")
    summary = {
        "audit_schema_version": config["audit_schema_version"], "timestamp_utc": timestamp, "status": status,
        "status_scope": "Audit checks only; not scientific validity or model quality.",
        "source": {"filename": source.name, "byte_size": source_size, "sha256_before": before_hash, "sha256_after": after_hash,
                   "unchanged_during_audit": before_hash == after_hash, "reference_sha256": config["reference"]["sha256"],
                   "matches_reference_sha256": before_hash == config["reference"]["sha256"]},
        "runtime": {"python_version": platform.python_version(), "platform": platform.platform(), "audit_script_sha256": sha256_file(Path(__file__))},
        "configuration": {"filename": str(config_path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(config_path), "effective": config},
        "structure": {**structure, "encoding": "utf-8-sig", "csv_strict_mode": True, "source_row_definition": "One-based logical data-record index excluding the header; records may span physical lines."},
        "analysis": analysis, "reference_comparisons": comparisons, "warnings": warnings, "failures": failures,
        "not_run": ["dataset splits", "model training", "calibration", "selective prediction", "model evaluation", "tokenizer truncation evaluation"],
    }
    with tempfile.TemporaryDirectory(prefix="phase01-audit-stage-", dir=PROJECT_ROOT) as temp_dir:
        stage_root = Path(temp_dir)
        staged = {}
        for key in ("summary", "row_manifest", "exclusions", "overlap_links", "report"):
            relative = output_paths[key].relative_to(PROJECT_ROOT)
            staged[key] = stage_root / relative
        write_json(staged["summary"], summary)
        write_manifest(staged["row_manifest"], rows, group_ids)
        write_exclusions(staged["exclusions"], rows)
        write_links(staged["overlap_links"], links)
        staged["report"].parent.mkdir(parents=True, exist_ok=True)
        staged["report"].write_text(markdown_report(summary), encoding="utf-8", newline="\n")
        if sha256_file(source) != before_hash:
            raise AuditInputError("Source changed before staged outputs could be published")
        for key in ("row_manifest", "exclusions", "overlap_links", "report", "summary"):
            output_paths[key].parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged[key], output_paths[key])
    write_status(status_path, "COMPLETED", f"Audit completed with result {status}; inspect summary.json.", timestamp)
    print(f"Phase 1 audit completed: {status}")
    print(f"Summary: {output_paths['summary'].relative_to(PROJECT_ROOT)}")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the read-only SANDARBH Phase 1 dataset audit.")
    parser.add_argument("--config", default="configs/audit.json", help="Configuration path relative to project root")
    args = parser.parse_args(argv)
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    default_status = PROJECT_ROOT / "results" / "audit" / "run_status.json"
    try:
        config_path = safe_project_path(args.config, "configuration")
        return run_audit(config_path)
    except AuditInputError as exc:
        try:
            write_status(default_status, "FAILED", f"Input/configuration/parsing failure: {exc}", timestamp)
        except OSError:
            pass
        print(f"Audit failed: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        try:
            write_status(default_status, "FAILED", f"Unexpected failure ({type(exc).__name__}); normal outputs are not current.", timestamp)
        except OSError:
            pass
        print(f"Audit failed unexpectedly: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
