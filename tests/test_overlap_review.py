from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("review_overlap_groups", ROOT / "scripts" / "review_overlap_groups.py")
review = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(review)

HEADER = ["preceding", "target", "following", "A1_Score", "A2_Score", "A3_Score"]


class GroupingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def rows(self, values):
        path = self.root / "fixture.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(HEADER)
            writer.writerows(values)
        return review.audit.read_rows(path, HEADER)[0], path

    def test_substantial_target_context_and_context_context_grouping(self):
        shared_target = "substantial neutral target context match"
        shared_context = "substantial neutral context shared here"
        rows, _ = self.rows([
            ["", shared_target, shared_context, "0", "0", "0"],
            [shared_target, "Another neutral target has words", "", "0", "0", "0"],
            ["", "Third neutral target has words", shared_context.upper(), "0", "0", "0"],
        ])
        grouping = review.build_groupings(rows, 5)
        self.assertEqual(len({grouping["rule_a_membership"][row["example_id"]] for row in rows}), 1)

    def test_repeated_short_targets_grouped_by_a(self):
        rows, _ = self.rows([
            ["", "brief neutral", "", "0", "0", "0"],
            ["", " BRIEF   NEUTRAL ", "", "1", "0", "0"],
        ])
        grouping = review.build_groupings(rows, 5)
        ids = [row["example_id"] for row in rows]
        self.assertEqual(grouping["rule_a_membership"][ids[0]], grouping["rule_a_membership"][ids[1]])
        self.assertNotEqual(grouping["five_membership"][ids[0]], grouping["five_membership"][ids[1]])

    def test_empty_never_groups_and_singletons_preserved(self):
        rows, _ = self.rows([
            ["", "Unique neutral target one", "", "0", "0", "0"],
            ["", "Unique neutral target two", "", "0", "0", "0"],
        ])
        grouping = review.build_groupings(rows, 5)
        self.assertEqual(len(grouping["rule_a_groups"]), 2)
        self.assertEqual(len(grouping["rule_b_groups"]), 2)
        self.assertTrue(all(len(group) == 1 for group in grouping["rule_b_groups"].values()))

    def test_short_context_excluded_by_a_included_by_b(self):
        rows, _ = self.rows([
            ["shared phrase", "First distinct neutral target", "", "0", "0", "0"],
            ["", "Second distinct neutral target", "SHARED  PHRASE", "0", "0", "0"],
        ])
        grouping = review.build_groupings(rows, 5)
        ids = [row["example_id"] for row in rows]
        self.assertNotEqual(grouping["rule_a_membership"][ids[0]], grouping["rule_a_membership"][ids[1]])
        self.assertEqual(grouping["rule_b_membership"][ids[0]], grouping["rule_b_membership"][ids[1]])

    def test_transitive_connections(self):
        alpha = "alpha neutral sentence has five words"
        beta = "beta neutral sentence has five words"
        rows, _ = self.rows([
            ["", alpha, "", "0", "0", "0"],
            [alpha, "Bridge target remains entirely neutral", beta, "0", "0", "0"],
            ["", beta, "", "0", "0", "0"],
        ])
        grouping = review.build_groupings(rows, 5)
        self.assertEqual(len(set(grouping["rule_a_membership"].values())), 1)

    def test_excluded_rows_omitted(self):
        rows, _ = self.rows([
            ["", "", "", "0", "0", "0"],
            ["", "Eligible neutral target", "", "0", "0", "0"],
        ])
        eligible = [row for row in rows if row["eligible"]]
        grouping = review.build_groupings(eligible, 5)
        self.assertEqual(set(grouping["rule_a_membership"]), {eligible[0]["example_id"]})

    def test_input_order_does_not_change_memberships_or_ids(self):
        shared = "shared substantial neutral sentence here"
        rows, _ = self.rows([
            ["", shared, "", "0", "0", "0"],
            [shared, "Different neutral target", "", "0", "0", "0"],
            ["", "Independent neutral target", "", "0", "0", "0"],
        ])
        first = review.build_groupings(rows, 5)
        second = review.build_groupings(list(reversed(rows)), 5)
        self.assertEqual(first["rule_a_membership"], second["rule_a_membership"])
        self.assertEqual(first["rule_b_membership"], second["rule_b_membership"])

    def test_rule_a_containment_and_counts_reconcile(self):
        rows, _ = self.rows([
            ["brief shared", "First distinct target", "", "1", "1", "0"],
            ["", "Second distinct target", "brief shared", "0", "0", "0"],
            ["", "Third unique target", "", "0", "0", "0"],
        ])
        grouping = review.build_groupings(rows, 5)
        checks = review.validate_groupings(grouping, rows, 5)
        self.assertTrue(checks["every_rule_a_group_contained_in_one_rule_b_group"])
        stats = review.group_statistics(grouping["rule_a_groups"], {row["example_id"]: row for row in rows})
        self.assertEqual(sum(int(size) * count for size, count in stats["group_size_distribution"].items()), len(rows))

    def test_review_selection_is_deterministic_and_text_free(self):
        long_text = "shared substantial neutral sentence here"
        rows, _ = self.rows([
            ["short phrase", long_text, "another substantial neutral context sentence", "0", "0", "0"],
            [long_text, "brief repeat", "SHORT PHRASE", "0", "0", "0"],
            ["another substantial neutral context sentence", " BRIEF REPEAT ", "", "0", "0", "0"],
        ])
        grouping = review.build_groupings(rows, 5)
        by_id = {row["example_id"]: row for row in rows}
        first = review.select_review_cases(grouping, by_id, 5)
        second = review.select_review_cases(grouping, by_id, 5)
        self.assertEqual(first, second)
        output = json.dumps(first)
        self.assertNotIn(long_text, output)
        self.assertTrue(all(not case["reviewer_observation"] and not case["reviewer_notes"] for case in first))

    def test_existing_human_notes_preserved_and_orphans_stop(self):
        path = self.root / "manual.csv"
        base = {field: "" for field in review.MANUAL_FIELDS}
        base.update({"review_case_id": "review-1", "reviewer_observation": "Observed", "reviewer_notes": "Keep this"})
        review.write_csv(path, review.MANUAL_FIELDS, [base])
        cases = [{field: "" for field in review.MANUAL_FIELDS}]
        cases[0]["review_case_id"] = "review-1"
        review.preserve_manual_notes(path, cases)
        self.assertEqual(cases[0]["reviewer_notes"], "Keep this")
        with self.assertRaisesRegex(review.ReviewInputError, "choose a new output directory"):
            review.preserve_manual_notes(path, [{**cases[0], "review_case_id": "review-2"}])

    def test_source_and_artifact_bytes_unchanged_by_grouping(self):
        rows, source = self.rows([["", "Neutral target remains unchanged", "", "0", "0", "0"]])
        artifact = self.root / "artifact.json"
        artifact.write_text('{"status":"COMPLETED"}\n', encoding="utf-8")
        before = source.read_bytes(), artifact.read_bytes()
        review.build_groupings(rows, 5)
        self.assertEqual((source.read_bytes(), artifact.read_bytes()), before)


class InputConsistencyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "source.csv"
        with self.source.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(HEADER)
            writer.writerow(["", "Neutral target", "", "0", "0", "0"])
        self.rows = review.audit.read_rows(self.source, HEADER)[0]
        self.paths = {"source": self.source, "row_manifest": self.root / "manifest.csv", "exclusions": self.root / "exclusions.csv", "overlap_links": self.root / "links.csv"}
        review.write_csv(self.paths["exclusions"], ("example_id", "source_row", "reason"), [])
        review.write_csv(self.paths["overlap_links"], ("example_id_1", "example_id_2"), [])
        self.summary = {"configuration": {"effective": {"expected_columns": HEADER}}, "analysis": {"row_accounting": {"eligible_rows": 1}}}

    def tearDown(self):
        self.temp.cleanup()

    def write_manifest(self, example_id):
        row = self.rows[0]
        review.write_csv(self.paths["row_manifest"], ("source_row", "example_id", "eligible", "exclusion_reason", "A1_Score", "A2_Score", "A3_Score", "diagnostic_binary_majority"), [{
            "source_row": row["source_row"], "example_id": example_id, "eligible": True, "exclusion_reason": "",
            "A1_Score": 0, "A2_Score": 0, "A3_Score": 0, "diagnostic_binary_majority": 0,
        }])

    def test_missing_and_unknown_ids_fail_explicitly(self):
        self.write_manifest("unknown-id")
        with self.assertRaisesRegex(review.ReviewInputError, "missing=.*unknown"):
            review.validate_inputs(self.paths, self.summary)


class FailedCliTests(unittest.TestCase):
    def test_tiny_valid_cli_preserves_phase1_inputs(self):
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            (root / "scripts").mkdir()
            (root / "configs").mkdir()
            shutil.copy2(ROOT / "scripts" / "audit_dataset.py", root / "scripts" / "audit_dataset.py")
            shutil.copy2(ROOT / "scripts" / "review_overlap_groups.py", root / "scripts" / "review_overlap_groups.py")
            source = root / "fixture.csv"
            shared = "shared substantial neutral context sentence"
            with source.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(HEADER)
                writer.writerow(["", shared, "", "0", "0", "0"])
                writer.writerow([shared, "Another neutral target", "", "1", "1", "0"])
            audit_config = json.loads((ROOT / "configs" / "audit.json").read_text(encoding="utf-8"))
            audit_config["input_csv"] = "fixture.csv"
            audit_config["reference"] = {"sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "counts": {}}
            (root / "configs" / "audit.json").write_text(json.dumps(audit_config), encoding="utf-8")
            phase1 = subprocess.run([sys.executable, "scripts/audit_dataset.py", "--config", "configs/audit.json"], cwd=root, text=True, capture_output=True, check=False)
            self.assertEqual(phase1.returncode, 0, phase1.stderr)
            overlap_config = json.loads((ROOT / "configs" / "overlap_review.json").read_text(encoding="utf-8"))
            overlap_config["input_csv"] = "fixture.csv"
            (root / "configs" / "overlap_review.json").write_text(json.dumps(overlap_config), encoding="utf-8")
            protected = [source, root / "results/audit/summary.json", root / "results/audit/row_manifest.csv",
                         root / "results/audit/exclusions.csv", root / "results/audit/overlap_links.csv"]
            before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
            phase2 = subprocess.run([sys.executable, "scripts/review_overlap_groups.py", "--config", "configs/overlap_review.json"], cwd=root, text=True, capture_output=True, check=False)
            self.assertEqual(phase2.returncode, 0, phase2.stderr)
            self.assertEqual(before, {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in protected})
            status = json.loads((root / "results/overlap_review/run_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "COMPLETED")
            with (root / "results/overlap_review/group_membership.csv").open("r", encoding="utf-8", newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 2)

    def test_failed_run_status_not_success(self):
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            (root / "scripts").mkdir()
            (root / "configs").mkdir()
            shutil.copy2(ROOT / "scripts" / "audit_dataset.py", root / "scripts" / "audit_dataset.py")
            shutil.copy2(ROOT / "scripts" / "review_overlap_groups.py", root / "scripts" / "review_overlap_groups.py")
            config = json.loads((ROOT / "configs" / "overlap_review.json").read_text(encoding="utf-8"))
            (root / "configs" / "overlap_review.json").write_text(json.dumps(config), encoding="utf-8")
            result = subprocess.run([sys.executable, "scripts/review_overlap_groups.py", "--config", "configs/overlap_review.json"], cwd=root, text=True, capture_output=True, check=False)
            self.assertEqual(result.returncode, 2)
            status = json.loads((root / "results" / "overlap_review" / "run_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "FAILED")
            self.assertFalse((root / "results" / "overlap_review" / "policy_comparison.json").exists())


if __name__ == "__main__":
    unittest.main()
