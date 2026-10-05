from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "audit_dataset.py"
SPEC = importlib.util.spec_from_file_location("audit_dataset", MODULE_PATH)
audit = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(audit)

HEADER = ["preceding", "target", "following", "A1_Score", "A2_Score", "A3_Score"]


class DatasetAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write_csv(self, rows, header=HEADER, name="fixture.csv"):
        path = self.root / name
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(header)
            writer.writerows(rows)
        return path

    def read(self, rows, header=HEADER):
        return audit.read_rows(self.write_csv(rows, header), HEADER)[0]

    def test_valid_schema_labels_and_missing_preceding_retained(self):
        rows = self.read([["", "A neutral target has enough words", "Neutral context", " -1 ", "0", "1"]])
        self.assertTrue(rows[0]["eligible"])
        self.assertFalse(rows[0]["available"]["preceding"])
        self.assertEqual(rows[0]["annotations"], (-1, 0, 1))

    def test_missing_duplicate_and_unexpected_headers_fail(self):
        cases = [
            HEADER[:-1],
            ["preceding", "target", "following", "A1_Score", "A2_Score", "A2_Score"],
            HEADER + ["extra"],
        ]
        for index, header in enumerate(cases):
            with self.subTest(header=header):
                path = self.write_csv([], header, f"bad-{index}.csv")
                with self.assertRaises(audit.AuditInputError):
                    audit.read_rows(path, HEADER)

    def test_malformed_record_fails(self):
        path = self.write_csv([["p", "t", "f", "0", "1"]])
        with self.assertRaisesRegex(audit.AuditInputError, "Malformed record"):
            audit.read_rows(path, HEADER)

    def test_missing_and_invalid_annotations_fail(self):
        for value in ("", "2", "0.0", "unknown"):
            with self.subTest(value=value):
                path = self.write_csv([["p", "t", "f", value, "0", "1"]], name=f"label-{repr(value)}.csv")
                with self.assertRaises(audit.AuditInputError):
                    audit.read_rows(path, HEADER)

    def test_all_empty_and_target_empty_reasons(self):
        rows = self.read([
            [" ", "", "\t", "0", "0", "0"],
            ["Available neutral context", " ", "", "0", "0", "0"],
        ])
        self.assertEqual(rows[0]["exclusion_reason"], "all_text_empty")
        self.assertEqual(rows[1]["exclusion_reason"], "target_empty")
        self.assertFalse(any(row["eligible"] for row in rows))

    def test_agreement_majority_and_fractions(self):
        rows = self.read([
            ["", "Neutral target one", "", "1", "1", "1"],
            ["", "Neutral target two", "", "1", "1", "0"],
            ["", "Neutral target three", "", "-1", "0", "1"],
            ["", "Neutral target four", "", "-1", "1", "1"],
        ])
        self.assertEqual([row["agreement"] for row in rows[:3]], ["unanimous", "two-versus-one", "all-different"])
        self.assertEqual(rows[2]["raw_majority"], "tie")
        self.assertEqual([row["binary_majority"] for row in rows], [1, 1, 0, 1])
        for row in rows:
            self.assertAlmostEqual(sum(row["fractions"].values()), 1.0)
        summary = audit.annotation_summary(rows)
        self.assertEqual(list(summary["label_counts_overall"]), ["-1", "0", "1"])

    def test_normalized_repeated_target_and_conflicting_annotations(self):
        rows = self.read([
            ["", "A Calm   Neutral Target Here", "", "0", "0", "0"],
            ["", " a calm neutral target HERE ", "", "1", "1", "0"],
        ])
        summary, links, _ = audit.analyze_overlaps(rows, 5)
        self.assertEqual(summary["repeated_targets"]["normalized_distinct_strings"], 1)
        self.assertEqual(summary["repeated_targets"]["groups_with_conflicting_annotation_triples"], 1)
        self.assertTrue(any(link["link_type"] == "target_target_substantial" for link in links))

    def test_target_to_context_overlap_with_different_triplets(self):
        sentence = "five neutral words form this sentence"
        rows = self.read([
            ["Earlier note", sentence, "Later note", "0", "0", "0"],
            [sentence.upper(), "Entirely different target wording", "Different ending", "0", "0", "0"],
        ])
        summary, _, _ = audit.analyze_overlaps(rows, 5)
        self.assertEqual(summary["shared_normalized_text"]["substantial_target_examples_in_other_context"], 1)
        self.assertEqual(summary["repeated_complete_triplets"]["original_groups"], 0)

    def test_empty_and_short_strings_do_not_form_substantial_groups(self):
        rows = self.read([
            ["", "Short phrase", "", "0", "0", "0"],
            ["", "short   PHRASE", "", "0", "0", "0"],
        ])
        summary, _, groups = audit.analyze_overlaps(rows, 5)
        self.assertEqual(summary["shared_normalized_text"]["substantial_affected_examples"], 0)
        self.assertEqual(groups, {})

    def test_connected_component_transitivity(self):
        alpha = "alpha neutral sentence contains five words"
        beta = "beta neutral sentence contains five words"
        rows = self.read([
            ["", alpha, "", "0", "0", "0"],
            [alpha, "Bridge target is independently neutral", beta, "0", "0", "0"],
            ["", beta, "", "0", "0", "0"],
        ])
        summary, _, groups = audit.analyze_overlaps(rows, 5)
        self.assertEqual(summary["candidate_components"]["size_distribution"], {"3": 1})
        self.assertEqual(len(set(groups.values())), 1)

    def test_overlap_links_are_unique_and_text_free(self):
        sentence = "shared neutral sentence has enough words"
        rows = self.read([
            ["", sentence, "", "0", "0", "0"],
            [sentence, "Another fully neutral target sentence", "", "0", "0", "0"],
        ])
        _, links, _ = audit.analyze_overlaps(rows, 5)
        serialized = [tuple(link.items()) for link in links]
        self.assertEqual(len(serialized), len(set(serialized)))
        self.assertTrue(all(sentence not in json.dumps(link) for link in links))

    def test_original_bytes_unchanged(self):
        path = self.write_csv([["Context", "Neutral target", "Ending", "-1", "0", "1"]])
        before = path.read_bytes()
        rows, _ = audit.read_rows(path, HEADER)
        audit.analyze(rows, 5)
        self.assertEqual(path.read_bytes(), before)

    def test_stable_ids_and_deterministic_substantive_analysis(self):
        data = [["Context", "A neutral repeatable target", "Ending", "-1", "0", "1"]]
        path = self.write_csv(data)
        first, _ = audit.read_rows(path, HEADER)
        second, _ = audit.read_rows(path, HEADER)
        self.assertEqual(first[0]["example_id"], second[0]["example_id"])
        self.assertEqual(audit.analyze(first, 5), audit.analyze(second, 5))

    def test_quality_diagnostics_do_not_echo_text(self):
        rows = self.read([["NAN", "Neutral \ufffd target", "Ending\x01", "0", "0", "0"]])
        quality = audit.text_quality(rows)
        self.assertEqual(quality["literal_missing_value_placeholder"]["affected_examples"], 1)
        self.assertEqual(quality["replacement_character"]["affected_examples"], 1)
        self.assertEqual(quality["unexpected_control_character"]["affected_examples"], 1)
        self.assertNotIn("Neutral", json.dumps(quality))


class SyntheticCliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "scripts").mkdir()
        (self.root / "configs").mkdir()
        (self.root / "scripts" / "audit_dataset.py").write_bytes(MODULE_PATH.read_bytes())

    def tearDown(self):
        self.temp.cleanup()

    def make_config(self, csv_path):
        config = {
            "audit_schema_version": "test-1", "input_csv": csv_path.name, "expected_columns": HEADER,
            "text_columns": ["preceding", "target", "following"], "annotation_columns": ["A1_Score", "A2_Score", "A3_Score"],
            "outputs": {"summary": "results/audit/summary.json", "row_manifest": "results/audit/row_manifest.csv",
                        "exclusions": "results/audit/exclusions.csv", "overlap_links": "results/audit/overlap_links.csv",
                        "report": "reports/phase_01_dataset_audit.md", "run_status": "results/audit/run_status.json"},
            "reference": {"sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(), "counts": {}},
            "overlap": {"normalization": "test", "minimum_substantial_word_count": 5,
                        "comparison_settings": ["original", "whitespace_casefold"]},
        }
        (self.root / "configs" / "audit.json").write_text(json.dumps(config), encoding="utf-8")

    def write_source(self, malformed=False):
        path = self.root / "fixture.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(HEADER)
            writer.writerow(["Neutral context", "Neutral target has several words", "Neutral ending", "0", "0"] if malformed else
                            ["Neutral context", "Neutral target has several words", "Neutral ending", "0", "0", "0"])
        return path

    def run_cli(self):
        return subprocess.run([sys.executable, "scripts/audit_dataset.py", "--config", "configs/audit.json"], cwd=self.root,
                              text=True, capture_output=True, check=False)

    def test_tiny_valid_cli_and_failed_rerun_status(self):
        source = self.write_source()
        self.make_config(source)
        success = self.run_cli()
        self.assertEqual(success.returncode, 0, success.stderr)
        summary = self.root / "results" / "audit" / "summary.json"
        self.assertTrue(summary.is_file())
        old_summary = summary.read_bytes()

        source = self.write_source(malformed=True)
        self.make_config(source)
        failed = self.run_cli()
        self.assertEqual(failed.returncode, 2)
        status = json.loads((self.root / "results" / "audit" / "run_status.json").read_text(encoding="utf-8"))
        self.assertEqual(status["status"], "FAILED")
        self.assertEqual(summary.read_bytes(), old_summary)


if __name__ == "__main__":
    unittest.main()
