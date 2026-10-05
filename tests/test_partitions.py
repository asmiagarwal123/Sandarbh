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
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


prepare = load("prepare_partitions_tests", ROOT / "scripts" / "prepare_partitions.py")
validator = load("validate_partitions_tests", ROOT / "scripts" / "validate_partitions.py")
HEADER = ["preceding", "target", "following", "A1_Score", "A2_Score", "A3_Score"]


class TargetAndAllocationTests(unittest.TestCase):
    def fake_row(self, labels, index=1):
        distinct = len(set(labels))
        agreement = "unanimous" if distinct == 1 else ("two-versus-one" if distinct == 2 else "all-different")
        return {"example_id": f"row-{index}", "source_row": index, "annotations": labels, "agreement": agreement}

    def test_all_three_vote_combinations_map_correctly(self):
        for index, labels in enumerate(product((-1, 0, 1), repeat=3), 1):
            with self.subTest(labels=labels):
                target = prepare.make_target(self.fake_row(labels, index), f"group-{index}")
                positives = labels.count(1)
                self.assertEqual(target["positive_vote_count"], positives)
                self.assertEqual(target["hard_label"], int(positives >= 2))
                self.assertAlmostEqual(target["soft_positive"], positives / 3)
                self.assertAlmostEqual(target["soft_positive"] + target["soft_other"], 1.0)

    def test_minus1_distinction_and_all_different_preserved(self):
        target = prepare.make_target(self.fake_row((-1, 0, 1)), "group")
        self.assertEqual(target["hard_label"], 0)
        self.assertAlmostEqual(target["soft_positive"], 1 / 3)
        self.assertAlmostEqual(target["soft_other"], 2 / 3)
        self.assertAlmostEqual(target["fraction_minus1"], 1 / 3)
        self.assertAlmostEqual(target["fraction_zero"], 1 / 3)
        self.assertTrue(target["has_minus1"])
        self.assertEqual(target["agreement_category"], "all-different")

    def synthetic_config(self):
        return {
            "partitions": {p: 0.25 for p in prepare.PARTITIONS},
            "allocation": {"base_seed": 42, "candidate_budget": 32, "candidate_seed_stride": 1000003,
                           "example_fraction_tolerance": 0.20, "positive_rate_tolerance": 0.50,
                           "require_both_hard_classes": True},
        }

    def groups(self):
        return [{"group_id": f"g-{i:02d}", "size": 1, "positive": i % 2, "other": 1 - i % 2} for i in range(12)]

    def test_reordering_groups_does_not_change_assignment(self):
        config = self.synthetic_config()
        first = prepare.select_assignment(self.groups(), config)
        second = prepare.select_assignment(list(reversed(self.groups())), config)
        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])

    def test_fixed_configuration_reproduces_substantive_output(self):
        config = self.synthetic_config()
        self.assertEqual(prepare.select_assignment(self.groups(), config), prepare.select_assignment(self.groups(), config))

    def test_infeasible_class_presence_reported_honestly(self):
        config = self.synthetic_config()
        groups = [{"group_id": f"g-{i}", "size": 1, "positive": 0, "other": 1} for i in range(8)]
        _, metrics, _ = prepare.select_assignment(groups, config)
        self.assertFalse(metrics["feasible"])
        self.assertGreater(metrics["constraint_violation"], 0)

    def test_excluded_rows_never_enter_targets(self):
        eligible = [self.fake_row((0, 0, 0), 1)]
        memberships = [{"example_id": "row-1", "rule_b_group_id": "group-1"}]
        targets = prepare.build_targets(eligible, memberships)
        self.assertEqual([row["example_id"] for row in targets], ["row-1"])


class SyntheticProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_temp = tempfile.TemporaryDirectory()
        cls.base = Path(cls.base_temp.name) / "base"
        cls._build_frozen_project(cls.base)

    @classmethod
    def tearDownClass(cls):
        cls.base_temp.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "project"
        shutil.copytree(self.base, self.root)

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def run_cli(root, script, config="configs/partition.json"):
        return subprocess.run([sys.executable, script, "--config", config], cwd=root, text=True, capture_output=True, check=False)

    @classmethod
    def _build_frozen_project(cls, root):
        (root / "scripts").mkdir(parents=True)
        (root / "configs").mkdir()
        for name in ("audit_dataset.py", "review_overlap_groups.py", "prepare_partitions.py", "validate_partitions.py"):
            shutil.copy2(ROOT / "scripts" / name, root / "scripts" / name)
        source = root / "fixture.csv"
        rows = []
        for index in range(8):
            preceding = "shared neutral greeting" if index < 2 else ""
            labels = (1, 1, 0) if index % 2 == 0 else (0, 0, -1)
            rows.append([preceding, f"synthetic neutral target number {index}", "", *labels])
        rows.append(["", "", "", 0, 0, 0])
        with source.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(HEADER)
            writer.writerows(rows)
        audit_config = json.loads((ROOT / "configs/audit.json").read_text(encoding="utf-8"))
        audit_config["input_csv"] = "fixture.csv"
        audit_config["reference"] = {"sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "counts": {}}
        (root / "configs/audit.json").write_text(json.dumps(audit_config), encoding="utf-8")
        phase1 = cls.run_cli(root, "scripts/audit_dataset.py", "configs/audit.json")
        if phase1.returncode != 0:
            raise AssertionError(phase1.stderr)
        overlap_config = json.loads((ROOT / "configs/overlap_review.json").read_text(encoding="utf-8"))
        overlap_config["input_csv"] = "fixture.csv"
        (root / "configs/overlap_review.json").write_text(json.dumps(overlap_config), encoding="utf-8")
        phase2a = cls.run_cli(root, "scripts/review_overlap_groups.py", "configs/overlap_review.json")
        if phase2a.returncode != 0:
            raise AssertionError(phase2a.stderr)
        partition_config = json.loads((ROOT / "configs/partition.json").read_text(encoding="utf-8"))
        partition_config["input_csv"] = "fixture.csv"
        partition_config["partitions"] = {p: 0.25 for p in prepare.PARTITIONS}
        partition_config["allocation"].update({"candidate_budget": 128, "example_fraction_tolerance": 0.20,
                                                "positive_rate_tolerance": 0.50})
        partition_config["reference_observations"] = {"eligible_examples": 8, "excluded_examples": 1,
            "hard_positive_examples": 4, "hard_other_examples": 4, "rule_b_groups": 7,
            "rule_b_singletons": 6, "rule_b_largest_group": 2}
        (root / "configs/partition.json").write_text(json.dumps(partition_config), encoding="utf-8")
        phase2b = cls.run_cli(root, "scripts/prepare_partitions.py")
        if phase2b.returncode != 0:
            raise AssertionError(phase2b.stderr + phase2b.stdout)

    def csv_rows(self, relative):
        with (self.root / relative).open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def test_every_eligible_once_groups_intact_and_roles_reconcile(self):
        manifest = self.csv_rows("results/partitions/split_manifest.csv")
        self.assertEqual(len(manifest), 8)
        self.assertEqual(len({row["example_id"] for row in manifest}), 8)
        by_group = {}
        for row in manifest:
            by_group.setdefault(row["group_id"], set()).add(row["partition"])
            expected_outer = "validation" if row["partition"].startswith("dev_") else row["partition"]
            self.assertEqual(row["outer_split"], expected_outer)
        self.assertTrue(all(len(values) == 1 for values in by_group.values()))

    def test_independent_validator_passes_and_summary_reconciles(self):
        result = self.run_cli(self.root, "scripts/validate_partitions.py")
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        parsed = json.loads(result.stdout)
        self.assertTrue(parsed["passed"])
        self.assertTrue(parsed["checks"]["saved_summary_reconciles"])
        self.assertTrue(parsed["checks"]["no_cross_partition_nonempty_exact_normalized_field_overlap"])

    def test_unknown_missing_and_duplicate_ids_fail(self):
        for mode in ("unknown", "missing", "duplicate"):
            with self.subTest(mode=mode):
                shutil.rmtree(self.root)
                shutil.copytree(self.base, self.root)
                path = self.root / "results/partitions/targets.csv"
                rows = self.csv_rows("results/partitions/targets.csv")
                fields = list(rows[0])
                if mode == "unknown":
                    rows[0]["example_id"] = "unknown-id"
                elif mode == "missing":
                    rows.pop()
                else:
                    rows.append(rows[0].copy())
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
                result = self.run_cli(self.root, "scripts/validate_partitions.py")
                self.assertNotEqual(result.returncode, 0)

    def test_altered_source_annotation_and_hash_detected(self):
        source = self.root / "fixture.csv"
        content = source.read_text(encoding="utf-8-sig")
        source.write_text(content.replace(",1,1,0", ",0,1,0", 1), encoding="utf-8-sig")
        result = self.run_cli(self.root, "scripts/validate_partitions.py")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("provenance", result.stdout.lower())

    def test_validator_detects_cross_partition_overlap(self):
        path = self.root / "results/partitions/split_manifest.csv"
        rows = self.csv_rows("results/partitions/split_manifest.csv")
        fields = list(rows[0])
        shared = [row for row in rows if row["source_row"] in {"1", "2"}]
        self.assertEqual(len(shared), 2)
        new_partition = next(p for p in prepare.PARTITIONS if p != shared[0]["partition"])
        shared[1]["partition"] = new_partition
        shared[1]["outer_split"] = "validation" if new_partition.startswith("dev_") else new_partition
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
        result = self.run_cli(self.root, "scripts/validate_partitions.py")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("cross-partition exact normalized field overlap", result.stdout.lower())

    def test_saved_summary_corruption_detected(self):
        path = self.root / "results/partitions/split_summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        summary["partition_diagnostics"]["train"]["examples"] += 1
        path.write_text(json.dumps(summary), encoding="utf-8")
        result = self.run_cli(self.root, "scripts/validate_partitions.py")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("reconcile", result.stdout.lower())

    def test_frozen_output_reused_without_membership_change(self):
        manifest = self.root / "results/partitions/split_manifest.csv"
        before = hashlib.sha256(manifest.read_bytes()).hexdigest()
        result = self.run_cli(self.root, "scripts/prepare_partitions.py")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("reused unchanged", result.stdout)
        self.assertEqual(hashlib.sha256(manifest.read_bytes()).hexdigest(), before)

    def test_changed_provenance_refuses_overwrite_and_marks_latest_failure(self):
        manifest = self.root / "results/partitions/split_manifest.csv"
        before = hashlib.sha256(manifest.read_bytes()).hexdigest()
        config_path = self.root / "configs/partition.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["allocation"]["candidate_budget"] += 1
        config_path.write_text(json.dumps(config), encoding="utf-8")
        result = self.run_cli(self.root, "scripts/prepare_partitions.py")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(hashlib.sha256(manifest.read_bytes()).hexdigest(), before)
        status = json.loads((self.root / "results/partitions/run_status.json").read_text(encoding="utf-8"))
        self.assertEqual(status["status"], "FAILED")

    def test_source_and_upstream_artifacts_unchanged(self):
        protected = ["fixture.csv", "results/audit/summary.json", "results/audit/row_manifest.csv",
                     "results/overlap_review/policy_comparison.json", "results/overlap_review/group_membership.csv",
                     "results/overlap_review/manual_review.csv"]
        before = {name: hashlib.sha256((self.root / name).read_bytes()).hexdigest() for name in protected}
        result = self.run_cli(self.root, "scripts/validate_partitions.py")
        self.assertEqual(result.returncode, 0)
        after = {name: hashlib.sha256((self.root / name).read_bytes()).hexdigest() for name in protected}
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
