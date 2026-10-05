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
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


train = load("train_baselines_tests", ROOT / "scripts" / "train_baselines.py")
validate = load("validate_baseline_tests", ROOT / "scripts" / "validate_baseline_run.py")
HEADER = ["preceding", "target", "following", "A1_Score", "A2_Score", "A3_Score"]


class BaselineUnitTests(unittest.TestCase):
    def test_target_and_context_input_construction(self):
        texts = {"preceding": "  Earlier\tneutral phrase ", "target": " Target\ntext! ", "following": " Later   phrase. "}
        self.assertEqual(train.construct_input(texts, "target"), "Target text!")
        self.assertEqual(train.construct_input(texts, "context"), "Earlier neutral phrase\nTarget text!\nLater phrase.")

    def test_missing_context_and_no_metadata_in_features(self):
        texts = {"preceding": "", "target": "Only target", "following": ""}
        output = train.construct_input(texts, "context")
        self.assertEqual(output, "Only target")
        for forbidden in ("row-000001", "group", "dev_tune", "hard_label"):
            self.assertNotIn(forbidden, output)

    def test_training_only_vocabulary_and_idf(self):
        vectorizer = TfidfVectorizer(min_df=1)
        vectorizer.fit(["common trainword", "common anothertrain"])
        vectorizer.transform(["common devonlytoken"])
        self.assertNotIn("devonlytoken", vectorizer.vocabulary_)
        self.assertEqual(len(vectorizer.idf_), len(vectorizer.vocabulary_))

    def test_calibration_and_test_access_rejected_before_reading(self):
        for partition in ("dev_calibration", "test"):
            with self.subTest(partition=partition), self.assertRaisesRegex(train.BaselineError, "access denied"):
                train.load_modeling_partition(partition, {}, "target")

    def test_candidate_selection_and_tie_breaking(self):
        rows = []
        for C in (0.1, 1.0):
            for weight in (None, "balanced"):
                rows.append({"C": C, "class_weight": weight, "converged": True, "metrics": {"macro_f1": 0.5}})
        selected = train.select_candidate(list(reversed(rows)))
        self.assertEqual(selected["C"], 0.1)
        self.assertIsNone(selected["class_weight"])
        rows[-1]["metrics"]["macro_f1"] = 0.6
        self.assertIs(train.select_candidate(rows), rows[-1])

    def test_known_metrics_and_no_positive_prediction(self):
        result = train.calculate_metrics([0, 0, 1, 1], [0, 0, 0, 0], [0.1, 0.2, 0.3, 0.4])
        self.assertAlmostEqual(result["accuracy"], 0.5)
        self.assertAlmostEqual(result["balanced_accuracy"], 0.5)
        self.assertEqual(result["confusion_matrix_0_1"], [[2, 0], [2, 0]])
        self.assertEqual(result["support_0_1"], [2, 2])
        self.assertEqual(result["positive_precision"], 0.0)
        self.assertTrue(result["no_positive_predictions"])
        self.assertAlmostEqual(result["average_precision"], 1.0)

    def test_class_order_and_score_interpretation(self):
        texts = ["calm neutral", "calm ordinary", "strong signal", "strong marker"]
        labels = [0, 0, 1, 1]
        for family, classifier in (
            ("logistic_regression", LogisticRegression(l1_ratio=0.0, solver="lbfgs", random_state=42)),
            ("linear_svc", LinearSVC(dual=True, random_state=42)),
        ):
            with self.subTest(family=family):
                pipeline = Pipeline([("tfidf", TfidfVectorizer(min_df=1)), ("classifier", classifier)]).fit(texts, labels)
                metrics, pred, decision, probability = train.score_pipeline(pipeline, family, texts, labels)
                self.assertEqual(pipeline.named_steps["classifier"].classes_.tolist(), [0, 1])
                self.assertTrue(np.isfinite(decision).all())
                if family == "logistic_regression":
                    self.assertIsNotNone(probability)
                    self.assertTrue(np.all((probability >= 0) & (probability <= 1)))
                else:
                    self.assertIsNone(probability)

    def test_convergence_warning_recognition(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.warn("synthetic nonconvergence", ConvergenceWarning)
        self.assertTrue(train.contains_convergence_warning(caught))
        with warnings.catch_warnings(record=True) as ordinary:
            warnings.simplefilter("always")
            warnings.warn("ordinary", UserWarning)
        self.assertFalse(train.contains_convergence_warning(ordinary))

    def test_serialization_roundtrip(self):
        pipeline = Pipeline([("tfidf", TfidfVectorizer(min_df=1)),
                             ("classifier", LogisticRegression(l1_ratio=0.0, solver="lbfgs"))])
        texts, labels = ["neutral alpha", "neutral beta", "signal gamma", "signal delta"], [0, 0, 1, 1]
        pipeline.fit(texts, labels)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "pipeline.joblib"
            joblib.dump(pipeline, path)
            loaded = joblib.load(path)
            np.testing.assert_array_equal(pipeline.predict(texts), loaded.predict(texts))


class SyntheticBaselineProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_temp = tempfile.TemporaryDirectory()
        cls.base = Path(cls.base_temp.name) / "base"
        cls._build_project(cls.base)

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
    def run_cli(root, script, config):
        return subprocess.run([sys.executable, script, "--config", config], cwd=root, text=True, capture_output=True, check=False)

    @classmethod
    def _build_project(cls, root):
        (root / "scripts").mkdir(parents=True)
        (root / "configs").mkdir()
        for name in ("audit_dataset.py", "review_overlap_groups.py", "prepare_partitions.py", "validate_partitions.py", "train_baselines.py", "validate_baseline_run.py"):
            shutil.copy2(ROOT / "scripts" / name, root / "scripts" / name)
        shutil.copy2(ROOT / "requirements-baselines.txt", root / "requirements-baselines.txt")
        source = root / "fixture.csv"
        rows = []
        for index in range(16):
            labels = (1, 1, 0) if index % 2 == 0 else (0, 0, -1)
            preceding = f"preceding neutral section {index}" if index % 3 else ""
            following = f"following neutral section {index}" if index % 4 else ""
            rows.append([preceding, f"common neutral target text item{index}", following, *labels])
        rows.append(["", "", "", 0, 0, 0])
        with source.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n"); writer.writerow(HEADER); writer.writerows(rows)
        audit_config = json.loads((ROOT / "configs/audit.json").read_text(encoding="utf-8"))
        audit_config["input_csv"] = "fixture.csv"
        audit_config["reference"] = {"sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "counts": {}}
        (root / "configs/audit.json").write_text(json.dumps(audit_config), encoding="utf-8")
        if cls.run_cli(root, "scripts/audit_dataset.py", "configs/audit.json").returncode != 0: raise AssertionError("synthetic audit failed")
        overlap = json.loads((ROOT / "configs/overlap_review.json").read_text(encoding="utf-8")); overlap["input_csv"] = "fixture.csv"
        (root / "configs/overlap_review.json").write_text(json.dumps(overlap), encoding="utf-8")
        if cls.run_cli(root, "scripts/review_overlap_groups.py", "configs/overlap_review.json").returncode != 0: raise AssertionError("synthetic overlap failed")
        partition = json.loads((ROOT / "configs/partition.json").read_text(encoding="utf-8")); partition["input_csv"] = "fixture.csv"
        partition["partitions"] = {p: 0.25 for p in ("train", "dev_tune", "dev_calibration", "test")}
        partition["allocation"].update({"candidate_budget": 128, "example_fraction_tolerance": 0.20, "positive_rate_tolerance": 0.50})
        partition["reference_observations"] = {"eligible_examples": 16, "excluded_examples": 1, "hard_positive_examples": 8,
            "hard_other_examples": 8, "rule_b_groups": 16, "rule_b_singletons": 16, "rule_b_largest_group": 1}
        (root / "configs/partition.json").write_text(json.dumps(partition), encoding="utf-8")
        phase2b = cls.run_cli(root, "scripts/prepare_partitions.py", "configs/partition.json")
        if phase2b.returncode != 0: raise AssertionError(phase2b.stderr + phase2b.stdout)
        psummary = json.loads((root / "results/partitions/split_summary.json").read_text(encoding="utf-8"))
        baseline = json.loads((ROOT / "configs/baselines.json").read_text(encoding="utf-8")); baseline["input_csv"] = "fixture.csv"
        baseline["required_partition_fingerprint"] = psummary["provenance_fingerprint"]
        baseline["reference_hashes"] = {"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "targets_sha256": hashlib.sha256((root / "results/partitions/targets.csv").read_bytes()).hexdigest(),
            "split_manifest_sha256": hashlib.sha256((root / "results/partitions/split_manifest.csv").read_bytes()).hexdigest()}
        (root / "configs/baselines.json").write_text(json.dumps(baseline), encoding="utf-8")
        result = cls.run_cli(root, "scripts/train_baselines.py", "configs/baselines.json")
        if result.returncode != 0: raise AssertionError(result.stderr + result.stdout)

    def test_end_to_end_validation_and_output_counts(self):
        result = self.run_cli(self.root, "scripts/validate_baseline_run.py", "configs/baselines.json")
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        validated = json.loads(result.stdout)
        self.assertTrue(validated["passed"])
        with (self.root / "results/baselines/candidate_metrics.csv").open("r", encoding="utf-8", newline="") as handle:
            candidates = list(csv.DictReader(handle))
        with (self.root / "results/baselines/dev_tune_predictions.csv").open("r", encoding="utf-8", newline="") as handle:
            predictions = list(csv.DictReader(handle))
        self.assertEqual(len(candidates), 24)
        self.assertEqual(len(predictions), 4 * 4)
        self.assertEqual({row["partition"] for row in predictions}, {"dev_tune"})

    def test_saved_output_corruption_detected(self):
        path = self.root / "results/baselines/dev_tune_predictions.csv"
        content = path.read_text(encoding="utf-8")
        path.write_text(content.replace(",dev_tune,", ",test,", 1), encoding="utf-8")
        result = self.run_cli(self.root, "scripts/validate_baseline_run.py", "configs/baselines.json")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("forbidden partition", result.stdout.lower())

    def test_completed_run_reused_without_model_change(self):
        model = self.root / "models/baselines/E1_pipeline.joblib"
        before = hashlib.sha256(model.read_bytes()).hexdigest()
        result = self.run_cli(self.root, "scripts/train_baselines.py", "configs/baselines.json")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("reused unchanged", result.stdout)
        self.assertEqual(hashlib.sha256(model.read_bytes()).hexdigest(), before)

    def test_changed_config_refuses_completed_run_and_sets_failed_status(self):
        model = self.root / "models/baselines/E1_pipeline.joblib"
        before = hashlib.sha256(model.read_bytes()).hexdigest()
        config_path = self.root / "configs/baselines.json"
        config = json.loads(config_path.read_text(encoding="utf-8")); config["tfidf"]["max_features"] = 29999
        config_path.write_text(json.dumps(config), encoding="utf-8")
        result = self.run_cli(self.root, "scripts/train_baselines.py", "configs/baselines.json")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(hashlib.sha256(model.read_bytes()).hexdigest(), before)
        status = json.loads((self.root / "results/baselines/run_status.json").read_text(encoding="utf-8"))
        self.assertEqual(status["status"], "FAILED")

    def test_source_and_frozen_partitions_preserved(self):
        protected = ["fixture.csv", "results/partitions/targets.csv", "results/partitions/split_manifest.csv",
                     "results/partitions/split_summary.json"]
        before = {name: hashlib.sha256((self.root / name).read_bytes()).hexdigest() for name in protected}
        result = self.run_cli(self.root, "scripts/validate_baseline_run.py", "configs/baselines.json")
        self.assertEqual(result.returncode, 0)
        after = {name: hashlib.sha256((self.root / name).read_bytes()).hexdigest() for name in protected}
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
