from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("phase4b", ROOT / "scripts" / "train_transformers.py")
phase4b = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(phase4b)
HAS_TORCH = importlib.util.find_spec("torch") is not None


class FakeTokenizer:
    bos_token_id = 0
    eos_token_id = 2
    pad_token_id = 1
    is_fast = True

    MARKERS = {
        "Preceding context:\n": [101, 102, 103, 104, 105, 106],
        "Target:\n": [111, 112, 113],
        "Following context:\n": [121, 122, 123, 124],
        " …": [131, 132],
    }

    def __call__(self, text, add_special_tokens=False, truncation=False, **kwargs):
        if text in self.MARKERS:
            ids = list(self.MARKERS[text])
        else:
            ids = [200 + index for index, _ in enumerate(text.split())]
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        return {"input_ids": ids}


def synthetic_config():
    return {
        "input_construction": {
            "maximum_sequence_length": 256,
            "expected_outer_special_tokens": 2,
            "expected_target_content_budget": 241,
            "markers": {"preceding": "Preceding context:\n", "target": "Target:\n", "following": "Following context:\n"},
            "overlong_target_ellipsis": " …",
        },
        "class_weights": {"expected_train_examples": 4, "expected_class_counts_0_1": [3, 1]},
    }


def synthetic_partitions(target_words=3, preceding="p1 p2", following="f1 f2"):
    target = " ".join(f"t{i}" for i in range(target_words))
    base = {"hard_label": 0, "soft_other": 1.0, "soft_positive": 0.0, "annotations": [0, 0, 0], "preceding": preceding, "target": target, "following": following}
    return {
        "train": [{**base, "example_id": "train-1", "partition": "train"}],
        "dev_tune": [{**base, "example_id": "dev-1", "partition": "dev_tune"}],
    }


class Phase4BTokenizationTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()

    def test_exact_marker_and_special_token_accounting(self):
        encoded, rows, metadata = phase4b.construct_encodings(self.tokenizer, synthetic_partitions(), synthetic_config())
        self.assertEqual(metadata["marker_token_counts"], {"preceding": 6, "target": 3, "following": 4})
        self.assertEqual(metadata["outer_special_token_count"], 2)
        self.assertEqual(metadata["target_content_budget"], 241)
        sequence = encoded["context"]["train-1"]["input_ids"]
        self.assertEqual((sequence[0], sequence[-1]), (0, 2))

    def test_same_target_token_hash_in_target_and_context(self):
        _, rows, _ = phase4b.construct_encodings(self.tokenizer, synthetic_partitions(), synthetic_config())
        train = [row for row in rows if row["example_id"] == "train-1"]
        self.assertEqual(len({row["target_retained_sha256"] for row in train}), 1)

    def test_overlong_target_retains_beginning_ellipsis_and_end(self):
        retained, info = phase4b.truncate_target(list(range(20)), 10, [90, 91])
        self.assertEqual(retained, [0, 1, 2, 3, 90, 91, 16, 17, 18, 19])
        self.assertEqual((info["beginning"], info["ellipsis"], info["ending"]), (4, 2, 4))

    def test_odd_capacity_gives_extra_to_beginning(self):
        retained, info = phase4b.truncate_target(list(range(20)), 11, [90, 91])
        self.assertEqual((info["beginning"], info["ending"]), (5, 4))
        self.assertEqual(retained[:5], [0, 1, 2, 3, 4])

    def test_preceding_tail_and_following_head(self):
        preceding, following = phase4b.allocate_context([1, 2, 3, 4], [10, 11, 12, 13], 4)
        self.assertEqual(preceding, [3, 4])
        self.assertEqual(following, [10, 11])

    def test_unused_capacity_redistributed_deterministically(self):
        preceding, following = phase4b.allocate_context([1], list(range(10, 20)), 6)
        self.assertEqual(preceding, [1])
        self.assertEqual(following, [10, 11, 12, 13, 14])

    def test_missing_context_retains_markers_and_zero_content(self):
        encoded, rows, _ = phase4b.construct_encodings(self.tokenizer, synthetic_partitions(preceding="", following=""), synthetic_config())
        row = next(row for row in rows if row["example_id"] == "train-1" and row["input_mode"] == "context")
        self.assertEqual(row["preceding_retained_tokens"], 0)
        self.assertEqual(row["following_retained_tokens"], 0)
        self.assertGreater(len(encoded["context"]["train-1"]["input_ids"]), row["target_retained_tokens"] + 2)

    def test_e6_e7_encoded_manifest_hashes_identical(self):
        _, _, metadata = phase4b.construct_encodings(self.tokenizer, synthetic_partitions(), synthetic_config())
        self.assertEqual(metadata["e6_encoded_manifest_sha256"], metadata["e7_encoded_manifest_sha256"])

    def test_forbidden_partition_rejected_before_read(self):
        with self.assertRaisesRegex(phase4b.TrainingError, "access denied"):
            phase4b.load_partitions(["dev_calibration"], {})
        with self.assertRaisesRegex(phase4b.TrainingError, "access denied"):
            phase4b.load_partitions(["test"], {})


@unittest.skipUnless(HAS_TORCH, "PyTorch is available only in the transformer environment")
class Phase4BLossTests(unittest.TestCase):
    def test_weighted_hard_loss(self):
        import torch
        logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        logp = torch.log_softmax(logits, dim=1)
        weights = torch.tensor([0.5, 2.0])
        labels = torch.tensor([0, 1])
        observed = phase4b.weighted_hard_loss(logp, labels, weights)
        expected = -(0.5 * logp[0, 0] + 2.0 * logp[1, 1]) / 2
        self.assertTrue(torch.allclose(observed, expected))

    def test_weighted_soft_loss(self):
        import torch
        logits = torch.tensor([[1.0, -1.0]])
        logp = torch.log_softmax(logits, dim=1)
        weights = torch.tensor([0.5, 2.0])
        targets = torch.tensor([[2 / 3, 1 / 3]])
        observed = phase4b.weighted_soft_loss(logp, targets, weights)
        expected = -(2 / 3 * 0.5 * logp[0, 0] + 1 / 3 * 2.0 * logp[0, 1])
        self.assertTrue(torch.allclose(observed, expected))

    def test_one_hot_soft_equals_hard_loss(self):
        import torch
        logp = torch.log_softmax(torch.tensor([[0.2, 0.8], [1.2, -0.4]]), dim=1)
        labels = torch.tensor([1, 0])
        one_hot = torch.nn.functional.one_hot(labels, num_classes=2).float()
        weights = torch.tensor([0.6, 4.0])
        self.assertTrue(torch.allclose(phase4b.weighted_hard_loss(logp, labels, weights), phase4b.weighted_soft_loss(logp, one_hot, weights)))


class Phase4BTrainingLogicTests(unittest.TestCase):
    def test_final_partial_accumulation_group(self):
        sizes = [phase4b.accumulation_group_size(index, 10, 8) for index in range(10)]
        self.assertEqual(sizes[:8], [8] * 8)
        self.assertEqual(sizes[8:], [2, 2])

    def test_checkpoint_selection_and_earliest_tie(self):
        rows = [{"epoch": 1, "macro_f1": 0.5}, {"epoch": 2, "macro_f1": 0.7}, {"epoch": 3, "macro_f1": 0.7}]
        self.assertEqual(phase4b.select_epoch(rows)["epoch"], 2)

    def test_primary_seed_is_fixed_by_config_not_metric(self):
        selected = [{"seed": 42, "macro_f1": 0.1}, {"seed": 43, "macro_f1": 0.9}]
        primary = next(row for row in selected if row["seed"] == 42)
        self.assertEqual(primary["seed"], 42)

    def test_corrupt_unit_is_rejected(self):
        config = {"training": {"primary_seed": 42}}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "E5_seed42"
            path.mkdir()
            with self.assertRaisesRegex(phase4b.TrainingError, "Incomplete/corrupt"):
                phase4b.verify_completed_unit(path, "fingerprint", config)

    def test_metrics_and_seed_summary_reconcile(self):
        metrics = phase4b.calculate_metrics([0, 0, 1, 1], [0, 1, 0, 1], [0.1, 0.7, 0.4, 0.8], [0.0, 1 / 3, 2 / 3, 1.0])
        self.assertEqual(metrics["confusion_matrix_0_1"], [[1, 1], [1, 1]])
        records = []
        for experiment in ("E5", "E6", "E7"):
            for seed, value in zip((42, 43, 44), (0.4, 0.5, 0.6)):
                row = {"experiment_id": experiment, "seed": seed}
                row.update({name: value for name in phase4b.SUMMARY_METRICS})
                records.append(row)
        summary = phase4b.summarize_seeds(records)
        self.assertAlmostEqual(summary["experiments"]["E5"]["metrics"]["macro_f1"]["mean"], 0.5)
        self.assertAlmostEqual(summary["experiments"]["E5"]["metrics"]["macro_f1"]["sample_standard_deviation"], 0.1)

    def test_tabular_schemas_and_report_are_text_free(self):
        forbidden = {"text", "decoded_text", "tokens", "input_ids"}
        self.assertFalse(forbidden & set(phase4b.TOKEN_MANIFEST_FIELDS))
        self.assertFalse(forbidden & set(phase4b.PREDICTION_FIELDS))

    def test_tokenization_preserves_input_objects(self):
        partitions = synthetic_partitions(target_words=300)
        before = hashlib.sha256(json.dumps(partitions, sort_keys=True).encode()).hexdigest()
        phase4b.construct_encodings(FakeTokenizer(), partitions, synthetic_config())
        after = hashlib.sha256(json.dumps(partitions, sort_keys=True).encode()).hexdigest()
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
