from __future__ import annotations

import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("phase4a", ROOT / "scripts" / "transformer_preflight.py")
phase4a = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(phase4a)


class FakeTokenizer:
    is_fast = True
    vocab_size = 1000
    special_tokens_map = {"bos_token": "<s>", "eos_token": "</s>"}

    @staticmethod
    def _tokens(text):
        tokens = []
        cursor = 0
        for word in text.split():
            start = text.find(word, cursor)
            end = start + len(word)
            tokens.append((len(tokens) + 10, (start, end)))
            cursor = end
        return tokens

    def __call__(self, text, add_special_tokens=True, truncation=False, max_length=None, return_offsets_mapping=False, **kwargs):
        tokens = self._tokens(text)
        ids = [token[0] for token in tokens]
        offsets = [token[1] for token in tokens]
        if add_special_tokens:
            ids = [0] + ids + [2]
            offsets = [(0, 0)] + offsets + [(0, 0)]
        if truncation and max_length is not None:
            ids = ids[:max_length]
            offsets = offsets[:max_length]
        value = {"input_ids": ids}
        if return_offsets_mapping:
            value["offset_mapping"] = offsets
        return value

    @staticmethod
    def build_inputs_with_special_tokens(ids):
        return [0] + list(ids) + [2]


class Phase4AUnitTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()

    def test_partition_allowlist_rejects_calibration_and_test_before_read(self):
        with self.assertRaisesRegex(phase4a.PreflightError, "access denied"):
            phase4a.load_tokenization_partitions(["dev_calibration"], {})
        with self.assertRaisesRegex(phase4a.PreflightError, "access denied"):
            phase4a.load_tokenization_partitions(["test"], {})

    def test_token_and_special_token_accounting(self):
        row = {"example_id": "x", "partition": "train", "preceding": "one two", "target": "three four", "following": "five"}
        lengths, _ = phase4a.diagnose_example(self.tokenizer, row, [256, 512])
        self.assertEqual(lengths["target"], (2, 4))
        self.assertEqual(lengths["context"], (5, 7))

    def test_partial_and_complete_target_truncation_detected(self):
        row = {"example_id": "x", "partition": "train", "preceding": "p1 p2 p3 p4 p5", "target": "t1 t2 t3", "following": "f1"}
        _, affected = phase4a.diagnose_example(self.tokenizer, row, [4, 8])
        by_limit = {item["max_length"]: item for item in affected}
        self.assertEqual(by_limit[4]["target_retention"], "complete_removal")
        self.assertEqual(by_limit[8]["target_retention"], "partial_removal")
        self.assertEqual(by_limit[8]["retained_target_token_count"], 2)

    def test_empty_context_is_deterministic(self):
        row = {"example_id": "x", "partition": "dev_tune", "preceding": "", "target": "target only", "following": ""}
        context, span = phase4a.context_with_target_span(row)
        self.assertEqual(context, "target only")
        self.assertEqual(span, (0, len(context)))

    def test_overlong_target_policy_has_explicit_cap(self):
        config = {"proposed_input_policy": {"sequence_length": 12, "markers": {"preceding": "pre", "target": "target", "following": "post"}, "overlong_target": "prefix", "context_allocation": "near"}}
        diagnostics = {"combined": {"ordinary_right_truncation_target_effect": {"12": {"partial_removal": 1, "complete_removal": 0}}}}
        policy = phase4a.proposed_policy_record(self.tokenizer, config, diagnostics)
        self.assertEqual(policy["maximum_retained_target_tokens"], 7)
        self.assertEqual(policy["overlong_target"], "prefix")

    def test_diagnostics_are_deterministic(self):
        rows = {"train": [{"example_id": "x", "partition": "train", "preceding": "a", "target": "b c", "following": "d"}], "dev_tune": [{"example_id": "y", "partition": "dev_tune", "preceding": "", "target": "e", "following": ""}]}
        first = phase4a.token_diagnostics(self.tokenizer, rows, [4, 8])
        second = phase4a.token_diagnostics(self.tokenizer, rows, [4, 8])
        self.assertEqual(first, second)

    def test_text_free_output_schema_detection(self):
        self.assertTrue(phase4a.text_free_schema({"example_id": "x", "metrics": [1, 2]}))
        self.assertFalse(phase4a.text_free_schema({"text": "forbidden"}))
        self.assertFalse(phase4a.text_free_schema({"nested": {"input_ids": [1]}}))

    def test_nonfinite_output_detection(self):
        class Result:
            def all(self): return self
            def item(self): return False
        fake_torch = SimpleNamespace(isfinite=lambda value: Result())
        with self.assertRaisesRegex(phase4a.PreflightError, "nonfinite"):
            phase4a.require_finite_tensor(fake_torch, object(), "logits")

    def test_timeout_is_reported_honestly(self):
        config = {"benchmark": {"microbatch_sizes": [1, 2], "overall_timeout_seconds": 300, "per_configuration_timeout_seconds": 10, "sequence_length": 256, "warmup_steps": 1, "timed_steps": 3, "seed": 42, "precision": "float32", "training_examples": 1595, "epochs_for_estimate": 3}}
        with mock.patch.object(phase4a.subprocess, "run", side_effect=subprocess.TimeoutExpired(["python"], 10)):
            result = phase4a.run_benchmarks(Path("snapshot"), config, {"actual_device": "cpu", "memory": {"available_bytes": 100}})
        self.assertEqual(result["configurations"][0]["status"], "TIMED_OUT")
        self.assertEqual(len(result["configurations"]), 1)

    def test_oom_is_reported_honestly(self):
        config = {"benchmark": {"microbatch_sizes": [1], "overall_timeout_seconds": 300, "per_configuration_timeout_seconds": 10, "sequence_length": 256, "warmup_steps": 1, "timed_steps": 3, "seed": 42, "precision": "float32", "training_examples": 1595, "epochs_for_estimate": 3}}
        completed = subprocess.CompletedProcess(["python"], 1, stdout="", stderr="RuntimeError: out of memory")
        with mock.patch.object(phase4a.subprocess, "run", return_value=completed):
            result = phase4a.run_benchmarks(Path("snapshot"), config, {"actual_device": "cpu", "memory": {"available_bytes": 100}})
        self.assertEqual(result["configurations"][0]["status"], "OOM")

    def test_empty_target_is_rejected(self):
        with self.assertRaisesRegex(phase4a.PreflightError, "empty target"):
            phase4a.context_with_target_span({"preceding": "context", "target": "", "following": ""})


if __name__ == "__main__":
    unittest.main()
