"""Synthetic tests for the SANDARBH Phase 5B policy freezer."""
import copy,importlib.util,json,tempfile,unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location("phase5b",ROOT/"scripts"/"freeze_evaluation_policy.py")
phase5b=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(phase5b)

class FrozenPolicyTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.config=json.loads((ROOT/"configs"/"final_evaluation_policy.json").read_text(encoding="utf-8"))
 def test_canonical_serialization_and_stable_hash(self):
  a={"b":2,"a":1};b={"a":1,"b":2};self.assertEqual(phase5b.canonical_bytes(a),phase5b.canonical_bytes(b));self.assertEqual(phase5b.policy_hash(a),phase5b.policy_hash(b))
 def test_changed_policy_changes_hash(self):self.assertNotEqual(phase5b.policy_hash({"x":1}),phase5b.policy_hash({"x":2}))
 def test_exact_temperatures(self):
  phase5b.validate_decisions(self.config);self.assertEqual({x["experiment_id"]:x["temperature"] for x in self.config["experiments"]},{k:v[3] for k,v in phase5b.EXPERIMENTS.items()})
 def test_shared_threshold_enforced(self):
  c=copy.deepcopy(self.config);c["selective_prediction"]["shared_operating_threshold"]=.61
  with self.assertRaises(phase5b.PolicyError):phase5b.validate_decisions(c)
 def test_model_specific_thresholds_rejected(self):
  c=copy.deepcopy(self.config);c["selective_prediction"]["model_specific_thresholds"]={"E5":.6}
  with self.assertRaises(phase5b.PolicyError):phase5b.validate_decisions(c)
 def test_missing_experiment_fails(self):
  c=copy.deepcopy(self.config);c["experiments"].pop()
  with self.assertRaises(phase5b.PolicyError):phase5b.validate_decisions(c)
 def test_wrong_seed_fails(self):
  c=copy.deepcopy(self.config);c["experiments"][0]["seed"]=43
  with self.assertRaises(phase5b.PolicyError):phase5b.validate_decisions(c)
 def test_wrong_epoch_fails(self):
  c=copy.deepcopy(self.config);c["experiments"][1]["selected_epoch"]=3
  with self.assertRaises(phase5b.PolicyError):phase5b.validate_decisions(c)
 def test_unknown_metric_fails(self):
  c=copy.deepcopy(self.config);c["metrics"]["classification"].append("mystery")
  with self.assertRaises(phase5b.PolicyError):phase5b.validate_decisions(c)
 def test_required_metric_omission_fails(self):
  c=copy.deepcopy(self.config);c["metrics"]["selective"].remove("positive_label_coverage")
  with self.assertRaises(phase5b.PolicyError):phase5b.validate_decisions(c)
 def test_ece_definition_validation(self):
  c=copy.deepcopy(self.config);c["definitions"]["ece"]["bin_count"]=9
  with self.assertRaises(phase5b.PolicyError):phase5b.validate_decisions(c)
 def test_selective_risk_definition_validation(self):
  c=copy.deepcopy(self.config);c["definitions"]["selective_risk"]="wrong"
  with self.assertRaises(phase5b.PolicyError):phase5b.validate_decisions(c)
 def test_cluster_bootstrap_preserves_complete_groups(self):
  groups={"a":["a1","a2"],"b":["b1"]};sample=phase5b.cluster_bootstrap_samples(groups,1,7)[0];self.assertEqual(sample.count("a1"),sample.count("a2"))
 def test_paired_bootstrap_identical_clusters(self):
  models={"E5":{"a":["E5-a"],"b":["E5-b"]},"E6":{"a":["E6-a"],"b":["E6-b"]}};sample=phase5b.paired_cluster_bootstrap(models,1,3)[0];self.assertEqual([x.split("-")[1] for x in sample["E5"]],[x.split("-")[1] for x in sample["E6"]])
 def test_bootstrap_deterministic(self):
  groups={"a":[1,2],"b":[3]};self.assertEqual(phase5b.cluster_bootstrap_samples(groups,5,20261006),phase5b.cluster_bootstrap_samples(groups,5,20261006))
 def test_single_class_unavailable_not_zero(self):self.assertIsNone(phase5b.percentile_interval([]))
 def test_percentile_interval(self):
  lo,hi=phase5b.percentile_interval([0,1,2,3,4],.80);self.assertAlmostEqual(lo,.4);self.assertAlmostEqual(hi,3.6)
 def test_policy_cannot_be_silently_overwritten(self):
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaises(phase5b.PolicyError):phase5b.ensure_output_available(Path(d))
 def test_changed_provenance_fails_hash_comparison(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/"evidence";p.write_text("changed",encoding="utf-8");self.assertNotEqual(phase5b.sha(p),"0"*64)
 def test_test_access_evidence_fails(self):
  old=phase5b.ROOT
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/"results").mkdir();(root/"results"/"test_predictions.csv").write_text("x",encoding="utf-8");phase5b.ROOT=root
   try:self.assertEqual([x.replace("\\","/") for x in phase5b.test_artifacts()],["results/test_predictions.csv"])
   finally:phase5b.ROOT=old
 def test_import_does_not_freeze_or_infer(self):self.assertTrue(callable(phase5b.freeze));self.assertEqual(phase5b.__name__,"phase5b")

if __name__=="__main__":unittest.main()
