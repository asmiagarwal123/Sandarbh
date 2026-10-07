"""Synthetic tests for the SANDARBH Phase 5C policy amendment."""
import copy,importlib.util,json,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location("phase5c",ROOT/"scripts"/"amend_frozen_policy.py");phase5c=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(phase5c)

class AmendedPolicyTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.config=json.loads((ROOT/"configs"/"final_evaluation_policy_v1_1.json").read_text(encoding="utf-8"))
 def test_original_policy_remains_unchanged(self):self.assertEqual({k:phase5c.sha(phase5c.safe(v)) for k,v in self.config["original_policy"].items()},self.config["original_policy_file_hashes"])
 def test_successor_links_correct_original_hash(self):self.assertEqual(self.config["supersedes_policy_sha256"],phase5c.ORIGINAL_HASH)
 def test_baseline_definitions(self):self.assertTrue(phase5c.validate_config(self.config))
 def test_pipeline_hash_change_rejected(self):
  c=copy.deepcopy(self.config);c["baselines"][0]["pipeline_sha256"]="0"*64
  with self.assertRaises(phase5c.AmendmentError):phase5c.validate_config(c)
 def test_logistic_probability_handling(self):
  b={x["experiment_id"]:x for x in self.config["baselines"]};self.assertEqual(b["E1"]["continuous_score_interface"],"predict_proba_positive_class");self.assertEqual(b["E2"]["probability_metrics"],["negative_log_likelihood","brier_score"])
 def test_linear_svc_decision_scores(self):
  b={x["experiment_id"]:x for x in self.config["baselines"]};self.assertTrue(all(b[e]["continuous_score_interface"]=="decision_function" for e in ("E3","E4")))
 def test_svc_cannot_be_labelled_probability(self):
  c=copy.deepcopy(self.config);c["baselines"][2]["continuous_score_semantics"]="probability"
  with self.assertRaises(phase5c.AmendmentError):phase5c.validate_config(c)
 def test_dummy_metric_availability(self):self.assertEqual(self.config["dummy"]["probability_metrics"],[]);self.assertEqual(self.config["dummy"]["ranking_metrics"],[])
 def test_unavailable_values_remain_null(self):self.assertIsNone(self.config["dummy"]["probability_output"]);self.assertIsNone(self.config["dummy"]["continuous_score"])
 def test_added_comparisons(self):self.assertEqual([x["contrast"] for x in self.config["added_comparisons"]],["E5-E1","E5-E3","E6-E2","E6-E4"])
 def test_shared_bootstrap_samples_all_models(self):
  models={m:{"a":[f"{m}-a1",f"{m}-a2"],"b":[f"{m}-b"]} for m in self.config["bootstrap_extension"]["identical_cluster_samples_across_models"]};sample=phase5c.paired_cluster_samples(models,1,20261006)[0];patterns=[[x.split("-",1)[1] for x in sample[m]] for m in sorted(sample)];self.assertTrue(all(x==patterns[0] for x in patterns))
 def test_transformer_decisions_cannot_change(self):
  original={"experiments":[{"experiment_id":"E5","temperature":1.0}],"selective_prediction":{"shared_operating_threshold":.6},"comparisons":[],"uncertainty":{},"interpretation_restrictions":[]};core=phase5c.successor(self.config,{"policy":original},{"config":"a","run_manifest":"b","selected_models":"c"},{e:x[3] for e,x in phase5c.BASELINES.items()},"time");self.assertEqual(core["original_transformer_policy"],original)
 def test_thresholds_cannot_change(self):
  original=json.loads((ROOT/"results"/"frozen_policy"/"frozen_policy.json").read_text())["policy"];self.assertEqual(original["selective_prediction"]["shared_operating_threshold"],.6);self.assertEqual(original["selective_prediction"]["descriptive_thresholds"],[.5,.6,.7,.8])
 def test_temperatures_cannot_change(self):
  original=json.loads((ROOT/"results"/"frozen_policy"/"frozen_policy.json").read_text())["policy"];self.assertEqual([x["temperature"] for x in original["experiments"]],[1.0237168508133028,.6020567132971402,.6154062772305571])
 def test_missing_baseline_fails(self):
  c=copy.deepcopy(self.config);c["baselines"].pop()
  with self.assertRaises(phase5c.AmendmentError):phase5c.validate_config(c)
 def test_baseline_retraining_evidence_fails(self):
  c=copy.deepcopy(self.config);c["declarations"]["no_baseline_retraining"]=False
  with self.assertRaises(phase5c.AmendmentError):phase5c.validate_config(c)
 def test_test_access_evidence_fails(self):
  old=phase5c.ROOT
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/"results").mkdir();(root/"results"/"test_metrics.csv").write_text("x");phase5c.ROOT=root
   try:self.assertEqual(len(phase5c.prohibited()),1)
   finally:phase5c.ROOT=old
 def test_canonical_successor_hash_stable(self):self.assertEqual(phase5c.policy_hash({"b":2,"a":1}),phase5c.policy_hash({"a":1,"b":2}))
 def test_silent_overwrite_refused(self):
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaises(phase5c.AmendmentError):phase5c.ensure_output_available(Path(d))
 def test_import_no_amendment_or_inference(self):self.assertTrue(callable(phase5c.amend));self.assertEqual(phase5c.__name__,"phase5c")

if __name__=="__main__":unittest.main()
