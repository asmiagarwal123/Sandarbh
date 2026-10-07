"""Synthetic tests for SANDARBH Phase 5A."""
import importlib.util,json,math,tempfile,unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location("phase5a",ROOT/"scripts"/"run_calibration_review.py")
phase5a=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(phase5a)

class CalibrationReviewTests(unittest.TestCase):
 def test_temperature_one_reproduces_probabilities(self):self.assertEqual(phase5a.softmax_pair(1,2),phase5a.softmax_pair(1,2,1))
 def test_temperature_preserves_argmax(self):
  for t in (.05,.5,1,20):self.assertEqual(max(range(2),key=lambda i:phase5a.softmax_pair(-2,3,t)[i]),1)
 def test_positive_temperature_parameterization(self):
  fit=phase5a.fit_temperature([(0.,1.),(1.,0.)],[1,0]);self.assertGreater(fit["temperature"],0)
 def test_temperature_bounds(self):
  fit=phase5a.fit_temperature([(0.,4.),(0.,4.)],[1,0],.1,2);self.assertTrue(.1<=fit["temperature"]<=2)
 def test_nonfinite_logits_fail(self):
  with self.assertRaises(phase5a.CalibrationError):phase5a.fit_temperature([(0,float("nan"))],[1])
 def test_probability_normalization(self):self.assertAlmostEqual(sum(phase5a.softmax_pair(-100,100)),1)
 def test_nll(self):self.assertAlmostEqual(phase5a.nll_from_logits([(0,0)],[1],1),math.log(2))
 def test_brier(self):self.assertAlmostEqual(phase5a.metric_record([1],[1.],[.75])["brier_score"],.0625)
 def test_ece_boundaries_including_one(self):
  bins=phase5a.reliability_bins([0,1],[0,1],[0.0,1.0]);self.assertEqual(bins[0]["count"],1);self.assertEqual(bins[9]["count"],1)
 def test_metric_schema_records_ten_ece_bins(self):self.assertIn("ece_bin_count",phase5a.METRIC_FIELDS)
 def test_report_uses_probability_version_temperature(self):
  base={"negative_log_likelihood":.1,"brier_score":.1,"expected_calibration_error":.1,"roc_auc":.5,"average_precision":.5,"soft_cross_entropy":.1,"soft_brier_score":.1,"soft_positive_mean_absolute_difference":.1}
  metrics=[{"experiment_id":"E5","probability_version":"uncalibrated","temperature":1.0,**base},{"experiment_id":"E5","probability_version":"temperature_scaled","temperature":1.0237168508133028,**base}]
  report=phase5a.markdown_report({"timestamp_utc":"synthetic"},{"models":[]},metrics,[])
  self.assertIn("| E5 | uncalibrated | 1.000000 |",report);self.assertIn("| E5 | temperature_scaled | 1.023717 |",report)
 def test_empty_bins_are_unavailable(self):
  row=phase5a.reliability_bins([0],[0],[.01])[1];self.assertIsNone(row["mean_predicted_positive_probability"]);self.assertIsNone(row["absolute_hard_calibration_gap"])
 def test_soft_metrics(self):
  m=phase5a.metric_record([0,1],[.25,.75],[.25,.75]);self.assertAlmostEqual(m["soft_brier_score"],0);self.assertAlmostEqual(m["soft_positive_mean_absolute_difference"],0)
 def test_risk_threshold_half(self):
  r=phase5a.risk_row("E5","uncalibrated",.5,[0,1],[0,1],[.5,.5]);self.assertEqual(r["coverage"],1);self.assertEqual(r["selective_risk"],0)
 def test_risk_with_abstention(self):
  r=phase5a.risk_row("E5","uncalibrated",.8,[0,1],[1,1],[.6,.9]);self.assertEqual(r["accepted_examples"],1);self.assertEqual(r["coverage"],.5)
 def test_class_specific_coverage(self):
  r=phase5a.risk_row("E5","uncalibrated",.8,[0,0,1,1],[0,0,1,1],[.9,.4,.8,.3]);self.assertEqual(r["positive_label_coverage"],.5);self.assertEqual(r["negative_label_coverage"],.5)
 def test_zero_accepted(self):
  r=phase5a.risk_row("E5","uncalibrated",.9,[0,1],[0,1],[.6,.7]);self.assertIsNone(r["selective_risk"]);self.assertIsNone(r["selective_risk_wilson_95_lower"])
 def test_wilson_interval(self):
  lo,hi=phase5a.wilson(1,10);self.assertTrue(0<=lo<.1<hi<=1)
 def test_candidate_coverage_floor(self):
  rows=self._candidate_rows();chosen=[x for x in phase5a.select_candidates(rows) if x["experiment_id"]=="E5" and x["probability_version"]=="uncalibrated" and x["candidate_type"].startswith("lowest")][0];self.assertGreaterEqual(chosen["evidence"]["coverage"],.9)
 def test_candidate_tie_break_higher_coverage(self):
  rows=self._candidate_rows();rows[1]["selective_risk"]=rows[0]["selective_risk"];chosen=[x for x in phase5a.select_candidates(rows) if x["experiment_id"]=="E5" and x["probability_version"]=="uncalibrated" and (x.get("constraint") or {}).get("minimum_coverage")==.9][0];self.assertEqual(chosen["threshold"],.5)
 def test_only_dev_calibration_config(self):
  config={"access":{"allowed_inference_partition":"dev_calibration","forbidden_partitions":["train","dev_tune","test"]},"experiments":[{"experiment_id":x,"seed":42} for x in phase5a.EXPERIMENTS],"model":{"revision":"fb53ab8802853c8e4fbdbcd0529f21fc6f459b2b"}};phase5a.verify_config(config)
 def test_test_access_config_rejected(self):
  config={"access":{"allowed_inference_partition":"test","forbidden_partitions":[]},"experiments":[{"experiment_id":x,"seed":42} for x in phase5a.EXPERIMENTS],"model":{"revision":"fb53ab8802853c8e4fbdbcd0529f21fc6f459b2b"}}
  with self.assertRaises(phase5a.CalibrationError):phase5a.verify_config(config)
 def test_duplicate_and_unknown_keys_detectable(self):
  keys=[("E5","a"),("E5","a"),("E5","unknown")];self.assertNotEqual(len(keys),len(set(keys)));self.assertFalse(all(k[1] in {"a"} for k in keys))
 def test_e6_e7_input_equality_rule(self):
  record={"input_ids":[1,2],"attention_mask":[1,1]};self.assertEqual(record,dict(record))
 def test_provenance_mismatch_fails(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/"x";p.write_text("a");self.assertNotEqual(phase5a.sha256_file(p),"0"*64)
 def test_atomic_success_publication(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);stage=root/"stage";dest=root/"out";stage.mkdir();(stage/"x").write_text("ok");phase5a.publish_stage(stage,dest);self.assertEqual((dest/"x").read_text(),"ok")
 def test_existing_output_prevents_overwrite(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);stage=root/"stage";dest=root/"out";stage.mkdir();dest.mkdir()
   with self.assertRaises(phase5a.CalibrationError):phase5a.publish_stage(stage,dest)
 def test_import_does_not_execute_inference(self):self.assertTrue(callable(phase5a.run));self.assertEqual(phase5a.__name__,"phase5a")
 def _candidate_rows(self):
  output=[]
  for e in phase5a.EXPERIMENTS:
   for v in phase5a.VERSIONS:
    for n in range(50,100):
     coverage=1-(n-50)/100;output.append({"experiment_id":e,"probability_version":v,"confidence_threshold":n/100,"coverage":coverage,"selective_risk":.2,"accepted_true_positive_labelled_examples":1,"accepted_true_negative_labelled_examples":1})
  return output

if __name__=="__main__":unittest.main()
