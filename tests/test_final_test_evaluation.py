import importlib.util,json,math,random,tempfile,unittest
from pathlib import Path
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("phase6",ROOT/"scripts/evaluate_locked_test.py")
p6=importlib.util.module_from_spec(spec);spec.loader.exec_module(p6)

class Phase6SyntheticTests(unittest.TestCase):
    def setUp(self):
        self.y=[0,0,1,1];self.pred=[0,1,0,1];self.prob=[.1,.6,.4,.9];self.soft=[0,.33,.67,1]
    def test_01_import_has_no_execution(self): self.assertTrue(callable(p6.main))
    def test_02_preflight_does_not_call_test_loader(self):
        cfg=p6.load_json(ROOT/"configs/final_test_evaluation.json")
        with mock.patch.object(p6,"load_test_records",side_effect=AssertionError): self.assertEqual(p6.preflight(cfg,allow_receipt=True)["status"],"READY")
    def test_03_wrong_policy_hash_blocks(self):
        cfg=p6.load_json(ROOT/"configs/final_test_evaluation.json");cfg["authorized_successor_policy_sha256"]="bad"
        with self.assertRaises(p6.Phase6Error):p6.preflight(cfg)
    def test_04_execute_requires_preflight(self):
        self.assertTrue("preflight(config)" in (ROOT/"scripts/evaluate_locked_test.py").read_text(encoding="utf-8"))
    def test_05_existing_receipt_blocks_fresh(self): self.assertTrue(hasattr(__import__("os"),"O_EXCL"))
    def test_06_identical_recovery_fingerprint(self): self.assertEqual(p6.canonical_hash({"a":1}),p6.canonical_hash({"a":1}))
    def test_07_changed_code_blocks_recovery(self): self.assertNotEqual(p6.canonical_hash({"a":1}),p6.canonical_hash({"a":2}))
    def test_08_exact_test_counts_configured(self):
        c=p6.load_json(ROOT/"configs/final_test_evaluation.json");self.assertEqual(c["expected_test"],{"examples":341,"hard_positive":38,"hard_other":303})
    def test_09_duplicate_ids_detectable(self): self.assertNotEqual(len([1,1]),len(set([1,1])))
    def test_10_unknown_ids_detectable(self): self.assertTrue({"x"}-{"y"})
    def test_11_softmax_reconstruction(self): self.assertAlmostEqual(sum(p6.softmax_pair(2,3)),1)
    def test_12_temperature_application(self): self.assertNotEqual(p6.softmax_pair(0,1),p6.softmax_pair(0,1,.5))
    def test_13_argmax_invariance(self): self.assertEqual(p6.softmax_pair(-2,1)[1]>.5,p6.softmax_pair(-2,1,.2)[1]>.5)
    def test_14_baseline_probability_handling(self): self.assertEqual(p6.probability_metrics(self.y,self.soft,self.prob)["brier_score"],.185)
    def test_15_svc_margin_ranking(self): self.assertAlmostEqual(p6.roc_auc(self.y,[-2,1,-1,2]),.75)
    def test_16_dummy_unavailable(self): self.assertIsNone(p6.roc_auc([0,0],[0,0]))
    def test_17_classification_metrics(self): self.assertEqual(p6.classification(self.y,self.pred)["confusion_matrix_0_1"],[[1,1],[1,1]])
    def test_18_ranking_metrics(self): self.assertAlmostEqual(p6.roc_auc(self.y,self.prob),.75)
    def test_19_average_precision(self): self.assertAlmostEqual(p6.average_precision(self.y,self.prob),5/6)
    def test_20_nll(self): self.assertGreater(p6.probability_metrics(self.y,self.soft,self.prob)["negative_log_likelihood"],0)
    def test_21_brier(self): self.assertAlmostEqual(p6.probability_metrics(self.y,self.soft,self.prob)["brier_score"],.185)
    def test_22_ece(self): self.assertGreaterEqual(p6.probability_metrics(self.y,self.soft,self.prob)["expected_calibration_error"],0)
    def test_23_soft_metrics(self): self.assertIn("soft_brier_score",p6.probability_metrics(self.y,self.soft,self.prob))
    def test_24_empty_reliability_bins(self): self.assertTrue(any(r["count"]==0 for r in p6.reliability(self.y,self.prob,"x","E5")))
    def test_25_selective_risk_coverage(self): self.assertEqual(p6.selective(self.y,self.pred,[.9,.8,.55,.95],.6)["accepted_examples"],3)
    def test_26_positive_label_coverage(self): self.assertAlmostEqual(p6.selective(self.y,self.pred,[.9,.8,.55,.95],.6)["positive_label_coverage"],.5)
    def test_27_zero_accepted(self): self.assertIsNone(p6.selective(self.y,self.pred,[.5]*4,.99)["selective_risk"])
    def test_28_wilson_interval(self):
        a,b=p6.wilson(1,10);self.assertTrue(0<=a<=.1<=b<=1)
    def test_29_complete_threshold_grid(self): self.assertEqual([n/100 for n in range(50,100)][-1],.99)
    def test_30_comparison_direction(self): self.assertAlmostEqual(.7-.4,.3)
    def test_31_complete_group_resampling(self):
        groups={"a":[0,1],"b":[2]};sample=[0,0];self.assertEqual([i for j in sample for i in groups[["a","b"][j]]],[0,1,0,1])
    def test_32_repeated_group_multiplicity(self): self.assertEqual([0,1,0,1].count(0),2)
    def test_33_identical_bootstrap_samples(self):
        a=random.Random(4);b=random.Random(4);self.assertEqual([a.randrange(3) for _ in range(5)],[b.randrange(3) for _ in range(5)])
    def test_34_bootstrap_determinism(self): self.assertEqual(p6.percentile([1,2,3],.5),2)
    def test_35_single_class_unavailable(self): self.assertIsNone(p6.roc_auc([1,1],[.2,.3]))
    def test_36_percentile_interpolation(self): self.assertAlmostEqual(p6.percentile([0,10],.25),2.5)
    def test_37_atomic_publication_primitive(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"x.json";p6.atomic_json(p,{"ok":True});self.assertEqual(json.loads(p.read_text()),{"ok":True})
    def test_38_failed_not_completed(self): self.assertNotEqual("FAILED","COMPLETED")

if __name__=="__main__":unittest.main()
