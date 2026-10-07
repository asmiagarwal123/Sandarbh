import importlib.util,json,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("final_validator",ROOT/"scripts/validate_final_project.py");v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)

class FinalProjectTests(unittest.TestCase):
    def test_import_no_execution(self):self.assertTrue(callable(v.validate))
    def test_canonical_policy_hash(self):
        p=v.loadj(ROOT/"results/frozen_policy_v1_1/frozen_policy.json");self.assertEqual(v.ch(p["policy"]),v.AUTH)
    def test_phase6_completed(self):self.assertEqual(v.loadj(ROOT/"results/final_test/run_status.json")["test_status"],"EVALUATED_ONCE")
    def test_phase6_validation(self):self.assertEqual(v.loadj(ROOT/"results/final_test/validation_summary.json")["passed_count"],42)
    def test_summary_metric_reconciliation(self):
        s=v.loadj(ROOT/"results/final/final_summary.json");c={r["experiment_id"]:r for r in v.rows(ROOT/"results/final_test/classification_metrics.csv")};self.assertTrue(v.close(s["primary_metrics"]["E6"]["macro_f1"],c["E6"]["macro_f1"]))
    def test_required_figures(self):self.assertEqual(len(list((ROOT/"reports/figures").glob("*.png"))),6)
    def test_no_winner_language(self):self.assertNotIn("universally best",(ROOT/"reports/FINAL_RESEARCH_REPORT.md").read_text(encoding="utf-8").lower())
    def test_no_significance_language(self):self.assertNotIn("statistically significant",(ROOT/"reports/FINAL_RESEARCH_REPORT.md").read_text(encoding="utf-8").lower())
    def test_ui_threshold(self):self.assertIn('threshold":.60',(ROOT/"src/sandarbh_ui/inference.py").read_text(encoding="utf-8"))
    def test_ui_no_dataset_browser(self):self.assertNotIn("AUTALIC",(ROOT/"app.py").read_text(encoding="utf-8"))
    def test_smoke_receipt_text_free(self):self.assertFalse(v.loadj(ROOT/"results/final/ui_smoke_test.json").get("text_saved",True))
    def test_gitignore_requirements(self):
        x=(ROOT/".gitignore").read_text(encoding="utf-8");self.assertIn(".streamlit/secrets.toml",x);self.assertIn(".venv-*/",x)
    def test_handoff_map_forbidden_absent(self):
        names=set(v.load_map());self.assertNotIn("AUTALIC.csv",names);self.assertFalse(any(x.endswith((".joblib",".safetensors")) for x in names))
    def test_source_leak_scanner_clean(self):self.assertIsNone(v.source_leaks([ROOT/"reports/FINAL_RESEARCH_REPORT.md",ROOT/"app.py"]))
    def test_validation_prepackage_passes(self):
        checks,_=v.validate(False);self.assertTrue(all(checks.values()),[k for k,x in checks.items() if not x])

if __name__=="__main__":unittest.main()
