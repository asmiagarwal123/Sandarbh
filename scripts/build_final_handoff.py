#!/usr/bin/env python3
"""Create and verify the flattened, text-free final handoff package."""
from __future__ import annotations
import hashlib,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PKG=ROOT/"Prompt_Packages/Prompt_07_Phase_06_07_Final"
MAP={
"PHASE_06_FINAL_TEST_PROTOCOL.md":"reports/PHASE_06_FINAL_TEST_PROTOCOL.md","phase_06_final_test.md":"reports/phase_06_final_test_results.md","FINAL_RESEARCH_REPORT.md":"reports/FINAL_RESEARCH_REPORT.md","UI_GUIDE.md":"reports/UI_GUIDE.md","final_test_config.json":"configs/final_test_evaluation.json","final_test_run_status.json":"results/final_test/run_status.json","final_test_run_manifest.json":"results/final_test/run_manifest.json","test_access_receipt.json":"results/final_test/test_access_receipt.json","classification_metrics.csv":"results/final_test/classification_metrics.csv","probability_metrics.csv":"results/final_test/probability_metrics.csv","risk_coverage.csv":"results/final_test/risk_coverage.csv","paired_comparisons.csv":"results/final_test/paired_comparisons.csv","bootstrap_intervals.csv":"results/final_test/bootstrap_intervals.csv","bootstrap_manifest.json":"results/final_test/bootstrap_manifest.json","research_question_summary.json":"results/final_test/research_question_summary.json","final_summary.json":"results/final/final_summary.json","final_validation_summary.json":"results/final/final_validation_summary.json","validate_final_test.py":"scripts/validate_final_test_run.py","validate_final_project.py":"scripts/validate_final_project.py","test_final_test.py":"tests/test_final_test_evaluation.py","test_final_project.py":"tests/test_final_project.py","app.py":"app.py","requirements-ui.txt":"requirements-ui.txt",
"ui_inference.py":"src/sandarbh_ui/inference.py","ui_input_builder.py":"src/sandarbh_ui/input_builder.py","ui_result_formatter.py":"src/sandarbh_ui/result_formatter.py","test_ui.py":"tests/test_ui.py","build_final_project.py":"scripts/build_final_project.py",
"command_execution_summary.json":"results/final/command_execution_summary.json","dataset_partition_summary.csv":"results/final/tables/dataset_partition_summary.csv","baseline_results.csv":"results/final/tables/baseline_results.csv","transformer_results.csv":"results/final/tables/transformer_results.csv","calibration_results.csv":"results/final/tables/calibration_results.csv","selective_prediction_060.csv":"results/final/tables/selective_prediction_060.csv","research_question_evidence.csv":"results/final/tables/research_question_evidence.csv",
"01_macro_f1_by_experiment.png":"reports/figures/01_macro_f1_by_experiment.png","02_positive_f1_recall.png":"reports/figures/02_positive_f1_recall.png","03_calibration_before_after.png":"reports/figures/03_calibration_before_after.png","04_coverage_risk_curves.png":"reports/figures/04_coverage_risk_curves.png","05_paired_comparison_intervals.png":"reports/figures/05_paired_comparison_intervals.png","06_methodology_flow.png":"reports/figures/06_methodology_flow.png"}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    missing=[src for src in MAP.values() if not (ROOT/src).is_file()]
    if missing:raise RuntimeError(f"Missing handoff originals: {missing}")
    if PKG.exists():
        extras=[p.name for p in PKG.iterdir() if p.is_file() and p.name not in set(MAP)|{"PACKAGE_CONTENTS.md"}]
        if extras:raise RuntimeError(f"Refusing to overwrite package containing unexpected files: {extras}")
    PKG.mkdir(parents=True,exist_ok=True)
    for name,src in MAP.items():shutil.copy2(ROOT/src,PKG/name)
    lines=["# Final Handoff Package Contents","","All copied artifacts were verified byte-identical to their project originals. The package is flattened; UI module filenames map back to `src/sandarbh_ui/`. Per-example predictions, source data, model files, environments, caches, secrets, and Git internals are excluded.","","| Copied filename | Original path | SHA-256 | Verified |","|---|---|---|---|"]
    for name,src in MAP.items():
        a=sha(ROOT/src);b=sha(PKG/name)
        if a!=b:raise RuntimeError(f"Copy verification failed: {name}")
        lines.append(f"| `{name}` | `{src}` | `{b}` | yes |")
    lines += ["","`PACKAGE_CONTENTS.md` is the generated package index; its hash is reported by the final validator and completion report because a file cannot contain its own final hash.",""]
    (PKG/"PACKAGE_CONTENTS.md").write_text("\n".join(lines),encoding="utf-8",newline="\n")
    expected=set(MAP)|{"PACKAGE_CONTENTS.md"};actual={p.name for p in PKG.iterdir() if p.is_file()}
    if actual!=expected or any(p.is_dir() for p in PKG.iterdir()):raise RuntimeError("Package inventory mismatch")
    print(f"PACKAGE_FILES={len(actual)} PACKAGE_CONTENTS_SHA256={sha(PKG/'PACKAGE_CONTENTS.md')} VERIFIED=True")
if __name__=="__main__":sys.exit(main())
