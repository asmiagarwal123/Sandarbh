#!/usr/bin/env python3
"""Independent final-project validator; does not load models or rerun inference."""
from __future__ import annotations
import argparse,csv,hashlib,importlib.util,json,re,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];AUTH="b5765e9431a4bf01e8abfeeec1762061b1c0a26eec4ea26c326abb7245ac7541"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def loadj(p):
    with p.open(encoding="utf-8") as f:return json.load(f)
def rows(p):
    with p.open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def ch(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def close(a,b):return a is None and b is None or a is not None and b is not None and abs(float(a)-float(b))<2e-9
def load_map():
    p=ROOT/"scripts/build_final_handoff.py";s=importlib.util.spec_from_file_location("handoff_map",p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m.MAP
def source_leaks(paths):
    with (ROOT/"AUTALIC.csv").open(encoding="utf-8-sig",newline="") as f:
        source={x.strip() for r in csv.DictReader(f) for k in ("preceding","target","following") if (x:=r[k]).strip() and len(x.strip())>=20}
    for p in paths:
        if p.suffix.lower() not in {".md",".py",".json",".csv",".txt"}:continue
        text=p.read_text(encoding="utf-8",errors="strict")
        for x in source:
            if x in text:return str(p.relative_to(ROOT)),hashlib.sha256(x.encode()).hexdigest()
    return None
def validate(require_package=True):
    C={};detail={}
    def ck(name,value,note=""):C[name]=bool(value);detail[name]=note
    p6=ROOT/"results/final_test";final=ROOT/"results/final";summary=loadj(final/"final_summary.json");status=loadj(p6/"run_status.json");v6=loadj(p6/"validation_summary.json");receipt=loadj(p6/"test_access_receipt.json")
    ck("01_phase6_completed",status.get("status")=="COMPLETED" and status.get("test_status")=="EVALUATED_ONCE")
    ck("02_phase6_validation",v6.get("passed") and v6.get("passed_count")==42 and not v6.get("failures"))
    ck("03_receipt_consistency",receipt.get("status")=="TEST_ACCESSED" and receipt.get("successor_policy_sha256")==AUTH and status.get("receipt_sha256")==sha(p6/"test_access_receipt.json"))
    succ=loadj(ROOT/"results/frozen_policy_v1_1/frozen_policy.json");ck("04_policy_canonical_hash",succ.get("canonical_policy_sha256")==AUTH and ch(succ["policy"])==AUTH)
    ck("05_final_report_exists",(ROOT/"reports/FINAL_RESEARCH_REPORT.md").stat().st_size>5000)
    ck("06_final_summary_completed",summary.get("status")=="COMPLETED" and summary.get("policy_sha256")==AUTH)
    class_rows=rows(p6/"classification_metrics.csv");cm={r["experiment_id"]:r for r in class_rows};primary=summary["primary_metrics"]
    ck("07_summary_primary_metrics",set(primary)==set(cm) and all(close(primary[m][k],cm[m][k] or None) for m in cm for k in primary[m]))
    comparisons=rows(p6/"paired_comparisons.csv");ck("08_summary_comparisons",summary["frozen_comparisons"]==comparisons)
    boots=rows(p6/"bootstrap_intervals.csv");ck("09_summary_bootstrap",summary["bootstrap_intervals"]==boots)
    risk=rows(p6/"risk_coverage.csv");sel={r["experiment_id"]:r for r in risk if float(r["threshold"])==.6};ck("10_summary_selective",summary["selective_operating_point"]==sel)
    tables=ROOT/"results/final/tables";required_tables={"dataset_partition_summary.csv","baseline_results.csv","transformer_results.csv","calibration_results.csv","selective_prediction_060.csv","frozen_paired_comparisons.csv","bootstrap_intervals.csv","research_question_evidence.csv"};ck("11_publication_tables",required_tables=={p.name for p in tables.glob("*.csv")} and all(p.stat().st_size>20 for p in tables.glob("*.csv")))
    ck("12_table_reconciliation",sha(tables/"frozen_paired_comparisons.csv")==sha(p6/"paired_comparisons.csv") and sha(tables/"bootstrap_intervals.csv")==sha(p6/"bootstrap_intervals.csv"))
    figs=ROOT/"reports/figures";pngs=sorted(figs.glob("*.png"));svgs=sorted(figs.glob("*.svg"));ck("13_figures_exist",len(pngs)==6 and len(svgs)==6 and all(p.stat().st_size>5000 for p in pngs+svgs))
    ck("14_figure_hashes",all((ROOT/x["path"]).is_file() and sha(ROOT/x["path"])==x["sha256"] for x in summary["figures"]))
    report=(ROOT/"reports/FINAL_RESEARCH_REPORT.md").read_text(encoding="utf-8");rq=summary["research_questions"];ck("15_rq_conclusions_numeric",all(rq[x]=="mixed evidence on this locked test" for x in ("RQ1","RQ2","RQ3")) and "-0.2632" in report and "0.2632" in report)
    low=report.lower();ck("16_no_unsupported_winner","universally best" not in low and "overall winner" not in low and "unconditional winner" in low)
    ck("17_no_significance_claim","statistically significant" not in low and "significant improvement" not in low)
    ck("18_no_pvalues",not re.search(r"\bp\s*[<=>]\s*0?\.\d+",low) and all("p_value" not in r and "p-value" not in r for r in comparisons))
    cfg=loadj(ROOT/"configs/final_test_evaluation.json");ck("19_no_policy_changes",cfg["authorized_successor_policy_sha256"]==AUTH and cfg["selective_thresholds"]=={"start":.5,"end":.99,"step":.01,"frozen_primary":.6})
    ui=[ROOT/"app.py",ROOT/"src/sandarbh_ui/inference.py",ROOT/"src/sandarbh_ui/input_builder.py",ROOT/"src/sandarbh_ui/result_formatter.py",ROOT/"reports/UI_GUIDE.md",ROOT/"requirements-ui.txt"];ck("20_ui_files",all(p.is_file() and p.stat().st_size>20 for p in ui))
    infer=ui[1].read_text(encoding="utf-8");app=ui[0].read_text(encoding="utf-8");ck("21_ui_frozen_models",all(x in infer for x in ("E5_seed42","E6_seed42","E7_seed42","1.0237168508133028","0.6020567132971402","0.6154062772305571")))
    ck("22_ui_thresholds",'threshold":.60' in infer and "p1>=.5" in infer)
    ck("23_ui_no_dataset_exposure","AUTALIC" not in app+infer and "transformer_test_predictions" not in app+infer and "baseline_test_predictions" not in app+infer)
    smoke=loadj(final/"ui_smoke_test.json");server_smoke=loadj(final/"ui_server_smoke_test.json");ck("24_ui_smoke",smoke.get("status")=="PASSED" and close(smoke.get("probability_sum"),1) and smoke.get("text_saved") is False and server_smoke.get("status")=="PASSED" and server_smoke.get("process_stopped_cleanly") is True)
    docs=[ROOT/"README.md",ROOT/"reports/IMPLEMENTATION_STATUS.md",ROOT/"reports/RESEARCH_PROTOCOL.md"];ck("25_documentation_updated",all("Phase 7" in p.read_text(encoding="utf-8") for p in docs) and "streamlit run app.py" in docs[0].read_text(encoding="utf-8"))
    ck("26_source_unchanged",sha(ROOT/"AUTALIC.csv")==cfg["source_sha256"])
    ck("27_partitions_unchanged",sha(ROOT/cfg["split_manifest"])==cfg["split_manifest_sha256"] and sha(ROOT/cfg["targets"])==cfg["targets_sha256"])
    immutable=True
    for m,s in cfg["baseline_models"].items():immutable &= sha(ROOT/s["path"])==s["sha256"]
    for m,s in cfg["transformers"].items():immutable &= sha(ROOT/s["path"]/"model.safetensors")==s["model_sha256"]
    ck("28_models_unchanged",immutable)
    policy=succ["policy"]["original_transformer_policy"]["provenance"];ck("29_calibration_unchanged",sha(ROOT/"results/calibration_review/calibration_parameters.json")==policy["phase5a_calibration_parameter_sha256"])
    original=loadj(ROOT/"results/frozen_policy/frozen_policy.json");ck("30_frozen_policies_unchanged",original.get("canonical_policy_sha256")==cfg["superseded_policy_sha256"] and succ.get("canonical_policy_sha256")==AUTH)
    model_files={p.name for p in (ROOT/"models/baselines").glob("*.joblib")};model_dirs={p.name for p in (ROOT/"models/transformers").iterdir() if p.is_dir()};ck("31_no_unexpected_models",model_files=={f"E{i}_pipeline.joblib" for i in range(1,5)} and model_dirs=={"E5_seed42","E6_seed42","E7_seed42"})
    final_paths=[ROOT/"reports/FINAL_RESEARCH_REPORT.md",ROOT/"reports/UI_GUIDE.md",ROOT/"app.py",*list((ROOT/"src/sandarbh_ui").glob("*.py")),*list(tables.glob("*.csv"))];leak=source_leaks(final_paths);ck("32_source_text_leakage_scan",leak is None,"" if leak is None else str(leak))
    serialized="\n".join(p.read_text(encoding="utf-8",errors="ignore") for p in [final/"final_summary.json",*tables.glob("*.csv")]);ck("33_no_nan_serialization",not re.search(r"\b(?:NaN|Infinity|-Infinity)\b",serialized))
    ignore=(ROOT/".gitignore").read_text(encoding="utf-8");ck("34_gitignore_hygiene",all(x in ignore for x in ("AUTALIC.csv",".venv/",".venv-*/","__pycache__/","*.py[cod]","results/","models/","checkpoints/",".cache/",".pytest_cache/",".streamlit/secrets.toml")))
    diff=subprocess.run(["git","diff","--check"],cwd=ROOT,capture_output=True,text=True);ck("35_git_diff_check",diff.returncode==0,diff.stderr.strip())
    secret_text="\n".join(p.read_text(encoding="utf-8",errors="ignore") for p in final_paths);ck("36_no_secrets",not re.search(r"(?:sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|password\s*=\s*['\"][^'\"]+)",secret_text,re.I))
    if require_package:
        pkg=ROOT/"Prompt_Packages/Prompt_07_Phase_06_07_Final";mapping=load_map();actual={p.name for p in pkg.iterdir() if p.is_file()} if pkg.is_dir() else set();expected=set(mapping)|{"PACKAGE_CONTENTS.md"};ck("37_handoff_inventory",actual==expected and not any(p.is_dir() for p in pkg.iterdir()) if pkg.is_dir() else False)
        matches=pkg.is_dir() and all((pkg/n).is_file() and sha(pkg/n)==sha(ROOT/s) for n,s in mapping.items() if n!="final_validation_summary.json")
        if pkg.is_dir() and (pkg/"final_validation_summary.json").is_file() and (final/"final_validation_summary.json").is_file():matches &= sha(pkg/"final_validation_summary.json")==sha(final/"final_validation_summary.json")
        ck("38_handoff_hashes",matches)
        forbidden={".joblib",".safetensors",".bin",".pt"};ck("39_handoff_exclusions",pkg.is_dir() and not any(p.suffix.lower() in forbidden or p.name=="AUTALIC.csv" for p in pkg.iterdir()))
    return C,detail
def main():
    p=argparse.ArgumentParser();p.add_argument("--prepare-package",action="store_true");a=p.parse_args();checks,detail=validate(not a.prepare_package);fail=[k for k,v in checks.items() if not v];summary={"validator":"scripts/validate_final_project.py","read_only_except_summary":True,"preliminary":a.prepare_package,"check_count":len(checks),"passed_count":sum(checks.values()),"checks":checks,"details":{k:v for k,v in detail.items() if v},"failures":fail,"passed":not fail}
    out=ROOT/"results/final/final_validation_summary.json";tmp=out.with_name(out.name+".tmp");tmp.write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8");tmp.replace(out);print(json.dumps(summary,indent=2));return 0 if not fail else 1
if __name__=="__main__":sys.exit(main())
