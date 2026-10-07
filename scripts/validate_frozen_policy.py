#!/usr/bin/env python3
"""Read-only independent validation of the SANDARBH Phase 5B frozen policy."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
TEMPS={"E5":1.0237168508133028,"E6":.6020567132971402,"E7":.6154062772305571}
CLASSIFICATION={"macro_f1","positive_precision","positive_recall","positive_f1","negative_precision","negative_recall","negative_f1","accuracy","balanced_accuracy","confusion_matrix_0_1","support_0_1","predicted_positive_count","roc_auc","average_precision"}
PROBABILITY={"negative_log_likelihood","brier_score","expected_calibration_error","reliability_bins","roc_auc","average_precision"}
SELECTIVE={"total_examples","accepted_examples","abstained_examples","coverage","abstention_rate","accepted_correct_predictions","accepted_errors","selective_risk","accepted_accuracy","accepted_confusion_matrix_0_1","accepted_macro_f1","positive_label_coverage","negative_label_coverage","accepted_true_positive_labelled_examples","accepted_true_negative_labelled_examples","accepted_predicted_positive_count","accepted_positive_precision","overall_positive_recall_with_abstentions","selective_risk_wilson_95_interval"}
def safe(value):
 p=(ROOT/value).resolve();p.relative_to(ROOT);return p
def load(path):
 with path.open(encoding="utf-8") as f:return json.load(f)
def sha(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for c in iter(lambda:f.read(1048576),b""):h.update(c)
 return h.hexdigest()
def canonical_hash(value):return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf-8")).hexdigest()
def prohibited_artifacts():
 terms=("test_prediction","test_predictions","test_metric","test_metrics","test_evaluation","test_logits","test_probabilities");found=[]
 for base in (ROOT/"results",ROOT/"models",ROOT/"reports"):
  if base.exists():
   for p in base.rglob("*"):
    if p.is_file() and any(t in p.name.casefold() for t in terms):found.append(str(p.relative_to(ROOT)))
 return found
def validate(config_path,output_override=None,compare_saved=True,quiet=False):
 checks={};failures=[]
 try:
  config=load(config_path);out=Path(output_override) if output_override else safe(config["outputs"]["directory"]);envelope=load(out/"frozen_policy.json");policy=envelope["policy"];manifest=load(out/"run_manifest.json");status=load(out/"run_status.json")
  ev={k:safe(v) for k,v in config["evidence"].items()};ref=config["reference"]
  pstatus,tstatus,a5status=load(ev["partition_status"]),load(ev["transformer_status"]),load(ev["phase5a_status"]);v5=load(ev["phase5a_validation"])
  checks["01_upstream_completion_statuses"]=pstatus.get("status")==tstatus.get("status")==a5status.get("status")=="COMPLETED" and a5status.get("policy_status")=="PENDING_RESEARCHER_REVIEW"
  checks["02_phase5a_validation_30_of_30"]=v5.get("passed") is True and v5.get("check_count")==30 and len(v5.get("checks",{}))==30 and all(v5["checks"].values())
  report=ev["phase5a_report"].read_text(encoding="utf-8");checks["03_phase5a_report_correction"]=all(f"| {e} | uncalibrated | 1.000000 |" in report and f"| {e} | temperature_scaled | {t:.6f} |" in report for e,t in TEMPS.items())
  hashes={k:sha(p) for k,p in ev.items()};reference_keys=("source","split_manifest","targets","transformer_manifest","phase5a_manifest","phase5a_predictions","phase5a_parameters","phase5a_metrics","phase5a_report")
  checks["04_frozen_source_partition_provenance"]=all(hashes[k]==ref[f"{k}_sha256"] for k in reference_keys) and pstatus.get("provenance_fingerprint")==ref["partition_fingerprint"]==policy["provenance"]["partition_fingerprint"]
  m4=load(ev["transformer_manifest"]);primary={u["experiment_id"]:u for u in m4["unit_summaries"] if u["seed"]==42};checkpoint_ok=True
  for spec in config["experiments"]:
   files={p.name:sha(p) for p in safe(spec["checkpoint"]).iterdir() if p.is_file()};checkpoint_ok &= files==primary[spec["experiment_id"]]["checkpoint"]["files"]==policy["provenance"]["checkpoint_hashes"][spec["experiment_id"]]
  checks["05_checkpoint_hashes"]=checkpoint_ok
  experiments={e["experiment_id"]:e for e in policy["experiments"]};checks["06_experiments_seeds_epochs_inputs"]=set(experiments)==set(TEMPS) and [(experiments[e]["seed"],experiments[e]["selected_epoch"],experiments[e]["input_mode"]) for e in TEMPS]==[(42,3,"target"),(42,2,"context"),(42,2,"context")]
  checks["07_exact_temperatures"]=all(math.isclose(experiments[e]["temperature"],t,rel_tol=0,abs_tol=1e-15) for e,t in TEMPS.items())
  checks["08_temperature_method"]=policy["calibration"]==config["calibration"] and policy["calibration"]["method"]=="scalar_temperature_scaling"
  checks["09_primary_argmax_rule"]=policy["classification"]["primary_rule"]=="argmax_calibrated_probabilities" and policy["classification"]["coverage"]==1.0 and policy["classification"]["abstention"]=="none"
  checks["10_primary_threshold"]=policy["classification"]["equivalent_positive_probability_threshold"]==.50
  checks["11_shared_selective_threshold"]=policy["selective_prediction"]["shared_operating_threshold"]==.60 and policy["selective_prediction"]["accept_rule"]=="confidence >= 0.60"
  checks["12_descriptive_thresholds"]=policy["selective_prediction"]["descriptive_thresholds"]==[.50,.60,.70,.80]
  checks["13_full_descriptive_grid"]=policy["selective_prediction"]["full_descriptive_grid"]==[round(n/100,2) for n in range(50,100)]
  checks["14_classification_metrics"]=set(policy["metrics"]["classification"])==CLASSIFICATION and policy["metrics"]["primary"]=="macro_f1_labels_0_1"
  checks["15_probability_metrics"]=set(policy["metrics"]["probability"])==PROBABILITY and set(policy["metrics"]["soft_target_descriptive"])=={"soft_cross_entropy","soft_brier_score","soft_positive_mean_absolute_difference"}
  checks["16_selective_metrics"]=set(policy["metrics"]["selective"])==SELECTIVE
  checks["17_paired_comparisons"]=policy["comparisons"]==config["comparisons"] and [x["contrast"] for x in policy["comparisons"]]==["E6-E5","E7-E6"] and len(policy["comparison_metrics"])==12
  u=policy["uncertainty"];checks["18_group_aware_bootstrap"]=u["method"]=="paired_cluster_bootstrap" and u["cluster_variable"]=="group_id" and u["interval_type"]=="percentile" and u["confidence_level"]==.95 and not u["resample_until_valid"] and not u["unavailable_values_are_zero"]
  checks["19_bootstrap_seed"]=u["random_seed"]==20261006
  checks["20_bootstrap_replicates"]=u["replicates"]==5000
  checks["21_interpretation_restrictions"]=policy["interpretation_restrictions"]==config["interpretation_restrictions"] and len(policy["interpretation_restrictions"])==14
  checks["22_all_experiments_proceed"]=policy["decisions"]["all_three_experiments_proceed"] is True and policy["freeze_declarations"]["all_three_experiments_proceed"] is True
  checks["23_no_winner_selected"]=policy["decisions"]["overall_winner_selected"] is False and policy["freeze_declarations"]["no_winner_selected"] is True
  checks["24_no_test_prediction_or_evaluation_artifact"]=not prohibited_artifacts()
  checks["25_no_test_tokenization_or_inference"]=policy["test_status"]=="NOT_ACCESSED" and policy["decisions"]["test_access_during_freezing"] is False and all(x in manifest["operations_not_performed"] for x in ("test ID loading","test label loading","test text loading","test tokenization","test inference","test prediction","test metric calculation"))
  checks["26_upstream_and_checkpoints_unchanged"]={k:sha(p) for k,p in ev.items()}==manifest["upstream_hashes_before_and_after"] and checkpoint_ok
  checks["27_policy_hash_correct"]=canonical_hash(policy)==envelope["canonical_policy_sha256"]==manifest["canonical_policy_sha256"]==status["canonical_policy_sha256"]
  checks["28_output_hashes_correct"]=all((out/name).is_file() and sha(out/name)==digest for name,digest in manifest["output_hashes"].items())
  code={"config":sha(config_path),"freezer":sha(ROOT/"scripts"/"freeze_evaluation_policy.py"),"validator":sha(Path(__file__)),"tests":sha(ROOT/"tests"/"test_frozen_policy.py"),"protocol":sha(ROOT/"reports"/"PHASE_05B_FROZEN_POLICY.md")};checks["29_run_status_completed"]=status["status"]==manifest["status"]=="COMPLETED" and code==manifest["code_and_config_hashes"]
  checks["30_policy_frozen"]=status["policy_status"]==manifest["policy_status"]==policy["policy_status"]=="FROZEN" and status["test_status"]==manifest["test_status"]==policy["test_status"]=="NOT_ACCESSED"
  failures=[k for k,v in checks.items() if not v]
 except Exception as exc:failures.append(f"validator_exception: {type(exc).__name__}: {exc}")
 result={"passed":not failures,"check_count":len(checks),"checks":checks,"failures":failures,"validator":"scripts/validate_frozen_policy.py","read_only":True}
 if compare_saved and not failures:
  try:
   saved=load(out/"validation_summary.json");checks_match=saved.get("passed")==result["passed"] and saved.get("check_count")==result["check_count"] and saved.get("checks")==result["checks"] and saved.get("failures")==result["failures"]
   if not checks_match:result["passed"]=False;result["failures"].append("saved validation summary differs from recomputation")
  except Exception as exc:result["passed"]=False;result["failures"].append(f"cannot compare saved validation summary: {exc}")
 if not quiet:print(f"Phase 5B validation: {'PASSED' if result['passed'] else 'FAILED'} ({sum(checks.values())}/{len(checks)} checks)");[print(f"- {x}",file=sys.stderr) for x in result["failures"]]
 return result
def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--config",default="configs/final_evaluation_policy.json");a=p.parse_args(argv);r=validate(safe(a.config));return 0 if r["passed"] else 2
if __name__=="__main__":raise SystemExit(main())
