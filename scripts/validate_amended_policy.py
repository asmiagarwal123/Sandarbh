#!/usr/bin/env python3
"""Read-only independent validation of the SANDARBH Phase 5C successor policy."""
from __future__ import annotations
import argparse,hashlib,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];ORIGINAL="9c420cd4600bb50246977390aebe399aea143b836956a691e35c18d709db8f0c"
PIPELINES={"E1":"1e87824ca409500932348c0f0d9992f60e14b1fac36b598a40318cef9df67bd3","E2":"4a97c435c77235f019c5328fe8c7fb1025807418402d85c71f35ad76cbff7c2d","E3":"13a0d8544e1d61fc58dc53222137e1d7b05d381cc93bcf172dac67fcef3521a1","E4":"076895be3321c70a379970414e1a790bddad8638485978b971d794fcf6bb1c29"}
CLASSIFICATION={"macro_f1","positive_precision","positive_recall","positive_f1","negative_precision","negative_recall","negative_f1","accuracy","balanced_accuracy","confusion_matrix_0_1","support_0_1","predicted_positive_count"}
def safe(v):p=(ROOT/v).resolve();p.relative_to(ROOT);return p
def load(p):
 with p.open(encoding="utf-8") as f:return json.load(f)
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for c in iter(lambda:f.read(1048576),b""):h.update(c)
 return h.hexdigest()
def canonical(v):return hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def prohibited():
 terms=("test_prediction","test_predictions","test_metric","test_metrics","test_evaluation","test_logits","test_probabilities","test_token");return [p for base in (ROOT/"results",ROOT/"models",ROOT/"reports") if base.exists() for p in base.rglob("*") if p.is_file() and any(t in p.name.casefold() for t in terms)]
def validate(config_path,output_override=None,compare_saved=True,quiet=False):
 checks={};failures=[]
 try:
  c=load(config_path);out=Path(output_override) if output_override else safe(c["outputs"]["directory"]);wrapped=load(out/"frozen_policy.json");p=wrapped["policy"];status=load(out/"run_status.json");manifest=load(out/"run_manifest.json")
  original={k:safe(v) for k,v in c["original_policy"].items()};base={k:safe(v) for k,v in c["baseline_evidence"].items()};oenv=load(original["frozen_policy"]);ostatus=load(original["run_status"]);oval=load(original["validation_summary"])
  checks["01_original_canonical_hash"]=oenv["canonical_policy_sha256"]==ostatus["canonical_policy_sha256"]==ORIGINAL
  checks["02_original_policy_byte_identical"]={k:sha(v) for k,v in original.items()}==c["original_policy_file_hashes"]
  checks["03_pre_amendment_test_not_accessed"]=ostatus["status"]=="COMPLETED" and ostatus["policy_status"]=="FROZEN" and ostatus["test_status"]=="NOT_ACCESSED" and oval["passed"] and oval["check_count"]==30
  checks["04_no_test_artifact"]=not prohibited()
  bm=load(base["run_manifest"]);bs=load(base["run_status"]);checks["05_phase3_status_and_validation"]=bs["status"]==bm["status"]=="COMPLETED" and bm["validation"]["status"]=="PASSED" and len(bm["validation"]["checks"])==16 and all(bm["validation"]["checks"].values())
  bspec={x["experiment_id"]:x for x in p["baselines"]};checks["06_pipeline_hashes"]=all(sha(safe(bspec[e]["pipeline"]))==PIPELINES[e]==p["baseline_provenance"]["pipeline_hashes"][e] for e in PIPELINES)
  checks["07_exact_baseline_definitions"]=set(bspec)==set(PIPELINES) and [(bspec[e]["classifier"],bspec[e]["input_mode"],bspec[e]["training_target"]) for e in PIPELINES]==[("logistic_regression","target","hard_label"),("logistic_regression","context","hard_label"),("linear_svc","target","hard_label"),("linear_svc","context","hard_label")]
  checks["08_no_baseline_retraining"]=p["declarations"]["no_baseline_retraining"] and bm["modeling_partitions_accessed"]==["train","dev_tune"] and "test vectorization or prediction" in bm["not_run"] and "baseline retraining" in manifest["operations_not_performed"]
  checks["09_dummy_definition"]=p["dummy"]["definition"]=="always predicts hard label 0" and p["dummy"]["majority_class"]==0 and p["dummy"]["probability_output"] is None and p["dummy"]["continuous_score"] is None
  checks["10_transformer_decisions_unchanged"]=p["original_transformer_policy"]==oenv["policy"]
  exp={x["experiment_id"]:x for x in p["original_transformer_policy"]["experiments"]};checks["11_transformer_temperatures_unchanged"]=all(math.isclose(exp[e]["temperature"],t,rel_tol=0,abs_tol=1e-15) for e,t in {"E5":1.0237168508133028,"E6":.6020567132971402,"E7":.6154062772305571}.items())
  sel=p["original_transformer_policy"]["selective_prediction"];checks["12_selective_thresholds_unchanged"]=sel["shared_operating_threshold"]==.60 and sel["descriptive_thresholds"]==[.50,.60,.70,.80] and sel["full_descriptive_grid"]==[round(n/100,2) for n in range(50,100)]
  u=p["uncertainty"];checks["13_bootstrap_settings_unchanged"]=u==oenv["policy"]["uncertainty"] and (u["method"],u["cluster_variable"],u["replicates"],u["random_seed"],u["confidence_level"],u["interval_type"])==("paired_cluster_bootstrap","group_id",5000,20261006,.95,"percentile")
  checks["14_transformer_comparisons_unchanged"]=p["all_comparisons"][:2]==oenv["policy"]["comparisons"] and [x["contrast"] for x in p["all_comparisons"][:2]]==["E6-E5","E7-E6"]
  checks["15_added_baseline_comparisons"]=[x["contrast"] for x in p["added_comparisons"]]==["E5-E1","E5-E3","E6-E2","E6-E4"] and p["all_comparisons"][2:]==p["added_comparisons"]
  checks["16_baseline_classification_metrics"]=set(p["baseline_classification_metrics"])==CLASSIFICATION
  checks["17_probability_metrics_restricted"]=all(bspec[e]["probability_metrics"]==["negative_log_likelihood","brier_score"] for e in ("E1","E2")) and all(bspec[e]["probability_metrics"]==[] for e in ("E3","E4"))
  checks["18_svc_scores_not_probabilities"]=all(bspec[e]["continuous_score_interface"]=="decision_function" and bspec[e]["continuous_score_semantics"]=="signed_margin_not_probability" for e in ("E3","E4"))
  checks["19_unavailable_metrics_null"]=p["dummy"]["probability_output"] is None and p["dummy"]["continuous_score"] is None and p["dummy"]["probability_metrics"]==[] and p["dummy"]["ranking_metrics"]==[]
  checks["20_same_clusters_all_models"]=p["bootstrap_extension"]["identical_cluster_samples_across_models"]==["E1","E2","E3","E4","E5","E6","E7","DUMMY"]
  checks["21_no_winner_selected"]=p["declarations"]["no_winner_selected"] is True and oenv["policy"]["decisions"]["overall_winner_selected"] is False
  checks["22_successor_schema_version"]=p["schema_version"]==manifest["schema_version"]=="5B-1.1.0"
  checks["23_superseded_policy_hash"]=p["supersedes_policy_sha256"]==manifest["supersedes_policy_sha256"]==status["supersedes_policy_sha256"]==ORIGINAL
  checks["24_amendment_reason"]=p["amendment_reason"]=="add previously frozen Phase 3 baselines before first test access"
  checks["25_successor_canonical_hash"]=canonical(p)==wrapped["canonical_policy_sha256"]==manifest["canonical_policy_sha256"]==status["canonical_policy_sha256"]
  checks["26_output_hashes"]=all((out/name).is_file() and sha(out/name)==digest for name,digest in manifest["output_hashes"].items())
  checks["27_source_partition_provenance"]=p["original_transformer_policy"]["provenance"]["source_sha256"]=="66ccaa43f9f3d8694c4826d8b999a4e562041637cad3c8c396590fbe78d3d408" and p["original_transformer_policy"]["provenance"]["partition_fingerprint"]=="61c9415766e4f7b342eb37df7b09adf2f89cd86f2818ee05645ae9409dbbba86"
  protected={**{f"original_{k}":v for k,v in original.items()},**{f"baseline_{k}":v for k,v in base.items()},**{f"pipeline_{e}":safe(bspec[e]["pipeline"]) for e in PIPELINES}};checks["28_model_checkpoint_immutability"]={k:sha(v) for k,v in protected.items()}==manifest["protected_hashes_before_and_after"]
  code={"config":sha(config_path),"amender":sha(ROOT/"scripts"/"amend_frozen_policy.py"),"validator":sha(Path(__file__)),"tests":sha(ROOT/"tests"/"test_amended_policy.py"),"protocol":sha(ROOT/"reports"/"PHASE_05C_AMENDED_FROZEN_POLICY.md")};checks["29_run_completed"]=status["status"]==manifest["status"]=="COMPLETED" and code==manifest["code_and_config_hashes"]
  checks["30_policy_frozen"]=status["policy_status"]==manifest["policy_status"]==p["policy_status"]=="FROZEN"
  checks["31_test_not_accessed"]=status["test_status"]==manifest["test_status"]==p["test_status"]=="NOT_ACCESSED" and p["declarations"]["test_not_accessed_before_amendment"] is True
  failures=[k for k,v in checks.items() if not v]
 except Exception as exc:failures.append(f"validator_exception: {type(exc).__name__}: {exc}")
 result={"passed":not failures,"check_count":len(checks),"checks":checks,"failures":failures,"validator":"scripts/validate_amended_policy.py","read_only":True}
 if compare_saved and not failures:
  try:
   saved=load(out/"validation_summary.json")
   if not (saved.get("passed")==result["passed"] and saved.get("check_count")==result["check_count"] and saved.get("checks")==result["checks"] and saved.get("failures")==result["failures"]):result["passed"]=False;result["failures"].append("saved validation summary differs")
  except Exception as exc:result["passed"]=False;result["failures"].append(f"cannot compare validation summary: {exc}")
 if not quiet:print(f"Phase 5C validation: {'PASSED' if result['passed'] else 'FAILED'} ({sum(checks.values())}/{len(checks)} checks)");[print(f"- {x}",file=sys.stderr) for x in result["failures"]]
 return result
def main(argv=None):
 q=argparse.ArgumentParser(description=__doc__);q.add_argument("--config",default="configs/final_evaluation_policy_v1_1.json");a=q.parse_args(argv);r=validate(safe(a.config));return 0 if r["passed"] else 2
if __name__=="__main__":raise SystemExit(main())
