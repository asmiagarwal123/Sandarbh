#!/usr/bin/env python3
"""Freeze the SANDARBH Phase 5B final-evaluation policy without test access."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os,random,shutil,sys,tempfile
from datetime import datetime,timezone
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[1]
EXPERIMENTS={"E5":(42,3,"target",1.0237168508133028),"E6":(42,2,"context",0.6020567132971402),"E7":(42,2,"context",0.6154062772305571)}
CLASSIFICATION_METRICS={"macro_f1","positive_precision","positive_recall","positive_f1","negative_precision","negative_recall","negative_f1","accuracy","balanced_accuracy","confusion_matrix_0_1","support_0_1","predicted_positive_count","roc_auc","average_precision"}
PROBABILITY_METRICS={"negative_log_likelihood","brier_score","expected_calibration_error","reliability_bins","roc_auc","average_precision"}
SELECTIVE_METRICS={"total_examples","accepted_examples","abstained_examples","coverage","abstention_rate","accepted_correct_predictions","accepted_errors","selective_risk","accepted_accuracy","accepted_confusion_matrix_0_1","accepted_macro_f1","positive_label_coverage","negative_label_coverage","accepted_true_positive_labelled_examples","accepted_true_negative_labelled_examples","accepted_predicted_positive_count","accepted_positive_precision","overall_positive_recall_with_abstentions","selective_risk_wilson_95_interval"}
class PolicyError(Exception):pass
def utc_now():return datetime.now(timezone.utc).isoformat().replace("+00:00","Z")
def safe(value):
 p=(ROOT/value).resolve()
 try:p.relative_to(ROOT)
 except ValueError as exc:raise PolicyError("Path escapes project root") from exc
 return p
def load(path):
 try:
  with path.open(encoding="utf-8") as f:return json.load(f)
 except Exception as exc:raise PolicyError(f"Cannot read {path}: {exc}") from exc
def sha(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for c in iter(lambda:f.read(1048576),b""):h.update(c)
 return h.hexdigest()
def canonical_bytes(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf-8")
def policy_hash(value):return hashlib.sha256(canonical_bytes(value)).hexdigest()
def write_json(path,value):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open("w",encoding="utf-8",newline="\n") as f:json.dump(value,f,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False);f.write("\n")
def percentile_interval(values,confidence=.95):
 values=sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
 if not values:return None
 def q(p):
  x=(len(values)-1)*p;lo=int(math.floor(x));hi=int(math.ceil(x));return values[lo] if lo==hi else values[lo]+(x-lo)*(values[hi]-values[lo])
 alpha=(1-confidence)/2;return [q(alpha),q(1-alpha)]
def cluster_bootstrap_samples(groups,replicates=5000,seed=20261006):
 if not groups or replicates<1:raise PolicyError("Bootstrap requires groups and positive replicates")
 keys=sorted(groups);rng=random.Random(seed);result=[]
 for _ in range(replicates):
  sampled=[keys[rng.randrange(len(keys))] for _ in keys];result.append([item for key in sampled for item in groups[key]])
 return result
def paired_cluster_bootstrap(groups_by_model,replicates=5000,seed=20261006):
 if not groups_by_model:raise PolicyError("No models for paired bootstrap")
 models=sorted(groups_by_model);keys=sorted(groups_by_model[models[0]])
 if any(sorted(groups_by_model[m])!=keys for m in models):raise PolicyError("Paired models have different clusters")
 rng=random.Random(seed);out=[]
 for _ in range(replicates):
  sampled=[keys[rng.randrange(len(keys))] for _ in keys];out.append({m:[item for key in sampled for item in groups_by_model[m][key]] for m in models})
 return out
def full_grid():return [round(n/100,2) for n in range(50,100)]
def validate_decisions(config):
 observed={e["experiment_id"]:(e["seed"],e["selected_epoch"],e["input_mode"],e["temperature"]) for e in config.get("experiments",[])}
 if set(observed)!=set(EXPERIMENTS):raise PolicyError("Experiments must be exactly E5, E6, and E7")
 for key,expected in EXPERIMENTS.items():
  if observed[key][:3]!=expected[:3] or not math.isclose(observed[key][3],expected[3],rel_tol=0,abs_tol=1e-15):raise PolicyError(f"Frozen experiment mismatch: {key}")
 selective=config["selective_prediction"]
 if selective["shared_operating_threshold"]!=.60 or "model_specific_thresholds" in selective:raise PolicyError("One shared 0.60 threshold is required")
 if selective["descriptive_thresholds"]!=[.50,.60,.70,.80]:raise PolicyError("Descriptive thresholds changed")
 if config["calibration"]["method"]!="scalar_temperature_scaling" or config["classification"]["equivalent_positive_probability_threshold"]!=.50:raise PolicyError("Calibration or primary rule changed")
 if set(config["metrics"]["classification"])!=CLASSIFICATION_METRICS or set(config["metrics"]["probability"])!=PROBABILITY_METRICS or set(config["metrics"]["selective"])!=SELECTIVE_METRICS:raise PolicyError("Required metric set changed")
 if config["definitions"]["ece"]!={"bin_count":10,"binning":"equal_width_positive_probability","intervals":"[i/10,(i+1)/10), final bin includes 1.0","empty_bin_value":None}:raise PolicyError("ECE definition changed")
 if config["definitions"]["selective_risk"]!="accepted_errors / accepted_examples; unavailable when accepted_examples is zero":raise PolicyError("Selective-risk definition changed")
 u=config["uncertainty"]
 if (u["method"],u["cluster_variable"],u["replicates"],u["confidence_level"],u["interval_type"],u["random_seed"])!=("paired_cluster_bootstrap","group_id",5000,.95,"percentile",20261006):raise PolicyError("Bootstrap settings changed")
 if config["decisions"]!={"all_three_experiments_proceed":True,"overall_winner_selected":False,"test_access_during_freezing":False}:raise PolicyError("Controller decisions changed")
 return True
def test_artifacts():
 terms=("test_prediction","test_predictions","test_metric","test_metrics","test_evaluation","test_logits","test_probabilities")
 found=[]
 for base in (ROOT/"results",ROOT/"models",ROOT/"reports"):
  if base.exists():
   for p in base.rglob("*"):
    if p.is_file() and any(t in p.name.casefold() for t in terms):found.append(str(p.relative_to(ROOT)))
 return sorted(found)
def evidence(config):
 paths={k:safe(v) for k,v in config["evidence"].items()};missing=[k for k,p in paths.items() if not p.is_file()]
 if missing:raise PolicyError(f"Missing evidence: {missing}")
 hashes={k:sha(p) for k,p in paths.items()};ref=config["reference"]
 for key in ("source","split_manifest","targets","transformer_manifest","phase5a_manifest","phase5a_predictions","phase5a_parameters","phase5a_metrics","phase5a_report"):
  if hashes[key]!=ref[f"{key}_sha256"]:raise PolicyError(f"Provenance mismatch: {key}")
 ps,ts,p5=load(paths["partition_status"]),load(paths["transformer_status"]),load(paths["phase5a_status"]);v5=load(paths["phase5a_validation"]);m4=load(paths["transformer_manifest"]);m5=load(paths["phase5a_manifest"])
 if ps.get("status")!="COMPLETED" or ts.get("status")!="COMPLETED" or p5.get("status")!="COMPLETED" or p5.get("policy_status")!="PENDING_RESEARCHER_REVIEW":raise PolicyError("Upstream completion status mismatch")
 if not v5.get("passed") or v5.get("check_count")!=30 or not all(v5.get("checks",{}).values()):raise PolicyError("Phase 5A validation is not 30/30")
 if ps.get("provenance_fingerprint")!=ref["partition_fingerprint"] or m4.get("partition_fingerprint")!=ref["partition_fingerprint"] or m5.get("partition_fingerprint")!=ref["partition_fingerprint"]:raise PolicyError("Partition fingerprint mismatch")
 report=paths["phase5a_report"].read_text(encoding="utf-8")
 for exp,temp in (("E5",1.0237168508133028),("E6",.6020567132971402),("E7",.6154062772305571)):
  if f"| {exp} | uncalibrated | 1.000000 |" not in report or f"| {exp} | temperature_scaled | {temp:.6f} |" not in report:raise PolicyError("Phase 5A report temperature correction missing")
 with paths["phase5a_predictions"].open(encoding="utf-8",newline="") as f:
  prediction_rows=sum(1 for _ in csv.DictReader(f))
 if prediction_rows!=513:raise PolicyError("Phase 5A prediction count mismatch")
 primary={u["experiment_id"]:u for u in m4["unit_summaries"] if u["seed"]==42};checkpoint_hashes={}
 for spec in config["experiments"]:
  directory=safe(spec["checkpoint"]);files={p.name:sha(p) for p in directory.iterdir() if p.is_file()}
  if files!=primary[spec["experiment_id"]]["checkpoint"]["files"]:raise PolicyError("Checkpoint hash mismatch")
  checkpoint_hashes[spec["experiment_id"]]=files
 if test_artifacts():raise PolicyError(f"Prohibited test artifact found: {test_artifacts()}")
 return paths,hashes,checkpoint_hashes
def core_policy(config,hashes,checkpoint_hashes,timestamp):
 return {"schema_version":config["policy_schema_version"],"policy_status":"FROZEN","freeze_timestamp_utc":timestamp,"test_status":"NOT_ACCESSED","provenance":{"source_sha256":hashes["source"],"partition_fingerprint":config["reference"]["partition_fingerprint"],"split_manifest_sha256":hashes["split_manifest"],"targets_sha256":hashes["targets"],"phase4b_manifest_sha256":hashes["transformer_manifest"],"phase5a_manifest_sha256":hashes["phase5a_manifest"],"phase5a_prediction_sha256":hashes["phase5a_predictions"],"phase5a_calibration_parameter_sha256":hashes["phase5a_parameters"],"phase5a_metrics_sha256":hashes["phase5a_metrics"],"checkpoint_hashes":checkpoint_hashes},"experiments":config["experiments"],"calibration":config["calibration"],"classification":config["classification"],"selective_prediction":{**config["selective_prediction"],"full_descriptive_grid":full_grid()},"metrics":config["metrics"],"definitions":config["definitions"],"comparisons":config["comparisons"],"comparison_metrics":config["comparison_metrics"],"uncertainty":config["uncertainty"],"interpretation_restrictions":config["interpretation_restrictions"],"decisions":config["decisions"],"freeze_declarations":{"all_three_experiments_proceed":True,"no_winner_selected":True,"test_not_accessed_during_freezing":True,"policy_changes_require_explicit_new_version":True}}
def ensure_output_available(out):
 if out.exists():raise PolicyError("Frozen policy cannot be silently overwritten; create an explicit new version before test access")
 return True
def freeze(config_path):
 config=load(config_path);validate_decisions(config);out=safe(config["outputs"]["directory"])
 if out.exists():
  status=load(out/"run_status.json") if (out/"run_status.json").exists() else {}
  if status.get("status")=="COMPLETED":print("Existing completed frozen policy found; refusing overwrite.");return 0
  raise PolicyError("Conflicting frozen-policy output exists")
 paths,hashes,checkpoints=evidence(config);before=dict(hashes);timestamp=utc_now();core=core_policy(config,hashes,checkpoints,timestamp);digest=policy_hash(core);envelope={"canonical_policy_sha256":digest,"canonicalization":"UTF-8 JSON, sorted keys, compact separators, no NaN, hashing the policy object only","policy":core}
 out.parent.mkdir(parents=True,exist_ok=True);stage=Path(tempfile.mkdtemp(prefix="phase5b-stage-",dir=out.parent))
 try:
  write_json(stage/"frozen_policy.json",envelope);write_json(stage/"run_status.json",{"status":"COMPLETED","policy_status":"FROZEN","test_status":"NOT_ACCESSED","timestamp_utc":timestamp,"canonical_policy_sha256":digest})
  code={"config":sha(config_path),"freezer":sha(Path(__file__)),"validator":sha(ROOT/"scripts"/"validate_frozen_policy.py"),"tests":sha(ROOT/"tests"/"test_frozen_policy.py"),"protocol":sha(ROOT/"reports"/"PHASE_05B_FROZEN_POLICY.md")}
  manifest={"schema_version":"5B-1.0.0","status":"COMPLETED","policy_status":"FROZEN","test_status":"NOT_ACCESSED","timestamp_utc":timestamp,"canonical_policy_sha256":digest,"upstream_hashes_before_and_after":before,"checkpoint_hashes":checkpoints,"code_and_config_hashes":code,"output_hashes":{"frozen_policy.json":sha(stage/"frozen_policy.json"),"run_status.json":sha(stage/"run_status.json")},"operations_not_performed":["test ID loading","test label loading","test text loading","test tokenization","test inference","test prediction","test metric calculation","retraining","recalibration"]};write_json(stage/"run_manifest.json",manifest)
  if {k:sha(p) for k,p in paths.items()}!=before:raise PolicyError("Upstream artifact changed during freeze")
  sys.path.insert(0,str(ROOT/"scripts"));import validate_frozen_policy as validator
  result=validator.validate(config_path,output_override=stage,compare_saved=False,quiet=True);write_json(stage/"validation_summary.json",result)
  if not result["passed"]:raise PolicyError(f"Prepublication validation failed: {result['failures']}")
  os.replace(stage,out);print(f"Phase 5B policy frozen: {digest}");return 0
 except Exception:
  shutil.rmtree(stage,ignore_errors=True);raise
def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--config",default="configs/final_evaluation_policy.json");a=p.parse_args(argv)
 try:return freeze(safe(a.config))
 except PolicyError as exc:print(f"Phase 5B freeze failed: {exc}",file=sys.stderr);return 2
 except Exception as exc:print(f"Phase 5B freeze failed unexpectedly: {type(exc).__name__}: {exc}",file=sys.stderr);return 3
if __name__=="__main__":raise SystemExit(main())
