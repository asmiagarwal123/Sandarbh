#!/usr/bin/env python3
"""Create the SANDARBH Phase 5C pre-test successor policy without inference."""
from __future__ import annotations
import argparse,hashlib,json,math,os,random,shutil,sys,tempfile
from datetime import datetime,timezone
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[1]
ORIGINAL_HASH="9c420cd4600bb50246977390aebe399aea143b836956a691e35c18d709db8f0c"
BASELINES={"E1":("logistic_regression","target",10.0,"1e87824ca409500932348c0f0d9992f60e14b1fac36b598a40318cef9df67bd3"),"E2":("logistic_regression","context",.1,"4a97c435c77235f019c5328fe8c7fb1025807418402d85c71f35ad76cbff7c2d"),"E3":("linear_svc","target",.1,"13a0d8544e1d61fc58dc53222137e1d7b05d381cc93bcf172dac67fcef3521a1"),"E4":("linear_svc","context",.1,"076895be3321c70a379970414e1a790bddad8638485978b971d794fcf6bb1c29")}
class AmendmentError(Exception):pass
def utc():return datetime.now(timezone.utc).isoformat().replace("+00:00","Z")
def safe(v):
 p=(ROOT/v).resolve()
 try:p.relative_to(ROOT)
 except ValueError as exc:raise AmendmentError("Path escapes project root") from exc
 return p
def load(p):
 try:
  with p.open(encoding="utf-8") as f:return json.load(f)
 except Exception as exc:raise AmendmentError(f"Cannot read {p}: {exc}") from exc
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for c in iter(lambda:f.read(1048576),b""):h.update(c)
 return h.hexdigest()
def canonical_bytes(v):return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf-8")
def policy_hash(v):return hashlib.sha256(canonical_bytes(v)).hexdigest()
def paired_cluster_samples(groups_by_model,replicates=1,seed=20261006):
 models=sorted(groups_by_model);keys=sorted(groups_by_model[models[0]])
 if any(sorted(groups_by_model[m])!=keys for m in models):raise AmendmentError("Models have different bootstrap clusters")
 rng=random.Random(seed);out=[]
 for _ in range(replicates):
  sampled=[keys[rng.randrange(len(keys))] for _ in keys];out.append({m:[v for key in sampled for v in groups_by_model[m][key]] for m in models})
 return out
def ensure_output_available(path):
 if path.exists():raise AmendmentError("Successor policy cannot be silently overwritten")
 return True
def write(p,v):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open("w",encoding="utf-8",newline="\n") as f:json.dump(v,f,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False);f.write("\n")
def prohibited():
 terms=("test_prediction","test_predictions","test_metric","test_metrics","test_evaluation","test_logits","test_probabilities","test_token");found=[]
 for base in (ROOT/"results",ROOT/"models",ROOT/"reports"):
  if base.exists():
   for p in base.rglob("*"):
    if p.is_file() and any(t in p.name.casefold() for t in terms):found.append(str(p.relative_to(ROOT)))
 return sorted(found)
def validate_config(c):
 if c["schema_version"]!="5B-1.1.0" or c["supersedes_policy_sha256"]!=ORIGINAL_HASH or c["amendment_reason"]!="add previously frozen Phase 3 baselines before first test access":raise AmendmentError("Versioning or amendment reason changed")
 observed={b["experiment_id"]:b for b in c["baselines"]}
 if set(observed)!=set(BASELINES):raise AmendmentError("Baselines must be exactly E1-E4")
 for e,(classifier,mode,C,digest) in BASELINES.items():
  b=observed[e]
  if (b["classifier"],b["input_mode"],float(b["C"]),b["pipeline_sha256"])!=(classifier,mode,C,digest):raise AmendmentError(f"Baseline definition mismatch: {e}")
  if classifier=="logistic_regression" and (b["continuous_score_interface"]!="predict_proba_positive_class" or b["probability_metrics"]!=["negative_log_likelihood","brier_score"]):raise AmendmentError("Logistic probability handling changed")
  if classifier=="linear_svc" and (b["continuous_score_interface"]!="decision_function" or b["continuous_score_semantics"]!="signed_margin_not_probability" or b["probability_metrics"]):raise AmendmentError("SVC score handling changed")
 if c["dummy"]!={"experiment_id":"DUMMY","definition":"always predicts hard label 0","majority_class":0,"training_source":"train majority class","competitive_model":False,"probability_output":None,"continuous_score":None,"probability_metrics":[],"ranking_metrics":[]}:raise AmendmentError("Dummy definition changed")
 if [x["contrast"] for x in c["added_comparisons"]]!=["E5-E1","E5-E3","E6-E2","E6-E4"]:raise AmendmentError("Added comparison definitions changed")
 if c["bootstrap_extension"]["identical_cluster_samples_across_models"]!=["E1","E2","E3","E4","E5","E6","E7","DUMMY"]:raise AmendmentError("Shared cluster sampling changed")
 if not all(c["declarations"].values()):raise AmendmentError("Required declaration is false")
 return True
def inspect(c):
 original={k:safe(v) for k,v in c["original_policy"].items()};base={k:safe(v) for k,v in c["baseline_evidence"].items()};original_hashes={k:sha(p) for k,p in original.items()};base_hashes={k:sha(p) for k,p in base.items()}
 if original_hashes!=c["original_policy_file_hashes"]:raise AmendmentError("Original Phase 5B policy files changed")
 envelope=load(original["frozen_policy"]);status=load(original["run_status"]);validation=load(original["validation_summary"])
 if envelope["canonical_policy_sha256"]!=ORIGINAL_HASH or status.get("canonical_policy_sha256")!=ORIGINAL_HASH:raise AmendmentError("Original canonical policy hash mismatch")
 if status.get("status")!="COMPLETED" or status.get("policy_status")!="FROZEN" or status.get("test_status")!="NOT_ACCESSED" or not validation.get("passed") or validation.get("check_count")!=30:raise AmendmentError("Original policy precondition failed")
 if any(base_hashes[k]!=c["baseline_reference_hashes"][k] for k in ("config","run_manifest","selected_models")):raise AmendmentError("Phase 3 reference hash mismatch")
 manifest=load(base["run_manifest"]);bstatus=load(base["run_status"]);selected=load(base["selected_models"])
 if bstatus.get("status")!="COMPLETED" or manifest.get("status")!="COMPLETED" or manifest.get("validation",{}).get("status")!="PASSED" or len(manifest.get("validation",{}).get("checks",{}))!=16 or not all(manifest["validation"]["checks"].values()):raise AmendmentError("Phase 3 is not completed and independently valid (16 embedded checks plus completed run-status check)")
 pipeline_hashes={}
 for spec in c["baselines"]:
  e=spec["experiment_id"];path=safe(spec["pipeline"]);digest=sha(path);pipeline_hashes[e]=digest
  if digest!=spec["pipeline_sha256"] or digest!=selected["models"][e]["sha256"] or digest!=manifest["model_artifacts"][e]["sha256"]:raise AmendmentError(f"Pipeline mismatch: {e}")
 if manifest.get("modeling_partitions_accessed")!=["train","dev_tune"] or "test vectorization or prediction" not in manifest.get("not_run",[]):raise AmendmentError("Baseline access/retraining evidence conflict")
 if prohibited():raise AmendmentError(f"Prohibited test artifact exists: {prohibited()}")
 return original,base,original_hashes,base_hashes,envelope,pipeline_hashes
def successor(c,envelope,base_hashes,pipeline_hashes,timestamp):
 return {"schema_version":"5B-1.1.0","policy_status":"FROZEN","test_status":"NOT_ACCESSED","freeze_timestamp_utc":timestamp,"supersedes_policy_sha256":ORIGINAL_HASH,"amendment_reason":c["amendment_reason"],"original_transformer_policy":envelope["policy"],"baseline_provenance":{"phase3_config_sha256":base_hashes["config"],"phase3_run_manifest_sha256":base_hashes["run_manifest"],"phase3_selected_models_sha256":base_hashes["selected_models"],"pipeline_hashes":pipeline_hashes},"baselines":c["baselines"],"dummy":c["dummy"],"baseline_classification_metrics":c["baseline_classification_metrics"],"added_comparisons":c["added_comparisons"],"baseline_comparison_metrics":c["baseline_comparison_metrics"],"all_comparisons":envelope["policy"]["comparisons"]+c["added_comparisons"],"uncertainty":envelope["policy"]["uncertainty"],"bootstrap_extension":c["bootstrap_extension"],"interpretation_restrictions":envelope["policy"]["interpretation_restrictions"],"declarations":c["declarations"]}
def amend(config_path):
 c=load(config_path);validate_config(c);out=safe(c["outputs"]["directory"])
 if out.exists():
  status=load(out/"run_status.json") if (out/"run_status.json").exists() else {}
  if status.get("status")=="COMPLETED":print("Existing completed amended policy found; refusing overwrite.");return 0
  raise AmendmentError("Conflicting amended-policy output exists")
 original,base,oh,bh,envelope,pipelines=inspect(c);all_paths={**{f"original_{k}":v for k,v in original.items()},**{f"baseline_{k}":v for k,v in base.items()},**{f"pipeline_{b['experiment_id']}":safe(b["pipeline"]) for b in c["baselines"]}};before={k:sha(v) for k,v in all_paths.items()};timestamp=utc();core=successor(c,envelope,bh,pipelines,timestamp);digest=policy_hash(core);wrapped={"canonical_policy_sha256":digest,"canonicalization":"UTF-8 JSON, sorted keys, compact separators, no NaN, hashing the policy object only","policy":core}
 out.parent.mkdir(parents=True,exist_ok=True);stage=Path(tempfile.mkdtemp(prefix="phase5c-stage-",dir=out.parent))
 try:
  write(stage/"frozen_policy.json",wrapped);write(stage/"run_status.json",{"status":"COMPLETED","policy_status":"FROZEN","test_status":"NOT_ACCESSED","timestamp_utc":timestamp,"canonical_policy_sha256":digest,"supersedes_policy_sha256":ORIGINAL_HASH})
  code={"config":sha(config_path),"amender":sha(Path(__file__)),"validator":sha(ROOT/"scripts"/"validate_amended_policy.py"),"tests":sha(ROOT/"tests"/"test_amended_policy.py"),"protocol":sha(ROOT/"reports"/"PHASE_05C_AMENDED_FROZEN_POLICY.md")};manifest={"schema_version":"5B-1.1.0","status":"COMPLETED","policy_status":"FROZEN","test_status":"NOT_ACCESSED","timestamp_utc":timestamp,"canonical_policy_sha256":digest,"supersedes_policy_sha256":ORIGINAL_HASH,"protected_hashes_before_and_after":before,"code_and_config_hashes":code,"output_hashes":{"frozen_policy.json":sha(stage/"frozen_policy.json"),"run_status.json":sha(stage/"run_status.json")},"operations_not_performed":["test ID loading","test label loading","test text loading","test vectorization","test tokenization","test inference","test prediction","test metric calculation","baseline retraining","transformer retraining","recalibration"]};write(stage/"run_manifest.json",manifest)
  if {k:sha(v) for k,v in all_paths.items()}!=before:raise AmendmentError("Protected artifact changed during amendment")
  sys.path.insert(0,str(ROOT/"scripts"));import validate_amended_policy as validator
  result=validator.validate(config_path,output_override=stage,compare_saved=False,quiet=True);write(stage/"validation_summary.json",result)
  if not result["passed"]:raise AmendmentError(f"Prepublication validation failed: {result['failures']}")
  os.replace(stage,out);print(f"Phase 5C successor policy frozen: {digest}");return 0
 except Exception:
  shutil.rmtree(stage,ignore_errors=True);raise
def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--config",default="configs/final_evaluation_policy_v1_1.json");a=p.parse_args(argv)
 try:return amend(safe(a.config))
 except AmendmentError as exc:print(f"Phase 5C amendment failed: {exc}",file=sys.stderr);return 2
 except Exception as exc:print(f"Phase 5C amendment failed unexpectedly: {type(exc).__name__}: {exc}",file=sys.stderr);return 3
if __name__=="__main__":raise SystemExit(main())
