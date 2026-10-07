#!/usr/bin/env python3
"""Read-only independent validation of saved SANDARBH Phase 5A evidence."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os,sys,tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT_ROOT/"scripts"))
import train_transformers as phase4b

class ValidationError(Exception): pass
def sha(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for c in iter(lambda:f.read(1048576),b""):h.update(c)
 return h.hexdigest()
def safe(value):
 p=(PROJECT_ROOT/value).resolve()
 try:p.relative_to(PROJECT_ROOT)
 except ValueError as exc:raise ValidationError("Path escapes project root") from exc
 return p
def jload(path):
 with path.open(encoding="utf-8") as f:return json.load(f)
def rows(path):
 with path.open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))
def close(a,b,tol=1e-9):
 if a is None or b is None:return a is None and b is None
 try:return math.isclose(float(a),float(b),rel_tol=tol,abs_tol=tol)
 except (ValueError,TypeError):return False
def softmax(a,b,t):
 m=max(a/t,b/t); x,y=math.exp(a/t-m),math.exp(b/t-m);return x/(x+y),y/(x+y)
def ap(labels,scores):
 positives=sum(labels); ordered=sorted(zip(scores,labels),reverse=True); tp=fp=0; prev=total=0.;i=0
 while i<len(ordered):
  score=ordered[i][0]; group=[]
  while i<len(ordered) and ordered[i][0]==score:group.append(ordered[i][1]);i+=1
  tp+=sum(group);fp+=len(group)-sum(group);rec=tp/positives;total+=(rec-prev)*tp/(tp+fp);prev=rec
 return total
def auc(labels,scores):
 pos=sum(labels);neg=len(labels)-pos; order=sorted(range(len(scores)),key=lambda i:scores[i]); ranks=[0.]*len(scores);i=0
 while i<len(order):
  e=i+1
  while e<len(order) and scores[order[e]]==scores[order[i]]:e+=1
  rank=((i+1)+e)/2
  for k in range(i,e):ranks[order[k]]=rank
  i=e
 return (sum(r for r,y in zip(ranks,labels) if y)-pos*(pos+1)/2)/(pos*neg)
def calc_metrics(labels,soft,scores):
 pred=[int(x>=.5) for x in scores];tn=sum(y==0 and p==0 for y,p in zip(labels,pred));fp=sum(y==0 and p==1 for y,p in zip(labels,pred));fn=sum(y==1 and p==0 for y,p in zip(labels,pred));tp=sum(y==1 and p==1 for y,p in zip(labels,pred));n=len(labels)
 p1=tp/(tp+fp) if tp+fp else 0.;r1=tp/(tp+fn);f1=2*p1*r1/(p1+r1) if p1+r1 else 0.;p0=tn/(tn+fn) if tn+fn else 0.;r0=tn/(tn+fp);f0=2*p0*r0/(p0+r0) if p0+r0 else 0.;clip=[min(max(x,1e-15),1-1e-15) for x in scores];ece=0.
 for b in range(10):
  ids=[i for i,x in enumerate(scores) if x>=b/10 and (x<(b+1)/10 or b==9 and x<=1)]
  if ids:ece+=len(ids)/n*abs(sum(scores[i] for i in ids)/len(ids)-sum(labels[i] for i in ids)/len(ids))
 return {"negative_log_likelihood":-sum(y*math.log(x)+(1-y)*math.log(1-x) for y,x in zip(labels,clip))/n,"brier_score":sum((x-y)**2 for x,y in zip(scores,labels))/n,"expected_calibration_error":ece,"roc_auc":auc(labels,scores),"average_precision":ap(labels,scores),"macro_f1":(f0+f1)/2,"positive_precision":p1,"positive_recall":r1,"positive_f1":f1,"accuracy":(tn+tp)/n,"balanced_accuracy":(r0+r1)/2,"confusion_matrix_0_1":[[tn,fp],[fn,tp]],"support_0_1":[tn+fp,fn+tp],"predicted_positive_count":sum(pred),"soft_cross_entropy":-sum((1-s)*math.log(1-x)+s*math.log(x) for s,x in zip(soft,clip))/n,"soft_brier_score":sum((x-s)**2 for x,s in zip(scores,soft))/n,"soft_positive_mean_absolute_difference":sum(abs(x-s) for x,s in zip(scores,soft))/n}
def wilson(errors,n,z=1.959963984540054):
 if not n:return None,None
 p=errors/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;return max(0,c-h),min(1,c+h)
def reconstruct_risk(exp,version,t,labels,preds,conf):
 ids=[i for i,c in enumerate(conf) if c+1e-15>=t];a=len(ids);err=sum(preds[i]!=labels[i] for i in ids);pos=sum(labels);neg=len(labels)-pos;tn=sum(labels[i]==0 and preds[i]==0 for i in ids);fp=sum(labels[i]==0 and preds[i]==1 for i in ids);fn=sum(labels[i]==1 and preds[i]==0 for i in ids);tp=sum(labels[i]==1 and preds[i]==1 for i in ids);lo,hi=wilson(err,a)
 return {"coverage":a/len(labels),"accepted_examples":a,"abstained_examples":len(labels)-a,"accepted_errors":err,"selective_risk":err/a if a else None,"positive_label_coverage":sum(labels[i]==1 for i in ids)/pos,"negative_label_coverage":sum(labels[i]==0 for i in ids)/neg,"accepted_true_positive_labelled_examples":sum(labels[i]==1 for i in ids),"accepted_true_negative_labelled_examples":sum(labels[i]==0 for i in ids),"accepted_predicted_positive_count":sum(preds[i] for i in ids),"overall_positive_recall_with_abstentions":tp/pos,"selective_risk_wilson_95_lower":lo,"selective_risk_wilson_95_upper":hi,"accepted_confusion_matrix_0_1":[[tn,fp],[fn,tp]]}
def validate(config_path,quiet=False):
 checks={}; failures=[]
 try:
  config=jload(config_path);out=safe(config["outputs"]["directory"]);manifest=jload(out/"run_manifest.json");status=jload(out/"run_status.json");pred=rows(out/"dev_calibration_predictions.csv");metrics=rows(out/"calibration_metrics.csv");bins=rows(out/"reliability_bins.csv");risk=rows(out/"risk_coverage.csv");params=jload(out/"calibration_parameters.json");policies=jload(out/"policy_candidates.json")
  upstream={"source":safe(config["input_csv"]),"partition_config":safe(config["partition_config"]),"transformer_config":safe(config["transformer_config"]),"partition_run_status":safe(config["partition_run_status"]),"split_manifest":safe(config["split_manifest"]),"targets":safe(config["targets"]),"transformer_run_status":safe(config["transformer_run_status"]),"transformer_run_manifest":safe(config["transformer_run_manifest"]),"transformer_selected_metrics":safe(config["transformer_selected_metrics"]),"transformer_tokenization_manifest":safe(config["transformer_tokenization_manifest"])}; hashes={k:sha(p) for k,p in upstream.items()}
  checks["01_upstream_statuses_and_hashes"]=jload(upstream["partition_run_status"])["status"]==jload(upstream["transformer_run_status"])["status"]=="COMPLETED" and hashes==manifest["upstream_hashes_before_and_after"] and all(hashes[k]==config["reference"][f"{k}_sha256"] for k in ("source","partition_config","transformer_config","split_manifest","targets","transformer_run_manifest","transformer_selected_metrics","transformer_tokenization_manifest"))
  checks["02_frozen_partition_fingerprint"]=manifest["partition_fingerprint"]==config["reference"]["partition_fingerprint"]=="61c9415766e4f7b342eb37df7b09adf2f89cd86f2818ee05645ae9409dbbba86"
  split=rows(upstream["split_manifest"]);targets=rows(upstream["targets"]);cal=[r for r in split if r["partition"]=="dev_calibration"];ids={r["example_id"] for r in cal};tmap={r["example_id"]:r for r in targets}
  checks["03_exact_calibration_ids"]=len(cal)==len(ids)==171
  checks["04_exact_hard_positives"]=sum(int(tmap[i]["hard_label"]) for i in ids)==19
  checks["05_prediction_rows"]=len(pred)==513
  keys=[(r["experiment_id"],r["example_id"]) for r in pred];checks["06_one_prediction_per_model_example"]=len(set(keys))==513 and {k[0] for k in keys}=={"E5","E6","E7"}
  checks["07_only_calibration_partition"]=all(r["partition"]=="dev_calibration" for r in pred) and manifest["partition_accessed_for_inference"]==["dev_calibration"]
  checks["08_no_unknown_or_duplicate_ids"]=all(r["example_id"] in ids for r in pred) and len(keys)==len(set(keys))
  checks["09_labels_correct"]=all(int(r["true_hard_label"])==int(tmap[r["example_id"]]["hard_label"]) and close(r["soft_positive"],tmap[r["example_id"]]["soft_positive"]) and close(r["soft_other"],tmap[r["example_id"]]["soft_other"]) for r in pred)
  probfields=["probability_0_uncalibrated","probability_1_uncalibrated","probability_0_calibrated","probability_1_calibrated","confidence_uncalibrated","confidence_calibrated"]
  checks["10_probability_finiteness_bounds"]=all(math.isfinite(float(r[k])) and 0<=float(r[k])<=1 for r in pred for k in probfields)
  checks["11_probability_pairs_normalized"]=all(close(float(r["probability_0_uncalibrated"])+float(r["probability_1_uncalibrated"]),1) and close(float(r["probability_0_calibrated"])+float(r["probability_1_calibrated"]),1) for r in pred)
  pmap={m["experiment_id"]:m for m in params["models"]};checks["12_temperature_bounds"]=all(.05<=float(m["temperature"])<=20 for m in pmap.values())
  checks["13_temperature_reconstruction"]=all(all(close(x,y,1e-8) for x,y in zip(softmax(float(r["logit_0"]),float(r["logit_1"]),float(r["temperature"])),(float(r["probability_0_calibrated"]),float(r["probability_1_calibrated"])))) for r in pred)
  checks["14_argmax_preserved"]=all(r["predicted_label_uncalibrated"]==r["predicted_label_calibrated"] for r in pred)
  metric_ok=True;soft_ok=True;by={}
  for exp in ("E5","E6","E7"):
   rs=sorted([r for r in pred if r["experiment_id"]==exp],key=lambda r:r["example_id"]); labels=[int(r["true_hard_label"]) for r in rs];soft=[float(r["soft_positive"]) for r in rs]
   for version,suffix in (("uncalibrated","uncalibrated"),("temperature_scaled","calibrated")):
    scores=[float(r[f"probability_1_{suffix}"]) for r in rs];expected=calc_metrics(labels,soft,scores);saved=next(r for r in metrics if r["experiment_id"]==exp and r["probability_version"]==version);by[(exp,version)]=(rs,labels,soft,scores)
    metric_ok &= int(saved["ece_bin_count"])==10
    for k,v in expected.items():
     observed=json.loads(saved[k]) if k in ("confusion_matrix_0_1","support_0_1") else saved[k]
     if isinstance(v,list):metric_ok &= observed==v
     elif k.startswith("soft_"):soft_ok &= close(observed,v)
     else:metric_ok &= close(observed,v)
  checks["15_classification_metrics_recalculated"]=metric_ok
  checks["16_nll_brier_ece_recalculated"]=metric_ok
  checks["17_soft_metrics_recalculated"]=soft_ok
  bin_ok=len(bins)==60
  for exp,version in by:
   rs,labels,soft,scores=by[(exp,version)]
   for b in range(10):
    saved=next(x for x in bins if x["experiment_id"]==exp and x["probability_version"]==version and int(x["bin_index"])==b);members=[i for i,x in enumerate(scores) if x>=b/10 and (x<(b+1)/10 or b==9 and x<=1)];bin_ok &= int(saved["count"])==len(members)
    for field,values in (("mean_predicted_positive_probability",scores),("observed_hard_positive_fraction",labels),("mean_soft_positive_fraction",soft)):
     expected=sum(values[i] for i in members)/len(members) if members else None; observed=None if saved[field]=="" else float(saved[field]);bin_ok &= close(observed,expected)
  checks["18_all_reliability_bins"]=bin_ok
  checks["19_complete_risk_grid"]=len(risk)==300 and all(len([r for r in risk if r["experiment_id"]==e and r["probability_version"]==v])==50 for e in ("E5","E6","E7") for v in ("uncalibrated","temperature_scaled"))
  risk_ok=wilson_ok=True
  for exp,version in by:
   rs,labels,soft,scores=by[(exp,version)];preds=[int(r[f"predicted_label_{'uncalibrated' if version=='uncalibrated' else 'calibrated'}"]) for r in rs];conf=[float(r[f"confidence_{'uncalibrated' if version=='uncalibrated' else 'calibrated'}"]) for r in rs]
   for saved in [x for x in risk if x["experiment_id"]==exp and x["probability_version"]==version]:
    ex=reconstruct_risk(exp,version,float(saved["confidence_threshold"]),labels,preds,conf)
    for k,v in ex.items():
     observed=json.loads(saved[k]) if k=="accepted_confusion_matrix_0_1" else (None if saved[k]=="" else saved[k]);good=observed==v if isinstance(v,list) else close(observed,v); risk_ok &= good
     if k.startswith("selective_risk_wilson"):wilson_ok &= good
  checks["20_risk_coverage_class_coverage"]=risk_ok;checks["21_wilson_intervals"]=wilson_ok
  candidate_ok=policies["policy_status"]=="PENDING_RESEARCHER_REVIEW" and policies["approved_policy"] is None and len(policies["candidates"])==36
  for c in policies["candidates"]:
   if c["candidate_type"]=="lowest_risk_at_coverage_floor":
    subset=[r for r in risk if r["experiment_id"]==c["experiment_id"] and r["probability_version"]==c["probability_version"] and float(r["coverage"])+1e-15>=c["constraint"]["minimum_coverage"] and int(r["accepted_true_positive_labelled_examples"])>0 and int(r["accepted_true_negative_labelled_examples"])>0];expected=min(subset,key=lambda r:(float(r["selective_risk"]),-float(r["coverage"]),float(r["confidence_threshold"]))) if subset else None;candidate_ok &= close(c["threshold"],None if expected is None else expected["confidence_threshold"])
  checks["22_candidate_selection_deterministic"]=candidate_ok
  training=jload(upstream["transformer_config"]);training_paths={"source":upstream["source"],"partition_split_manifest":upstream["split_manifest"],"partition_targets":upstream["targets"]};old=phase4b.ALLOWED_PARTITIONS;phase4b.ALLOWED_PARTITIONS=frozenset({"dev_calibration"})
  try:parts=phase4b.load_partitions(["dev_calibration"],training_paths)
  finally:phase4b.ALLOWED_PARTITIONS=old
  from transformers import AutoTokenizer
  tok=AutoTokenizer.from_pretrained(safe(config["experiments"][0]["checkpoint"]),local_files_only=True,trust_remote_code=False,use_fast=True);encoded,trows,meta=phase4b.construct_encodings(tok,parts,training)
  checks["23_e6_e7_identical_inputs"]=manifest["e6_e7_input_ids_and_masks_identical"] is True and meta["e6_encoded_manifest_sha256"]==meta["e7_encoded_manifest_sha256"]
  observed_max=max(int(r["sequence_length"]) for r in trows);checks["24_sequence_length_limit"]=observed_max<=256 and observed_max==manifest["maximum_observed_sequence_length"]
  checks["25_no_test_artifact_or_access"]=not any("test" in p.name.lower() and "prediction" in p.name.lower() for p in out.iterdir()) and manifest["forbidden_partitions_accessed_for_inference"]==[]
  def output_path(name):return safe(config["outputs"]["report"]) if name=="phase_05a_calibration_review.md" else out/name
  checks["26_output_hashes"]=all(output_path(name).is_file() and sha(output_path(name))==digest for name,digest in manifest["output_hashes"].items())
  phase4=jload(upstream["transformer_run_manifest"]);primary={u["experiment_id"]:u for u in phase4["unit_summaries"] if u["seed"]==42};checks["27_checkpoint_hashes_immutable"]=all({p.name:sha(p) for p in safe(next(x["checkpoint"] for x in config["experiments"] if x["experiment_id"]==e)).iterdir() if p.is_file()}==primary[e]["checkpoint"]["files"]==manifest["checkpoint_hashes"][e] for e in ("E5","E6","E7"))
  forbidden={"text","target","preceding","following","decoded_text","tokens","input_ids"};headers=set(pred[0])|set(metrics[0])|set(bins[0])|set(risk[0]);checks["28_no_raw_text_leakage"]=not (headers&forbidden)
  checks["29_run_and_policy_status"]=manifest["status"]==status["status"]=="COMPLETED" and manifest["policy_status"]==status["policy_status"]=="PENDING_RESEARCHER_REVIEW"
  code={"config":sha(config_path),"runner":sha(PROJECT_ROOT/"scripts"/"run_calibration_review.py"),"validator":sha(Path(__file__)),"tests":sha(PROJECT_ROOT/"tests"/"test_calibration_review.py"),"protocol":sha(PROJECT_ROOT/"reports"/"PHASE_05_CALIBRATION_PROTOCOL.md")};checks["30_code_and_config_hashes"]=code==manifest["code_and_config_hashes"]
  failures=[name for name,passed in checks.items() if not passed]
 except Exception as exc:failures.append(f"validator_exception: {type(exc).__name__}: {exc}")
 result={"passed":not failures,"check_count":len(checks),"checks":checks,"failures":failures,"validator":"scripts/validate_calibration_review.py"}
 if 'config' in locals():
  destination=safe(config["outputs"]["validation_summary"]);destination.parent.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(prefix=".validation-",suffix=".tmp",dir=destination.parent);os.close(fd)
  try:
   with open(tmp,"w",encoding="utf-8",newline="\n") as f:json.dump(result,f,indent=2,sort_keys=True);f.write("\n")
   os.replace(tmp,destination)
  finally:
   if os.path.exists(tmp):os.unlink(tmp)
 if not quiet:print(f"Phase 5A validation: {'PASSED' if result['passed'] else 'FAILED'} ({sum(checks.values())}/{len(checks)} checks)");[print(f"- {x}",file=sys.stderr) for x in failures]
 return result
def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--config",default="configs/calibration_review.json");a=p.parse_args(argv);result=validate(safe(a.config));return 0 if result["passed"] else 2
if __name__=="__main__":raise SystemExit(main())
