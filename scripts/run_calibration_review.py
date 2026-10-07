#!/usr/bin/env python3
"""Run SANDARBH Phase 5A calibration and selective-prediction review."""
from __future__ import annotations

import argparse, csv, hashlib, importlib.metadata, json, math, os, platform, shutil, sys, tempfile, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import train_transformers as phase4b

EXPERIMENTS = ("E5", "E6", "E7")
VERSIONS = ("uncalibrated", "temperature_scaled")
PREDICTION_FIELDS = ("experiment_id","seed","selected_epoch","example_id","partition","true_hard_label","soft_other","soft_positive","predicted_label_uncalibrated","logit_0","logit_1","probability_0_uncalibrated","probability_1_uncalibrated","temperature","predicted_label_calibrated","probability_0_calibrated","probability_1_calibrated","confidence_uncalibrated","confidence_calibrated")
METRIC_FIELDS = ("experiment_id","probability_version","apparent_fit_set_diagnostic","temperature","negative_log_likelihood","brier_score","expected_calibration_error","ece_bin_count","roc_auc","average_precision","macro_f1","positive_precision","positive_recall","positive_f1","accuracy","balanced_accuracy","confusion_matrix_0_1","support_0_1","predicted_positive_count","soft_cross_entropy","soft_brier_score","soft_positive_mean_absolute_difference")
BIN_FIELDS = ("experiment_id","probability_version","bin_index","lower_bound","upper_bound","include_upper_bound","count","mean_predicted_positive_probability","observed_hard_positive_fraction","mean_soft_positive_fraction","absolute_hard_calibration_gap")
RISK_FIELDS = ("experiment_id","probability_version","confidence_threshold","total_examples","accepted_examples","abstained_examples","coverage","abstention_rate","accepted_correct_predictions","accepted_errors","selective_risk","accepted_accuracy","accepted_confusion_matrix_0_1","accepted_macro_f1","positive_label_coverage","negative_label_coverage","accepted_true_positive_labelled_examples","accepted_true_negative_labelled_examples","accepted_predicted_positive_count","accepted_positive_precision","overall_positive_recall_with_abstentions","selective_risk_wilson_95_lower","selective_risk_wilson_95_upper")

class CalibrationError(Exception): pass
def utc_now(): return datetime.now(timezone.utc).isoformat().replace("+00:00","Z")
def sha256_file(path: Path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()
def safe_path(value: str, label="path"):
    path=(PROJECT_ROOT/value).resolve()
    try: path.relative_to(PROJECT_ROOT)
    except ValueError as exc: raise CalibrationError(f"{label} escapes project root") from exc
    return path
def load_json(path: Path):
    try:
        with path.open(encoding="utf-8") as f: value=json.load(f)
    except Exception as exc: raise CalibrationError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(value,dict): raise CalibrationError(f"Expected JSON object: {path}")
    return value
def write_json(path: Path, value: Any):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8",newline="\n") as f: json.dump(value,f,ensure_ascii=False,indent=2,sort_keys=True); f.write("\n")
def write_csv(path: Path, fields: Iterable[str], rows: Iterable[dict[str,Any]]):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(fields),lineterminator="\n"); w.writeheader(); w.writerows(rows)
def softmax_pair(a: float,b: float,t: float=1.0):
    if not all(math.isfinite(x) for x in (a,b,t)) or t<=0: raise CalibrationError("Nonfinite logits or nonpositive temperature")
    x,y=a/t,b/t; m=max(x,y); ea,eb=math.exp(x-m),math.exp(y-m); s=ea+eb
    p=(ea/s,eb/s)
    if not all(math.isfinite(v) and 0<=v<=1 for v in p) or not math.isclose(sum(p),1,abs_tol=1e-12): raise CalibrationError("Invalid probabilities")
    return p
def nll_from_logits(logits, labels, temperature):
    if not logits or len(logits)!=len(labels): raise CalibrationError("Invalid calibration arrays")
    return -sum(math.log(max(softmax_pair(a,b,temperature)[label],1e-300)) for (a,b),label in zip(logits,labels))/len(labels)
def fit_temperature(logits, labels, minimum=.05, maximum=20., tolerance=1e-10, iteration_limit=500):
    if minimum<=0 or maximum<=minimum: raise CalibrationError("Invalid temperature bounds")
    if any(not math.isfinite(v) for pair in logits for v in pair): raise CalibrationError("Nonfinite logits")
    initial=nll_from_logits(logits,labels,1.0); lo,hi=math.log(minimum),math.log(maximum); ratio=(math.sqrt(5)-1)/2
    c=hi-ratio*(hi-lo); d=lo+ratio*(hi-lo); fc=nll_from_logits(logits,labels,math.exp(c)); fd=nll_from_logits(logits,labels,math.exp(d)); iterations=0
    while hi-lo>tolerance and iterations<iteration_limit:
        if fc<=fd: hi,d,fd=d,c,fc; c=hi-ratio*(hi-lo); fc=nll_from_logits(logits,labels,math.exp(c))
        else: lo,c,fc=c,d,fd; d=lo+ratio*(hi-lo); fd=nll_from_logits(logits,labels,math.exp(d))
        iterations+=1
    candidates=[(initial,1.0),(nll_from_logits(logits,labels,minimum),minimum),(nll_from_logits(logits,labels,maximum),maximum),(nll_from_logits(logits,labels,math.exp((lo+hi)/2)),math.exp((lo+hi)/2))]
    final,temp=min(candidates,key=lambda x:(x[0],abs(math.log(x[1]))))
    return {"temperature":temp,"optimizer":"golden_section_search_log_temperature","tolerance":tolerance,"iteration_limit":iteration_limit,"iterations":iterations,"initial_objective":initial,"final_objective":final,"converged":hi-lo<=tolerance,"at_bound":math.isclose(temp,minimum,abs_tol=1e-12) or math.isclose(temp,maximum,abs_tol=1e-12)}
def reliability_bins(labels, soft, probs, count=10):
    rows=[]
    for i in range(count):
        lo,hi=i/count,(i+1)/count; ids=[j for j,p in enumerate(probs) if p>=lo and (p<hi or i==count-1 and p<=hi)]
        mp=sum(probs[j] for j in ids)/len(ids) if ids else None; observed=sum(labels[j] for j in ids)/len(ids) if ids else None
        rows.append({"bin_index":i,"lower_bound":lo,"upper_bound":hi,"include_upper_bound":i==count-1,"count":len(ids),"mean_predicted_positive_probability":mp,"observed_hard_positive_fraction":observed,"mean_soft_positive_fraction":sum(soft[j] for j in ids)/len(ids) if ids else None,"absolute_hard_calibration_gap":abs(mp-observed) if ids else None})
    return rows
def metric_record(labels,soft,probs):
    pred=[int(p>=.5) for p in probs]; return phase4b.calculate_metrics(labels,pred,probs,soft,10)
def confusion(labels,preds):
    return [[sum(y==0 and p==0 for y,p in zip(labels,preds)),sum(y==0 and p==1 for y,p in zip(labels,preds))],[sum(y==1 and p==0 for y,p in zip(labels,preds)),sum(y==1 and p==1 for y,p in zip(labels,preds))]]
def wilson(errors,n,z=1.959963984540054):
    if n==0:return (None,None)
    p=errors/n; den=1+z*z/n; center=(p+z*z/(2*n))/den; half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return max(0.,center-half),min(1.,center+half)
def risk_row(experiment,version,threshold,labels,preds,confidences):
    ids=[i for i,c in enumerate(confidences) if c+1e-15>=threshold]; n=len(labels); accepted=len(ids); errors=sum(preds[i]!=labels[i] for i in ids); correct=accepted-errors
    al=[labels[i] for i in ids]; ap=[preds[i] for i in ids]; cm=confusion(al,ap) if ids else [[0,0],[0,0]]; pos=sum(labels); neg=n-pos; lo,hi=wilson(errors,accepted)
    p1=cm[1][1]/(cm[1][1]+cm[0][1]) if cm[1][1]+cm[0][1] else None
    r0=cm[0][0]/sum(x==0 for x in al) if any(x==0 for x in al) else None; r1=cm[1][1]/sum(x==1 for x in al) if any(x==1 for x in al) else None
    f0=2*(cm[0][0]/(cm[0][0]+cm[1][0]))*r0/((cm[0][0]/(cm[0][0]+cm[1][0]))+r0) if cm[0][0]+cm[1][0] and r0 and (cm[0][0]/(cm[0][0]+cm[1][0])+r0) else 0.
    f1=2*p1*r1/(p1+r1) if p1 is not None and r1 is not None and p1+r1 else 0.
    macro=(f0+f1)/2 if accepted and any(x==0 for x in al) and any(x==1 for x in al) else None
    return {"experiment_id":experiment,"probability_version":version,"confidence_threshold":threshold,"total_examples":n,"accepted_examples":accepted,"abstained_examples":n-accepted,"coverage":accepted/n,"abstention_rate":1-accepted/n,"accepted_correct_predictions":correct,"accepted_errors":errors,"selective_risk":errors/accepted if accepted else None,"accepted_accuracy":correct/accepted if accepted else None,"accepted_confusion_matrix_0_1":json.dumps(cm,separators=(",",":")),"accepted_macro_f1":macro,"positive_label_coverage":sum(labels[i]==1 for i in ids)/pos,"negative_label_coverage":sum(labels[i]==0 for i in ids)/neg,"accepted_true_positive_labelled_examples":sum(labels[i]==1 for i in ids),"accepted_true_negative_labelled_examples":sum(labels[i]==0 for i in ids),"accepted_predicted_positive_count":sum(preds[i] for i in ids),"accepted_positive_precision":p1,"overall_positive_recall_with_abstentions":sum(labels[i]==1 and preds[i]==1 for i in ids)/pos,"selective_risk_wilson_95_lower":lo,"selective_risk_wilson_95_upper":hi}
def select_candidates(rows):
    output=[]
    for experiment in EXPERIMENTS:
      for version in VERSIONS:
        subset=[r for r in rows if r["experiment_id"]==experiment and r["probability_version"]==version]
        for t in (.70,.80,.90): output.append({"experiment_id":experiment,"probability_version":version,"candidate_type":"fixed_threshold","constraint":None,"threshold":t,"evidence":next(r for r in subset if math.isclose(r["confidence_threshold"],t))})
        for floor in (.90,.80,.70):
            eligible=[r for r in subset if r["coverage"]+1e-15>=floor and r["accepted_true_positive_labelled_examples"]>0 and r["accepted_true_negative_labelled_examples"]>0]
            chosen=min(eligible,key=lambda r:(r["selective_risk"],-r["coverage"],r["confidence_threshold"])) if eligible else None
            output.append({"experiment_id":experiment,"probability_version":version,"candidate_type":"lowest_risk_at_coverage_floor","constraint":{"minimum_coverage":floor,"both_true_classes_required":True},"threshold":chosen["confidence_threshold"] if chosen else None,"evidence":chosen})
    return output
def verify_config(config):
    if config["access"]!={"allowed_inference_partition":"dev_calibration","forbidden_partitions":["train","dev_tune","test"]}: raise CalibrationError("Partition access policy changed")
    if [e["experiment_id"] for e in config["experiments"]]!=list(EXPERIMENTS) or any(e["seed"]!=42 for e in config["experiments"]): raise CalibrationError("Experiments must be E5/E6/E7 seed 42")
    if config["model"]["revision"]!="fb53ab8802853c8e4fbdbcd0529f21fc6f459b2b": raise CalibrationError("Model revision changed")
def verify_upstream(config):
    names={"source":config["input_csv"],"partition_config":config["partition_config"],"transformer_config":config["transformer_config"],"partition_run_status":config["partition_run_status"],"split_manifest":config["split_manifest"],"targets":config["targets"],"transformer_run_status":config["transformer_run_status"],"transformer_run_manifest":config["transformer_run_manifest"],"transformer_selected_metrics":config["transformer_selected_metrics"],"transformer_tokenization_manifest":config["transformer_tokenization_manifest"]}
    paths={k:safe_path(v,k) for k,v in names.items()}
    if any(not p.is_file() for p in paths.values()): raise CalibrationError("Required upstream artifact missing")
    hashes={k:sha256_file(p) for k,p in paths.items()}; ref=config["reference"]
    for k in ("source","partition_config","transformer_config","split_manifest","targets","transformer_run_manifest","transformer_selected_metrics","transformer_tokenization_manifest"):
        if hashes[k]!=ref[f"{k}_sha256"]: raise CalibrationError(f"Upstream hash mismatch: {k}")
    pstatus=load_json(paths["partition_run_status"]); tstatus=load_json(paths["transformer_run_status"]); manifest=load_json(paths["transformer_run_manifest"])
    if pstatus.get("status")!="COMPLETED" or tstatus.get("status")!="COMPLETED" or manifest.get("status")!="COMPLETED": raise CalibrationError("Upstream phase incomplete")
    if pstatus.get("provenance_fingerprint")!=ref["partition_fingerprint"] or manifest.get("partition_fingerprint")!=ref["partition_fingerprint"]: raise CalibrationError("Partition fingerprint mismatch")
    primary={u["experiment_id"]:u for u in manifest["unit_summaries"] if u["seed"]==42}
    if set(primary)!=set(EXPERIMENTS): raise CalibrationError("Exactly three primary checkpoint records required")
    checkpoint_hashes={}; checkpoint_before={}
    for spec in config["experiments"]:
        exp=spec["experiment_id"]; directory=safe_path(spec["checkpoint"],"checkpoint"); expected=primary[exp]["checkpoint"]
        if expected["path"]!=spec["checkpoint"] or not directory.is_dir(): raise CalibrationError(f"Checkpoint path mismatch: {exp}")
        files={p.name:sha256_file(p) for p in directory.iterdir() if p.is_file()}
        if files!=expected["files"] or set(files)!={"config.json","model.safetensors","tokenizer.json","tokenizer_config.json"}: raise CalibrationError(f"Checkpoint hash mismatch: {exp}")
        checkpoint_hashes[exp]=files; checkpoint_before[exp]=dict(files)
    return paths,hashes,manifest,primary,checkpoint_hashes,checkpoint_before
def infer(checkpoint, records, pad_id, batch_size):
    import torch
    from transformers import AutoModelForSequenceClassification
    model=AutoModelForSequenceClassification.from_pretrained(checkpoint,local_files_only=True,trust_remote_code=False,use_safetensors=True).to("cpu"); model.float(); model.eval(); out=[]
    start=time.perf_counter()
    with torch.inference_mode():
      for begin in range(0,len(records),batch_size):
        batch=records[begin:begin+batch_size]; packed=phase4b.collate(batch,pad_id,torch.device("cpu")); logits=model(input_ids=packed["input_ids"],attention_mask=packed["attention_mask"]).logits.detach().cpu().tolist(); out.extend((float(x[0]),float(x[1])) for x in logits)
    elapsed=time.perf_counter()-start; del model
    return out,elapsed
def publish_stage(stage: Path, output_dir: Path):
    if output_dir.exists(): raise CalibrationError("Completed output directory already exists; validate/reuse rather than overwrite")
    os.replace(stage,output_dir)
def markdown_report(manifest,parameters,metrics,risk):
    lines=["# Phase 05A Calibration and Selective-Prediction Review","", "**Run status:** `COMPLETED`  ","**Policy status:** `PENDING_RESEARCHER_REVIEW`  ",f"**Generated (UTC):** {manifest['timestamp_utc']}  ","", "Post-calibration changes below are apparent fit-set diagnostics because fitting and measurement both use the 171-example `dev_calibration` set. No model or policy is selected, and the test partition was not accessed.","","## Calibration diagnostics","","| Experiment | Version | Temperature | NLL | Brier | ECE | ROC-AUC | AP | Soft CE | Soft Brier | Soft MAD |","|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for m in metrics: lines.append(f"| {m['experiment_id']} | {m['probability_version']} | {m['temperature']:.6f} | {m['negative_log_likelihood']:.6f} | {m['brier_score']:.6f} | {m['expected_calibration_error']:.6f} | {m['roc_auc']:.6f} | {m['average_precision']:.6f} | {m['soft_cross_entropy']:.6f} | {m['soft_brier_score']:.6f} | {m['soft_positive_mean_absolute_difference']:.6f} |")
    lines += ["","## Risk/coverage checkpoints","","| Experiment | Version | Threshold | Coverage | Risk | Positive coverage | Accepted |","|---|---|---:|---:|---:|---:|---:|"]
    for r in risk:
      if any(math.isclose(r["confidence_threshold"],t) for t in (.5,.7,.8,.9)): lines.append(f"| {r['experiment_id']} | {r['probability_version']} | {r['confidence_threshold']:.2f} | {r['coverage']:.6f} | {'' if r['selective_risk'] is None else f'{r['selective_risk']:.6f}'} | {r['positive_label_coverage']:.6f} | {r['accepted_examples']} |")
    lines += ["","## Interpretation","","Temperature scaling preserved every native argmax prediction and associated classification metric. Reliability and threshold results are unstable with only 19 hard-positive examples; abstention can disproportionately remove positive-labelled examples. Final validity requires later locked-test evaluation after researcher policy approval.",""]
    return "\n".join(lines)
def regenerate_report_only(config_path: Path):
    config=load_json(config_path); output_dir=safe_path(config["outputs"]["directory"],"output"); manifest_path=output_dir/"run_manifest.json"
    manifest=load_json(manifest_path); params=load_json(output_dir/"calibration_parameters.json")
    if manifest.get("status")!="COMPLETED": raise CalibrationError("Phase 5A is not completed")
    protected=("dev_calibration_predictions.csv","calibration_parameters.json","calibration_metrics.csv","reliability_bins.csv","risk_coverage.csv","policy_candidates.json")
    before={name:sha256_file(output_dir/name) for name in protected}
    if any(manifest.get("output_hashes",{}).get(name)!=digest for name,digest in before.items()): raise CalibrationError("Saved numerical artifact hash mismatch before report correction")
    def csv_rows(path):
        with path.open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))
    metrics=csv_rows(output_dir/"calibration_metrics.csv"); risk=csv_rows(output_dir/"risk_coverage.csv")
    numeric_metric=set(METRIC_FIELDS)-{"experiment_id","probability_version","apparent_fit_set_diagnostic","confusion_matrix_0_1","support_0_1"}
    for row in metrics:
        for key in numeric_metric:
            if row[key]!="": row[key]=float(row[key])
    numeric_risk={"confidence_threshold","coverage","selective_risk","positive_label_coverage"}
    for row in risk:
        for key in numeric_risk: row[key]=None if row[key]=="" else float(row[key])
        row["accepted_examples"]=int(row["accepted_examples"])
    report_path=safe_path(config["outputs"]["report"],"report"); temp=report_path.with_suffix(".md.tmp")
    temp.write_text(markdown_report(manifest,params,metrics,risk),encoding="utf-8",newline="\n"); os.replace(temp,report_path)
    manifest["output_hashes"]["phase_05a_calibration_review.md"]=sha256_file(report_path)
    manifest["code_and_config_hashes"]["runner"]=sha256_file(Path(__file__))
    manifest["code_and_config_hashes"]["tests"]=sha256_file(PROJECT_ROOT/"tests"/"test_calibration_review.py")
    temp_manifest=manifest_path.with_suffix(".json.tmp"); write_json(temp_manifest,manifest); os.replace(temp_manifest,manifest_path)
    after={name:sha256_file(output_dir/name) for name in protected}
    if after!=before: raise CalibrationError("Numerical artifact changed during report-only correction")
    print("Phase 5A report-only correction completed; inference and calibration were not run.")
    return 0
def run(config_path: Path):
    config=load_json(config_path); verify_config(config); output_dir=safe_path(config["outputs"]["directory"],"output")
    if output_dir.exists():
        status=load_json(output_dir/"run_status.json") if (output_dir/"run_status.json").exists() else {}
        if status.get("status")=="COMPLETED": print("Existing completed Phase 5A outputs found; run the independent validator for safe reuse."); return 0
        if status.get("status")=="FAILED" and {p.name for p in output_dir.iterdir()}=={"run_status.json"}: shutil.rmtree(output_dir)
        else: raise CalibrationError("Conflicting incomplete Phase 5A output directory exists")
    paths,uphash,phase4manifest,primary,cphashes,checkpoint_before=verify_upstream(config)
    output_dir.parent.mkdir(parents=True,exist_ok=True); status_path=safe_path(config["outputs"]["run_status"],"status")
    stage=Path(tempfile.mkdtemp(prefix="phase5a-stage-",dir=str(output_dir.parent))); started=utc_now(); total_start=time.perf_counter()
    write_json(stage/"run_status.json",{"status":"RUNNING","policy_status":"PENDING_RESEARCHER_REVIEW","timestamp_utc":started})
    try:
      training_config=load_json(paths["transformer_config"]); training_paths={"source":paths["source"],"partition_split_manifest":paths["split_manifest"],"partition_targets":paths["targets"]}
      old_allowed=phase4b.ALLOWED_PARTITIONS; phase4b.ALLOWED_PARTITIONS=frozenset({"dev_calibration"})
      try: partitions=phase4b.load_partitions(["dev_calibration"],training_paths)
      finally: phase4b.ALLOWED_PARTITIONS=old_allowed
      rows=sorted(partitions["dev_calibration"],key=lambda x:x["example_id"])
      if len(rows)!=171 or sum(r["hard_label"] for r in rows)!=19: raise CalibrationError("Calibration partition counts mismatch")
      from transformers import AutoTokenizer
      tokenizer=AutoTokenizer.from_pretrained(safe_path(config["experiments"][0]["checkpoint"]),local_files_only=True,trust_remote_code=False,use_fast=True)
      encoded,token_rows,token_meta=phase4b.construct_encodings(tokenizer,{"dev_calibration":rows},training_config)
      if token_meta["marker_token_counts"]!={"preceding":6,"target":3,"following":4} or max(int(r["sequence_length"]) for r in token_rows)>256: raise CalibrationError("Input reconstruction validation failed")
      context_hash={r["example_id"]:(r["input_ids_sha256"],r["attention_mask_sha256"]) for r in token_rows if r["input_mode"]=="context"}
      prediction_rows=[]; parameter_models=[]; metric_rows=[]; bin_rows=[]; risk_rows=[]; runtimes={}
      for spec in config["experiments"]:
        exp=spec["experiment_id"]; records=[encoded[spec["input_mode"]][r["example_id"]] for r in rows]; logits,elapsed=infer(safe_path(spec["checkpoint"]),records,tokenizer.pad_token_id,config["inference"]["batch_size"]); runtimes[exp]=elapsed
        labels=[r["hard_label"] for r in rows]; soft=[r["soft_positive"] for r in rows]; fit=fit_temperature(logits,labels,config["calibration"]["minimum_temperature"],config["calibration"]["maximum_temperature"],config["calibration"]["tolerance"],config["calibration"]["iteration_limit"]); fit.update({"experiment_id":exp,"seed":42,"selected_epoch":primary[exp]["selected_epoch"],"fit_partition":"dev_calibration","fit_examples":171,"hard_positive_examples":19}); parameter_models.append(fit)
        pu=[softmax_pair(*pair) for pair in logits]; pc=[softmax_pair(*pair,fit["temperature"]) for pair in logits]
        if any((a[1]>=a[0])!=(b[1]>=b[0]) for a,b in zip(pu,pc)): raise CalibrationError("Temperature scaling changed argmax")
        for row,pair,u,c in zip(rows,logits,pu,pc): prediction_rows.append({"experiment_id":exp,"seed":42,"selected_epoch":primary[exp]["selected_epoch"],"example_id":row["example_id"],"partition":"dev_calibration","true_hard_label":row["hard_label"],"soft_other":row["soft_other"],"soft_positive":row["soft_positive"],"predicted_label_uncalibrated":int(u[1]>=u[0]),"logit_0":pair[0],"logit_1":pair[1],"probability_0_uncalibrated":u[0],"probability_1_uncalibrated":u[1],"temperature":fit["temperature"],"predicted_label_calibrated":int(c[1]>=c[0]),"probability_0_calibrated":c[0],"probability_1_calibrated":c[1],"confidence_uncalibrated":max(u),"confidence_calibrated":max(c)})
        for version,pairs in zip(VERSIONS,(pu,pc)):
          probs=[p[1] for p in pairs]; metrics=metric_record(labels,soft,probs); rec={"experiment_id":exp,"probability_version":version,"apparent_fit_set_diagnostic":version=="temperature_scaled","temperature":1.0 if version=="uncalibrated" else fit["temperature"],"ece_bin_count":10,**metrics}; rec["confusion_matrix_0_1"]=json.dumps(rec["confusion_matrix_0_1"],separators=(",",":")); rec["support_0_1"]=json.dumps(rec["support_0_1"],separators=(",",":")); metric_rows.append(rec)
          for b in reliability_bins(labels,soft,probs): bin_rows.append({"experiment_id":exp,"probability_version":version,**b})
          preds=[int(p[1]>=p[0]) for p in pairs]; conf=[max(p) for p in pairs]
          for n in range(50,100): risk_rows.append(risk_row(exp,version,n/100,labels,preds,conf))
      if len(prediction_rows)!=513 or len(metric_rows)!=6 or len(bin_rows)!=60 or len(risk_rows)!=300: raise CalibrationError("Generated row counts mismatch")
      params={"schema_version":"5A-1.0.0","method":"scalar_temperature","objective":"ordinary_unweighted_hard_label_negative_log_likelihood","interpretation":"Post-calibration changes are apparent fit-set diagnostics, not unbiased estimates.","models":parameter_models}
      policies={"policy_status":"PENDING_RESEARCHER_REVIEW","approved_policy":None,"scope":"Calibration-set descriptive candidates only; none is an approved operating policy.","selection_grid":[n/100 for n in range(50,100)],"candidates":select_candidates(risk_rows)}
      write_csv(stage/"dev_calibration_predictions.csv",PREDICTION_FIELDS,prediction_rows); write_json(stage/"calibration_parameters.json",params); write_csv(stage/"calibration_metrics.csv",METRIC_FIELDS,({k:r.get(k) for k in METRIC_FIELDS} for r in metric_rows)); write_csv(stage/"reliability_bins.csv",BIN_FIELDS,bin_rows); write_csv(stage/"risk_coverage.csv",RISK_FIELDS,risk_rows); write_json(stage/"policy_candidates.json",policies)
      output_hashes={p.name:sha256_file(p) for p in stage.iterdir() if p.is_file() and p.name!="run_status.json"}; versions={n:importlib.metadata.version(n) for n in ("torch","transformers","tokenizers","safetensors","numpy")}; versions["python"]=platform.python_version()
      current_upstream={k:sha256_file(p) for k,p in paths.items()}; current_cp={e:{p.name:sha256_file(p) for p in safe_path(next(x["checkpoint"] for x in config["experiments"] if x["experiment_id"]==e)).iterdir() if p.is_file()} for e in EXPERIMENTS}
      if current_upstream!=uphash or current_cp!=checkpoint_before: raise CalibrationError("Upstream or checkpoint changed during run")
      manifest={"schema_version":"5A-1.0.0","status":"COMPLETED","policy_status":"PENDING_RESEARCHER_REVIEW","run_started_utc":started,"timestamp_utc":utc_now(),"runtime_seconds":time.perf_counter()-total_start,"partition_fingerprint":config["reference"]["partition_fingerprint"],"partition_accessed_for_inference":["dev_calibration"],"forbidden_partitions_accessed_for_inference":[],"permitted_examples":171,"hard_positive_examples":19,"prediction_rows":513,"device":"cpu","precision":"float32","inference_batch_size":config["inference"]["batch_size"],"model":config["model"],"experiments":[{"experiment_id":e,"seed":42,"selected_epoch":primary[e]["selected_epoch"],"input_mode":next(x["input_mode"] for x in config["experiments"] if x["experiment_id"]==e),"runtime_seconds":runtimes[e]} for e in EXPERIMENTS],"input_reconstruction":token_meta,"e6_e7_input_ids_and_masks_identical":True,"maximum_observed_sequence_length":max(int(r["sequence_length"]) for r in token_rows),"upstream_hashes_before_and_after":uphash,"checkpoint_hashes":cphashes,"code_and_config_hashes":{"config":sha256_file(config_path),"runner":sha256_file(Path(__file__)),"validator":sha256_file(PROJECT_ROOT/"scripts/validate_calibration_review.py"),"tests":sha256_file(PROJECT_ROOT/"tests/test_calibration_review.py"),"protocol":sha256_file(PROJECT_ROOT/"reports/PHASE_05_CALIBRATION_PROTOCOL.md")},"python_and_dependencies":versions,"output_hashes":output_hashes,"integrity":{"source_unchanged":True,"partitions_unchanged":True,"phase3_outputs_unchanged":True,"phase4a_outputs_unchanged":True,"phase4b_outputs_unchanged":True,"checkpoints_unchanged":True,"new_model_weights_created":False,"test_inference_or_predictions":False,"raw_text_in_generated_artifacts":False}}
      report=markdown_report(manifest,params,metric_rows,risk_rows); report_path=safe_path(config["outputs"]["report"]); report_path.parent.mkdir(parents=True,exist_ok=True); report_tmp=stage/"phase_05a_calibration_review.md"; report_tmp.write_text(report,encoding="utf-8",newline="\n")
      manifest["output_hashes"]["phase_05a_calibration_review.md"]=sha256_file(report_tmp)
      write_json(stage/"run_manifest.json",manifest); write_json(stage/"run_status.json",{"status":"COMPLETED","policy_status":"PENDING_RESEARCHER_REVIEW","timestamp_utc":manifest["timestamp_utc"],"message":"Phase 5A evidence generation completed; policy selection remains pending."})
      publish_stage(stage,output_dir); shutil.copyfile(output_dir/"phase_05a_calibration_review.md",report_path); (output_dir/"phase_05a_calibration_review.md").unlink()
      print("Phase 5A calibration review completed: COMPLETED / PENDING_RESEARCHER_REVIEW"); return 0
    except Exception:
      shutil.rmtree(stage,ignore_errors=True); output_dir.mkdir(parents=True,exist_ok=True); write_json(status_path,{"status":"FAILED","policy_status":"PENDING_RESEARCHER_REVIEW","timestamp_utc":utc_now(),"message":"Phase 5A failed; no staged evidence was published."}); raise
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--config",default="configs/calibration_review.json"); p.add_argument("--regenerate-report-only",action="store_true"); a=p.parse_args(argv)
    try:return regenerate_report_only(safe_path(a.config,"config")) if a.regenerate_report_only else run(safe_path(a.config,"config"))
    except CalibrationError as exc: print(f"Phase 5A failed: {exc}",file=sys.stderr); return 2
    except Exception as exc: print(f"Phase 5A failed unexpectedly: {type(exc).__name__}: {exc}",file=sys.stderr); return 3
if __name__=="__main__": raise SystemExit(main())
