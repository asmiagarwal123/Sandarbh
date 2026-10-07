#!/usr/bin/env python3
"""One-time, policy-frozen SANDARBH Phase 6 locked-test evaluation."""
from __future__ import annotations

import argparse, csv, hashlib, importlib.util, json, math, os, random, statistics, subprocess, sys, time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
AUTHORIZED = "b5765e9431a4bf01e8abfeeec1762061b1c0a26eec4ea26c326abb7245ac7541"
SYSTEMS = ("E1","E2","E3","E4","E5","E6","E7","DUMMY")
TRANSFORMERS = ("E5","E6","E7")
COMPARISONS = (("E6","E5"),("E7","E6"),("E5","E1"),("E5","E3"),("E6","E2"),("E6","E4"))
CLASS_FIELDS = ("macro_f1","positive_precision","positive_recall","positive_f1","negative_precision","negative_recall","negative_f1","accuracy","balanced_accuracy")
OUTPUT_NAMES = ("transformer_test_predictions.csv","baseline_test_predictions.csv","classification_metrics.csv","probability_metrics.csv","reliability_bins.csv","risk_coverage.csv","paired_comparisons.csv","bootstrap_intervals.csv","bootstrap_manifest.json","research_question_summary.json","run_manifest.json")

class Phase6Error(Exception): pass

def utcnow(): return datetime.now(timezone.utc).isoformat().replace("+00:00","Z")
def sha_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""): h.update(b)
    return h.hexdigest()
def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f: value=json.load(f)
    if not isinstance(value,dict): raise Phase6Error(f"JSON object required: {path}")
    return value
def read_csv(path: Path) -> list[dict[str,str]]:
    with path.open(encoding="utf-8-sig",newline="") as f: return list(csv.DictReader(f,strict=True))
def atomic_json(path: Path, value: Any):
    tmp=path.with_name(path.name+".tmp")
    with tmp.open("w",encoding="utf-8",newline="\n") as f: json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False); f.write("\n")
    os.replace(tmp,path)
def write_csv(path: Path, fields: list[str], rows: list[dict]):
    tmp=path.with_name(path.name+".tmp")
    with tmp.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore",lineterminator="\n"); w.writeheader(); w.writerows(rows)
    os.replace(tmp,path)
def fnum(x): return "" if x is None else repr(float(x))

def safe_div(a,b): return a/b if b else 0.0
def classification(y, pred):
    tn=sum(a==0 and b==0 for a,b in zip(y,pred)); fp=sum(a==0 and b==1 for a,b in zip(y,pred))
    fn=sum(a==1 and b==0 for a,b in zip(y,pred)); tp=sum(a==1 and b==1 for a,b in zip(y,pred))
    p1=safe_div(tp,tp+fp); r1=safe_div(tp,tp+fn); f1=safe_div(2*p1*r1,p1+r1)
    p0=safe_div(tn,tn+fn); r0=safe_div(tn,tn+fp); f0=safe_div(2*p0*r0,p0+r0)
    return {"macro_f1":(f0+f1)/2,"positive_precision":p1,"positive_recall":r1,"positive_f1":f1,
      "negative_precision":p0,"negative_recall":r0,"negative_f1":f0,"accuracy":safe_div(tn+tp,len(y)),
      "balanced_accuracy":(r0+r1)/2,"confusion_matrix_0_1":[[tn,fp],[fn,tp]],"support_0_1":[tn+fp,fn+tp],
      "predicted_positive_count":sum(pred)}
def roc_auc(y,s):
    pos=sum(y); neg=len(y)-pos
    if not pos or not neg:return None
    order=sorted(range(len(s)),key=lambda i:s[i]); ranks=[0.0]*len(s); i=0
    while i<len(order):
        j=i+1
        while j<len(order) and s[order[j]]==s[order[i]]:j+=1
        rank=((i+1)+j)/2
        for k in range(i,j):ranks[order[k]]=rank
        i=j
    return (sum(r for r,a in zip(ranks,y) if a==1)-pos*(pos+1)/2)/(pos*neg)
def average_precision(y,s):
    pos=sum(y)
    if not pos:return None
    pairs=sorted(zip(s,y),reverse=True); tp=fp=0; prev=out=0.0; i=0
    while i<len(pairs):
        score=pairs[i][0]; group=[]
        while i<len(pairs) and pairs[i][0]==score:group.append(pairs[i][1]);i+=1
        tp+=sum(group);fp+=len(group)-sum(group); rec=tp/pos; out+=(rec-prev)*tp/(tp+fp);prev=rec
    return out
def probability_metrics(y,soft,p,bins=10):
    q=[min(max(x,1e-15),1-1e-15) for x in p]
    n=len(y)
    return {"negative_log_likelihood":-sum(a*math.log(x)+(1-a)*math.log(1-x) for a,x in zip(y,q))/n,
      "brier_score":sum((x-a)**2 for a,x in zip(y,p))/n,"expected_calibration_error":sum(
       len(ix)/n*abs(sum(p[j] for j in ix)/len(ix)-sum(y[j] for j in ix)/len(ix)) for b in range(bins)
       if (ix:=[j for j,x in enumerate(p) if x>=b/bins and (x<(b+1)/bins or b==bins-1)])),
      "roc_auc":roc_auc(y,p),"average_precision":average_precision(y,p),
      "soft_cross_entropy":-sum((1-a)*math.log(1-x)+a*math.log(x) for a,x in zip(soft,q))/n,
      "soft_brier_score":sum((x-a)**2 for a,x in zip(soft,p))/n,
      "soft_positive_mean_absolute_difference":sum(abs(x-a) for a,x in zip(soft,p))/n}
def reliability(y,p,state,model):
    out=[]
    for b in range(10):
        lo=b/10;hi=(b+1)/10; ix=[i for i,x in enumerate(p) if x>=lo and (x<hi or b==9)]
        out.append({"experiment_id":model,"probability_state":state,"bin_index":b,"lower_bound":lo,"upper_bound":hi,
          "upper_inclusive":str(b==9),"count":len(ix),"mean_probability":fnum(sum(p[i] for i in ix)/len(ix) if ix else None),
          "observed_positive_rate":fnum(sum(y[i] for i in ix)/len(ix) if ix else None),
          "absolute_gap":fnum(abs(sum(p[i] for i in ix)/len(ix)-sum(y[i] for i in ix)/len(ix)) if ix else None)})
    return out
def wilson(errors,n,z=1.959963984540054):
    if not n:return (None,None)
    p=errors/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return max(0,c-h),min(1,c+h)
def selective(y,pred,conf,t):
    ix=[i for i,x in enumerate(conf) if x>=t]; yy=[y[i] for i in ix]; pp=[pred[i] for i in ix]
    errors=sum(a!=b for a,b in zip(yy,pp)); lo,hi=wilson(errors,len(ix)); cm=classification(yy,pp) if ix else None
    pos=sum(y);neg=len(y)-pos; accepted_pos=sum(y[i]==1 for i in ix);accepted_neg=sum(y[i]==0 for i in ix)
    return {"total_examples":len(y),"accepted_examples":len(ix),"abstained_examples":len(y)-len(ix),"coverage":len(ix)/len(y),
      "abstention_rate":1-len(ix)/len(y),"accepted_correct_predictions":len(ix)-errors,"accepted_errors":errors,
      "selective_risk":errors/len(ix) if ix else None,"wilson_95_lower":lo,"wilson_95_upper":hi,
      "accepted_accuracy":1-errors/len(ix) if ix else None,"accepted_confusion_matrix_0_1":cm["confusion_matrix_0_1"] if cm else None,
      "accepted_macro_f1":cm["macro_f1"] if cm else None,"positive_label_coverage":accepted_pos/pos if pos else None,
      "negative_label_coverage":accepted_neg/neg if neg else None,"accepted_true_positive_labelled_examples":accepted_pos,
      "accepted_true_negative_labelled_examples":accepted_neg,"accepted_predicted_positive_count":sum(pp),
      "accepted_positive_precision":cm["positive_precision"] if cm else None,
      "overall_positive_recall_with_abstentions":sum(y[i]==1 and pred[i]==1 for i in ix)/pos if pos else None}

def preflight(config: dict, allow_receipt=False):
    if config.get("authorized_successor_policy_sha256")!=AUTHORIZED:raise Phase6Error("Wrong authorized successor policy hash")
    checks={}; required={"partitions":"results/partitions/run_status.json","baselines":"results/baselines/run_status.json","transformers":"results/transformers/run_status.json","calibration":"results/calibration_review/run_status.json"}
    for name,rel in required.items(): checks[name]=load_json(ROOT/rel).get("status")=="COMPLETED"
    checks["original_policy"]=load_json(ROOT/"results/frozen_policy/run_status.json").get("policy_status")=="FROZEN"
    successor=load_json(ROOT/config["successor_policy"]); policy=successor.get("policy")
    checks["successor_policy"]=successor.get("canonical_policy_sha256")==AUTHORIZED and canonical_hash(policy)==AUTHORIZED and policy.get("policy_status")=="FROZEN" and policy.get("test_status")=="NOT_ACCESSED"
    v=load_json(ROOT/"results/frozen_policy_v1_1/validation_summary.json");checks["successor_validation"]=v.get("passed") and v.get("check_count")==31 and all(v.get("checks",{}).values())
    checks["source_hash"]=sha_file(ROOT/config["source"])==config["source_sha256"]
    checks["split_hash"]=sha_file(ROOT/config["split_manifest"])==config["split_manifest_sha256"]
    checks["targets_hash"]=sha_file(ROOT/config["targets"])==config["targets_sha256"]
    split=read_csv(ROOT/config["split_manifest"]); test=[r for r in split if r["partition"]=="test"]; ids=[r["example_id"] for r in test]; groups=[r["group_id"] for r in test]
    other_groups={r["group_id"] for r in split if r["partition"]!="test"}
    checks["test_structure"]=len(ids)==341 and len(set(ids))==341 and all(groups) and not(set(groups)&other_groups)
    targets={r["example_id"]:r for r in read_csv(ROOT/config["targets"])}; counts=sum(int(targets[i]["hard_label"]) for i in ids)
    checks["test_counts"]=counts==38 and len(ids)-counts==303
    model_hashes={}
    for m,s in config["baseline_models"].items():model_hashes[m]=sha_file(ROOT/s["path"]);checks[f"model_{m}"]=model_hashes[m]==s["sha256"]
    for m,s in config["transformers"].items():model_hashes[m]=sha_file(ROOT/s["path"]/"model.safetensors");checks[f"model_{m}"]=model_hashes[m]==s["model_sha256"]
    out=ROOT/config["outputs"]["directory"]; receipt=out/"test_access_receipt.json"
    if receipt.exists() and not allow_receipt: checks["no_prior_receipt"]=False
    elif not receipt.exists():checks["no_prior_receipt"]=True
    prior=[out/n for n in OUTPUT_NAMES+("run_status.json","validation_summary.json") if (out/n).exists()]
    checks["no_prior_outputs"]=not prior if not allow_receipt else True
    if not all(checks.values()):raise Phase6Error("Preflight failed: "+", ".join(k for k,v in checks.items() if not v))
    return {"status":"READY","checks":checks,"test_ids":ids,"test_id_set_sha256":canonical_hash(sorted(ids)),"model_hashes":model_hashes,"test_group_count":len(set(groups))}

def baseline_worker():
    import joblib
    payload=json.load(sys.stdin); out=[]
    for model,spec in payload["models"].items():
        pipe=joblib.load(spec["path"]); texts=payload["texts"][spec["input_mode"]]; pred=pipe.predict(texts).astype(int).tolist()
        clf=pipe.named_steps["classifier"]
        if clf.classes_.astype(int).tolist()!=[0,1]:raise RuntimeError("class order")
        if spec["score_type"]=="native_positive_probability":
            probs=pipe.predict_proba(texts); scores=probs[:,1].tolist(); p0=probs[:,0].tolist();p1=scores
        else:scores=pipe.decision_function(texts).tolist();p0=[None]*len(texts);p1=[None]*len(texts)
        out.append({"experiment_id":model,"predicted":pred,"scores":scores,"p0":p0,"p1":p1,"score_type":spec["score_type"]})
    json.dump(out,sys.stdout,separators=(",",":"),allow_nan=False)

def import_training():
    path=ROOT/"scripts/train_transformers.py";spec=importlib.util.spec_from_file_location("phase4_training_reuse",path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
def load_test_records(config):
    mod=import_training();mod.ALLOWED_PARTITIONS=frozenset({"test"})
    paths={"source":ROOT/config["source"],"partition_split_manifest":ROOT/config["split_manifest"],"partition_targets":ROOT/config["targets"]}
    rows=mod.load_partitions(["test"],paths)["test"]
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(ROOT/config["transformers"]["E5"]["path"],local_files_only=True)
    tcfg=load_json(ROOT/"configs/transformer_training.json");encoded,manifest,meta=mod.construct_encodings(tokenizer,{"test":rows},tcfg)
    return mod,rows,encoded,manifest,meta,tokenizer
def softmax_pair(a,b,temp=1.0):
    a/=temp;b/=temp;m=max(a,b);ea=math.exp(a-m);eb=math.exp(b-m);return ea/(ea+eb),eb/(ea+eb)

def run_inference(config, rows, encoded, tokenizer):
    import torch
    from transformers import AutoModelForSequenceClassification
    trans=[]; start=time.monotonic()
    for model,s in config["transformers"].items():
        net=AutoModelForSequenceClassification.from_pretrained(ROOT/s["path"],local_files_only=True).to(torch.device("cpu"));net.float();net.eval()
        records=[encoded[s["input_mode"]][r["example_id"]] for r in rows]
        with torch.inference_mode():
            for pos in range(0,len(records),8):
                batch=records[pos:pos+8];maximum=max(len(x["input_ids"]) for x in batch)
                ids=torch.tensor([x["input_ids"]+[tokenizer.pad_token_id]*(maximum-len(x["input_ids"])) for x in batch]);mask=torch.tensor([x["attention_mask"]+[0]*(maximum-len(x["attention_mask"])) for x in batch])
                logits=net(input_ids=ids,attention_mask=mask).logits.cpu().tolist()
                for source,(l0,l1) in zip(rows[pos:pos+8],logits):
                    u0,u1=softmax_pair(l0,l1);c0,c1=softmax_pair(l0,l1,s["temperature"]);pu=int(l1>l0);pc=int(c1>c0)
                    if pu!=pc:raise Phase6Error("Temperature scaling changed argmax")
                    trans.append({"experiment_id":model,"seed":s["seed"],"selected_epoch":s["selected_epoch"],"example_id":source["example_id"],"group_id":source["group_id"],"partition":"test","true_hard_label":source["hard_label"],"soft_other":source["soft_other"],"soft_positive":source["soft_positive"],"logit_0":l0,"logit_1":l1,"probability_0_uncalibrated":u0,"probability_1_uncalibrated":u1,"probability_0_calibrated":c0,"probability_1_calibrated":c1,"predicted_label_uncalibrated":pu,"predicted_label_calibrated":pc,"confidence_calibrated":max(c0,c1),"temperature":s["temperature"]})
        del net
    texts={"target":[" ".join(r["target"].split()) for r in rows],"context":["\n".join(x for x in (" ".join(r["preceding"].split())," ".join(r["target"].split())," ".join(r["following"].split())) if x) for r in rows]}
    payload={"models":{m:{**s,"path":str(ROOT/s["path"])} for m,s in config["baseline_models"].items()},"texts":texts}
    proc=subprocess.run([str(ROOT/config["baseline_python"]),str(Path(__file__).resolve()),"--baseline-worker"],input=json.dumps(payload),text=True,capture_output=True)
    if proc.returncode:raise Phase6Error("Baseline worker failed: "+proc.stderr[-500:])
    worker=json.loads(proc.stdout); base=[]
    for result in worker:
        for i,r in enumerate(rows):base.append({"experiment_id":result["experiment_id"],"example_id":r["example_id"],"group_id":r["group_id"],"partition":"test","true_hard_label":r["hard_label"],"soft_other":r["soft_other"],"soft_positive":r["soft_positive"],"predicted_label":result["predicted"][i],"score_type":result["score_type"],"continuous_score":result["scores"][i],"probability_0":result["p0"][i],"probability_1":result["p1"][i]})
    for r in rows:base.append({"experiment_id":"DUMMY","example_id":r["example_id"],"group_id":r["group_id"],"partition":"test","true_hard_label":r["hard_label"],"soft_other":r["soft_other"],"soft_positive":r["soft_positive"],"predicted_label":0,"score_type":"unavailable","continuous_score":None,"probability_0":None,"probability_1":None})
    return trans,base,time.monotonic()-start

def derive(config,trans,base):
    by={m:[] for m in SYSTEMS}
    for r in base:by[r["experiment_id"]].append(r)
    for r in trans:by[r["experiment_id"]].append(r)
    class_rows=[]; metrics={}; prob_rows=[]; rel=[];risk=[]
    for m in SYSTEMS:
        rr=by[m];y=[int(x["true_hard_label"]) for x in rr];pred=[int(x.get("predicted_label",x.get("predicted_label_calibrated"))) for x in rr];cm=classification(y,pred);score=None
        if m in ("E1","E2"):score=[float(x["probability_1"]) for x in rr]
        elif m in ("E3","E4"):score=[float(x["continuous_score"]) for x in rr]
        elif m in TRANSFORMERS:score=[float(x["probability_1_calibrated"]) for x in rr]
        cm.update({"experiment_id":m,"roc_auc":roc_auc(y,score) if score else None,"average_precision":average_precision(y,score) if score else None});metrics[m]=cm
        class_rows.append({**cm,"confusion_matrix_0_1":json.dumps(cm["confusion_matrix_0_1"],separators=(",",":")),"support_0_1":json.dumps(cm["support_0_1"],separators=(",",":"))})
        soft=[float(x["soft_positive"]) for x in rr]
        if m in ("E1","E2"):
            pm=probability_metrics(y,soft,score);prob_rows.append({"experiment_id":m,"probability_state":"native_uncalibrated",**{k:(pm[k] if k in ("negative_log_likelihood","brier_score","roc_auc","average_precision") else None) for k in pm}})
        if m in TRANSFORMERS:
            for state,key in (("uncalibrated","probability_1_uncalibrated"),("calibrated","probability_1_calibrated")):
                p=[float(x[key]) for x in rr];pm=probability_metrics(y,soft,p);prob_rows.append({"experiment_id":m,"probability_state":state,**pm});rel+=reliability(y,p,state,m)
            conf=[float(x["confidence_calibrated"]) for x in rr]
            for n in range(50,100):risk.append({"experiment_id":m,"threshold":n/100,"is_frozen_primary":str(n==60),**selective(y,pred,conf,n/100)})
    comp=[]
    trans_metrics=list(CLASS_FIELDS)+["negative_log_likelihood","brier_score","expected_calibration_error","roc_auc","average_precision","soft_brier_score","soft_cross_entropy"]
    calibrated={(r["experiment_id"],r["probability_state"]):r for r in prob_rows}
    for left,right in COMPARISONS:
        names=trans_metrics if (left,right) in COMPARISONS[:2] else ["macro_f1","positive_f1","positive_recall","balanced_accuracy","accuracy","roc_auc","average_precision"]
        for name in names:
            lv=calibrated.get((left,"calibrated"),{}).get(name,metrics[left].get(name));rv=calibrated.get((right,"calibrated"),{}).get(name,metrics[right].get(name))
            comp.append({"comparison":f"{left}-{right}","direction":f"{left} minus {right}","metric":name,"left_value":lv,"right_value":rv,"difference":lv-rv if lv is not None and rv is not None else None})
    return by,class_rows,metrics,prob_rows,rel,risk,comp

def percentile(v,q):
    if not v:return None
    a=sorted(v);x=(len(a)-1)*q;lo=int(math.floor(x));hi=int(math.ceil(x));return a[lo]+(a[hi]-a[lo])*(x-lo)
def bootstrap(by,metrics,prob_rows,config):
    ref=by["E5"];groups=sorted({r["group_id"] for r in ref});indices={g:[i for i,r in enumerate(ref) if r["group_id"]==g] for g in groups};rng=random.Random(config["bootstrap"]["seed"]);h=hashlib.sha256()
    model_names={m:["macro_f1","positive_f1","positive_recall","balanced_accuracy"] for m in SYSTEMS}; values=defaultdict(list);unavail=defaultdict(int)
    for m in TRANSFORMERS:model_names[m]+=["negative_log_likelihood","brier_score"]
    probs={(r["experiment_id"],r["probability_state"]):r for r in prob_rows}
    for _ in range(config["bootstrap"]["replicates"]):
        sampled=[rng.randrange(len(groups)) for _ in groups];h.update((json.dumps(sampled,separators=(",",":"))+"\n").encode());ix=[i for j in sampled for i in indices[groups[j]]]
        replicate={}
        for m in SYSTEMS:
            rr=by[m];y=[int(rr[i]["true_hard_label"]) for i in ix];p=[int(rr[i].get("predicted_label",rr[i].get("predicted_label_calibrated"))) for i in ix]
            c=classification(y,p) if len(set(y))==2 else {}
            for name in model_names[m]:
                key=("model",m,name);val=c.get(name)
                if name in ("negative_log_likelihood","brier_score"):
                    ps=[float(rr[i]["probability_1_calibrated"]) for i in ix];soft=[float(rr[i]["soft_positive"]) for i in ix];val=probability_metrics(y,soft,ps)[name]
                if val is None:unavail[key]+=1
                else:values[key].append(val)
            replicate[m]=c
        for left,right in COMPARISONS:
            for name in ("macro_f1","positive_f1","positive_recall"):
                key=("difference",f"{left}-{right}",name);a=replicate[left].get(name);b=replicate[right].get(name)
                if a is None or b is None:unavail[key]+=1
                else:values[key].append(a-b)
    rows=[]
    for key,vals in sorted(values.items()):
        kind,subject,name=key;point=(metrics[subject][name] if kind=="model" and name in metrics[subject] else probs[(subject,"calibrated")][name] if kind=="model" else metrics[subject.split("-")[0]][name]-metrics[subject.split("-")[1]][name])
        rows.append({"interval_type":kind,"subject":subject,"metric":name,"point_estimate":point,"lower_95":percentile(vals,.025),"upper_95":percentile(vals,.975),"requested_replicates":config["bootstrap"]["replicates"],"valid_replicates":len(vals),"unavailable_replicates":unavail[key]})
    manifest={"seed":config["bootstrap"]["seed"],"method":"paired_cluster_bootstrap","cluster":"group_id","cluster_count":len(groups),"replicate_count":config["bootstrap"]["replicates"],"sampled_cluster_indices_sha256":h.hexdigest(),"availability_counts":[{"interval_type":r["interval_type"],"subject":r["subject"],"metric":r["metric"],"valid":r["valid_replicates"],"unavailable":r["unavailable_replicates"]} for r in rows],"implementation_sha256":sha_file(Path(__file__).resolve()),"p_values_generated":False,"resample_until_valid":False}
    return rows,manifest

def execute(config_path: Path,config: dict):
    out=ROOT/config["outputs"]["directory"];receipt_path=out/"test_access_receipt.json";script_hash=sha_file(Path(__file__).resolve());config_hash=sha_file(config_path)
    if receipt_path.exists():
        rec=load_json(receipt_path);expected=(rec.get("evaluation_script_sha256")==script_hash and rec.get("config_sha256")==config_hash and rec.get("successor_policy_sha256")==AUTHORIZED)
        status=load_json(out/"run_status.json") if (out/"run_status.json").exists() else {}
        if status.get("status")=="COMPLETED":raise Phase6Error("Fresh execution refused: locked test already evaluated once")
        if not expected:raise Phase6Error("Recovery refused: code/config provenance changed")
        pf=preflight(config,allow_receipt=True);recovery=True
    else:
        pf=preflight(config);out.mkdir(parents=True,exist_ok=True)
        receipt={"timestamp_utc":utcnow(),"successor_policy_sha256":AUTHORIZED,"config_sha256":config_hash,"evaluation_script_sha256":script_hash,"metric_implementation_sha256":script_hash,"bootstrap_implementation_sha256":script_hash,"test_id_set_sha256":pf["test_id_set_sha256"],"split_manifest_sha256":config["split_manifest_sha256"],"targets_sha256":config["targets_sha256"],"model_hashes":pf["model_hashes"],"status":"TEST_ACCESSED","declaration":"Policy, models, metrics, temperatures and thresholds may not change after this receipt."}
        flags=os.O_WRONLY|os.O_CREAT|os.O_EXCL
        fd=os.open(receipt_path,flags)
        with os.fdopen(fd,"w",encoding="utf-8",newline="\n") as f:json.dump(receipt,f,indent=2);f.write("\n")
        recovery=False
    started=time.monotonic();atomic_json(out/"run_status.json",{"status":"RUNNING","policy_status":"FROZEN","test_status":"TEST_ACCESSED","successor_policy_sha256":AUTHORIZED,"recovery":recovery})
    try:
        mod,rows,encoded,token_manifest,meta,tokenizer=load_test_records(config)
        trans,base,inference_seconds=run_inference(config,rows,encoded,tokenizer)
        by,class_rows,metrics,prob_rows,rel,risk,comp=derive(config,trans,base);boots,bmanifest=bootstrap(by,metrics,prob_rows,config)
        tfields=list(trans[0]);bfields=list(base[0]);write_csv(out/"transformer_test_predictions.csv",tfields,trans);write_csv(out/"baseline_test_predictions.csv",bfields,base)
        write_csv(out/"classification_metrics.csv",list(class_rows[0]),class_rows);write_csv(out/"probability_metrics.csv",list(prob_rows[0]),prob_rows);write_csv(out/"reliability_bins.csv",list(rel[0]),rel)
        risk2=[{k:(json.dumps(v,separators=(",",":")) if isinstance(v,list) else v) for k,v in r.items()} for r in risk];write_csv(out/"risk_coverage.csv",list(risk2[0]),risk2);write_csv(out/"paired_comparisons.csv",list(comp[0]),comp);write_csv(out/"bootstrap_intervals.csv",list(boots[0]),boots);atomic_json(out/"bootstrap_manifest.json",bmanifest)
        intervals={(r["subject"],r["metric"]):r for r in boots if r["interval_type"]=="difference"}
        def evidence(c):
            vals=[next(r["difference"] for r in comp if r["comparison"]==c and r["metric"]==m) for m in ("macro_f1","positive_f1","positive_recall")]
            return "favourable" if all(x>0 for x in vals) else "unfavourable" if all(x<0 for x in vals) else "mixed"
        rq={"language_scope":"this locked test only","rq1":{"question":"Does context help?","evidence":"E6-E5","assessment":evidence("E6-E5"),"bootstrap_intervals":[intervals[("E6-E5",m)] for m in ("macro_f1","positive_f1","positive_recall")]},"rq2":{"question":"Do soft labels help?","evidence":"E7-E6","assessment":evidence("E7-E6"),"caveat":"Soft vote fractions are descriptive annotator fractions, not ground-truth probabilities.","bootstrap_intervals":[intervals[("E7-E6",m)] for m in ("macro_f1","positive_f1","positive_recall")]},"rq3":{"question":"Do calibration and abstention improve reliability?","assessment":"mixed evidence on this locked test","evidence":"E5-E7 uncalibrated versus calibrated metrics and frozen 0.60 selective policy","caveat":"Lower selective risk does not imply safety and may accompany reduced positive-label coverage."},"baseline_comparison":{"comparisons":["E5-E1","E5-E3","E6-E2","E6-E4"],"winner_selected":False},"deployment_ready":False}
        atomic_json(out/"research_question_summary.json",rq)
        report=["# SANDARBH Phase 6 Final Test Results","",f"One-time locked-test evaluation under successor policy `{AUTHORIZED}`. No model winner was selected.","","## Cohort","",f"341 examples: 38 positive-labelled, 303 other; {pf['test_group_count']} Rule B groups.","","## Classification","","| System | Macro-F1 | Positive F1 | Positive recall | Balanced accuracy | Accuracy | ROC-AUC | AP |","|---|---:|---:|---:|---:|---:|---:|---:|"]
        for m in SYSTEMS:
            x=metrics[m];report.append("| "+" | ".join([m]+[("NA" if x.get(k) is None else f"{x[k]:.6f}") for k in ("macro_f1","positive_f1","positive_recall","balanced_accuracy","accuracy","roc_auc","average_precision")])+" |")
        report += ["","## Frozen comparisons","","| Direction | Metric | Difference |","|---|---|---:|"]
        for r in comp:
            difference="NA" if r["difference"] is None else f"{r['difference']:.6f}"
            report.append(f"| {r['direction']} | {r['metric']} | {difference} |")
        report += ["","## Research questions","",f"- RQ1 (context; E6−E5): {rq['rq1']['assessment']} evidence on this locked test.",f"- RQ2 (soft-label training; E7−E6): {rq['rq2']['assessment']} evidence on this locked test. Soft vote fractions are not ground-truth probabilities.","- RQ3 (calibration and abstention): mixed evidence on this locked test; interpret probability metrics and the frozen 0.60 operating point jointly with complete and class-specific coverage.","","## Limitations","","Only 38 positives occur in the test set. Findings are limited to AUTALIC and the frozen partition. Label 0 is not a definitive safe/non-ableist judgment; soft votes are descriptive annotator fractions. Exact-overlap grouping cannot detect every paraphrase. Calibration used 171 examples with 19 positives. Selective prediction may reject positive-labelled examples. No causal, universal-benefit, safety, or deployment-readiness claim is made.",""]
        report_path=ROOT/config["outputs"]["report"];tmp=report_path.with_name(report_path.name+".tmp");tmp.write_text("\n".join(report),encoding="utf-8");os.replace(tmp,report_path)
        output_hashes={n:sha_file(out/n) for n in OUTPUT_NAMES if (out/n).exists()}
        manifest={"schema_version":"6.0.0","status":"COMPLETED","successor_policy_sha256":AUTHORIZED,"test_access_receipt_sha256":sha_file(receipt_path),"config_sha256":config_hash,"evaluation_script_sha256":script_hash,"source_sha256":config["source_sha256"],"partition_fingerprint":config["partition_fingerprint"],"test_examples":341,"hard_positive":38,"hard_other":303,"test_group_count":pf["test_group_count"],"models":pf["model_hashes"],"runtime":{"inference_seconds":inference_seconds,"total_seconds":time.monotonic()-started,"device":"cpu","precision":"float32"},"input_evidence":{"maximum_sequence_length":max(int(r["sequence_length"]) for r in token_manifest),"e6_e7_encoded_inputs_identical":meta["e6_encoded_manifest_sha256"]==meta["e7_encoded_manifest_sha256"],"context_encoded_manifest_sha256":meta["context_encoded_manifest_sha256"]},"row_counts":{"transformer_predictions":len(trans),"baseline_predictions":len(base)},"report_sha256":sha_file(report_path),"output_hashes":output_hashes,"no_retraining":True,"no_winner_selected":True,"raw_text_saved":False}
        atomic_json(out/"run_manifest.json",manifest);atomic_json(out/"run_status.json",{"status":"COMPLETED","policy_status":"FROZEN","test_status":"EVALUATED_ONCE","successor_policy_sha256":AUTHORIZED,"timestamp_utc":utcnow(),"receipt_sha256":sha_file(receipt_path)})
        print(json.dumps({"status":"COMPLETED","test_status":"EVALUATED_ONCE","runtime_seconds":manifest["runtime"]["total_seconds"]},indent=2))
    except Exception as e:
        atomic_json(out/"run_status.json",{"status":"FAILED","policy_status":"FROZEN","test_status":"TEST_ACCESSED","successor_policy_sha256":AUTHORIZED,"timestamp_utc":utcnow(),"error":str(e),"recovery_permitted_only_with_exact_provenance":True});raise

def main():
    p=argparse.ArgumentParser();p.add_argument("--config");p.add_argument("--preflight-only",action="store_true");p.add_argument("--execute-once",action="store_true");p.add_argument("--baseline-worker",action="store_true");a=p.parse_args()
    if a.baseline_worker:return baseline_worker()
    if bool(a.preflight_only)==bool(a.execute_once):raise Phase6Error("Choose exactly one of --preflight-only or --execute-once")
    path=(ROOT/a.config).resolve();config=load_json(path)
    if a.preflight_only:
        result=preflight(config);print(json.dumps({k:v for k,v in result.items() if k!="test_ids"},indent=2))
    else:execute(path,config)
if __name__=="__main__":
    try:main()
    except Phase6Error as e:print(f"BLOCKED: {e}",file=sys.stderr);sys.exit(2)
