#!/usr/bin/env python3
"""Independent, read-only Phase 6 validator; never loads a model or performs inference."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,random,sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; AUTH="b5765e9431a4bf01e8abfeeec1762061b1c0a26eec4ea26c326abb7245ac7541"
SYSTEMS=("E1","E2","E3","E4","E5","E6","E7","DUMMY"); TRANS=("E5","E6","E7"); COMPS=(("E6","E5"),("E7","E6"),("E5","E1"),("E5","E3"),("E6","E2"),("E6","E4"))
def loadj(p):
    with p.open(encoding="utf-8") as f:return json.load(f)
def rows(p):
    with p.open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def sha(p):
    h=hashlib.sha256();h.update(p.read_bytes());return h.hexdigest()
def ch(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def val(x):return None if x in (None,"") else float(x)
def close(a,b,t=2e-9):return a is None and b is None or a is not None and b is not None and math.isclose(float(a),float(b),rel_tol=t,abs_tol=t)
def div(a,b):return a/b if b else 0.0
def cls(y,p):
    tn=sum(a==0 and b==0 for a,b in zip(y,p));fp=sum(a==0 and b==1 for a,b in zip(y,p));fn=sum(a==1 and b==0 for a,b in zip(y,p));tp=sum(a==1 and b==1 for a,b in zip(y,p));p1=div(tp,tp+fp);r1=div(tp,tp+fn);p0=div(tn,tn+fn);r0=div(tn,tn+fp);f1=div(2*p1*r1,p1+r1);f0=div(2*p0*r0,p0+r0)
    return {"macro_f1":(f0+f1)/2,"positive_precision":p1,"positive_recall":r1,"positive_f1":f1,"negative_precision":p0,"negative_recall":r0,"negative_f1":f0,"accuracy":div(tn+tp,len(y)),"balanced_accuracy":(r0+r1)/2,"confusion_matrix_0_1":[[tn,fp],[fn,tp]],"support_0_1":[tn+fp,fn+tp],"predicted_positive_count":sum(p)}
def auc(y,s):
    pos=sum(y);neg=len(y)-pos
    if not pos or not neg:return None
    order=sorted(range(len(s)),key=lambda i:s[i]);rank=[0.]*len(s);i=0
    while i<len(order):
        j=i+1
        while j<len(order) and s[order[j]]==s[order[i]]:j+=1
        for k in range(i,j):rank[order[k]]=((i+1)+j)/2
        i=j
    return (sum(r for r,a in zip(rank,y) if a)-pos*(pos+1)/2)/(pos*neg)
def ap(y,s):
    if not sum(y):return None
    z=sorted(zip(s,y),reverse=True);tp=fp=0;prev=out=0.;i=0
    while i<len(z):
        score=z[i][0];g=[]
        while i<len(z) and z[i][0]==score:g.append(z[i][1]);i+=1
        tp+=sum(g);fp+=len(g)-sum(g);r=tp/sum(y);out+=(r-prev)*tp/(tp+fp);prev=r
    return out
def pm(y,soft,p):
    q=[min(max(x,1e-15),1-1e-15) for x in p];n=len(y);ece=0
    for b in range(10):
        ix=[i for i,x in enumerate(p) if x>=b/10 and (x<(b+1)/10 or b==9)]
        if ix:ece+=len(ix)/n*abs(sum(p[i] for i in ix)/len(ix)-sum(y[i] for i in ix)/len(ix))
    return {"negative_log_likelihood":-sum(a*math.log(x)+(1-a)*math.log(1-x) for a,x in zip(y,q))/n,"brier_score":sum((x-a)**2 for a,x in zip(y,p))/n,"expected_calibration_error":ece,"roc_auc":auc(y,p),"average_precision":ap(y,p),"soft_cross_entropy":-sum((1-a)*math.log(1-x)+a*math.log(x) for a,x in zip(soft,q))/n,"soft_brier_score":sum((x-a)**2 for a,x in zip(soft,p))/n,"soft_positive_mean_absolute_difference":sum(abs(x-a) for x,a in zip(p,soft))/n}
def pct(v,q):
    if not v:return None
    a=sorted(v);x=(len(a)-1)*q;l=int(x);u=math.ceil(x);return a[l]+(a[u]-a[l])*(x-l)
def main():
    a=argparse.ArgumentParser();a.add_argument("--config",required=True);args=a.parse_args();cfgp=ROOT/args.config;cfg=loadj(cfgp);out=ROOT/cfg["outputs"]["directory"]
    C={}; D={}
    def ck(n,v,d=""):C[n]=bool(v);D[n]=d
    succ=loadj(ROOT/cfg["successor_policy"]);pol=succ["policy"];receipt=loadj(out/"test_access_receipt.json");manifest=loadj(out/"run_manifest.json");status=loadj(out/"run_status.json")
    split=rows(ROOT/cfg["split_manifest"]);test=[r for r in split if r["partition"]=="test"];ids=[r["example_id"] for r in test];idset=set(ids);target={r["example_id"]:r for r in rows(ROOT/cfg["targets"])};groups={r["example_id"]:r["group_id"] for r in test}
    tr=rows(out/"transformer_test_predictions.csv");br=rows(out/"baseline_test_predictions.csv");cr=rows(out/"classification_metrics.csv");pr=rows(out/"probability_metrics.csv");rr=rows(out/"reliability_bins.csv");risk=rows(out/"risk_coverage.csv");comp=rows(out/"paired_comparisons.csv");bi=rows(out/"bootstrap_intervals.csv");bm=loadj(out/"bootstrap_manifest.json");rq=loadj(out/"research_question_summary.json")
    ck("01_successor_policy_canonical_hash",succ.get("canonical_policy_sha256")==AUTH and ch(pol)==AUTH)
    ck("02_superseded_policy_relationship",pol.get("supersedes_policy_sha256")==cfg["superseded_policy_sha256"])
    ck("03_test_access_receipt",receipt.get("status")=="TEST_ACCESSED" and receipt.get("successor_policy_sha256")==AUTH)
    ck("04_phase6_script_config_hashes",receipt.get("config_sha256")==sha(cfgp) and receipt.get("evaluation_script_sha256")==sha(ROOT/"scripts/evaluate_locked_test.py"))
    ck("05_exactly_341_test_ids",len(ids)==341 and len(idset)==341)
    labels={i:int(target[i]["hard_label"]) for i in ids};ck("06_exact_class_counts",sum(labels.values())==38 and sum(1-x for x in labels.values())==303)
    ck("07_transformer_row_count",len(tr)==1023)
    ck("08_baseline_row_count",len(br)==1705)
    allrows=tr+br;counts=defaultdict(list)
    for r in allrows:counts[r["experiment_id"]].append(r)
    ck("09_one_prediction_per_model_id",all(len(v)==341 and len({r["example_id"] for r in v})==341 for v in counts.values()) and set(counts)==set(SYSTEMS))
    ck("10_identical_ids",all({r["example_id"] for r in counts[m]}==idset for m in SYSTEMS))
    ck("11_correct_group_ids",all(r["group_id"]==groups.get(r["example_id"]) for r in allrows))
    ck("12_test_partition_only",all(r["partition"]=="test" for r in allrows))
    ck("13_frozen_labels",all(int(r["true_hard_label"])==labels.get(r["example_id"]) for r in allrows))
    probs_ok=all(all(math.isfinite(float(r[k])) and 0<=float(r[k])<=1 for k in ("probability_0_uncalibrated","probability_1_uncalibrated","probability_0_calibrated","probability_1_calibrated")) and close(float(r["probability_0_uncalibrated"])+float(r["probability_1_uncalibrated"]),1) and close(float(r["probability_0_calibrated"])+float(r["probability_1_calibrated"]),1) for r in tr);ck("14_probability_finiteness_normalization",probs_ok)
    def sm(a,b,t=1):a/=t;b/=t;m=max(a,b);x=math.exp(a-m);y=math.exp(b-m);return x/(x+y),y/(x+y)
    ck("15_logit_probability_reconstruction",all(all(close(x,y) for x,y in zip(sm(float(r["logit_0"]),float(r["logit_1"])),(float(r["probability_0_uncalibrated"]),float(r["probability_1_uncalibrated"])))) for r in tr))
    ck("16_temperature_reconstruction",all(all(close(x,y) for x,y in zip(sm(float(r["logit_0"]),float(r["logit_1"]),float(r["temperature"])),(float(r["probability_0_calibrated"]),float(r["probability_1_calibrated"])))) for r in tr))
    ck("17_argmax_invariance",all(r["predicted_label_uncalibrated"]==r["predicted_label_calibrated"] for r in tr))
    ck("18_e6_e7_encoded_equality_evidence",manifest["input_evidence"].get("e6_e7_encoded_inputs_identical") is True)
    ck("19_sequence_length_limit",int(manifest["input_evidence"].get("maximum_sequence_length",999))<=256)
    ck("20_baseline_score_types",all(r["score_type"]==("native_positive_probability" if r["experiment_id"] in ("E1","E2") else "decision_margin" if r["experiment_id"] in ("E3","E4") else "unavailable") for r in br))
    ck("21_svc_margins_not_probabilities",all(r["probability_0"]==r["probability_1"]=="" and r["continuous_score"]!="" for r in br if r["experiment_id"] in ("E3","E4")))
    ck("22_dummy_unavailable_fields",all(r["continuous_score"]==r["probability_0"]==r["probability_1"]=="" for r in br if r["experiment_id"]=="DUMMY"))
    cm_saved={r["experiment_id"]:r for r in cr};calc={};rank={}
    for m,v in counts.items():
        v=sorted(v,key=lambda r:r["example_id"]);y=[int(r["true_hard_label"]) for r in v];p=[int(r.get("predicted_label") or r.get("predicted_label_calibrated")) for r in v];calc[m]=cls(y,p)
        s=None
        if m in ("E1","E2"):s=[float(r["probability_1"]) for r in v]
        elif m in ("E3","E4"):s=[float(r["continuous_score"]) for r in v]
        elif m in TRANS:s=[float(r["probability_1_calibrated"]) for r in v]
        rank[m]=(auc(y,s),ap(y,s)) if s else (None,None)
    ck("23_classification_metrics",all(all(close(calc[m][k],val(cm_saved[m][k])) for k in ("macro_f1","positive_precision","positive_recall","positive_f1","negative_precision","negative_recall","negative_f1","accuracy","balanced_accuracy")) and json.loads(cm_saved[m]["confusion_matrix_0_1"])==calc[m]["confusion_matrix_0_1"] for m in SYSTEMS))
    ck("24_ranking_metrics",all(close(rank[m][0],val(cm_saved[m]["roc_auc"])) and close(rank[m][1],val(cm_saved[m]["average_precision"])) for m in SYSTEMS))
    precalc={};
    for m in TRANS:
        v=sorted(counts[m],key=lambda r:r["example_id"]);y=[int(r["true_hard_label"]) for r in v];soft=[float(r["soft_positive"]) for r in v]
        for state,key in (("uncalibrated","probability_1_uncalibrated"),("calibrated","probability_1_calibrated")):precalc[(m,state)]=pm(y,soft,[float(r[key]) for r in v])
    psaved={(r["experiment_id"],r["probability_state"]):r for r in pr}
    ck("25_probability_metrics",all(close(precalc[k][n],val(psaved[k][n])) for k in precalc for n in ("negative_log_likelihood","brier_score","expected_calibration_error")))
    ck("26_soft_metrics",all(close(precalc[k][n],val(psaved[k][n])) for k in precalc for n in ("soft_cross_entropy","soft_brier_score","soft_positive_mean_absolute_difference")))
    rel_ok=len(rr)==60
    for r in rr:
        v=sorted(counts[r["experiment_id"]],key=lambda x:x["example_id"]);key="probability_1_"+r["probability_state"];b=int(r["bin_index"]);ix=[i for i,x in enumerate(v) if float(x[key])>=b/10 and (float(x[key])<(b+1)/10 or b==9)];rel_ok &= int(r["count"])==len(ix)
    ck("27_reliability_reconciliation",rel_ok)
    ck("28_complete_selective_grid",len(risk)==150 and all({round(float(r["threshold"]),2) for r in risk if r["experiment_id"]==m}=={i/100 for i in range(50,100)} for m in TRANS))
    ck("29_frozen_060_policy",sum(r["is_frozen_primary"]=="True" for r in risk)==3 and float(cfg["selective_thresholds"]["frozen_primary"])==.60)
    sel_ok=wil_ok=True
    for r in risk:
        v=counts[r["experiment_id"]];t=float(r["threshold"]);ix=[x for x in v if float(x["confidence_calibrated"])>=t];err=sum(int(x["true_hard_label"])!=int(x["predicted_label_calibrated"]) for x in ix);sel_ok &= int(r["accepted_examples"])==len(ix) and close(val(r["selective_risk"]),err/len(ix) if ix else None)
        if ix:
            z=1.959963984540054;p=err/len(ix);d=1+z*z/len(ix);c=(p+z*z/(2*len(ix)))/d;h=z*math.sqrt(p*(1-p)/len(ix)+z*z/(4*len(ix)**2))/d;wil_ok &= close(val(r["wilson_95_lower"]),max(0,c-h)) and close(val(r["wilson_95_upper"]),min(1,c+h))
    ck("30_selective_risk",sel_ok);ck("31_wilson_intervals",wil_ok)
    comp_saved={(r["comparison"],r["metric"]):r for r in comp};ck("32_paired_comparisons",all(any(k[0]==f"{a}-{b}" for k in comp_saved) and all(close(val(r["difference"]),val(r["left_value"])-val(r["right_value"])) for k,r in comp_saved.items() if k[0]==f"{a}-{b}" and r["difference"]!="") for a,b in COMPS))
    ck("33_bootstrap_settings",bm.get("seed")==20261006 and bm.get("replicate_count")==5000 and bm.get("method")=="paired_cluster_bootstrap" and not bm.get("p_values_generated") and not bm.get("resample_until_valid"))
    ref=sorted(counts["E5"],key=lambda r:r["example_id"]);gs=sorted({r["group_id"] for r in ref});gix={g:[i for i,r in enumerate(ref) if r["group_id"]==g] for g in gs};rng=random.Random(20261006);h=hashlib.sha256();vals=defaultdict(list);un=defaultdict(int)
    ordered={m:sorted(counts[m],key=lambda r:r["example_id"]) for m in SYSTEMS}
    for _ in range(5000):
        sample=[rng.randrange(len(gs)) for _ in gs];h.update((json.dumps(sample,separators=(",",":"))+"\n").encode());ix=[i for j in sample for i in gix[gs[j]]];rep={}
        for m in SYSTEMS:
            y=[int(ordered[m][i]["true_hard_label"]) for i in ix];p=[int(ordered[m][i].get("predicted_label") or ordered[m][i].get("predicted_label_calibrated")) for i in ix];rep[m]=cls(y,p) if len(set(y))==2 else {}
            names=["macro_f1","positive_f1","positive_recall","balanced_accuracy"]
            for n in names:
                k=("model",m,n);x=rep[m].get(n);un[k]+=x is None
                if x is not None:vals[k].append(x)
            if m in TRANS:
                pp=[float(ordered[m][i]["probability_1_calibrated"]) for i in ix];soft=[float(ordered[m][i]["soft_positive"]) for i in ix];q=pm(y,soft,pp)
                for n in ("negative_log_likelihood","brier_score"):vals[("model",m,n)].append(q[n])
        for x,y in COMPS:
            for n in ("macro_f1","positive_f1","positive_recall"):
                k=("difference",f"{x}-{y}",n);v=rep[x].get(n);w=rep[y].get(n);un[k]+=v is None or w is None
                if v is not None and w is not None:vals[k].append(v-w)
    ck("34_deterministic_cluster_sampling",h.hexdigest()==bm.get("sampled_cluster_indices_sha256"))
    boot_saved={(r["interval_type"],r["subject"],r["metric"]):r for r in bi};recon=all(k in boot_saved and close(pct(v,.025),val(boot_saved[k]["lower_95"])) and close(pct(v,.975),val(boot_saved[k]["upper_95"])) and len(v)==int(boot_saved[k]["valid_replicates"]) and un[k]==int(boot_saved[k]["unavailable_replicates"]) for k,v in vals.items());ck("35_bootstrap_interval_reconstruction",recon)
    ck("36_rq_summary_traceability",rq["rq1"]["evidence"]=="E6-E5" and rq["rq2"]["evidence"]=="E7-E6" and "E5-E7" in rq["rq3"]["evidence"])
    ck("37_no_model_winner",rq["baseline_comparison"]["winner_selected"] is False and manifest["no_winner_selected"] is True)
    ck("38_no_threshold_changes",cfg["selective_thresholds"]=={"start":.5,"end":.99,"step":.01,"frozen_primary":.6})
    immutable=sha(ROOT/cfg["source"])==cfg["source_sha256"] and sha(ROOT/cfg["split_manifest"])==cfg["split_manifest_sha256"] and sha(ROOT/cfg["targets"])==cfg["targets_sha256"]
    for m,s in cfg["baseline_models"].items():immutable &= sha(ROOT/s["path"])==s["sha256"]
    for m,s in cfg["transformers"].items():immutable &= sha(ROOT/s["path"]/"model.safetensors")==s["model_sha256"]
    ck("39_upstream_model_immutability",immutable)
    ck("40_output_hashes",all(sha(out/n)==x for n,x in manifest["output_hashes"].items()))
    forbidden=("preceding","following","target","input_ids","decoded_text","source_text");ck("41_no_raw_text_leakage",all(not any(x in {k.lower() for k in r} for x in forbidden) for r in allrows))
    ck("42_final_one_time_status",status.get("status")=="COMPLETED" and status.get("policy_status")=="FROZEN" and status.get("test_status")=="EVALUATED_ONCE")
    failures=[k for k,v in C.items() if not v];summary={"validator":"scripts/validate_final_test_run.py","read_only":True,"check_count":42,"passed_count":sum(C.values()),"checks":C,"failures":failures,"passed":not failures}
    tmp=out/"validation_summary.json.tmp";tmp.write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8");tmp.replace(out/"validation_summary.json");print(json.dumps(summary,indent=2));return 0 if not failures else 1
if __name__=="__main__":sys.exit(main())
