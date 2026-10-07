#!/usr/bin/env python3
"""Build text-free Phase 7 tables, figures, final report, and summary from validated Phase 6 outputs."""
from __future__ import annotations
import csv,hashlib,json,os,sys
from datetime import datetime,timezone
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[1];P6=ROOT/"results/final_test";FINAL=ROOT/"results/final";TABLES=FINAL/"tables";FIGURES=ROOT/"reports/figures"
POLICY="b5765e9431a4bf01e8abfeeec1762061b1c0a26eec4ea26c326abb7245ac7541"
COLORS={"DUMMY":"#777777","E1":"#4477AA","E2":"#66CCEE","E3":"#228833","E4":"#CCBB44","E5":"#EE6677","E6":"#AA3377","E7":"#BBBBBB"}
def sha(p:Path):return hashlib.sha256(p.read_bytes()).hexdigest()
def loadj(p):
    with p.open(encoding="utf-8") as f:return json.load(f)
def read(p):
    with p.open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def atomic_text(p:Path,s:str):p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+".tmp");q.write_text(s,encoding="utf-8",newline="\n");os.replace(q,p)
def atomic_json(p,v):atomic_text(p,json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
def write(p,fields,rows):
    p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+".tmp")
    with q.open("w",encoding="utf-8",newline="") as f:w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n",extrasaction="ignore");w.writeheader();w.writerows(rows)
    os.replace(q,p)
def f(x):return None if x in (None,"") else float(x)
def md(x):return "NA" if x is None else f"{float(x):.4f}"
def table(headers,rows):return "\n".join(["| "+" | ".join(headers)+" |","|"+"|".join(["---"]+ ["---:"]*(len(headers)-1))+"|"]+["| "+" | ".join(map(str,r))+" |" for r in rows])
def savefig(fig,name):
    png=FIGURES/f"{name}.png";svg=FIGURES/f"{name}.svg";fig.savefig(png,dpi=220,bbox_inches="tight",facecolor="white");fig.savefig(svg,bbox_inches="tight",facecolor="white");return [png,svg]
def main():
    status=loadj(P6/"run_status.json");validation=loadj(P6/"validation_summary.json")
    if status.get("status")!="COMPLETED" or status.get("test_status")!="EVALUATED_ONCE" or not validation.get("passed") or validation.get("passed_count")!=42:raise RuntimeError("Phase 6 is not completed and validated 42/42")
    c=read(P6/"classification_metrics.csv");p=read(P6/"probability_metrics.csv");risk=read(P6/"risk_coverage.csv");comp=read(P6/"paired_comparisons.csv");boot=read(P6/"bootstrap_intervals.csv")
    cmap={r["experiment_id"]:r for r in c};pmap={(r["experiment_id"],r["probability_state"]):r for r in p};bmap={(r["interval_type"],r["subject"],r["metric"]):r for r in boot}
    split=read(ROOT/"results/partitions/split_manifest.csv");targets={r["example_id"]:r for r in read(ROOT/"results/partitions/targets.csv")}
    dataset=[]
    for part in ("train","dev_tune","dev_calibration","test"):
        rs=[r for r in split if r["partition"]==part];pos=sum(int(targets[r["example_id"]]["hard_label"]) for r in rs);dataset.append({"partition":part,"examples":len(rs),"hard_positive":pos,"hard_other":len(rs)-pos,"rule_b_groups":len({r["group_id"] for r in rs})})
    write(TABLES/"dataset_partition_summary.csv",list(dataset[0]),dataset)
    fields=list(c[0]);write(TABLES/"baseline_results.csv",fields,[r for r in c if r["experiment_id"] in ("DUMMY","E1","E2","E3","E4")]);write(TABLES/"transformer_results.csv",fields,[r for r in c if r["experiment_id"] in ("E5","E6","E7")])
    write(TABLES/"calibration_results.csv",list(p[0]),p);sel=[r for r in risk if float(r["threshold"])==.6];write(TABLES/"selective_prediction_060.csv",list(risk[0]),sel);write(TABLES/"frozen_paired_comparisons.csv",list(comp[0]),comp);write(TABLES/"bootstrap_intervals.csv",list(boot[0]),boot)
    evidence=[]
    for rq,contrast,statement in (("RQ1","E6-E5","Transformer context evidence is mixed; positive recall decreased."),("RQ2","E7-E6","Soft-target training increased recall but worsened several probability metrics.")):
        for metric in ("macro_f1","positive_f1","positive_recall"):
            x=bmap[("difference",contrast,metric)];evidence.append({"research_question":rq,"contrast":contrast,"metric":metric,"point_estimate":x["point_estimate"],"lower_95":x["lower_95"],"upper_95":x["upper_95"],"interpretation":statement})
    for m in ("E5","E6","E7"):
        for metric in ("negative_log_likelihood","brier_score","expected_calibration_error"):
            a=f(pmap[(m,"uncalibrated")][metric]);b=f(pmap[(m,"calibrated")][metric]);evidence.append({"research_question":"RQ3","contrast":f"{m} calibrated-uncalibrated","metric":metric,"point_estimate":b-a,"lower_95":"","upper_95":"","interpretation":"Negative values favour calibration for this loss/error metric."})
    write(TABLES/"research_question_evidence.csv",list(evidence[0]),evidence)
    import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt;import numpy as np
    plt.rcParams.update({"font.size":10,"axes.titlesize":13,"axes.labelsize":11});FIGURES.mkdir(parents=True,exist_ok=True);figs=[];systems=["DUMMY","E1","E2","E3","E4","E5","E6","E7"]
    fig,ax=plt.subplots(figsize=(9,5));ax.bar(systems,[f(cmap[m]["macro_f1"]) for m in systems],color=[COLORS[m] for m in systems]);ax.set_ylim(0,1);ax.set_ylabel("Macro-F1");ax.set_title("Full-coverage macro-F1 on the locked test (n=341; positives=38)");ax.grid(axis="y",alpha=.25);figs+=savefig(fig,"01_macro_f1_by_experiment");plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5));x=np.arange(len(systems));w=.38;ax.bar(x-w/2,[f(cmap[m]["positive_f1"]) for m in systems],w,label="Positive F1",color="#4477AA");ax.bar(x+w/2,[f(cmap[m]["positive_recall"]) for m in systems],w,label="Positive recall",color="#EE6677");ax.set_xticks(x,systems);ax.set_ylim(0,1);ax.legend();ax.set_title("Positive-class performance on the locked test (38 positives)");ax.grid(axis="y",alpha=.25);figs+=savefig(fig,"02_positive_f1_recall");plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=False);metrics=(("negative_log_likelihood","NLL"),("brier_score","Brier"),("expected_calibration_error","ECE"))
    for ax,(metric,label) in zip(axes,metrics):
        x=np.arange(3);ax.bar(x-.2,[f(pmap[(m,"uncalibrated")][metric]) for m in ("E5","E6","E7")],.4,label="Before",color="#999999");ax.bar(x+.2,[f(pmap[(m,"calibrated")][metric]) for m in ("E5","E6","E7")],.4,label="After",color="#4477AA");ax.set_xticks(x,("E5","E6","E7"));ax.set_ylim(bottom=0);ax.set_title(label);ax.grid(axis="y",alpha=.2)
    axes[0].legend();fig.suptitle("Temperature calibration on the locked test (lower is better)");figs+=savefig(fig,"03_calibration_before_after");plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,6))
    for m in ("E5","E6","E7"):
        rs=sorted((r for r in risk if r["experiment_id"]==m),key=lambda r:f(r["coverage"]));ax.plot([f(r["coverage"]) for r in rs],[f(r["selective_risk"]) for r in rs],label=m,color=COLORS[m],linewidth=2)
    ax.set_xlim(0,1);ax.set_ylim(0,1);ax.set_xlabel("Coverage");ax.set_ylabel("Selective risk");ax.set_title("Frozen coverage–risk curves; low coverage is not a safety guarantee");ax.legend();ax.grid(alpha=.25);figs+=savefig(fig,"04_coverage_risk_curves");plt.close(fig)
    diffs=[r for r in boot if r["interval_type"]=="difference" and r["metric"] in ("macro_f1","positive_f1","positive_recall")];labels=[f"{r['subject']} · {r['metric'].replace('_',' ')}" for r in diffs];points=np.array([f(r["point_estimate"]) for r in diffs]);lo=np.array([f(r["lower_95"]) for r in diffs]);hi=np.array([f(r["upper_95"]) for r in diffs]);y=np.arange(len(diffs));fig,ax=plt.subplots(figsize=(10,9));ax.errorbar(points,y,xerr=[points-lo,hi-points],fmt="o",color="#334455",ecolor="#7799AA",capsize=3);ax.axvline(0,color="black",linewidth=1);ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlabel("Left-minus-right difference with percentile 95% interval");ax.set_title("Frozen paired Rule B cluster-bootstrap comparisons");ax.grid(axis="x",alpha=.25);figs+=savefig(fig,"05_paired_comparison_intervals");plt.close(fig)
    fig,ax=plt.subplots(figsize=(14,4));ax.axis("off");nodes=[("Audit & overlap",.07),("Frozen partitions",.24),("E1–E7 training",.41),("Calibration & policy",.59),("One-time test",.77),("Phase 7 & UI",.94)]
    for label,x0 in nodes:ax.text(x0,.5,label,fontsize=9,ha="center",va="center",bbox=dict(boxstyle="round,pad=.42",facecolor="#E7F1F8",edgecolor="#4477AA"),transform=ax.transAxes)
    for (_,a),(_,b) in zip(nodes,nodes[1:]):ax.annotate("",xy=(b-.075,.5),xytext=(a+.075,.5),xycoords=ax.transAxes,arrowprops=dict(arrowstyle="->",color="#555555"))
    ax.set_title("SANDARBH leakage-controlled research workflow",pad=20);figs+=savefig(fig,"06_methodology_flow");plt.close(fig)
    class_rows=[]
    for m in systems:
        x=cmap[m];class_rows.append([m,md(f(x["macro_f1"])),md(f(x["positive_precision"])),md(f(x["positive_recall"])),md(f(x["positive_f1"])),md(f(x["accuracy"])),md(f(x["balanced_accuracy"])),md(f(x["roc_auc"])),md(f(x["average_precision"]))])
    prob_rows=[]
    for x in p:prob_rows.append([x["experiment_id"],x["probability_state"],md(f(x["negative_log_likelihood"])),md(f(x["brier_score"])),md(f(x["expected_calibration_error"])),md(f(x["soft_brier_score"]))])
    selrows=[]
    for x in sel:selrows.append([x["experiment_id"],x["accepted_examples"],md(f(x["coverage"])),md(f(x["selective_risk"])),md(f(x["positive_label_coverage"])),md(f(x["negative_label_coverage"]))])
    e65={r["metric"]:f(r["difference"]) for r in comp if r["comparison"]=="E6-E5"};e76={r["metric"]:f(r["difference"]) for r in comp if r["comparison"]=="E7-E6"}
    report=f"""# SANDARBH: Final Research Report

## 1. Plain-language abstract

SANDARBH studies whether conversational context, soft annotation targets, probability calibration, and confidence-based abstention change binary detection behaviour on AUTALIC. All decisions were frozen before a single locked-test evaluation. The test contained 341 examples, including 38 positive-labelled examples. Evidence was mixed: context changed the balance between accuracy and positive recall; soft-target training restored recall but worsened several probability-quality measures relative to contextual hard-label training; calibration helped E6 and E7 more consistently than E5; abstention reduced observed error while also excluding positive-labelled examples. These are research findings, not a deployment or safety guarantee.

## 2. Motivation

Target-only classification can miss surrounding context, while contextual models may change false-positive and false-negative patterns. This project evaluates those trade-offs under a leakage-controlled, predeclared protocol.

## 3. Research questions

1. Does conversational context help compared with target-only input?
2. How does soft-vote-fraction training compare with hard-label training under identical contextual input construction?
3. How do frozen temperature calibration and confidence-based abstention affect probability quality, coverage, and risk?

## 4. Dataset description

The frozen modeling set contains {sum(int(x['examples']) for x in dataset)} eligible examples. The locked test contains 341 examples: 303 label 0 and 38 label 1. Source sentences are never reproduced in this report.

## 5. Annotation interpretation

Class 1 means a majority ableist annotation under this benchmark mapping. Class 0 means no majority positive annotation; it does not establish that language is safe or non-ableist. Soft targets are annotator vote fractions, not ground-truth probabilities.

## 6. Eligibility and exclusions

Eligibility and exclusions were fixed during the audit. Excluded rows never entered modeling partitions, and Phase 7 does not revise those decisions.

## 7. Overlap grouping and leakage control

Exact normalized overlap links were consolidated into Rule B groups and kept intact across partitions. This controls detected exact overlap but cannot detect every paraphrase or semantic near-duplicate.

## 8. Frozen partition design

Train, development-tuning, calibration, and test partitions were frozen before model fitting. The test contained 213 Rule B groups, with no group crossing partitions.

## 9. Classical baselines

E1/E2 are saved TF-IDF logistic pipelines; E3/E4 are saved TF-IDF LinearSVC pipelines. E1/E3 use target text and E2/E4 use context. DUMMY always predicts class 0. No baseline was retrained or refitted.

## 10. Transformer architecture

E5–E7 use frozen DistilRoBERTa seed-42 checkpoints. E5 is target-only hard-label training; E6 is contextual hard-label training; E7 is contextual soft-target training. Inference used CPU FP32.

## 11. Target-only versus contextual inputs

Classical context comparisons were mixed: E2−E1 changed macro-F1 by {f(cmap['E2']['macro_f1'])-f(cmap['E1']['macro_f1']):.4f}, and E4−E3 by {f(cmap['E4']['macro_f1'])-f(cmap['E3']['macro_f1']):.4f}. For transformers, E6−E5 changed macro-F1 by {e65['macro_f1']:.4f} but positive recall by {e65['positive_recall']:.4f}; the recall interval was [{f(bmap[('difference','E6-E5','positive_recall')]['lower_95']):.4f}, {f(bmap[('difference','E6-E5','positive_recall')]['upper_95']):.4f}]. Context therefore did not yield uniform improvement across metrics.

## 12. Hard versus soft targets

E7−E6 changed macro-F1 by {e76['macro_f1']:.4f}, positive F1 by {e76['positive_f1']:.4f}, and positive recall by {e76['positive_recall']:.4f}. The positive-recall interval was [{f(bmap[('difference','E7-E6','positive_recall')]['lower_95']):.4f}, {f(bmap[('difference','E7-E6','positive_recall')]['upper_95']):.4f}]. E7 improved ranking metrics but had worse NLL, Brier, ECE, soft cross-entropy, and soft Brier than E6. This is a trade-off, not an unconditional winner.

## 13. Calibration methodology

Scalar temperatures were fitted only on the 171-example calibration partition, which had 19 positives, then frozen. Test logits were divided by the frozen temperature before softmax; every argmax was preserved.

## 14. Selective-prediction methodology

The primary evaluation uses complete coverage. The secondary policy accepts predictions with maximum calibrated probability at least 0.60. Thresholds 0.50–0.99 are descriptive and were frozen before test access.

## 15. Frozen final evaluation policy

The authoritative policy hash is `{POLICY}`. It fixes models, temperatures, thresholds, metrics, comparisons, bootstrap settings, and interpretation constraints.

## 16. Locked-test classification results

{table(['System','Macro-F1','Pos P','Pos R','Pos F1','Accuracy','Balanced accuracy','ROC-AUC','AP'],class_rows)}

## 17. Probability and calibration results

{table(['System','State','NLL','Brier','ECE','Soft Brier'],prob_rows)}

## 18. Selective prediction at 0.60

{table(['System','Accepted','Coverage','Risk','Positive coverage','Negative coverage'],selrows)}

Lower observed selective risk accompanies reduced coverage and must not be interpreted as safety.

## 19. Bootstrap uncertainty

Intervals use 5,000 paired complete-group bootstrap replicates with seed 20261006. They summarize uncertainty for this sample and are not p-values or universal proof.

## 20. RQ1 conclusion — context

Observed evidence is mixed. Classical contextual systems increased macro-F1 relative to their target-only counterparts. E6 slightly increased macro-F1 over E5 but reduced positive recall substantially. Different metrics support different interpretations.

## 21. RQ2 conclusion — soft annotation targets

E7 increased positive recall and ranking quality relative to E6, while lowering accuracy and macro-F1 and worsening multiple calibrated probability-quality metrics. Soft vote fractions must not be treated as ground-truth probabilities.

## 22. RQ3 conclusion — calibration and abstention

Calibration improved E6 and E7 NLL, Brier, and ECE. E5 had small NLL/Brier improvements but slightly worse ECE. At 0.60, E5/E6/E7 coverage was {f(sel[0]['coverage']):.4f}/{f(sel[1]['coverage']):.4f}/{f(sel[2]['coverage']):.4f}, with positive-label coverage {f(sel[0]['positive_label_coverage']):.4f}/{f(sel[1]['positive_label_coverage']):.4f}/{f(sel[2]['positive_label_coverage']):.4f}. Evidence is mixed and does not establish deployment reliability.

## 23. Error, risk, ethics, and prohibited uses

Errors can be harmful in either direction. The system is not diagnostic, clinical, moderation-ready, or suitable for automated punishment. A negative prediction does not make language safe. Human review and broader validation would be essential for any consequential use. Individual test examples were not inspected or exposed.

## 24. Reproducibility and integrity

The test was accessed once under an immutable receipt. Predictions are text-free. The independent Phase 6 validator passed 42/42 checks. No retraining, recalibration, threshold tuning, seed shopping, p-values, post-test winner selection, or policy change occurred.

## 25. Limitations

The test has only 38 positives. Results are limited to AUTALIC and this frozen split. Exact-overlap grouping misses some paraphrases. Calibration used 171 examples with 19 positives. Bootstrap intervals depend on the observed groups. Selective prediction can reject positive-labelled examples. The UI is a local research demonstration and may produce incorrect or harmful predictions.

## 26. Conclusion and future work

SANDARBH provides a reproducible locked evaluation of context, soft targets, calibration, and abstention. Future work should use new external data, broader overlap detection, larger positive-class samples, independent annotation studies, and separately pre-registered evaluations. The locked test must not be retuned or rerun.
"""
    report_path=ROOT/"reports/FINAL_RESEARCH_REPORT.md";atomic_text(report_path,report)
    figure_entries=[{"path":str(x.relative_to(ROOT)).replace("\\","/"),"sha256":sha(x)} for x in figs]
    table_entries=[{"path":str(x.relative_to(ROOT)).replace("\\","/"),"sha256":sha(x)} for x in sorted(TABLES.glob("*.csv"))]
    summary={"schema_version":"7.0.0","status":"COMPLETED","generated_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),"policy_sha256":POLICY,"source_sha256":"66ccaa43f9f3d8694c4826d8b999a4e562041637cad3c8c396590fbe78d3d408","partition_fingerprint":"61c9415766e4f7b342eb37df7b09adf2f89cd86f2818ee05645ae9409dbbba86","experiments":systems,"primary_metrics":{m:{k:f(cmap[m][k]) for k in ("macro_f1","positive_precision","positive_recall","positive_f1","accuracy","balanced_accuracy","roc_auc","average_precision")} for m in systems},"frozen_comparisons":comp,"bootstrap_intervals":boot,"selective_operating_point":{r["experiment_id"]:r for r in sel},"research_questions":{"RQ1":"mixed evidence on this locked test","RQ2":"mixed evidence on this locked test","RQ3":"mixed evidence on this locked test"},"limitations":["38 test positives","AUTALIC and frozen-split scope","class 0 is not definitively safe","soft votes are not ground-truth probabilities","exact-overlap grouping misses some paraphrases","171-example calibration set with 19 positives","abstention can exclude positive-labelled examples","no causal or deployment-readiness claim"],"integrity":{"phase6_validation":"42/42","test_access_once":True,"no_retraining":True,"no_recalibration":True,"no_threshold_tuning":True,"no_winner_selection":True,"no_source_text":True},"final_report":{"path":"reports/FINAL_RESEARCH_REPORT.md","sha256":sha(report_path)},"tables":table_entries,"figures":figure_entries}
    FINAL.mkdir(parents=True,exist_ok=True);atomic_json(FINAL/"final_summary.json",summary);print(json.dumps({"status":"COMPLETED","tables":len(table_entries),"figures":len(figs),"report_sha256":sha(report_path)},indent=2))
if __name__=="__main__":sys.exit(main())
