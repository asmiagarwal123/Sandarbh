#!/usr/bin/env python3
"""Independently validate the saved SANDARBH Phase 4B transformer run."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALLOWED = frozenset({"train", "dev_tune"})
SOURCE_COLUMNS = ("preceding", "target", "following", "A1_Score", "A2_Score", "A3_Score")
SUMMARY_METRICS = (
    "macro_f1", "positive_precision", "positive_recall", "positive_f1", "accuracy", "balanced_accuracy",
    "average_precision", "roc_auc", "negative_log_likelihood", "brier_score", "expected_calibration_error",
    "soft_cross_entropy", "soft_brier_score", "soft_positive_mean_absolute_difference",
)


class ValidationError(Exception):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def hash_ids(ids: Iterable[int]) -> str:
    return sha256_json([int(value) for value in ids])


def record_digest(values: Iterable[str]) -> str:
    return hashlib.sha256(json.dumps(list(values), ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def collapse(text: str) -> str:
    return " ".join(text.split())


def safe_path(relative: str, label: str) -> Path:
    path = (PROJECT_ROOT / relative).resolve()
    try:
        path.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValidationError(f"{label} escapes project root") from exc
    return path


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"Cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValidationError(f"{label} is not an object")
    return value


def read_csv(path: Path, label: str) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise ValidationError(f"Invalid header in {label}")
            return list(reader.fieldnames), list(reader)
    except ValidationError:
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValidationError(f"Cannot read {label}: {exc}") from exc


def average_precision(y_true: list[int], scores: list[float]) -> float:
    positives = sum(y_true)
    ordered = sorted(zip(scores, y_true), key=lambda pair: pair[0], reverse=True)
    tp = fp = 0
    previous_recall = result = 0.0
    index = 0
    while index < len(ordered):
        score = ordered[index][0]
        group = []
        while index < len(ordered) and ordered[index][0] == score:
            group.append(ordered[index][1]); index += 1
        tp += sum(group); fp += len(group) - sum(group)
        recall = tp / positives
        result += (recall - previous_recall) * tp / (tp + fp)
        previous_recall = recall
    return result


def roc_auc(y_true: list[int], scores: list[float]) -> float:
    positives, negatives = sum(y_true), len(y_true) - sum(y_true)
    order = sorted(range(len(scores)), key=lambda index: scores[index])
    ranks = [0.0] * len(scores)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and scores[order[end]] == scores[order[index]]:
            end += 1
        rank = ((index + 1) + end) / 2.0
        for position in range(index, end): ranks[order[position]] = rank
        index = end
    rank_sum = sum(rank for rank, label in zip(ranks, y_true) if label == 1)
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def metrics(rows: list[dict[str, str]], bins: int) -> dict[str, Any]:
    y_true = [int(row["true_hard_label"]) for row in rows]
    y_pred = [int(row["predicted_label"]) for row in rows]
    probability = [float(row["probability_1"]) for row in rows]
    soft = [float(row["soft_positive"]) for row in rows]
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in probability):
        raise ValidationError("Invalid saved probability")
    tn = sum(t == 0 and p == 0 for t, p in zip(y_true, y_pred)); fp = sum(t == 0 and p == 1 for t, p in zip(y_true, y_pred))
    fn = sum(t == 1 and p == 0 for t, p in zip(y_true, y_pred)); tp = sum(t == 1 and p == 1 for t, p in zip(y_true, y_pred))
    support = [tn + fp, fn + tp]
    p1 = tp / (tp + fp) if tp + fp else 0.0; r1 = tp / support[1] if support[1] else 0.0
    f1 = 2 * p1 * r1 / (p1 + r1) if p1 + r1 else 0.0
    p0 = tn / (tn + fn) if tn + fn else 0.0; r0 = tn / support[0] if support[0] else 0.0
    f0 = 2 * p0 * r0 / (p0 + r0) if p0 + r0 else 0.0
    clipped = [min(max(value, 1e-15), 1 - 1e-15) for value in probability]
    nll = -sum(y * math.log(p) + (1-y) * math.log(1-p) for y, p in zip(y_true, clipped)) / len(rows)
    brier = sum((p-y)**2 for y,p in zip(y_true, probability)) / len(rows)
    ece = 0.0
    for i in range(bins):
        low, high = i/bins, (i+1)/bins
        members = [j for j,p in enumerate(probability) if p >= low and (p < high or (i == bins-1 and p <= high))]
        if members:
            ece += len(members)/len(rows) * abs(sum(probability[j] for j in members)/len(members) - sum(y_true[j] for j in members)/len(members))
    soft_ce = -sum((1-s)*math.log(1-p)+s*math.log(p) for s,p in zip(soft,clipped))/len(rows)
    result = {
        "macro_f1": (f0+f1)/2, "positive_precision": p1, "positive_recall": r1, "positive_f1": f1,
        "accuracy": (tn+tp)/len(rows), "balanced_accuracy": (r0+r1)/2,
        "confusion_matrix_0_1": [[tn,fp],[fn,tp]], "support_0_1": support,
        "average_precision": average_precision(y_true,probability), "roc_auc": roc_auc(y_true,probability),
        "predicted_positive_count": sum(y_pred), "no_positive_predictions": sum(y_pred)==0,
        "negative_log_likelihood": nll, "brier_score": brier, "expected_calibration_error": ece,
        "soft_cross_entropy": soft_ce, "soft_brier_score": sum((p-s)**2 for s,p in zip(soft,probability))/len(rows),
        "soft_positive_mean_absolute_difference": sum(abs(p-s) for s,p in zip(soft,probability))/len(rows),
    }
    return result


def close(left: Any, right: Any, tolerance: float = 1e-10) -> bool:
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def reconstruct_rows(config: dict[str, Any], tokenizer: Any) -> tuple[list[dict[str, Any]], str, dict[str, int]]:
    split_path = safe_path(config["partition_artifacts"]["split_manifest"], "split")
    targets_path = safe_path(config["partition_artifacts"]["targets"], "targets")
    _, split = read_csv(split_path, "split")
    _, targets = read_csv(targets_path, "targets")
    selected = [row for row in split if row["partition"] in ALLOWED]
    desired = {row["example_id"] for row in selected}
    source: dict[str, dict[str,str]] = {}
    with safe_path(config["input_csv"], "source").open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, strict=True); header = next(reader)
        if tuple(header) != SOURCE_COLUMNS: raise ValidationError("Source header mismatch")
        for source_row, fields in enumerate(reader,1):
            example_id=f"row-{source_row:06d}-{record_digest(fields)[:12]}"
            if example_id in desired:
                raw=dict(zip(header,fields)); source[example_id]={field:collapse(raw[field]) for field in ("preceding","target","following")}
    target_by={row["example_id"]:row for row in targets}
    c=config["input_construction"]; markers={k:tokenizer(v,add_special_tokens=False,truncation=False)["input_ids"] for k,v in c["markers"].items()}
    ellipsis=tokenizer(c["overlong_target_ellipsis"],add_special_tokens=False,truncation=False)["input_ids"]
    budget=256-2-sum(len(v) for v in markers.values())
    if budget != 241: raise ValidationError("Recalculated target budget is not 241")
    manifest=[]; context_signature=[]
    for split_row in sorted(selected,key=lambda r:(r["partition"],r["example_id"])):
        eid=split_row["example_id"]; texts=source[eid]
        target=tokenizer(texts["target"],add_special_tokens=False,truncation=False)["input_ids"]
        pre=tokenizer(texts["preceding"],add_special_tokens=False,truncation=False)["input_ids"]
        fol=tokenizer(texts["following"],add_special_tokens=False,truncation=False)["input_ids"]
        if len(target)<=budget:
            retained=list(target); beginning=len(target); ellipsis_count=ending=0; truncated=False
        else:
            capacity=budget-len(ellipsis); beginning=(capacity+1)//2; ending=capacity//2
            retained=list(target[:beginning])+list(ellipsis)+list(target[-ending:]); ellipsis_count=len(ellipsis); truncated=True
        capacity=256-2-sum(len(v) for v in markers.values())-len(retained)
        pa=(capacity+1)//2; fa=capacity//2
        if len(pre)<pa: fa += pa-len(pre); pa=len(pre)
        if len(fol)<fa: pa += fa-len(fol); fa=len(fol)
        pre_r=list(pre[-min(len(pre),pa):] if min(len(pre),pa) else []); fol_r=list(fol[:min(len(fol),fa)])
        sequences={
            "target":[tokenizer.bos_token_id]+list(markers["target"])+retained+[tokenizer.eos_token_id],
            "context":[tokenizer.bos_token_id]+list(markers["preceding"])+pre_r+list(markers["target"])+retained+list(markers["following"])+fol_r+[tokenizer.eos_token_id],
        }
        for mode,sequence in sequences.items():
            row={
                "example_id":eid,"partition":split_row["partition"],"input_mode":mode,"sequence_length":len(sequence),
                "target_original_tokens":len(target),"target_retained_tokens":len(retained),"target_beginning_tokens":beginning,
                "target_ellipsis_tokens":ellipsis_count,"target_ending_tokens":ending,"target_truncated":str(truncated),
                "preceding_original_tokens":len(pre),"preceding_retained_tokens":len(pre_r) if mode=="context" else 0,
                "following_original_tokens":len(fol),"following_retained_tokens":len(fol_r) if mode=="context" else 0,
                "input_ids_sha256":hash_ids(sequence),"attention_mask_sha256":hash_ids([1]*len(sequence)),
                "target_original_sha256":hash_ids(target),"target_retained_sha256":hash_ids(retained),
            }
            manifest.append(row)
            if mode=="context": context_signature.append({k:row[k] for k in ("example_id","partition","input_ids_sha256","attention_mask_sha256")})
    manifest.sort(key=lambda r:(r["partition"],r["example_id"],r["input_mode"]))
    counts=Counter(int(target_by[row["example_id"]]["hard_label"]) for row in selected if row["partition"]=="train")
    return manifest,sha256_json(context_signature),dict(counts)


def checkpoint_hashes(directory: Path) -> dict[str,str]:
    files=sorted(path for path in directory.rglob("*") if path.is_file())
    return {str(path.relative_to(directory)).replace("\\","/"):sha256_file(path) for path in files}


def validate(config_path: Path, quiet: bool=False) -> dict[str,Any]:
    checks:dict[str,bool]={}; failures:list[str]=[]
    try:
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        config=load_json(config_path,"training config")
        outputs={k:safe_path(v,f"output {k}") for k,v in config["outputs"].items() if k not in {"directory","units_directory","checkpoint_root"}}
        manifest=load_json(outputs["run_manifest"],"run manifest"); status=load_json(outputs["run_status"],"run status")
        checks["run_completed"]=manifest.get("status")==status.get("status")=="COMPLETED" and manifest.get("completed_unit_count")==9
        upstream={"source":safe_path(config["input_csv"],"source")}
        upstream.update({f"partition_{k}":safe_path(v,k) for k,v in config["partition_artifacts"].items()})
        upstream.update({f"baseline_{k}":safe_path(v,k) for k,v in config["baseline_artifacts"].items()})
        upstream["preflight_config"]=safe_path(config["preflight_artifacts"]["config"],"preflight config")
        current_hashes={k:sha256_file(v) for k,v in upstream.items()}
        checks["upstream_artifacts_unchanged"]=current_hashes==manifest.get("protected_upstream_hashes_before")==manifest.get("upstream_hashes")
        checks["source_partition_preflight_references_match"]=current_hashes["source"]==config["reference"]["source_sha256"] and current_hashes["partition_targets"]==config["reference"]["targets_sha256"] and current_hashes["partition_split_manifest"]==config["reference"]["split_manifest_sha256"] and current_hashes["preflight_config"]==config["reference"]["preflight_config_sha256"]
        preflight=load_json(upstream["preflight_config"],"preflight config")
        checks["preflight_and_model_revision_match"]=preflight.get("model",{}).get("revision")==config["model"]["revision"]==manifest.get("model",{}).get("revision")
        checks["phase4a_external_validation_receipts_recorded"]=manifest.get("phase4a_external_validation_receipts")=={key:value for key,value in config["reference"].items() if key.startswith("preflight_")}
        code_hashes={"training_config":sha256_file(config_path),"training_script":sha256_file(PROJECT_ROOT/"scripts"/"train_transformers.py"),"validation_script":sha256_file(Path(__file__)),"tests":sha256_file(PROJECT_ROOT/"tests"/"test_transformer_training.py"),"protocol":sha256_file(PROJECT_ROOT/"reports"/"PHASE_04B_TRANSFORMER_PROTOCOL.md"),"requirements":sha256_file(PROJECT_ROOT/"requirements-transformers.txt")}
        checks["code_config_hashes_match"]=code_hashes==manifest.get("code_and_config_hashes")
        snapshot=Path(snapshot_download(repo_id=config["model"]["repository"],revision=config["model"]["revision"],allow_patterns=["config.json","model.safetensors","tokenizer.json","tokenizer_config.json","vocab.json","merges.txt"],local_files_only=True))
        tokenizer=AutoTokenizer.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False,use_fast=True)
        reconstructed,context_hash,class_counts=reconstruct_rows(config,tokenizer)
        fields,saved_tokens=read_csv(outputs["tokenization_manifest"],"tokenization manifest")
        normalized=[{key:str(value) for key,value in row.items()} for row in reconstructed]
        checks["tokenization_manifest_exact"]=saved_tokens==normalized and all(int(row["sequence_length"])<=256 for row in saved_tokens)
        checks["e6_e7_encodings_identical"]=context_hash==manifest.get("input_construction",{}).get("e6_encoded_manifest_sha256")==manifest.get("input_construction",{}).get("e7_encoded_manifest_sha256") and manifest.get("e6_e7_encoded_inputs_identical") is True
        checks["class_weights_match"]=class_counts=={0:1415,1:180} and all(close(value,1595/(2*class_counts[index])) for index,value in enumerate(manifest.get("class_weights_0_1",[])))
        units_dir=safe_path(config["outputs"]["units_directory"],"units")
        units=[]; all_epochs=[]
        for experiment in ("E5","E6","E7"):
            for seed in (42,43,44):
                unit_dir=units_dir/f"{experiment}_seed{seed}"; unit=load_json(unit_dir/"unit.json","unit")
                _,epochs=read_csv(unit_dir/"epoch_metrics.csv","unit epochs"); _,preds=read_csv(unit_dir/"predictions.csv","unit predictions")
                expected_selected=min(epochs,key=lambda row:(-float(row["macro_f1"]),int(row["epoch"])))
                if len(epochs)!=3 or len(preds)!=171 or int(unit["selected_epoch"])!=int(expected_selected["epoch"]): raise ValidationError(f"Invalid epochs/selection for {experiment} seed {seed}")
                if unit.get("partitions_accessed")!=["train","dev_tune"] or unit.get("train_count")!=1595 or unit.get("dev_tune_count")!=171: raise ValidationError(f"Invalid access/count for {experiment} seed {seed}")
                calculated=metrics(preds,config["calibration_description"]["ece_bin_count"]); selected=unit["selected_checkpoint_metrics"]
                for key,value in calculated.items():
                    if isinstance(value,list):
                        if value!=selected[key]: raise ValidationError(f"Metric mismatch {key} for {experiment} seed {seed}")
                    elif isinstance(value,bool):
                        if value is not selected[key]: raise ValidationError(f"Metric mismatch {key} for {experiment} seed {seed}")
                    elif not close(value,selected[key]): raise ValidationError(f"Metric mismatch {key} for {experiment} seed {seed}")
                required_loss="weighted_soft_cross_entropy" if experiment=="E7" else "weighted_hard_cross_entropy"
                if unit.get("loss_definition")!=required_loss or unit.get("class_weights_0_1")!=manifest.get("class_weights_0_1"): raise ValidationError(f"Loss/weights mismatch {experiment} seed {seed}")
                checkpoint=unit.get("checkpoint")
                checkpoint_path=safe_path(f"{config['outputs']['checkpoint_root']}/{experiment}_seed{seed}","checkpoint")
                if seed==42:
                    if checkpoint is None or checkpoint_hashes(checkpoint_path)!=checkpoint["files"]: raise ValidationError(f"Checkpoint hash mismatch {experiment} seed42")
                    AutoModelForSequenceClassification.from_pretrained(checkpoint_path,local_files_only=True,trust_remote_code=False,use_safetensors=True)
                elif checkpoint is not None or checkpoint_path.exists(): raise ValidationError(f"Robustness weights retained for {experiment} seed {seed}")
                units.append(unit); all_epochs.extend(epochs)
        checks["exactly_nine_valid_units_three_epochs"] = len(units)==9 and len(all_epochs)==27
        checks["primary_seed_fixed_and_robustness_weights_absent"] = all(unit["primary_seed"]==(unit["seed"]==42) for unit in units)
        _,aggregate_epochs=read_csv(outputs["epoch_metrics"],"aggregate epochs"); _,aggregate_selected=read_csv(outputs["selected_checkpoint_metrics"],"aggregate selected"); _,aggregate_predictions=read_csv(outputs["dev_tune_predictions"],"aggregate predictions")
        checks["aggregate_row_counts"] = len(aggregate_epochs)==27 and len(aggregate_selected)==9 and len(aggregate_predictions)==1539 and all(row["partition"]=="dev_tune" for row in aggregate_predictions)
        selected_records=[unit["selected_checkpoint_metrics"] for unit in units]
        expected_summary={"description":"Dev_tune selected-checkpoint descriptive summary; sample standard deviation uses n-1.","experiments":{}}
        for exp in ("E5","E6","E7"):
            recs=sorted([r for r in selected_records if r["experiment_id"]==exp],key=lambda r:int(r["seed"])); metric_summary={}
            for name in SUMMARY_METRICS:
                values=[float(r[name]) for r in recs]; metric_summary[name]={"mean":statistics.mean(values),"sample_standard_deviation":statistics.stdev(values),"minimum":min(values),"maximum":max(values)}
            expected_summary["experiments"][exp]={"seeds":[42,43,44],"metrics":metric_summary}
        checks["seed_summary_reconciles"] = load_json(outputs["seed_summary"],"seed summary")==expected_summary
        by={(r["experiment_id"],int(r["seed"])):r for r in selected_records}
        expected_comparisons={"description":"Paired dev_tune differences are descriptive only; no significance tests or winner selection.","difference_direction":{"E6_minus_E5":"context hard minus target-only hard","E7_minus_E6":"context soft minus context hard"},"comparisons":{}}
        for name,left,right in (("E6_minus_E5","E6","E5"),("E7_minus_E6","E7","E6")):
            expected_comparisons["comparisons"][name]=[{"seed":seed,"differences":{metric:float(by[(left,seed)][metric])-float(by[(right,seed)][metric]) for metric in SUMMARY_METRICS}} for seed in (42,43,44)]
        checks["comparisons_reconcile"] = load_json(outputs["comparisons"],"comparisons")==expected_comparisons
        named={key:outputs[key] for key in ("tokenization_manifest","epoch_metrics","selected_checkpoint_metrics","dev_tune_predictions","seed_summary","comparisons")}
        checks["output_hashes_match"] = all(manifest.get("output_hashes",{}).get(key)==sha256_file(path) for key,path in named.items())
        checks["no_forbidden_partition_or_later_stage"] = manifest.get("partitions_accessed")==["train","dev_tune"] and all(term in manifest.get("not_performed",[]) for term in ("dev_calibration inference","test inference","probability calibration","selective prediction"))
        checks["text_free_outputs"] = not any(key in fields for key in ("text","decoded_text","tokens","input_ids")) and all(not any(key in row for key in ("text","decoded_text","tokens","input_ids")) for row in aggregate_predictions)
        for name,passed in checks.items():
            if not passed: failures.append(f"Check failed: {name}")
    except (ValidationError,OSError,KeyError,ValueError,TypeError) as exc:
        failures.append(str(exc))
    result={"passed":not failures,"checks":checks,"failures":failures}
    if not quiet: print(json.dumps(result,indent=2,sort_keys=True))
    return result


def main(argv:list[str]|None=None)->int:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--config",default="configs/transformer_training.json"); args=parser.parse_args(argv)
    return 0 if validate(safe_path(args.config,"config"))["passed"] else 1


if __name__=="__main__":
    raise SystemExit(main())
