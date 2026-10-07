"""Lazy, deterministic inference for the three frozen transformer demonstrations."""
from __future__ import annotations
import hashlib,math,time
from functools import lru_cache
from pathlib import Path
from .input_builder import build_encodings,validate_target
from .result_formatter import format_result
ROOT=Path(__file__).resolve().parents[2]
MODEL_SPECS={
 "E5":{"path":"models/transformers/E5_seed42","sha256":"96c100bfce0c547f385636ecc9f15b6e2772cf7ca3d995442543ec9aecb0ba9e","mode":"target","temperature":1.0237168508133028,"description":"Target-only; hard-label training; seed 42, epoch 3."},
 "E6":{"path":"models/transformers/E6_seed42","sha256":"ff5d477ab875e5d487f1fee9c329c419bdfdaecf50e15da2f5edda3f250c4b49","mode":"context","temperature":0.6020567132971402,"description":"Contextual; hard-label training; seed 42, epoch 2."},
 "E7":{"path":"models/transformers/E7_seed42","sha256":"ea85aef0135587f1390de5e8ed5f6b6ab7f342136bf263184c2ce212ba4e29a2","mode":"context","temperature":0.6154062772305571,"description":"Contextual cost-weighted soft-vote training; seed 42, epoch 2."}}
def sha(path:Path):return hashlib.sha256(path.read_bytes()).hexdigest()
def calibrated_probabilities(logit0:float,logit1:float,temperature:float):
    if not math.isfinite(temperature) or temperature<=0:raise ValueError("Temperature must be positive.")
    a=logit0/temperature;b=logit1/temperature;m=max(a,b);x=math.exp(a-m);y=math.exp(b-m);return x/(x+y),y/(x+y)
def acceptance(confidence:float,threshold:float=.60):return confidence>=threshold
@lru_cache(maxsize=3)
def load_model_bundle(experiment_id:str):
    if experiment_id not in MODEL_SPECS:raise ValueError(f"Unknown experiment: {experiment_id}")
    spec=MODEL_SPECS[experiment_id];directory=ROOT/spec["path"];weight=directory/"model.safetensors"
    if not weight.is_file():raise FileNotFoundError(f"Required frozen checkpoint for {experiment_id} is unavailable.")
    if sha(weight)!=spec["sha256"]:raise RuntimeError(f"Frozen checkpoint hash mismatch for {experiment_id}.")
    import torch
    from transformers import AutoModelForSequenceClassification,AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(directory,local_files_only=True,trust_remote_code=False)
    model=AutoModelForSequenceClassification.from_pretrained(directory,local_files_only=True,trust_remote_code=False).to(torch.device("cpu"));model.float();model.eval()
    return tokenizer,model
def predict(experiment_id:str,preceding:str,target:str,following:str,loader=load_model_bundle):
    validate_target(target)
    if experiment_id not in MODEL_SPECS:raise ValueError(f"Unknown experiment: {experiment_id}")
    import torch
    started=time.perf_counter();tokenizer,model=loader(experiment_id);encoded,manifest,_=build_encodings(tokenizer,preceding,target,following);spec=MODEL_SPECS[experiment_id];record=encoded[spec["mode"]]
    ids=torch.tensor([record["input_ids"]],dtype=torch.long);mask=torch.tensor([record["attention_mask"]],dtype=torch.long);model.eval()
    with torch.inference_mode():logits=model(input_ids=ids,attention_mask=mask).logits[0].cpu().tolist()
    p0,p1=calibrated_probabilities(float(logits[0]),float(logits[1]),spec["temperature"]);pred=int(p1>=.5);confidence=max(p0,p1)
    return format_result({"experiment_id":experiment_id,"description":spec["description"],"context_used":spec["mode"]=="context","temperature":spec["temperature"],"predicted_label":pred,"probability_0":p0,"probability_1":p1,"confidence":confidence,"accepted":acceptance(confidence),"threshold":.60,"sequence_length":len(record["input_ids"]),"inference_seconds":time.perf_counter()-started})
