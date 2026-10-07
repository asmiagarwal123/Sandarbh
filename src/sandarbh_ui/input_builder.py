"""Exact Phase 4B input construction reused without model loading."""
from __future__ import annotations
import importlib.util,json
from pathlib import Path
from typing import Any
ROOT=Path(__file__).resolve().parents[2]
def collapse(text:str)->str:return " ".join(text.split())
def validate_target(target:str)->str:
    value=collapse(target)
    if not value:raise ValueError("Target sentence is required.")
    return value
def load_training_module():
    path=ROOT/"scripts/train_transformers.py";spec=importlib.util.spec_from_file_location("sandarbh_ui_phase4",path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
def build_encodings(tokenizer:Any,preceding:str,target:str,following:str):
    target=validate_target(target);row={"example_id":"ui-input","partition":"ui","hard_label":0,"soft_other":1.0,"soft_positive":0.0,"annotations":[0,0,0],"preceding":collapse(preceding),"target":target,"following":collapse(following)}
    with (ROOT/"configs/transformer_training.json").open(encoding="utf-8") as f:cfg=json.load(f)
    mod=load_training_module();encoded,manifest,meta=mod.construct_encodings(tokenizer,{"ui":[row]},cfg)
    return {"target":encoded["target"]["ui-input"],"context":encoded["context"]["ui-input"]},manifest,meta

