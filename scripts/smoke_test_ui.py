#!/usr/bin/env python3
"""One-input UI inference smoke test; deliberately prints no entered text."""
from pathlib import Path
import json,os,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from sandarbh_ui.inference import predict
def main():
    result=predict("E6","Synthetic preceding context.","Synthetic target for local smoke testing.","Synthetic following context.")
    summary={"status":"PASSED","experiment_id":result["experiment_id"],"probability_sum":result["probability_0"]+result["probability_1"],"sequence_length":result["sequence_length"],"accepted":result["accepted"],"text_saved":False}
    path=ROOT/"results/final/ui_smoke_test.json";path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+".tmp");tmp.write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8");os.replace(tmp,path);print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
