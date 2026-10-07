import importlib.util
import json
import math
import sys
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"));sys.path.insert(0,str(ROOT/"src"))
import train_transformers as training
import validate_v1_1 as validation
from sandarbh_ui.input_builder import build_encodings


class FakeTokenizer:
    bos_token_id=0;eos_token_id=2;pad_token_id=1
    def __call__(self,text,**kwargs):
        markers={"Preceding context:\n":list(range(10,16)),"Target:\n":[20,21,22],"Following context:\n":[30,31,32,33]," …":[40]}
        return {"input_ids":markers.get(text,list(range(100,100+len(text.split()))))}


class V11ValidationTests(unittest.TestCase):
    def setUp(self): self.tok=FakeTokenizer()
    def test_weighted_soft_loss_and_optimum(self):
        try: import torch
        except ImportError: self.skipTest("PyTorch is installed in the transformer environment")
        weights=torch.tensor([1595/(2*1415),1595/(2*180)],dtype=torch.float64)
        for q,expected in ((0,0),(1/3,0.7971830985915492),(2/3,0.9401993355481727),(1,1)):
            optimum=weights[1].item()*q/(weights[0].item()*(1-q)+weights[1].item()*q) if q else 0
            self.assertAlmostEqual(optimum,expected,12)
            if 0<q<1:
                logit=torch.tensor([[0.0,math.log(optimum/(1-optimum))]],dtype=torch.float64,requires_grad=True)
                target=torch.tensor([[1-q,q]],dtype=torch.float64)
                loss=training.weighted_soft_loss(torch.log_softmax(logit,dim=1),target,weights);loss.backward()
                self.assertAlmostEqual(logit.grad[0,1].item(),0.0,12)
    def test_short_target_both_contexts(self):
        enc,manifest,_=build_encodings(self.tok,"before words","target words","after words")
        self.assertLess(len(enc["context"]["input_ids"]),256)
    def test_missing_preceding_context(self):
        enc,_,_=build_encodings(self.tok,"","target","after words");self.assertLessEqual(len(enc["context"]["input_ids"]),256)
    def test_missing_following_context(self):
        enc,_,_=build_encodings(self.tok,"before words","target","");self.assertLessEqual(len(enc["context"]["input_ids"]),256)
    def test_long_preceding_keeps_tail_and_long_following_keeps_head(self):
        pre=list(range(300));fol=list(range(400,700));a,b=training.allocate_context(pre,fol,20)
        self.assertEqual(a,pre[-10:]);self.assertEqual(b,fol[:10])
    def test_overlong_target_is_head_ellipsis_tail(self):
        out,meta=training.truncate_target(list(range(300)),241,[999]);self.assertEqual(len(out),241);self.assertEqual(out[120],999);self.assertEqual(out[:2],[0,1]);self.assertEqual(out[-2:],[298,299]);self.assertTrue(meta["truncated"])
    def test_deterministic_redistribution(self):
        pre=list(range(2));fol=list(range(100,200));a,b=training.allocate_context(pre,fol,11);self.assertEqual(a,pre);self.assertEqual(len(b),9)
    def test_e6_e7_encoding_identity(self):
        enc,manifest,meta=build_encodings(self.tok,"before","target","after");self.assertEqual(meta["e6_encoded_manifest_sha256"],meta["e7_encoded_manifest_sha256"]);self.assertEqual(enc["context"],enc["context"])
    def test_exact_length_boundary(self):
        enc,_,_=build_encodings(self.tok,"p "*200,"t "*241,"f "*200);self.assertEqual(len(enc["context"]["input_ids"]),256)
    def test_binary_positive_brier(self):
        value=validation.probability_metrics([0,1],[.25,.75])["brier_score"];self.assertAlmostEqual(value,.0625)
    def test_immutable_critical_hashes(self):
        config=json.loads((ROOT/"configs/v1_1_validation.json").read_text());manifest=validation.integrity_manifest(config,"test");self.assertEqual(manifest["status"],"PASS")
    def test_outputs_are_text_free(self):
        forbidden={"preceding","target","following","sentence","text"}
        for name in ("probability_null_comparison.json","posthoc_summary.json","seed_robustness.json"):
            data=json.loads((ROOT/"results/v1_1_validation"/name).read_text())
            self.assertFalse(any(k in forbidden for k in self._keys(data)))
    def _keys(self,value):
        if isinstance(value,dict):
            for k,v in value.items(): yield k;yield from self._keys(v)
        elif isinstance(value,list):
            for v in value: yield from self._keys(v)


if __name__=="__main__": unittest.main()
