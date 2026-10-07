import importlib,logging,sys,unittest
from pathlib import Path
from unittest import mock
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from sandarbh_ui import inference
from sandarbh_ui.input_builder import build_encodings,validate_target
from sandarbh_ui.result_formatter import clear_input_fields,format_result

class FakeTokenizer:
    bos_token_id=0;eos_token_id=2;pad_token_id=1
    def __call__(self,text,**kwargs):
        special={"Preceding context:\n":list(range(10,16)),"Target:\n":[20,21,22],"Following context:\n":[30,31,32,33]," â€¦":[40]}
        return {"input_ids":special.get(text,list(range(100,100+len(text.split()))))}
class FakeOutput:
    def __init__(self,logits):self.logits=logits
class FakeModel:
    def eval(self):return self
    def __call__(self,**kwargs):
        import torch
        return FakeOutput(torch.tensor([[0.0,1.0]],dtype=torch.float32))
def fake_loader(_):return FakeTokenizer(),FakeModel()

class UITests(unittest.TestCase):
    def test_empty_target_rejected(self):
        with self.assertRaisesRegex(ValueError,"required"):validate_target(" \n ")
    def test_model_selector_mapping(self):self.assertEqual(set(inference.MODEL_SPECS),{"E5","E6","E7"})
    def test_e5_target_only_behavior(self):self.assertEqual(inference.MODEL_SPECS["E5"]["mode"],"target")
    def test_e6_e7_contextual_parity(self):self.assertEqual(inference.MODEL_SPECS["E6"]["mode"],inference.MODEL_SPECS["E7"]["mode"])
    def test_e7_cost_weighted_soft_vote_wording(self):
        phrase="Contextual cost-weighted soft-vote training";self.assertIn(phrase,inference.MODEL_SPECS["E7"]["description"]);self.assertIn(phrase,(ROOT/"app.py").read_text(encoding="utf-8"))
    def test_target_preserving_truncation(self):
        enc,manifest,_=build_encodings(FakeTokenizer(),"a b"," ".join(["x"]*400),"c d");self.assertEqual(len(enc["context"]["input_ids"]),256);self.assertTrue(any(r["target_truncated"]=="True" for r in manifest))
    def test_temperature_application(self):self.assertNotEqual(inference.calibrated_probabilities(0,1,1),inference.calibrated_probabilities(0,1,.5))
    def test_probability_normalization(self):self.assertAlmostEqual(sum(inference.calibrated_probabilities(-1,2,.6)),1)
    def test_argmax_preservation(self):
        for t in (.1,.6,1,2):self.assertGreater(inference.calibrated_probabilities(-1,2,t)[1],.5)
    def test_confidence(self):self.assertAlmostEqual(max(inference.calibrated_probabilities(0,1,1)),.7310585786)
    def test_acceptance_rule(self):self.assertTrue(inference.acceptance(.60));self.assertFalse(inference.acceptance(.599999))
    def test_result_formatting(self):
        x=format_result({"predicted_label":0,"accepted":False,"experiment_id":"E6","confidence":.59});self.assertEqual(x["status"],"REVIEW / BELOW RESEARCH THRESHOLD");self.assertIn("does not prove",x["class_meaning"]);self.assertIn("not a safety guarantee",x["explanation"]);self.assertIn("research estimate",x["probability_notice"])
    def test_above_threshold_wording_is_not_approval(self):
        x=format_result({"predicted_label":1,"accepted":True,"experiment_id":"E5","confidence":.75});self.assertEqual(x["status"],"ABOVE RESEARCH THRESHOLD");self.assertNotIn("ACCEPTED",x["status"]);self.assertIn("not a safety guarantee",x["explanation"])
    def test_clear_callback_resets_only_input_fields(self):
        state={"preceding":"before","target":"target","following":"after","model":"E7"};clear_input_fields(state);self.assertEqual({k:state[k] for k in ("preceding","target","following")},{"preceding":"","target":"","following":""});self.assertEqual(state["model"],"E7");self.assertIn("on_click=clear_inputs",(ROOT/"app.py").read_text(encoding="utf-8"))
    def test_no_source_text_logging(self):self.assertNotIn("logging.",(ROOT/"src/sandarbh_ui/inference.py").read_text(encoding="utf-8"))
    def test_lazy_model_loading(self):self.assertIn("@lru_cache",(ROOT/"src/sandarbh_ui/inference.py").read_text(encoding="utf-8"));self.assertEqual(inference.load_model_bundle.cache_info().currsize,0)
    def test_missing_checkpoint_error(self):
        with mock.patch.dict(inference.MODEL_SPECS,{"E5":{**inference.MODEL_SPECS["E5"],"path":"models/missing"}}):
            inference.load_model_bundle.cache_clear()
            with self.assertRaises(FileNotFoundError):inference.load_model_bundle("E5")
    def test_predict_with_mock(self):
        try:import torch
        except ImportError:self.skipTest("PyTorch available only in transformer environment")
        x=inference.predict("E6","before","target","after",loader=fake_loader);self.assertEqual(x["predicted_label"],1);self.assertTrue(x["context_used"]);self.assertLessEqual(x["sequence_length"],256)
    def test_streamlit_import_and_app_compile(self):
        try:import streamlit
        except ImportError:self.skipTest("Streamlit available only in transformer/UI environment")
        compile((ROOT/"app.py").read_text(encoding="utf-8"),str(ROOT/"app.py"),"exec")

if __name__=="__main__":unittest.main()
