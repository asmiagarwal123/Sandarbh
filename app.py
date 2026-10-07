"""SANDARBH local Streamlit research demonstration."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/"src"))
import streamlit as st
from sandarbh_ui.inference import MODEL_SPECS,load_model_bundle,predict
from sandarbh_ui.result_formatter import clear_input_fields

st.set_page_config(page_title="SANDARBH Research Demo",page_icon="🔬",layout="wide")
@st.cache_resource(show_spinner="Loading frozen checkpoint on CPU…")
def cached_bundle(experiment_id):return load_model_bundle(experiment_id)
def clear_inputs():clear_input_fields(st.session_state)
st.title("SANDARBH research demonstration")
st.warning("Research demonstration only—not a moderation system, safety guarantee, diagnosis, or basis for punishment. The benchmark is limited, domain-specific, and class-imbalanced; about 11% of the locked test set has a positive label.")
prediction,models,results,ethics=st.tabs(["Prediction","Model information","Research results","Limitations and ethics"])
with prediction:
    model=st.selectbox("Frozen experiment",list(MODEL_SPECS),index=1,format_func=lambda x:f"{x} — {MODEL_SPECS[x]['description']}",help="E6 is the neutral contextual demonstration default, not a test-selected winner.")
    preceding=st.text_area("Preceding context",key="preceding");target=st.text_area("Target sentence (required)",key="target");following=st.text_area("Following context",key="following")
    col1,col2=st.columns(2)
    if col1.button("Run frozen model",type="primary"):
        try:
            with st.spinner("Running deterministic CPU inference…"):out=predict(model,preceding,target,following,loader=cached_bundle)
            st.subheader(out["status"]);a,b,c=st.columns(3);a.metric("Model-estimated P(class 1)",f"{out['probability_1']:.1%}");b.metric("Predicted class",str(out["predicted_label"]));c.metric("Confidence",f"{out['confidence']:.1%}")
            st.caption(out["probability_notice"]);st.write(out["explanation"]);st.caption(f"Review status: {out['review_status']}. {out['description']} Context used: {out['context_used']}. Temperature: {out['temperature']:.12g}. CPU time: {out['inference_seconds']:.2f}s. Encoded length: {out['sequence_length']}/256.")
        except (ValueError,FileNotFoundError,RuntimeError) as exc:st.error(str(exc))
    col2.button("Clear inputs",on_click=clear_inputs)
with models:
    st.markdown("**E5:** target-only hard-label training. Locked-test positive precision/recall: 0.270/0.711.\n\n**E6:** preceding, target, and following context with hard-label training. Locked-test positive precision/recall: 0.315/0.447.\n\n**E7:** Contextual cost-weighted soft-vote training with the same contextual encoding as E6. Locked-test positive precision/recall: 0.255/0.711. E7 does not isolate the effect of soft labels, so E7-versus-E6 is not a clean causal soft-label comparison.\n\nThese are descriptive results on a locked test set with approximately 11% positive prevalence. The selector is a scientific comparison control; the UI never chooses a model based on test performance.")
with results:
    st.write("Aggregate locked-test results only; no test examples or individual predictions are exposed.")
    st.image(str(ROOT/"reports/figures/01_macro_f1_by_experiment.png"));st.image(str(ROOT/"reports/figures/03_calibration_before_after.png"));st.image(str(ROOT/"reports/figures/04_coverage_risk_curves.png"))
with ethics:
    st.markdown("""- Research prototype; not deployment-ready.
- No diagnostic or clinical use.
- No automated punishment or moderation decision.
- Class 1 means majority ableist annotation under this benchmark.
- Class 0 means no majority ableist annotation; it does **not** mean definitively safe or non-ableist.
- The dataset is limited and class-imbalanced; predictions can be incorrect and harmful.
- Intra-community, reclaimed, quoted, sarcastic, or critical language can be misclassified.
- Confidence-based abstention can reject positive-labelled cases and does not guarantee safety.
- Consequential interpretation requires human review and external validation.""")
