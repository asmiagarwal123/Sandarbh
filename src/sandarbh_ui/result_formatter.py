"""Plain-language, non-deployment interpretation of UI outputs."""
INPUT_STATE_KEYS=("preceding","target","following")
def clear_input_fields(state):
    """Reset only the three widget-bound input fields in a supplied session state."""
    for key in INPUT_STATE_KEYS:state[key]=""
def format_result(result:dict)->dict:
    label=result["predicted_label"]
    meaning=("Majority ableist annotation under this benchmark's binary mapping." if label==1 else "No majority ableist annotation under this benchmark's binary mapping. This does not prove that the text is harmless or acceptable.")
    status=("ABOVE RESEARCH THRESHOLD" if result["accepted"] else "REVIEW / BELOW RESEARCH THRESHOLD")
    explanation=(f"The frozen {result['experiment_id']} model predicts class {label}: {meaning}. "
      f"Its confidence is {result['confidence']:.1%}. ")
    if result["accepted"]: explanation+="Above the frozen 0.60 research confidence threshold."
    else: explanation+="Below the frozen 0.60 research operating point; route this output to review."
    explanation+=" The threshold is not a safety guarantee. This is research output, not a safety, diagnostic, clinical, or moderation decision."
    return {**result,"status":status,"review_status":status,"class_meaning":meaning,
      "probability_notice":"Model-estimated probability after temperature scaling — a research estimate, not a guarantee.",
      "explanation":explanation}
