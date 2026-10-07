# SANDARBH Local UI Guide

The Streamlit interface is a local research demonstration of the frozen E5, E6, and E7 checkpoints. It never searches or displays the dataset, test examples, individual saved predictions, tokens, embeddings, or local filesystem paths.

From the project root, install the bounded UI dependencies and start the application:

```powershell
.\.venv-transformers\Scripts\python.exe -m pip install -r requirements-ui.txt
.\.venv-transformers\Scripts\python.exe -m streamlit run app.py
```

Open the local URL Streamlit prints, normally `http://localhost:8501`. Stop it with `Ctrl+C`.

Enter optional preceding and following context and a required target sentence. Choose E5, E6, or E7 explicitly; E6 is merely the neutral contextual default and is not a test-selected winner. The UI applies the exact Phase 4B 256-token construction, frozen checkpoint, frozen temperature, 0.50 classification boundary, and 0.60 selective-confidence threshold.

The first request for a model loads its checkpoint on CPU and may take several seconds; subsequent requests reuse the cached model. CPU inference latency depends on hardware. Missing checkpoints or an empty target produce a clear error. Entered text is held only in memory for inference and is not written to project files or logs by the application.

Outputs describe the benchmark class, calibrated probabilities, confidence, and `ACCEPTED`/`ABSTAIN` status. Class 1 means majority ableist annotation under the project benchmark. Class 0 means no majority ableist annotation; it does not mean that language is safe or non-ableist.

This interface is not diagnostic, clinical, moderation-ready, or suitable for automated punishment. Predictions can be wrong or harmful and require human interpretation. The aggregate research-results tab exposes only validated summary figures.
