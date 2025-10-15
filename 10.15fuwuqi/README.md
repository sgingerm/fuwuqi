# jiang (example repository)

This bundle contains:
- `vector_graph_pipeline/`: the module implementing query-only encoding, one-shot similarity, local APPNP, BFS, and reranking.
- `eval/`: evaluation scripts to compute per-question precision/recall/F1@k and export top 30% by F1.
- `requirements.txt`: Python dependencies.
- `run_pipeline.bat`: Windows one-click runner (edits the QUESTION line to test).
- `run_eval_k_2_3_4_5_6.bat`: Windows one-click batch to evaluate K in {2,3,4,5,6}.

## Quick start (Windows)
1. Unzip this repo.
2. Open a terminal in the repo folder and run:
```
pip install -r requirements.txt
```
3. To test the pipeline on a single question:
```
python -m vector_graph_pipeline.code.util.pipeline "What are the adverse effects of curare?"
```
4. To evaluate the JSONL dataset (defaults to `D:\datanew\question-answer-passages_test.filtered.strict.jsonl`):
```
python -m eval.run_eval_all
```
Outputs are written to `D:\kg_out\eval\`:
- `eval_k{k}_all.csv` / `eval_k{k}_top30.csv`
- `eval_k{k}_all.json` / `eval_k{k}_top30.json`
- `eval_k{k}_summary.json`
