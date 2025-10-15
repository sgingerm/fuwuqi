@echo off
REM Evaluate at K in (2,3,4,5,6) one by one
SET DATASET=D:\datanew\question-answer-passages_test.filtered.strict.jsonl
SET OUTDIR=D:\kg_out\eval

echo Installing dependencies (if needed)...
pip install -r requirements.txt

REM You can set --device cuda to use GPU for the query encoder and reranker
for %%K in (2 3 4 5 6) do (
  echo ===== Evaluating @k=%%K =====
  python -m eval.evaluate_dataset --dataset "%DATASET%" --out-dir "%OUTDIR%" --k %%K
)

echo All done. Results saved under %OUTDIR%.
pause
