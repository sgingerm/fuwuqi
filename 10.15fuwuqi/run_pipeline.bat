@echo off
echo Installing dependencies (if needed)...
pip install -r requirements.txt

set "QUESTION=What are the adverse effects of curare?"
python -m vector_graph_pipeline.code.util.pipeline "%QUESTION%"
pause
