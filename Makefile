ENV_NAME ?= cs789
CONDA ?= ~/anaconda3/bin/conda
REQ_PROFILE ?= requirements-laptop-cpu.txt
PY := $(CONDA) run -n $(ENV_NAME) python3

.PHONY: install spacy build-2017 build-2024 smoke test eval-taxbench-dry eval-sara-dry viz

install:
	$(PY) -m pip install -r $(REQ_PROFILE)

spacy:
	$(PY) -m spacy download en_core_web_sm

build-2017:
	KNOWLEDGE_PROFILE=2017 $(PY) scripts/build_pipeline.py

build-2024:
	KNOWLEDGE_PROFILE=2024-2026 $(PY) scripts/build_pipeline.py

smoke:
	$(PY) scripts/test_query.py "standard deduction for single filers"

test:
	$(PY) -m pytest -q

eval-taxbench-dry:
	$(PY) evaluation/run_eval.py --dataset taxbench --mode hybrid --dry-run --limit 5 --overwrite

eval-sara-dry:
	SARA_SPLIT=test $(PY) evaluation/run_eval.py --dataset sara_v3 --mode none --dry-run --limit 20 --overwrite

viz:
	$(PY) scripts/viz_graph.py --sample-n 300
