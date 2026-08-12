# Local Apple-silicon pipeline. Every target runs on this laptop.
.PHONY: help setup test detect prep smoke pilot run-structured run-structured-misspec \
        run-llm-subset tools tools-misspec kg herbs templates calibrate sweep \
        evaluate report clean all

VENV    := .venv
PY      := $(VENV)/bin/python
PIP     := $(VENV)/bin/pip
PYTEST  := $(VENV)/bin/pytest
export PYTHONPATH := src

# Keep thread pools modest: unified memory is shared, and oversubscription
# competes with MLX for bandwidth rather than adding throughput.
export OMP_NUM_THREADS      := 4
export MKL_NUM_THREADS      := 4
export TOKENIZERS_PARALLELISM := false

help:
	@echo "Targets:"
	@echo "  make setup            create .venv and install pinned deps (never touches an existing env)"
	@echo "  make detect           print detected hardware and chosen model tier"
	@echo "  make test             run unit tests"
	@echo "  make prep             build the condition x feature matrix and KG"
	@echo "  make smoke            10-20 cases, all non-LLM parts + one real MLX generation"
	@echo "  make pilot            100-300 cases, timed, prints a full-run estimate"
	@echo "  make run-structured   full non-LLM evaluation over the feasible test set"
	@echo "  make run-llm-subset   LLM baselines on a bounded subset (resumable)"
	@echo "  make evaluate         metrics, statistical tests"
	@echo "  make report           figures and the written report"
	@echo ""
	@echo "Overnight:  caffeinate -i make run-llm-subset"

$(VENV):
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip wheel

setup: $(VENV)
	$(PIP) install -r requirements-macos.txt
	@$(PY) -m ayur.env_detect

detect:
	@$(PY) -m ayur.env_detect

test:
	$(PYTEST) tests/ -q

prep:
	$(PY) -m ayur.data.prep

smoke: prep
	$(PY) -m ayur.experiments.run --stage smoke

pilot: prep
	$(PY) -m ayur.experiments.run --stage pilot

run-structured: prep
	$(PY) -m ayur.experiments.run --stage structured

run-structured-misspec: prep
	$(PY) -m ayur.experiments.run --stage structured-misspecified

run-llm-subset: prep
	$(PY) -m ayur.experiments.run --stage llm-subset

# --- heterogeneous action space (the multi-tool contribution) ---------------
tools: prep
	$(PY) -m ayur.experiments.tools_run --n 200 --tag wellspec

tools-misspec: prep
	$(PY) -m ayur.experiments.tools_run --n 200 --tag misspec \
	    --env-noise 0.15 --omission 0.25 --assumed-noise 0.05

# Defends the headline multi-tool claim against "you chose the costs".
cost-sweep: prep
	$(PY) -m ayur.experiments.cost_sweep --n 120

# --- individual components --------------------------------------------------
kg:
	$(PY) -m ayur.kg.graph

nosology:
	$(PY) -m ayur.kg.nosology

retrieval:
	$(PY) -m ayur.tools.retrieval

bhashabench:
	$(PY) -m ayur.experiments.bhashabench --n 400

bhashabench-full:
	$(PY) -m ayur.experiments.bhashabench --all

llm-planner:
	$(PY) -m ayur.experiments.llm_baseline --n 60 --budget 10

herbs:
	$(PY) -m ayur.tools.herb_verify

templates:
	$(PY) -m ayur.env.translate
	$(PY) -m ayur.env.templates

calibrate:
	$(PY) -m ayur.experiments.calibrate

sweep: prep
	$(PY) -m ayur.experiments.sweep --n 400 --budget 20

evaluate:
	$(PY) -m ayur.experiments.evaluate --stage structured
	$(PY) -m ayur.experiments.evaluate --stage structured-misspecified

report:
	$(PY) -m ayur.experiments.report

# Everything that needs no LLM, in dependency order.
all: prep test nosology kg herbs retrieval templates sweep tools tools-misspec \
     cost-sweep calibrate report

clean:
	rm -rf data/processed results/checkpoints
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
