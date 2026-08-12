#!/usr/bin/env bash
# Remaining non-LLM experiments, run sequentially so they never compete for
# unified memory. Launch with:  caffeinate -i bash scripts/overnight.sh
#
# Each step is resumable and writes its own JSON, so an interruption costs at
# most the step in flight.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
PY=.venv/bin/python

step() {
  echo ""
  echo "############################################################"
  echo "# $1"
  echo "# started $(date '+%H:%M:%S')"
  echo "############################################################"
  shift
  "$@"
  echo "# -> exit $? at $(date '+%H:%M:%S')"
}

# Wait for any experiment still running from an earlier launch.
while pgrep -f "ayur.experiments.run" > /dev/null; do
  echo "waiting for in-flight run to finish ... $(date '+%H:%M:%S')"
  sleep 30
done

step "evaluate structured"            $PY -m ayur.experiments.evaluate --stage structured
step "evaluate structured-misspec"    $PY -m ayur.experiments.evaluate --stage structured-misspecified
step "budget sweep"                   $PY -m ayur.experiments.sweep --n 400 --budget 20
step "tools: well-specified n=200"    $PY -m ayur.experiments.tools_run --n 200 --tag wellspec
step "tools: misspecified n=200"      $PY -m ayur.experiments.tools_run --n 200 --tag misspec \
                                          --env-noise 0.15 --omission 0.25 --assumed-noise 0.05
step "calibration"                    $PY -m ayur.experiments.calibrate --n-calib 400 --n-test 400
step "knowledge graph"                $PY -m ayur.kg.graph
step "herb verification"              $PY -m ayur.tools.herb_verify
step "bilingual templates"            $PY -m ayur.env.templates
step "report + figures"               $PY -m ayur.experiments.report

echo ""
echo "############################################################"
echo "# ALL DONE $(date '+%H:%M:%S')"
echo "############################################################"
ls -la results/*.json results/figures/*.png 2>/dev/null
