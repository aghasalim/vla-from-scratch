#!/usr/bin/env bash
# Run every independent recomputation. Skip the ones whose toolchain is missing,
# say so, and fail if any that ran disagreed.
#
#   ./verify/verify.sh
set -uo pipefail

cd "$(dirname "$0")/.." || exit 2
ROOT=$PWD
pass=0; fail=0; skip=0
failed_names=()

have() { command -v "$1" >/dev/null 2>&1; }

run() {   # run <label> <command...>
  local label=$1; shift
  echo "--- $label"
  if "$@"; then
    pass=$((pass + 1))
  else
    fail=$((fail + 1)); failed_names+=("$label")
  fi
}

skipped() {
  echo "--- $1"
  echo "    skipped: $2"
  skip=$((skip + 1))
}

# SQLite reads standard input, which inside a script is the rest of this script,
# so the redirect from /dev/null is load bearing.
if have sqlite3; then
  run "sqlite3  summaries.sql" \
    sh -c 'sqlite3 :memory: -cmd ".read verify/summaries.sql" < /dev/null'
else
  skipped "sqlite3  summaries.sql" "sqlite3 not installed"
fi

if have cc; then
  bin=$(mktemp -d)/route
  if cc -O2 -std=c99 -Wall -o "$bin" verify/route.c -lm; then
    run "C        route.c" "$bin" verify/golden_routes.csv
  else
    fail=$((fail + 1)); failed_names+=("C route.c did not compile")
  fi
else
  skipped "C        route.c" "no C compiler on PATH"
fi

if have go; then
  run "Go       gocheck" sh -c 'cd verify/gocheck && go run . -root ..'
else
  skipped "Go       gocheck" "go not installed"
fi

if have Rscript; then
  run "R        verify.R" Rscript verify/verify.R "$ROOT"
else
  skipped "R        verify.R" "Rscript not installed"
fi

if have cargo; then
  run "Rust     demomc" sh -c \
    'cd verify/demomc && cargo run --release --quiet -- ../demo_stats.json'
else
  skipped "Rust     demomc" "cargo not installed"
fi

if have node; then
  run "Node     derived.js" node verify/derived.js "$ROOT"
else
  skipped "Node     derived.js" "node not installed"
fi

if have ruby; then
  run "Ruby     tables.rb" ruby verify/tables.rb "$ROOT"
else
  skipped "Ruby     tables.rb" "ruby not installed"
fi

# The golden files are only reference data if they still match the code that
# wrote them. Regenerate into a scratch directory and diff. Needs torch, so it
# is skipped wherever the experiment itself could not be run either.
py=""
for cand in ${VERIFY_PYTHON:-} .venv/bin/python python3 python; do
  if have "$cand" || [ -x "$cand" ]; then
    if "$cand" -c "import torch" >/dev/null 2>&1; then py=$cand; break; fi
  fi
done
if [ -n "$py" ]; then
  tmp=$(mktemp -d)
  run "Python   golden files are current" sh -c \
    "PYTHONPATH='$ROOT' '$py' verify/export_golden.py '$tmp' >/dev/null &&
     diff -q verify/golden_routes.csv '$tmp/golden_routes.csv' &&
     diff -q verify/demo_stats.json '$tmp/demo_stats.json'"
else
  skipped "Python   golden files are current" "no python with torch on PATH"
fi

echo
echo "$pass passed, $fail failed, $skip skipped"
for n in "${failed_names[@]:-}"; do [ -n "$n" ] && echo "  failed: $n"; done
[ "$fail" -eq 0 ] || exit 1
