#!/bin/bash
# Lint gate for OWLBAN GROUP.
#
# Fails on:
#   1. Any ruff finding in tests/ (the test suite must be clean).
#   2. Any ruff finding outside tests/ that is not in .ruff-baseline.txt
#      (application modules only carry a frozen backlog).
#   3. A pylint score below the threshold.
#
# Regenerate the baseline after fixing findings:
#   .venv/bin/python -m ruff check . --output-format=concise \
#     | grep -E '^[^:]+:[0-9]+:[0-9]+: [A-Z]+[0-9]+' > .ruff-baseline.txt
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY="${PYTHON:-.venv/bin/python}"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
status=0

# --- ruff -------------------------------------------------------------------
# tests/ is held clean outright: no baseline applies to it.
if ! "$PY" -m ruff check tests/; then
    echo "FAIL: ruff findings in tests/ (tests must be clean)"
    status=1
fi

# Application modules: only baseline findings are tolerated. Findings are
# filtered before diffing against the baseline so that a change in ruff's
# summary output can never be mistaken for a new finding.
"$PY" -m ruff check . --output-format=concise > "$tmp/ruff-raw.txt" 2>&1 || true
grep -E '^[^:]+:[0-9]+:[0-9]+: [A-Z]+[0-9]+' "$tmp/ruff-raw.txt" \
    > "$tmp/ruff-findings.txt" || true
grep -vxF -f .ruff-baseline.txt "$tmp/ruff-findings.txt" \
    > "$tmp/ruff-new.txt" || true
if [ -s "$tmp/ruff-new.txt" ]; then
    echo "FAIL: new ruff findings not in .ruff-baseline.txt:"
    cat "$tmp/ruff-new.txt"
    status=1
fi

# --- pylint -----------------------------------------------------------------
if ! "$PY" -m pylint --rcfile=.pylintrc auth_lib.py \
        --disable=import-error --fail-under=9.5; then
    echo "FAIL: pylint score below threshold"
    status=1
fi

if [ "$status" -eq 0 ]; then
    echo "PASS: lint checks clean"
fi
exit "$status"
