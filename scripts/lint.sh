#!/bin/bash
# Lint gate for OWLBAN GROUP.
#
# Fails on:
#   1. ANY ruff finding anywhere in the repo (no baseline, no exceptions).
#   2. A pylint score below the threshold.
#
# The baseline mechanism was removed on purpose: a frozen list of violations
# is only as good as its upkeep, and it silently ratchets in dead code.
# When this gate fails, fix the finding.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY="${PYTHON:-.venv/bin/python}"
status=0

# --- ruff -------------------------------------------------------------------
if ! "$PY" -m ruff check . --no-fix --exit-non-zero-on-fix; then
    echo "FAIL: ruff findings (all must be zero)"
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
