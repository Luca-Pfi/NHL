#!/usr/bin/env bash
# Full pipeline: fetch -> build -> models/features -> analysis (figures/ + results/)
set -euo pipefail
cd "$(dirname "$0")"
python3 src/fetch_games.py "$@"
python3 src/build_dataset.py
python3 src/features.py > /dev/null
python3 src/analysis.py > /dev/null
echo "Done: see REPORT.md, figures/ and results/summary.json"
