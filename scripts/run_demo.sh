#!/usr/bin/env bash
set -euo pipefail

# Run a quick demo using data in data/data_raw and write artifacts to artifacts/
python -m titanic_automl.cli --mode demo --data-dir data/data_raw --output-dir artifacts
