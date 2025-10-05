#!/bin/bash
# Run the Titanic AutoML pipeline in FULL mode (comprehensive search)

set -e

echo "Running Titanic AutoML in FULL mode..."
python -m titanic_automl.cli --mode full
