#!/bin/bash
# Run the Titanic AutoML pipeline in DEMO mode (fast execution)

set -e

echo "Running Titanic AutoML in DEMO mode..."
python -m titanic_automl.cli --mode demo
