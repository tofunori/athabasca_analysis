#!/bin/bash
# Multi-glacier analysis script
# Usage: ./run_analysis.sh [glacier_name]

set -e

# Default glacier if none specified
GLACIER=${1:-"Athabasca"}

# Check if config.yaml exists
if [[ ! -f "config.yaml" ]]; then
    echo "Error: config.yaml not found!"
    exit 1
fi

# Extract glacier configuration using yq (or python if yq not available)
if command -v yq &> /dev/null; then
    CSV_PATH=$(yq eval ".glaciers.${GLACIER}.csv_path" config.yaml)
    MASK_PATH=$(yq eval ".glaciers.${GLACIER}.mask_path" config.yaml)
    LAT=$(yq eval ".glaciers.${GLACIER}.lat" config.yaml)
    LON=$(yq eval ".glaciers.${GLACIER}.lon" config.yaml)
    OUTLIER_THRESHOLD=$(yq eval ".analysis.outlier_threshold" config.yaml)
else
    # Fallback to python for YAML parsing
    CSV_PATH=$(python -c "import yaml; c=yaml.safe_load(open('config.yaml')); print(c['glaciers']['${GLACIER}']['csv_path'])")
    MASK_PATH=$(python -c "import yaml; c=yaml.safe_load(open('config.yaml')); print(c['glaciers']['${GLACIER}']['mask_path'])")
    LAT=$(python -c "import yaml; c=yaml.safe_load(open('config.yaml')); print(c['glaciers']['${GLACIER}']['lat'])")
    LON=$(python -c "import yaml; c=yaml.safe_load(open('config.yaml')); print(c['glaciers']['${GLACIER}']['lon'])")
    OUTLIER_THRESHOLD=$(python -c "import yaml; c=yaml.safe_load(open('config.yaml')); print(c['analysis']['outlier_threshold'])")
fi

echo "Running analysis for ${GLACIER} glacier..."
echo "CSV Path: ${CSV_PATH}"
echo "Mask Path: ${MASK_PATH}"
echo "Coordinates: ${LAT}, ${LON}"
echo "Outlier Threshold: ${OUTLIER_THRESHOLD}"

# Create output directories
mkdir -p executed results figs reports

# Run analysis with Papermill
echo "Executing notebook with Papermill..."
papermill compare_albedo.ipynb "executed/${GLACIER}_analysis.ipynb" \
    -p GLACIER "${GLACIER}" \
    -p CSV_PATH "${CSV_PATH}" \
    -p MASK_PATH "${MASK_PATH}" \
    -p LAT ${LAT} \
    -p LON ${LON} \
    -p OUTLIER_THRESHOLD ${OUTLIER_THRESHOLD} \
    -p TARGET_MONTHS "[6, 7, 8, 9]"

# Convert to HTML report
echo "Converting to HTML report..."
jupyter nbconvert --to html "executed/${GLACIER}_analysis.ipynb" \
    --output-dir reports \
    --output "${GLACIER}_report.html"

echo "Analysis complete!"
echo "Results saved in:"
echo "  - Executed notebook: executed/${GLACIER}_analysis.ipynb"
echo "  - HTML report: reports/${GLACIER}_report.html"
echo "  - Figures: figs/${GLACIER}/"
echo "  - Results: results/${GLACIER}/"
