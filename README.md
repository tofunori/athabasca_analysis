# Multi-Glacier MODIS-AWS Albedo Analysis Pipeline

This repository contains a parameterized pipeline for analyzing MODIS satellite albedo data against AWS ground-truth measurements for multiple glaciers.

## Features

- **Multi-glacier support**: Easily run analysis for any glacier by updating configuration
- **Parameterized notebooks**: Uses Papermill to inject glacier-specific parameters
- **Automated pipeline**: Snakemake workflow or simple bash/batch scripts
- **Organized outputs**: Separate directories for each glacier's results
- **Configurable analysis**: Adjustable outlier thresholds and analysis parameters

## Quick Start

### Prerequisites

```bash
# Install required packages
pip install papermill snakemake jupyter numpy pandas matplotlib scipy
# or with conda:
conda install -c conda-forge papermill snakemake jupyter-lab
```

### Basic Usage

#### Option 1: Simple Script (Recommended)

**Linux/Mac:**
```bash
chmod +x run_analysis.sh
./run_analysis.sh Athabasca
```

**Windows:**
```cmd
run_analysis.bat Athabasca
```

#### Option 2: Snakemake Pipeline

```bash
# Run all glaciers
snakemake

# Run specific glacier
snakemake run_glacier --config glacier=Athabasca

# Clean outputs
snakemake clean
```

#### Option 3: Direct Papermill

```bash
papermill compare_albedo.ipynb executed/Athabasca_analysis.ipynb \\
    -p GLACIER "Athabasca" \\
    -p CSV_PATH "data/csv/iceAWS_Atha_albedo_daily_20152020_filled_clean.csv" \\
    -p OUTLIER_THRESHOLD 2.5
```

## Configuration

Edit `config.yaml` to add new glaciers or modify analysis parameters:

```yaml
glaciers:
  Athabasca:
    csv_path: "data/aws/Athabasca.csv"
    mask_path: "data/masks/Athabasca.tif"
    lat: 52.19
    lon: -117.28
    description: "Athabasca Glacier, Columbia Icefield"
  
  NewGlacier:
    csv_path: "data/aws/NewGlacier.csv"
    mask_path: "data/masks/NewGlacier.tif"
    lat: 50.0
    lon: -115.0
    description: "New Glacier Example"

analysis:
  outlier_threshold: 2.5
  target_months: [6, 7, 8, 9]  # June through September
```

## Directory Structure

```
project_root/
├── compare_albedo.ipynb        # Parameterized analysis notebook
├── config.yaml                # Glacier configurations
├── scripts/
│   └── analysis.py            # Reusable analysis functions
├── data/
│   ├── csv/                   # AWS and MODIS data files
│   └── masks/                 # Glacier mask files
├── executed/                  # Executed notebooks (git-ignored)
├── results/                   # Analysis results by glacier
├── figs/                      # Generated figures by glacier
├── reports/                   # HTML reports by glacier
├── run_analysis.sh            # Linux/Mac script
├── run_analysis.bat           # Windows script
└── Snakefile                  # Snakemake workflow
```

## Parameters

The notebook accepts these parameters via Papermill:

- `GLACIER`: Glacier name (string)
- `CSV_PATH`: Path to AWS data CSV (string) 
- `MASK_PATH`: Path to glacier mask file (string)
- `LAT`: Glacier latitude (float)
- `LON`: Glacier longitude (float)
- `OUTLIER_THRESHOLD`: Z-score threshold for outlier detection (float, default: 2.5)
- `TARGET_MONTHS`: List of months for analysis (list, default: [6,7,8,9])
- `METHOD_COLORS`: Color scheme for plotting (dict)

## Outputs

For each glacier, the pipeline generates:

- **Executed notebook**: `executed/{glacier}_analysis.ipynb`
- **HTML report**: `reports/{glacier}_report.html`
- **Figures**: `figs/{glacier}/` (PNG files)
- **Results**: `results/{glacier}/` (CSV statistics)

## Analysis Features

- **Multi-method comparison**: MCD43A3, MOD09GA, MOD10A1 vs AWS
- **Statistical analysis**: Correlation, RMSE, MAE, bias calculations
- **Temporal analysis**: Monthly, 16-day, and weekly composites
- **Outlier detection**: Configurable Z-score based outlier identification
- **Pixel-level analysis**: Spatial performance evaluation
- **Comprehensive visualization**: Scatter plots, time series, box plots
- **Month-coded outliers**: Color-coded outlier visualization by month

## Adding New Glaciers

1. **Add data files**: Place AWS CSV and mask files in appropriate directories
2. **Update config.yaml**: Add glacier entry with paths and coordinates
3. **Run analysis**: Use any of the execution methods above

Example data file naming convention:
- AWS data: `data/csv/iceAWS_{glacier}_albedo_daily_*.csv`
- MODIS data: `data/csv/{glacier}_Terra_Aqua_MultiProduct_*.csv`
- Mask file: `data/masks/{glacier}.tif`

## Dependencies

- Python 3.7+
- papermill: Notebook parameterization and execution
- jupyter: Notebook environment
- pandas: Data manipulation
- numpy: Numerical computations
- matplotlib: Plotting
- scipy: Statistical analysis
- snakemake: Workflow management (optional)
- yq: YAML parsing in bash (optional)

## Troubleshooting

**Common issues:**

1. **Missing data files**: Check file paths in config.yaml match actual file locations
2. **Permission errors**: Ensure execute permissions on bash scripts (`chmod +x run_analysis.sh`)
3. **Module import errors**: Install missing dependencies with pip/conda
4. **Memory issues**: For large datasets, consider processing in chunks

**Debugging:**

- Check executed notebooks in `executed/` directory for detailed error messages
- Use `papermill --help` for additional options
- Enable verbose output with `snakemake -v` for Snakemake debugging

## Contributing

To extend the pipeline:

1. Add new analysis functions to `scripts/analysis.py`
2. Update the notebook template with new analysis sections
3. Modify `config.yaml` schema for additional parameters
4. Update documentation

## License

This project is licensed under the MIT License - see LICENSE file for details.
