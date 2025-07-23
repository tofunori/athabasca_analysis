"""
Snakefile for multi-glacier MODIS-AWS albedo analysis pipeline
"""

import yaml
from pathlib import Path

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Get all glacier names
GLACIERS = list(config["glaciers"].keys())

# Define output directories
OUTPUT_DIRS = config["output_dirs"]

rule all:
    """
    Run analysis for all glaciers
    """
    input:
        expand(f"{OUTPUT_DIRS['executed']}/{{glacier}}_analysis.ipynb", glacier=GLACIERS),
        expand(f"{OUTPUT_DIRS['reports']}/{{glacier}}_report.html", glacier=GLACIERS)

rule run_analysis:
    """
    Execute notebook for a specific glacier using Papermill
    """
    input:
        notebook="compare_albedo.ipynb",
        config="config.yaml"
    output:
        f"{OUTPUT_DIRS['executed']}/{{glacier}}_analysis.ipynb"
    params:
        glacier=lambda wildcards: wildcards.glacier,
        glacier_config=lambda wildcards: config["glaciers"][wildcards.glacier]
    shell:
        """
        papermill {input.notebook} {output} \\
            -p GLACIER {params.glacier} \\
            -p CSV_PATH {params.glacier_config[csv_path]} \\
            -p MASK_PATH {params.glacier_config[mask_path]} \\
            -p LAT {params.glacier_config[lat]} \\
            -p LON {params.glacier_config[lon]} \\
            -p OUTLIER_THRESHOLD {config[analysis][outlier_threshold]} \\
            -p TARGET_MONTHS {config[analysis][target_months]} \\
            -p METHOD_COLORS {config[analysis][method_colors]}
        """

rule convert_to_html:
    """
    Convert executed notebook to HTML report
    """
    input:
        f"{OUTPUT_DIRS['executed']}/{{glacier}}_analysis.ipynb"
    output:
        f"{OUTPUT_DIRS['reports']}/{{glacier}}_report.html"
    shell:
        """
        jupyter nbconvert --to html --execute {input} --output-dir {OUTPUT_DIRS[reports]} \\
            --output {wildcards.glacier}_report.html
        """

rule clean:
    """
    Clean generated files
    """
    shell:
        f"""
        rm -rf {OUTPUT_DIRS['executed']}/*
        rm -rf {OUTPUT_DIRS['reports']}/*
        rm -rf {OUTPUT_DIRS['figures']}/*
        rm -rf {OUTPUT_DIRS['results']}/*
        """

rule run_glacier:
    """
    Run analysis for a specific glacier
    Usage: snakemake run_glacier --config glacier=Athabasca
    """
    input:
        f"{OUTPUT_DIRS['executed']}/{{config[glacier]}}_analysis.ipynb",
        f"{OUTPUT_DIRS['reports']}/{{config[glacier]}}_report.html"
