#!/usr/bin/env python3
"""
Simple and clean script to reorganize the multi-method data from long format to wide format.
"""

import pandas as pd
import numpy as np

def reorganize_data_simple(input_file, output_file):
    """
    Reorganize data from long format to wide format using pandas pivot.
    """
    
    print("Loading data...")
    df = pd.read_csv(input_file)
    
    print(f"Original data shape: {df.shape}")
    print(f"Available methods: {list(df['method'].unique())}")
    print("Method counts:")
    print(df['method'].value_counts())
    
    # Method-specific columns to pivot
    value_cols = ['albedo', 'ndsi', 'glacier_fraction', 'solar_zenith']
    
    # Columns that identify each unique observation
    id_cols = ['pixel_id', 'date', 'qa_mode']
    
    # Metadata columns (should be the same for all methods for a given pixel_id/date)
    metadata_cols = ['elevation', 'slope', 'aspect', 'longitude', 'latitude', 
                    'tile_h', 'tile_v', 'pixel_row', 'pixel_col']
    
    # Spectral reflectance columns
    spectral_cols = [col for col in df.columns if col.startswith('sur_refl_b')]
    
    print("\nPivoting data...")
    
    # Create a list to store pivoted dataframes
    pivoted_data = []
    
    # Pivot each value column separately
    for value_col in value_cols:
        print(f"Pivoting {value_col}...")
        
        pivot_df = df.pivot_table(
            index=id_cols,
            columns='method',
            values=value_col,
            aggfunc='first'  # Use first value if there are duplicates
        )
        
        # Rename columns to include the original column name
        pivot_df.columns = [f'{value_col}_{method}' for method in pivot_df.columns]
        
        # Reset index to make id_cols regular columns
        pivot_df = pivot_df.reset_index()
        
        pivoted_data.append(pivot_df)
    
    # Merge all pivoted dataframes
    print("Merging pivoted data...")
    result_df = pivoted_data[0]
    
    for pivot_df in pivoted_data[1:]:
        result_df = result_df.merge(pivot_df, on=id_cols, how='outer')
    
    # Add metadata columns (take first occurrence for each unique combination)
    print("Adding metadata...")
    metadata_df = df[id_cols + metadata_cols + spectral_cols].drop_duplicates(subset=id_cols)
    
    result_df = result_df.merge(metadata_df, on=id_cols, how='left')
    
    # Sort by pixel_id and date
    result_df = result_df.sort_values(['pixel_id', 'date']).reset_index(drop=True)
    
    print(f"\nFinal data shape: {result_df.shape}")
    
    # Save the data
    print(f"Saving to {output_file}...")
    result_df.to_csv(output_file, index=False)
    
    # Summary statistics
    print("\n=== SUMMARY ===")
    print(f"Original records: {len(df)}")
    print(f"Reorganized records: {len(result_df)}")
    print(f"Unique combinations: {len(result_df[id_cols].drop_duplicates())}")
    
    # Show data availability for each method
    methods = df['method'].unique()
    print("\nData availability by method:")
    for method in methods:
        albedo_col = f'albedo_{method}'
        if albedo_col in result_df.columns:
            available = result_df[albedo_col].notna().sum()
            percentage = (available / len(result_df)) * 100
            print(f"  {method}: {available} records ({percentage:.1f}%)")
    
    # Show sample of reorganized data focusing on albedo values
    print(f"\nSample of reorganized data (showing albedo columns):")
    albedo_cols = ['pixel_id', 'date'] + [f'albedo_{method}' for method in methods if f'albedo_{method}' in result_df.columns]
    print(result_df[albedo_cols].head(10))
    
    # Show column structure
    print(f"\nTotal columns: {len(result_df.columns)}")
    print("Column groups:")
    print(f"  - ID columns: {len(id_cols)}")
    print(f"  - Metadata columns: {len(metadata_cols)}")
    print(f"  - Method-specific columns: {len([col for col in result_df.columns if any(val in col for val in value_cols)])}")
    print(f"  - Spectral columns: {len(spectral_cols)}")
    
    return result_df

if __name__ == "__main__":
    input_file = "data/csv/Athabasca_Terra_Aqua_MultiProduct_2014-01-01_to_2021-01-01.csv"
    output_file = "data/csv/Athabasca_MultiProduct_Organized.csv"
    
    reorganized_df = reorganize_data_simple(input_file, output_file)
    print(f"\n✅ Successfully created reorganized dataset: {output_file}")
    print("\nNow each method has its own columns instead of being in separate rows!")
