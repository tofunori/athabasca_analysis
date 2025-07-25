#!/usr/bin/env python3
"""
Script to reorganize the multi-method data from long format to wide format.
Instead of having methods in rows, each method will have its own columns.
"""

import pandas as pd
import numpy as np

def reorganize_multimethod_data(input_file, output_file):
    """
    Reorganize data from long format (methods in rows) to wide format (methods as columns).
    
    Parameters:
    input_file (str): Path to input CSV file
    output_file (str): Path to output CSV file
    """
    
    print("Loading data...")
    df = pd.read_csv(input_file)
    
    print(f"Original data shape: {df.shape}")
    print(f"Available methods: {df['method'].unique()}")
    print(f"Method counts:\n{df['method'].value_counts()}")
    
    # Identify the columns that vary by method (measurement columns)
    method_specific_cols = ['albedo', 'ndsi', 'glacier_fraction', 'solar_zenith']
    
    # Identify columns that should be the same for all methods (metadata columns)
    # These will be kept as single columns
    metadata_cols = ['pixel_id', 'date', 'qa_mode', 'elevation', 'slope', 'aspect', 
                    'longitude', 'latitude', 'tile_h', 'tile_v', 'pixel_row', 'pixel_col']
    
    # Spectral reflectance columns (these should be the same for Terra/Aqua products on same date)
    spectral_cols = [col for col in df.columns if col.startswith('sur_refl_b')]
    
    print("\nReorganizing data...")
    
    # Create the pivot table
    # First, let's create a base dataframe with metadata
    base_df = df[metadata_cols].drop_duplicates().reset_index(drop=True)
    
    print(f"Base metadata shape: {base_df.shape}")
    
    # For each method, create columns with method suffix
    method_dfs = []
    
    for method in df['method'].unique():
        method_data = df[df['method'] == method].copy()
        
        # Rename method-specific columns to include method name
        rename_dict = {}
        for col in method_specific_cols:
            if col in method_data.columns:
                rename_dict[col] = f'{col}_{method}'
        
        method_data = method_data.rename(columns=rename_dict)
        
        # Select relevant columns for this method
        method_cols = metadata_cols + [f'{col}_{method}' for col in method_specific_cols if f'{col}_{method}' in method_data.columns]
        method_df = method_data[method_cols]
        
        method_dfs.append(method_df)
        print(f"Method {method}: {len(method_df)} records")
    
    # Merge all method dataframes
    print("\nMerging all methods...")
    result_df = base_df.copy()
    
    for method_df in method_dfs:
        result_df = result_df.merge(method_df, on=metadata_cols, how='outer')
    
    # Add spectral reflectance data (taking from any available record for each pixel_id/date)
    if spectral_cols:
        spectral_data = df[metadata_cols + spectral_cols].drop_duplicates(subset=['pixel_id', 'date'])
        result_df = result_df.merge(spectral_data, on=['pixel_id', 'date'], how='left')
    
    print(f"\nFinal reorganized data shape: {result_df.shape}")
    
    # Sort by pixel_id and date for better organization
    result_df = result_df.sort_values(['pixel_id', 'date']).reset_index(drop=True)
    
    # Save the reorganized data
    print(f"Saving to {output_file}...")
    result_df.to_csv(output_file, index=False)
    
    # Print summary statistics
    print("\n=== REORGANIZATION SUMMARY ===")
    print(f"Original records: {len(df)}")
    print(f"Reorganized records: {len(result_df)}")
    print(f"Unique pixel_id/date combinations: {len(result_df[['pixel_id', 'date']].drop_duplicates())}")
    
    # Show data availability for each method
    print("\nData availability by method:")
    for method in df['method'].unique():
        albedo_col = f'albedo_{method}'
        if albedo_col in result_df.columns:
            non_null_count = result_df[albedo_col].notna().sum()
            print(f"  {method}: {non_null_count} records ({non_null_count/len(result_df)*100:.1f}%)")
    
    # Show first few rows
    print(f"\nFirst 5 rows of reorganized data:")
    print(result_df.head())
    
    print(f"\nColumn names in reorganized data:")
    for i, col in enumerate(result_df.columns):
        print(f"  {i+1:2d}. {col}")
    
    return result_df

if __name__ == "__main__":
    input_file = "data/csv/Athabasca_Terra_Aqua_MultiProduct_2014-01-01_to_2021-01-01.csv"
    output_file = "data/csv/Athabasca_MultiProduct_Organized.csv"
    
    reorganized_df = reorganize_multimethod_data(input_file, output_file)
    print(f"\n✅ Data successfully reorganized and saved to {output_file}")
