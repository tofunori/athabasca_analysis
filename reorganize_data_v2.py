#!/usr/bin/env python3
"""
Improved script to reorganize the multi-method data from long format to wide format.
This version handles the metadata columns better to avoid duplicates.
"""

import pandas as pd
import numpy as np

def reorganize_multimethod_data_v2(input_file, output_file):
    """
    Reorganize data from long format (methods in rows) to wide format (methods as columns).
    """
    
    print("Loading data...")
    df = pd.read_csv(input_file)
    
    print(f"Original data shape: {df.shape}")
    print(f"Available methods: {df['method'].unique()}")
    
    # Method-specific columns that should be pivoted
    method_specific_cols = ['albedo', 'ndsi', 'glacier_fraction', 'solar_zenith']
    
    # Metadata columns that should remain single columns
    id_cols = ['pixel_id', 'date']
    metadata_cols = ['qa_mode', 'elevation', 'slope', 'aspect', 
                    'longitude', 'latitude', 'tile_h', 'tile_v', 
                    'pixel_row', 'pixel_col']
    
    # Spectral reflectance columns
    spectral_cols = [col for col in df.columns if col.startswith('sur_refl_b')]
    
    # Create pivot for method-specific data
    print("Creating pivot for method-specific data...")
    
    # Pivot each method-specific column separately
    pivoted_dfs = []
    
    for col in method_specific_cols:
        if col in df.columns:
            pivot_df = df.pivot_table(
                index=id_cols, 
                columns='method', 
                values=col, 
                aggfunc='first'  # Take first value if duplicates exist
            ).reset_index()
            
            # Flatten column names
            pivot_df.columns = [f'{col}_{method_name}' if method_name != '' else col_name 
                              for col_name, method_name in pivot_df.columns]
            
            # Fix the id columns names (they get modified by pivot)
            for id_col in id_cols:
                if id_col in pivot_df.columns:
                    continue
                # The id columns might have been renamed, find them
                for c in pivot_df.columns:
                    if id_col in str(c):
                        pivot_df = pivot_df.rename(columns={c: id_col})
                        break
            
            pivoted_dfs.append(pivot_df)
    
    # Merge all pivoted dataframes
    print("Merging pivoted data...")
    result_df = pivoted_dfs[0]
    for pivot_df in pivoted_dfs[1:]:
        result_df = result_df.merge(pivot_df, on=id_cols, how='outer')
    
    # Add metadata (take first occurrence for each pixel_id/date combination)
    print("Adding metadata...")
    metadata_df = df[id_cols + metadata_cols].drop_duplicates(subset=id_cols)
    result_df = result_df.merge(metadata_df, on=id_cols, how='left')
    
    # Add spectral data
    if spectral_cols:
        print("Adding spectral data...")
        spectral_df = df[id_cols + spectral_cols].drop_duplicates(subset=id_cols)
        result_df = result_df.merge(spectral_df, on=id_cols, how='left')
    
    # Reorder columns for better organization
    # Start with id columns
    column_order = id_cols.copy()
    
    # Add metadata columns
    column_order.extend(metadata_cols)
    
    # Add method-specific columns grouped by method
    methods = df['method'].unique()
    for method in methods:
        for col in method_specific_cols:
            method_col = f'{col}_{method}'
            if method_col in result_df.columns:
                column_order.append(method_col)
    
    # Add spectral columns
    column_order.extend(spectral_cols)
    
    # Reorder dataframe
    result_df = result_df[column_order]
    
    # Sort by pixel_id and date
    result_df = result_df.sort_values(id_cols).reset_index(drop=True)
    
    print(f"\nFinal reorganized data shape: {result_df.shape}")
    
    # Save the reorganized data
    print(f"Saving to {output_file}...")
    result_df.to_csv(output_file, index=False)
    
    # Print summary
    print("\n=== REORGANIZATION SUMMARY ===")
    print(f"Original records: {len(df)}")
    print(f"Reorganized records: {len(result_df)}")
    
    print("\nData availability by method:")
    for method in methods:
        albedo_col = f'albedo_{method}'
        if albedo_col in result_df.columns:
            non_null_count = result_df[albedo_col].notna().sum()
            print(f"  {method}: {non_null_count} records ({non_null_count/len(result_df)*100:.1f}%)")
    
    # Show sample of the reorganized data
    print(f"\nSample of reorganized data:")
    display_cols = ['pixel_id', 'date'] + [f'albedo_{method}' for method in methods[:3]]
    print(result_df[display_cols].head(10))
    
    print(f"\nColumn structure:")
    print("ID & Metadata columns:")
    for col in id_cols + metadata_cols:
        print(f"  - {col}")
    
    print("\nMethod-specific columns:")
    for method in methods:
        print(f"  {method}:")
        for col in method_specific_cols:
            method_col = f'{col}_{method}'
            if method_col in result_df.columns:
                print(f"    - {method_col}")
    
    if spectral_cols:
        print(f"\nSpectral columns: {len(spectral_cols)} bands")
    
    return result_df

if __name__ == "__main__":
    input_file = "data/csv/Athabasca_Terra_Aqua_MultiProduct_2014-01-01_to_2021-01-01.csv"
    output_file = "data/csv/Athabasca_MultiProduct_Organized.csv"
    
    reorganized_df = reorganize_multimethod_data_v2(input_file, output_file)
    print(f"\n✅ Data successfully reorganized and saved to {output_file}")
