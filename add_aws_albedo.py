#!/usr/bin/env python3
"""
Script to add AWS albedo values to the reorganized satellite data.
Matches AWS albedo data with satellite data based on date.
"""

import pandas as pd
import numpy as np
from datetime import datetime

def add_aws_albedo_to_satellite_data(satellite_file, aws_file, output_file):
    """
    Add AWS albedo values to the reorganized satellite data.
    
    Parameters:
    satellite_file (str): Path to reorganized satellite CSV
    aws_file (str): Path to AWS albedo CSV
    output_file (str): Path for output CSV with AWS data added
    """
    
    print("Loading datasets...")
    satellite_df = pd.read_csv(satellite_file)
    aws_df = pd.read_csv(aws_file)
    
    print(f"Satellite data shape: {satellite_df.shape}")
    print(f"AWS data shape: {aws_df.shape}")
    
    # Convert AWS date format to match satellite format
    print("Processing AWS dates...")
    # AWS format: "12-Sep-2014 00:00:00" -> "2014-09-12"
    aws_df['date'] = pd.to_datetime(aws_df['Time'], format='%d-%b-%Y %H:%M:%S').dt.strftime('%Y-%m-%d')
    
    # Clean up AWS data
    aws_df = aws_df[['date', 'Albedo']].rename(columns={'Albedo': 'albedo_AWS'})
    aws_df = aws_df.dropna(subset=['albedo_AWS'])  # Remove NaN values
    
    print(f"AWS data after processing: {len(aws_df)} records")
    print(f"AWS date range: {aws_df['date'].min()} to {aws_df['date'].max()}")
    
    # Convert satellite dates to same format (should already be YYYY-MM-DD)
    print("Processing satellite dates...")
    satellite_df['date'] = pd.to_datetime(satellite_df['date']).dt.strftime('%Y-%m-%d')
    
    print(f"Satellite date range: {satellite_df['date'].min()} to {satellite_df['date'].max()}")
    
    # Find overlapping date range
    sat_dates = set(satellite_df['date'])
    aws_dates = set(aws_df['date'])
    overlap_dates = sat_dates.intersection(aws_dates)
    
    print(f"Overlapping dates: {len(overlap_dates)}")
    print(f"Date overlap range: {min(overlap_dates)} to {max(overlap_dates)}")
    
    # Merge the datasets
    print("Merging AWS albedo with satellite data...")
    result_df = satellite_df.merge(aws_df, on='date', how='left')
    
    # Count matches
    matches = result_df['albedo_AWS'].notna().sum()
    total_satellite_records = len(satellite_df)
    
    print(f"Successful matches: {matches} out of {total_satellite_records} satellite records ({matches/total_satellite_records*100:.1f}%)")
    
    # Check matches for each satellite method
    print("\nAWS matches for each satellite method:")
    methods = ['MOD09GA', 'MYD09GA', 'mcd43a3', 'mod10a1', 'myd10a1']
    
    for method in methods:
        albedo_col = f'albedo_{method}'
        if albedo_col in result_df.columns:
            # Records where both satellite method and AWS have data
            both_available = result_df[(result_df[albedo_col].notna()) & 
                                     (result_df['albedo_AWS'].notna())]
            
            method_total = result_df[albedo_col].notna().sum()
            matches_for_method = len(both_available)
            
            print(f"  {method}: {matches_for_method} matches out of {method_total} records ({matches_for_method/method_total*100:.1f}%)")
    
    # Reorder columns to put AWS albedo after the satellite albedo columns
    print("Reordering columns...")
    cols = list(result_df.columns)
    
    # Find position after albedo columns
    albedo_cols = [col for col in cols if col.startswith('albedo_') and col != 'albedo_AWS']
    
    if albedo_cols:
        # Insert AWS albedo after the last satellite albedo column
        aws_col_idx = cols.index('albedo_AWS')
        last_albedo_idx = max([cols.index(col) for col in albedo_cols])
        
        # Remove AWS column from current position
        cols.pop(aws_col_idx)
        # Insert it after last satellite albedo column
        cols.insert(last_albedo_idx + 1, 'albedo_AWS')
        
        result_df = result_df[cols]
    
    # Save the result
    print(f"Saving merged data to {output_file}...")
    result_df.to_csv(output_file, index=False)
    
    # Show sample of merged data
    print("\n=== SAMPLE OF MERGED DATA ===")
    sample_cols = ['pixel_id', 'date', 'qa_mode', 'albedo_MOD09GA', 'albedo_mcd43a3', 'albedo_AWS']
    sample = result_df[result_df['albedo_AWS'].notna()][sample_cols].head(10)
    print(sample)
    
    # Show statistics
    print(f"\n=== FINAL STATISTICS ===")
    print(f"Total records: {len(result_df)}")
    print(f"Records with AWS albedo: {result_df['albedo_AWS'].notna().sum()}")
    print(f"AWS albedo range: {result_df['albedo_AWS'].min():.3f} to {result_df['albedo_AWS'].max():.3f}")
    
    # Show some comparison examples
    print(f"\n=== COMPARISON EXAMPLES ===")
    comparison_data = result_df[(result_df['albedo_MOD09GA'].notna()) & 
                               (result_df['albedo_AWS'].notna())][['date', 'albedo_MOD09GA', 'albedo_AWS']].head(5)
    if len(comparison_data) > 0:
        print("MOD09GA vs AWS albedo examples:")
        comparison_data['difference'] = comparison_data['albedo_MOD09GA'] - comparison_data['albedo_AWS']
        print(comparison_data)
    
    return result_df

if __name__ == "__main__":
    satellite_file = "data/csv/Athabasca_MultiProduct_Organized.csv"
    aws_file = "data/csv/iceAWS_Atha_albedo_daily_20152020_filled_clean.csv"
    output_file = "data/csv/Athabasca_MultiProduct_with_AWS.csv"
    
    merged_df = add_aws_albedo_to_satellite_data(satellite_file, aws_file, output_file)
    print(f"\n✅ Successfully merged AWS albedo data!")
    print(f"Output saved to: {output_file}")
    print("Now you can compare satellite methods directly with AWS ground truth!")
