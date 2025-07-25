#!/usr/bin/env python3
"""
Example Usage of Improved Data Handler
======================================

This script demonstrates various ways to use the improved data handler
for albedo data import and merging with different configurations.
"""

import pandas as pd
import numpy as np
from improved_data_handler import (
    DataConfig, DataLoader, DataMerger, 
    load_and_merge_data, validate_merged_data
)
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def example_basic_usage():
    """Example 1: Basic usage with default configuration."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Usage with Default Configuration")
    print("="*60)
    
    try:
        # Load data with default settings
        merged_data = load_and_merge_data()
        
        # Validate the data
        validation = validate_merged_data(merged_data)
        
        print(f"✓ Dataset shape: {merged_data.shape}")
        print(f"✓ Date range: {validation['date_range'][0]} to {validation['date_range'][1]}")
        print(f"✓ Total observations: {validation['total_observations']}")
        
        return merged_data
        
    except Exception as e:
        print(f"✗ Basic usage failed: {str(e)}")
        return None


def example_custom_configuration():
    """Example 2: Custom configuration with different file paths."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Custom Configuration")
    print("="*60)
    
    try:
        # Create custom configuration
        custom_config = DataConfig(
            modis_file='data/csv/athabasca_2014-01-01_to_2021-01-01.csv',
            aws_file='data/csv/iceAWS_Atha_albedo_daily_20152020_filled_clean.csv',
            albedo_range=(0.0, 1.0),
            outlier_threshold=2.5,
            min_observations=5
        )
        
        # Load data with custom configuration
        merged_data = load_and_merge_data(
            config=custom_config,
            merge_strategy='outer',  # Use outer join to keep all data
            add_temporal_features=True
        )
        
        print(f"✓ Custom config dataset shape: {merged_data.shape}")
        print(f"✓ Available columns: {list(merged_data.columns)}")
        
        return merged_data
        
    except Exception as e:
        print(f"✗ Custom configuration failed: {str(e)}")
        return None


def example_step_by_step():
    """Example 3: Step-by-step loading and merging."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Step-by-Step Loading and Merging")
    print("="*60)
    
    try:
        # Initialize loader and merger
        config = DataConfig()
        loader = DataLoader(config)
        merger = DataMerger(config)
        
        # Step 1: Load MODIS data
        print("Step 1: Loading MODIS data...")
        modis_data = loader.load_modis_data()
        print(f"✓ MODIS data loaded: {modis_data.shape}")
        
        # Step 2: Load AWS data
        print("Step 2: Loading AWS data...")
        aws_data = loader.load_aws_data()
        print(f"✓ AWS data loaded: {aws_data.shape}")
        
        # Step 3: Create pivot table
        print("Step 3: Creating pivot table...")
        modis_pivot = merger.create_optimized_pivot(modis_data)
        print(f"✓ Pivot table created: {modis_pivot.shape}")
        
        # Step 4: Merge with AWS
        print("Step 4: Merging with AWS data...")
        merged_data = merger.merge_with_aws(modis_pivot, aws_data, merge_strategy='inner')
        print(f"✓ Data merged: {merged_data.shape}")
        
        # Step 5: Add temporal features
        print("Step 5: Adding temporal features...")
        final_data = merger.add_temporal_features(merged_data)
        print(f"✓ Final dataset: {final_data.shape}")
        
        return final_data
        
    except Exception as e:
        print(f"✗ Step-by-step process failed: {str(e)}")
        return None


def example_different_merge_strategies():
    """Example 4: Compare different merge strategies."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Different Merge Strategies Comparison")
    print("="*60)
    
    strategies = ['inner', 'outer', 'left', 'right']
    results = {}
    
    for strategy in strategies:
        try:
            merged_data = load_and_merge_data(
                merge_strategy=strategy,
                add_temporal_features=False  # Skip for speed
            )
            
            validation = validate_merged_data(merged_data)
            results[strategy] = {
                'shape': merged_data.shape,
                'total_obs': validation['total_observations'],
                'coverage': validation['method_coverage']
            }
            
            print(f"✓ {strategy.upper()} merge: {merged_data.shape[0]} observations")
            
        except Exception as e:
            print(f"✗ {strategy.upper()} merge failed: {str(e)}")
            results[strategy] = None
    
    # Summary comparison
    print("\nMERGE STRATEGY COMPARISON:")
    print("-" * 40)
    for strategy, result in results.items():
        if result:
            print(f"{strategy.upper():<8}: {result['total_obs']} observations")
    
    return results


def example_data_validation():
    """Example 5: Comprehensive data validation."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Comprehensive Data Validation")
    print("="*60)
    
    try:
        # Load data
        merged_data = load_and_merge_data()
        
        # Validate the merged data
        validation = validate_merged_data(merged_data)
        
        print("DATA VALIDATION REPORT:")
        print("-" * 30)
        
        # Basic info
        print(f"Total observations: {validation['total_observations']}")
        print(f"Date range: {validation['date_range'][0]} to {validation['date_range'][1]}")
        
        # Method coverage
        print("\nMethod Coverage:")
        for method, stats in validation['method_coverage'].items():
            print(f"  {method:<10}: {stats['valid_observations']:>4} obs ({stats['coverage_percentage']:>5.1f}%)")
        
        # Outlier detection
        print("\nOutlier Detection (values outside [0,1]):")
        total_outliers = sum(validation['outlier_counts'].values())
        if total_outliers == 0:
            print("  ✓ No outliers detected")
        else:
            for method, count in validation['outlier_counts'].items():
                if count > 0:
                    print(f"  ⚠ {method}: {count} outliers")
        
        # Data completeness analysis
        print(f"\nData Completeness Analysis:")
        complete_cases = merged_data.dropna().shape[0]
        completeness_rate = (complete_cases / len(merged_data)) * 100
        print(f"  Complete cases: {complete_cases}/{len(merged_data)} ({completeness_rate:.1f}%)")
        
        # Missing data by column
        print(f"\nMissing Data by Column:")
        for col in ['MCD43A3', 'MOD09GA', 'MOD10A1', 'AWS']:
            if col in merged_data.columns:
                missing = merged_data[col].isnull().sum()
                missing_pct = (missing / len(merged_data)) * 100
                print(f"  {col:<10}: {missing:>4} missing ({missing_pct:>5.1f}%)")
        
        return validation
        
    except Exception as e:
        print(f"✗ Data validation failed: {str(e)}")
        return None


def example_performance_comparison():
    """Example 6: Performance comparison with original approach."""
    print("\n" + "="*60)
    print("EXAMPLE 6: Performance Analysis")
    print("="*60)
    
    import time
    
    try:
        # Time the improved data loader
        start_time = time.time()
        merged_data = load_and_merge_data()
        load_time = time.time() - start_time
        
        print(f"✓ Data loading completed in {load_time:.2f} seconds")
        print(f"✓ Memory usage: ~{merged_data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        print(f"✓ Dataset efficiency: {merged_data.shape[0] / load_time:.0f} rows/second")
        
        # Analyze data types for memory efficiency
        print(f"\nData Types Analysis:")
        for col, dtype in merged_data.dtypes.items():
            print(f"  {col:<15}: {dtype}")
        
        return load_time
        
    except Exception as e:
        print(f"✗ Performance analysis failed: {str(e)}")
        return None


def main():
    """Run all examples to demonstrate the improved data handler."""
    print("IMPROVED DATA HANDLER DEMONSTRATION")
    print("=" * 70)
    print("This script demonstrates the enhanced data import and merging capabilities.")
    
    # Run all examples
    examples = [
        example_basic_usage,
        example_custom_configuration,
        example_step_by_step,
        example_different_merge_strategies,
        example_data_validation,
        example_performance_comparison
    ]
    
    results = {}
    for i, example_func in enumerate(examples, 1):
        try:
            result = example_func()
            results[example_func.__name__] = result
        except Exception as e:
            logger.error(f"Example {i} failed: {str(e)}")
            results[example_func.__name__] = None
    
    # Final summary
    print("\n" + "="*70)
    print("DEMONSTRATION SUMMARY")
    print("="*70)
    
    successful = sum(1 for result in results.values() if result is not None)
    total = len(examples)
    
    print(f"Successfully completed: {successful}/{total} examples")
    
    if successful == total:
        print("✓ All examples completed successfully!")
        print("The improved data handler is ready for use.")
    else:
        print("⚠ Some examples failed. Check the error messages above.")
    
    print("\nKey Improvements Demonstrated:")
    print("- ✓ Robust error handling and validation")
    print("- ✓ Flexible configuration system")
    print("- ✓ Multiple merge strategies")
    print("- ✓ Automatic data quality checks")
    print("- ✓ Memory-efficient data loading")
    print("- ✓ Comprehensive logging and monitoring")
    print("- ✓ Temporal feature engineering")


if __name__ == "__main__":
    main()