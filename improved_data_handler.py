"""
Improved Data Handler for Albedo Analysis
==========================================

This module provides enhanced data loading, validation, and merging capabilities
for albedo analysis workflows with better error handling, performance, and maintainability.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from functools import lru_cache
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration class for data file paths and column mappings."""
    
    # File paths
    modis_file: str = 'data/csv/athabasca_2014-01-01_to_2021-01-01.csv'
    aws_file: str = 'data/csv/iceAWS_Atha_albedo_daily_20152020_filled_clean.csv'
    
    # Column mappings for flexibility
    modis_columns: Dict[str, str] = None
    aws_columns: Dict[str, str] = None
    
    # Data validation parameters
    albedo_range: Tuple[float, float] = (0.0, 1.0)
    outlier_threshold: float = 3.0  # Standard deviations
    min_observations: int = 10
    
    def __post_init__(self):
        if self.modis_columns is None:
            self.modis_columns = {
                'date': 'date',
                'method': 'method', 
                'albedo': 'albedo_value',
                'albedo_alt': 'albedo',  # Alternative column name
                'pixel_id': 'pixel_id',
                'latitude': 'latitude',
                'longitude': 'longitude'
            }
        
        if self.aws_columns is None:
            self.aws_columns = {
                'date': 'Time',
                'albedo': 'Albedo'
            }


class DataLoader:
    """Enhanced data loader with validation and error handling."""
    
    def __init__(self, config: DataConfig = None):
        self.config = config or DataConfig()
        self._modis_data = None
        self._aws_data = None
    
    def _validate_file_exists(self, filepath: str) -> Path:
        """Validate that file exists and is readable."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {filepath}")
        return path
    
    def _standardize_modis_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize MODIS column names for consistency."""
        df = df.copy()
        
        # Handle albedo column name variations
        albedo_col = None
        if self.config.modis_columns['albedo'] in df.columns:
            albedo_col = self.config.modis_columns['albedo']
        elif self.config.modis_columns['albedo_alt'] in df.columns:
            albedo_col = self.config.modis_columns['albedo_alt']
            df = df.rename(columns={albedo_col: self.config.modis_columns['albedo']})
            logger.info(f"Renamed column '{albedo_col}' to '{self.config.modis_columns['albedo']}'")
        
        if albedo_col is None:
            available_cols = list(df.columns)
            raise ValueError(f"No albedo column found. Available columns: {available_cols}")
        
        # Standardize method names to uppercase
        if 'method' in df.columns:
            df['method'] = df['method'].str.upper()
        
        return df
    
    def _validate_data_quality(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Validate data quality and log warnings for issues."""
        df = df.copy()
        initial_rows = len(df)
        
        # Check for completely empty rows
        empty_rows = df.isnull().all(axis=1).sum()
        if empty_rows > 0:
            logger.warning(f"{data_type}: Found {empty_rows} completely empty rows")
            df = df.dropna(how='all')
        
        # Validate albedo values are in reasonable range
        albedo_cols = [col for col in df.columns if 'albedo' in col.lower()]
        for col in albedo_cols:
            if col in df.columns:
                invalid_range = (
                    (df[col] < self.config.albedo_range[0]) | 
                    (df[col] > self.config.albedo_range[1])
                ).sum()
                if invalid_range > 0:
                    logger.warning(f"{data_type}: {invalid_range} values in {col} outside range {self.config.albedo_range}")
        
        final_rows = len(df)
        if final_rows != initial_rows:
            logger.info(f"{data_type}: Cleaned {initial_rows - final_rows} rows, {final_rows} remaining")
        
        return df
    
    @lru_cache(maxsize=1)
    def load_modis_data(self, reload: bool = False) -> pd.DataFrame:
        """Load and preprocess MODIS data with caching."""
        if self._modis_data is not None and not reload:
            return self._modis_data.copy()
        
        logger.info(f"Loading MODIS data from {self.config.modis_file}")
        
        # Validate file exists
        self._validate_file_exists(self.config.modis_file)
        
        try:
            # Load data with optimized dtypes
            dtype_dict = {
                'method': 'category',
                'pixel_id': 'int32',
                'tile_h': 'int16',
                'tile_v': 'int16'
            }
            
            df = pd.read_csv(self.config.modis_file, dtype=dtype_dict, low_memory=False)
            
            # Standardize columns
            df = self._standardize_modis_columns(df)
            
            # Convert date column
            df['date'] = pd.to_datetime(df[self.config.modis_columns['date']], errors='coerce')
            
            # Remove rows with invalid dates
            invalid_dates = df['date'].isnull().sum()
            if invalid_dates > 0:
                logger.warning(f"Removing {invalid_dates} rows with invalid dates")
                df = df.dropna(subset=['date'])
            
            # Validate data quality
            df = self._validate_data_quality(df, "MODIS")
            
            self._modis_data = df
            logger.info(f"Successfully loaded MODIS data: {len(df)} rows, {len(df.columns)} columns")
            
            return df.copy()
            
        except Exception as e:
            logger.error(f"Failed to load MODIS data: {str(e)}")
            raise
    
    @lru_cache(maxsize=1) 
    def load_aws_data(self, reload: bool = False) -> pd.DataFrame:
        """Load and preprocess AWS data with caching."""
        if self._aws_data is not None and not reload:
            return self._aws_data.copy()
        
        logger.info(f"Loading AWS data from {self.config.aws_file}")
        
        # Validate file exists
        self._validate_file_exists(self.config.aws_file)
        
        try:
            df = pd.read_csv(self.config.aws_file, low_memory=False)
            
            # Convert date column
            date_col = self.config.aws_columns['date']
            if date_col not in df.columns:
                raise ValueError(f"AWS date column '{date_col}' not found in data")
            
            df['date'] = pd.to_datetime(df[date_col], errors='coerce')
            
            # Remove rows with invalid dates
            invalid_dates = df['date'].isnull().sum()
            if invalid_dates > 0:
                logger.warning(f"Removing {invalid_dates} rows with invalid dates")
                df = df.dropna(subset=['date'])
            
            # Validate data quality
            df = self._validate_data_quality(df, "AWS")
            
            self._aws_data = df
            logger.info(f"Successfully loaded AWS data: {len(df)} rows, {len(df.columns)} columns")
            
            return df.copy()
            
        except Exception as e:
            logger.error(f"Failed to load AWS data: {str(e)}")
            raise


class DataMerger:
    """Enhanced data merger with optimized merging strategies."""
    
    def __init__(self, config: DataConfig = None):
        self.config = config or DataConfig()
        self.loader = DataLoader(config)
    
    def create_optimized_pivot(self, modis_df: pd.DataFrame, 
                              aggregation_func: str = 'mean') -> pd.DataFrame:
        """Create an optimized pivot table from MODIS data."""
        logger.info("Creating optimized pivot table from MODIS data")
        
        # Filter to only required columns to reduce memory
        required_cols = ['date', 'method', self.config.modis_columns['albedo']]
        available_cols = [col for col in required_cols if col in modis_df.columns]
        
        if len(available_cols) != len(required_cols):
            missing = set(required_cols) - set(available_cols)
            raise ValueError(f"Missing required columns: {missing}")
        
        # Create pivot with specified aggregation
        pivot_df = modis_df[available_cols].pivot_table(
            index='date', 
            columns='method', 
            values=self.config.modis_columns['albedo'], 
            aggfunc=aggregation_func
        )
        
        logger.info(f"Created pivot table: {len(pivot_df)} dates, {len(pivot_df.columns)} methods")
        return pivot_df
    
    def merge_with_aws(self, modis_pivot: pd.DataFrame, aws_df: pd.DataFrame,
                      merge_strategy: str = 'inner') -> pd.DataFrame:
        """Merge MODIS pivot data with AWS data using specified strategy."""
        logger.info(f"Merging MODIS pivot with AWS data using '{merge_strategy}' strategy")
        
        # Prepare AWS data for merging
        aws_cols = ['date', self.config.aws_columns['albedo']]
        aws_merge = aws_df[aws_cols].copy()
        aws_merge = aws_merge.rename(columns={self.config.aws_columns['albedo']: 'AWS'})
        
        # Remove any duplicate dates in AWS data
        duplicates = aws_merge.duplicated(subset=['date']).sum()
        if duplicates > 0:
            logger.warning(f"Removing {duplicates} duplicate dates from AWS data")
            aws_merge = aws_merge.drop_duplicates(subset=['date'], keep='first')
        
        # Perform merge
        merged = pd.merge(modis_pivot, aws_merge, on='date', how=merge_strategy)
        
        if merge_strategy == 'inner':
            logger.info(f"Inner merge resulted in {len(merged)} overlapping observations")
        else:
            logger.info(f"{merge_strategy.title()} merge resulted in {len(merged)} total observations")
        
        # Set date as index for time series analysis
        merged = merged.set_index('date')
        
        return merged
    
    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features to the merged dataset."""
        df = df.copy()
        
        # Add month, season, and year features
        df['month'] = df.index.month
        df['year'] = df.index.year
        df['day_of_year'] = df.index.dayofyear
        
        # Add season (meteorological seasons)
        def get_season(month):
            if month in [12, 1, 2]:
                return 'Winter'
            elif month in [3, 4, 5]:
                return 'Spring'
            elif month in [6, 7, 8]:
                return 'Summer'
            else:
                return 'Fall'
        
        df['season'] = df['month'].apply(get_season)
        
        logger.info("Added temporal features: month, year, day_of_year, season")
        return df
    
    def create_comprehensive_dataset(self, merge_strategy: str = 'inner',
                                   add_temporal: bool = True,
                                   aggregation_func: str = 'mean') -> pd.DataFrame:
        """Create a comprehensive merged dataset with all enhancements."""
        logger.info("Creating comprehensive merged dataset")
        
        # Load data
        modis_df = self.loader.load_modis_data()
        aws_df = self.loader.load_aws_data()
        
        # Create optimized pivot
        modis_pivot = self.create_optimized_pivot(modis_df, aggregation_func)
        
        # Merge with AWS
        merged = self.merge_with_aws(modis_pivot, aws_df, merge_strategy)
        
        # Add temporal features if requested
        if add_temporal:
            merged = self.add_temporal_features(merged)
        
        # Data quality summary
        total_obs = len(merged)
        complete_cases = merged.dropna().shape[0]
        logger.info(f"Final dataset: {total_obs} observations, {complete_cases} complete cases")
        
        return merged


# Convenience functions for backward compatibility and easy usage
def load_and_merge_data(config: DataConfig = None, 
                       merge_strategy: str = 'inner',
                       add_temporal_features: bool = True) -> pd.DataFrame:
    """
    Convenience function to load and merge all data in one call.
    
    Parameters:
    -----------
    config : DataConfig, optional
        Configuration object with file paths and column mappings
    merge_strategy : str, default 'inner'
        Merge strategy ('inner', 'outer', 'left', 'right')
    add_temporal_features : bool, default True
        Whether to add temporal features (month, season, etc.)
    
    Returns:
    --------
    pd.DataFrame
        Merged dataset with MODIS and AWS data
    """
    merger = DataMerger(config)
    return merger.create_comprehensive_dataset(
        merge_strategy=merge_strategy,
        add_temporal=add_temporal_features
    )


def validate_merged_data(df: pd.DataFrame, 
                        methods: List[str] = None,
                        aws_col: str = 'AWS') -> Dict:
    """
    Validate the quality of merged data and return summary statistics.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Merged dataset to validate
    methods : List[str], optional
        List of MODIS methods to validate
    aws_col : str, default 'AWS'
        Name of AWS albedo column
    
    Returns:
    --------
    Dict
        Dictionary with validation results and summary statistics
    """
    if methods is None:
        methods = ['MCD43A3', 'MOD09GA', 'MOD10A1']
    
    validation_results = {
        'total_observations': len(df),
        'date_range': (df.index.min(), df.index.max()),
        'method_coverage': {},
        'data_completeness': {},
        'outlier_counts': {}
    }
    
    # Check coverage for each method
    for method in methods:
        if method in df.columns:
            valid_count = df[method].notna().sum()
            validation_results['method_coverage'][method] = {
                'valid_observations': valid_count,
                'coverage_percentage': (valid_count / len(df)) * 100
            }
    
    # Check AWS data completeness
    if aws_col in df.columns:
        aws_valid = df[aws_col].notna().sum()
        validation_results['method_coverage'][aws_col] = {
            'valid_observations': aws_valid,
            'coverage_percentage': (aws_valid / len(df)) * 100
        }
    
    # Check for potential outliers (values outside [0, 1] range)
    for col in df.columns:
        if col in methods + [aws_col]:
            outliers = ((df[col] < 0) | (df[col] > 1)).sum()
            validation_results['outlier_counts'][col] = outliers
    
    return validation_results


if __name__ == "__main__":
    # Example usage
    try:
        # Load data with default configuration
        merged_data = load_and_merge_data()
        
        # Validate the merged data
        validation = validate_merged_data(merged_data)
        
        print("Data loading and merging completed successfully!")
        print(f"Dataset shape: {merged_data.shape}")
        print(f"Date range: {validation['date_range'][0]} to {validation['date_range'][1]}")
        
        # Display coverage information
        print("\nMethod coverage:")
        for method, stats in validation['method_coverage'].items():
            print(f"  {method}: {stats['valid_observations']} obs ({stats['coverage_percentage']:.1f}%)")
        
    except Exception as e:
        logger.error(f"Failed to load and merge data: {str(e)}")
        raise