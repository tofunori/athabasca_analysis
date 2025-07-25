#!/usr/bin/env python3
"""
Improved Albedo Comparison Analysis
===================================

This script provides an enhanced version of the albedo comparison analysis
using the improved data handler for better performance and maintainability.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import logging
from pathlib import Path

# Import our improved data handler
from improved_data_handler import DataConfig, load_and_merge_data, validate_merged_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional spatial libraries
try:
    import geopandas as gpd
    import contextily as ctx
    from shapely.geometry import Point
    SPATIAL_LIBS = True
except ImportError:
    SPATIAL_LIBS = False
    logger.warning("Spatial libraries not available. Install geopandas and contextily for spatial analysis.")


class AlbedoAnalyzer:
    """Enhanced albedo analysis with improved data handling."""
    
    def __init__(self, config: DataConfig = None):
        self.config = config or DataConfig()
        self.merged_data = None
        self.stats_results = {}
        
    def load_data(self, merge_strategy: str = 'inner'):
        """Load and merge data using the improved data handler."""
        logger.info("Loading and merging data...")
        
        try:
            self.merged_data = load_and_merge_data(
                config=self.config,
                merge_strategy=merge_strategy,
                add_temporal_features=True
            )
            
            # Validate the merged data
            validation = validate_merged_data(self.merged_data)
            
            logger.info(f"Dataset loaded successfully: {self.merged_data.shape}")
            logger.info(f"Date range: {validation['date_range'][0]} to {validation['date_range'][1]}")
            
            # Display coverage information
            for method, stats in validation['method_coverage'].items():
                logger.info(f"{method}: {stats['valid_observations']} obs ({stats['coverage_percentage']:.1f}%)")
                
            return self.merged_data
            
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise
    
    def calculate_statistics(self, methods: list = None):
        """Calculate comprehensive statistics for all methods."""
        if self.merged_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        if methods is None:
            methods = ['MCD43A3', 'MOD09GA', 'MOD10A1']
        
        logger.info("Calculating statistics for all methods...")
        
        for method in methods:
            if method not in self.merged_data.columns:
                logger.warning(f"Method {method} not found in data")
                continue
                
            # Create mask for valid data pairs
            mask = self.merged_data[[method, 'AWS']].notna().all(axis=1)
            
            if mask.sum() == 0:
                logger.warning(f"No valid data pairs for {method}")
                continue
            
            x = self.merged_data.loc[mask, method]
            y = self.merged_data.loc[mask, 'AWS']
            
            # Calculate statistics
            try:
                r, p = stats.pearsonr(x, y)
                rmse = np.sqrt(np.mean((x - y)**2))
                mae = np.mean(np.abs(x - y))
                bias = np.mean(x - y)
                
                # Calculate additional metrics
                slope, intercept, _, _, _ = stats.linregress(y, x)  # y as predictor
                
                self.stats_results[method] = {
                    'n': mask.sum(),
                    'r': r,
                    'p': p,
                    'rmse': rmse,
                    'mae': mae,
                    'bias': bias,
                    'slope': slope,
                    'intercept': intercept,
                    'mean_modis': x.mean(),
                    'mean_aws': y.mean(),
                    'std_modis': x.std(),
                    'std_aws': y.std()
                }
                
                logger.info(f"{method}: n={mask.sum()}, r={r:.3f}, RMSE={rmse:.3f}")
                
            except Exception as e:
                logger.error(f"Failed to calculate statistics for {method}: {str(e)}")
                
        return self.stats_results
    
    def create_overall_comparison_plots(self, save_path: str = 'albedo_comparison_improved.png'):
        """Create comprehensive comparison plots."""
        if self.merged_data is None or not self.stats_results:
            raise ValueError("Data not loaded or statistics not calculated.")
        
        logger.info("Creating overall comparison plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        methods = ['MCD43A3', 'MOD09GA', 'MOD10A1']
        colors = ['blue', 'green', 'red']
        
        # Scatter plots for each method vs AWS
        for i, (method, color) in enumerate(zip(methods, colors)):
            if method not in self.stats_results:
                continue
                
            mask = self.merged_data[[method, 'AWS']].notna().all(axis=1)
            if mask.sum() == 0:
                continue
                
            x = self.merged_data.loc[mask, 'AWS']
            y = self.merged_data.loc[mask, method]
            
            # Scatter plot
            axes[i].scatter(x, y, alpha=0.6, s=20, color=color, edgecolors='black', linewidth=0.5)
            
            # 1:1 line
            axes[i].plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.7, label='1:1 line')
            
            # Regression line
            stats_data = self.stats_results[method]
            x_line = np.linspace(0, 1, 100)
            y_line = stats_data['slope'] * x_line + stats_data['intercept']
            axes[i].plot(x_line, y_line, 'r-', lw=2, alpha=0.8, label=f'Regression')
            
            # Formatting
            axes[i].set_xlabel('AWS Albedo', fontsize=12)
            axes[i].set_ylabel(f'{method} Albedo', fontsize=12)
            axes[i].set_title(
                f'{method} vs AWS\n'
                f'n={stats_data["n"]}, r={stats_data["r"]:.3f}, '
                f'RMSE={stats_data["rmse"]:.3f}',
                fontsize=11
            )
            axes[i].set_xlim(0, 1)
            axes[i].set_ylim(0, 1)
            axes[i].grid(True, alpha=0.3)
            axes[i].legend(fontsize=10)
        
        # Time series plot
        ax = axes[3]
        merged_plot = self.merged_data.dropna(subset=['AWS'])
        
        # Plot AWS data
        ax.plot(merged_plot.index, merged_plot['AWS'], 'k-', lw=2, 
               label='AWS', alpha=0.8, zorder=3)
        
        # Plot MODIS data
        for method, color in zip(methods, colors):
            if method in merged_plot.columns:
                valid_data = merged_plot[merged_plot[method].notna()]
                ax.scatter(valid_data.index, valid_data[method], 
                          s=15, alpha=0.7, label=method, color=color, zorder=2)
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Albedo', fontsize=12)
        ax.set_title('Albedo Time Series Comparison', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plots saved to {save_path}")
        plt.show()
    
    def create_monthly_analysis(self, save_path: str = 'monthly_analysis_improved.png'):
        """Create monthly comparison analysis."""
        if self.merged_data is None:
            raise ValueError("Data not loaded.")
        
        logger.info("Creating monthly analysis...")
        
        methods = ['MCD43A3', 'MOD09GA', 'MOD10A1']
        colors = ['blue', 'green', 'red']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        # Monthly correlation analysis
        months = sorted(self.merged_data['month'].unique())
        
        for i, (method, color) in enumerate(zip(methods, colors)):
            if method not in self.merged_data.columns:
                continue
                
            monthly_stats = []
            month_labels = []
            
            for month in months:
                month_data = self.merged_data[self.merged_data['month'] == month]
                mask = month_data[[method, 'AWS']].notna().all(axis=1)
                
                if mask.sum() >= 5:  # Minimum observations for meaningful correlation
                    x = month_data.loc[mask, method]
                    y = month_data.loc[mask, 'AWS']
                    
                    try:
                        r, _ = stats.pearsonr(x, y)
                        monthly_stats.append(r)
                        month_labels.append(month)
                    except:
                        continue
            
            if monthly_stats:
                axes[i].bar(month_labels, monthly_stats, color=color, alpha=0.7, 
                           edgecolor='black', linewidth=1)
                axes[i].set_title(f'{method} - Monthly Correlations with AWS', fontsize=11)
                axes[i].set_xlabel('Month', fontsize=10)
                axes[i].set_ylabel('Correlation (r)', fontsize=10)
                axes[i].set_ylim(0, 1)
                axes[i].grid(True, alpha=0.3, axis='y')
                
                # Add value labels on bars
                for j, (month, r_val) in enumerate(zip(month_labels, monthly_stats)):
                    axes[i].text(month, r_val + 0.02, f'{r_val:.2f}', 
                               ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # Seasonal box plot
        ax = axes[3]
        seasonal_data = []
        season_labels = []
        
        for season in ['Winter', 'Spring', 'Summer', 'Fall']:
            season_subset = self.merged_data[self.merged_data['season'] == season]
            if len(season_subset) > 0:
                # Combine all methods for seasonal comparison
                all_modis = []
                for method in methods:
                    if method in season_subset.columns:
                        valid_vals = season_subset[method].dropna()
                        all_modis.extend(valid_vals.tolist())
                
                if all_modis:
                    seasonal_data.append(all_modis)
                    season_labels.append(season)
        
        if seasonal_data:
            bp = ax.boxplot(seasonal_data, labels=season_labels, patch_artist=True)
            colors_seasonal = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
            for patch, color in zip(bp['boxes'], colors_seasonal):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_title('Seasonal Distribution of MODIS Albedo Values', fontsize=11)
            ax.set_ylabel('Albedo', fontsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Monthly analysis saved to {save_path}")
        plt.show()
    
    def generate_summary_report(self, output_file: str = 'albedo_analysis_summary.txt'):
        """Generate a comprehensive summary report."""
        if not self.stats_results:
            raise ValueError("Statistics not calculated.")
        
        logger.info(f"Generating summary report: {output_file}")
        
        with open(output_file, 'w') as f:
            f.write("ENHANCED ALBEDO COMPARISON ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            # Dataset information
            f.write("DATASET INFORMATION:\n")
            f.write("-" * 20 + "\n")
            if self.merged_data is not None:
                f.write(f"Analysis period: {self.merged_data.index.min().strftime('%Y-%m-%d')} to {self.merged_data.index.max().strftime('%Y-%m-%d')}\n")
                f.write(f"Total merged observations: {self.merged_data.shape[0]}\n")
                f.write(f"Total features: {self.merged_data.shape[1]}\n\n")
            
            # Method coverage
            f.write("METHOD COVERAGE:\n")
            f.write("-" * 16 + "\n")
            for method, stats in self.stats_results.items():
                f.write(f"{method:<10}: {stats['n']} observations\n")
            f.write("\n")
            
            # Statistical summary
            f.write("STATISTICAL SUMMARY:\n")
            f.write("-" * 19 + "\n")
            f.write(f"{'Method':<10} {'N':<6} {'r':<7} {'p-value':<9} {'RMSE':<7} {'MAE':<7} {'Bias':<7} {'Slope':<7}\n")
            f.write("-" * 70 + "\n")
            
            for method, stats in self.stats_results.items():
                f.write(f"{method:<10} {stats['n']:<6} {stats['r']:<7.3f} {stats['p']:<9.3e} "
                       f"{stats['rmse']:<7.3f} {stats['mae']:<7.3f} {stats['bias']:<7.3f} {stats['slope']:<7.3f}\n")
            
            f.write("\n" + "=" * 70 + "\n")
            f.write("Analysis completed using improved data handler\n")
            f.write(f"Report generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        logger.info(f"Summary report saved to {output_file}")


def main():
    """Main execution function."""
    logger.info("Starting enhanced albedo comparison analysis...")
    
    try:
        # Initialize analyzer
        analyzer = AlbedoAnalyzer()
        
        # Load and merge data
        merged_data = analyzer.load_data(merge_strategy='inner')
        
        # Calculate statistics
        stats_results = analyzer.calculate_statistics()
        
        # Create visualizations
        analyzer.create_overall_comparison_plots()
        analyzer.create_monthly_analysis()
        
        # Generate summary report
        analyzer.generate_summary_report()
        
        logger.info("Enhanced albedo comparison analysis completed successfully!")
        
        # Print summary to console
        print("\nANALYSIS SUMMARY:")
        print("=" * 40)
        print(f"Dataset shape: {merged_data.shape}")
        print(f"Date range: {merged_data.index.min()} to {merged_data.index.max()}")
        print("\nMethod Performance (vs AWS):")
        for method, stats in stats_results.items():
            print(f"  {method}: r={stats['r']:.3f}, RMSE={stats['rmse']:.3f}, n={stats['n']}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()