# ==============================================================================
# IMPORTS
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
try:
    import geopandas as gpd
    import contextily as ctx
    from shapely.geometry import Point
    SPATIAL_LIBS = True
except ImportError:
    SPATIAL_LIBS = False
    print("Warning: Spatial libraries not available. Install geopandas and contextily for spatial analysis.")

# ==============================================================================
# DATA LOADING AND PREPROCESSING
# ==============================================================================
modis = pd.read_csv('data/csv/athabasca_2014-01-01_to_2021-01-01.csv')
aws = pd.read_csv('data/csv/iceAWS_Atha_albedo_daily_20152020_filled_clean.csv')

# ------------------------------------------------------------------------------
# Ensure expected column names exist (support older/newer CSV schemas)
# ------------------------------------------------------------------------------
# Some ATHA MODIS CSVs provide an "albedo" column instead of the newer
# "albedo_value" naming convention used throughout the analysis scripts.  To
# remain backward- and forward-compatible, create an alias if necessary.

if 'albedo_value' not in modis.columns and 'albedo' in modis.columns:
    modis = modis.rename(columns={'albedo': 'albedo_value'})

# Standardize method names to uppercase
modis['method'] = modis['method'].str.upper()

modis['date'] = pd.to_datetime(modis['date'])
aws['date'] = pd.to_datetime(aws['Time'])

modis_pivot = modis.pivot_table(index='date', columns='method', values='albedo_value', aggfunc='mean')
merged = pd.merge(modis_pivot, aws[['date', 'Albedo']], on='date', how='inner')
merged.rename(columns={'Albedo': 'AWS'}, inplace=True)
merged.set_index('date', inplace=True)

# ==============================================================================
# OVERALL STATISTICS CALCULATION
# ==============================================================================
stats_results = {}
for modis_col in ['MCD43A3', 'MOD09GA', 'MOD10A1']:
    mask = merged[[modis_col, 'AWS']].notna().all(axis=1)
    if mask.sum() > 0:
        x = merged.loc[mask, modis_col]
        y = merged.loc[mask, 'AWS']
        
        r, p = stats.pearsonr(x, y)
        rmse = np.sqrt(np.mean((x - y)**2))
        mae = np.mean(np.abs(x - y))
        bias = np.mean(x - y)
        
        stats_results[modis_col] = {
            'n': mask.sum(), 'r': r, 'p': p,
            'rmse': rmse, 'mae': mae, 'bias': bias
        }


# ==============================================================================
# OVERALL VISUALIZATIONS
# ==============================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for i, (modis_col, color) in enumerate(zip(['MCD43A3', 'MOD09GA', 'MOD10A1'], ['blue', 'green', 'red'])):
    mask = merged[[modis_col, 'AWS']].notna().all(axis=1)
    if mask.sum() > 0:
        x, y = merged.loc[mask, modis_col], merged.loc[mask, 'AWS']
        
        axes[i].scatter(y, x, alpha=0.5, s=10, color=color)
        axes[i].plot([0, 1], [0, 1], 'k--', lw=1)
        axes[i].set_xlabel('AWS Albedo')
        axes[i].set_ylabel(f'{modis_col} Albedo')
        axes[i].set_title(f'{modis_col} vs AWS (n={mask.sum()}, r={stats_results[modis_col]["r"]:.3f})')
        axes[i].set_xlim(0, 1)
        axes[i].set_ylim(0, 1)
        axes[i].grid(True, alpha=0.3)

ax = axes[3]
merged_plot = merged.dropna(subset=['AWS'])
ax.plot(merged_plot.index, merged_plot['AWS'], 'k-', lw=1, label='AWS', alpha=0.8)
for modis_col, color in zip(['MCD43A3', 'MOD09GA', 'MOD10A1'], ['blue', 'green', 'red']):
    ax.scatter(merged_plot.index, merged_plot[modis_col], s=5, alpha=0.5, label=modis_col, color=color)
ax.set_xlabel('Date')
ax.set_ylabel('Albedo')
ax.set_title('Albedo Time Series Comparison')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('albedo_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# ==============================================================================
# SUMMARY TABLE
# ==============================================================================
for method, s in stats_results.items():
    print(f"{method:<10} {s['n']:<6} {s['r']:<6.3f} {s['rmse']:<6.3f} {s['mae']:<6.3f} {s['bias']:<6.3f}")

# ==============================================================================
# MONTHLY ANALYSIS
# ==============================================================================
merged['month'] = merged.index.month
monthly_stats = {}

for month in sorted(merged['month'].unique()):
    month_data = merged[merged['month'] == month]
    monthly_stats[month] = {}
    
    for modis_col in ['MCD43A3', 'MOD09GA', 'MOD10A1']:
        mask = month_data[[modis_col, 'AWS']].notna().all(axis=1)
        if mask.sum() > 5:
            x, y = month_data.loc[mask, modis_col], month_data.loc[mask, 'AWS']
            r, p = stats.pearsonr(x, y)
            rmse = np.sqrt(np.mean((x - y)**2))
            
            monthly_stats[month][modis_col] = {'n': mask.sum(), 'r': r, 'rmse': rmse, 'data': (x, y)}

# ==============================================================================
# MONTHLY SCATTER PLOTS
# ==============================================================================
months_with_data = [m for m in monthly_stats.keys() if any(len(v) > 0 for v in monthly_stats[m].values())]

if len(months_with_data) > 0:
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.ravel()
    
    for plot_idx, month in enumerate(months_with_data[:12]):
        ax = axes[plot_idx]
        
        for modis_col, color in zip(['MCD43A3', 'MOD09GA', 'MOD10A1'], ['blue', 'green', 'red']):
            if modis_col in monthly_stats[month] and monthly_stats[month][modis_col]['n'] > 5:
                x, y = monthly_stats[month][modis_col]['data']
                r = monthly_stats[month][modis_col]['r']
                ax.scatter(y, x, alpha=0.6, s=15, color=color, label=f'{modis_col} (r={r:.2f})')
        
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set_xlabel('AWS Albedo')
        ax.set_ylabel('MODIS Albedo')
        ax.set_title(f'Month {month}')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    for i in range(len(months_with_data), 12):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('monthly_albedo_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

# ==============================================================================
# PIXEL-LEVEL COMPARISON (MOD09GA vs MOD10A1)
# ==============================================================================

# Filter for MOD09GA and MOD10A1 only
mod_data = modis[modis['method'].isin(['MOD09GA', 'MOD10A1'])].copy()

# Find matching observations (same date and pixel_id)
mod09_data = mod_data[mod_data['method'] == 'MOD09GA'][['date', 'pixel_id', 'albedo_value']]
mod10_data = mod_data[mod_data['method'] == 'MOD10A1'][['date', 'pixel_id', 'albedo_value']]

# Merge on exact date and pixel_id matches
pixel_matched = pd.merge(mod09_data, mod10_data, on=['date', 'pixel_id'], suffixes=('_MOD09GA', '_MOD10A1'))

if len(pixel_matched) > 0:
    # Add AWS data to matched pixels
    pixel_aws = pd.merge(pixel_matched, aws[['date', 'Albedo']], on='date', how='inner')
    pixel_aws.rename(columns={'Albedo': 'AWS'}, inplace=True)
    
    # Statistics for direct MOD09GA vs MOD10A1 comparison
    r_direct, p_direct = stats.pearsonr(pixel_matched['albedo_value_MOD09GA'], 
                                       pixel_matched['albedo_value_MOD10A1'])
    rmse_direct = np.sqrt(np.mean((pixel_matched['albedo_value_MOD09GA'] - 
                                  pixel_matched['albedo_value_MOD10A1'])**2))
    
    # Statistics vs AWS for matched pixels
    if len(pixel_aws) > 0:
        # Clean data for correlation analysis
        clean_09 = pixel_aws[['albedo_value_MOD09GA', 'AWS']].dropna()
        clean_10 = pixel_aws[['albedo_value_MOD10A1', 'AWS']].dropna()
        
        # Calculate correlations on clean data
        if len(clean_09) > 1 and clean_09['albedo_value_MOD09GA'].std() > 0 and clean_09['AWS'].std() > 0:
            r_09_aws, p_09_aws = stats.pearsonr(clean_09['albedo_value_MOD09GA'], clean_09['AWS'])
        else:
            r_09_aws, p_09_aws = np.nan, np.nan
            
        if len(clean_10) > 1 and clean_10['albedo_value_MOD10A1'].std() > 0 and clean_10['AWS'].std() > 0:
            r_10_aws, p_10_aws = stats.pearsonr(clean_10['albedo_value_MOD10A1'], clean_10['AWS'])
        else:
            r_10_aws, p_10_aws = np.nan, np.nan
        
        # Calculate RMSE on clean data
        if len(clean_09) > 0:
            rmse_09_aws = np.sqrt(np.mean((clean_09['albedo_value_MOD09GA'] - clean_09['AWS'])**2))
        else:
            rmse_09_aws = np.nan
            
        if len(clean_10) > 0:
            rmse_10_aws = np.sqrt(np.mean((clean_10['albedo_value_MOD10A1'] - clean_10['AWS'])**2))
        else:
            rmse_10_aws = np.nan
        
        # Pixel-level analysis
        pixel_stats = pixel_aws.groupby('pixel_id').agg({
            'albedo_value_MOD09GA': ['count', 'mean', 'std'],
            'albedo_value_MOD10A1': ['count', 'mean', 'std'],
            'AWS': ['count', 'mean', 'std']
        }).round(3)
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # MOD09GA vs MOD10A1 direct comparison
    axes[0].scatter(pixel_matched['albedo_value_MOD09GA'], pixel_matched['albedo_value_MOD10A1'], 
                   alpha=0.6, s=20, color='purple')
    axes[0].plot([0, 1], [0, 1], 'k--', lw=1)
    axes[0].set_xlabel('MOD09GA Albedo')
    axes[0].set_ylabel('MOD10A1 Albedo')
    axes[0].set_title(f'MOD09GA vs MOD10A1\n(Same pixel+date, n={len(pixel_matched)}, r={r_direct:.3f})')
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3)
    
    if len(pixel_aws) > 0:
        # MOD09GA vs AWS (matched pixels)
        axes[1].scatter(pixel_aws['AWS'], pixel_aws['albedo_value_MOD09GA'], 
                       alpha=0.6, s=20, color='green')
        axes[1].plot([0, 1], [0, 1], 'k--', lw=1)
        axes[1].set_xlabel('AWS Albedo')
        axes[1].set_ylabel('MOD09GA Albedo')
        if not np.isnan(r_09_aws):
            axes[1].set_title(f'MOD09GA vs AWS\n(Matched pixels, n={len(pixel_aws)}, r={r_09_aws:.3f})')
        else:
            axes[1].set_title(f'MOD09GA vs AWS\n(Matched pixels, n={len(pixel_aws)}, r=low_var)')
        axes[1].set_xlim(0, 1)
        axes[1].set_ylim(0, 1)
        axes[1].grid(True, alpha=0.3)
        
        # MOD10A1 vs AWS (matched pixels)
        axes[2].scatter(pixel_aws['AWS'], pixel_aws['albedo_value_MOD10A1'], 
                       alpha=0.6, s=20, color='red')
        axes[2].plot([0, 1], [0, 1], 'k--', lw=1)
        axes[2].set_xlabel('AWS Albedo')
        axes[2].set_ylabel('MOD10A1 Albedo')
        if not np.isnan(r_10_aws):
            axes[2].set_title(f'MOD10A1 vs AWS\n(Matched pixels, n={len(pixel_aws)}, r={r_10_aws:.3f})')
        else:
            axes[2].set_title(f'MOD10A1 vs AWS\n(Matched pixels, n={len(pixel_aws)}, r=low_var)')
        axes[2].set_xlim(0, 1)
        axes[2].set_ylim(0, 1)
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pixel_level_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
else:
    print("No matching observations found for same date and pixel_id")

# ==============================================================================
# INDIVIDUAL PIXEL ID ANALYSIS
# ==============================================================================

if len(pixel_matched) > 0:
    unique_pixels = pixel_matched['pixel_id'].unique()
    
    pixel_results = {}
    
    for pixel_id in unique_pixels:
        # Filter data for this pixel
        pixel_data = pixel_matched[pixel_matched['pixel_id'] == pixel_id]
        pixel_aws_data = pixel_aws[pixel_aws['pixel_id'] == pixel_id] if len(pixel_aws) > 0 else pd.DataFrame()
        
        # Direct MOD09GA vs MOD10A1 for this pixel
        if len(pixel_data) > 1:
            r_direct, p_direct = stats.pearsonr(pixel_data['albedo_value_MOD09GA'], 
                                              pixel_data['albedo_value_MOD10A1'])
            rmse_direct = np.sqrt(np.mean((pixel_data['albedo_value_MOD09GA'] - 
                                         pixel_data['albedo_value_MOD10A1'])**2))
        else:
            r_direct, rmse_direct = np.nan, np.nan
        
        # AWS comparisons for this pixel
        if len(pixel_aws_data) > 1:
            clean_09 = pixel_aws_data[['albedo_value_MOD09GA', 'AWS']].dropna()
            clean_10 = pixel_aws_data[['albedo_value_MOD10A1', 'AWS']].dropna()
            
            # MOD09GA vs AWS
            if len(clean_09) > 1 and clean_09['albedo_value_MOD09GA'].std() > 0 and clean_09['AWS'].std() > 0:
                r_09, p_09 = stats.pearsonr(clean_09['albedo_value_MOD09GA'], clean_09['AWS'])
                rmse_09 = np.sqrt(np.mean((clean_09['albedo_value_MOD09GA'] - clean_09['AWS'])**2))
            else:
                r_09, rmse_09 = np.nan, np.nan
            
            # MOD10A1 vs AWS
            if len(clean_10) > 1 and clean_10['albedo_value_MOD10A1'].std() > 0 and clean_10['AWS'].std() > 0:
                r_10, p_10 = stats.pearsonr(clean_10['albedo_value_MOD10A1'], clean_10['AWS'])
                rmse_10 = np.sqrt(np.mean((clean_10['albedo_value_MOD10A1'] - clean_10['AWS'])**2))
            else:
                r_10, rmse_10 = np.nan, np.nan
                
        else:
            r_09, rmse_09, r_10, rmse_10 = np.nan, np.nan, np.nan, np.nan
        
        # Store results
        pixel_results[pixel_id] = {
            'n_matched': len(pixel_data),
            'n_aws': len(pixel_aws_data),
            'r_direct': r_direct,
            'rmse_direct': rmse_direct,
            'r_09_aws': r_09,
            'rmse_09_aws': rmse_09,
            'r_10_aws': r_10,
            'rmse_10_aws': rmse_10,
            'data': pixel_data,
            'aws_data': pixel_aws_data
        }
    
    # Summary table for individual pixels
    print(f"\n\nSUMMARY TABLE - INDIVIDUAL PIXELS:")
    print(f"{'Pixel_ID':<12} {'n_match':<8} {'n_aws':<6} {'MOD09_vs_MOD10':<15} {'MOD09_vs_AWS':<15} {'MOD10_vs_AWS':<15}")
    print("-" * 80)
    
    for pixel_id, results in pixel_results.items():
        r_direct_str = f"{results['r_direct']:.3f}" if not np.isnan(results['r_direct']) else "n/a"
        r_09_str = f"{results['r_09_aws']:.3f}" if not np.isnan(results['r_09_aws']) else "n/a"
        r_10_str = f"{results['r_10_aws']:.3f}" if not np.isnan(results['r_10_aws']) else "n/a"
        
        print(f"{pixel_id:<12} {results['n_matched']:<8} {results['n_aws']:<6} {r_direct_str:<15} {r_09_str:<15} {r_10_str:<15}")
    
    # Individual pixel visualizations
    n_pixels = len(unique_pixels)
    if n_pixels > 0:
        fig, axes = plt.subplots(n_pixels, 3, figsize=(15, 5*n_pixels))
        if n_pixels == 1:
            axes = axes.reshape(1, -1)
        
        for i, pixel_id in enumerate(unique_pixels):
            results = pixel_results[pixel_id]
            pixel_data = results['data']
            pixel_aws_data = results['aws_data']
            
            # MOD09GA vs MOD10A1 for this pixel
            if len(pixel_data) > 0:
                axes[i,0].scatter(pixel_data['albedo_value_MOD09GA'], pixel_data['albedo_value_MOD10A1'], 
                                alpha=0.7, s=25, color='purple')
                axes[i,0].plot([0, 1], [0, 1], 'k--', lw=1)
                axes[i,0].set_xlabel('MOD09GA Albedo')
                axes[i,0].set_ylabel('MOD10A1 Albedo')
                r_str = f"r={results['r_direct']:.3f}" if not np.isnan(results['r_direct']) else "r=n/a"
                axes[i,0].set_title(f'Pixel {pixel_id}\nMOD09GA vs MOD10A1 (n={len(pixel_data)}, {r_str})')
                axes[i,0].set_xlim(0, 1)
                axes[i,0].set_ylim(0, 1)
                axes[i,0].grid(True, alpha=0.3)
            
            # MOD09GA vs AWS for this pixel
            if len(pixel_aws_data) > 0:
                clean_09 = pixel_aws_data[['albedo_value_MOD09GA', 'AWS']].dropna()
                if len(clean_09) > 0:
                    axes[i,1].scatter(clean_09['AWS'], clean_09['albedo_value_MOD09GA'], 
                                    alpha=0.7, s=25, color='green')
                    axes[i,1].plot([0, 1], [0, 1], 'k--', lw=1)
                    axes[i,1].set_xlabel('AWS Albedo')
                    axes[i,1].set_ylabel('MOD09GA Albedo')
                    r_str = f"r={results['r_09_aws']:.3f}" if not np.isnan(results['r_09_aws']) else "r=n/a"
                    axes[i,1].set_title(f'Pixel {pixel_id}\nMOD09GA vs AWS (n={len(clean_09)}, {r_str})')
                    axes[i,1].set_xlim(0, 1)
                    axes[i,1].set_ylim(0, 1)
                    axes[i,1].grid(True, alpha=0.3)
            
            # MOD10A1 vs AWS for this pixel
            if len(pixel_aws_data) > 0:
                clean_10 = pixel_aws_data[['albedo_value_MOD10A1', 'AWS']].dropna()
                if len(clean_10) > 0:
                    axes[i,2].scatter(clean_10['AWS'], clean_10['albedo_value_MOD10A1'], 
                                    alpha=0.7, s=25, color='red')
                    axes[i,2].plot([0, 1], [0, 1], 'k--', lw=1)
                    axes[i,2].set_xlabel('AWS Albedo')
                    axes[i,2].set_ylabel('MOD10A1 Albedo')
                    r_str = f"r={results['r_10_aws']:.3f}" if not np.isnan(results['r_10_aws']) else "r=n/a"
                    axes[i,2].set_title(f'Pixel {pixel_id}\nMOD10A1 vs AWS (n={len(clean_10)}, {r_str})')
                    axes[i,2].set_xlim(0, 1)
                    axes[i,2].set_ylim(0, 1)
                    axes[i,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('individual_pixel_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()


# ==============================================================================
# SPATIAL ANALYSIS AND MAPPING
# ==============================================================================

if SPATIAL_LIBS:
    try:
        # Load shapefiles
        aws_station = gpd.read_file('data/mask/Point_Custom.shp')
        glacier_mask = gpd.read_file('data/mask/masque_athabasa_zone_ablation.shp')
        
        # Get unique pixel locations from MODIS data for each method
        pixel_locations = {}
        method_colors = {'MOD09GA': 'green', 'MOD10A1': 'red', 'MCD43A3': 'blue'}
        
        for method in ['MOD09GA', 'MOD10A1', 'MCD43A3']:
            method_data = modis[modis['method'] == method]
            unique_pixels = method_data[['pixel_id', 'latitude', 'longitude']].drop_duplicates()
            pixel_locations[method] = unique_pixels
        
        # Create separate spatial maps for each method
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Get common bounds for all maps
        all_bounds = []
        if len(aws_station) > 0:
            all_bounds.append(aws_station.total_bounds)
        if len(glacier_mask) > 0:
            all_bounds.append(glacier_mask.total_bounds)
        for method_pixels in pixel_locations.values():
            if len(method_pixels) > 0:
                lon_bounds = [method_pixels['longitude'].min(), method_pixels['longitude'].max()]
                lat_bounds = [method_pixels['latitude'].min(), method_pixels['latitude'].max()]
                all_bounds.append([lon_bounds[0], lat_bounds[0], lon_bounds[1], lat_bounds[1]])
        
        if all_bounds:
            all_bounds = np.array(all_bounds)
            minx, miny = all_bounds[:, 0].min(), all_bounds[:, 1].min()
            maxx, maxy = all_bounds[:, 2].max(), all_bounds[:, 3].max()
            buffer = 0.001
        else:
            minx, miny, maxx, maxy, buffer = 0, 0, 1, 1, 0.1
        
        for idx, (method, color) in enumerate(method_colors.items()):
            ax = axes[idx]
            
            # Plot glacier ablation zone
            glacier_mask.plot(ax=ax, color='lightblue', alpha=0.5, edgecolor='blue', linewidth=1)
            
            # Plot AWS station
            aws_station.plot(ax=ax, color='black', marker='*', markersize=150, zorder=5)
            
            # Plot pixels for this method only
            if method in pixel_locations and len(pixel_locations[method]) > 0:
                pixels_gdf = gpd.GeoDataFrame(
                    pixel_locations[method],
                    geometry=[Point(xy) for xy in zip(pixel_locations[method]['longitude'], pixel_locations[method]['latitude'])],
                    crs='EPSG:4326'
                )
                pixels_gdf.plot(ax=ax, color=color, marker='s', markersize=120, alpha=0.8, zorder=4)
                
                # Add pixel ID labels
                for _, row in pixels_gdf.iterrows():
                    ax.annotate(str(int(row['pixel_id'])), (row.geometry.x, row.geometry.y), 
                              xytext=(8, 8), textcoords='offset points', fontsize=10, fontweight='bold',
                              bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7, edgecolor='black'))
            
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_title(f'{method} Pixels\n({len(pixel_locations.get(method, []))} locations)')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(minx - buffer, maxx + buffer)
            ax.set_ylim(miny - buffer, maxy + buffer)
        
        # Add common legend
        legend_elements = [
            plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='black', markersize=12, label='AWS Station'),
            plt.Rectangle((0, 0), 1, 1, facecolor='lightblue', alpha=0.5, edgecolor='blue', label='Glacier Ablation Zone'),
        ]
        fig.legend(handles=legend_elements, bbox_to_anchor=(0.5, 0.02), loc='lower center', ncol=2)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        plt.savefig('spatial_pixel_maps_separate.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Store AWS point for export
        if len(aws_station) > 0:
            aws_point = aws_station.geometry.iloc[0]
        
    except Exception as e:
        pass

else:
    # Basic coordinate analysis without spatial libraries (silent)
    pass

# ==============================================================================
# PIXEL-SPECIFIC CHARACTERISTICS
# ==============================================================================

# Get all unique pixels across all methods
all_pixel_data = []
for method in ['MOD09GA', 'MOD10A1', 'MCD43A3']:
    method_data = modis[modis['method'] == method]
    # Get first occurrence of each pixel to extract characteristics
    unique_pixels = method_data.drop_duplicates(subset=['pixel_id']).copy()
    unique_pixels['method'] = method
    all_pixel_data.append(unique_pixels)

combined_pixels = pd.concat(all_pixel_data, ignore_index=True)

# Get characteristics for all unique pixel locations
pixel_chars = combined_pixels.groupby('pixel_id').agg({
    'latitude': 'first',
    'longitude': 'first', 
    'elevation': 'first',
    'slope': 'first',
    'aspect': 'first',
    'pixel_row': 'first',
    'pixel_col': 'first',
    'tile_h': 'first',
    'tile_v': 'first',
    'glacier_fraction': 'first'
}).round(4)

# Count observations per pixel per method
pixel_obs_counts = {}
for method in ['MOD09GA', 'MOD10A1', 'MCD43A3']:
    method_data = modis[modis['method'] == method]
    counts = method_data.groupby('pixel_id').size()
    pixel_obs_counts[method] = counts

# Extract terrain data for plots
elevations = pixel_chars['elevation'].values
slopes = pixel_chars['slope'].values
aspects = pixel_chars['aspect'].values
glacier_fractions = pixel_chars['glacier_fraction'].values

# Create enhanced pixel characteristics visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Elevation vs Glacier Fraction scatter
scatter = axes[0,0].scatter(elevations, glacier_fractions, c=slopes, s=80, alpha=0.8, 
                           cmap='RdYlBu_r', edgecolors='black', linewidth=0.5)
axes[0,0].set_xlabel('Elevation (m)')
axes[0,0].set_ylabel('Glacier Fraction')
axes[0,0].set_title('Elevation vs Glacier Fraction\n(Color = Slope)')
axes[0,0].grid(True, alpha=0.3)
cbar = plt.colorbar(scatter, ax=axes[0,0], shrink=0.8)
cbar.set_label('Slope (°)', rotation=270, labelpad=15)

# Glacier fraction by pixel (bar plot)
pixel_ids_sorted = sorted(pixel_chars.index)
glacier_fracs_sorted = [pixel_chars.loc[pid, 'glacier_fraction'] for pid in pixel_ids_sorted]
colors = plt.cm.Blues(np.linspace(0.3, 1, len(pixel_ids_sorted)))

bars = axes[0,1].bar(range(len(pixel_ids_sorted)), glacier_fracs_sorted, 
                     color=colors, alpha=0.8, edgecolor='black', linewidth=1)
axes[0,1].set_xlabel('Pixel ID')
axes[0,1].set_ylabel('Glacier Fraction')
axes[0,1].set_title('Glacier Fraction by Pixel')
axes[0,1].set_xticks(range(len(pixel_ids_sorted)))
axes[0,1].set_xticklabels([str(int(pid)) for pid in pixel_ids_sorted])
axes[0,1].set_ylim(0, 1)
axes[0,1].grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, value in zip(bars, glacier_fracs_sorted):
    height = bar.get_height()
    axes[0,1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.2f}', ha='center', va='bottom', fontweight='bold')

# Terrain characteristics (elevation vs slope)
scatter2 = axes[1,0].scatter(elevations, slopes, c=glacier_fractions, s=80, alpha=0.8,
                            cmap='Blues', edgecolors='black', linewidth=0.5)
axes[1,0].set_xlabel('Elevation (m)')
axes[1,0].set_ylabel('Slope (degrees)')
axes[1,0].set_title('Elevation vs Slope\n(Color = Glacier Fraction)')
axes[1,0].grid(True, alpha=0.3)
cbar2 = plt.colorbar(scatter2, ax=axes[1,0], shrink=0.8)
cbar2.set_label('Glacier Fraction', rotation=270, labelpad=15)

# Enhanced observation count with method breakdown
pixel_ids = [int(pid) for pid in sorted(pixel_chars.index)]
obs_mod09 = [pixel_obs_counts['MOD09GA'].get(pid, 0) for pid in pixel_chars.index]
obs_mod10 = [pixel_obs_counts['MOD10A1'].get(pid, 0) for pid in pixel_chars.index]
obs_mcd43 = [pixel_obs_counts['MCD43A3'].get(pid, 0) for pid in pixel_chars.index]

x_pos = range(len(pixel_ids))
width = 0.25
axes[1,1].bar([x - width for x in x_pos], obs_mod09, width, label='MOD09GA', 
              color='green', alpha=0.8, edgecolor='black')
axes[1,1].bar(x_pos, obs_mod10, width, label='MOD10A1', 
              color='red', alpha=0.8, edgecolor='black')
axes[1,1].bar([x + width for x in x_pos], obs_mcd43, width, label='MCD43A3', 
              color='blue', alpha=0.8, edgecolor='black')
axes[1,1].set_xlabel('Pixel ID')
axes[1,1].set_ylabel('Observations Count')
axes[1,1].set_title('Observations per Pixel by Method')
axes[1,1].set_xticks(x_pos)
axes[1,1].set_xticklabels([str(pid) for pid in pixel_ids], rotation=45)
axes[1,1].legend(fontsize=9)
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pixel_characteristics_enhanced.png', dpi=300, bbox_inches='tight')
plt.show()

# ==============================================================================
# OUTLIER DETECTION
# ==============================================================================
# Detect Z-score outliers (|z| > 2.5)
outliers = {}
outlier_counts = {}
outlier_values = {}
for col in ['MCD43A3', 'MOD09GA', 'MOD10A1', 'AWS']:
    if col in merged.columns:
        clean_data = merged[col].dropna()
        if len(clean_data) > 0:
            z = np.abs(stats.zscore(clean_data))
            outlier_mask = z > 3.0  # Less strict: only extreme outliers
            outliers[col] = pd.Series(outlier_mask, index=clean_data.index)
            outlier_counts[col] = outlier_mask.sum()
            # Store actual outlier values
            outlier_values[col] = clean_data[outlier_mask].sort_values()
        else:
            outliers[col] = pd.Series([], dtype=bool)
            outlier_counts[col] = 0
            outlier_values[col] = pd.Series([])

# Scatter plots with outliers highlighted
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
methods = ['MCD43A3', 'MOD09GA', 'MOD10A1']
for i, method in enumerate(methods):
    if method in merged.columns:
        mask = merged[[method, 'AWS']].notna().all(axis=1)
        if mask.sum() > 0:
            aws_outliers = outliers['AWS'].reindex(merged.index).fillna(False).astype(bool)
            method_outliers = outliers[method].reindex(merged.index).fillna(False).astype(bool)
            is_outlier = aws_outliers | method_outliers
            
            # Normal points
            normal_mask = mask & ~is_outlier
            if normal_mask.sum() > 0:
                axes[i].scatter(merged.loc[normal_mask, 'AWS'], merged.loc[normal_mask, method], 
                               alpha=0.6, s=15, color='blue', label='Normal')
            
            # Outliers in red
            outlier_mask = mask & is_outlier
            if outlier_mask.sum() > 0:
                axes[i].scatter(merged.loc[outlier_mask, 'AWS'], merged.loc[outlier_mask, method], 
                               alpha=0.9, s=30, color='red', label=f'Outliers ({outlier_mask.sum()})')
            
            axes[i].plot([0,1], [0,1], 'k--', lw=1)
            axes[i].set_xlabel('AWS Albedo')
            axes[i].set_ylabel(f'{method} Albedo')
            axes[i].set_title(f'{method} vs AWS\n({outlier_counts[method] + outlier_counts["AWS"]} outliers)')
            axes[i].set_xlim(0, 1)
            axes[i].set_ylim(0, 1)
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()

plt.tight_layout()
plt.savefig('outliers_scatter.png', dpi=300, bbox_inches='tight')
plt.show()

# Box plot for outlier overview
fig, ax = plt.subplots(figsize=(10, 6))
merged[['MCD43A3', 'MOD09GA', 'MOD10A1', 'AWS']].boxplot(ax=ax, patch_artist=True)
ax.set_ylabel('Albedo Value')
ax.set_title('Albedo Distribution with Outliers')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outliers_boxplot.png', dpi=300, bbox_inches='tight')
plt.show()

# Calculate statistics with and without outliers (using residual-based outlier detection)
stats_with_outliers = {}
stats_without_outliers = {}

for method in ['MCD43A3', 'MOD09GA', 'MOD10A1']:
    if method in merged.columns:
        mask = merged[[method, 'AWS']].notna().all(axis=1)
        if mask.sum() > 0:
            x_all = merged.loc[mask, 'AWS']
            y_all = merged.loc[mask, method]
            
            # Stats with outliers
            r_all, _ = stats.pearsonr(x_all, y_all)
            rmse_all = np.sqrt(np.mean((y_all - x_all)**2))
            mae_all = np.mean(np.abs(y_all - x_all))
            bias_all = np.mean(y_all - x_all)
            stats_with_outliers[method] = {'n': len(x_all), 'r': r_all, 'rmse': rmse_all, 'mae': mae_all, 'bias': bias_all}
            
            # Remove residual outliers (better approach for correlation analysis)
            slope, intercept = np.polyfit(x_all, y_all, 1)
            predicted = slope * x_all + intercept
            residuals = y_all - predicted
            residual_threshold = 3.0 * residuals.std()  # Less strict: only extreme outliers
            residual_outliers = np.abs(residuals) > residual_threshold
            clean_mask = mask & ~pd.Series(residual_outliers, index=mask[mask].index).reindex(merged.index).fillna(False)
            
            if clean_mask.sum() > 0:
                x_clean = merged.loc[clean_mask, 'AWS']
                y_clean = merged.loc[clean_mask, method]
                
                # Stats without outliers
                r_clean, _ = stats.pearsonr(x_clean, y_clean)
                rmse_clean = np.sqrt(np.mean((y_clean - x_clean)**2))
                mae_clean = np.mean(np.abs(y_clean - x_clean))
                bias_clean = np.mean(y_clean - x_clean)
                stats_without_outliers[method] = {'n': len(x_clean), 'r': r_clean, 'rmse': rmse_clean, 'mae': mae_clean, 'bias': bias_clean}

# ==============================================================================
# EXPORT ALL STATISTICS TO TEXT FILE
# ==============================================================================

with open('albedo_analysis_results.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("ATHABASCA GLACIER ALBEDO ANALYSIS RESULTS\n")
    f.write("="*80 + "\n\n")
    
    # Data overview
    f.write("DATA OVERVIEW:\n")
    f.write("-"*30 + "\n")
    f.write(f"Analysis period: {merged.index.min().strftime('%Y-%m-%d')} to {merged.index.max().strftime('%Y-%m-%d')}\n")
    f.write(f"Total merged observations: {merged.shape[0]}\n\n")
    
    f.write("Data availability by method:\n")
    for col in ['MCD43A3', 'MOD09GA', 'MOD10A1', 'AWS']:
        count = merged[col].notna().sum() if col in merged.columns else 0
        f.write(f"  {col}: {count} observations\n")
    f.write("\n")
    
    # Overall statistics
    f.write("OVERALL COMPARISON STATISTICS:\n")
    f.write("-"*40 + "\n")
    for method, s in stats_results.items():
        f.write(f"\n{method} vs AWS:\n")
        f.write(f"  Sample size (n): {s['n']}\n")
        f.write(f"  Correlation (r): {s['r']:.3f}\n")
        f.write(f"  P-value: {s['p']:.6f}\n")
        f.write(f"  RMSE: {s['rmse']:.3f}\n")
        f.write(f"  MAE: {s['mae']:.3f}\n")
        f.write(f"  Bias: {s['bias']:.3f}\n")
    f.write("\n")
    
    # Monthly statistics
    f.write("MONTHLY STATISTICS:\n")
    f.write("-"*25 + "\n")
    f.write(f"{'Month':<6} {'Method':<10} {'n':<4} {'r':<6} {'RMSE':<6}\n")
    f.write("-" * 40 + "\n")
    
    for month in sorted(merged['month'].unique()):
        month_data = merged[merged['month'] == month]
        for modis_col in ['MCD43A3', 'MOD09GA', 'MOD10A1']:
            mask = month_data[[modis_col, 'AWS']].notna().all(axis=1)
            if mask.sum() > 5:
                x, y = month_data.loc[mask, modis_col], month_data.loc[mask, 'AWS']
                r, p = stats.pearsonr(x, y)
                rmse = np.sqrt(np.mean((x - y)**2))
                f.write(f"{month:<6} {modis_col:<10} {mask.sum():<4} {r:<6.3f} {rmse:<6.3f}\n")
    f.write("\n")
    
    # Pixel-level comparison
    if len(pixel_matched) > 0:
        f.write("PIXEL-LEVEL COMPARISON (SAME DATE + PIXEL_ID):\n")
        f.write("-"*50 + "\n")
        f.write(f"MOD09GA total observations: {len(mod09_data)}\n")
        f.write(f"MOD10A1 total observations: {len(mod10_data)}\n")
        f.write(f"Matched observations (same date + pixel_id): {len(pixel_matched)}\n")
        
        if len(pixel_aws) > 0:
            f.write(f"Matched observations with AWS data: {len(pixel_aws)}\n")
            f.write(f"AWS missing values: {pixel_aws['AWS'].isna().sum()}\n\n")
            
            # Direct comparison
            f.write(f"Direct MOD09GA vs MOD10A1 (matched pixels):\n")
            f.write(f"  Sample size: {len(pixel_matched)}\n")
            f.write(f"  Correlation: {r_direct:.3f} (p={p_direct:.6f})\n")
            f.write(f"  RMSE: {rmse_direct:.3f}\n\n")
            
            # AWS comparisons
            clean_09 = pixel_aws[['albedo_value_MOD09GA', 'AWS']].dropna()
            clean_10 = pixel_aws[['albedo_value_MOD10A1', 'AWS']].dropna()
            
            f.write(f"Pixel-level vs AWS comparisons:\n")
            f.write(f"  MOD09GA vs AWS: n={len(clean_09)}\n")
            if len(clean_09) > 1:
                f.write(f"    Correlation: {r_09_aws:.3f} (p={p_09_aws:.6f})\n")
                f.write(f"    RMSE: {rmse_09_aws:.3f}\n")
            
            f.write(f"  MOD10A1 vs AWS: n={len(clean_10)}\n")
            if len(clean_10) > 1:
                f.write(f"    Correlation: {r_10_aws:.3f} (p={p_10_aws:.6f})\n")
                f.write(f"    RMSE: {rmse_10_aws:.3f}\n")
        f.write("\n")
    
    # Individual pixel analysis
    if 'pixel_results' in locals():
        f.write("INDIVIDUAL PIXEL ANALYSIS:\n")
        f.write("-"*35 + "\n")
        f.write(f"{'Pixel_ID':<12} {'n_match':<8} {'n_aws':<6} {'MOD09_vs_MOD10':<15} {'MOD09_vs_AWS':<15} {'MOD10_vs_AWS':<15}\n")
        f.write("-" * 80 + "\n")
        
        for pixel_id, results in pixel_results.items():
            r_direct_str = f"{results['r_direct']:.3f}" if not np.isnan(results['r_direct']) else "n/a"
            r_09_str = f"{results['r_09_aws']:.3f}" if not np.isnan(results['r_09_aws']) else "n/a"
            r_10_str = f"{results['r_10_aws']:.3f}" if not np.isnan(results['r_10_aws']) else "n/a"
            
            f.write(f"{int(pixel_id):<12} {results['n_matched']:<8} {results['n_aws']:<6} {r_direct_str:<15} {r_09_str:<15} {r_10_str:<15}\n")
        f.write("\n")
    
    # Spatial analysis
    f.write("SPATIAL ANALYSIS:\n")
    f.write("-"*20 + "\n")
    f.write(f"AWS Station location: {aws_point.y:.4f}°N, {aws_point.x:.4f}°W\n\n")
    
    for method in ['MOD09GA', 'MOD10A1', 'MCD43A3']:
        if method in pixel_locations and len(pixel_locations[method]) > 0:
            pixels = pixel_locations[method]
            f.write(f"{method} pixel locations ({len(pixels)}):\n")
            f.write(f"  Latitude range: {pixels['latitude'].min():.4f} to {pixels['latitude'].max():.4f}\n")
            f.write(f"  Longitude range: {pixels['longitude'].min():.4f} to {pixels['longitude'].max():.4f}\n")
            
            f.write(f"  Distances from AWS:\n")
            for _, row in pixels.iterrows():
                pixel_point = Point(row['longitude'], row['latitude'])
                distance_deg = ((pixel_point.x - aws_point.x)**2 + (pixel_point.y - aws_point.y)**2)**0.5
                distance_m = distance_deg * 111000
                f.write(f"    Pixel {int(row['pixel_id'])}: {distance_m:.0f}m\n")
            f.write("\n")
    
    # Pixel overlap analysis
    f.write("PIXEL OVERLAP ANALYSIS:\n")
    f.write("-"*30 + "\n")
    all_pixels = set()
    for method in ['MOD09GA', 'MOD10A1', 'MCD43A3']:
        if method in pixel_locations:
            method_pixels = set(pixel_locations[method]['pixel_id'])
            all_pixels.update(method_pixels)
            f.write(f"{method}: {len(method_pixels)} unique pixels\n")
    
    f.write(f"Total unique pixel locations: {len(all_pixels)}\n\n")
    
    methods = list(pixel_locations.keys())
    for i in range(len(methods)):
        for j in range(i+1, len(methods)):
            method1, method2 = methods[i], methods[j]
            if method1 in pixel_locations and method2 in pixel_locations:
                pixels1 = set(pixel_locations[method1]['pixel_id'])
                pixels2 = set(pixel_locations[method2]['pixel_id'])
                overlap = pixels1.intersection(pixels2)
                overlap_list = [int(x) for x in sorted(overlap)]
                f.write(f"{method1} & {method2}: {len(overlap)} shared pixels {overlap_list}\n")
    f.write("\n")
    
    # Pixel characteristics
    f.write("PIXEL-SPECIFIC CHARACTERISTICS:\n")
    f.write("-"*40 + "\n")
    f.write(f"{'Pixel_ID':<12} {'Lat':<8} {'Lon':<9} {'Elev(m)':<7} {'Slope':<6} {'Aspect':<7} {'Row':<6} {'Col':<6}\n")
    f.write("-" * 70 + "\n")
    
    for pixel_id in sorted(pixel_chars.index):
        char = pixel_chars.loc[pixel_id]
        f.write(f"{int(pixel_id):<12} {char['latitude']:<8.4f} {char['longitude']:<9.4f} {char['elevation']:<7.0f} "
              f"{char['slope']:<6.1f} {char['aspect']:<7.1f} {int(char['pixel_row']):<6} {int(char['pixel_col']):<6}\n")
    
    f.write(f"\nOBSERVATION COUNTS PER PIXEL:\n")
    f.write(f"{'Pixel_ID':<12} {'MOD09GA':<8} {'MOD10A1':<8} {'MCD43A3':<8} {'Total':<8}\n")
    f.write("-" * 50 + "\n")
    
    for pixel_id in sorted(pixel_chars.index):
        counts = []
        total = 0
        for method in ['MOD09GA', 'MOD10A1', 'MCD43A3']:
            count = pixel_obs_counts[method].get(pixel_id, 0)
            counts.append(count)
            total += count
        f.write(f"{int(pixel_id):<12} {counts[0]:<8} {counts[1]:<8} {counts[2]:<8} {total:<8}\n")
    
    # Terrain analysis summary
    elevations = pixel_chars['elevation'].values
    slopes = pixel_chars['slope'].values
    aspects = pixel_chars['aspect'].values
    
    f.write(f"\nTERRAIN ANALYSIS SUMMARY:\n")
    f.write("-" * 30 + "\n")
    f.write(f"Elevation range: {elevations.min():.0f}m to {elevations.max():.0f}m\n")
    f.write(f"Mean elevation: {elevations.mean():.0f}m ± {elevations.std():.0f}m\n")
    f.write(f"Slope range: {slopes.min():.1f}° to {slopes.max():.1f}°\n")
    f.write(f"Mean slope: {slopes.mean():.1f}° ± {slopes.std():.1f}°\n")
    f.write(f"Aspect range: {aspects.min():.1f}° to {aspects.max():.1f}°\n")
    f.write(f"Mean aspect: {aspects.mean():.1f}° (0°=N, 90°=E, 180°=S, 270°=W)\n")
    f.write(f"Aspect std: {aspects.std():.1f}°\n\n")
    
    # Outlier analysis
    f.write("OUTLIER ANALYSIS (Z-score > 3.0):\n")
    f.write("-" * 35 + "\n")
    for method, count in outlier_counts.items():
        total = merged[method].notna().sum() if method in merged.columns else 0
        pct = (count/total*100) if total > 0 else 0
        f.write(f"{method}: {count} outliers ({pct:.1f}% of {total} observations)\n")
    f.write("\n")
    
    # Show actual outlier values with dates
    f.write("ACTUAL OUTLIER VALUES WITH DATES:\n")
    f.write("-" * 50 + "\n")
    for method in ['MCD43A3', 'MOD09GA', 'MOD10A1', 'AWS']:
        if method in outlier_values and len(outlier_values[method]) > 0:
            values = outlier_values[method]
            f.write(f"\n{method} outliers ({len(values)} total):\n")
            f.write(f"  Range: {values.min():.3f} to {values.max():.3f}\n")
            f.write(f"  {'Date':<12} {'Value':<8}\n")
            f.write(f"  {'-'*20}\n")
            
            # Show first 10 outliers with dates
            for date, value in values[:10].items():
                f.write(f"  {date.strftime('%Y-%m-%d'):<12} {value:<8.3f}\n")
            
            if len(values) > 10:
                f.write(f"  ... ({len(values)-10} more outliers)\n")
            f.write("\n")
    f.write("\n")
    
    # Comparison WITH vs WITHOUT outliers
    f.write("STATISTICS COMPARISON (WITH vs WITHOUT OUTLIERS):\n")
    f.write("-" * 55 + "\n")
    f.write(f"{'Method':<10} {'Condition':<12} {'n':<4} {'r':<6} {'RMSE':<6} {'MAE':<6} {'Bias':<6}\n")
    f.write("-" * 55 + "\n")
    
    for method in ['MCD43A3', 'MOD09GA', 'MOD10A1']:
        if method in stats_with_outliers:
            # With outliers
            s_with = stats_with_outliers[method]
            f.write(f"{method:<10} {'With':<12} {s_with['n']:<4} {s_with['r']:<6.3f} {s_with['rmse']:<6.3f} {s_with['mae']:<6.3f} {s_with['bias']:<6.3f}\n")
            
            # Without outliers
            if method in stats_without_outliers:
                s_without = stats_without_outliers[method]
                f.write(f"{method:<10} {'Without':<12} {s_without['n']:<4} {s_without['r']:<6.3f} {s_without['rmse']:<6.3f} {s_without['mae']:<6.3f} {s_without['bias']:<6.3f}\n")
                
                # Improvement metrics
                r_improvement = ((s_without['r'] - s_with['r']) / abs(s_with['r'])) * 100 if s_with['r'] != 0 else 0
                rmse_improvement = ((s_with['rmse'] - s_without['rmse']) / s_with['rmse']) * 100 if s_with['rmse'] != 0 else 0
                f.write(f"{method:<10} {'Improvement':<12} {'':<4} {r_improvement:<6.1f}% {rmse_improvement:<6.1f}% {'':<6} {'':<6}\n")
            f.write("\n")
    f.write("\n")
    
    # Generated files
    f.write("GENERATED FILES:\n")
    f.write("-" * 20 + "\n")
    f.write("- albedo_comparison.png (overall scatter plots and time series)\n")
    f.write("- monthly_albedo_comparison.png (monthly scatter plots)\n")
    f.write("- pixel_level_comparison.png (pixel-level comparisons)\n")
    f.write("- individual_pixel_comparison.png (individual pixel analysis)\n")
    f.write("- spatial_pixel_maps_separate.png (separate maps for each method)\n")
    f.write("- pixel_characteristics_enhanced.png (enhanced terrain and glacier analysis)\n")
    f.write("- outliers_scatter.png (scatter plots with outliers highlighted)\n")
    f.write("- outliers_boxplot.png (box plots showing outlier distribution)\n")
    f.write("- albedo_analysis_results.txt (this summary file)\n\n")
    
    f.write("Analysis completed: " + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S') + "\n")
    f.write("="*80 + "\n")

