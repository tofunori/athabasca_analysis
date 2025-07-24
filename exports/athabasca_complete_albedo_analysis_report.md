# ATHABASCA Glacier - MODIS vs AWS Albedo Analysis Report

**Generated:** 2025-07-24 11:39:12

**Analysis includes:** Terra/Aqua fused data

---

## 1. Data Summary

- **Study period:** 2014-09-12 to 2020-09-18
- **Total observations:** 515
- **MODIS methods analyzed:** MCD43A3, MOD09GA (Terra+Aqua fused), MOD10A1 (Terra+Aqua fused)
- **AWS ground truth observations:** 2199
- **Overlapping MODIS-AWS observations:** 515

## 2. Overall Statistics

### 2.1 Basic Performance Metrics

| Method | n | r | R² | p-value | RMSE | MAE | Bias |
|--------|---|---|----|---------|----|-----|------|
| MCD43A3 | 332 | 0.642 | 0.412 | 0.000000 | 0.150 | 0.090 | -0.070 |
| MOD09GA | 252 | 0.508 | 0.258 | 0.000000 | 0.123 | 0.078 | -0.028 |
| MOD10A1 | 139 | 0.653 | 0.427 | 0.000000 | 0.144 | 0.107 | -0.062 |

### 2.2 Regression Statistics

| Method | Slope | Intercept | Std Error | Confidence Level |
|--------|-------|-----------|-----------|------------------|
| MCD43A3 | 0.8753 | 0.1000 | 0.0575 | High |
| MOD09GA | 0.5284 | 0.1434 | 0.0567 | High |
| MOD10A1 | 0.7069 | 0.1345 | 0.0700 | High |

### 2.3 Data Characteristics

| Method | MODIS Mean±SD | AWS Mean±SD | Agreement (±5%) | Agreement (±10%) | Agreement (±15%) |
|--------|---------------|--------------|----------------|------------------|------------------|
| MCD43A3 | 0.237±0.126 | 0.307±0.172 | 16.3% | 30.7% | 43.7% |
| MOD09GA | 0.245±0.119 | 0.273±0.124 | 11.9% | 23.8% | 34.9% |
| MOD10A1 | 0.248±0.150 | 0.310±0.162 | 6.5% | 16.5% | 25.2% |

### 2.4 Error Distribution (Absolute Differences)

| Method | Q25 | Median | Q75 | Q90 | Q95 |
|--------|-----|--------|-----|-----|-----|
| MCD43A3 | 0.017 | 0.049 | 0.089 | 0.290 | 0.402 |
| MOD09GA | 0.025 | 0.052 | 0.096 | 0.166 | 0.227 |
| MOD10A1 | 0.035 | 0.077 | 0.149 | 0.253 | 0.326 |

## 3. 16-Day Composite Analysis (Detailed)

### 3.1 Performance Summary

| Method | Periods | r | R² | RMSE | MAE | Bias | Within ±10% |
|--------|---------|---|----|----|-----|------|-------------|
| MCD43A3 | 37 | 0.613 | 0.375 | 0.157 | 0.104 | -0.092 | 29.7% |
| MOD09GA | 45 | 0.842 | 0.709 | 0.111 | 0.082 | -0.062 | 26.7% |
| MOD10A1 | 43 | 0.641 | 0.410 | 0.142 | 0.107 | -0.058 | 18.6% |

### 3.2 Detailed Composite Statistics

**MCD43A3:**
- Data periods: 37
- Correlation: r=0.613 (R²=0.375)
- Error metrics: RMSE=0.157, MAE=0.104, Bias=-0.092
- MODIS range: 0.569, AWS range: 0.651
- Agreement: ±5%=13.5%, ±10%=29.7%, ±15%=40.5%

**MOD09GA:**
- Data periods: 45
- Correlation: r=0.842 (R²=0.709)
- Error metrics: RMSE=0.111, MAE=0.082, Bias=-0.062
- MODIS range: 0.424, AWS range: 0.651
- Agreement: ±5%=17.8%, ±10%=26.7%, ±15%=31.1%

**MOD10A1:**
- Data periods: 43
- Correlation: r=0.641 (R²=0.410)
- Error metrics: RMSE=0.142, MAE=0.107, Bias=-0.058
- MODIS range: 0.645, AWS range: 0.651
- Agreement: ±5%=9.3%, ±10%=18.6%, ±15%=25.6%

## 4. Weekly Composite Analysis (Detailed)

### 4.1 Performance Summary

| Method | Periods | r | R² | RMSE | MAE | Bias | Within ±10% |
|--------|---------|---|----|----|-----|------|-------------|
| MCD43A3 | 69 | 0.574 | 0.329 | 0.161 | 0.109 | -0.088 | 30.4% |
| MOD09GA | 85 | 0.624 | 0.390 | 0.126 | 0.087 | -0.048 | 22.4% |
| MOD10A1 | 70 | 0.723 | 0.522 | 0.134 | 0.103 | -0.060 | 12.9% |

### 4.2 Detailed Weekly Statistics

**MCD43A3:**
- Weekly periods: 69
- Correlation: r=0.574 (R²=0.329, p=2.519e-07)
- Error metrics: RMSE=0.161, MAE=0.109, Bias=-0.088
- Mean absolute relative error: 27.3%
- Data means: MODIS=0.246±0.120, AWS=0.334±0.160

**MOD09GA:**
- Weekly periods: 85
- Correlation: r=0.624 (R²=0.390, p=1.712e-10)
- Error metrics: RMSE=0.126, MAE=0.087, Bias=-0.048
- Mean absolute relative error: 25.2%
- Data means: MODIS=0.270±0.113, AWS=0.317±0.147

**MOD10A1:**
- Weekly periods: 70
- Correlation: r=0.723 (R²=0.522, p=1.639e-12)
- Error metrics: RMSE=0.134, MAE=0.103, Bias=-0.060
- Mean absolute relative error: 31.2%
- Data means: MODIS=0.272±0.155, AWS=0.332±0.166

## 5. Monthly Analysis (Detailed)

### 5.1 Monthly Summary by Method

| Method | Months | Avg r | Avg R² | Avg RMSE | Avg MAE | Avg Bias | Best Month (r) | Worst Month (r) |
|--------|--------|-------|--------|----------|---------|----------|----------------|------------------|
| MCD43A3 | 4 | 0.385 | 0.220 | 0.154 | 0.104 | -0.086 | Sep (0.704) | Jun (0.089) |
| MOD09GA | 4 | 0.477 | 0.242 | 0.120 | 0.081 | -0.030 | Sep (0.632) | Aug (0.363) |
| MOD10A1 | 4 | 0.532 | 0.368 | 0.135 | 0.103 | -0.047 | Aug (0.829) | Jul (0.074) |

### 5.2 Detailed Monthly Statistics

**June:**

| Method | n | r | R² | RMSE | MAE | Bias | Within ±10% | MARE% |
|--------|---|---|----|----|-----|------|-------------|-------|
| MCD43A3 | 54 | 0.089 | 0.008 | 0.217 | 0.153 | -0.133 | 14.8% | 34.2% |
| MOD09GA | 27 | 0.552 | 0.304 | 0.119 | 0.080 | -0.011 | 18.5% | 27.0% |
| MOD10A1 | 22 | 0.494 | 0.244 | 0.153 | 0.112 | 0.010 | 27.3% | 36.3% |

**July:**

| Method | n | r | R² | RMSE | MAE | Bias | Within ±10% | MARE% |
|--------|---|---|----|----|-----|------|-------------|-------|
| MCD43A3 | 100 | 0.151 | 0.023 | 0.070 | 0.052 | -0.047 | 32.0% | 19.1% |
| MOD09GA | 69 | 0.363 | 0.132 | 0.075 | 0.048 | -0.012 | 31.9% | 21.3% |
| MOD10A1 | 28 | 0.074 | 0.005 | 0.124 | 0.088 | -0.029 | 14.3% | 37.3% |

**August:**

| Method | n | r | R² | RMSE | MAE | Bias | Within ±10% | MARE% |
|--------|---|---|----|----|-----|------|-------------|-------|
| MCD43A3 | 115 | 0.596 | 0.355 | 0.095 | 0.057 | -0.022 | 34.8% | 20.5% |
| MOD09GA | 95 | 0.363 | 0.132 | 0.121 | 0.068 | -0.000 | 27.4% | 30.4% |
| MOD10A1 | 39 | 0.829 | 0.686 | 0.076 | 0.063 | -0.046 | 12.8% | 26.8% |

**September:**

| Method | n | r | R² | RMSE | MAE | Bias | Within ±10% | MARE% |
|--------|---|---|----|----|-----|------|-------------|-------|
| MCD43A3 | 63 | 0.704 | 0.495 | 0.233 | 0.155 | -0.143 | 34.9% | 27.3% |
| MOD09GA | 61 | 0.632 | 0.400 | 0.166 | 0.127 | -0.095 | 11.5% | 31.6% |
| MOD10A1 | 50 | 0.731 | 0.535 | 0.186 | 0.149 | -0.124 | 16.0% | 39.1% |

## 6. Outlier Impact Analysis (Detailed)

### 6.1 Outlier Detection Summary

**Z-score Outlier Detection (threshold > 3):**

- **MCD43A3:** 22 outliers (5.6% of 391 observations)
- **MOD09GA:** 8 outliers (2.7% of 294 observations)
- **MOD10A1:** 3 outliers (1.8% of 167 observations)
- **AWS:** 11 outliers (2.5% of 432 observations)

### 6.2 Statistics Comparison (With vs Without Outliers)

| Method | Condition | n | r | RMSE | MAE | Bias | r Improvement | RMSE Improvement |
|--------|-----------|---|---|------|-----|------|---------------|------------------|
| MCD43A3 | With | 332 | 0.642 | 0.150 | 0.090 | -0.070 | - | - |
| MCD43A3 | Without | 314 | 0.675 | 0.135 | 0.081 | -0.075 | +5.1% | +10.2% |
| MOD09GA | With | 252 | 0.508 | 0.123 | 0.078 | -0.028 | - | - |
| MOD09GA | Without | 245 | 0.717 | 0.093 | 0.067 | -0.037 | +41.3% | +24.5% |
| MOD10A1 | With | 139 | 0.653 | 0.144 | 0.107 | -0.062 | - | - |
| MOD10A1 | Without | 134 | 0.731 | 0.132 | 0.100 | -0.075 | +11.9% | +8.5% |

## 7. Data Quality and Availability

### 7.1 Data Completeness by Method

| Method | Available | Missing | Completeness | Valid with AWS |
|--------|-----------|---------|--------------|----------------|
| MCD43A3 | 391 | 124 | 75.9% | 332 |
| MOD09GA | 294 | 221 | 57.1% | 252 |
| MOD10A1 | 167 | 348 | 32.4% | 139 |

## 8. Summary and Conclusions

### 8.1 Overall Performance Ranking

1. **MOD10A1** - r=0.653, RMSE=0.144, Agreement(±10%)=16.5%
2. **MCD43A3** - r=0.642, RMSE=0.150, Agreement(±10%)=30.7%
3. **MOD09GA** - r=0.508, RMSE=0.123, Agreement(±10%)=23.8%

### 8.2 Key Findings

- **Analysis period:** 2198 days
- **Total observations:** 515 overlapping MODIS-AWS pairs
- **Methods analyzed:** Terra/Aqua fusion applied to MOD09GA and MOD10A1 products
- **Temporal scales:** Daily, weekly composites, 16-day composites, monthly analysis
- **Quality control:** Comprehensive outlier analysis performed
- **Validation approach:** Multiple statistical metrics and agreement assessments

---

*End of Comprehensive Analysis Report*
*Generated: 2025-07-24 11:39:12*
