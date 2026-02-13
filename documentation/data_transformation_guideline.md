# Data Cleaning and Integration Pipeline Guide

## Overview

This script performs comprehensive data cleaning, transformation, and integration on the combined timeseries data with full logging.

## Quick Start
bash
python clean_and_integrate.py

## Pipeline Steps

### 1. Data Reduction
- Filters data to 2012-2024 (12 years exactly)
- Logs rows removed

### 2. Data Cleaning
- Handles missing values using mean imputation
- Imputes by indicator to maintain data integrity
- Logs imputation details per indicator

### 3. Data Transformation - Regression Smoothing
- Applies linear regression smoothing to Namibia records only
- Smooths each indicator separately
- Logs regression coefficients

### 4. Data Integration - Conflict Resolution
- Identifies duplicate records (same country, indicator, year)
- Resolves conflicts using source priority: WHO_GHO > IMF_WEO
- Logs conflicts found and resolved

### 5. Data Integration - Remove Overlaps
- Removes any remaining overlapping records
- Keeps first occurrence of exact duplicates
- Logs overlaps removed

### 6. Data Transformation - Decimal Rounding
- Rounds all values to 2 decimal places (nearest 100th)
- Logs value range after rounding

### 7. Output Generation
- Saves integrated dataset to `data/processed/integrated_timeseries.csv`
- Generates comprehensive summary statistics
- Logs all outputs

## Input

data/processed/combined_timeseries_long.csv

Expected schema:
- country
- indicator
- year
- value
- source
- country_code

## Output

### Data File
data/final/integrated_timeseries.csv

Final integrated dataset with all transformations applied.

### Log File
logs/data_pipeline_YYYYMMDD_HHMMSS.log

Comprehensive log of all pipeline operations.

## Logging Details

The script logs:
- Load operations: Rows, columns, indicators, sources, year range
- Data reduction: Original rows, filtered rows, rows removed
- Missing values: Before/after counts, imputation details per indicator
- Regression smoothing: Coefficients, records smoothed per indicator
- Conflict resolution: Duplicates found, duplicates removed
- Overlap removal: Overlapping records removed
- Decimal rounding: Value range after rounding
- Final summary: 
  - Total rows and indicators
  - Year range
  - Indicator breakdown with counts and means
  - Source breakdown with counts
  - Value statistics (min, max, mean, median, std dev)

## Configuration

### Year Rangepython
df = filter_year_range(df, start_year=2012, end_year=2024)

### Decimal Placespython
df = round_decimals(df, decimals=2)  # Nearest 100th

### Source Prioritypython
source_priority = {'WHO_GHO': 1, 'IMF_WEO': 2}  # Lower = higher priority

## Example Log Output

2024-02-12 10:30:15 - INFO - ================================================================================
2024-02-12 10:30:15 - INFO - DATA CLEANING AND TRANSFORMATION PIPELINE STARTED
2024-02-12 10:30:15 - INFO - ================================================================================

2024-02-12 10:30:15 - INFO - Loading data from: data/processed/combined_timeseries_long.csv
2024-02-12 10:30:15 - INFO -   Loaded successfully: 520 rows, 6 columns

2024-02-12 10:30:15 - INFO - DATA REDUCTION: Filtering to years 2012-2024
2024-02-12 10:30:15 - INFO -   Original rows: 520
2024-02-12 10:30:15 - INFO -   Filtered rows: 312
2024-02-12 10:30:15 - INFO -   Rows removed: 208

2024-02-12 10:30:15 - INFO - DATA CLEANING: Handling missing values
2024-02-12 10:30:15 - INFO -   Missing values before: 5
2024-02-12 10:30:15 - INFO -   gdp_per_capita: Imputed 2 values with mean 5234.67
2024-02-12 10:30:15 - INFO -   inflation: Imputed 3 values with mean 6.45
2024-02-12 10:30:15 - INFO -   Missing values after: 0
...

## Error Handling

- All operations wrapped in try-except blocks
- Errors logged with full stack traces
- Pipeline stops on critical errors
- Logs saved even if pipeline fails

## Validation

After running, check:
1. Log file for any warnings or errors
2. Final dataset has expected rows (≤ 52 rows: 4 indicators × 13 years)
3. Year range is 2012-2024
4. No missing values in output
5. All values rounded to 2 decimals