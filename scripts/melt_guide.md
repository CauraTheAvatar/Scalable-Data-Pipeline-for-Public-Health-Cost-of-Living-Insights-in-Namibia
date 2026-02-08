# Time Series Data Transformation Guide

## Quick Start
python melt.py


## What This Script Does
Transforms wide-format data = long-format data
Processes 3 IMF economic indicators
Processes 1 WHO health indicators
Filters data to Namibia only
Standardizes schema across all datasets
Adds ISO country codes

## Expected Input Files

### IMF Data (Economic Indicators)
Place these in 'data/raw/':
- 'gdp-per-capita-imf.csv'
- 'general-government-gross-debt-imf.csv'
- 'inflation-rate-imf.csv'

Country column: 'Respective indicator'
Format: Countries as rows, years as columns

### WHO Data (Health Indicators)
Place these in 'data/raw/':
- 'consumption_expenditure_%_of_gdp_who.csv'

Country column: 'Country'
Format: Countries as rows, years as columns

## Output Files
data/processed/imf_timeseries_long.csv
data/processed/who_timeseries_long.csv

## Final Schema
All output files have this structure:

'country'       Country name
'indicator'     Indicator code
'year'          Year (integer)
'value'         Numeric value 
'source'        Data source
'country_code'  ISO code

## Core Function: 'melt_timeseries()'

def melt_timeseries(df, country_col, indicator_name, source_name):
    """
    Reusable function to transform wide → long format
    
    Steps:
    1. Identifies year columns (numeric column names)
    2. Uses pd.melt() to pivot data
    3. Renames country column to 'country'
    4. Adds 'indicator' column
    5. Adds 'source' column
    6. Converts year to integer
    7. Drops missing values
    8. Returns standardized DataFrame
    """

## Customization

### Adding More IMF Indicators

imf_datasets = {
    "gdp_per_capita": "data/raw/gdp-per-capita-imf.csv
",
    "inflation": "data/raw/general-government-gross-debt-imf
.csv",
    "gov_debt": "data/raw/inflation-rate-imf.csv",
    "your_new_indicator": "data/raw/imf_your_file.csv",  # Add here
}


### Adding More Countries

# In process_imf_data(), change:
df = df[df[country_col] == "Namibia"]

# To:
df = df[df[country_col].isin(["Namibia", "South Africa", "Botswana"])]

# Update country mapping:
country_map = {
    "Namibia": "NAM",
    "South Africa": "ZAF",
    "Botswana": "BWA"
}

## Data Cleaning Steps Applied
1. Convert year column to integer
2. Ensure value column is named 'value'
3. Drop missing values in 'value' column
4. Filter to Namibia only
5. Standardize column names
6. Add ISO country codes

## Troubleshooting

File not found errors:
- Check file names match exactly (case-sensitive)
- Ensure files are in 'data/raw/' directory

Column name errors:
- Verify country column name in your CSV
- Update 'country_col' variable if different

Year column detection issues:
- Function auto-detects numeric column names
- Years should be column headers (e.g., 2010, 2011, 2012...)

No data after filtering:
- Check country name spelling matches exactly
- Verify "Namibia" exists in your dataset