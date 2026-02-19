# Scalable Data Pipeline for Public Health Cost of Living Insights in Namibia
# Processing script for dataset integration, cleaning, and transformation with logging

## Author: Laura Conceicao Uuyuni
## Date: 2026-02-11
## This reusable scripts cleans and integrates the timeseries data from three sources with logging. 
## It includes steps for filtering by year range, handling missing values, applying regression smoothing to Namibia data, 
## resolving data conflicts, removing overlapping records, rounding values, and saving the cleaned dataset. 
## Summary statistics are generated at the end of the process.

"""
Data cleaning, transformation, and integration pipeline.
The processes combined the timeseries data from three sources with logging.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from sklearn.linear_model import LinearRegression
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
input_filepath = BASE_DIR / "data" / "combined_timeseries.csv"
output_filepath = BASE_DIR / "data" / "cleaned_integrated_timeseries.csv"

# Configure logging to file and console
def setup_logging():
    # Create logs directory if it doesn't exist
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a timestamped log file
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file = f'{log_dir}/data_transform_{timestamp}.log'
    
    # Configure logging to write to file and console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        # Log to both file and console
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info("="*70)
    logging.info("DATA CLEANING AND INTEGRATION PIPELINE STARTED")
    logging.info("="*70)
    logging.info(f"Log file created: {log_file}")
    
    return log_file

# Load combined timeseries data
def load_data(filepath):
    """ Load combined timeseries data"""
    logging.info(f"\nLoading data from: {filepath}")
    
    try:
        # Load data with parsing timestamp column
        df = pd.read_csv(filepath)
        
        # Schema validation
        required_columns = ['country', 'indicator', 'year', 'value', 'source']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f" Missing required columns: {missing_columns}")
        
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        
        # Log basic data info
        logging.info(f" Loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        logging.info(f" Columns: {list(df.columns)}")
        logging.info(f" Indicators: {df['indicator'].unique().tolist()}")
        logging.info(f" Sources: {df['source'].unique().tolist()}")
        logging.info(f" Year range: {df['year'].min()} - {df['year'].max()}")
        return df
    
    # Catch and log any exceptions during data loading
    except Exception as e:
        logging.error(f" Error loading data: {str(e)}")
        raise
    
# Data cleaning and transformation functions    
def filter_year_range(df, start_year=2000, end_year=2024):
    """Reduce data to specified year range"""
    logging.info(f"\n DATA REDUCTION: Filtering data to years {start_year} - {end_year}")
    
    # Log original row count before filtering
    original_rows = len(df)
    
    # Filter data to specified year range
    df_filtered = df[(df['year'] >= start_year) & (df['year'] <= end_year)]
    
    # Log filtering results
    logging.info(f" Original rows: {original_rows}")
    logging.info(f" Filtered rows: {len(df_filtered)}")
    logging.info(f" Rows removed: {original_rows - len(df_filtered)}")
    return df_filtered

def handle_missing_values(df):
    """Handle missing values using mean imputation by indicator"""
    logging.info("\n DATA CLEANING: Handling missing values")
    
    # Log missing value counts before imputation
    missing_before = df['value'].isna().sum()
    logging.info(f" Missing values before: {missing_before}")
    
    # Impute missing values with mean by indicator
    if missing_before > 0:
        # Log indicators with missing values
        for (country, indicator), group in df.groupby(['country', 'indicator']):
            missing_count = group['value'].isna().sum()
            
            if missing_count == 0:
                continue
            
            mean_value = np.nanmean(group['value'].values)
            
            # Log mean value for imputation
            if not pd.isna(mean_value):
                df.loc[group.index, 'value'] = group['value'].fillna(mean_value)
                logging.info(
                    f" {indicator}: Imputed {missing_count} values with mean {mean_value:.2f}"
                )
            else:
                logging.warning(
                    f" {indicator}: Cannot impute - mean is NaN (no valid values available)"
                )
              
    # Log missing value counts after imputation            
    missing_after = df['value'].isna().sum()
    logging.info(f" Missing values after: {missing_after}")
    return df

# Apply regression smoothing to Namibia data only
def apply_regression_smoothing(df):
    """Apply regression smoothing to Namibia data only"""
    logging.info("\n DATA TRANSFORMATION: Applying regression smoothing to Namibia data")
    
    namibia_mask = df['country'] == 'Namibia'
    logging.info(f" Processing {namibia_mask.sum()} rows for Namibia")
    
    # Create a copy of the dataframe to store smoothed values
    df_smoothed = df.copy()
    
    # Apply regression smoothing for each indicator separately
    for indicator in df['indicator'].unique():
        mask = namibia_mask & (df['indicator'] == indicator)
        data_subset = df.loc[mask].dropna(subset=['value'])
            
        if len(data_subset) < 2:
            logging.warning(
                f" {indicator}: Insufficient data points for regression smoothing "
                f" (only {len(data_subset)} valid rows)" 
            )
            continue
            
        # Prepare data for regression
        X = data_subset['year'].values.reshape(-1, 1)
        y = data_subset['value'].values
        
        # Fit linear regression model
        model = LinearRegression()
        model.fit(X, y)
        y_smoothed = model.predict(X)    
        
        # Update smoothed values in the dataframe
        df_smoothed.loc[data_subset.index, 'value'] = y_smoothed
        
        logging.info(f" {indicator}: Smoothed {len(data_subset)} values")
        logging.info(
            f" Regression coefficients: y = {model.coef_[0]:.4f} * year + {model.intercept_:.2f}"
            )

    return df_smoothed

def resolve_data_conflicts(df):
    """Resolve conflicts by using latest year coverage and source priority"""
    logging.info("\n DATA INTEGRATION: Resolving data conflicts")
    
    original_rows = len(df)
    
    # Calculate max available year per source and indicator
    max_year_per_source = (
        df.groupby(['indicator', 'source'])['year']
        .max()
        .reset_index()
        .rename(columns={'year': 'max_source_year'})
    )
    
    # Merge max year info back to original dataframe
    df = df.merge(
        max_year_per_source,
        on=['indicator', 'source'],
        how='left'
    )
    
    # Define source priority (lower number = higher priority)
    source_priority = {'WHO_GHO': 1, 'IMF_WEO': 2}
    df['source_priority'] = df['source'].map(source_priority)
    
    # Sort by max_source_year (descending) and lowest source_priority
    df_sorted = df.sort_values(
        by=['country', 'indicator', 'year', 'max_source_year', 'source_priority'],
        ascending=[True, True, True, False, True]
    )
    
    # Drop duplicates, keeping best candidate based on sorting
    df_resolved = df_sorted.drop_duplicates(
        subset=['country', 'indicator', 'year'],
        keep='first'
    )
    
    # Clean up temporary columns
    df_resolved = df_resolved.drop(
        ['source_priority', 'max_source_year'],
        axis=1
    )
    
    removed_count = original_rows - len(df_resolved)
    
    # Log conflict resolution results
    if removed_count > 0:
        logging.info(f" Removed {removed_count} conflicting records")
    else:
        logging.info(" No duplicate records found")
    
    logging.info(f" Remaining rows: {len(df_resolved)}")
    return df_resolved

# Remove any remaining overlapping records after conflict resolution    
def remove_overlapping_records(df):
    """Remove any remaining overlapping records after conflict resolution"""
    logging.info("\n DATA INTEGRATION: Removing overlapping records")
    
    original_rows = len(df)
    
    df_unique = df.drop_duplicates(
        subset=['country', 'indicator', 'year', 'value'],
        keep='first'
    )
    
    removed_count = original_rows - len(df_unique)
    
    if removed_count > 0:
        logging.info(f" Removed {removed_count} conflicting records")
    else:
        logging.info(" No duplicate records found")
        
    logging.info(f" Remaining rows: {len(df_unique)}")
    return df_unique

# Round values to specified decimal places for consistency
def round_decimals(df, decimal_places=2):
    """Round values to specified decimal places"""
    logging.info(f"\n DATA TRANSFORMATION: Rounding values to {decimal_places} decimal places")
    
    df['value'] = df['value'].round(decimal_places)
    logging.info(f" Value range: {df['value'].min():.2f} - {df['value'].max():.2f}")
    
    return df

# Save processed data to CSV file
def save_cleaned_data(df, output_filepath):
    """Save processed data to CSV file"""
    logging.info(f"\n Saving cleaned data to: {output_filepath}")
    
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    
    df.to_csv(output_filepath, index=False)
    
    logging.info(f" Saved successfully")
    logging.info(f" Final dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    logging.info(f" Columns: {list(df.columns)}")
    
    return output_filepath

# Generate summary statistics for the cleaned dataset
def generate_summary_statistics(df):
    """Generate summary statistics for the cleaned dataset"""
    logging.info("\n" + "="*70)
    logging.info("\n FINAL DATASET SUMMARY STATISTICS:")
    logging.info("="*70)
    
    total_records = len(df)
    year_range = (df['year'].min(), df['year'].max())
    
    logging.info(f"\n Overall Statistics:")
    logging.info(f" Total records: {total_records}")
    logging.info(f" Total rows: {len(df)}")
    logging.info(f" Total indicators: {df['indicator'].nunique()}")
    logging.info(f" Total countries: {df['country'].unique().tolist()}")
    logging.info(f" Year range: {year_range[0]} - {year_range[1]}")
    
    logging.info(f" \n Indicator-specific Statistics:")
    for indicator in df['indicator'].unique():
        subset = df[df['indicator'] == indicator]
        logging.info(f" {indicator}:")
        logging.info(f"  Records: {len(subset)}")
        logging.info(f"  Year range: {subset['year'].min()} - {subset['year'].max()}")
        logging.info(f"  Value range: {subset['value'].min():.2f} - {subset['value'].max():.2f}")
        
    logging.info(f" \n Source-specific Statistics:")
    for source in df['source'].unique():
        subset = df[df['source'] == source]
        logging.info(f" {source}:")
        logging.info(f"  Records: {len(subset)}")
        logging.info(f"  Year range: {subset['year'].min()} - {subset['year'].max()}")
        logging.info(f"  Value range: {subset['value'].min():.2f} - {subset['value'].max():.2f}")
        
    logging.info(f"\n Value Statistics:")
    logging.info(f"  Overall value range: {df['value'].min():.2f} - {df['value'].max():.2f}")
    logging.info(f"  Mean value: {df['value'].mean():.2f}")
    logging.info(f"  Median value: {df['value'].median():.2f}")
    logging.info(f"  Standard deviation: {df['value'].std():.2f}")

# Main function to execute the data cleaning and integration pipeline    
def main(
    # Main function parameters
    input_filepath='data/processed/combined_timeseries_data.csv',
    output_filepath='data/processed/cleaned_integrated_timeseries.csv',
    start_year=2000,
    end_year=2024,
    decimal_places=2
):
    """Main function to execute the data cleaning and integration pipeline"""
    log_file = setup_logging()
    
    try:
        # Load combined timeseries data
        df = load_data(input_filepath)
        
        if df.empty:
            logging.warning(" Loaded dataset is empty. Exiting pipeline.")
            return
        
        # Apply data cleaning and transformation steps
        df = filter_year_range(df, start_year=start_year, end_year=end_year)
        
        # Handle missing values with mean imputation by indicator
        df = handle_missing_values(df)
        
        # Resolve data conflicts by keeping the most recent data source
        df = resolve_data_conflicts(df)
        
        # Remove any remaining overlapping records after conflict resolution
        df = remove_overlapping_records(df)
        
        # Apply regression smoothing to Namibia data only
        df = apply_regression_smoothing(df)
        
        # Round values to specified decimal places for consistency
        df = round_decimals(df, decimal_places=decimal_places)
        
        # Save output to CSV file
        save_cleaned_data(df, output_filepath)
        
        # Generate summary statistics for the cleaned dataset
        generate_summary_statistics(df)
        
    except Exception as e:
        logging.error(f" An error occurred during processing: {str(e)}")
        raise
    finally:
        logging.info("\n" + "="*70)
        logging.info("DATA CLEANING AND INTEGRATION PIPELINE COMPLETED")
        logging.info("="*70)
        
if __name__ == "__main__":
    main()