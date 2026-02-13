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

# Configure logging to file and console
def setup_logging():
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file = f'{log_dir}/data_transform_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
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

def load_data(filepath):
    """ Load combined timeseries data"""
    logging.info(f"\nLoading data from: {filepath}")
    
    try:
        df = pd.read_csv(filepath, parse_dates=['timestamp'])
        logging.info(f" Loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        logging.info(f" Columns: {list(df.columns)}")
        logging.info(f" Indicators: {df['indicator'].unique().tolist()}")
        logging.info(f" Sources: {df['source'].unique().tolist()}")
        logging.info(f" Year range: {df['year'].min()} - {df['year'].max()}")
        return df
    except Exception as e:
        logging.error(f" Error loading data: {str(e)}")
        raise
    
def filter_year_range(df, start_year=2000, end_year=2024):
    """Reduce data to specified year range"""
    logging.info(f"\n DATA REDUCTION: Filtering data to years {start_year} - {end_year}")
    
    original_rows = len(df)
    df_filtered = df[(df['year'] >= start_year) & (df['year'] <= end_year)]
    
    logging.info(f" Original rows: {original_rows}")
    logging.info(f" Filtered rows: {len(df_filtered)}")
    logging.info(f" Rows removed: {original_rows - len(df_filtered)}")
    return df_filtered

def handle_missing_values(df):
    """Handle missing values using mean imputation by indicator"""
    logging.info("\n DATA CLEANING: Handling missing values")
    
    missing_before = df['value'].isna().sum()
    logging.info(f" Missing values before: {missing_before}")
    
    if missing_before > 0:
        for indicator in df['indicator'].unique():
            mask = df['indicator'] == indicator
            mean_value = df.loc[mask, 'value'].mean()
            
            missing_count = df.loc[mask, 'value'].isna().sum()
            if missing_count > 0:
                df.loc[mask & df['value'].isna(), 'value'] = mean_value
                logging.info(f" {indicator}: Imputed {missing_count} values with mean {mean_value:.2f}")
                
    missing_after = df['value'].isna().sum()
    logging.info(f" Missing values after: {missing_after}")
    return df

def apply_regression_smoothing(df):
    """Apply regression smoothing to Namibia data only"""
    logging.info("\n DATA TRANSFORMATION: Applying regression smoothing to Namibia data")
    
    namibia_mask = df['country'] == 'Namibia'
    logging.info(f" Processing {namibia_mask.sum()} rows for Namibia")
    
    df_smoothed = df.copy()
    
    for indicator in df['indicator'].unique():
        mask = namibia_mask & (df['indicator'] == indicator)
        data_subset = df[mask].copy()
        
        if len(data_subset) < 2:
            logging.warning(f" {indicator}: Insufficient data for regression smoothing (only {len(data_subset)} rows)")
            continue
        
        X = data_subset['year'].values.reshape(-1, 1)
        y = data_subset['value'].values
        
        model = LinearRegression()
        model.fit(X, y)
        y_smoothed = model.predict(X)
        
        df_smoothed.loc[mask, 'value'] = y_smoothed
        
        logging.info(f" {indicator}: Smoothed {len(data_subset)} values")
        logging.info(f" Regression: y = {model.coef_[0]:.4f} * x + {model.intercept_:.2f}")
    return df_smoothed

def resolve_data_conflicts(df):
    """Resolve conflicts by keeping most recent data source"""
    logging.info("\n DATA INTEGRATION: Resolving data conflicts")
    
    original_rows = len(df)
    
    duplicates = df.duplicated(subset=['country', 'indicator', 'year'], keep=False)
    duplicate_count = duplicates.sum()
    
    if duplicate_count > 0:
        logging.inifo(f" Found {duplicate_count} duplicate rows based on country, indicator, and year")
        
        source_priority = {'WHO_GHO': 1, 'IMF':3}
        df['source_priority'] = df['source'].map(source_priority)
        
        df_resolved = df.sort_values('source_priority').drop_duplicates(
            subset=['country', 'indicator', 'year'], 
            keep='first'
        )
        
        df_resolved = df_resolved.drop('source_priority', axis=1)
        
        removed_count = original_rows - len(df_resolved)
        logging.info(f" Removed {removed_count} duplicate records")
        logging.info(f" Remaining rows: {len(df_resolved)}")
        
        return df_resolved
    else:
        logging.info(" No duplicate records found")
        return df
    
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
        logging.info(f" Removed {removed_count} overlapping records")
    else:
        logging.info(" No overlapping records found")
        
    logging.info(f" Remaining rows: {len(df_unique)}")
    return df_unique

def round_decimals(df, decimal_places=2):
    """Round values to specified decimal places"""
    logging.info(f"\n DATA TRANSFORMATION: Rounding values to {decimal_places} decimal places")
    
    df['value'] = df['value'].round(decimal_places)
    logging.info(f" Value range: {df['value'].min():.2f} - {df['value'].max():.2f}")
    
    return df

def save_cleaned_data(df, output_filepath):
    """Save processed data to CSV file"""
    logging.info(f"\n Saving cleaned data to: {output_filepath}")
    
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    
    df.to_csv(output_filepath, index=False)
    
    logging.info(f" Saved successfully")
    logging.info(f" Final dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    logging.info(f" Columns: {list(df.columns)}")
    
    return output_filepath

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
    
def main():
    """Main function to execute the data cleaning and integration pipeline"""
    log_file = setup_logging()
    
    try:
        # Load combined timeseries data
        df = load_data('data/combined_timeseries.csv')
        
        # Apply data cleaning and transformation steps
        df = filter_year_range(df, start_year=2000, end_year=2024)
        
        # Handle missing values with mean imputation by indicator
        df = handle_missing_values(df)
        
        # Apply regression smoothing to Namibia data only
        df = apply_regression_smoothing(df)
        
        # Resolve data conflicts by keeping the most recent data source
        df = resolve_data_conflicts(df)
        
        # Remove any remaining overlapping records after conflict resolution
        df = remove_overlapping_records(df)
        
        # Round values to 2 decimal places for consistency
        df = round_decimals(df, decimal_places=2)
        
        # Save output to CSV file
        output_filepath = 'data/cleaned_integrated_timeseries.csv'
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