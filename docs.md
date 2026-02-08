# Scalable Data Pipeline for Public Health & Cost of Living Insights in Namibia

## List datasets with source, type, and time coverage.
Sourced from WHO Global Health Observatory 
- Consumption expenditure % of GDP

Sourced from IMF WEO Namibia dataset
- GDP per capita
- Inflation
- General government gross debt

## Datasets considered but dropped due to lack of relevant data
WHO Global Health Observatory 
- physicians per 1000
- life expectancy
- mortality due to unsafe WASH

World Bank Open Data
- GDP per capita
- inflation, consumer prices
- consumption expenditure (% of GDP)

## Discoveries upon dataset inspections
All three datasets sourced from the World Bank Open Data were void of values. As a result, I opted to source similar datasets with actual values from the International Monetary Fund World Economic Outlook (IMF WEO). Furthermore, I found that both IMF and WHO datasets had wide formats.

## Data Cleaning Steps Applied
1. Convert year column to integer
2. Ensure value column is named 'value'
3. Drop missing values in 'value' column
4. Filter to Namibia only
5. Standardize column names
6. Add ISO country codes

## Canonical Schema
String      country	
String      indicator	
Int         year	
Float       value	
String      source
String      country_code

## For data melting documentation check out melt_guide.md
