import pandas as pd
import numpy as np
import re
from datetime import datetime

def clean_startup_data(file_path):
    """
    Clean and process the startup funding data
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Clean column names
    df.columns = df.columns.str.strip().str.lower()
    
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Extract year and month
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['month_year'] = df['date'].dt.to_period('M')
    
    # Clean startup names - remove quotes and special characters
    df['startup'] = df['startup'].astype(str)
    df['startup'] = df['startup'].str.replace('"', '').str.replace("'", "").str.strip()
    df['startup'] = df['startup'].str.replace(r'\\x[a-fA-F0-9]{2}', '', regex=True)
    
    # Clean city names
    df['city'] = df['city'].astype(str)
    df['city'] = df['city'].str.replace(',', '').str.strip()
    df['city'] = df['city'].str.split(',').str[0]  # Take first city if multiple
    
    # Standardize major city names
    city_mapping = {
        'Gurgaon': 'Gurugram',
        'Noida': 'Noida',
        'New Delhi': 'Delhi',
        'Mumbai': 'Mumbai',
        'Bangalore': 'Bengaluru',
        'Pune': 'Pune',
        'Hyderabad': 'Hyderabad',
        'Chennai': 'Chennai',
        'Kolkata': 'Kolkata',
        'Ahmedabad': 'Ahmedabad'
    }
    
    # Apply city name standardization
    for old_name, new_name in city_mapping.items():
        df['city'] = df['city'].str.replace(old_name, new_name, case=False)
    
    # Clean amount column
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    
    # Fill missing amounts with 0
    df['amount'] = df['amount'].fillna(0)
    
    # Create amount categories
    df['amount_category'] = pd.cut(df['amount'], 
                                   bins=[0, 1, 10, 50, 100, 500, float('inf')],
                                   labels=['<1Cr', '1-10Cr', '10-50Cr', '50-100Cr', '100-500Cr', '>500Cr'])
    
    # Clean round column
    df['round'] = df['round'].astype(str).str.strip()
    df['round'] = df['round'].str.replace('Series ', 'Series-')
    
    # Standardize round names
    round_mapping = {
        'Seed Funding': 'Seed',
        'Seed/ Angel Funding': 'Seed',
        'Seed/Angel Funding': 'Seed',
        'Angel Funding': 'Angel',
        'Angel / Seed Funding': 'Angel',
        'Private Equity Round': 'Private Equity',
        'Private Equity': 'Private Equity',
        'Debt Funding': 'Debt',
        'Bridge Round': 'Bridge',
        'Series A': 'Series A',
        'Series B': 'Series B',
        'Series C': 'Series C',
        'Series D': 'Series D',
        'Series E': 'Series E',
        'Series F': 'Series F'
    }
    
    for old, new in round_mapping.items():
        df['round'] = df['round'].str.replace(old, new)
    
    # Clean vertical and subvertical
    df['vertical'] = df['vertical'].astype(str).str.strip()
    df['subvertical'] = df['subvertical'].astype(str).str.strip()
    
    # Standardize vertical names
    vertical_mapping = {
        'E-Commerce': 'eCommerce',
        'Ecommerce': 'eCommerce',
        'ECommerce': 'eCommerce',
        'Consumer Internet': 'Consumer Tech',
        'Technology': 'Tech',
        'FinTech': 'Fintech',
        'Fin-Tech': 'Fintech',
        'Food and Beverage': 'Food & Beverage',
        'Food & Beverages': 'Food & Beverage',
        'Healthcare': 'Health',
        'Health and Wellness': 'Health',
        'Education': 'EdTech',
        'Ed-Tech': 'EdTech'
    }
    
    for old, new in vertical_mapping.items():
        df['vertical'] = df['vertical'].str.replace(old, new)
    
    # Clean investors column
    df['investors'] = df['investors'].astype(str)
    df['investor_count'] = df['investors'].str.count(',') + 1
    df['investor_count'] = df['investor_count'].where(df['investors'] != 'nan', 0)
    
    # Remove rows with missing critical data
    df = df.dropna(subset=['date', 'startup'])
    
    # Remove outliers (amounts > 99th percentile might be data errors)
    amount_99th = df['amount'].quantile(0.99)
    df = df[df['amount'] <= amount_99th * 2]  # Allow some extreme values
    
    return df

def get_funding_insights(df):
    """
    Generate insights from the funding data
    """
    insights = {}
    
    # Total funding and deals
    insights['total_funding'] = df['amount'].sum()
    insights['total_deals'] = len(df)
    insights['avg_deal_size'] = df['amount'].mean()
    
    # Year-wise trends
    insights['yearly_funding'] = df.groupby('year')['amount'].sum().sort_index()
    insights['yearly_deals'] = df.groupby('year').size().sort_index()
    
    # Top categories
    insights['top_cities'] = df.groupby('city')['amount'].sum().sort_values(ascending=False).head(10)
    insights['top_verticals'] = df.groupby('vertical')['amount'].sum().sort_values(ascending=False).head(10)
    insights['top_rounds'] = df.groupby('round')['amount'].sum().sort_values(ascending=False)
    
    # Growth metrics
    current_year = df['year'].max()
    previous_year = current_year - 1
    
    current_year_funding = df[df['year'] == current_year]['amount'].sum()
    previous_year_funding = df[df['year'] == previous_year]['amount'].sum()
    
    if previous_year_funding > 0:
        insights['yoy_growth'] = ((current_year_funding - previous_year_funding) / previous_year_funding) * 100
    else:
        insights['yoy_growth'] = 0
    
    return insights

if __name__ == "__main__":
    # Clean the data
    df_clean = clean_startup_data('startup_cleaned.csv')
    
    # Save cleaned data
    df_clean.to_csv('startup_data_processed.csv', index=False)
    
    # Generate insights
    insights = get_funding_insights(df_clean)
    
    print("Data cleaning completed!")
    print(f"Total records: {len(df_clean)}")
    print(f"Date range: {df_clean['date'].min()} to {df_clean['date'].max()}")
    print(f"Top cities: {df_clean['city'].value_counts().head().index.tolist()}")
    print(f"Top sectors: {df_clean['vertical'].value_counts().head().index.tolist()}")
