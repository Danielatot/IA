import pandas as pd
import os
import re

def load_financial_data(balance_path, pnl_path):
    """
    Load balance sheet and PnL data from Excel files into pandas DataFrames organized by year.
    
    Args:
        balance_path (str): Path to directory containing balance sheet Excel files
        pnl_path (str): Path to directory containing PnL Excel files
    
    Returns:
        dict: Dictionary where keys are years and values are tuples of (balance_df, pnl_df)
    
    Raises:
        FileNotFoundError: If either directory doesn't exist
        ValueError: If files don't follow expected naming pattern
    """
    # Initialize output dictionary
    financial_data = {}
    
    # Compile regex pattern to extract year from filename
    year_pattern = re.compile(r'.*_(\d{4})\.xlsx$')
    
    # Process balance sheets
    balance_files = {}
    for file in os.listdir(balance_path):
        match = year_pattern.match(file)
        if match:
            year = match.group(1)
            balance_files[year] = os.path.join(balance_path, file)
    
    # Process PnL statements
    pnl_files = {}
    for file in os.listdir(pnl_path):
        match = year_pattern.match(file)
        if match:
            year = match.group(1)
            pnl_files[year] = os.path.join(pnl_path, file)
    
    # Verify we have matching files for each year
    common_years = set(balance_files.keys()) & set(pnl_files.keys())
    if not common_years:
        raise ValueError("No matching year files found between balance and PnL directories")
    
    # Load data for each year
    for year in common_years:
        balance_df = pd.read_excel(balance_files[year])
        pnl_df = pd.read_excel(pnl_files[year])
        financial_data[year] = (balance_df, pnl_df)
    
    return financial_data
