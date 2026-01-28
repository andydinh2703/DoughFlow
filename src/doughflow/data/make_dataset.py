"""
Data loading and processing functions for DoughFlow

This module handles reading raw bakery sales data and combining all
years into a single dataset.
"""

import pandas as pd 
from pathlib import Path
from typing import Optional

def load_raw_data(data_dir: str = "data/raw") -> pd.DataFrame:
    """
    Load and combine all raw CSV files from the data directory.

    Parameters
    ----------
    data_dir: str
        Path to directory containing all the CSV files
        Default is "data/raw".

    Returns
    -----------
    pd.DataFrame
        Combined dataframe with all years of sales data.
    
    Raises
    -------
    FileNotFoundError
        If no CSV files are found in the directory

    Example
    -------
    >>>> df = load_raw_data()
    >>>> df.shape
    (1825,5)
    """

    data_path = Path(data_dir)
    csv_files = list(data_path.glob("bhb_*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    
    dataframes = []
    used_columns = ['Date', 'Time', 'Category', 'Item', 'Qty', 'Price Point Name']
    for file in csv_files: 
        df = pd.read_csv(file, header=0, usecols=used_columns)
        dataframes.append(df)

    combined = pd.concat(dataframes)
    # sort by date and reset index
    combined = combined.sort_values(by='Date').reset_index(drop=True)

    # only get 0 or positive number

    combined = combined[combined['Qty'] >= 0]

    return combined

# rename columns function 
def rename_data(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    new_data = data.rename(columns={
        'Date' : 'date',
        'Time' : 'time',
        'Category' : 'category',
        'Item' : 'item',
        'Qty' : 'quantity',
        'Price Point Name' : 'type_of_item'
    })

    return new_data

def aggregate_daily_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate transaction-level data to daily level.

 
    This function converts transaction-level sales data (multiple transactions
    per day per item) into daily aggregates suitable for time series forecasting.

    Return:
    --------
    A data frame contains
        - Date
        - Item
        _ Type of item
        - Quantity sold
    
    """

    df = df.copy()

    df = rename_data(df)

    df = df.groupby(['date','item','type_of_item'], as_index=False)['quantity'].agg('sum')

    return df

def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Generate a summary of the loaded dataset

    Parameters
    ----------
    df: pd.DataFrame
        The loaded bakery sales data.

    Returns
    -------
    dict
        Dictionary containing row count, date range, and columns.
    """
    return {
        "rows": len(df),
        "columns": list(df.columns)
    }

def filter_operational_days(df, start_date = '2021-01-01', days_of_week = [3,4,5,6]):
    """
    Filter data to only include operational days (Thu-Sun) from stable period.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'date' column
    start_date : str
        Date when schedule stabilized (default: 2021-01-01)
    days_of_week : list
        Days to include (0=Mon, 3=Thu, 4=Fri, 5=Sat, 6=Sun)
    
    Returns:
    --------
    pd.DataFrame : Filtered data
    """

    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    # Filter by date range 
    df_filtered = df[df['date'] >= pd.to_datetime(start_date)]

    # Filter by day of week 
    df_filtered['day_of_week'] = df_filtered['date'].dt.dayofweek 
    df_filtered= df_filtered[df_filtered['day_of_week'].isin(days_of_week)]

    df_filtered =  df_filtered.drop(columns = ['day_of_week'])

    df_filtered = df_filtered.reset_index(drop=True)

    return df_filtered

def categorize_missing_days(missing_dates, holidays_calendar):
    """
    Categorize missing operational days.
    
    Parameters:
    -----------
    missing_dates : list or set of datetime
        Dates that are missing from dataset
    holidays_calendar : holidays.US() object
        US holidays calendar
    
    Returns:
    --------
    dict with keys:
        - 'holidays': List of dates that are US holidays
        - 'unexpected': List of dates that are NOT holidays
    """
    categorized = {
        'holidays': [],
        'unexpected': []
    }
    
    for date in missing_dates:
        if date in holidays_calendar:
            categorized['holidays'].append(date)
        else:
            categorized['unexpected'].append(date)
    
    return categorized


# Creating complete date range of the whole data set
def create_complete_date_range(df, start_date = '2021-01-01', days_of_week = [3,4,5,6]):
    """
    Create complete date range for operational days with both items.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Existing sales data
    start_date : str
        Starting date for the range
    days_of_week : list
        Operational days (3=Thu, 4=Fri, 5=Sat, 6=Sun)
    
    Returns:
    --------
    pd.DataFrame with all dates (including missing ones) for both items
    """

    # Create date range = 
    date_range = pd.date_range(
        start= start_date,
        end= df['date'].max(),
        freq= 'D'
    )

    # Filter operational days only
    operational_dates = date_range[date_range.dayofweek.isin(days_of_week)]

    # Create complete skeleton with both items
    items = ['Danish', 'Croissant']
    complete_df = pd.DataFrame([
        {'date': date, 'item': item}
        for date in operational_dates
        for item in items
    ])

    # Merge with actual data 
    merged = complete_df.merge(
        df[['date', 'item', 'quantity']],
        on= ['date', 'item'],
        how= 'left'
    )

    return merged

import holidays
# Filling missing values for days closed on holidays
def fill_holidays_missing_values(df, holidays_calendar = None):
    """
    Fill missing quantity values based on business logic.
    
    Strategy:
    - If date is a US holiday → Fill with 0 (bakery closed)
    - Otherwise → Leave as NaN (let lag feature fallback handle it)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Complete date range with NaN for missing values
    holidays_calendar : holidays.US() object, optional
        If None, creates US holidays calendar
    
    Returns:
    --------
    pd.DataFrame with filled values
    """

    if holidays_calendar is None: 
        holidays_calendar = holidays.US()

    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    # mask for holiday 
    is_holiday = df['date'].apply(lambda x: x in holidays_calendar)

    # Fill holidays with 0 
    df.loc[is_holiday & df['quantity'].isna(), 'quantity'] = 0

    # Log what we have filled
    holidays_filled = is_holiday & (df['quantity'] == 0)
    print(f"Filled {holidays_filled.sum()} holiday entries with 0")
    print(f"Remaining NaN values : {df['quantity'].isna().sum()}")

    
    return df

def filling_non_holiday_missing_values(df):
    """
    Fill non-holiday missing values with item-day averages.
    
    Logic: If Danish on Thursday is missing, fill with 
           average Danish sales on all other Thursdays.
    """

    df = df.copy()

    # Count initial missing values
    initial_missing = df['quantity'].isna().sum()
    
    # Strategy 1: Fill with lag_1w_ago
    if 'lag_1w_ago' in df.columns:
        df['quantity'] = df['quantity'].fillna(df['lag_1w_ago'])
    
    # Strategy 2: Fill with rolling_mean_7d
    if 'rolling_mean_7d' in df.columns:
        df['quantity'] = df['quantity'].fillna(df['rolling_mean_7d'])
    
    # Strategy 3: Fill remaining with 0
    df['quantity'] = df['quantity'].fillna(0)
    
    # Log what was filled
    final_missing = df['quantity'].isna().sum()
    filled_count = initial_missing - final_missing
    
    print(f"Filled {filled_count} non-holiday entries using cascading strategy")
    print(f"Remaining NaN values: {final_missing}")

    return df