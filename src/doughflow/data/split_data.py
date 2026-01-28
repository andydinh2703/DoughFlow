"""
Train/Test spliiting functions for time series data 
"""

import pandas as pd 
from typing import Tuple, Dict, Any
from pathlib import Path
import json 

def fixed_holdout_split(
        df: pd.DataFrame,
        split_date: str,
        date_column: str = 'date',
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splitting time series data into train and test sets at a specific date.

    Args:
        df: DataFrame with time series data
        split_date: Date to split on (format: 'YYYY-MM-DD')
                    Train gets dates <= split_date
                    Test gets dates > split_date
        date_column: name of the date column 

    Returns: 
        train_df, test_df: Training and test DataFrames
    """

    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])

    # convert split_date to date time
    split_date = pd.to_datetime(split_date)

    # Split: train gets <= split_date, test gets > split_date
    train_df = df[df[date_column] <= split_date].copy()
    test_df = df[df[date_column] > split_date].copy()

    # sort by date to ensure chronological order
    train_df = train_df.sort_values(date_column).reset_index(drop=True)
    test_df = test_df.sort_values(date_column).reset_index(drop=True)

    return train_df, test_df

# Validation function 
def validate_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    date_column: str = 'date',
    item_column: str = 'item'
) -> Dict[str, Any]:
    """
    Validate that the train/test split is correct.
    
    Checks:
    1. No date overlap (no leakage)
    2. Train dates come before test dates
    3. Both items (Croissant, Danish) present in both sets
    4. No missing dates in the gap
    
    Returns:
        Dictionary with validation results
    
    """
    results = {
        'valid': True,
        'errors': [],
        'warning': []
    }

    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df[date_column] = pd.to_datetime(train_df[date_column])
    test_df[date_column] = pd.to_datetime(test_df[date_column])

    # Check 1: No overlap 
    train_max = train_df[date_column].max()
    test_min = test_df[date_column].min()

    if train_max >= test_min:
        results['valid'] = False
        results['errors'].append(
            f"Data leakage detected! Train max date ({train_max.date()}) "
            f">= Test min date ({test_min.date()})"
        )

    
    # Check 2: Chronological order 
    if test_min <= train_max:
        results['warning'].append(
            "Test period should start after train period"
        )

    # Check 3: Both items present
    train_items = set(train_df[item_column].unique())
    test_items = set(test_df[item_column].unique())

    if train_items != test_items:
        results['errors'].append(
            f"Item mismatch! Train: {train_items}, Test: {test_items}"
        )
        results['valid'] = False

    # Check 4: Gap between train and test
    gap_days = (test_min - train_max).days - 1
    if gap_days > 0:
        results['warning'].append(
            f"Gap of {gap_days} days between train and test"
        )
    
    return results

def split_summary(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    date_column: str = 'date',
    item_column: str = 'item'
) -> None:
    """
    Print a summary of the train/test split.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df[date_column] = pd.to_datetime(train_df[date_column])
    test_df[date_column] = pd.to_datetime(test_df[date_column])
    
    total_rows = len(train_df) + len(test_df)
    
    print("="*70)
    print("TRAIN/TEST SPLIT SUMMARY")
    print("="*70)
    
    print(f"\nTraining Set:")
    print(f"  Rows: {len(train_df)} ({len(train_df)/total_rows*100:.1f}%)")
    print(f"  Date range: {train_df[date_column].min().date()} to {train_df[date_column].max().date()}")
    print(f"  Duration: {(train_df[date_column].max() - train_df[date_column].min()).days} days")
    
    for item in sorted(train_df[item_column].unique()):
        count = len(train_df[train_df[item_column] == item])
        print(f"  {item}: {count} samples")
    
    print(f"\nTest Set:")
    print(f"  Rows: {len(test_df)} ({len(test_df)/total_rows*100:.1f}%)")
    print(f"  Date range: {test_df[date_column].min().date()} to {test_df[date_column].max().date()}")
    print(f"  Duration: {(test_df[date_column].max() - test_df[date_column].min()).days} days")
    
    for item in sorted(test_df[item_column].unique()):
        count = len(test_df[test_df[item_column] == item])
        print(f"  {item}: {count} samples")
    
    print(f"\nTotal: {total_rows} rows")
    print("="*70)


def save_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str = 'data/processed',
    metadata: Dict[str, Any] = None
) -> None:
    """
    Save train and test sets to CSV files with metadata.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok= True)

    train_path = output_path / 'train.csv'
    test_path = output_path / 'test.csv'
    metadata_path = output_path / 'split_metadata.json'

    # Save CSVs
    train_df.to_csv(train_path, index= False)
    test_df.to_csv(test_path, index=False)

    # Log message
    print(f"✓ Saved training set to: {train_path}")
    print(f"✓ Saved test set to: {test_path}")

    # save metadata
    if metadata:
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent= 2, default='str')
        print(f"✓ Saved metadata to: {metadata_path}")

def load_split(
    input_dir: str = 'data/processed'
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Load train and test sets from CSV files.
    """

    input_path = Path(input_dir)

    train_df = pd.read_csv(input_path/'train.csv')
    test_df = pd.read_csv(input_path/'test.csv')

    # Load metadata if exists
    metadata_path = input_path / 'split_metadata.json'
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, 'r') as f: 
            metadata = json.load(f)

    return train_df, test_df, metadata
