"""
Complete DoughFlow pipeline: Raw data → Engineered features → Train/Test split
"""

import pandas as pd
import holidays

from doughflow.data.make_dataset import (
    filter_operational_days,
    create_complete_date_range,
    fill_holidays_missing_values,
    filling_non_holiday_missing_values
)
from doughflow.features.build_features import (
    TemporalFeatureExtractor,
    LagFeatureExtractor
)
from doughflow.data.split_data import (
    fixed_holdout_split,
    validate_split,
    split_summary,
    save_split
)

def validate_data_before_split(df): 
    """
    Run checks to ensure data is ready for train/test split.
    
    Checks:
    1. No missing values in quantity or features
    2. Correct date range (no gaps in operational days)
    3. Both items present for all dates
    """

    checks = {
        'No NaN in quantity': df['quantity'].isna().sum() == 0,
        'No NaN in lag features': df.filter(like='lag_').isna().sum().sum() == 0,
        'No NaN in rolling features': df.filter(like='rolling').isna().sum().sum() == 0,
        'Both items present': len(df['item'].unique()) == 2,
        'Only operational days': set(df['day_of_week'].unique()).issubset({3,4,5,6})
    }

    print("Data Validation Results:")
    for check, passed in checks.items():
        status = "Good" if passed else "Not yet"
        print(f"{check} : {status}")

    return all(checks.values())


def run_complete_pipeline(split_date: str):
    """
    Run the complete pipeline from raw data to train/test split.
    
    Steps:
    1. Load raw data
    2. Filter operational days
    3. Create complete date range
    4. Fill holiday missing values
    5. Add temporal features
    6. Add lag features
    7. Fill non-holiday missing values
    8. Validate data
    9. Split into train/test
    10. Save results
    """

    print("=" * 70)
    print('DOUGHFLOW COMPLETE PIPELINE')
    print("=" * 70)

    # Step 1-2: Load & filter 
    print("\n Loading Data...")
    df = pd.read_csv('data/processed/danish_croissant_data.csv')
    print(f"✓ Loaded {len(df)} rows")

    # Filter operational days
    print('\n Filtering operational days...')
    operating_data = filter_operational_days(df)
    print(f"✓ Filtered to {len(operating_data)} operational day records")

    # Step 3: Completing date range for the data
    print('\n Creating complete date range...')
    complete_data = create_complete_date_range(operating_data)
    print(f"✓ Created complete range: {len(complete_data)} rows")

    # Step 4: Fill holidays
    print('\n Filling holiday missing values...')
    us_holidays = holidays.US()
    data_with_filled_holidays = fill_holidays_missing_values(complete_data, us_holidays)


    # Step 5: Temporal features
    print('\n Adding temporal features...')
    temporal_extractor = TemporalFeatureExtractor()
    data_with_temporal = temporal_extractor.fit_transform(data_with_filled_holidays)
    print(f"✓ Added temporal features")

    # Step 6: Lag features
    print("\n Adding lag features...")
    lag_extractor = LagFeatureExtractor(fill_strategy='cascading')
    data_with_lags = lag_extractor.fit_transform(data_with_temporal)
    print(f"✓ Added lag features")

    # Step 7: Fill remaining missing values
    print('\n Filling non-holiday missing values...')
    final_data = filling_non_holiday_missing_values(data_with_lags)
    print(f"✓ Final data shape: {final_data.shape}")

    # Step 8: Validate
    print('\n Validating data...')
    is_ready = validate_data_before_split(final_data)

    if not is_ready: 
        print("\n✗ Data validation FAILED. Cannot proceed with split.")
        return None, None
    
    print("\n✓ Data is ready for train/test split!")

    # Step 9: Split
    print(f"\n Splitting data at {split_date}...")
    train, test = fixed_holdout_split(final_data,split_date=split_date)

    # Show summary
    split_summary(train, test)

    # validate split 
    validation = validate_split(train, test)

    print("\n" + "=" * 70)
    print("SPLIT VALIDATION RESULTS")
    print("=" * 70)

    if validation['valid']:
        print("✓ Split is VALID - No data leakage!")
    else:
        print("✗ Split has ERRORS:")
        for error in validation['errors']:
            print(f"  - {error}")
    
    if validation['warning']:
        print("\nWarnings:")
        for warning in validation['warning']:
            print(f"  - {warning}")
    
    print("=" * 70)


    # Step 10: Save 
    print('\n Saving train/test sets...')
    metadata = {
        'strategy': 'fixed_holdout',
        'split_date': split_date,
        'total_rows': len(final_data),
        'train_rows': len(train),
        'test_rows': len(test),
        'train_date_range': [str(train['date'].min()), str(train['date'].max())],
        'test_date_range': [str(test['date'].min()), str(test['date'].max())],
        'features': list(train.columns)

    }

    save_split(train, test, metadata=metadata)

    print("\n" + "=" * 70)
    print("✓ PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\nNext steps:")
    print("1. Train your model on: data/processed/train.csv")
    print("2. Evaluate on: data/processed/test.csv")
    print("3. Check metadata: data/processed/split_metadata.json")
    

    return train, test


if __name__ == "__main__":
    train, test = run_complete_pipeline(split_date='2024-12-31')

