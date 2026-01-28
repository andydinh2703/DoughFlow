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



# Run validation 

# Load & filter operational days
from doughflow.data.make_dataset import filter_operational_days,create_complete_date_range, fill_holidays_missing_values, filling_non_holiday_missing_values
                        
from doughflow.features.build_features import TemporalFeatureExtractor, LagFeatureExtractor
import holidays 
import pandas as pd

df = pd.read_csv('data/processed/danish_croissant_data.csv')


operating_data = filter_operational_days(df)

# Create complete date range 
complete_data = create_complete_date_range(operating_data)

# Fill missing values
us_holidays = holidays.US()
data_with_filled_holidays = fill_holidays_missing_values(complete_data, us_holidays)


# Apply feature engineering to handle remaining missing values
temporal_extractor = TemporalFeatureExtractor()
data_with_temporal = temporal_extractor.fit_transform(data_with_filled_holidays)

# Extract lag features 
lag_extractor = LagFeatureExtractor(fill_strategy='cascading')
data_with_lags = lag_extractor.fit_transform(data_with_temporal)
final_data = filling_non_holiday_missing_values(data_with_lags)

is_ready = validate_data_before_split(final_data)

if is_ready:
    print("Data is ready for train/test split")

else: 
    print("Data is NOT good yet!")


print(final_data.head(80))