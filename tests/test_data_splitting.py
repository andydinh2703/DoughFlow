import pandas as pd 
from doughflow.data.split_data import (
    fixed_holdout_split,
    validate_split,
    split_summary,
    save_split
)

# Load data 
df = pd.read_csv('data/processed/danish_croissant_data.csv')

print("Testing Fixed Holdout Split...\n")

# Split the data
train, test = fixed_holdout_split(df, split_date= '2024-12-31')

# Show summary 
split_summary(train, test)

# Validate the split 
print("\nValidation Results:")
validation =validate_split(train, test)

if validation['valid']:
    print("✓ Split is VALID - no data leakage detected!")
else:
    print("✗ Split has ERRORS:")
    for error in validation['errors']:
        print(f"  - {error}")

if validation['warning']:
    print("\nWarnings:")
    for warning in validation['warning']:
        print(f"  - {warning}")

# Save the split
print("\nSaving split to disk...")
metadata = {
    'strategy': 'fixed_holdout',
    'split_date': '2024-12-31',
    'total_rows': len(df),
    'train_rows': len(train),
    'test_rows': len(test)
}
save_split(train, test, metadata=metadata)

print("\n✓ You now have working train/test splitting.")