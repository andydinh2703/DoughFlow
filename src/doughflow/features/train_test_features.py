"""
Train & Test Features Builder
"""

import pandas as pd

def prepare_features_and_target(df):
    # columns to exclude (not features)
    exclude_cols = ['date', 'quantity']

    df_encoded = df.copy()

    # encode item: 0 for Danish, 1 for Croissant
    df_encoded['item_encoded'] = (df_encoded['item'] == 'Croissant').astype(int)

    # exclude item column
    exclude_cols.append('item')

    feature_cols = [col for col in df_encoded.columns if col not in exclude_cols]

    X = df_encoded[feature_cols]
    y = df_encoded['quantity']

    dates = pd.to_datetime(df_encoded['date'])
    items = df_encoded['item']

    return X, y, dates, items

