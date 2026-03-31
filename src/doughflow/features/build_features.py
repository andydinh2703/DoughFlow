"""
Feature engineering for bakery sales forecasting.

This module transforms raw date information into features suitable for
machine learning models
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extract multiple temporal features from the date column.

    Features created:
    - day_of_week: 0 (Monday) to 6 (Sunday)
    - month
    - year
    - week of year: 1,2,3,... 52
    - day of month: 1 to 31
    - is_weekend: yes | no

    Optional cyclical encoding:
    - day_of_week_sin, day_of_week_cos
    - month_sin, month_cos
    - week_of_year_sin, week_of_year_cos
    """

    def __init__(self, add_cyclical_encoding=False):
        """
        Parameters
        ----------
        add_cyclical_encoding : bool
            If True, add sin/cos pairs for cyclical features (day_of_week, month, week_of_year).
            This helps models capture circular relationships (e.g., Monday is close to Sunday).
        """
        self.add_cyclical_encoding = add_cyclical_encoding

    def fit(self, X, y = None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()

        # converting to datetime
        X_copy['date'] = pd.to_datetime(X_copy['date'])

        # extracting features
        X_copy['day_of_week'] = X_copy['date'].dt.dayofweek
        X_copy['day_of_month'] = X_copy['date'].dt.day
        X_copy['month'] = X_copy['date'].dt.month
        X_copy['year'] = X_copy['date'].dt.year
        X_copy['week_of_year'] = X_copy['date'].dt.isocalendar().week
        X_copy['is_weekend'] = (X_copy['day_of_week'].isin([5,6])).astype(int)

        # Add cyclical encoding if requested
        if self.add_cyclical_encoding:
            X_copy = self._create_cyclical_features(X_copy)

        return X_copy

    def _create_cyclical_features(self, X):
        """
        Create sin/cos pairs for cyclical temporal features.

        This preserves circular relationships: Monday (0) and Sunday (6) are
        actually adjacent in the weekly cycle, but numerically far apart.
        Sin/cos encoding fixes this by mapping them to points on a circle.
        """
        # Day of week (period 7)
        X['day_of_week_sin'] = np.sin(2 * np.pi * X['day_of_week'] / 7)
        X['day_of_week_cos'] = np.cos(2 * np.pi * X['day_of_week'] / 7)

        # Month (period 12)
        X['month_sin'] = np.sin(2 * np.pi * X['month'] / 12)
        X['month_cos'] = np.cos(2 * np.pi * X['month'] / 12)

        # Week of year (period 52)
        X['week_of_year_sin'] = np.sin(2 * np.pi * X['week_of_year'] / 52)
        X['week_of_year_cos'] = np.cos(2 * np.pi * X['week_of_year'] / 52)

        return X
    

class ProductFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extract specific products: Danish and Croissant.

    Parameters: 
    - aggregation_level = 'item' (default) or 'item_level'

    If include type level: 
    --------------------

    Type of each item: 
    ------------------
    Danish: Apple, Cheese, Mixed, Lemon, Snail
    Croissant: Almond, Chocolate, Plain, Savory. Ham And Gruyere Cheese, Spinach Ricotta
    """

    def __init__(self, aggregation_level = 'item'):

            self.danish_types = ['Apple','Cheese','Mixed','Lemon','Snail']

            self.croissant_types = ['Almond', 'Chocolate', 'Plain', 'Savory. Ham And Gruyere Cheese', 'Spinach Ricotta']

            if aggregation_level not in ['item','item_type']:
                raise ValueError("Aggregation level must be 'item' or 'item_type'.")
            
            self.aggregation_level = aggregation_level

    def fit(self, X, y = None):
        return self
    
    def transform(self, X): 
        X = X.copy()

        danish_mask = (X['item'] == 'Danish') & (X['type_of_item'].isin(self.danish_types))
        croissant_mask = (X['item'] == 'Croissant') & (X['type_of_item'].isin(self.croissant_types))
        X = X[danish_mask | croissant_mask]
        
        # aggregating at item level 
        if self.aggregation_level == 'item': 
            X = X.groupby(['date', 'item'], as_index =False)['quantity'].agg('sum')

        # sort columns for each level 
        sorted_cols = ['date', 'item'] if self.aggregation_level == 'item' else ['date', 'item', 'type_of_item']

        # sort X 
        X = X.sort_values(by = sorted_cols).reset_index(drop = True)

        return X



class LagFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Create lag features for time series forecasting.
    
    Features created for each item-type combination:
    
    1. Recent trend (last 4 weeks same-day):
       - lag_1w_ago: Last week Thursday
       - lag_2w_ago: 2 weeks ago Thursday  
       - lag_3w_ago: 3 weeks ago Thursday
       - lag_4w_ago: 4 weeks ago Thursday
    
    2. Year-over-year same-week patterns:
       - same_week_1yr_ago: Week 22 last year
       - same_week_2yr_ago: Week 22 two years ago
    
    3. Rolling stats
        - rolling_mean_operational: mean of last business days
        - rolling_std_operational: standard deviation of last business days

    """

    def __init__(self, include_recent_lags=True,
                 include_yearly_lags=True,
                 include_rolling_stats= True,
                 recent_lag_weeks = [1,2,3,4],
                 yearly_lags = [1,2],
                 fill_strategy ='cascading'):
        
        """
        Parameters
        ----------
        include_recent_lags : bool
            Include lag_1w_ago, lag_2w_ago, etc.
        include_yearly_lags : bool
            Include year-over-year same-week features
        include_rolling_stats : bool
            Include rolling mean/std features
        recent_lag_days : list
            Which recent lags to create (in days)
        yearly_lags : list
            How many years back for yearly features
        fill_strategy: 'cascading', 'mean', 'zero'
        """

        self.include_recent_lags = include_recent_lags
        self.include_yearly_lags = include_yearly_lags
        self.include_rolling_stats = include_rolling_stats
        self.recent_lag_weeks = recent_lag_weeks
        self.yearly_lags = yearly_lags
        self.fill_strategy = fill_strategy

    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        X = X.copy()

        # ensure that date is datetime type 
        X['date'] = pd.to_datetime(X['date'])

        # if item_type level 
        if 'type_of_item' in X.columns: 
            # Sort by item, type, and date for proper lag calculation
            X = X.sort_values(['item', 'type_of_item', 'date']).reset_index(drop=True)
            
            # Create a unique key for each item-type combination
            X['item_key'] = X['item'] + '_' + X['type_of_item']
        else: 
            # Sort by item and date 
            X = X.sort_values(['item', 'date']).reset_index(drop=True)

            X['item_key'] = X['item']

        

        # 1. Recent lags (day-based)
        if self.include_recent_lags:
            for lag_weeks in self.recent_lag_weeks:
                lag_days = lag_weeks * 7
                
                # Create a lagged version of the data
                lagged = X[['item_key', 'date', 'quantity']].copy()
                lagged['date'] = lagged['date'] + pd.Timedelta(days=lag_days)
                lagged = lagged.rename(columns={'quantity': f'lag_{lag_weeks}w_ago'})
                
                # Merge back to get the lag feature
                X = X.merge(
                    lagged[['item_key', 'date', f'lag_{lag_weeks}w_ago']],
                    on=['item_key', 'date'],
                    how='left'
                )
        
        # 2. Year-over-year same-week lags (week-based)
        if self.include_yearly_lags:
            X = self.create_yearly_lags(X)

        # 3. Rolling statistics: 
        if self.include_rolling_stats:
            # Rolling mean over last 7 days
            X['rolling_mean_operational'] = X.groupby('item_key')['quantity'].transform(lambda x: x.shift(1).rolling(window=4, min_periods=1).mean())

            # Rolling std dev over last 7 days
            X['rolling_std_operational'] = X.groupby('item_key')['quantity'].transform(lambda x: x.shift(1).rolling(window=4, min_periods=1).std())

        # 4. Fill missing values based on strategy 
        if self.fill_strategy is not None: 
            X = self.fill_missing_lags(X) # fill out missing values in lags features given the strategy
        
        # Clean up temporary column 
        X = X.drop(columns = ['item_key'])

        return X
    
    def create_yearly_lags(self,X):
        # ensure having required columns
        required_cols = ['day_of_week', 'week_of_year', 'year']

        if not all(col in X.columns for col in required_cols):
            raise ValueError(f'Missing required columns: {required_cols}')
        
        # creat a lookup key for matching 
        X['match_key'] = X['item_key'] + '_' + X['day_of_week'].astype(str) + '_' + X['week_of_year'].astype(str)


        # For each year lag, do a self merge 
        for years_back in self.yearly_lags:
            # Create a temporary df for N years ago 
            historical = X[['match_key','year', 'quantity', 'date']].copy()

            historical['year'] = historical['year'] + years_back

            historical = historical.rename(columns = {
                'quantity': f'lag_{years_back}y_same_week'
            })

            # merge back to main dataframe
            X = X.merge(
                historical[['match_key', 'year', f'lag_{years_back}y_same_week']],
                on = ['match_key', 'year'],
                how = 'left'
            )

        # Clean up temporary column
        X = X.drop(columns = ['match_key'])

        return X
    

    def fill_missing_lags(self, X): 
        """
        Fill missing lag values based on strategy.

        Parameter:
        ---------
        X: DataFrame
            Data with lags features that might contains NaN values

        Returns
        -------
        DataFrame 
            Data with filled lag values
        """

        X = X.copy()

        # identify lag columns 
        lag_cols = [col for col in X.columns if col.startswith('lag_')]

        if not lag_cols:
            return X
        
        # cascading
        if self.fill_strategy == 'cascading':
            # sort all lags columns by time proximity (1w, 2w,... then yearly)
            recent_lag_cols = sorted([col for col in lag_cols if 'w_ago' in col],
                                     key=lambda x: int(x.split('_')[1].replace('w','')))
            
            yearly_lag_cols = sorted([col for col in lag_cols if 'y_' in col],
                                     key=lambda x: int(x.split('_')[1].replace('y','')))
            
            # For each lag column, fill with next available lag
            for i, col in enumerate(recent_lag_cols):
                if X[col].isna().any():
                    # filling with subsequent lags
                    for fallback_col in recent_lag_cols[i+1:]:
                        X[col] = X[col].fillna(X[fallback_col])
                    
                    # if still NaN,  rolling mean
                    if 'rolling_mean_operational' in X.columns:
                        X[col] = X[col].fillna(X['rolling_mean_operational'])

                    # Final fallbacl: item-level mean 
                    if X[col].isna().any():
                        item_means = X.groupby('item_key')['quantity'].transform('mean')

                        X[col] = X[col].fillna(item_means)

            # For yearly lags, fill with item mean 
            for col in yearly_lag_cols:
                if X[col].isna().any():
                    item_means = X.groupby('item_key')['quantity'].transform('mean')
                    X[col] = X[col].fillna(item_means)

        elif self.fill_strategy == 'mean':
            # Fill each lag column with item-level mean
            for col in lag_cols: 
                if X[col].isna().any(): 
                    item_means = X.groupby('item_key')['quantity'].transform('mean')
                    X[col] = X[col].fillna(item_means)


        elif self.fill_strategy == 'zero':
            # Fill na with 0
            X[lag_cols] = X[lag_cols].fillna(0)

        # Fill any remaining NaN with 0 
        if 'rolling_std_operational' in X.columns: 
            X['rolling_std_operational'] = X['rolling_std_operational'].fillna(0)
            X['rolling_std_operational'] = X['rolling_std_operational'].round(2)

        if 'rolling_mean_operational' in X.columns: 
            X['rolling_mean_operational'] = X['rolling_mean_operational'].fillna(0)
            X['rolling_mean_operational'] = X['rolling_mean_operational'].round(2)

        X[lag_cols] = X[lag_cols].round(2)



        return X


class ExtendedRollingWindowExtractor(BaseEstimator, TransformerMixin):
    """
    Create additional rolling window features beyond the standard 7-day window.

    This captures longer-term trends that the basic rolling statistics might miss.
    Different window sizes capture different temporal scales:
    - 7-day: Recent weekly pattern
    - 14-day: Two-week trend
    - 30-day: Monthly trend
    """

    def __init__(self, window_sizes=[14, 30], statistics=['mean', 'std']):
        """
        Parameters
        ----------
        window_sizes : list
            Rolling window sizes in days (default: [14, 30])
        statistics : list
            Statistics to compute: 'mean', 'std', 'min', 'max' (default: ['mean', 'std'])
        """
        self.window_sizes = window_sizes
        self.statistics = statistics

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Ensure date is datetime and sorted
        X['date'] = pd.to_datetime(X['date'])

        # Create item_key for grouping
        if 'type_of_item' in X.columns:
            X['item_key'] = X['item'] + '_' + X['type_of_item']
        else:
            X['item_key'] = X['item']

        # Sort by item and date
        X = X.sort_values(['item_key', 'date']).reset_index(drop=True)

        # Create rolling features for each window size
        for window in self.window_sizes:
            for stat in self.statistics:
                col_name = f'rolling_{stat}_{window}d'

                if stat == 'mean':
                    X[col_name] = X.groupby('item_key')['quantity'].transform(
                        lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
                    )
                elif stat == 'std':
                    X[col_name] = X.groupby('item_key')['quantity'].transform(
                        lambda x: x.shift(1).rolling(window=window, min_periods=1).std()
                    )
                elif stat == 'min':
                    X[col_name] = X.groupby('item_key')['quantity'].transform(
                        lambda x: x.shift(1).rolling(window=window, min_periods=1).min()
                    )
                elif stat == 'max':
                    X[col_name] = X.groupby('item_key')['quantity'].transform(
                        lambda x: x.shift(1).rolling(window=window, min_periods=1).max()
                    )

                # Fill NaN with 0 and round
                X[col_name] = X[col_name].fillna(0).round(2)

        # Clean up temporary column
        X = X.drop(columns=['item_key'])

        return X


class InteractionFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Create interaction features between temporal and lag features.

    Interactions capture multiplicative effects that simple features miss.
    Example: Saturday in summer may have different patterns than Saturday in winter.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Ensure required features exist
        required = ['day_of_week', 'month', 'is_weekend']
        missing = [col for col in required if col not in X.columns]
        if missing:
            raise ValueError(f"Missing required features: {missing}")

        # Day-of-week × Season interactions
        # Saturday in summer (peak season)
        X['saturday_summer'] = ((X['day_of_week'] == 5) &
                                 (X['month'].isin([6, 7, 8]))).astype(int)

        # Weekend in summer (high demand period)
        X['weekend_summer'] = (X['is_weekend'] &
                               (X['month'].isin([6, 7, 8]))).astype(int)

        # Weekend in fall (still strong)
        X['weekend_fall'] = (X['is_weekend'] &
                             (X['month'].isin([9, 10, 11]))).astype(int)

        # Lag × temporal interactions (if lag features exist)
        if 'lag_1w_ago' in X.columns:
            # Lag × day of week (captures day-specific trends)
            X['lag_dow_interaction'] = X['lag_1w_ago'] * X['day_of_week']

            # Lag × is_weekend (different weekend behavior)
            X['lag_weekend_interaction'] = X['lag_1w_ago'] * X['is_weekend']

        return X
