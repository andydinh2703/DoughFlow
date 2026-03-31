"""
Rolling window backtest for time series model validation.

Implements expanding/rolling window cross-validation that respects
temporal ordering and prevents data leakage. Reports per-fold,
per-item, and per-day-of-week metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Callable, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class RollingBacktest:
    """
    Rolling window backtest for time series forecasting models.
    
    Splits data into chronologically ordered folds:
    
    Fold 1: [=========TRAIN=========][TEST]
    Fold 2: [===========TRAIN===========][TEST]
    Fold 3: [=============TRAIN=============][TEST]
    ...
    
    Each fold trains on all data before the test period (expanding window).
    
    Parameters:
    -----------
    min_train_weeks : int
        Minimum number of weeks of training data for the first fold.
        Default: 52 (1 year of data to capture seasonality)
    
    test_weeks : int
        Number of weeks per test fold.
        Default: 4 (1 month of test data per fold)
    
    step_weeks : int
        Number of weeks to slide forward between folds.
        Default: 4 (non-overlapping test folds)
    
    operational_days : list
        Days of week the bakery operates (3=Thu, 4=Fri, 5=Sat, 6=Sun).
        Used to calculate weeks correctly.
    """

    def __init__(
        self,
        min_train_weeks: int = 52,
        test_weeks: int = 4,
        step_weeks: int = 4,
        operational_days: list = None
    ):
        self.min_train_weeks = min_train_weeks
        self.test_weeks = test_weeks
        self.step_weeks = step_weeks
        self.operational_days = operational_days or [3, 4, 5, 6]
        self.results_ = []

    def _get_fold_boundaries(self, df: pd.DataFrame, date_column: str = 'date') -> List[dict]:
        """
        Calculate train/test boundaries for each fold.
        
        Returns:
        --------
        List of dicts with 'train_end' and 'test_end' dates
        """
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])

        min_date = df[date_column].min()
        max_date = df[date_column].max()

        # Calculate minimum training end date
        min_train_end = min_date + pd.Timedelta(weeks=self.min_train_weeks)
        test_duration = pd.Timedelta(weeks=self.test_weeks)
        step_duration = pd.Timedelta(weeks=self.step_weeks)

        folds = []
        train_end = min_train_end

        while train_end + test_duration <= max_date:
            test_end = train_end + test_duration
            folds.append({
                'train_end': train_end,
                'test_start': train_end + pd.Timedelta(days=1),
                'test_end': test_end
            })
            train_end += step_duration

        return folds

    def run(
        self,
        df: pd.DataFrame,
        model_class,
        feature_prep_fn: Callable,
        model_kwargs: dict = None,
        date_column: str = 'date',
        item_column: str = 'item'
    ) -> pd.DataFrame:
        """
        Run rolling backtest with the specified model.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Full dataset with features already engineered
        model_class : class
            Model class to instantiate (must follow BaseModel interface)
        feature_prep_fn : callable
            Function that takes a DataFrame and returns (X, y, dates, items)
        model_kwargs : dict, optional
            Keyword arguments to pass to model constructor
        date_column : str
            Name of date column
        item_column : str
            Name of item column
        
        Returns:
        --------
        pd.DataFrame with per-fold metrics
        """
        model_kwargs = model_kwargs or {}
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])

        folds = self._get_fold_boundaries(df, date_column)

        if len(folds) == 0:
            raise ValueError(
                f"Not enough data for backtest. Need at least "
                f"{self.min_train_weeks + self.test_weeks} weeks of data."
            )

        print(f"\n{'='*70}")
        print(f"ROLLING BACKTEST ({len(folds)} folds)")
        print(f"{'='*70}")
        print(f"Min train: {self.min_train_weeks} weeks | "
              f"Test: {self.test_weeks} weeks | "
              f"Step: {self.step_weeks} weeks\n")

        self.results_ = []

        for i, fold in enumerate(folds):
            # Split data
            train_data = df[df[date_column] <= fold['train_end']].copy()
            test_data = df[
                (df[date_column] > fold['train_end']) &
                (df[date_column] <= fold['test_end'])
            ].copy()

            if len(test_data) == 0:
                continue

            # Prepare features
            X_train, y_train, _, _ = feature_prep_fn(train_data)
            X_test, y_test, test_dates, test_items = feature_prep_fn(test_data)

            # Train and predict
            model = model_class(**model_kwargs)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Calculate overall metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100

            fold_result = {
                'fold': i + 1,
                'train_end': fold['train_end'].date(),
                'test_start': fold['test_start'].date(),
                'test_end': fold['test_end'].date(),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'mae': round(mae, 2),
                'rmse': round(rmse, 2),
                'mape': round(mape, 2),
                'r2': round(r2, 4)
            }

            # Per-item metrics
            if item_column in test_data.columns:
                for item_name in test_data[item_column].unique():
                    item_mask = test_items == item_name
                    if item_mask.sum() > 0:
                        item_r2 = r2_score(y_test[item_mask], y_pred[item_mask])
                        item_mae = mean_absolute_error(y_test[item_mask], y_pred[item_mask])
                        fold_result[f'r2_{item_name}'] = round(item_r2, 4)
                        fold_result[f'mae_{item_name}'] = round(item_mae, 2)

            # Per-day-of-week metrics
            test_data_with_dow = test_data.copy()
            test_data_with_dow['_pred'] = y_pred
            dow_map = {3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}

            if 'day_of_week' in test_data_with_dow.columns:
                for dow, dow_name in dow_map.items():
                    dow_mask = test_data_with_dow['day_of_week'] == dow
                    if dow_mask.sum() > 0:
                        dow_y = y_test[dow_mask.values]
                        dow_pred = y_pred[dow_mask.values]
                        if len(dow_y) > 1:
                            fold_result[f'r2_{dow_name}'] = round(
                                r2_score(dow_y, dow_pred), 4
                            )

            self.results_.append(fold_result)

            # Print fold summary
            print(f"  Fold {i+1}: "
                  f"train→{fold['train_end'].date()} | "
                  f"test {fold['test_start'].date()}→{fold['test_end'].date()} | "
                  f"R²={r2:.4f} MAE={mae:.2f}")

        results_df = pd.DataFrame(self.results_)

        # Print aggregate summary
        self._print_summary(results_df)

        return results_df

    def _print_summary(self, results_df: pd.DataFrame):
        """Print aggregate backtest summary."""
        print(f"\n{'='*70}")
        print("BACKTEST SUMMARY")
        print(f"{'='*70}\n")

        print(f"Overall R²:  {results_df['r2'].mean():.4f} ± {results_df['r2'].std():.4f}")
        print(f"Overall MAE: {results_df['mae'].mean():.2f} ± {results_df['mae'].std():.2f}")
        print(f"Overall RMSE: {results_df['rmse'].mean():.2f} ± {results_df['rmse'].std():.2f}")
        print(f"Overall MAPE: {results_df['mape'].mean():.2f}% ± {results_df['mape'].std():.2f}%")

        print(f"\nBest fold:  Fold {results_df.loc[results_df['r2'].idxmax(), 'fold']} "
              f"(R²={results_df['r2'].max():.4f})")
        print(f"Worst fold: Fold {results_df.loc[results_df['r2'].idxmin(), 'fold']} "
              f"(R²={results_df['r2'].min():.4f})")

        # Per-item summary
        item_cols = [c for c in results_df.columns if c.startswith('r2_') and c not in
                     ['r2_Thu', 'r2_Fri', 'r2_Sat', 'r2_Sun']]
        if item_cols:
            print(f"\nPer-Item R² (mean across folds):")
            for col in item_cols:
                item_name = col.replace('r2_', '')
                mean_r2 = results_df[col].mean()
                std_r2 = results_df[col].std()
                print(f"  {item_name}: {mean_r2:.4f} ± {std_r2:.4f}")

        # Per-day summary
        dow_cols = [c for c in results_df.columns if c in ['r2_Thu', 'r2_Fri', 'r2_Sat', 'r2_Sun']]
        if dow_cols:
            print(f"\nPer-Day R² (mean across folds):")
            for col in sorted(dow_cols):
                day_name = col.replace('r2_', '')
                valid = results_df[col].dropna()
                if len(valid) > 0:
                    print(f"  {day_name}: {valid.mean():.4f} ± {valid.std():.4f}")

        print(f"\n{'='*70}")
