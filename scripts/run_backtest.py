"""
Run rolling backtest on DoughFlow models.

Validates model performance across multiple time periods
to ensure R² is consistent and not an artifact of the test window.
"""

import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from doughflow.models.rolling_backtest import RollingBacktest
from doughflow.models.random_forest import RandomForestModel
from doughflow.features.train_test_features import prepare_features_and_target

# Try to import LightGBM model (may not exist yet)
try:
    from doughflow.models.lightgbm_model import LightGBMModel
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


def run_backtest_for_model(df, model_class, model_name, model_kwargs=None):
    """Run backtest for a single model class."""
    print(f"\n{'#'*70}")
    print(f"# BACKTEST: {model_name}")
    print(f"{'#'*70}")

    backtest = RollingBacktest(
        min_train_weeks=52,   # 1 year minimum training
        test_weeks=4,         # 4-week test windows
        step_weeks=4          # Non-overlapping folds
    )

    results = backtest.run(
        df=df,
        model_class=model_class,
        feature_prep_fn=prepare_features_and_target,
        model_kwargs=model_kwargs or {}
    )

    results['model'] = model_name
    return results


if __name__ == "__main__":
    # Load full processed dataset
    print("Loading data...")
    df = pd.read_csv('data/processed/train.csv')

    # Also append test data for full backtest coverage
    test_df = pd.read_csv('data/processed/test.csv')
    full_df = pd.concat([df, test_df], ignore_index=True)
    full_df['date'] = pd.to_datetime(full_df['date'])
    full_df = full_df.sort_values('date').reset_index(drop=True)

    print(f"Full dataset: {len(full_df)} rows")
    print(f"Date range: {full_df['date'].min().date()} to {full_df['date'].max().date()}")

    all_results = []

    # Backtest RandomForest
    rf_results = run_backtest_for_model(
        full_df,
        RandomForestModel,
        "RandomForest",
        model_kwargs={
            'n_estimators': 100,
            'min_samples_split': 5,
            'min_samples_leaf': 10,
            'random_state': 42
        }
    )
    all_results.append(rf_results)

    # Backtest LightGBM if available
    if HAS_LIGHTGBM:
        lgbm_results = run_backtest_for_model(
            full_df,
            LightGBMModel,
            "LightGBM",
            model_kwargs={
                'n_estimators': 300,
                'learning_rate': 0.05,
                'num_leaves': 31,
                'min_child_samples': 20,
                'random_state': 42
            }
        )
        all_results.append(lgbm_results)

    # Combine and save results
    combined_results = pd.concat(all_results, ignore_index=True)

    output_path = 'reports/backtest_results.csv'
    combined_results.to_csv(output_path, index=False)
    print(f"\n✓ All backtest results saved to {output_path}")

    # Final comparison summary
    print(f"\n{'='*70}")
    print("CROSS-MODEL BACKTEST COMPARISON")
    print(f"{'='*70}\n")

    for model_name in combined_results['model'].unique():
        model_data = combined_results[combined_results['model'] == model_name]
        print(f"{model_name}:")
        print(f"  R²:   {model_data['r2'].mean():.4f} ± {model_data['r2'].std():.4f}")
        print(f"  MAE:  {model_data['mae'].mean():.2f} ± {model_data['mae'].std():.2f}")
        print(f"  RMSE: {model_data['rmse'].mean():.2f} ± {model_data['rmse'].std():.2f}")
        print()

    print(f"{'='*70}")
