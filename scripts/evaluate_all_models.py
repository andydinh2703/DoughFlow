"""
Evaluate all 3 models and create comparison report.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json

# Load saved models
import joblib
from doughflow.features.train_test_features import prepare_features_and_target

def load_and_evaluate():
    """Load models and evaluate on test set."""

    # Load data
    train = pd.read_csv('data/processed/train.csv')
    test = pd.read_csv('data/processed/test.csv')

    # Prepare features
    X_train, y_train, _, _ = prepare_features_and_target(train)
    X_test, y_test, _, _ = prepare_features_and_target(test)

    print(f"\n{'='*70}")
    print("FEATURE ENGINEERING ENHANCEMENT RESULTS")
    print(f"{'='*70}\n")

    print(f"Training set: {len(X_train)} samples, {len(X_train.columns)} features")
    print(f"Test set: {len(X_test)} samples\n")

    print("Features used:")
    print(X_train.columns.tolist())
    print()

    # Models to evaluate
    models = {
        'LinearRegression': 'models/LinearRegression_model.pkl',
        'RandomForest': 'models/RandomForest_model.pkl',
        'XGBoost': 'models/XGBoost_model.pkl',
        'LightGBM': 'models/LightGBM_model.pkl'
    }

    results = []

    for model_name, model_path in models.items():
        # Load model
        model = joblib.load(model_path)

        # Predict on test
        y_pred = model.predict(X_test)

        # Calculate metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100

        results.append({
            'Model': model_name,
            'MAE': round(mae, 2),
            'RMSE': round(rmse, 2),
            'MAPE': round(mape, 2),
            'R²': round(r2, 4)
        })

    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('R²', ascending=False)

    print(f"\n{'='*70}")
    print("MODEL PERFORMANCE COMPARISON (Test Set)")
    print(f"{'='*70}\n")
    print(results_df.to_string(index=False))
    print()

    # Check if we hit target
    best_r2 = results_df['R²'].max()
    target_r2 = 0.70

    print(f"\n{'='*70}")
    print("TARGET ACHIEVEMENT")
    print(f"{'='*70}\n")
    print(f"Target R²: {target_r2}")
    print(f"Best R²:   {best_r2}")

    if best_r2 >= target_r2:
        print(f"\n✅ TARGET ACHIEVED! (+{(best_r2 - target_r2):.4f} above target)")
    else:
        gap = target_r2 - best_r2
        print(f"\n⚠️  Target not yet reached. Gap: {gap:.4f}")
        print(f"   Need {gap:.4f} more R² improvement")

        print(f"\n   Recommendations:")
        print(f"   1. Try hyperparameter tuning (GridSearch/Optuna)")
        print(f"   2. Add more domain-specific features (holidays, events)")
        print(f"   3. Try ensemble methods (averaging predictions)")
        print(f"   4. Feature selection to remove noise")

    print(f"\n{'='*70}\n")

    # Save results
    results_df.to_csv('reports/model_comparison_enhanced.csv', index=False)
    print("✓ Results saved to: reports/model_comparison_enhanced.csv\n")

if __name__ == "__main__":
    load_and_evaluate()
