"""
LightGBM Training Script

Trains a LightGBM model with Huber loss for robust demand forecasting.
"""

import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from doughflow.models.lightgbm_model import LightGBMModel
from doughflow.models.evaluate import (
    calculate_metrics,
    evaluate_by_item,
    plot_predictions,
    plot_residuals,
    print_evaluation_summary
)
from doughflow.features.train_test_features import prepare_features_and_target

# Load train/test splits
train_df = pd.read_csv('data/processed/train.csv')
test_df = pd.read_csv('data/processed/test.csv')

# Prepare features
X_train, y_train, train_dates, train_items = prepare_features_and_target(train_df)
X_test, y_test, test_dates, test_items = prepare_features_and_target(test_df)

print(f"\n{'='*70}")
print("LIGHTGBM MODEL TRAINING (Huber Loss)")
print(f"{'='*70}\n")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Features: {len(X_train.columns)}\n")

# Create model with time-series-tuned defaults
model = LightGBMModel(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42
)

# Train
print("Training LightGBM...")
model.fit(X_train, y_train)
print("✓ Training complete\n")

# Make predictions
test_predictions = model.predict(X_test)

# Calculate metrics
test_metrics = calculate_metrics(y_test.values, test_predictions)
test_metrics_by_item = evaluate_by_item(y_test, test_predictions, test_items)

# Print results
print_evaluation_summary(test_metrics, test_metrics_by_item)

# Learning curve analysis
train_predictions = model.predict(X_train)
train_metrics = calculate_metrics(y_train.values, train_predictions)

print("\nLearning Curve Analysis:")
print(f"Train MAE: {train_metrics['mae']}")
print(f"Test MAE:  {test_metrics['mae']}")
print(f"Train R²:  {train_metrics['r2']}")
print(f"Test R²:   {test_metrics['r2']}\n")

# Feature importance
importances = model.get_feature_importances()
print('Top 10 features:')
for feature, importance in list(importances.items())[:10]:
    print(f"  {feature}: {importance:.4f}")

# Visualize predictions vs actual
plot_predictions(
    y_test,
    test_predictions,
    test_dates,
    test_items,
    save_path='reports/figures/lgbm_predictions.png'
)

plot_residuals(
    y_test,
    test_predictions,
    test_items,
    save_path='reports/figures/lgbm_residuals.png'
)

# Save model
model.save('models/')
print(f"\n{'='*70}")
print("✓ LightGBM model saved to models/LightGBM_model.pkl")
print("✓ Metadata saved to models/LightGBM_metadata.json")
print(f"{'='*70}\n")
