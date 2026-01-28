"""
RandomForestRegressor Training Script
"""

import pandas as pd
from pathlib import Path
import sys

# Add src to path 
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from doughflow.models.random_forest import RandomForestModel
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

# prepare features
X_train, y_train, train_dates, train_items = prepare_features_and_target(train_df)
X_test, y_test, test_dates, test_items = prepare_features_and_target(test_df)


# Create model 
model = RandomForestModel(
    n_estimators=100,
    max_depth= None, 
    min_samples_split= 5,
    min_samples_leaf= 10,
    random_state= 42
)

# Train
model.fit(X_train, y_train)

# Make Predictions
test_predictions = model.predict(X_test)

# Calculate metrics 
test_metrics = calculate_metrics(y_test.values, test_predictions)
test_metrics_by_item = evaluate_by_item(y_test, test_predictions, test_items)

# print results 
print_evaluation_summary(test_metrics, test_metrics_by_item)


#check if model is learning
train_predictions = model.predict(X_train)
train_metrics = calculate_metrics(y_train.values, train_predictions)

print("\nLearning Curve Analysis:")
print(f"Train MAE: {train_metrics['mae']}")
print(f"Test MAE:  {test_metrics['mae']}")
print(f"Train R²:  {train_metrics['r2']}")
print(f"Test R²:   {test_metrics['r2']}\n")

# Check features importance
importances = model.get_feature_importances()
print('Top 5 features:')
for feature, importance in list(importances.items())[:5]:
    print(f"  {feature}: {importance:.4f}")

# Visualize predictions vs actual 
plot_predictions(
    y_test,
    test_predictions,
    test_dates,
    test_items,
    save_path= 'reports/figures/rf_predictions.png'
)

plot_residuals(
    y_test,
    test_predictions,
    test_items,
    save_path= 'reports/figures/rf_residuals.png'
)

# Save model 
model.save('models/')

# This creates:
# - models/RandomForest_model.pkl (the trained model)
# - models/RandomForest_metadata.json (metrics, features, etc.)