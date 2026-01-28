"""
LinearRegression Training Script.
"""
import pandas as pd
from pathlib import Path
import numpy as np
import sys

# Add src to path 
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from doughflow.models.linear_regression import LinearRegressionModel
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

# Removing outliers 
q1 = np.percentile(train_df['quantity'], 25)
q3 = np.percentile(train_df['quantity'], 75)
IQR = q3 - q1
lower = q1 - 1.5 * IQR
upper = q3 + 1.5 * IQR 
iqr_outliers = (train_df['quantity'].values < lower) | (train_df['quantity'].values > upper)

train_df = train_df[iqr_outliers == False]


# prepare features
X_train, y_train, train_dates, train_items = prepare_features_and_target(train_df)
X_test, y_test, test_dates, test_items = prepare_features_and_target(test_df)


# Create model 
model = LinearRegressionModel(
    fit_intercept= True
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

# Check features coefficients
coefficients = model.get_coefficients()
print('Top 5 features by coefficient magnitude:')
for feature, coef in list(coefficients.items())[:8]:
    print(f"  {feature}: {coef:.4f}")

print('\n')

# Visualize predictions vs actual 
plot_predictions(
    y_test,
    test_predictions,
    test_dates,
    test_items,
    save_path= 'reports/figures/lr_predictions.png'
)

plot_residuals(
    y_test,
    test_predictions,
    test_items,
    save_path= 'reports/figures/lr_residuals.png'
)

# Save model 
model.save('models/')

# This creates:
# - models/LinearRegression_model.pkl (the trained model)
# - models/LinearRegression_metadata.json (metrics, features, etc.)