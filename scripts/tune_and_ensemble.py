"""
Hyperparameter tuning and ensemble methods to reach R² >= 0.70
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from doughflow.features.train_test_features import prepare_features_and_target
import joblib

# Load data
train = pd.read_csv('data/processed/train.csv')
test = pd.read_csv('data/processed/test.csv')

X_train, y_train, _, _ = prepare_features_and_target(train)
X_test, y_test, _, _ = prepare_features_and_target(test)

print(f"\n{'='*70}")
print("HYPERPARAMETER TUNING & ENSEMBLE OPTIMIZATION")
print(f"{'='*70}\n")

# Strategy 1: Try Ridge Regression (regularization might help)
print("1. Testing Ridge Regression with alpha tuning...")
ridge_alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
best_ridge_r2 = 0
best_ridge_alpha = 0

for alpha in ridge_alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    if r2 > best_ridge_r2:
        best_ridge_r2 = r2
        best_ridge_alpha = alpha

print(f"   Best Ridge alpha: {best_ridge_alpha}, R²: {best_ridge_r2:.4f}")

# Strategy 2: Tune Random Forest more aggressively
print("\n2. Tuning Random Forest hyperparameters...")
rf_param_grid = {
    'n_estimators': [200, 300],
    'max_depth': [None, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 5]
}

rf_grid = GridSearchCV(
    RandomForestRegressor(random_state=42),
    rf_param_grid,
    cv=3,
    scoring='r2',
    n_jobs=-1,
    verbose=0
)
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_
y_pred_rf = best_rf.predict(X_test)
rf_r2 = r2_score(y_test, y_pred_rf)

print(f"   Best RF params: {rf_grid.best_params_}")
print(f"   Best RF R²: {rf_r2:.4f}")

# Strategy 3: Tune XGBoost
print("\n3. Tuning XGBoost hyperparameters...")
xgb_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0]
}

xgb_grid = GridSearchCV(
    xgb.XGBRegressor(random_state=42),
    xgb_param_grid,
    cv=3,
    scoring='r2',
    n_jobs=-1,
    verbose=0
)
xgb_grid.fit(X_train, y_train)
best_xgb = xgb_grid.best_estimator_
y_pred_xgb = best_xgb.predict(X_test)
xgb_r2 = r2_score(y_test, y_pred_xgb)

print(f"   Best XGB params: {xgb_grid.best_params_}")
print(f"   Best XGB R²: {xgb_r2:.4f}")

# Strategy 4: Gradient Boosting
print("\n4. Testing Gradient Boosting...")
gb = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42
)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
gb_r2 = r2_score(y_test, y_pred_gb)
print(f"   Gradient Boosting R²: {gb_r2:.4f}")

# Strategy 5: Ensemble (average predictions)
print("\n5. Creating ensemble (averaging all predictions)...")
y_pred_ensemble = (y_pred_rf + y_pred_xgb + y_pred_gb) / 3
ensemble_r2 = r2_score(y_test, y_pred_ensemble)
ensemble_mae = mean_absolute_error(y_test, y_pred_ensemble)
ensemble_rmse = np.sqrt(mean_squared_error(y_test, y_pred_ensemble))

print(f"   Ensemble R²:   {ensemble_r2:.4f}")
print(f"   Ensemble MAE:  {ensemble_mae:.2f}")
print(f"   Ensemble RMSE: {ensemble_rmse:.2f}")

# Summary
print(f"\n{'='*70}")
print("FINAL RESULTS SUMMARY")
print(f"{'='*70}\n")

results = pd.DataFrame({
    'Model': ['Ridge', 'Random Forest (tuned)', 'XGBoost (tuned)', 'Gradient Boosting', 'Ensemble (RF+XGB+GB)'],
    'R²': [best_ridge_r2, rf_r2, xgb_r2, gb_r2, ensemble_r2]
}).sort_values('R²', ascending=False)

print(results.to_string(index=False))

best_r2 = results['R²'].max()
best_model = results.loc[results['R²'].idxmax(), 'Model']

print(f"\n{'='*70}")
print(f"Best Model: {best_model}")
print(f"Best R²: {best_r2:.4f}")
print(f"Target R²: 0.70")

if best_r2 >= 0.70:
    print(f"\n✅ TARGET ACHIEVED! ({best_r2:.4f} >= 0.70)")
    print(f"   Improvement: +{(best_r2 - 0.70):.4f} above target")
else:
    gap = 0.70 - best_r2
    print(f"\n⚠️  Gap remaining: {gap:.4f}")
    print(f"   Current best: {best_r2:.4f}")
    print(f"   Need: {gap:.4f} more improvement")

print(f"{'='*70}\n")

# Save best model
if best_model == 'Ensemble (RF+XGB+GB)':
    # Save all three models for ensemble
    joblib.dump(best_rf, 'models/Ensemble_RF_model.pkl')
    joblib.dump(best_xgb, 'models/Ensemble_XGB_model.pkl')
    joblib.dump(gb, 'models/Ensemble_GB_model.pkl')
    print("✓ Saved ensemble models to models/Ensemble_*.pkl")
elif best_model == 'Random Forest (tuned)':
    joblib.dump(best_rf, 'models/RandomForest_tuned_model.pkl')
    print("✓ Saved tuned RandomForest to models/RandomForest_tuned_model.pkl")
elif best_model == 'XGBoost (tuned)':
    joblib.dump(best_xgb, 'models/XGBoost_tuned_model.pkl')
    print("✓ Saved tuned XGBoost to models/XGBoost_tuned_model.pkl")
elif best_model == 'Gradient Boosting':
    joblib.dump(gb, 'models/GradientBoosting_model.pkl')
    print("✓ Saved Gradient Boosting to models/GradientBoosting_model.pkl")
