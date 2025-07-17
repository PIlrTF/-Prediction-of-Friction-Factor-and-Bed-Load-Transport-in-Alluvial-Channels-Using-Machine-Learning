# 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
import xgboost as xgb

# Code Analysis - Step 1: Understanding the existing code structure
'''
The original code:
1. Loads a preprocessed dataset
2. Defines inputs (X) and outputs (y = ['B', 'f'])
3. Splits data into training and test sets
4. Creates a MultiOutputRegressor with XGBoost for predicting multiple targets
5. Trains the model and evaluates using MSE and R²
6. Plots predictions vs actual values
'''

# Code Analysis - Step 2: Adding cross-validation for more robust evaluation
# Load the data
data = pd.read_csv('preprocessed_dataset.csv')

# Define inputs and outputs
X = data.drop(['B', 'f'], axis=1)
y = data[['B', 'f']]

# ENHANCEMENT 1: Examine data characteristics
print("Dataset shape:", data.shape)
print("\nFeature statistics:")
print(X.describe())
print("\nTarget statistics:")
print(y.describe())

# ENHANCEMENT 2: Check for missing values
print("\nMissing values in features:", X.isnull().sum().sum())
print("Missing values in targets:", y.isnull().sum().sum())

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ENHANCEMENT 3: Implement cross-validation for model evaluation
# Define a custom function to calculate MSE for both outputs
def multi_output_mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, multioutput='raw_values').mean()

# Create a scorer for GridSearchCV
mse_scorer = make_scorer(multi_output_mse, greater_is_better=False)

# ENHANCEMENT 4: Hyperparameter tuning with cross-validation
param_grid = {
    'estimator__n_estimators': [200, 500],
    'estimator__learning_rate': [0.01, 0.05, 0.1],
    'estimator__max_depth': [4, 6, 8],
    'estimator__subsample': [0.7, 0.8, 0.9],
    'estimator__colsample_bytree': [0.7, 0.8, 0.9]
}

# Create base model
base_model = xgb.XGBRegressor(random_state=42)
multi_output_model = MultiOutputRegressor(base_model)

# Grid search with cross-validation
grid_search = GridSearchCV(
    estimator=multi_output_model,
    param_grid=param_grid,
    scoring=mse_scorer,
    cv=5,  # 5-fold cross-validation
    n_jobs=-1,  # Use all available cores
    verbose=1
)

print("\nPerforming grid search with cross-validation...")
# Uncomment to run grid search (can be time-consuming)
# grid_search.fit(X_train, y_train)
# print("Best parameters:", grid_search.best_params_)
# print("Best cross-validation score:", -grid_search.best_score_)  # Negative because of scoring metric
# best_model = grid_search.best_estimator_

# For demonstration, we'll use the original parameters
# Build the model with original parameters
model = MultiOutputRegressor(xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
))

# ENHANCEMENT 5: K-fold cross-validation for more robust evaluation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Lists to store cross-validation results
cv_mse_b = []
cv_mse_f = []
cv_r2_b = []
cv_r2_f = []

print("\nPerforming 5-fold cross-validation...")
for train_index, val_index in kf.split(X_train):
    # Split data
    X_cv_train, X_cv_val = X_train.iloc[train_index], X_train.iloc[val_index]
    y_cv_train, y_cv_val = y_train.iloc[train_index], y_train.iloc[val_index]

    # Train model
    model.fit(X_cv_train, y_cv_train)

    # Predict
    y_cv_pred = model.predict(X_cv_val)

    # Calculate metrics
    mse = mean_squared_error(y_cv_val, y_cv_pred, multioutput='raw_values')
    r2 = r2_score(y_cv_val, y_cv_pred, multioutput='raw_values')

    # Store results
    cv_mse_b.append(mse[0])
    cv_mse_f.append(mse[1])
    cv_r2_b.append(r2[0])
    cv_r2_f.append(r2[1])

# Print cross-validation results
print("\nCross-validation results:")
print(f"MSE for B: {np.mean(cv_mse_b):.6f} ± {np.std(cv_mse_b):.6f}")
print(f"MSE for f: {np.mean(cv_mse_f):.6f} ± {np.std(cv_mse_f):.6f}")
print(f"R² for B: {np.mean(cv_r2_b):.4f} ± {np.std(cv_r2_b):.4f}")
print(f"R² for f: {np.mean(cv_r2_f):.4f} ± {np.std(cv_r2_f):.4f}")

# Train the final model on the full training set
print("\nTraining final model on full training set...")
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate on test set
test_mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
test_r2 = r2_score(y_test, y_pred, multioutput='raw_values')

print("\nTest set evaluation:")
print(f"Mean Squared Error for B: {test_mse[0]:.6f}")
print(f"Mean Squared Error for f: {test_mse[1]:.6f}")
print(f"R² Score for B: {test_r2[0]:.4f}")
print(f"R² Score for f: {test_r2[1]:.4f}")

# ENHANCEMENT 6: More informative visualizations with residual plots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# True vs Predicted for B
axes[0,0].scatter(y_test['B'], y_pred[:, 0], alpha=0.6, color='blue')
axes[0,0].plot([y_test['B'].min(), y_test['B'].max()],
             [y_test['B'].min(), y_test['B'].max()], 'k--')
axes[0,0].set_title('True vs Predicted for B')
axes[0,0].set_xlabel('True B')
axes[0,0].set_ylabel('Predicted B')
axes[0,0].grid(True, alpha=0.3)

# True vs Predicted for f
axes[0,1].scatter(y_test['f'], y_pred[:, 1], alpha=0.6, color='green')
axes[0,1].plot([y_test['f'].min(), y_test['f'].max()],
             [y_test['f'].min(), y_test['f'].max()], 'k--')
axes[0,1].set_title('True vs Predicted for f')
axes[0,1].set_xlabel('True f')
axes[0,1].set_ylabel('Predicted f')
axes[0,1].grid(True, alpha=0.3)

# Residual plot for B
residuals_b = y_test['B'] - y_pred[:, 0]
axes[1,0].scatter(y_pred[:, 0], residuals_b, alpha=0.6, color='blue')
axes[1,0].axhline(y=0, color='k', linestyle='--')
axes[1,0].set_title('Residuals for B')
axes[1,0].set_xlabel('Predicted B')
axes[1,0].set_ylabel('Residual (True - Predicted)')
axes[1,0].grid(True, alpha=0.3)

# Residual plot for f
residuals_f = y_test['f'] - y_pred[:, 1]
axes[1,1].scatter(y_pred[:, 1], residuals_f, alpha=0.6, color='green')
axes[1,1].axhline(y=0, color='k', linestyle='--')
axes[1,1].set_title('Residuals for f')
axes[1,1].set_xlabel('Predicted f')
axes[1,1].set_ylabel('Residual (True - Predicted)')
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.suptitle('Model Evaluation: XGBoost MultiOutput Regression', fontsize=16)
plt.subplots_adjust(top=0.93)
# plt.show()

# ENHANCEMENT 7: Feature importance analysis
feature_names = X.columns

# Get feature importance for each target
b_importances = model.estimators_[0].feature_importances_
f_importances = model.estimators_[1].feature_importances_

# Create DataFrames for easy sorting
b_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': b_importances
}).sort_values('Importance', ascending=False)

f_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': f_importances
}).sort_values('Importance', ascending=False)

print("\nFeature importance for B:")
print(b_importance_df.head(10))

print("\nFeature importance for f:")
print(f_importance_df.head(10))

# ENHANCEMENT 8: Visualization of feature importance
plt.figure(figsize=(14, 10))

# Feature importance for B
plt.subplot(2, 1, 1)
plt.barh(b_importance_df['Feature'][:10], b_importance_df['Importance'][:10])
plt.title('Top 10 Feature Importance for B')
plt.xlabel('Importance')
plt.gca().invert_yaxis()

# Feature importance for f
plt.subplot(2, 1, 2)
plt.barh(f_importance_df['Feature'][:10], f_importance_df['Importance'][:10])
plt.title('Top 10 Feature Importance for f')
plt.xlabel('Importance')
plt.gca().invert_yaxis()

plt.tight_layout()
# plt.show()

# ENHANCEMENT 9: Function for predicting new values with confidence estimates
def predict_with_confidence(model, new_data, cv_results):
    """
    Make predictions and estimate confidence based on cross-validation performance

    Parameters:
    model: Trained MultiOutputRegressor model
    new_data: DataFrame with feature data
    cv_results: Dictionary with cv_mse_b, cv_mse_f values

    Returns:
    Dictionary with predictions and confidence intervals
    """
    # Make predictions
    pred = model.predict(new_data)

    # Calculate 95% confidence intervals based on CV performance
    std_b = np.sqrt(np.mean(cv_results['cv_mse_b']))
    std_f = np.sqrt(np.mean(cv_results['cv_mse_f']))

    # For each prediction
    results = []
    for i in range(len(pred)):
        results.append({
            'B_pred': pred[i, 0],
            'B_95_lower': pred[i, 0] - 1.96 * std_b,
            'B_95_upper': pred[i, 0] + 1.96 * std_b,
            'f_pred': pred[i, 1],
            'f_95_lower': pred[i, 1] - 1.96 * std_f,
            'f_95_upper': pred[i, 1] + 1.96 * std_f
        })

    return results

# Example usage of prediction function with confidence intervals
# (Create sample data using the first row of the test set for demonstration)
new_data_example = X_test.iloc[[0]]
print("\nExample prediction with confidence intervals:")
cv_results = {'cv_mse_b': cv_mse_b, 'cv_mse_f': cv_mse_f}
prediction_with_conf = predict_with_confidence(model, new_data_example, cv_results)
print(pd.DataFrame(prediction_with_conf))

# ENHANCEMENT 10: Save the model for later use
import joblib

# Save the model
# joblib.dump(model, 'xgboost_multioutput_model.pkl')

# And here's how you would load it
# loaded_model = joblib.load('xgboost_multioutput_model.pkl')

print("\nAnalysis and enhancement of the XGBoost model complete!")
