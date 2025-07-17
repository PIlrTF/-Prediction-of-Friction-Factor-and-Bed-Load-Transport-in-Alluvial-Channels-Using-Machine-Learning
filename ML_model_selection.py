# XGBOOST

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

# Load the data
data = pd.read_csv('/content/preprocessed_dataset.csv')

# Define inputs and outputs
X = data.drop(['B', 'f'], axis=1)
y = data[['B', 'f']]

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = MultiOutputRegressor(xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
))

# Train the model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
r2 = r2_score(y_test, y_pred, multioutput='raw_values')

print(f"Mean Squared Error for B and f: {mse}")
print(f"R2 Score for B and f: {r2}")

# Plotting true vs predicted for B and f separately
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# For B
axes[0].scatter(y_test['B'], y_pred[:, 0], alpha=0.5, color='blue')
axes[0].plot([y_test['B'].min(), y_test['B'].max()], [y_test['B'].min(), y_test['B'].max()], 'k--')
axes[0].set_title('True vs Predicted for B')
axes[0].set_xlabel('True B')
axes[0].set_ylabel('Predicted B')

# For f
axes[1].scatter(y_test['f'], y_pred[:, 1], alpha=0.5, color='green')
axes[1].plot([y_test['f'].min(), y_test['f'].max()], [y_test['f'].min(), y_test['f'].max()], 'k--')
axes[1].set_title('True vs Predicted for f')
axes[1].set_xlabel('True f')
axes[1].set_ylabel('Predicted f')

plt.tight_layout()
plt.show()

# Example: Predicting new values
# new_data = pd.DataFrame({...})
# predictions = model.predict(new_data)
# Manual Cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

r2_scores_B = []
r2_scores_f = []

for train_index, val_index in kf.split(X):
    X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
    y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

    model_fold = MultiOutputRegressor(xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    ))

    model_fold.fit(X_train_fold, y_train_fold)
    y_val_pred = model_fold.predict(X_val_fold)

    r2_B = r2_score(y_val_fold['B'], y_val_pred[:, 0])
    r2_f = r2_score(y_val_fold['f'], y_val_pred[:, 1])

    r2_scores_B.append(r2_B)
    r2_scores_f.append(r2_f)

print(f"Cross-validated R2 scores for B: {r2_scores_B}")
print(f"Mean R2 for B: {np.mean(r2_scores_B)}")

print(f"Cross-validated R2 scores for f: {r2_scores_f}")
print(f"Mean R2 for f: {np.mean(r2_scores_f)}")
