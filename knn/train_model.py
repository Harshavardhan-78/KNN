import pandas as pd
import pickle

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Load dataset
df = pd.read_csv(
    r'C:\Users\HARSHAVARDHAN\OneDrive\Desktop\TekWorks\phase2\25may\slr\Realestate.csv'
)

# Show columns
print("Dataset Columns:")
print(df.columns)

# Features and Target
x = df.drop(
    columns=['No', 'Y house price of unit area']
)

y = df['Y house price of unit area']

print("\nFeature Count:")
print(x.shape[1])

# Train Test Split
xtr, xte, ytr, yte = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

# Base Model
knn = KNeighborsRegressor()

# Hyperparameter Grid
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

# Grid Search CV
grid = GridSearchCV(
    estimator=knn,
    param_grid=param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1
)

# Train
grid.fit(xtr, ytr)

# Best Model
best_model = grid.best_estimator_

# Best Parameters
print("\nBest Parameters:")
print(grid.best_params_)

# Cross Validation Score
print("\nBest Cross Validation Score:")
print(grid.best_score_)

# Predictions
ypred = best_model.predict(xte)

# =========================
# Metrics
# =========================

mae = mean_absolute_error(yte, ypred)

mse = mean_squared_error(yte, ypred)

rmse = mse ** 0.5

r2 = r2_score(yte, ypred)

# =========================
# Display Metrics
# =========================

print("\n========== MODEL METRICS ==========")

print(f"\nMAE  : {mae:.2f}")

print(f"MSE  : {mse:.2f}")

print(f"RMSE : {rmse:.2f}")

print(f"R2 Score : {r2:.2f}")

# Feature Verification
print("\nModel expects features:")
print(best_model.n_features_in_)
metrics = {
    "MAE": mae,
    "MSE": mse,
    "RMSE": rmse,
    "R2": r2
}

pickle.dump(
    metrics,
    open("models/metrics.pkl", "wb")
)

# Save Model
pickle.dump(
    best_model,
    open(
        r"C:\Users\HARSHAVARDHAN\OneDrive\Desktop\TekWorks\phase2\26may\knn\models\knn_model.pkl",
        "wb"
    )
)

print("\nModel saved successfully!")