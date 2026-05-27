import pandas as pd
import pickle

from sklearn.datasets import load_iris

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# =========================
# Load Dataset
# =========================

iris = load_iris()

x = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

y = iris.target

# =========================
# Train Test Split
# =========================

xtr, xte, ytr, yte = train_test_split(
    x,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# =========================
# Pipeline
# =========================

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

# =========================
# Hyperparameter Tuning
# =========================

param_grid = {
    'knn__n_neighbors': [5, 7, 9, 11],
    'knn__weights': ['uniform', 'distance'],
    'knn__metric': ['euclidean', 'manhattan']
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=10,
    scoring='accuracy',
    n_jobs=-1
)

# =========================
# Train
# =========================

grid.fit(xtr, ytr)

best_model = grid.best_estimator_

# =========================
# Predictions
# =========================

ypred = best_model.predict(xte)

# =========================
# Metrics
# =========================

accuracy = accuracy_score(yte, ypred)

precision = precision_score(
    yte,
    ypred,
    average='weighted'
)

recall = recall_score(
    yte,
    ypred,
    average='weighted'
)

f1 = f1_score(
    yte,
    ypred,
    average='weighted'
)

print("\n========== METRICS ==========")

print(f"Accuracy : {accuracy:.4f}")

print(f"Precision : {precision:.4f}")

print(f"Recall : {recall:.4f}")

print(f"F1 Score : {f1:.4f}")

print("\nBest Parameters:")
print(grid.best_params_)

print("\nCross Validation Score:")
print(grid.best_score_)

# =========================
# Save Metrics
# =========================

metrics = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1,
    "CV Score": grid.best_score_
}

pickle.dump(
    metrics,
    open("models/metrics.pkl", "wb")
)

# =========================
# Save Model
# =========================

pickle.dump(
    best_model,
    open("models/knn_classifier.pkl", "wb")
)

print("\nModel Saved Successfully!")