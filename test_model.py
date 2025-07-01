import os
import cupy as cp
import joblib
import pickle
import numpy as np

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)

def is_binary(y_np):
    """Return True if y_np contains exactly two unique values {0,1}."""
    uniq = np.unique(y_np)
    return len(uniq) == 2 and set(uniq).issubset({0, 1})

def load_model(model_path):
    """Load a model from .joblib or .pkl/.pickle file."""
    ext = str(model_path).split('.')[-1].lower()
    if ext == "joblib":
        return joblib.load(model_path)
    elif ext in ("pkl", "pickle"):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unsupported model extension '{ext}'")

def predict(model, X):
    """Run model.predict on X and return a CuPy array of predictions."""
    return model.predict(X)

def to_numpy(cp_array):
    """Convert a CuPy array to a flattened NumPy array."""
    return cp.asnumpy(cp_array).ravel()

def print_sample_predictions(y_pred_np, y_true_np, binary, n=10):
    """Print the first n predictions versus true values."""
    print("\nSample predictions:")
    for i in range(min(n, len(y_true_np))):
        if binary:
            pred_label = int(y_pred_np[i] >= 0.5)
            print(f"#{i+1}: pred={pred_label} (p={y_pred_np[i]:.3f})  true={int(y_true_np[i])}")
        else:
            print(f"#{i+1}: pred={y_pred_np[i]:.3f}  true={y_true_np[i]:.3f}")

def compute_classification_metrics(y_true_np, y_prob_np):
    """Compute and print binary classification metrics."""
    y_hat = (y_prob_np >= 0.5).astype(int)
    print("\n=== Classification Metrics ===")
    print(f"Accuracy       : {accuracy_score(y_true_np, y_hat):.4f}")
    print(f"Precision      : {precision_score(y_true_np, y_hat, zero_division=0):.4f}")
    print(f"Recall         : {recall_score(y_true_np, y_hat, zero_division=0):.4f}")
    print(f"F1 Score       : {f1_score(y_true_np, y_hat, zero_division=0):.4f}")
    print(f"ROC AUC        : {roc_auc_score(y_true_np, y_prob_np):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true_np, y_hat))

def compute_regression_metrics(y_true_np, y_pred_np, feature_count=None):
    """Compute and print regression metrics. feature_count is number of features for adjusted R²."""
    mse = mean_squared_error(y_true_np, y_pred_np)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_np, y_pred_np)
    r2 = r2_score(y_true_np, y_pred_np)
    n = len(y_true_np)
    p = feature_count if feature_count is not None else 1
    adj_r2 = 1 - (1 - r2) * (n - 1) / max(n - p - 1, 1)

    print("\n=== Regression Metrics ===")
    print(f"MSE             : {mse:.6f}")
    print(f"RMSE            : {rmse:.6f}")
    print(f"MAE             : {mae:.6f}")
    print(f"R²              : {r2:.6f}")
    print(f"Adjusted R²     : {adj_r2:.6f}")

def start_testing_model(model_path, split_data):
    """
    Load a model, get predictions on split_data,
    and dispatch to classification or regression metrics.
    """
    print("Starting testing...")
    model = load_model(model_path)

    X_test = split_data['X_test']
    y_test_cp = split_data['y_test']

    # Predictions as CuPy
    y_pred_cp = predict(model, X_test)

    # Convert to NumPy
    y_test_np = to_numpy(y_test_cp)
    y_pred_np = to_numpy(y_pred_cp)

    # Determine problem type
    binary = is_binary(y_test_np)

    # Print some sample predictions
    print_sample_predictions(y_pred_np, y_test_np, binary)

    # Compute and print metrics
    if binary:
        compute_classification_metrics(y_test_np, y_pred_np)
    else:
        # If you know number of features, pass feature_count=X_test.shape[1]
        compute_regression_metrics(y_test_np, y_pred_np, feature_count=None)

