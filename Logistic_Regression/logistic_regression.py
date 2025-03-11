import sys
import os

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add parent directory to sys.path
sys.path.append(parent_dir)

from preprocessing import get_preprocessed_data
from config import get_model_path
from test_model import start_testing_model
from cuml.linear_model import LogisticRegression
from cuml.model_selection import StratifiedKFold
import cupy as cp
import joblib


def train_model(split_data):
    X_train = split_data['X_train']  # CuPy array
    y_train = split_data['y_train']  # CuPy array

    X_train = cp.ascontiguousarray(X_train)
    y_train = cp.ascontiguousarray(y_train)

    param_grid = {
        'penalty': ['l2'],
        'C': [0.1, 1, 10, 100],
        'max_iter': [100, 200, 500, 1000, 5000]
    }

    best_score = -float('inf')
    best_model = None
    skf = StratifiedKFold(n_splits=5)

    for C in param_grid['C']:
        for max_iter in param_grid['max_iter']:
            model = LogisticRegression(penalty='l2', C=C, max_iter=max_iter)
            scores = []
            for train_idx, val_idx in skf.split(X_train, y_train):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]
                model.fit(X_tr, y_tr)
                score = model.score(X_val, y_val)
                scores.append(score)
            mean_score = cp.mean(cp.array(scores))
            print(f"C: {C}, max_iter: {max_iter}, CV Score: {mean_score}")
            if mean_score > best_score:
                best_score = mean_score
                best_model = model.fit(X_train, y_train)  # Refit on full data

    print("Best CV Score:", best_score)
    return best_model

# Load model
def load_model_from_path(model_name):
    model = joblib.load(get_model_path(model_name))
    return model

# Save model
def save_model_to_path(model, model_name):
    joblib.dump(model, get_model_path(model_name))

# Start logistic regression
def start_logistic_regression(model_name, split_data):
    model = train_model(split_data)
    save_model_to_path(model, model_name)

    return model


if __name__ == '__main__':
    cuml_model_name = 'logistic_regression_cuml.joblib'

    dataset_name = 'diabetes.csv'
    split_data_train_test = get_preprocessed_data(dataset_name, 'Outcome')

    if os.path.exists(get_model_path(cuml_model_name)):
        print('Model already exists')
        print('Moving to testing the model....')
    else:
        print('No model found. Training the model...')

        start_logistic_regression(cuml_model_name, split_data_train_test)
        print('Training completed.')
        print('Testing the model...')

    start_testing_model(get_model_path(cuml_model_name), split_data_train_test)

