from preprocessing import get_preprocessed_data
from config import get_model_path
from test_model import start_testing_model
from cuml.model_selection import GridSearchCV
from cuml.linear_model import LogisticRegression

import joblib
import os

# Start training
def train_model(split_data):
    X_train = split_data['X_train']
    y_train = split_data['y_train']

    # Define logistic regression model
    model = LogisticRegression()

    # Define hyperparameters for grid search
    # Define hyperparameters for tuning
    param_grid = {
        'penalty': ['l2'],  # Only 'l2' is supported in cuML
        'C': [0.1, 1, 10, 100],  # Regularization strength
        'solver': ['lbfgs'],  # Supported solver in cuML
        'max_iter': [100, 500, 1000]  # Number of iterations
    }

    # Define grid search object
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', verbose=1)
    grid_search.fit(X_train, y_train)

    # Print the best params
    print("Best Parameters:", grid_search.best_params_)
    print("Best Score:", grid_search.best_score_)

    # Get the best model and return it
    best_model = grid_search.best_estimator_
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
    cuml_model = None

    dataset_name = 'diabetes.csv'
    split_data_train_test = get_preprocessed_data(dataset_name, 'Outcome')

    if os.path.exists(get_model_path(cuml_model_name)):
        print('Model already exists')
        print('Moving to testing the model....')
        cuml_model = load_model_from_path(cuml_model_name)
    else:
        print('No model found. Training the model...')
        cuml_model = start_logistic_regression(cuml_model_name, split_data_train_test)
        print('Training completed.')
        print('Testing the model...')

    start_testing_model(get_model_path(cuml_model_name), split_data_train_test)

