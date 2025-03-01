from cuml.linear_model import LinearRegression
from cuml.model_selection import GridSearchCV
from cuml.preprocessing import StandardScaler, PolynomialFeatures
from cuml.pipeline import Pipeline
import os
from preprocess_df import get_preprocessed_data
from config import get_model_path
from test_model import start_testing_model
import joblib

# Training the model
def train_model(split_data):
    X_train = split_data['X_train']
    y_train = split_data['y_train']

    # Create pipeline with scaling, polynomial features, and regression
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures()),
        ('regression', LinearRegression())
    ])

    # Define parameter grid including all transformations
    param_grid = {
        'poly__degree': [1, 2, 3],
        'regression__fit_intercept': [True, False],
        'regression__algorithm': ['eig', 'svd'],
        'regression__normalize': [True, False]
    }

    # Perform grid search
    grid_search = GridSearchCV(pipeline, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    print("Best Parameters:", grid_search.best_params_)
    print("Best Score:", grid_search.best_score_)

    best_model = grid_search.best_estimator_
    return best_model

# Load model
def load_model_from_path(model_name):
    model = joblib.load(get_model_path(model_name))
    return model

# Save model
def save_model_to_path(model, model_name):
    joblib.dump(model, get_model_path(model_name))

# Start the regression
def start_regression(model_name, split_data):
    model = train_model(split_data)
    save_model_to_path(model, model_name)

    return model


if __name__ == '__main__':
    model_name_cuml = 'linear_regression_cuml.joblib'
    cuml_model = None
    split_data_train_test = get_preprocessed_data(scale_data=False)

    if os.path.exists(get_model_path(model_name_cuml)):
        print('Model already exists')
        print('Moving to testing the model....')
        cuml_model = load_model_from_path(model_name_cuml)
    else:
        print('No model found. Training the model...')
        cuml_model = start_regression(model_name_cuml, split_data_train_test)
        print('Training completed.')
        print('Testing the model...')

    start_testing_model(get_model_path(model_name_cuml), split_data_train_test)