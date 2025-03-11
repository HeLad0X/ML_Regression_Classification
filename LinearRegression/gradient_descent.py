import sys
import os

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add parent directory to sys.path
sys.path.append(parent_dir)

from config import get_model_path, GradientDescent
from preprocessing import get_preprocessed_data
import cupy as cp
import pickle
from test_model import start_testing_model

# Initialize the parameters
def initialize_parameters(X):
    feature_size = X.shape[1]
    params = cp.random.uniform(-1 / cp.sqrt(feature_size),
                               1 / cp.sqrt(feature_size), feature_size)

    return params

# Calculate the cost function
def cost_function(X, y, params):
    m = len(y)
    predictions = X.dot(params)
    errors = predictions - y
    cost = (1 / (2 * m)) * cp.sum(errors ** 2)

    return cost

# Calculate the gradients
def calculate_gradients(X, y, params):
    m = len(y)
    predictions = X.dot(params)
    errors = predictions - y
    gradients = (1 / m) * X.T.dot(errors)

    return gradients

# Update the parameters
def update_parameters(params, gradients, learning_rate):
    params -= learning_rate * gradients

    return params

# start the gradient descent
def gradient_descent(data, learning_rate, epochs):
    """
    Steps to take:
    1. Initialize the parameters (w0, w1, w2, w3, w4)
    2. Calculate the predictions (y_pred) using the parameters
    3. Calculate the cost function (mse)
    4. Calculate the gradients
    5. Update the parameters
    6. Repeat steps 2-5 until convergence
    """

    mse_curr = None
    cost_diff = 1
    cost_threshold = 0.000001
    X_train = data['X_train']
    y_train = data['y_train']

    # Add bias term to X_train
    X_train = cp.c_[cp.ones((X_train.shape[0], 1)), X_train]

    # Initialize the parameters (w0, w1, w2, w3, w4)
    params = initialize_parameters(X_train)

    current_epoch = 0
    while cost_diff > cost_threshold and current_epoch < epochs:
        mse_prev = mse_curr

        # Calculate the cost function
        mse_curr = cost_function(X_train, y_train, params)
        if mse_prev is not None:
            cost_diff = abs(mse_prev - mse_curr)

        # Calculate the gradients
        gradients = calculate_gradients(X_train, y_train, params)

        # Update the parameters
        params = update_parameters(params, gradients, learning_rate)

        if current_epoch % 10 == 0:
            print(f"Epoch: {current_epoch}, MSE: {mse_curr}")
        current_epoch += 1


    return mse_curr, params

# Save the model
def save_model(model, model_path):
     print('Saving model...')
     with open(model_path, 'wb') as file:
        pickle.dump(model, file)

# Load model
def load_model(model_path):
    print('Loading model...')
    with open(model_path, 'rb') as fl:
        model = pickle.load(fl)
    return model

# Start training the model
def train_model(model_path, data):
    learning_rates = [0.001, 0.01, 0.05, 0.1]
    epoch_size = [100, 200, 500]

    best_mse = float('inf')
    best_params = None

    for rate in learning_rates:
        for epoch in epoch_size:
            mse, params = gradient_descent(data, rate, epoch)
            if mse < best_mse:
                best_mse = mse
                best_params = params

    # Create model instance for saving
    print('Training complete')
    print('Best MSE:', best_mse)
    print('Best params:', best_params)
    model = GradientDescent(best_params)
    save_model(model, model_path)

    return model

if __name__ == '__main__':
    model_path_gd = get_model_path('linear_gradient_descent.pkl')

    dataset_name = 'Iris.csv'
    target_column = 'Species'
    iris_data = get_preprocessed_data(dataset_name, target_column)

    gradient_descent_model = None

    if os.path.exists(model_path_gd):
        print('Model already exists')
        print('Skipping training....')
    else:
        print('No model found')
        print('Starting training....')
        gradient_descent_model = train_model(model_path_gd, iris_data)

    start_testing_model(model_path_gd, iris_data)