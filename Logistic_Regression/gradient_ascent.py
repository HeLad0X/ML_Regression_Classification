"""
This module will be implementing the Logistic Regression model using Gradient ascent
The only packages used will be the cupy cudf and other libraries just for matrix and vector manipulation
It will be custom model built from ground up only using the math and minimizing the cost function and updating the gradient


The steps followed will be: 
    Initialize weights and bias (usually zeros or small random numbers).
    For each epoch:
        Compute linear output: z=X⋅W+b
        Compute loss (optional for monitoring)
        Compute gradients 
    Update wieghts and bias using gradient descent
"""
import sys
import os

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add parent directory to sys.path
sys.path.append(parent_dir)

from preprocessing import get_preprocessed_data
import pickle
from config import get_model_path, GradientAscent
from test_model import start_testing_model
import cupy as cp

# Initialize the parameters
def initialize_parameters(X):
    feature_size = X.shape[1]
    params = cp.random.uniform(-1 / cp.sqrt(feature_size),
                               1 / cp.sqrt(feature_size), feature_size)

    return params


def gradient_function(X : cp.ndarray, y : cp.ndarray, params : cp.ndarray) -> cp.ndarray:
    z = X.dot(params)
    prediction = sigmoid(z)

    gradient = (1 / len(y)) * (X.T.dot(prediction - y))

    return gradient


def loss_function(X, y, params):
    z = X.dot(params)
    prediction = sigmoid(z)
    epsilon = 1e-10
    prediction = cp.maximum(epsilon, prediction)
    prediction = cp.minimum(1 - epsilon, prediction)
    loss = -cp.mean(y * cp.log(prediction) + (1 - y) * cp.log(1 - prediction))
    return loss

def update_params(params, gradient, learning_rate):
    return params - learning_rate * gradient


def sigmoid(z):
    return 1 / (1 + cp.exp(-z))

def save_model_to_path(model, model_name):
    model_path = get_model_path(model_name=model_name)

    print('Saving model...')
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)

def load_model_from_path(model_name):
    model_path = get_model_path(model_name)
    
    print('Loading model...')
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

def train_model(split_data, learning_rate, epoch):
    X_train = split_data['X_train']
    y_train = split_data['y_train']

    # Add bias term to X_train
    X_train = cp.c_[cp.ones((X_train.shape[0], 1)), X_train]

    params = initialize_parameters(X_train)
    cost_curr = None

    current_epoch = 0
    while current_epoch < epoch:
        cost_curr = loss_function(X_train, y_train, params)

        gradient = gradient_function(X_train, y_train, params)
        params = update_params(params, gradient, learning_rate)

        if current_epoch % 50 == 0:
            print(f"Epoch: {current_epoch}, Cost: {cost_curr}")
        current_epoch += 1

    return params, cost_curr

def start_regression(pkl_model_path, split_data):
    
    best_params = None
    least_cost = float('inf')

    epochs = [10, 50, 100, 200, 500, 1000]
    learning_rates = [0.01, 0.05, 0.1, 0.5]

    for epoch in epochs:
        for learning_rate in learning_rates:
            params, cost = train_model(split_data, learning_rate, epoch)

            if cost < least_cost:
                least_cost = cost
                best_params = params

    model = GradientAscent(best_params)
    print('Training complete')
    print('Best Cost:', least_cost)
    print('Best params:', best_params)

    save_model_to_path(model, pkl_model_path)
    


if __name__ == '__main__':
    model_name_pkl = 'logistic_gradient_descent.pkl'
    dataset_name = 'diabetes.csv'
    target_column = 'Outcome'
    
    split_data_train_test = get_preprocessed_data(dataset_name, target_column)

    pkl_model_path = get_model_path(model_name_pkl)

    if os.path.exists(pkl_model_path):
        print('Model already exists')
        print('Moving to testing the model....')
    else:
        print('No model found. Training the model...')
        start_regression(pkl_model_path, split_data_train_test)
        print('Training completed.')
        print('Testing the model...')

    start_testing_model(get_model_path(pkl_model_path), split_data_train_test)
