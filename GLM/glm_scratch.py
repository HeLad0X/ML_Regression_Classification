import os
import sys
import cupy as cp
import pickle

# Add parent directory for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from preprocessing import get_preprocessed_data
from config import get_model_path
from test_model import start_testing_model

class GLM:
    def __init__(self, family='gaussian', learning_rate=0.01, epochs=1000, tol=1e-6):
        self.family = family
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.tol = tol
        self.weights = None

    def _initialize_weights(self, n_features):
        # n_features includes bias term
        self.weights = cp.random.uniform(
            -1 / cp.sqrt(n_features),
            1 / cp.sqrt(n_features),
            size=(n_features, 1)
        )

    def _link_inverse(self, z):
        if self.family == 'gaussian':
            return z
        elif self.family == 'binomial':
            return 1 / (1 + cp.exp(-z))
        else:
            raise NotImplementedError(f"Family '{self.family}' not implemented")

    def _loss(self, y_true, y_pred):
        if self.family == 'gaussian':
            return cp.mean((y_true - y_pred) ** 2)
        elif self.family == 'binomial':
            eps = 1e-10
            y_pred = cp.clip(y_pred, eps, 1 - eps)
            return -cp.mean(y_true * cp.log(y_pred) + (1 - y_true) * cp.log(1 - y_pred))
        else:
            raise NotImplementedError

    def _gradient(self, X, y_true, y_pred):
        return cp.dot(X.T, (y_pred - y_true)) / X.shape[0]

    def fit(self, X, y):
        # Convert to CuPy arrays and reshape
        X = cp.asarray(X)
        y = cp.asarray(y).reshape(-1, 1)

        # Add bias term
        X = cp.hstack((cp.ones((X.shape[0], 1)), X))
        n_features = X.shape[1]
        self._initialize_weights(n_features)

        for epoch in range(self.epochs):
            # Forward pass
            z = X.dot(self.weights)                        # (m, n+1) @ (n+1, 1) -> (m, 1)
            y_pred = self._link_inverse(z)

            # Compute loss and gradient
            loss = self._loss(y, y_pred)
            grad = self._gradient(X, y, y_pred)

            # Update weights
            self.weights -= self.learning_rate * grad

            # Logging and convergence check
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")
            if cp.linalg.norm(grad) < self.tol:
                print(f"Converged at epoch {epoch}")
                break

    def predict(self, X):
        X = cp.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X = cp.hstack((cp.ones((X.shape[0], 1)), X))
        z = X.dot(self.weights)                          # (m, n+1) @ (n+1, 1)
        return self._link_inverse(z)

# Save model to disk
def save_model(model, model_path):
    print("Saving model...")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

# Load model from disk
def load_model(model_path):
    print("Loading model...")
    with open(model_path, 'rb') as f:
        return pickle.load(f)

# Train over hyperparameter grid
def train_model(data, family='binomial'):
    X_train = data['X_train']
    y_train = data['y_train']

    best_model = None
    best_loss = float('inf')
    learning_rates = [0.01, 0.05]
    epochs_list = [100, 300, 500]

    for lr in learning_rates:
        for ep in epochs_list:
            model = GLM(family=family, learning_rate=lr, epochs=ep)
            model.fit(X_train, y_train)
            preds = model.predict(X_train)
            loss = model._loss(y_train.reshape(-1,1), preds)
            print(f"lr={lr}, ep={ep}, loss={loss:.6f}")
            if loss < best_loss:
                best_loss = loss
                best_model = model

    return best_model

# Entry point to train/load and test
def start_regression(model_name, dataset_name, target_column, family='binomial'):
    split_data = get_preprocessed_data(dataset_name, target_column)
    model_path = get_model_path(model_name)

    if os.path.exists(model_path):
        print('Model already exists. Loading...')
        model = load_model(model_path)
    else:
        print('Training new GLM model...')
        model = train_model(split_data, family=family)
        save_model(model, model_path)

    print('Testing model...')
    start_testing_model(model_path, split_data)

if __name__ == '__main__':
    model_file = 'glm_model.pkl'
    start_regression(model_file, 'diabetes.csv', 'Outcome', family='binomial')
