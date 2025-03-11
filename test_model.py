from cuml.metrics import mean_squared_error, mean_absolute_error, r2_score
import cupy as cp
import joblib
import pickle

# Predicting some random values from the model of cuml
def predict_model_cuml(model, split_data):
    X_test = split_data['X_test']
    y_test = split_data['y_test']
    y_pred = model.predict(X_test)
    for i in range(10):
        round_value = int(cp.round(y_pred[i]))
        print(round_value, ':', y_test[i], round_value == int(y_test[i]))

    return y_pred

# Sigmoid to get the actual value for the output
def sigmoid(z):
    return 1 / (1 + cp.exp(-z))

# Predicting some random values from the model of gradient descent
def predict_model_custom(model, split_data):
    print('Custom model')
    X_test = split_data['X_test']
    y_test = split_data['y_test']
    y_pred = model.predict(X_test)
    for i in range(10):
        round_value = int(cp.round(y_pred[i]))
        print(round_value, ':', y_test[i], round_value == int(y_test[i]))

    return y_pred

# Printing various metrics of the model
def test_model(y_pred, split_data):
    y_test = split_data['y_test'].astype('float32')
    X_test = split_data['X_test'].astype('float32')
    y_pred = y_pred.astype('float32')

    # Compute evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = cp.sqrt(mse)  # RMSE is sqrt of MSE
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Compute Adjusted R² (manual calculation)
    n = len(y_test)  # Number of samples
    p = X_test.shape[1]  # Number of features
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    # Print results
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R² Score: {r2}")
    print(f"Adjusted R² Score: {adjusted_r2}")

# Start testing the models
def start_testing_model(model_path, split_data_train_test):
    print('Starting testing....')

    model_type = str(model_path).split('.')[-1]

    if model_type == 'joblib':
        linear_model = joblib.load(model_path)
        y_pred = predict_model_cuml(linear_model, split_data_train_test)
    elif model_type == 'pkl':
        with open(model_path, 'rb') as fl:
            linear_model = pickle.load(fl)
            y_pred = predict_model_custom(linear_model, split_data_train_test)
    else:
        print('Model type not found')
        return

    test_model(y_pred, split_data_train_test)