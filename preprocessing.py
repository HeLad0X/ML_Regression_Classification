"""
This module implement preprocessing any csv document and is capable of handling most use case scenarios
    Scenarios:
    -> Missing value: used filling method. Mean for numerical and mode for categorical
    -> Categorical data handling: cudf.get_dummies is used for it
    -> Scaling data: Data is scaled using Standardization, Robust scaling(IQR), and log transformation for different skewness

    
    Steps to take:
    1. Get the files
    2. Handle missing values
    3. Handle categorical data
    4. Feature scaling
    5. Data splitting
    6. Class imbalance
"""

from config import get_df
import cupy as cp
import cudf as cd
from cuml.model_selection import train_test_split
from typing import List


# Handling missing values
def handle_na(df, categorical_features, numerical_features):
    print('Handling NaNs...')
    for col in list(df.columns):
        if col in numerical_features:
            df[col] = df[col].fillna(df[col].mean())
        elif col in categorical_features:
            df[col] = df[col].fillna(df[col].mode()[0])

    return df

# Handling categorical features
def handle_cat_features(df, categorical_features):
    encoded_df = cd.get_dummies(df[categorical_features])

    return encoded_df

# Encode categorical features
def encode_df(df, categorical_features):
    encoded_df = None
    encoded = False

    if len(categorical_features) > 0:
        print('Encoding categorical features...')
        encoded_df = handle_cat_features(df, categorical_features)
        encoded = True

    return encoded_df, encoded

# Function to get the skewness of the data
def get_skewness(df: cd.DataFrame) -> List[float]:
    skewness = []
    for col in df.columns:
        data = cp.asarray(df[col].values)
        mean = cp.mean(data)
        std_dev = cp.std(data, ddof=1)

        skewness.append(cp.mean(((data - mean) / std_dev) ** 3).item())

    return skewness


# Z-score scaling for normal data
def z_score_scaling(data: cp.array) -> cp.array:
    mean = cp.mean(data)
    std_dev = cp.std(data, ddof=1)
    standard_array =  (data - mean) / std_dev
    standard_array = cp.where(standard_array > 3, 3, standard_array)
    standard_array = cp.where(standard_array < -3, -3, standard_array)

    return standard_array


# IQR scaling for skewed data
def iqr_normalization(arr: cp.array) -> cp.array:
    q1 = cp.quantile(arr, 0.25)
    q3 = cp.quantile(arr, 0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    arr = cp.where(arr < lower_bound, lower_bound, arr)
    arr = cp.where(arr > upper_bound, upper_bound, arr)

    return arr

# Log transformation scaling
def log_transformation(arr: cp.array) -> cp.array:
    min_value = arr.min()

    # If there are negative values, shift the data
    if min_value < 0:
        arr = arr - min_value + 1  # Shift values to make all positive (adds |min| + 1)

    return cp.log1p(arr)

# Function to scale the data
def scale_numerical_features(df, numerical_features) -> cd.DataFrame:
    """
    Both z-score scaling and IQR scaling are used for scaling the data.
    They are also handling outliers by capping to max and min values.
    Log transformation is being used for highly skewed data while also handling negative values.
    """
    numerical_df = df[numerical_features]
    skewness = get_skewness(numerical_df)
    for i, col in enumerate(numerical_df.columns):
        if skewness[i] > 1.8:
            print(f'Feature {col} is of skewness: {skewness[i]}. Going with log transformation scaling...')
            numerical_df[col] = log_transformation(cp.asarray(numerical_df[col].values))
        elif skewness[i] > 0.75:
            print(f'Feature {col} is of skewness: {skewness[i]}. Going with robust scaling...')
            numerical_df[col] = iqr_normalization(cp.asarray(numerical_df[col].values))
        else:
            print(f'Feature {col} is of skewness: {skewness[i]}. Going with standardization...')
            numerical_df[col] = z_score_scaling(cp.asarray(numerical_df[col].values))

    return numerical_df

def scale_df(df: cd.DataFrame, numerical_features):
    scaled_df = None
    scaled = False

    # Scaling numerical features
    if len(numerical_features) > 0:
        print('Scaling numerical features...')
        scaled_df = scale_numerical_features(df, numerical_features)
        scaled = True

    return scaled_df, scaled


# Splitting into train and test data
def split_train_test(df: cd.DataFrame):
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    split_data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }

    return split_data


# Get preprocessed files
def get_preprocessed_data(file_name, target_column):
    print('Starting the preprocessing step...')

    # Get the dataframe
    df = get_df(file_name)
    temp_df = df.copy()

    # remove the target column
    temp_df.drop(target_column, axis=1, inplace=True)

    # Get the categorical and numerical columns
    categorical_cols = temp_df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = temp_df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    # Handle missing values
    temp_df = handle_na(temp_df, categorical_cols, numerical_cols)

    # Handle categorical data
    encoded_df, encoded = encode_df(temp_df, categorical_cols)

    # Scale data
    scaled_df, scaled = scale_df(df, numerical_cols)

    # Concat encoded and scaled df
    if scaled and encoded:
        temp_df = cd.concat([encoded_df, scaled_df], axis=1)
    elif scaled:
        temp_df = scaled_df
    elif encoded:
        temp_df = encoded_df
    else:
        print('Empty dataframe!')

    # Add the target column
    temp_df[target_column] = df[target_column]

    # Encode target column in case it is categorical
    if df[target_column].dtype == 'object':
        temp_df[target_column] = temp_df[target_column].astype('category')
        temp_df[target_column] = temp_df[target_column].cat.codes

    # Split the data
    split_data = split_train_test(temp_df)

    print('Preprocessing complete. Returning data...')
    return split_data
