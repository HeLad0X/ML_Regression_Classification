from config import get_df
import cupy as cp
import cudf as cd
from cuml.model_selection import train_test_split


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

# Function to get the skewness of the data
def get_skewness(df: cd.DataFrame) -> [float]:
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
    """
    numerical_df = df[numerical_features]
    skewness = get_skewness(numerical_df)
    for i, col in enumerate(numerical_df.columns):
        if skewness[i] > 1.8 or skewness[i] < -1.8:
            print(f'Feature {col} is of skewness: {skewness[i]}. Going with log transformation scaling...')
            numerical_df[col] = iqr_normalization(cp.asarray(numerical_df[col].values))
        elif skewness[i] > 0.75 or skewness[i] < -0.75:
            print(f'Feature {col} is of skewness: {skewness[i]}. Going with robust scaling...')
            numerical_df[col] = iqr_normalization(cp.asarray(numerical_df[col].values))
        else:
            print(f'Feature {col} is of skewness: {skewness[i]}. Going with standardization...')
            numerical_df[col] = z_score_scaling(cp.asarray(numerical_df[col].values))

    return numerical_df

# Splitting into train and test data
def split_train_test(df: cd.DataFrame):
    print('Splitting into training and testing data...')
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
    """
    Steps to take:
    1. Get the files
    2. Handle missing values
    3. Handle categorical data
    4. Feature scaling
    5. Data splitting
    6. Class imbalance
    """
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
    encoded_df = None
    scaled_df = None
    scaled = False
    encoded = False

    if len(categorical_cols) > 0:
        print('Encoding categorical features...')
        encoded_df = handle_cat_features(temp_df, categorical_cols)
        encoded = True

    # Scaling numerical features
    if len(numerical_cols) > 0:
        print('Scaling numerical features...')
        scaled_df = scale_numerical_features(df, numerical_cols)
        scaled = True

    # Concat encoded and scaled df
    if scaled and encoded:
        temp_df = cd.concat([encoded_df, scaled_df], axis=1)
    elif scaled:
        temp_df = scaled_df

    elif encoded:
        temp_df = encoded_df

    # Add the target column
    temp_df[target_column] = df[target_column]

    # Split the data
    split_df = split_train_test(temp_df)

    print('Preprocessing complete. Returning data...')
    return split_df

