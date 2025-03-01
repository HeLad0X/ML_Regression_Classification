from cuml.model_selection import train_test_split
from config import get_cudf_iris_data
import cudf as cd
import cupy as cp


# Function to fill empty values
def fill_empty_values(df: cd.DataFrame) -> cd.DataFrame:
    for col in df.columns[:-1]:
        df[col].fillna(df[col].mean(), inplace=True)

    return df


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


# Function to scale the data
def scale_df(df: cd.DataFrame) -> cd.DataFrame:
    """
    Both z-score scaling and IQR scaling are used for scaling the data.
    They are also handling outliers by capping to max and min values.
    """
    skewness = get_skewness(df)
    for i, col in enumerate(df.columns):
        if skewness[i] > 0.5:
            df[col] = iqr_normalization(cp.asarray(df[col].values))
        else:
            df[col] = z_score_scaling(cp.asarray(df[col].values))

    return df

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

def get_preprocessed_data(scale_data = True):
    df = get_cudf_iris_data()
    temp_df = df.copy()
    temp_df.drop(['target'], axis=1, inplace=True)

    temp_df = fill_empty_values(temp_df)
    if scale_data:
        temp_df = scale_df(temp_df)
    temp_df['target'] = df['target']
    split_data = split_train_test(temp_df)

    return split_data
