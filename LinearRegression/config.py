from sklearn import datasets
import cudf
import os
from pathlib import Path
import cupy as cp

class GradientDescent:
    def __init__(self, params):
        self.params = params

    def predict(self, X):
        # Add bias term for prediction
        X = cp.c_[cp.ones((X.shape[0], 1)), X]
        return X.dot(self.params)

class Config:
    def __init__(self):
        cwd = os.getcwd()
        self._file_name = 'iris_data.csv'
        self._dataset_path = Path(os.path.join(cwd, f'data/{self._file_name}'))
        self._model_folder = Path(os.path.join(cwd, 'models'))

    def get_dataset(self):
        if os.path.exists(self._dataset_path):
            return cudf.read_csv(self._dataset_path)
        else:
            dataset = datasets.load_iris()
            df = cudf.DataFrame(dataset.data, columns=dataset.feature_names)
            df['target'] = dataset.target
            print(df.head())
            df.to_csv(self._dataset_path, index=False)
            return df

    def get_model_path(self, model_name):
        return Path(os.path.join(self._model_folder, model_name))



configObj = Config()
def get_cudf_iris_data():
    return configObj.get_dataset()

def get_model_path(model_name):
    return configObj.get_model_path(model_name)