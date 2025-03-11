import kagglehub
import os
from pathlib import Path
import cudf as cd
import cupy as cp


class GradientAscent:
    def __init__(self, params):
        self.params = params

    def _sigmoid(self, z):
        return 1 / (1 + cp.exp(-z))
 
    def predict(self, X):
        # Add bias term for prediction
        X = cp.c_[cp.ones((X.shape[0], 1)), X]
        z = X.dot(self.params)
        preditcion = self._sigmoid(z)

        return preditcion
    

class GradientDescent:
    def __init__(self, params):
        self.params = params

    def predict(self, X):
        # Add bias term for prediction
        X = cp.c_[cp.ones((X.shape[0], 1)), X]
        return X.dot(self.params)

class Config:
    def __init__(self):
        # cwd = os.path.dirname(os.getcwd()) # For using in pycharm
        cwd = os.getcwd()
        self._data_folder = Path(os.path.join(cwd, 'files_models/data'))
        self._model_folder = Path(os.path.join(cwd, 'files_models/models'))

    def get_data_folder(self, file_name):
        print(f'Getting dataframe for {file_name}...')
        _file_path = Path(os.path.join(self._data_folder, file_name))
        if os.path.exists(_file_path):
            print(f'File found! Returning dataframe...')
            df = cd.read_csv(_file_path)
        else:
            print(f'File not found! Downloading file...')
            path = None
            if file_name == 'diabetes.csv':
                path = kagglehub.dataset_download("nancyalaswad90/review")

            else:
                path = kagglehub.dataset_download("uciml/iris")

            df = cd.read_csv(Path(os.path.join(path, file_name)))
            df.to_csv(_file_path, index=False)
            print('Download completed. Returning dataframe...')

        return df

    def get_model_path(self, model_name):
        return Path(os.path.join(self._model_folder, model_name))



configObj = Config()

# Get the insurance files_models df
def get_df(file_name):
    return configObj.get_data_folder(file_name)

# Get the model
def get_model_path(model_name):
    return configObj.get_model_path(model_name)