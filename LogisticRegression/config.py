import kagglehub
import os
from pathlib import Path
import cudf as cd

class Config:
    def __init__(self):
        cwd = os.getcwd()
        file_name = 'insurance.csv'
        self._data_folder = Path(os.path.join(cwd, 'data'))
        self._model_folder = Path(os.path.join(cwd, 'model'))
        self._file_path = Path(os.path.join(self._data_folder, file_name))

    def get_data_folder(self):
        if os.path.exists(self._file_path):
            insurance_df = cd.read_csv(self._file_path)
        else:
            path = kagglehub.dataset_download("mirichoi0218/insurance")
            insurance_df = cd.read_csv(Path(os.path.join(path, "insurance.csv")))
            insurance_df.to_csv(self._file_path)

        return insurance_df

    def get_model_path(self, model_name):
        return Path(os.path.join(self._model_folder, model_name))



configObj = Config()

# Get the insurance data df
def get_insurance_df():
    return configObj.get_data_folder()

# Get the model
def get_model(model_name):
    return configObj.get_model_path(model_name)