import numpy as np
import os
from datetime import datetime
def create_scenario_folder(directory_path, new_folder_name):

    try:
        new_folder_path = os.path.join(directory_path, new_folder_name)

        if not os.path.exists(new_folder_path):
            os.makedirs(new_folder_path)
            print(f"Created scenario folder: {new_folder_path}")
        else:
            print(f"Folder already exists: {new_folder_path}")

        return new_folder_path

    except Exception as e:
        print(f"An error occurred: {str(e)}")


def standardize(y):

    ys = (y - np.mean(y))/np.std(y)

    return ys

def sample_ensemble_member(ensemble):

    index = np.random.randint(0, ensemble.shape[0], 1)
    control = ensemble[index, :]
    ensemble = np.delete(ensemble, index, axis=0)

    return control, ensemble, index
