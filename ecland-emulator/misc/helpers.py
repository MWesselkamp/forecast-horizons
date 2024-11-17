import numpy as np
import torch
import os
import yaml

from sklearn.metrics import r2_score

def r2_score_multi(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Calculated the r-squared score between 2 arrays of values

    :param y_pred: predicted array
    :param y_true: "truth" array
    :return: r-squared metric
    """
    return r2_score(y_pred.flatten(), y_true.flatten())

def anomaly(y_hat, climatology):
    
    return y_hat - climatology
    
def standardized_anomaly(y_hat, climatology, climatology_std):
    
    return anomaly(forecast, climatology)/climatology_std

def anomaly_correlation(forecast, reference, climatology):
    
    anomaly_f = anomaly(forecast, climatology)
    anomaly_r = anomaly(reference, climatology)
    
    msse = np.mean(anomaly_f * anomaly_r)
    act = np.sqrt(np.mean(anomaly_f**2) * np.mean(anomaly_r**2))
    
    return msse/act

def set_global_seed(seed):

    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(config_path):

    with open(config_path) as stream:
        try:
            config = yaml.safe_load(stream)
            print(f"Opening {config_path} for experiment configuration.")
        except yaml.YAMLError as exc:
            print(exc)
    return config

def load_hpars(use_model):

    with open(os.path.join(use_model, "hparams.yaml"), "r") as stream:
        hpars = yaml.safe_load(stream)
    print(hpars)
    return hpars


def print_nested_dict_structure(d, indent=0):
    """
    Recursively prints the structure of a nested dictionary.

    Parameters:
    - d (dict): The dictionary to analyze.
    - indent (int): Current indentation level for pretty-printing.
    """
    for key, value in d.items():
        print(" " * indent + f"{key}: ", end="")
        if isinstance(value, dict):
            print("{")
            print_nested_dict_structure(value, indent + 2)
            print(" " * indent + "}")
        elif isinstance(value, list):
            print(f"List[{len(value)}]")
        elif isinstance(value, tuple):
            print(f"Tuple[{len(value)}]")
        elif isinstance(value, (int, float, str, bool)):
            print(type(value).__name__)
        elif value is None:
            print("None")
        else:
            print(type(value).__name__)