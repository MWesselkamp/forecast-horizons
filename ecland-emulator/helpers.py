import numpy as np
import os

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