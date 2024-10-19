import os
import time
import sys
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib.dates as mdates
import cftime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
print(SCRIPT_DIR)

from matplotlib.colors import BoundaryNorm
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, root_mean_squared_error
from helpers import r2_score_multi, anomaly_correlation, standardized_anomaly


class EvaluationModule:
    
    def __init__(self, 
                score, 
                layer_index,
                variable_indices,
                maximum_evaluation_time,
                path_to_results = None):
        
        self.init_score = score
        self.layer_index = layer_index
        self.variable_indices = variable_indices # select variable index based on layer
        self.maximum_evaluation_time = maximum_evaluation_time
        self.path_to_results = path_to_results
        
        score_methods = {
            'rmse': self.rmse,
            'mae': self.mae,
            'r2': self.r2,
            'acc': self.acc,
            'scaled_anom': self.scaled_anom
        }

        if score in score_methods:
            print("Evaluation with", score)
            self.score = score_methods[score]
        else:
            print("Don't know score!")

    def set_samples(self, observations, fc_numerical, fc_emulator):

        self.observations = observations
        self.fc_numerical = fc_numerical
        self.fc_emulator = fc_emulator

    def subset_samples(self):
        
        self.observations = self.observations[:self.maximum_evaluation_time, :, self.layer_index]
        self.fc_numerical = self.fc_numerical[:self.maximum_evaluation_time, :, self.variable_indices[self.layer_index]]
        self.fc_emulator = self.fc_emulator[:self.maximum_evaluation_time,:, self.variable_indices[self.layer_index]]

    def rmse(self, x_preds, x_ref, **kwargs):

        if not isinstance(x_preds, torch.Tensor):
            x_preds = torch.tensor(x_preds)
        if not isinstance(x_ref, torch.Tensor):
            x_ref = torch.tensor(x_ref)
        
        if np.isnan(x_preds).any() or np.isnan(x_ref).any():
           return torch.tensor(float('nan'))
        
        return root_mean_squared_error(x_preds, x_ref)


    def mae(self, x_preds, x_ref, **kwargs):

        if not isinstance(x_preds, torch.Tensor):
            x_preds = torch.tensor(x_preds)
        if not isinstance(x_ref, torch.Tensor):
            x_ref = torch.tensor(x_ref)
            
        if np.isnan(x_preds).any() or np.isnan(x_ref).any():
           return torch.tensor(float('nan'))
        
        return mean_absolute_error(x_preds, x_ref)

    def r2(self, x_preds, x_ref, **kwargs):
        return r2_score_multi(x_preds, x_ref)

    def acc(self, x_preds, x_ref, **kwargs):    
        return anomaly_correlation(x_preds, x_ref, kwargs["clim_mu"])

    def scaled_anom(self, x, **kwargs):    
        
        anom = standardized_anomaly(x, kwargs["clim_mu"], kwargs["clim_std"])
        anom = np.mean(anom)

        return anom

    def evaluate_total(self, y_prog_prediction):
        eval_array = np.array([self.score(self.observations[t, ...], 
                                          y_prog_prediction[t, ...]) for t in range(self.maximum_evaluation_time)])
        return eval_array

    def evaluate_target(self, model = "numerical"):
        
        y_prog_prediction = self.fc_numerical if model == "numerical" else self.fc_emulator

        eval_array = np.array([self.score(self.observations[t, :, np.newaxis], 
                                          y_prog_prediction[t, :, np.newaxis]) for t in range(self.maximum_evaluation_time)])
        
        return eval_array
    
    def get_score_increments(self):
        
        model_score = self.evaluate_target(model="emulator")
        return model_score[1:] - model_score[:-1]

    def get_skill_score(self):

        model_score = self.evaluate_target(model="emulator")
        reference_score = self.evaluate_target(model="numerical")
        return model_score/reference_score