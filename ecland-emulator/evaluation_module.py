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

from abc import ABC, abstractmethod
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
            'RMSE': self.rmse,
            'MAE': self.mae
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

    @abstractmethod
    def subset_samples(self):
        
        pass

    def _to_tensor(self, x):

        if not isinstance(x, torch.Tensor):
            return torch.tensor(x)
        else:
            return x
        
    @abstractmethod
    def rmse(self, x_preds, x_ref, **kwargs):

        pass

    @abstractmethod
    def mae(self, x_preds, x_ref, **kwargs):

        pass

    @abstractmethod
    def evaluate_stepwise(self, model = "numerical"):
        
        pass
    
    def get_score_increments(self):
        
        model_score = self.evaluate_stepwise(model="emulator")
        return model_score[1:] - model_score[:-1]

    def get_skill_score(self):

        model_score = self.evaluate_stepwise(model="emulator")
        reference_score = self.evaluate_stepwise(model="numerical")
        return model_score/reference_score
    

class PointEvaluation(EvaluationModule):

    def subset_samples(self):
        
        self.observations = self.observations[:self.maximum_evaluation_time, :, self.layer_index]
        self.fc_numerical = self.fc_numerical[:self.maximum_evaluation_time, :, self.variable_indices[self.layer_index]]
        self.fc_emulator = self.fc_emulator[:self.maximum_evaluation_time,:, self.variable_indices[self.layer_index]]

        print("Shape of subset: ", self.fc_emulator.shape)

    def rmse(self, x_preds, x_ref, **kwargs):

        x_preds = self._to_tensor(x_preds)
        x_ref = self._to_tensor(x_ref)
        
        if np.isnan(x_preds).any() or np.isnan(x_ref).any():
           return torch.tensor(float('nan'))
        
        return root_mean_squared_error(x_preds, x_ref)


    def mae(self, x_preds, x_ref, **kwargs):

        x_preds = self._to_tensor(x_preds)
        x_ref = self._to_tensor(x_ref)
            
        if np.isnan(x_preds).any() or np.isnan(x_ref).any():
           return torch.tensor(float('nan'))
        
        return mean_absolute_error(x_preds, x_ref)
    
    def evaluate_stepwise(self, model = "numerical"):
        
        y_prog_prediction = self.fc_numerical if model == "numerical" else self.fc_emulator

        eval_array = np.array([self.score(self.observations[t, :, np.newaxis], 
                                          y_prog_prediction[t, :, np.newaxis]) for t in range(self.maximum_evaluation_time)])
        
        return eval_array


class EnsembleEvaluation(EvaluationModule):

    def subset_samples(self):
        
        self.observations = self.observations[:self.maximum_evaluation_time, :, self.layer_index]
        self.fc_numerical = self.fc_numerical[:self.maximum_evaluation_time, :, self.variable_indices[self.layer_index]]
        self.fc_emulator = np.stack(
            [emulator[:self.maximum_evaluation_time,:, self.variable_indices[self.layer_index]] for key, emulator in self.fc_emulator.items()]
        )
        print("Shape of subset: ", self.fc_emulator.shape)

    def root_mean_squared_error(self, x_preds, x_ref):

        return np.sqrt(np.square(np.subtract(x_preds, x_ref)).mean())
    
    def mean_absolute_error(self, x_preds, x_ref):

        return abs(np.subtract(x_preds, x_ref)).mean()
    
    def rmse(self, x_preds, x_ref, **kwargs):

        x_preds = self._to_tensor(x_preds)
        x_ref = self._to_tensor(x_ref)
        
        if np.isnan(x_preds).any() or np.isnan(x_ref).any():
           return torch.tensor(float('nan'))
        
        return self.root_mean_squared_error(x_preds, x_ref)
    
    def mae(self, x_preds, x_ref, **kwargs):

        x_preds = self._to_tensor(x_preds)
        x_ref = self._to_tensor(x_ref)
            
        if np.isnan(x_preds).any() or np.isnan(x_ref).any():
           return torch.tensor(float('nan'))
        
        return self.mean_absolute_error(x_preds, x_ref)
    
    def evaluate_stepwise(self, model):

        if model == "numerical":
            eval_array = np.array([self.score(self.observations[t, :, np.newaxis], 
                                          self.fc_numerical[t, :, np.newaxis]) for t in range(self.maximum_evaluation_time)])
        else:
            eval_array = np.array([self.score(self.observations.squeeze()[np.newaxis, t], 
                                          self.fc_emulator[:, t]) for t in range(self.maximum_evaluation_time)])
        
        return eval_array