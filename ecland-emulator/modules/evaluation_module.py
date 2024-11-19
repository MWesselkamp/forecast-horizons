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

from modules.data_module import Transform
from abc import ABC, abstractmethod
from matplotlib.colors import BoundaryNorm
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, root_mean_squared_error
from misc.helpers import r2_score_multi, anomaly_correlation, standardized_anomaly


class EvaluationBasic:
    
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

    def transform(self, use_min_max=False):

        self.Transform = Transform(use_min_max=use_min_max)
        self.Transform.compute_global_min_max(self.observations, 
                                         self.fc_numerical,
                                         self.fc_emulator)

        self.observations = self.Transform.normalise(self.observations)
        self.fc_numerical = self.Transform.normalise(self.fc_numerical)
        self.fc_emulator = self.Transform.normalise(self.fc_emulator)

    def inv_transform(self, data_array):
        return self.Transform.inv_normalise(data_array)

    def subset_samples(self):
        
        self.observations = self.observations[..., self.layer_index]
        self.fc_numerical = self.fc_numerical[..., self.variable_indices[self.layer_index]]
        self.fc_emulator = self.fc_emulator[..., self.variable_indices[self.layer_index]] 
        
        print("Shape of subset fc_emulator: ", self.fc_emulator.shape)
        print("Shape of subset observations: ", self.observations.shape)
        print("Shape of subset fc_numerical: ", self.fc_numerical.shape)

        return [self.observations,self.fc_numerical, self.fc_emulator]
    
    def slice_evluation_times(self):
        
        self.observations = self.observations[:self.maximum_evaluation_time, ...]
        self.fc_numerical = self.fc_numerical[:self.maximum_evaluation_time, ...]
        self.fc_emulator = self.fc_emulator[:self.maximum_evaluation_time,...] 
        
        print("Shape of subset fc_emulator: ", self.fc_emulator.shape)
        print("Shape of subset observations: ", self.observations.shape)
        print("Shape of subset fc_numerical: ", self.fc_numerical.shape)

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
        
        model_score = self.evaluate_emulator()
        return model_score[1:] - model_score[:-1]

    def get_skill_score(self):

        model_score, _ = self.evaluate_emulator()
        reference_score = self.evaluate_numerical()
        skill_score = 1 - model_score/reference_score
        return skill_score
    

class PointEvaluation(EvaluationBasic):


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
    
    def evaluate_numerical(self):
        eval_array = np.array([self.score(self.observations[t, :, np.newaxis], 
                                          self.fc_numerical[t, :, np.newaxis]) for t in range(self.maximum_evaluation_time)])
        
        return eval_array
    
    def evaluate_emulator(self):
        eval_array = np.array([self.score(self.observations[t, :, np.newaxis], 
                                          self.fc_emulator[t, :, np.newaxis]) for t in range(self.maximum_evaluation_time)])
        
        return eval_array, None


class EnsembleEvaluation(EvaluationBasic):

    def set_samples(self, observations, fc_numerical, fc_emulator):

        self.observations = observations
        self.fc_numerical = fc_numerical

        min_length = min(emulator.shape[0] for emulator in fc_emulator.values())
        truncated_fc_emulator = {
            key: emulator[:min_length] for key, emulator in fc_emulator.items()
            }
        stacked_fc_emulator = np.stack(list(truncated_fc_emulator.values()), axis=0)
        self.fc_emulator = stacked_fc_emulator

        print("STACKED EMUATOR SHAPE: ", self.fc_emulator.shape)

    def slice_evluation_times(self):
        
        self.observations = self.observations[:self.maximum_evaluation_time,...]
        self.fc_numerical = self.fc_numerical[:self.maximum_evaluation_time, ...]
        self.fc_emulator = self.fc_emulator[:, :self.maximum_evaluation_time,...] 
        
        print("Shape of subset fc_emulator: ", self.fc_emulator.shape)
        print("Shape of subset observations: ", self.observations.shape)
        print("Shape of subset fc_numerical: ", self.fc_numerical.shape)

        return [self.observations,self.fc_numerical, self.fc_emulator]
    
    def root_mean_squared_error(self, x_preds, x_ref):

        return np.sqrt(np.square(np.subtract(x_preds, x_ref)).mean()), None
    
    def mean_absolute_error(self, x_preds, x_ref):

        mae_mean = abs(np.subtract(x_preds, x_ref)).mean()
        mae_dispersion = abs(np.subtract(x_preds, x_ref)).std()
        return mae_mean, mae_dispersion
    
    def rmse(self, x_preds, x_ref, **kwargs):

        x_preds = self._to_tensor(x_preds)
        x_ref = self._to_tensor(x_ref)
        
        if np.isnan(x_preds).any() or np.isnan(x_ref).any():
           return torch.tensor(float('nan'))
        
        mean_score, std_score = self.root_mean_squared_error(x_preds, x_ref)

        return mean_score, std_score
    
    def mae(self, x_preds, x_ref, **kwargs):

        x_preds = self._to_tensor(x_preds)
        x_ref = self._to_tensor(x_ref)
            
        if np.isnan(x_preds).any() or np.isnan(x_ref).any():
           return torch.tensor(float('nan'))
        
        mean_score, std_score = self.mean_absolute_error(x_preds, x_ref)

        return mean_score, std_score
    
    def evaluate_numerical(self):

        eval_array = []

        for t in range(self.maximum_evaluation_time):
            score_result = self.score(self.observations[t, :, np.newaxis], self.fc_numerical[t, :, np.newaxis])

            if isinstance(score_result, tuple):
                mean_score, _ = score_result  # Unpack the tuple
            else:
                mean_score = score_result 
    
            eval_array.append(mean_score)
        
        return np.array(eval_array)
    
    def evaluate_emulator(self):

        eval_array_mean = []
        eval_array_std = []

        for t in range(self.maximum_evaluation_time):
            score_result = self.score(self.observations.squeeze()[np.newaxis, t], self.fc_emulator[:, t]) 
            
            if isinstance(score_result, tuple):
                mean_score, std_score = score_result  # Unpack the tuple
                eval_array_std.append(std_score)      # Append std_score only if it exists
            else:
                mean_score = score_result
                eval_array_std.append(None)                # Set to None if no standard deviation is provided

            eval_array_mean.append(mean_score)
        
        eval_array_mean = np.array(eval_array_mean)
        eval_array_std = np.array(eval_array_std) if eval_array_std is not None else None

        return eval_array_mean, eval_array_std
    

    
