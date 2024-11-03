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
        
        model_score = self.evaluate_emulator()
        return model_score[1:] - model_score[:-1]

    @abstractmethod
    def get_skill_score(self):

        pass
    

class PointEvaluation(EvaluationBasic):

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
    
    def evaluate_numerical(self):
        eval_array = np.array([self.score(self.observations[t, :, np.newaxis], 
                                          self.fc_numerical[t, :, np.newaxis]) for t in range(self.maximum_evaluation_time)])
        
        return eval_array
    
    def evaluate_emulator(self):
        eval_array = np.array([self.score(self.observations[t, :, np.newaxis], 
                                          self.fc_emulator[t, :, np.newaxis]) for t in range(self.maximum_evaluation_time)])
        
        return eval_array

    def get_skill_score(self):

        model_score = self.evaluate_emulator()
        reference_score = self.evaluate_numerical()
        return model_score/reference_score

class EnsembleEvaluation(EvaluationBasic):

    def subset_samples(self):
        
        self.observations = self.observations[:self.maximum_evaluation_time, :, self.layer_index]
        self.fc_numerical = self.fc_numerical[:self.maximum_evaluation_time, :, self.variable_indices[self.layer_index]]
        self.fc_emulator = np.stack(
            [emulator[:self.maximum_evaluation_time,:, self.variable_indices[self.layer_index]] for key, emulator in self.fc_emulator.items()]
        )
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
    
    def get_skill_score(self):

        model_score, _ = self.evaluate_emulator()
        reference_score = self.evaluate_numerical()
        return model_score/reference_score
    

class EvaluationModule:

    def __init__(self,
                 observations, 
                 fc_numerical, 
                 fc_emulators,
                 evaluator):

        self.observations = observations
        self.fc_numerical = fc_numerical
        self.fc_emulators = fc_emulators
        self.evaluator = evaluator

    def initialise_evaluator(self,
                             score,
                             layer_index,
                             variable_indices,
                             maximum_evaluation_time):
        
        self.score = score
        self.variable_indices = variable_indices
        self.maximum_evaluation_time = maximum_evaluation_time
        
        if self.evaluator == "ensemble":
            self.EvaluateModel = EnsembleEvaluation(score =  score,
                                        layer_index = layer_index,
                                        variable_indices = variable_indices,
                                        maximum_evaluation_time = maximum_evaluation_time)
        elif self.evaluator == "point":
            self.EvaluateModel = PointEvaluation(score =  score,
                                    layer_index = layer_index,
                                    variable_indices = variable_indices,
                                    maximum_evaluation_time = maximum_evaluation_time)
        else:
            print("Don't know evaluator")

    def evaluate(self):

        self.EvaluateModel.set_samples(observations=self.observations ,
                                fc_numerical=self.fc_numerical,
                                fc_emulator=self.fc_emulators)
        self.EvaluateModel.subset_samples()
        
        numerical_score = self.EvaluateModel.evaluate_stepwise(model = "numerical")
        emulator_score = self.EvaluateModel.evaluate_stepwise(model = "emulator")
        skill = self.EvaluateModel.get_skill_score()

        return numerical_score, emulator_score, skill

    def _scores_container(self):
        scores = {}
        skill_scores = {}
        return scores, skill_scores

    def _layers_container(self):
        self.layers = {}

    def run_ensemble_evaluation(self):

        self._layers_container()
        for layer in [0,1,2]:
            scores, skill_scores = self._scores_container()
            self.EvaluateModel = EnsembleEvaluation(score = self.score,
                                        layer_index = layer,
                                        variable_indices = self.variable_indices,
                                        maximum_evaluation_time = self.maximum_evaluation_time)
            numerical_score, emulator_score, skill = self.evaluate()
            
    def run_point_evaluation(self):

        self._layers_container()
        for layer in [0,1,2]:
            scores, skill_scores = self._scores_container()
            self.EvaluateModel = PointEvaluation(score = self.score,
                                        layer_index = layer,
                                        variable_indices = self.variable_indices,
                                        maximum_evaluation_time = self.maximum_evaluation_time)
            for mod, fc_emulator in self.fc_emulators.items():
                numerical_score, emulator_score, skill = self.evaluate()