"""
Script contains a class for forecasting with each emulator type.
Author: @mariekewesselkamp
"""
import numpy as np
import os
import torch
import xgboost as xgb
import time
import torch

from torch import tensor
from misc.models import *
from modules.data_module import *
from abc import ABC, abstractmethod
from tests.test_model import *

if torch.cuda.is_available():
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"
print(DEVICE)

torch.cuda.empty_cache()

class MCdropoutForecastModule(ABC):

    """
    Initialize class with Pytorch lighning or DLMC xgb model type.
    """
    
    def __init__(self, hpars ,config, closest_grid_cell, my_device = None):
        
        self.hpars = hpars
        self.config = config
        self.config['x_slice_indices'] = closest_grid_cell

        if my_device is None:
            self.my_device = DEVICE
        else:
            self.my_device = my_device

    def load_test_data(self):
        
        self.x_static, self.x_met, self.y_prog, self.y_prog_initial_state = self.dataset.load_data() 
        self._set_forcing_device()

        return self.x_static, self.x_met, self.y_prog, self.y_prog_initial_state
    
    def match_indices(self, target_variables):
        matching_indices = [i for i, val in enumerate(self.dataset.targ_lst) if val in target_variables]
        return matching_indices
    
    def initialise_dataset(self, initial_time):

        self.dataset = EcDatasetMLP(self.config,
                    self.config["test_start"],
                    self.config["test_end"],
                    initial_time)

        self.input_clim_dim = len(self.dataset.static_feat_lst)
        self.input_met_dim = len(self.dataset.dynamic_feat_lst)
        self.input_state_dim = len(self.dataset.targ_lst)
        self.output_dim = len(self.dataset.targ_lst)  # Number of output targets
        self.output_diag_dim = 0 # len(dataset.targ_diag_lst)

    def transform_initial_vector(self, station_data, matching_indices):

        y_prog_means = self.dataset.y_prog_means[matching_indices]
        y_prog_stds = self.dataset.y_prog_stdevs[matching_indices]

        variable_data = self.dataset.prog_transform(station_data, 
                                                means=y_prog_means, 
                                                stds=y_prog_stds)
        
        return variable_data
    
    def transform_station_data(self, station_data, target_variable_list):

        y_prog_means, y_prog_stds, _ = self.dataset.get_prognostic_standardiser(target_variable_list)

        variable_data = self.dataset.prog_transform(station_data, 
                                                means=y_prog_means, 
                                                stds=y_prog_stds)
        
        return variable_data
    
    def load_model(self):

        self.model = MLPregressor(input_clim_dim = self.input_clim_dim,
                             input_met_dim = self.input_met_dim,
                             input_state_dim = self.input_state_dim,
                             hidden_dim = self.hpars['hidden_dim'],
                             output_dim = self.output_dim,
                             output_diag_dim = self.output_diag_dim,
                             batch_size=self.config["batch_size"],
                             learning_rate=self.hpars['learning_rate'], 
                             lookback = None, 
                             rollout=self.config["roll_out"], 
                             dropout=self.hpars['dropout'], 
                             weight_decay = self.hpars['weight_decay'], 
                             loss = self.hpars['loss'],
                             device = 'cpu', 
                             targets = self.config["targets_prog"],
                             db_path = self.config["file_path"])

        path_to_checkpoint = self.config["model_path"]
        use_checkpoint = os.listdir(path_to_checkpoint)[-1]
        path_to_best_checkpoint = os.path.join(path_to_checkpoint, use_checkpoint)  # trainer.checkpoint_callback.best_model_path
        print("Load model from checkpoint: ", path_to_best_checkpoint)
        checkpoint = torch.load(path_to_best_checkpoint, map_location=torch.device('cpu'), weights_only=True)
        torch.set_float32_matmul_precision("high")
        self.model.load_state_dict(checkpoint['state_dict'])

        return self.model

        
    def _set_initial_conditions(self, initial_conditions):

        return initial_conditions
    
        
    def _perturb_prediction(self, original_vector):

        if self.predictions_perturbation is not None:
            return torch.normal(mean=original_vector, std=0.05*abs(original_vector))
        else:
            return original_vector

    def _set_forcing_device(self):

        print(f"Model to device: {self.my_device}")
        self.x_static, self.x_met, self.y_prog = self.x_static.to(self.my_device), self.x_met.to(self.my_device), self.y_prog.to(self.my_device)
        self.model.to(self.my_device)

    def _create_prediction_container(self):
        return torch.full_like(self.y_prog, float('nan'))

    def run_forecast(self, 
                     initial_conditions,
                     mc_samples = 100 ,
                     predictions_perturbation = None):

        """
        Args:
            Takes as input the output of self.get_test_data. Suboptimal.
        """
        self.predictions_perturbation = predictions_perturbation

        self.y_prog_prediction = self._create_prediction_container()
        self.y_prog_prediction[:initial_conditions.shape[0]] = initial_conditions
        #self.y_prog_prediction[0,...] = self._perturb_initial_conditions()
        print("Initialised prediction.")
    
        start_time = time.time()

        mc_trajectories = self.step_forecast(mc_samples=mc_samples)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("--- %s seconds ---" % (elapsed_time))
        print("--- %s minutes ---" % (elapsed_time/60))
        
        return self.y_prog, mc_trajectories
    
    def backtransformation(self, y_vector):
        
        return self.dataset.prog_inv_transform(y_vector, means = self.dataset.y_prog_means, 
                                                                stds = self.dataset.y_prog_stdevs, 
                                                                maxs = self.dataset.y_prog_maxs)


    def step_forecast(self, mc_samples):

        """
        Perform iterative forecasting with MC dropout uncertainty propagation.
        
        Args:
            mc_samples (int): Number of MC samples to estimate uncertainty.
        
        Returns:
            torch.Tensor: Forecasted trajectories of shape (mc_samples, time_steps, ...)
        """
        
        print("Setting model to evaluation mode")
        self.model.eval()  # model starts in evaluation mode
        
        # Create a tensor to store all sample trajectories
        mc_trajectories = torch.zeros((mc_samples, self.y_prog_prediction.shape[0], *self.y_prog_prediction.shape[1:]))
        
        with torch.no_grad():
            
            for sample_idx in range(mc_samples):  # Iterate over MC samples

                y_temp = self.y_prog_prediction.clone()  # Copy initial conditions

                if sample_idx % 100 == 0 :
                        print(f"On step {sample_idx}...")
                
                #perturb initial states
                y_temp[0, ...] = self._perturb_prediction(y_temp[0, ...])

                for time_idx in range(y_temp.shape[0] - 1):
                    
                    # Activate MC Dropout by setting the model to train mode temporarily
                    self.model.train()
                    logits = self.model.forward(self.x_static, self.x_met[[time_idx]], y_temp[[time_idx]])
                    
                    prediction = y_temp[time_idx, ...] + logits.squeeze()
                    #prediction = self._perturb_prediction(prediction) if time_idx == 1 else prediction
                    y_temp[time_idx + 1, ...] = prediction

                # Store the full trajectory for thismc sample
                mc_trajectories[sample_idx] = y_temp

        self.model.eval()  

        return mc_trajectories  

