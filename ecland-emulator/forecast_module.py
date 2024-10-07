"""
Script contains a class for forecasting with each emulator type.
"""
import numpy as np
import os
import pandas as pd
import pytorch_lightning as pl
import torch
import random
import xarray as xr
import re
import xgboost as xgb
import time
import torch

from torch import tensor
from models import *
from data_module import *

if torch.cuda.is_available():
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"
print(DEVICE)

torch.cuda.empty_cache()

class ForecastModule:

    """
    Initialize class with Pytorch lighning or DLMC xgb model type.
    """
    
    def __init__(self, hpars ,config, my_device = None):
        
        self.hpars = hpars
        self.config = config

        if my_device is None:
            self.my_device = DEVICE
        else:
            self.my_device = my_device
        
    def initialise_dataset(self):

        datamodule = NonLinRegDataModule(self.config)
        dataset = EcDataset(self.config,
                    self.config["test_start"],
                    self.config["test_end"])

        self.input_clim_dim = len(dataset.static_feat_lst)
        self.input_met_dim = len(dataset.dynamic_feat_lst)
        self.input_state_dim = len(dataset.targ_lst)
        self.output_dim = len(dataset.targ_lst)  # Number of output targets
        self.output_diag_dim = 0 # len(dataset.targ_diag_lst)

        return dataset

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
        checkpoint = torch.load(path_to_best_checkpoint, map_location=torch.device('cpu'))
        torch.set_float32_matmul_precision("high")
        self.model.load_state_dict(checkpoint['state_dict'])

        return self.model

    def load_test_data(self, dataset):

        """
        Args:
            Takes a dataset created with data_module.EcDataset. 
            Chunk_idx specifies the spatial indices to use for prediction.
            
        Needs to be called before step forecast! 
        """
        
        X_static, X_met, Y_prog = dataset.load_data() 
        self.dataset = dataset

        return X_static, X_met, Y_prog

    def get_climatology(self):    

        climatology = xr.open_dataset(self.config['climatology_path'])
        climatology_mu = climatology['clim_6hr_mu'].sel(variable=self.config['targets_prog']).values
        climatology_mu = np.tile(climatology_mu, (2, 1, 1))
        climatology_std = climatology['clim_6hr_std'].sel(variable=self.config['targets_prog']).values
        climatology_std = np.tile(climatology_std, (2, 1, 1))
        return climatology_mu, climatology_std
        
    
    def step_forecast(self, X_static, X_met, Y_prog):

        """
        Args:
            Takes as input the output of self.get_test_data. Suboptimal.
        """
        
        if isinstance(self.model, pl.LightningModule):
            
            X_static, X_met, Y_prog = X_static.to(self.my_device), X_met.to(self.my_device), Y_prog.to(self.my_device)
            self.model.to(self.my_device)
        
            Y_prog_prediction = Y_prog.clone()
            
            print("Setting model to evaluation mode")
            self.model.eval()

        else:
            print("Don't know model type.")
        
        start_time = time.time()
        
        if self.config['model'] == 'mlp':
            
            with torch.no_grad():
                
                for time_idx in range(Y_prog_prediction.shape[0]-1):
                    
                    if time_idx % 1000 == 0:
                        print(f"on step {time_idx}...")

                    logits = self.model.forward(X_static, X_met[[time_idx]], Y_prog_prediction[[time_idx]])
                    Y_prog_prediction[time_idx+1, ...] = Y_prog_prediction[time_idx, ...] + logits.squeeze()
            

        end_time = time.time()
        elapsed_time = end_time - start_time
        print("--- %s seconds ---" % (elapsed_time))
        print("--- %s minutes ---" % (elapsed_time/60))

        print(Y_prog.shape)
        print(Y_prog_prediction.shape)
        
        self.dynamic_features = Y_prog
        self.dynamic_features_prediction = tensor(Y_prog_prediction) if self.config['model'] == 'xgb' else Y_prog_prediction
                
        print("Backtransforming")
        # make class object for succeeding access.
        self.dynamic_features_prediction = self.dataset.prog_inv_transform(self.dynamic_features_prediction.cpu(), means = self.dataset.y_prog_means, stds = self.dataset.y_prog_stdevs, maxs = self.dataset.y_prog_maxs).numpy()
        self.dynamic_features = self.dataset.prog_inv_transform(self.dynamic_features.cpu(), means = self.dataset.y_prog_means, stds = self.dataset.y_prog_stdevs, maxs = self.dataset.y_prog_maxs).numpy()

        return self.dynamic_features, self.dynamic_features_prediction

    def save_forecast(self, save_to):
        """
        Args:
            save_to: Specify results path.
        Not applicable for global data set sizes, also not recommended to use this with continental size.
        """
        
        print("Backtransforming")
        self.dynamic_features_prediction = self.dataset.dynamic_feat_scalar.inv_transform(self.dynamic_features_prediction.cpu()).numpy()
        self.dynamic_features = self.dataset.dynamic_feat_scalar.inv_transform(self.dynamic_features.cpu()).numpy()

        obs_array = xr.DataArray(self.dynamic_features,
                                  coords={"time": self.dataset.full_times[:self.dynamic_features.shape[0]],
                                          "location": self.dataset.ds_ecland['x'],
                                          "variable": self.dataset.dynamic_feat_lst},
                                     dims=["time", "x", "variable"])
        obs_array.to_netcdf(os.path.join(save_to, f'fc_ecland_{self.config['model']}.nc'))
        print("saved reference data to: ", os.path.join(save_to, f'fc_ecland_{self.config['model']}.nc'))

        preds_array = xr.DataArray(self.dynamic_features_prediction,
                                  coords={"time": self.dataset.full_times[:self.dynamic_features_prediction.shape[0]],
                                          "location": self.dataset.ds_ecland['x'],
                                          "variable": self.dataset.dynamic_feat_lst},
                                      dims=["time", "x", "variable"])
        preds_array.to_netcdf(os.path.join(save_to, f'fc_ailand_{self.config['model']}.nc'))
        print("saved forecast to: ", os.path.join(save_to, f'fc_ecland_{self.config['model']}.nc'))

        return self.dynamic_features, self.dynamic_features_prediction
