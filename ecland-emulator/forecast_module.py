"""
Script contains a class for forecasting with each emulator type.
"""
import numpy as np
import os
import torch
import xgboost as xgb
import time
import torch

from torch import tensor
from models import *
from data_module import *
from abc import ABC, abstractmethod
from tests.test_model import *

if torch.cuda.is_available():
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"
print(DEVICE)

torch.cuda.empty_cache()

class ForecastModule(ABC):

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
        
    @abstractmethod
    def initialise_dataset(self):

        pass

    @abstractmethod 
    def load_model(self):

        pass

    def load_test_data(self, dataset):
        
        self.dataset = dataset
        self.x_static, self.x_met, self.y_prog = self.dataset.load_data() 
        self._set_forcing()

        return self.x_static, self.x_met, self.y_prog 
        
    def set_initial_conditions(self):

        if self.initial_conditions is None:
            return self.y_prog[0,...]
        else:
            return self.initial_conditions
        
    def perturb_prediction(self, original_vector):

        if self.perturbation is not None:
            return torch.normal(mean=original_vector, std=self.perturbation)
        else:
            return original_vector


    def _set_forcing(self):

        print(f"Model to device: {self.my_device}")
        self.x_static, self.x_met, self.y_prog = self.x_static.to(self.my_device), self.x_met.to(self.my_device), self.y_prog.to(self.my_device)
        self.model.to(self.my_device)

    @abstractmethod
    def step_forecast(self):
        
        pass

    def _create_prediction_container(self):
        return torch.full_like(self.y_prog, float('nan'))

    def run_forecast(self, initial_conditions = None, perturbation = None):

        """
        Args:
            Takes as input the output of self.get_test_data. Suboptimal.
        """
        self.perturbation = perturbation
        self.initial_conditions = initial_conditions

        self.y_prog_prediction = self._create_prediction_container()
        
        self.y_prog_prediction[0,...] = self.set_initial_conditions()
        print("Initialised prediction.")
    
        start_time = time.time()

        self.step_forecast()
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("--- %s seconds ---" % (elapsed_time))
        print("--- %s minutes ---" % (elapsed_time/60))

        print("y_prog shape: ",self.y_prog.shape)
        print("y_prog_prediction shape: ", self.y_prog_prediction.shape)  
        
        return self.y_prog, self.y_prog_prediction
    
    def backtransformation(self):
        
        print("Backtransforming")
        # make class object for succeeding access.
        y_prog_prediction = self.dataset.prog_inv_transform(self.y_prog_prediction, 
                                                                           means = self.dataset.y_prog_means, 
                                                                           stds = self.dataset.y_prog_stdevs, 
                                                                           maxs = self.dataset.y_prog_maxs)
        y_prog = self.dataset.prog_inv_transform(self.y_prog, 
                                                                means = self.dataset.y_prog_means, 
                                                                stds = self.dataset.y_prog_stdevs, 
                                                                maxs = self.dataset.y_prog_maxs)

        return y_prog, y_prog_prediction


class ForecastModuleMLP(ForecastModule):

    def initialise_dataset(self):

        dataset = EcDatasetMLP(self.config,
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

    def step_forecast(self):
        
        print("Setting model to evaluation mode")
        self.model.eval()
        with torch.no_grad():
                
            for time_idx in range(self.y_prog_prediction.shape[0]-1):
                    
                if time_idx % 1000 == 0:
                    print(f"on step {time_idx}...")

                logits = self.model.forward(self.x_static, self.x_met[[time_idx]], self.y_prog_prediction[[time_idx]])
                prediction = self.y_prog_prediction[time_idx, ...] + logits.squeeze()
                # Perturn only on step after intitialisation
                prediction = self.perturb_prediction(prediction) if time_idx == 1 else prediction
                self.y_prog_prediction[time_idx+1, ...] = prediction


class ForecastModuleLSTM(ForecastModule):

    def initialise_dataset(self):

        dataset = EcDatasetLSTM(self.config,
                    self.config["test_start"],
                    self.config["test_end"])

        self.input_clim_dim = len(dataset.static_feat_lst)
        self.input_met_dim = len(dataset.dynamic_feat_lst)
        self.input_state_dim = len(dataset.targ_lst)
        self.output_dim = len(dataset.targ_lst)  # Number of output targets
        self.output_diag_dim = 0 # len(dataset.targ_diag_lst)

        return dataset
    
    def _process_data(self):
        return self.x_static, self.x_met[self.config["lookback"]:], self.y_prog[self.config["lookback"]:]
    
    def load_test_data(self, dataset):
        
        self.dataset = dataset
        self.x_static, self.x_met, self.y_prog = self.dataset.load_data() 
        self._set_forcing()

        return self._process_data()
    
    def load_model(self):

        print("Set up model")
        self.model = LSTMregressor(input_clim_dim = self.input_clim_dim,
                              input_met_dim = self.input_met_dim,
                              input_state_dim = self.input_state_dim,
                              lookback = self.config["lookback"], 
                              lead_time = self.config["roll_out"], 
                              device = self.config["device"], 
                              batch_size=self.config["batch_size"],
                              learning_rate=self.hpars['learning_rate'],
                              num_layers_en = self.hpars['num_layers_en'], 
                              num_layers_de = self.hpars['num_layers_de'], 
                              hidden_size_en = self.hpars['hidden_size_en'], 
                              hidden_size_de = self.hpars['hidden_size_de'], 
                              dropout = self.hpars['dropout'], 
                              weight_decay = self.hpars['weight_decay'],
                              loss = self.hpars['loss'], 
                              use_dlogits = True,
                              transform = self.config["prog_transform"], # prognostic transform informs model
                              db_path = self.config["file_path"])

        path_to_checkpoint = self.config["model_path"]
        use_checkpoint = os.listdir(path_to_checkpoint)[-1]
        path_to_best_checkpoint = os.path.join(path_to_checkpoint, use_checkpoint)  # trainer.checkpoint_callback.best_model_path
        print("Load model from checkpoint: ", path_to_best_checkpoint)
        checkpoint = torch.load(path_to_best_checkpoint, map_location=torch.device('cpu'), weights_only=True)
        torch.set_float32_matmul_precision("high")
        self.model.load_state_dict(checkpoint['state_dict'])
        print("Matched keys")

        return self.model
    
    def handle_hindcast(self, skip = True):
        self.skip_hindcast = skip

    def step_forecast(self):
        
        test_forecast_shapes(self.model, self.x_static, self.x_met, self.y_prog) # Test data structures before forecasting

        preds = self.model.forecast(self.x_static, self.x_met, self.y_prog)
        self.y_prog_prediction[self.model.lookback:, ...] = preds.squeeze(0)

        if self.skip_hindcast:
            self.y_prog_prediction = self.y_prog_prediction[self.model.lookback:, ...]
            self.y_prog = self.y_prog[self.model.lookback:, ...]

class ForecastModuleXGB(ForecastModule):

    def initialise_dataset(self):

        dataset = EcDatasetXGB(self.config,
                    self.config["test_start"],
                    self.config["test_end"])

        self.input_clim_dim = len(dataset.static_feat_lst)
        self.input_met_dim = len(dataset.dynamic_feat_lst)
        self.input_state_dim = len(dataset.targ_lst)
        self.output_dim = len(dataset.targ_lst)  # Number of output targets
        self.output_diag_dim = 0 # len(dataset.targ_diag_lst)

        return dataset

    def load_model(self):

        print("Set up model: XGB")
        self.model = xgb.Booster()
        self.model.load_model(os.path.join(self.config['model_path'], 'xgb_model.bin'))

        return self.model
    
    def load_test_data(self, dataset):
        
        self.dataset = dataset
        self.x_static, self.x_met, self.y_prog , _ = dataset.load_data() 
        self._set_forcing()

        return self.x_static, self.x_met, self.y_prog 
    
    def _set_forcing(self):
        self.x_static, self.x_met, self.y_prog = self.x_static.to(self.my_device), self.x_met.to(self.my_device), self.y_prog.to(self.my_device)

    def _create_prediction_container(self):
        return np.full_like(self.y_prog, np.nan, dtype=float)
    
    def perturb_prediction(self, original_vector):

        if self.perturbation is not None:
            return np.random.normal(loc=original_vector, scale=self.perturbation)
        else:
            return original_vector
    
    def step_forecast(self):
        
        for time_idx in range(self.y_prog_prediction.shape[0]-1):
                
            if time_idx % 10 == 0:
                print(f"on step {time_idx}...")    
            input_data = np.concatenate((self.x_static, self.x_met[[time_idx]], self.y_prog_prediction[[time_idx]]), axis=-1)
            input_data = input_data.squeeze(0)
            step_predictors = xgb.DMatrix(input_data)

            logits = self.model.predict(step_predictors)
            prediction = self.y_prog_prediction[time_idx, ...] + logits.squeeze()
            # Perturn only on step after intitialisation
            prediction = self.perturb_prediction(prediction) if time_idx == 1 else prediction
            self.y_prog_prediction[time_idx+1, ...] = prediction
