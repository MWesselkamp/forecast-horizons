import os
import sys 
import xarray as xr
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

from typing import Tuple

import cftime
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
import zarr
from torch import tensor
from torch.utils.data import DataLoader, Dataset
from abc import ABC, abstractmethod

#PATH = '../data'
## Open up experiment configs
#with open(os.path.join(PATH, "configs.yaml")) as stream:
#    try:
#        CONFIG = yaml.safe_load(stream)
#    except yaml.YAMLError as exc:
#        print(exc)

torch.cuda.empty_cache()

class EcDataset(Dataset):
    # load the dataset
    def __init__(
        self, 
        config,
        start_yr,
        end_yr,
        initial_time
    ):
        
        self.x_idxs=config["x_slice_indices"]
        if isinstance(self.x_idxs, np.integer):
            self.x_idxs = int(self.x_idxs)
        
        self.spatial_sample_size = config["spatial_sample_size"]
            
        path=config["file_path"]
        
        self.rollout=config["roll_out"]
        self.lookback=config["lookback"]
        self.model = config["model"]
        self.lookback = config["lookback"]

        self.dyn_transform, self.dyn_inv_transform = self.select_transform(config["dyn_transform"])
        self.stat_transform, self.stat_inv_transform = self.select_transform(config["stat_transform"])
        self.prog_transform, self.prog_inv_transform = self.select_transform(config["prog_transform"])
        self.diag_transform, self.diag_inv_transform = self.select_transform(config["diag_transform"])

        self.ds_ecland = xr.open_zarr(path)

        # Apply bounding box filter?
        if config['bounding_box'] is not None:
            self._apply_bounding_box_filter(self)

        # Create time index to select the appropriate data range
        try:
            date_times = pd.to_datetime(self.ds_ecland["time"].values)
        except Exception as e:
            if 'units' in self.ds_ecland["time"].attrs:
                time_units = self.ds_ecland["time"].attrs["units"]
                date_times = pd.to_datetime(cftime.num2pydate(self.ds_ecland["time"].values, time_units))
            else:
                raise KeyError("The 'time' variable does not have a 'units' attribute and direct conversion failed.")

        # Convert the specific initial time to datetime
        initial_time_dt = pd.to_datetime(initial_time)

        self.start_index = (np.abs(date_times - initial_time_dt)).argmin()
        self.start_index_lagged = self.start_index - self.lookback 

        self.end_index = max(np.argwhere(date_times.year == int(end_yr)))[0]

        print("Temporal start index", self.start_index, "Temporal end index", self.end_index)
        print("Temporal lagged start index", self.start_index_lagged)

        self.times = np.array(date_times[self.start_index : self.end_index])
        self.len_dataset = self.end_index - self.start_index
        print("Length of dataset:", self.len_dataset)

        # Will initialise x_idxs dependent on if we use one or multiple grid cells.
        self._initialize_spatial_indices()
        print("Spatial size of data set:", self.x_size)
        print("Spatial indices of data set:", self.x_idxs)
        
        # Initialise an appropriate spatial sample size based on a given reference size
        self._initialize_spatial_sampling()

        # List of climatological time-invariant features
        self.static_feat_lst = config["clim_feats"]
        self.clim_index = [
            list(self.ds_ecland["clim_variable"]).index(x) for x in config["clim_feats"]
        ]
        # List of features that change in time
        self.dynamic_feat_lst = config["dynamic_feats"]
        self.dynamic_index = [
            list(self.ds_ecland["variable"]).index(x) for x in config["dynamic_feats"]
        ]
        # Prognostic target list
        self.targ_lst = config["targets_prog"]
        self.targ_index = [
            list(self.ds_ecland["variable"]).index(x) for x in config["targets_prog"]
        ]
        # Diagnostic target list
        self.targ_diag_lst = config["targets_diag"]
        if self.targ_diag_lst is not None:
            self.targ_diag_index = [
                list(self.ds_ecland["variable"]).index(x) for x in config["targets_diag"]
            ]
            self.y_diag_means = tensor(self.ds_ecland.data_means[self.targ_diag_index])
            self.y_diag_stdevs = tensor(self.ds_ecland.data_stdevs[self.targ_diag_index])
        else:
            self.targ_diag_index = None
        
        self.variable_size = len(self.dynamic_index) + len(self.targ_index ) + len(self.clim_index)
        
        # Define the statistics used for normalising the data
        self.x_dynamic_means = tensor(self.ds_ecland.data_means.values[self.dynamic_index])
        self.x_dynamic_stdevs = tensor(self.ds_ecland.data_stdevs.values[self.dynamic_index])
        self.x_dynamic_maxs = tensor(self.ds_ecland.data_maxs.values[self.dynamic_index])

        self.clim_means = tensor(self.ds_ecland.clim_means.values[self.clim_index])
        self.clim_stdevs = tensor(self.ds_ecland.clim_stdevs.values[self.clim_index])
        self.clim_maxs = tensor(self.ds_ecland.clim_maxs.values[self.clim_index])

        # Define statistics for normalising the targets
        self.y_prog_means = tensor(self.ds_ecland.data_means.values[self.targ_index])
        self.y_prog_stdevs = tensor(self.ds_ecland.data_stdevs.values[self.targ_index])
        self.y_prog_maxs = tensor(self.ds_ecland.data_maxs.values[self.targ_index])

        # Create time-invariant static climatological features for one or multiple grid cells.
        if isinstance(self.x_idxs, int):
            print("isint")
            clim_data = self.ds_ecland.clim_data.values[self.x_idxs]
        else:
            clim_data = self.ds_ecland.clim_data.values[slice(*self.x_idxs), :]

        if clim_data.size == 0:
            raise ValueError("Selected climatological data is empty. Check the slicing indices and data availability.")
        x_static = tensor(clim_data)
        print(x_static.shape)
        print(self.clim_index)

        try:
            x_static = x_static[:, self.clim_index]
        except IndexError:
            x_static = x_static[self.clim_index]

        print(x_static.shape)
        self.x_static_scaled = self.stat_transform(
            x_static, means=self.clim_means, stds=self.clim_stdevs, maxs=self.clim_maxs
        ).reshape(1, self.x_size, -1)

        print(self.x_static_scaled.shape)

    def get_prognostic_standardiser(self, target_list):

        # Prognostic target list
        self.targ_lst = target_list
        self.targ_index = [
            list(self.ds_ecland["variable"]).index(x) for x in target_list
        ]

        # Define statistics for normalising the targets
        y_prog_means = tensor(self.ds_ecland.data_means.values[self.targ_index])
        y_prog_stdevs = tensor(self.ds_ecland.data_stdevs.values[self.targ_index])
        y_prog_maxs = tensor(self.ds_ecland.data_maxs.values[self.targ_index])

        return y_prog_means, y_prog_stdevs, y_prog_maxs
    
    def _initialize_spatial_indices(self):
        
        # Decide if we're dealing with single grid cells or slices 
        if isinstance(self.x_idxs, int):
            self._handle_single_grid_cell()
        else:
            self._handle_multiple_grid_cells()

    def _handle_single_grid_cell(self):

        # Handle the case where x_idxs is a single integer (one grid cell)
        print("Using a single grid cell")
        self.x_size = 1
        self.lats = self.ds_ecland["lat"][self.x_idxs]
        self.lons = self.ds_ecland["lon"][self.x_idxs]

    def _handle_multiple_grid_cells(self):

        # Handle the case where x_idxs represents multiple grid cells
        print("Using multiple grid cells")
        self.x_idxs = (0, None) if "None" in self.x_idxs else tuple(self.x_idxs)
        self.x_size = len(self.ds_ecland["x"][slice(*self.x_idxs)])
        self.lats = self.ds_ecland["lat"][slice(*self.x_idxs)]
        self.lons = self.ds_ecland["lon"][slice(*self.x_idxs)]

    def _apply_bounding_box_filter(self):

        bounding_box = self.config['bounding_box']
        print("Bounding box from config:", bounding_box)
            
        ds_ecland_idx = self.ds_ecland.x.where(
            (self.ds_ecland.lat > bounding_box[0]) & 
            (self.ds_ecland.lat < bounding_box[1]) & 
            (self.ds_ecland.lon > bounding_box[2]) & 
            (self.ds_ecland.lon < bounding_box[3])
        ).compute()  # Convert the Dask array to a NumPy array
            
        # Drop coordinates that do not satisfy the condition
        ds_ecland_idx = ds_ecland_idx.dropna(dim='x', how='all')

        # Update the dataset with the filtered indices
        self.ds_ecland = self.ds_ecland.sel(x=ds_ecland_idx)

        # Check the size of the spatial dimension after filtering
        spatial_size_after_filter = self.ds_ecland.dims['x']
        print(f"Size of the spatial dimension 'x' after bounding box filtering: {spatial_size_after_filter}")
    
        # If the spatial size is zero, raise an error
        if spatial_size_after_filter == 0:
            raise ValueError("The spatial dimension 'x' is empty after applying the bounding box filter. "
                                "Please check the bounding box configuration or data.")

    def _initialize_spatial_sampling(self):

        # Helper method to initialize spatial sampling
        # If spatial sampling is active, creates container with random indices.

        if self.spatial_sample_size is not None:
            print("Activate spatial sampling of x_idxs")
            self.spatial_sample_size = self.find_spatial_sample_size(self.spatial_sample_size)
            print("Spatial sample size:", self.spatial_sample_size)
            self.chunked_x_idxs = self.chunk_indices(self.spatial_sample_size)
            self.chunk_size = len(self.chunked_x_idxs)
            print("Chunk size:", self.chunk_size)
        else:
            print("Use all x_idx from data set.")
            self.spatial_sample_size = None
            self.chunk_size = len(self.x_idxs) if isinstance(self.x_idxs, list) else 1

    def chunk_indices(self, chunk_size = 2000):

        indices = list(range(self.x_size))
        random.shuffle(indices)
        spatial_chunks = [indices[i:i + chunk_size] for i in range(0, self.x_size, chunk_size)]
        
        return spatial_chunks

    def find_spatial_sample_size(self, limit):

        for i in range(limit, self.x_size):
            if self.x_size % i == 0:
                return i

    def select_transform(self, transform_spec):

        if transform_spec == "zscoring":
            self.transform = self.z_transform
            self.inv_transform = self.inv_z_transform
        elif transform_spec == "max":
            self.transform = self.max_transform
            self.inv_transform = self.inv_max_transform
        elif transform_spec == "identity":
            self.transform = self.id_transform
            self.inv_transform = self.inv_id_transform
            
        return self.transform, self.inv_transform
            
    def id_transform(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """Placeholder identity function

        :param x: data :param mean: mean :param std: standard deviation :return:
        normalised data
        """
        return x

    def inv_id_transform(
        self, x: np.ndarray, **kwargs
    ) -> np.ndarray:
        """Placeholder inverse identity function.

        :param x_norm: normlised data :param mean: mean :param std: standard deviation
        :return: unnormalised data
        """
        return x

    def z_transform(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """Transform data with mean and stdev.

        :param x: data :param mean: mean :param std: standard deviation :return:
        normalised data
        """
        x_norm = (x - kwargs["means"]) / (kwargs["stds"] + 1e-5)
        return x_norm

    def inv_z_transform(
        self, x_norm: np.ndarray, **kwargs
    ) -> np.ndarray:
        """Inverse transform on data with mean and stdev.

        :param x_norm: normlised data :param mean: mean :param std: standard deviation
        :return: unnormalised data
        """
        x = (x_norm * (kwargs["stds"] + 1e-5)) + kwargs["means"]
        return x
    
    def max_transform(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """Transform data with max.

        :param x: data :param mean: mean :param std: standard deviation :return:
        normalised data
        """
        x_norm = (x / kwargs["maxs"])  # + 1e-5
        return x_norm
    
    def inv_max_transform(self, x_norm: np.ndarray, **kwargs) -> np.ndarray:
        """Transform data with max.

        :param x: data :param mean: mean :param std: standard deviation :return:
        normalised data
        """
        x = (x * kwargs["maxs"])  # + 1e-5
        return x

    def _slice_dataset(self):

        if isinstance(self.x_idxs, int):
            print("Select one grid cell from data")
            return self.ds_ecland.isel(
                time=slice(self.start_index, self.end_index),
                x=self.x_idxs
            ).expand_dims("x", axis=1)
        else:
            print("Select slice from data")
            return self.ds_ecland.isel(
                time=slice(self.start_index, self.end_index),
                x=slice(*self.x_idxs)
            )
        
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        
        """Load data into memory. **CAUTION ONLY USE WHEN WORKING WITH DATASET THAT FITS IN MEM**
    
        :return: static_features, dynamic_features, prognostic_targets, diagnostic_targets
        """
    
        ds_slice = self._slice_dataset()

        # Static features are already precomputed and stored in x_static_scaled
        X_static = self.x_static_scaled
        X = tensor(ds_slice['data'].isel(variable=self.dynamic_index).values)
        Y_prog = tensor(ds_slice['data'].isel(variable=self.targ_index).values) 
        Y_prog_initial_states = Y_prog[0]

        X = self.dyn_transform(X, means=self.x_dynamic_means, stds=self.x_dynamic_stdevs, maxs=self.x_dynamic_maxs)
        Y_prog = self.prog_transform(Y_prog, means=self.y_prog_means, stds=self.y_prog_stdevs,)
    
        return X_static, X, Y_prog, Y_prog_initial_states
            
    def _calculate_effective_length(self):
        return self.len_dataset - 1 - self.rollout
    
    def _get_size_factor(self):
        return self.chunk_size if self.spatial_sample_size is not None else self.x_size
    
    # number of rows in the dataset
    def __len__(self):
        
        effective_length = self._calculate_effective_length()
        size_factor = self._get_size_factor()
    
        return effective_length * size_factor
    

class EcDatasetMLP(EcDataset):

    def __getitem__(self, idx):

        # print(f"roll: {self.rollout}, lends: {self.len_dataset}, x_size: {self.x_size}")
        effective_length = self._calculate_effective_length()

        t_start_idx = (idx % effective_length) + self.start_index
        t_end_idx = (idx % effective_length) + self.start_index + self.lookback + self.rollout + 1
        
        if self.spatial_sample_size is not None:
            x_idx = [x + self.x_idxs[0] for x in self.chunked_x_idxs[(idx % self.chunk_size)]]
        else:
            x_idx = (idx % self.x_size) + self.x_idxs[0]
        
        ds_slice = tensor(
            self.ds_ecland.data[
                slice(t_start_idx, t_end_idx), :, :
            ]
        )
        print("x_idx:", x_idx)
        print("ds_slice shape:", ds_slice.shape)
        ds_slice = ds_slice[:, x_idx, :]
        print("ds_slice shape:", ds_slice.shape)

        X = ds_slice[:, :, self.dynamic_index]
        X = self.dyn_transform(X, means = self.x_dynamic_means, stds = self.x_dynamic_stdevs, maxs = self.x_dynamic_maxs)
        
        X_static = self.x_static_scaled.expand(self.rollout+self.lookback, -1, -1)
        X_static = X_static[:, x_idx, :]
        
        Y_prog = ds_slice[:, :, self.targ_index]
        Y_prog = self.prog_transform(Y_prog, means = self.y_prog_means, stds = self.y_prog_stdevs, maxs = self.y_prog_maxs)

        # Calculate delta_x update for corresponding x state
        Y_inc = Y_prog[1:,:, :] - Y_prog[:-1, :, :]
        
        if self.targ_diag_index is not None:
            
            Y_diag = ds_slice[:, :, self.targ_diag_index]
            Y_diag = self.diag_transform(Y_diag,  means = self.y_diag_means, stds = self.y_diag_stdevs,  maxs = self.y_diag_maxs)
            Y_diag = Y_diag[:-1]
                
            return X_static, X[:-1], Y_prog[:-1], Y_inc, Y_diag
        else:
            return X_static, X[:-1], Y_prog[:-1], Y_inc
        
class EcDatasetLSTM(EcDataset):

    def _slice_dataset(self):

        # Overwrite class methods to consider time lag when slicing data.
        if isinstance(self.x_idxs, int):
            print("Select one grid cell from data")
            return self.ds_ecland.isel(
                time=slice(self.start_index_lagged, self.end_index),
                x=self.x_idxs
            ).expand_dims("x", axis=1)
        else:
            print("Select slice from data")
            return self.ds_ecland.isel(
                time=slice(self.start_index_lagged, self.end_index),
                x=slice(*self.x_idxs)
            )
        
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        
        """Load data into memory. **CAUTION ONLY USE WHEN WORKING WITH DATASET THAT FITS IN MEM**
    
        :return: static_features, dynamic_features, prognostic_targets, diagnostic_targets
        """
    
        ds_slice = self._slice_dataset()

        # Static features are already precomputed and stored in x_static_scaled
        X_static = self.x_static_scaled
        X = tensor(ds_slice['data'].isel(variable=self.dynamic_index).values)
        Y_prog = tensor(ds_slice['data'].isel(variable=self.targ_index).values) 
        Y_prog_initial_states = Y_prog[:self.lookback]

        X = self.dyn_transform(X, means=self.x_dynamic_means, stds=self.x_dynamic_stdevs, maxs=self.x_dynamic_maxs)
        Y_prog = self.prog_transform(Y_prog, means=self.y_prog_means, stds=self.y_prog_stdevs,)
    
        return X_static, X, Y_prog, Y_prog_initial_states
    
class EcDatasetXGB(EcDataset):

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        
        """Load data into memory. **CAUTION ONLY USE WHEN WORKING WITH DATASET THAT FITS IN MEM**
    
        :return: static_features, dynamic_features, prognostic_targets, diagnostic_targets
        """
    
        ds_slice = self._slice_dataset()
        
        # Static features are already precomputed and stored in x_static_scaled
        X_static = self.x_static_scaled
        X = tensor(ds_slice['data'].isel(variable=self.dynamic_index).values)
        Y_prog = tensor(ds_slice['data'].isel(variable=self.targ_index).values)
        Y_prog_initial_states = Y_prog[0]

        X = self.dyn_transform(X, means=self.x_dynamic_means, stds=self.x_dynamic_stdevs, maxs=self.x_dynamic_maxs)
        Y_prog = self.prog_transform(Y_prog, means=self.y_prog_means, stds=self.y_prog_stdevs,)

        return X_static, X[:-1], Y_prog[:-1], Y_prog_initial_states
    
