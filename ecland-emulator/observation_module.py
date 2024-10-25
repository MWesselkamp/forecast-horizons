import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys 
import torch
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

class ObservationModule:

    def __init__(self, 
                 network = 'soil_TERENO_ISMN_2022.nc', 
                 station = None, 
                 variable = 'st', 
                 depth = None):
        
        self.network = network
        self.station = station
        self.variable = variable
        self.depth = depth

        self.network_name = self.network.split('_')[1]

        self._process_variable = self._process_temperature if variable == 'st' else self._process_soilmoisture

    def load_station(self, 
                     years = None,
                     data_path = '/perm/dadf/HSAF_validation/in_situ_data/pre_processed_data/ismn_nc'):
        self.years = years
        self.data_path = data_path
        if (years is None) | (len(years) == 1):
            self._load_single_year()
        else:
            self._load_multiple_years()

    def _load_single_year(self):
        
        self.network_data = xr.open_dataset(os.path.join(self.data_path,self.network))
        if self.station is not None:
            print("Select station: ", self.station)
            self.network_data = self.network_data.sel(station_id = [self.station])
        if self.depth is not None:
            print("Select depth: ", self.depth)
            self.network_data = self.network_data.sel(depth = self.depth)

    def _load_multiple_years(self):
        
        file_names = [re.sub(r'(\d{4})', str(year), self.network) for year in self.years]
        file_paths = [os.path.join(self.data_path, file_name) for file_name in file_names]

        self.network_data = xr.open_mfdataset(file_paths, combine='by_coords')

        if self.station is not None:
            print("Select station: ", self.station)
            self.network_data = self.network_data.sel(station_id = [self.station])
        if self.depth is not None:
            print("Select depth: ", self.depth)
            self.network_data = self.network_data.sel(depth = self.depth)

    def load_forcing(self,
                    data_path = "/ec/res4/hpcperm/daep/ec_land_training_db/ecland_i6aj_o400_2010_2022_6h_euro.zarr"):
        
        match = re.search(r'\d{4}', self.network)

        if match:
            year = match.group(0)
            print(f"Extracted year: {year}")
        else:
            print("No year found.")

        self.forcing = xr.open_zarr(data_path).data.sel(time=slice(year)).to_dataset()

    def match_station_with_forcing(self):

        lat_a = self.network_data.lat.values  
        lat_b = self.forcing.lat.values 
        lon_a = self.network_data.lon.values  
        lon_b = self.forcing.lon.values 

        closest_indices = []
        for lat, lon in zip(lat_a, lon_a):
            
            distances = np.sqrt((lat_b - lat)**2 + (lon_b - lon)**2) #Euclidean distance
            closest_idx = distances.argmin() # closest distance
            
            closest_indices.append(closest_idx)

        self.closest_gridcell = self.forcing.isel(x=closest_indices)  # select the closest grid cell to station 
        print("Matched station with grid cell: ", closest_indices[0])

        return closest_indices[0]
    
    def _process_temperature(self):
        print("Converting celsius into kelvin")
        return self.variable_data + 273.15
    
    def _process_soilmoisture(self):
        return self.variable_data 

    def process_station_data(self, path_to_plots = None):

        self.variable_data = self.network_data[self.variable] 
        self.variable_data = self._process_variable()

        print("Resampling to 6-hourly mean.")
        self.variable_data = self.variable_data.resample(time='6h').mean()
        
        if path_to_plots is not None:
            self._plot_station_data(save_to=path_to_plots)

        #self.variable_data = self.variable_data.interpolate_na(dims = "depth", 
        #                                                       method="nearest")

        #print("Length of data set:", len(self.variable_data['time']))
        #variable_data_tensor = torch.tensor(self.variable_data.values, dtype=torch.float32)

        #return variable_data_tensor
    
    def slice_station_data(self, lookback = 0, t_0 = '2022-01-01T00:00:00'):

        t_0_datetime = pd.to_datetime(t_0)

        t_0_index = self.variable_data.time.get_index('time').get_loc(t_0_datetime)
        print("Selecting from index:", t_0_index)
        t_lookback_index = t_0_index - lookback
        print("Subtract lookback for new index:", t_lookback_index)

        variable_data_slice = self.variable_data.isel(time=slice(t_lookback_index, None))
        variable_data_tensor = torch.tensor(variable_data_slice.values, dtype=torch.float32)

        return variable_data_tensor
    
    def _plot_station_data(self, save_to):

        self.variable_data.plot()
        plt.savefig(os.path.join(save_to, f'{self.network_name}_image_plot.pdf'))
        plt.show()

    def match_indices(self, dataset, target_variables):
        self.dataset = dataset
        matching_indices = [i for i, val in enumerate(self.dataset.targ_lst) if val in target_variables]
        return matching_indices
    
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



        