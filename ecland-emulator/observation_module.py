import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
import sys 
import torch
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

class ObservationModule:

    def __init__(self, network = 'soil_TERENO_ISMN_2022.nc', station = None, variable = 'st', depth = None):
        
        self.network = network
        self.station = station
        self.variable = variable
        self.depth = depth

        self.network_name = self.network.split('_')[1]

        self._process_variable = self._process_temperature if variable == 'st' else self._process_soilmoisture

    def load_station(self, 
                     data_path = '/perm/dadf/HSAF_validation/in_situ_data/pre_processed_data/ismn_nc'):

        # Load measurements from one soil data station
        self.network_data = xr.open_dataset(os.path.join(data_path,self.network))
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

    def process_station_data(self):

        self.variable_data = self.network_data[self.variable] 
        self.variable_data = self._process_variable()

        print("Resampling to 6-hourly mean.")
        self.station_data_6hr_mean = self.variable_data.resample(time='6h').mean()

        print("Length of data set:", len(self.station_data_6hr_mean['time']))
        self.station_data_6hr_mean_tensor = torch.tensor(self.station_data_6hr_mean.values, dtype=torch.float32)

        return self.station_data_6hr_mean_tensor
    
    def plot_station_data(self, save_to):

        self.station_data_6hr_mean.plot()
        plt.savefig(os.path.join(save_to, f'{self.network_name}_image_plot.pdf'))
        plt.show()

    def transform_station_data(self, dataset, target_variables):

        self.matching_indices = [i for i, val in enumerate(dataset.targ_lst) if val in target_variables]
        station_data_transformed = dataset.prog_transform(self.station_data_6hr_mean_tensor, 
                                                          means=dataset.y_prog_means[self.matching_indices], 
                                                          stds=dataset.y_prog_stdevs[self.matching_indices])
        
        return station_data_transformed



        