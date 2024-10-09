import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
import sys 
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

class ObservationModule:

    def __init__(self, station = 'soil_TERENO_ISMN_2022.nc'):
        
        self.station = station

    def load_station(self, 
                     data_path = '/perm/dadf/HSAF_validation/in_situ_data/pre_processed_data/ismn_nc'):

        # Load measurements from one soil data station
        self.station_data = xr.open_dataset(os.path.join(data_path,self.station))

    def load_forcing(self,
                     year = "2022",
                    data_path = "/ec/res4/hpcperm/daep/ec_land_training_db/ecland_i6aj_o400_2010_2022_6h_euro.zarr"):
        
        self.forcing = xr.open_zarr(data_path).data.sel(time=slice(year)).to_dataset()

    def match_station_with_forcing(self):

        lat_a = self.station_data.lat.values  
        lat_b = self.forcing.lat.values 
        lon_a = self.station_data.lon.values  
        lon_b = self.forcing.lon.values 

        closest_indices = []
        for lat, lon in zip(lat_a, lon_a):
            print("Get Euclidean Distance.")
            #Euclidean distance
            distances = np.sqrt((lat_b - lat)**2 + (lon_b - lon)**2)
            # closest distance
            closest_idx = distances.argmin()
            
            closest_indices.append(closest_idx)
        print("Index of closest grid cell:", closest_indices)

        # select the closest grid cell to station 
        self.closest_gridcell = self.forcing.isel(x=closest_indices)
        print(self.closest_gridcell)

        return closest_indices

    def process_station_data(self, 
                             variable = 'st'):

        soil_temperature = self.station_data["st"] 

        print("Converting celsius into kelvin")
        soil_temperature = soil_temperature + 273.15

        print("Resampling to 6-hourly mean.")
        self.station_data_6hr_mean = soil_temperature.resample(time='6h').mean()
        print(self.station_data_6hr_mean.head())
        print("Length of data set:", len(self.station_data_6hr_mean['time']))

        print("Turn into tensor.")
        station_data_6hr_mean_tensor = torch.tensor(self.station_data_6hr_mean.values, dtype=torch.float32)

        return station_data_6hr_mean_tensor
    
    def plot_station_data(self):
        self.station_data_6hr_mean.plot()


        