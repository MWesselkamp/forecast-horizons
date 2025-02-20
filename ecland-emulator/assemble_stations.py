import torch
import os
import sys
import matplotlib.pyplot as plt
import yaml

parent_dir = os.path.abspath('..')
sys.path.append(parent_dir)

from modules.observation_module import *

SCRIPT_DIR = os.getcwd()
sys.path.append(os.path.dirname(SCRIPT_DIR))
print(SCRIPT_DIR) 

Station = ObservationModule(network = 'soil_SMOSMANIA_ISMN_2022.nc', 
                                    station = 'Condom' ,
                                    variable = 'st',
                                    depth=[0.05, 0.2, 0.3]) # Initialise the Observation Module with the default Station (Gevenich)
        

Station.load_station(years = [2021, 2022]) # Load two years of station data for lookback slicing
Station.load_forcing() # Load forcing for matching data base with station data
closest_grid_cell = Station.match_station_with_forcing() # Match the station with clostest grid cell and extract the index of the grid cell

#soil_type = Station.station_physiography()
Station.process_station_data()

print("Finished.")