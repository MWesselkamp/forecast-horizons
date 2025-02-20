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

station_ids = ['Condom', 'Villevielle', 'LaGrandCombe', 'Narbonne', 'Urgons',
                    'CabrieresdAvignon', 'Savenes', 'PeyrusseGrande','Sabres', 
                    'Mouthoumet','Mejannes-le-Clap',  'CreondArmagnac', 'SaintFelixdeLauragais',
                    'Mazan-Abbaye', 'LezignanCorbieres']

Stations = ObservationModule(network = 'soil_SMOSMANIA_ISMN_2022.nc', 
                                    station = station_ids ,
                                    variable = 'st',
                                    depth=[0.05, 0.2, 0.3],
                                    years = [2021, 2022]) # Initialise the Observation Module with the default Station (Gevenich)
        

Stations.load_station() # Load two years of station data for lookback slicing
Stations.load_forcing() # Load forcing for matching data base with station data
closest_grid_cell = Stations.match_station_with_forcing() # Match the station with clostest grid cell and extract the index of the grid cell

#soil_type = Station.station_physiography()
Stations.process_station_data()

print(Stations.closest_indices_dict["Condom"])
print("Finished.")

