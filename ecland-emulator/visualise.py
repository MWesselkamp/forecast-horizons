
import torch
import os
import sys
import matplotlib.pyplot as plt
import yaml
import argparse

parent_dir = os.path.abspath('..')
sys.path.append(parent_dir)

from data_module import *
from evaluation_module import *
from forecast_module import *
from observation_module import *
from visualisation_module import *

from helpers import *
from tests.test_model import *

set_global_seed(42)

SCRIPT_DIR = os.getcwd()
sys.path.append(os.path.dirname(SCRIPT_DIR))
print(SCRIPT_DIR) 

PATH_TO_PLOTS = 'ecland-emulator/plots/'
EX_STATION = "Condom"
EX_CONFIG = load_config(config_path = 'configs/smosmania_st.yaml')

Station = ObservationModule(network = EX_CONFIG['network'], 
                                station = EX_STATION ,
                                variable = EX_CONFIG['variable'],
                                depth=EX_CONFIG['depth']) # Initialise the Observation Module with the default Station (Gevenich)
    

Station.load_station(years = EX_CONFIG['years']) # Load two years of station data for lookback slicing
Station.load_forcing() # Load forcing for matching data base with station data
closest_grid_cell = Station.match_station_with_forcing() # Match the station with clostest grid cell and extract the index of the grid cell
Station.process_station_data() # specify path_to_plots, if you want to visualise
station_data = Station.slice_station_data(lookback=0,t_0=EX_CONFIG['initial_time'])

DOY_VECTOR = Station.doy_vector

use_stations = ['Condom', 'Villevielle', 'LaGrandCombe', 'Narbonne', 'Urgons',
                'CabrieresdAvignon', 'Savenes', 'PeyrusseGrande','Sabres', 
                'Mouthoumet','Mejannes-le-Clap',  'CreondArmagnac', 'SaintFelixdeLauragais']

stations_dict = {}
for station in use_stations:
    with open(f"ecland-emulator/results/SMOSMANIA_{station}_2022_st_ensemble.yaml", 'r') as f:
        layers = yaml.load(f, Loader=yaml.UnsafeLoader)
    stations_dict[station] = layers

forecast_dict = {}
for station in use_stations[:7]:
    with open(f"ecland-emulator/results/SMOSMANIA_{station}_2022_st_ensemble_fc.yaml", 'r') as f:
        layers_fc = yaml.load(f, Loader=yaml.UnsafeLoader)
    forecast_dict[station] = layers_fc

PlotStations = VisualisationMany(
                 network = EX_CONFIG["network"], 
                 station = "all", 
                 variable = EX_CONFIG["variable"], 
                 maximum_leadtime = EX_CONFIG["maximum_leadtime"], 
                 score = EX_CONFIG["score"],
                 doy_vector = DOY_VECTOR,
                 evaluation = "ens", 
                 path_to_plots = PATH_TO_PLOTS
)

PlotStations.assemble_scores(stations_dict)
PlotStations.plot_scores()
PlotStations.plot_horizons(EX_CONFIG["tolerance"])
PlotStations.plot_skill_scores()

PlotStations.assemble_forecasts(forecast_dict)
PlotStations.plot_forecasts()