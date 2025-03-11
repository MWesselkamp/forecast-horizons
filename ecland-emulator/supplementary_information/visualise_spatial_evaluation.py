
import torch
import os
import sys
import matplotlib.pyplot as plt
import yaml
import argparse

parent_dir = os.path.abspath('..')
sys.path.append(parent_dir)

from modules.data_module import *
from modules.evaluation_module import *
from modules.forecast_module import *
from modules.observation_module import *
from modules.visualisation_module import *

from misc.helpers import *
from tests.test_model import *

set_global_seed(42)

SCRIPT_DIR = os.getcwd()
sys.path.append(os.path.dirname(SCRIPT_DIR))
print(SCRIPT_DIR) 

def nullable_string(val):
    return None if not val else val

parser = argparse.ArgumentParser(description="Load different config files for training.")
parser.add_argument('--maximum_leadtime', type=int, nargs='?', const=56, help='Specify maximum lead time (6-hourly). Default: two weeks')
parser.add_argument('--variable', type=nullable_string, help='Specify variable from st or sm.')
args = parser.parse_args()

MAXIMUM_LEADTIME = args.maximum_leadtime
VARIABLE = args.variable
PATH_TO_PLOTS = 'ecland-emulator/plots/'
PATH_TO_RESULTS = 'ecland-emulator/results'
EX_STATION = "Condom"

EX_CONFIG = load_config(config_path = f"configs/smosmania_{VARIABLE}.yaml")

print("MAXIMUM LEADTIME:", MAXIMUM_LEADTIME)


if __name__ == "__main__":

    Station = ObservationModule(network = EX_CONFIG['network'], 
                                    station = EX_STATION ,
                                    variable = VARIABLE,
                                    depth=EX_CONFIG['depth'],
                                    years = EX_CONFIG['years']) # Initialise the Observation Module with the default Station (Gevenich)
        

    Station.load_station() # Load two years of station data for lookback slicing
    Station.load_forcing() # Load forcing for matching data base with station data
    Station.match_station_with_forcing() # Match the station with clostest grid cell and extract the index of the grid cell
    Station.process_station_data() # specify path_to_plots, if you want to visualise
    Station.slice_station_data(lookback=0,t_0=EX_CONFIG['initial_time'])

    DOY_VECTOR = Station.doy_vector[:MAXIMUM_LEADTIME]

    #use_stations = ['Condom', 'Villevielle', 'LaGrandCombe', 'Narbonne', 'Urgons',
    #                'CabrieresdAvignon', 'Savenes', 'PeyrusseGrande','Sabres', 
    #                'Mouthoumet','Mejannes-le-Clap',  'CreondArmagnac', 'SaintFelixdeLauragais',
    #                'Mazan-Abbaye', 'LezignanCorbieres']

    #use_stations = ['Condom', 'Villevielle', 'LaGrandCombe', 'Narbonne', 'SaintFelixdeLauragais','PeyrusseGrande',
    #                'Mouthoumet', 'Mejannes-le-Clap', 'CreondArmagnac']
    if VARIABLE == 'sm':
        use_stations = ['Savenes', 'Mouthoumet', 'Mazan-Abbaye', 'LezignanCorbieres', 
                        'LaGrandCombe', 'CreondArmagnac', 'Urgons', 'Condom']
    elif VARIABLE =='st':
        use_stations = ['Condom', 'Villevielle', 'LaGrandCombe', 'Narbonne', 'Urgons',
                    'CabrieresdAvignon', 'Savenes', 'PeyrusseGrande','Sabres', 
                    'Mouthoumet','Mejannes-le-Clap',  'CreondArmagnac', 'SaintFelixdeLauragais',
                    'Mazan-Abbaye', 'LezignanCorbieres']
    else:
        print("Don't know which stations to use")

    stations_dict = {}
    for station in use_stations:
        with open(f"ecland-emulator/results/SMOSMANIA_{station}_2022_{VARIABLE}_ensemble.yaml", 'r') as f:
            layers = yaml.load(f, Loader=yaml.UnsafeLoader)
        stations_dict[station] = layers

    forecast_dict = {}
    for station in use_stations:
        with open(f"ecland-emulator/results/SMOSMANIA_{station}_2022_{VARIABLE}_ensemble_fc.yaml", 'r') as f:
            layers_fc = yaml.load(f, Loader=yaml.UnsafeLoader)
        forecast_dict[station] = layers_fc

    PlotStations = VisualisationMany(
                    network = EX_CONFIG["network"], 
                    station = "all", 
                    variable = VARIABLE, 
                    maximum_leadtime = MAXIMUM_LEADTIME, 
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
