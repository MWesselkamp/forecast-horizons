import torch
import os
import sys
import matplotlib.pyplot as plt

parent_dir = os.path.abspath('..')
sys.path.append(parent_dir)

from data_module import *
from evaluation_module import *
from forecast_module import *
from observation_module import *
from visualisation_module import *

from misc.helpers import *
from tests.test_model import *

set_global_seed(42)

SCRIPT_DIR = os.getcwd()
sys.path.append(os.path.dirname(SCRIPT_DIR))
print(SCRIPT_DIR) 

path_to_plots = 'ecland-emulator/plots/'
path_to_results = 'ecland-emulator/results/'

#EX_CONFIG = load_config(config_path = '../../configs/smosmania_st.yaml')
EX_CONFIG = load_config(config_path = 'configs/tereno_st.yaml')

network =  EX_CONFIG['network'] #'soil_TERENO_ISMN_2022.nc'#'soil_SMOSMANIA_ISMN_2022.nc' # 'soil_TERENO_ISMN_2022.nc'
network_name = network.split('_')[1]
station = EX_CONFIG['station'] # 'Lahas'

variable = EX_CONFIG['variable'] 
depth = EX_CONFIG['depth']  # [0.05, 0.2, 0.3]

years = EX_CONFIG['years']
models = EX_CONFIG['models']# , 'xgb'

maximum_leadtime = EX_CONFIG['maximum_leadtime'] # medium range, ten days
tolerance = EX_CONFIG['tolerance']
score = EX_CONFIG['score']

print("Network: ", network)
print("Station: ", station)
print("Variable: ", variable)
print("Depth: ", depth)
print("Years: ", years)
print("Models: ", models)
print("Initial time: ", EX_CONFIG['initial_time'])

def perturb_initial_state(initial_state, perturbation_factor = 0.001):
    random_perturbation = torch.rand(initial_state.size()) * 2 - 1  # random values between -1 and 1
    return initial_state * (1 + perturbation_factor * random_perturbation)

def create_perturbation_ensemble(initial_state, size = 50):
    torch.manual_seed(42)
    perturbation_ensemble = [perturb_initial_state(initial_state) for i in range(size)]
    return perturbation_ensemble

if __name__ == "__main__":

    Station = ObservationModule(network = network, 
                                station = station,
                                variable = variable,
                                depth=depth) # Initialise the Observation Module with the default Station (Gevenich)

    Station.load_station(years = years) # Load two years of station data for lookback slicing
    Station.load_forcing() # Load forcing for matching data base with station data
    closest_grid_cell = Station.match_station_with_forcing() # Match the station with clostest grid cell and extract the index of the grid cell
    Station.process_station_data() # specify path_to_plots, if you want to visualise

    dynamic_features_ensemble = {}
    dynamic_features_prediction_ensemble = {}

    for mod in ["mlp", "lstm", "xgb"]:

        if mod == 'mlp':
            print('mlp')
            CONFIG = load_config(config_path = 'configs/mlp_emulator.yaml')
            HPARS = load_hpars(use_model = 'ecland-emulator/mlp')
            ForecastModel = ForecastModuleMLP(hpars=HPARS, config=CONFIG)    
        elif mod == 'lstm':
            CONFIG = load_config(config_path = 'configs/lstm_emulator.yaml')
            HPARS = load_hpars(use_model = 'ecland-emulator/lstm')
            ForecastModel = ForecastModuleLSTM(hpars=HPARS, config=CONFIG)
        elif mod == 'xgb':
            CONFIG = load_config(config_path = 'configs/xgb_emulator.yaml')
            HPARS = None
            ForecastModel = ForecastModuleXGB(hpars=HPARS, config=CONFIG)

        CONFIG['x_slice_indices'] = closest_grid_cell # adjust the index of the grid cell in the config file before initialising the models

        dataset = ForecastModel.initialise_dataset(EX_CONFIG['initial_time'])
        model = ForecastModel.load_model()
        x_static, x_met, y_prog, y_prog_initial_state = ForecastModel.load_test_data(dataset)  
        print("INITIAL STATE SHAPE:", y_prog_initial_state.shape)

        station_data = Station.slice_station_data(lookback=CONFIG["lookback"],
                                        t_0=EX_CONFIG['initial_time'])
        matching_indices = Station.match_indices(dataset=dataset,
                                                target_variables=EX_CONFIG['targets_eval'])
        y_prog_initial_state[..., matching_indices] = station_data[:y_prog_initial_state.shape[0]]
        
        perturbed_ensemble = create_perturbation_ensemble(y_prog_initial_state)

        ensemble_prediction = []
        for i in range(len(perturbed_ensemble)):
            matching_indices = Station.match_indices(dataset=dataset,
                                                target_variables=EX_CONFIG['targets_prog'])
            initial_vector =  Station.transform_station_data(station_data = perturbed_ensemble[i],
                                                    target_variable_list = EX_CONFIG['targets_prog'])

            dynamic_features, dynamic_features_prediction = ForecastModel.run_forecast(initial_conditions=initial_vector, 
                                                                                    initial_conditions_perturbation=None,
                                                                                    predictions_perturbation = None)
            dynamic_features, dynamic_features_prediction = ForecastModel.backtransformation()
            ensemble_prediction.append(dynamic_features_prediction)

        dynamic_features_ensemble[mod] = dynamic_features
        dynamic_features_prediction_ensemble[mod] = ensemble_prediction

    print(dynamic_features_prediction_ensemble['mlp'][0].shape)
    print(dynamic_features_prediction_ensemble['lstm'][0].shape)
    print(dynamic_features_prediction_ensemble['xgb'][0].shape)

    ensemble_dict = dynamic_features_prediction_ensemble
    for key, ensemble_prediction in ensemble_dict.items():
        ensemble_dict[key] = np.stack(ensemble_prediction).squeeze()[:,:maximum_leadtime,:]

    print(ensemble_dict['mlp'].shape)
    print(ensemble_dict['lstm'].shape)
    print(ensemble_dict['xgb'].shape)

    fc_numerical = dynamic_features_ensemble['lstm'][0].squeeze()[:maximum_leadtime]
    observations = station_data[:maximum_leadtime]

    PlotStation = VisualisationSingle(
                 network = EX_CONFIG["network"], 
                 station = EX_CONFIG["station"], 
                 variable = EX_CONFIG["variable"], 
                 maximum_leadtime = EX_CONFIG["maximum_leadtime"], 
                 score = EX_CONFIG["score"],
                 doy_vector = Station.doy_vector,
                 evaluation = "ens", 
                 path_to_plots = path_to_plots
    )

    PlotStation.plot_initial_state_perturbation(ensemble_dict, fc_numerical, observations, EX_CONFIG['ensemble_size'])