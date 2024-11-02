# %% [markdown]
# # Evaluating ECLand Emulator on ISMN data
# 
# In adjusting the flags, we can choose the network, station, soil variable and layer for evaluation. Here we run this example with soil temperature.

# %%
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

def nullable_string(val):
    return None if not val else val

parser = argparse.ArgumentParser(description="Load different config files for training.")
parser.add_argument('--station', type=nullable_string, help='Specify .yaml file from same directory.')
args = parser.parse_args()

STATION = args.station

PATH_TO_PLOTS = 'ecland-emulator/plots'
PATH_TO_RESULTS = 'ecland-emulator/results'

EX_CONFIG = load_config(config_path = 'configs/smosmania_st.yaml')

print("Network: ", EX_CONFIG['network'])
print("Station: ", STATION)
print("Variable: ", EX_CONFIG['variable'] )
print("Depth: ", EX_CONFIG['depth'])
print("Years: ", EX_CONFIG['years'])
print("Models: ", EX_CONFIG['models'])
print("Initial time: ", EX_CONFIG['initial_time'])


def evaluate_ensemble(observations, 
                fc_numerical, 
                fc_emulators,
                score, 
                maximum_leadtime):
        layers = {}
        for layer in [0,1,2]:
            EvaluateModel = EnsembleEvaluation(score =  score,
                                                layer_index = layer,
                                                variable_indices = matching_indices,
                                                maximum_evaluation_time = maximum_leadtime)

            EvaluateModel.set_samples(observations=observations,
                                fc_numerical=fc_numerical,
                                fc_emulator=fc_emulators)
            EvaluateModel.subset_samples()
            numerical_score = EvaluateModel.evaluate_numerical()
            ensemble_score = EvaluateModel.evaluate_emulator()
            ensemble_skill = EvaluateModel.get_skill_score()

            scores = {}
            scores_dispersion = {}
            skill_scores = {}
            scores["ECLand"] = numerical_score
            scores["Emulators"] = ensemble_score[0]
            scores_dispersion["Emulators"] = ensemble_score[1]
            skill_scores["Emulators"] = ensemble_skill

            layers[f"layer{layer}"] = {}
            layers[f"layer{layer}"]["scores"] = scores
            layers[f"layer{layer}"]["scores_dispersion"] = scores_dispersion
            layers[f"layer{layer}"]["skill_scores"] = skill_scores

        return layers


def evaluate_point(observations, 
                fc_numerical, 
                fc_emulators,
                score, 
                maximum_leadtime):
        
        layers = {}
        for layer in [0,1,2]:
            EvaluateModel = PointEvaluation(score =  score,
                                        layer_index = layer,
                                        variable_indices = matching_indices,
                                        maximum_evaluation_time = maximum_leadtime)
            scores = {}
            skill_scores = {}
            for mod, fc_emulator in fc_emulators.items():

                EvaluateModel.set_samples(observations=observations,
                                    fc_numerical=fc_numerical,
                                    fc_emulator=fc_emulator)
                EvaluateModel.subset_samples()
                numerical_score = EvaluateModel.evaluate_numerical()
                emulator_score = EvaluateModel.evaluate_emulator()
                skill_score = EvaluateModel.get_skill_score()

                scores["ECLand"] = numerical_score
                scores[mod] = emulator_score
                skill_scores[mod] = skill_score
                
            layers[f"layer{layer}"] = {}
            layers[f"layer{layer}"]["scores"] = scores
            layers[f"layer{layer}"]["skill_scores"] = skill_scores

        return layers


if __name__ == "__main__":

    Station = ObservationModule(network = EX_CONFIG['network'], 
                                station = STATION ,
                                variable = EX_CONFIG['variable'],
                                depth=EX_CONFIG['depth']) # Initialise the Observation Module with the default Station (Gevenich)
    

    Station.load_station(years = EX_CONFIG['years']) # Load two years of station data for lookback slicing
    Station.load_forcing() # Load forcing for matching data base with station data
    closest_grid_cell = Station.match_station_with_forcing() # Match the station with clostest grid cell and extract the index of the grid cell
    Station.process_station_data() # specify path_to_plots, if you want to visualise

    dynamic_features_dict = {}
    dynamic_features_prediction_dict = {}

    for mod in EX_CONFIG['models']:

        # initialise experiement and setup model with config file
        #CONFIG, HPARS, ForecastModel = setup_experiment(model = mod)
        if mod == 'mlp':
            print('mlp')
            CONFIG = load_config(config_path = 'configs/mlp_emulator_node.yaml')
            HPARS = load_hpars(use_model = 'ecland-emulator/mlp')
            ForecastModel = ForecastModuleMLP(hpars=HPARS, config=CONFIG)    
        elif mod == 'lstm':
            CONFIG = load_config(config_path = 'configs/lstm_emulator_node.yaml')
            HPARS = load_hpars(use_model = 'ecland-emulator/lstm')
            ForecastModel = ForecastModuleLSTM(hpars=HPARS, config=CONFIG)
        elif mod == 'xgb':
            CONFIG = load_config(config_path = 'configs/xgb_emulator_node.yaml')
            HPARS = None
            ForecastModel = ForecastModuleXGB(hpars=HPARS, config=CONFIG)

        CONFIG['x_slice_indices'] = closest_grid_cell # adjust the index of the grid cell in the config file before initialising the models

        dataset = ForecastModel.initialise_dataset(EX_CONFIG['initial_time'])
        model = ForecastModel.load_model()
        x_static, x_met, y_prog, y_prog_initial_state = ForecastModel.load_test_data(dataset)  

        station_data = Station.slice_station_data(lookback=CONFIG["lookback"],
                                    t_0=EX_CONFIG['initial_time'])
        matching_indices = Station.match_indices(dataset=dataset,
                                                target_variables=EX_CONFIG['targets_eval'])

        y_prog_initial_state[..., matching_indices] = station_data[:y_prog_initial_state.shape[0]]
        
        matching_indices = Station.match_indices(dataset=dataset,
                                                target_variables=EX_CONFIG['targets_prog'])
        print("MATCHING INDICES: ", matching_indices)
        initial_vector =  Station.transform_station_data(station_data = y_prog_initial_state, 
                                                    target_variable_list = EX_CONFIG['targets_prog'])

        dynamic_features, dynamic_features_prediction = ForecastModel.run_forecast(initial_conditions=initial_vector)
        dynamic_features, dynamic_features_prediction = ForecastModel.backtransformation()

        dynamic_features_dict[mod] = dynamic_features
        dynamic_features_prediction_dict[mod] = dynamic_features_prediction


    fc_numerical = dynamic_features_dict['mlp']
    fc_emulators = dynamic_features_prediction_dict
    matching_indices = Station.match_indices(dataset=dataset,
                                                target_variables=EX_CONFIG['targets_eval'])
    

    layers_ensemble = evaluate_ensemble(observations=station_data,
                    fc_numerical=fc_numerical,
                    fc_emulators=fc_emulators,
                    score=EX_CONFIG['score'] ,
                    maximum_leadtime= EX_CONFIG['maximum_leadtime'])

    save_to =os.path.join(PATH_TO_RESULTS, 
                          f"{EX_CONFIG['network'].split('_')[1]}_{STATION}_{max(EX_CONFIG['years'])}_{EX_CONFIG['variable']}_ensemble.yaml")
    print("Write layers to path:", save_to)

    with open(save_to, 'w') as f:
        yaml.dump(layers_ensemble, f, indent=4)

    layers_point = evaluate_point(observations=station_data,
                    fc_numerical=fc_numerical,
                    fc_emulators=fc_emulators,
                    score=EX_CONFIG['score'] ,
                    maximum_leadtime= EX_CONFIG['maximum_leadtime'])
    
    save_to =os.path.join(PATH_TO_RESULTS, 
                          f"{EX_CONFIG['network'].split('_')[1]}_{STATION}_{max(EX_CONFIG['years'])}_{EX_CONFIG['variable']}_point.yaml")
    print("Write layers to path:", save_to)

    with open(save_to, 'w') as f:
        yaml.dump(layers_point, f, indent=4)

    #PointPlots = VisualisationModule(network = EX_CONFIG['network'],
    #                                station = EX_CONFIG['station'],
    #                                variable = EX_CONFIG['variable'],
    #                                maximum_leadtime=EX_CONFIG['maximum_leadtime'],
    #                                doy_vector = Station.doy_vector,
    #                                evaluation = "poi", # ens
    #                                path_to_plots=PATH_TO_PLOTS)
    #
    #EnsemblePlots = VisualisationModule(network = EX_CONFIG['network'],
    #                                station = EX_CONFIG['station'],
    #                                variable = EX_CONFIG['variable'],
    #                                maximum_leadtime=EX_CONFIG['maximum_leadtime'],
    #                                doy_vector = Station.doy_vector,
    #                                evaluation = "ens", # ens
    #                                path_to_plots=PATH_TO_PLOTS)
    
    #PointPlots.plot_station_data_and_forecast(dynamic_features_dict, 
    #                                          dynamic_features_prediction_dict, station_data,
    #                                          matching_indices= matching_indices)
    #
    #EnsemblePlots.plot_scores(layers_ensemble['layer0']['scores'], layers_ensemble['layer1']['scores'], layers_ensemble['layer2']['scores'],
    #                      score = "MAE", log_y=False)
    #EnsemblePlots.plot_skill_scores(layers_ensemble['layer0']['skill_scores'], layers_ensemble['layer1']['skill_scores'], layers_ensemble['layer2']['skill_scores'],
    #                      score = "MAE", log_y=False, sharey = False, invert=True)
    #EnsemblePlots.plot_horizons(layers_ensemble['layer0']['scores'], layers_ensemble['layer1']['scores'], layers_ensemble['layer2']['scores'],
    #                      score = "MAE", threshold = EX_CONFIG['tolerance'], hod=None, log_y=False)

