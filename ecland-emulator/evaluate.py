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
parser.add_argument('--station', type=nullable_string, help='Station name.')
parser.add_argument('--variable', type=nullable_string, help='Specify variable from st or sm.')
parser.add_argument('--maximum_leadtime', type=int, nargs='?', const=56, help='Specify maximum lead time (6-hourly). Default: two weeks')
parser.add_argument('--make_plots', type=bool, nargs='?', const=False, help='Do you want to save the plots when evaluating.')
parser.add_argument('--initial_time', type=nullable_string, nargs='?', const='2022-02-01T00:00:00',  help='Specify variable from st or sm.')
args = parser.parse_args()

STATION = args.station
VARIABLE = args.variable
MAXIMUM_LEADTIME = args.maximum_leadtime
MAKE_PLOTS = args.make_plots


PATH_TO_PLOTS = 'ecland-emulator/plots'
PATH_TO_RESULTS = 'ecland-emulator/results'

if STATION == "Gevenich":
    EX_CONFIG = load_config(config_path = f'configs/tereno_{VARIABLE}.yaml')
else:
    EX_CONFIG = load_config(config_path = f"configs/smosmania_{VARIABLE}.yaml")

INITIAL_TIME = EX_CONFIG['initial_time']

print("Station: ", STATION)
print("Variable: ", VARIABLE)
print("Did you adjust the evaluation targets in the configuration script?")
print("Initial time: ", INITIAL_TIME)
print("Maximum Leadtime: ", MAXIMUM_LEADTIME)

print("Depth: ", EX_CONFIG['depth'])
print("Years: ", EX_CONFIG['years'])


def assemble_forecasts(observations, 
                fc_numerical, 
                fc_emulators,
                maximum_leadtime):
    
    forecasts = {}
    save_data = ["observations", "fc_numerical", "fc_emulators"]
    for layer in [0,1,2]:
        EvaluateModel = EnsembleEvaluation(score =  None,
                                            layer_index = layer,
                                            variable_indices = matching_indices,
                                            maximum_evaluation_time = maximum_leadtime)

        EvaluateModel.set_samples(observations=observations,
                                fc_numerical=fc_numerical,
                                fc_emulator=fc_emulators)
        subsetted_data = EvaluateModel.subset_samples()
        subsetted_data = EvaluateModel.slice_evluation_times()

        print("Observation subset shape:", subsetted_data[0].shape)
        print("Numerical subset shape:", subsetted_data[1].shape)
        print("Emulator subset shape:", subsetted_data[2].shape)
        forecasts[f"layer{layer}"] = dict(zip(save_data, subsetted_data))

        return forecasts
    
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
            EvaluateModel.slice_evluation_times()

            EvaluateModel.transform(use_min_max=False)
            
            numerical_score = EvaluateModel.evaluate_numerical()
            ensemble_score, ensemble_score_dispersion = EvaluateModel.evaluate_emulator()
            ensemble_skill = EvaluateModel.get_skill_score()

            ensemble_score_shifted = EX_CONFIG['tolerance'] - ensemble_score
            numerical_score_shifted = EX_CONFIG['tolerance'] - numerical_score

            scores = {}
            shifted_scores = {}
            scores_dispersion = {}
            skill_scores = {}
            scores["ECLand"] = EvaluateModel.inv_transform(numerical_score)
            scores["Emulators"] = EvaluateModel.inv_transform(ensemble_score)
            shifted_scores["ECLand"] = EvaluateModel.inv_transform(numerical_score_shifted)
            shifted_scores["Emulators"] = EvaluateModel.inv_transform(ensemble_score_shifted)
            scores_dispersion["Emulators"] = EvaluateModel.inv_transform(ensemble_score_dispersion)
            skill_scores["Emulators"] = EvaluateModel.inv_transform(ensemble_skill)

            layers[f"layer{layer}"] = {}
            layers[f"layer{layer}"]["scores"] = scores
            layers[f"layer{layer}"]["shifted_scores"] = shifted_scores
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
            shifted_scores = {}
            skill_scores = {}
            for mod, fc_emulator in fc_emulators.items():

                EvaluateModel.set_samples(observations=observations,
                                    fc_numerical=fc_numerical,
                                    fc_emulator=fc_emulator)
                
                EvaluateModel.subset_samples()
                EvaluateModel.slice_evluation_times()

                EvaluateModel.transform(use_min_max=False)

                numerical_score = EvaluateModel.evaluate_numerical()
                emulator_score, _ = EvaluateModel.evaluate_emulator()
                skill_score = EvaluateModel.get_skill_score()

                emulator_score_shifted = EX_CONFIG['tolerance'] - emulator_score
                numerical_score_shifted = EX_CONFIG['tolerance'] - numerical_score

                scores["ECLand"] = numerical_score
                scores[mod] = emulator_score
                shifted_scores["ECLand"] = numerical_score_shifted
                shifted_scores[mod] = emulator_score_shifted
                skill_scores[mod] = skill_score
                
            layers[f"layer{layer}"] = {}
            layers[f"layer{layer}"]["scores"] = scores
            layers[f"layer{layer}"]["shifted_scores"] = shifted_scores
            layers[f"layer{layer}"]["skill_scores"] = skill_scores

        return layers

def ensemble_horizons(stations_dict):

    layers = {}

    for layer, scores in stations_dict.items():

        layers[f"layer{layer}"] = {}

        layers[f"layer{layer}"]["h_ecland"] = np.argmax((EX_CONFIG['tolerance'] - scores["scores"]["ECLand"]) < 0)
        layers[f"layer{layer}"]["h_emulator"] = np.argmax((EX_CONFIG['tolerance'] - scores["scores"]["Emulators"]) < 0)
        layers[f"layer{layer}"]["h_skillscore"] = np.argmax((scores["skill_scores"]["Emulators"]) < 0)

    return layers

if __name__ == "__main__":

    Station = ObservationModule(network = EX_CONFIG['network'], 
                                station = STATION ,
                                variable = VARIABLE,
                                depth=EX_CONFIG['depth'],
                                years = EX_CONFIG['years']) # Initialise the Observation Module with the default Station (Gevenich)
    

    Station.load_station() # Load two years of station data for lookback slicing
    Station.load_forcing() # Load forcing for matching data base with station data
    Station.match_station_with_forcing() # Match the station with clostest grid cell and extract the index of the grid cell
    soil_type = Station.station_physiography()
    Station.process_station_data(PATH_TO_PLOTS) # specify path_to_plots, if you want to visualise

    dynamic_features_dict = {}
    dynamic_features_prediction_dict = {}

    for mod in EX_CONFIG['models']:

        # initialise experiement and setup model with config file
        #CONFIG, HPARS, ForecastModel = setup_experiment(model = mod)
        if mod == 'mlp':
            print('mlp')
            CONFIG = load_config(config_path = 'configs/mlp_emulator_node.yaml')
            HPARS = load_hpars(use_model = 'ecland-emulator/mlp')
            ForecastModel = ForecastModuleMLP(hpars=HPARS, config=CONFIG,
                                              closest_grid_cell= Station.closest_indices_dict[STATION])    
        elif mod == 'lstm':
            CONFIG = load_config(config_path = 'configs/lstm_emulator_node.yaml')
            HPARS = load_hpars(use_model = 'ecland-emulator/lstm')
            ForecastModel = ForecastModuleLSTM(hpars=HPARS, config=CONFIG,
                                              closest_grid_cell= Station.closest_indices_dict[STATION])
        elif mod == 'xgb':
            CONFIG = load_config(config_path = 'configs/xgb_emulator_node.yaml')
            HPARS = None
            ForecastModel = ForecastModuleXGB(hpars=HPARS, config=CONFIG,
                                              closest_grid_cell= Station.closest_indices_dict[STATION])

        Station.slice_station_data(lookback=CONFIG["lookback"], t_0=INITIAL_TIME)
        ForecastModel.initialise_dataset(INITIAL_TIME)
        model = ForecastModel.load_model()
        x_static, x_met, y_prog, y_prog_initial_state = ForecastModel.load_test_data()  

        matching_indices = ForecastModel.match_indices(target_variables=EX_CONFIG['targets_eval'])

        station_data = Station.select_station_data(STATION)
        y_prog_initial_state[..., matching_indices] = station_data[:y_prog_initial_state.shape[0]]
        
        matching_indices = ForecastModel.match_indices(target_variables=EX_CONFIG['targets_prog'])
        print("MATCHING INDICES: ", matching_indices)
        initial_vector =  ForecastModel.transform_station_data(station_data = y_prog_initial_state, 
                                                    target_variable_list = EX_CONFIG['targets_prog'])

        dynamic_features, dynamic_features_prediction = ForecastModel.run_forecast(initial_conditions=initial_vector)
        dynamic_features, dynamic_features_prediction = ForecastModel.backtransformation()

        dynamic_features_dict[mod] = dynamic_features
        dynamic_features_prediction_dict[mod] = dynamic_features_prediction

    fc_numerical = dynamic_features_dict['mlp']
    fc_emulators = dynamic_features_prediction_dict

    matching_indices = ForecastModel.match_indices(target_variables=EX_CONFIG['targets_eval'])
    
    forecast_ensemble = assemble_forecasts(observations=station_data,
                    fc_numerical=fc_numerical,
                    fc_emulators=fc_emulators,
                    maximum_leadtime= MAXIMUM_LEADTIME)
    
    save_to =os.path.join(PATH_TO_RESULTS, 
                          f"{EX_CONFIG['network'].split('_')[1]}_{STATION}_{max(EX_CONFIG['years'])}_{VARIABLE}_ensemble_fc.yaml")
    print("Write forecasts to path:", save_to)

    with open(save_to, 'w') as f:
        yaml.dump(forecast_ensemble, f, indent=4)

    
    #StationTransform = Transform(use_min_max=True)
    #NumericalTransform = Transform(use_min_max=True)
    #EmulatorTransform = Transform(use_min_max=True)

    #station_data = StationTransform.normalise(station_data)
    #fc_numerical = NumericalTransform.normalise(fc_numerical)
    #for key, array in fc_emulators.items():
    #    fc_emulators[key] = EmulatorTransform.normalise(array)

    print("Observation subset shape:", station_data.shape)
    print("Numerical subset shape:", fc_numerical.shape)
    
    layers_ensemble = evaluate_ensemble(observations=station_data,
                    fc_numerical=fc_numerical,
                    fc_emulators=fc_emulators,
                    score=EX_CONFIG['score'] ,
                    maximum_leadtime= MAXIMUM_LEADTIME)
    
    layers_point = evaluate_point(observations=station_data,
                    fc_numerical=fc_numerical,
                    fc_emulators=fc_emulators,
                    score=EX_CONFIG['score'] ,
                    maximum_leadtime= MAXIMUM_LEADTIME)

    layer_horizons = ensemble_horizons(layers_ensemble)

    with open(os.path.join(PATH_TO_RESULTS,  f"{EX_CONFIG['network'].split('_')[1]}_{STATION}_{max(EX_CONFIG['years'])}_{VARIABLE}_ensemble.yaml"), 'w') as f:
        yaml.dump(layers_ensemble, f, indent=4)

    with open(os.path.join(PATH_TO_RESULTS, f"{EX_CONFIG['network'].split('_')[1]}_{STATION}_{max(EX_CONFIG['years'])}_{VARIABLE}_point.yaml"), 'w') as f:
        yaml.dump(layers_point, f, indent=4)

    with open(os.path.join(PATH_TO_RESULTS, f"{EX_CONFIG['network'].split('_')[1]}_{STATION}_{max(EX_CONFIG['years'])}_{VARIABLE}_horizons.yaml"), 'w') as f:
        yaml.dump(layer_horizons, f, indent=4)

    if MAKE_PLOTS:
        print("Making plots!")

        PointPlots = VisualisationSingle(network = EX_CONFIG['network'],
                                        station = STATION,
                                        variable = VARIABLE,
                                        score = EX_CONFIG["score"],
                                        maximum_leadtime=MAXIMUM_LEADTIME,
                                        doy_vector = Station.doy_vector,
                                        evaluation = "poi", # ens
                                        path_to_plots=PATH_TO_PLOTS)
        
        EnsemblePlots = VisualisationSingle(network = EX_CONFIG['network'],
                                        station = STATION,
                                        variable = VARIABLE,
                                        score = EX_CONFIG["score"],
                                        maximum_leadtime=MAXIMUM_LEADTIME,
                                        doy_vector = Station.doy_vector,
                                        evaluation = "ens", # ens
                                        path_to_plots=PATH_TO_PLOTS)
        
        PointPlots.plot_station_data_and_forecast(fc_numerical, 
                                                fc_emulators, 
                                                station_data,
                                                matching_indices= matching_indices)
        
        PointPlots.plot_horizons(layers_point['layer0']['scores'], layers_point['layer1']['scores'], layers_point['layer2']['scores'],
                                threshold = EX_CONFIG["tolerance"])
        
        EnsemblePlots.plot_scores(layers_ensemble['layer0']['scores'], layers_ensemble['layer1']['scores'], layers_ensemble['layer2']['scores'], log_y=False)
        EnsemblePlots.plot_skill_scores(layers_ensemble['layer0']['skill_scores'], layers_ensemble['layer1']['skill_scores'], layers_ensemble['layer2']['skill_scores'], 
                                        log_y=False, sharey = False)
        EnsemblePlots.plot_horizons(layers_ensemble['layer0']['scores'], layers_ensemble['layer1']['scores'], layers_ensemble['layer2']['scores'],
                                    #scores_l1_std = layers_ensemble['layer0']['scores_dispersion'], 
                                    #scores_l2_std = layers_ensemble['layer1']['scores_dispersion'], 
                                    #scores_l3_std = layers_ensemble['layer2']['scores_dispersion'],
                                    threshold = EX_CONFIG['tolerance'], hod=None, log_y=False)

