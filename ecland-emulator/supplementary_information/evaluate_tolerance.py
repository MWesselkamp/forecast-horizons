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
parser.add_argument('--variable', type=nullable_string, help='Specify variable from st or sm.')
parser.add_argument('--maximum_leadtime', type=int, nargs='?', const=56, help='Specify maximum lead time (6-hourly). Default: two weeks')
args = parser.parse_args()

VARIABLE = args.variable
MAXIMUM_LEADTIME = args.maximum_leadtime


PATH_TO_PLOTS = 'ecland-emulator/plots'
PATH_TO_RESULTS = 'ecland-emulator/results'

EX_CONFIG = load_config(config_path = f"configs/smosmania_{VARIABLE}.yaml")

INITIAL_TIME = EX_CONFIG['initial_time']


print("Variable: ", VARIABLE)
print("Did you adjust the evaluation targets in the configuration script?")
print("Initial time: ", INITIAL_TIME)
print("Maximum Leadtime: ", MAXIMUM_LEADTIME)

print("Depth: ", EX_CONFIG['depth'])
print("Years: ", EX_CONFIG['years'])


def load_yaml(file_path):
    """Load a YAML file."""
    with open(file_path, 'r') as f:
        return yaml.load(f, Loader=yaml.UnsafeLoader)
    

def check_condition(condition):
    """Helper function to check if condition is met in any element.
    """
    condition = np.asarray(condition)
    
    # Debug print for validation
    print(f"Condition array: {condition}, Type: {type(condition)}")
    
    if np.any(condition):
        return np.argmax(condition)
    else:
        return len(condition)

def ensemble_horizons(stations_dict, tolerance):
    """Calculate horizons for each layer."""
    layers = {}
    for layer, scores in stations_dict.items():
        print(scores)
        layers[f"layer{layer}"] = {
            "h_ecland": check_condition((tolerance - scores["scores"]["ECLand"]) < 0),
            "h_emulator": check_condition((tolerance - scores["scores"]["Emulators"]) < 0)
        }
    return layers

def assemble_data_to_dataframe(station_dict, station_name, tolerance):
    """Assemble data from station_dict into a DataFrame."""
    
    data = []
    #for file_path in yaml_files:
    #    station_name = os.path.basename(file_path).split('_')[1]  # Adjust split if station name is located differently
    #    layer_horizons = load_yaml(file_path)
        
    for layer, values in station_dict.items():
        data.append({
                "station": station_name,
                "tolerance": tolerance,
                "layer": layer,
                "h_ecland": values["h_ecland"],
                "h_emulator": values["h_emulator"]
            })

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=["station", "tolerance", "layer", "h_ecland", "h_emulator"])
    return df

def visualise(df_subsets):

    label_properties = {'weight':'bold', 'size':16}
    legend_properties = {'weight':'bold', 'size':14}
    tick_properties = {'size':16}
    linewidth = 2
    figsize = (13, 5)

    titles = ["Surface Layer", "Subsurface Layer 1", "Subsurface Layer 2"]


    # Define subplots
    fig, axs = plt.subplots(1, 3, figsize=figsize, sharey=True)

    i=0
    for key, item in df_subsets.items():
        stations = item['station'].unique()
        for station in stations:
            station_data = item[item['station'] == station]
            axs[i].plot(station_data['threshold'], station_data['h_ecland'], linestyle='-', color='cyan')
            axs[i].plot(station_data['threshold'], station_data['h_emulator'], linestyle='-', color = 'magenta')

        axs[i].set_title(titles[i], fontdict=label_properties) 
        #axs[i].set_xlabel('Type', fontdict=label_properties)
        plt.setp(axs[i].get_yticklabels(), **tick_properties)
        plt.setp(axs[i].get_xticklabels(), **tick_properties)
        if i == 0:
            axs[i].set_ylabel('Horizon [6-hrly timesteps]', fontdict=label_properties)
        axs[i].set_xlabel('MAE tolerance', fontdict=label_properties)

        i+=1

    #fig.text(0.5, 0.04, 'Horizon Type', ha='center', fontdict=label_properties)

    plt.tight_layout()
    fig_path = os.path.join(PATH_TO_PLOTS, f'SMOSMANIA_2022_{VARIABLE}_aggregated_horizons.pdf')
    plt.savefig(fig_path)
    plt.show()


if __name__ == "__main__":

    if VARIABLE == 'st':
        use_stations = {'Condom':'Silty clay', 'Villevielle':'Sandy loam', 'LaGrandCombe':'Loamy sand', 
                        'Narbonne':'Clay', 'Urgons':'Silt loam','LezignanCorbieres':'Sandy clay loam',
                        'CabrieresdAvignon':'Sandy clay loam', 'Savenes':'Loam', 'PeyrusseGrande':'Silty clay',
                        'Sabres':'Sand', 'Montaut':'Silt loam', 'Mazan-Abbaye':'Sandy loam',
                        'Mouthoumet':'Clay loam','Mejannes-le-Clap':'Loam', 'CreondArmagnac':'Sand', 
                        'SaintFelixdeLauragais':'Loam'}
    elif VARIABLE == 'sm':
        use_stations = {'Condom':'Silty clay', 'LaGrandCombe':'Loamy sand', 
                        'Urgons':'Silt loam','LezignanCorbieres':'Sandy clay loam',
                        'Savenes':'Loam', 'Mazan-Abbaye':'Sandy loam',
                        'Mouthoumet':'Clay loam','CreondArmagnac':'Sand', 
                        }
    else:
        print("Don't know variable.")


    horizons_data = []
    tolerance_sequence = np.arange(0.4,2,0.2)

    for tolerance in tolerance_sequence:
        for key, item in use_stations.items():
            with open(f"ecland-emulator/results/SMOSMANIA_{key}_2022_{VARIABLE}_ensemble.yaml", 'r') as f:
                stations_dict = yaml.load(f, Loader=yaml.UnsafeLoader)

            print(stations_dict)
            print(tolerance)

            horizon_layers = ensemble_horizons(stations_dict, tolerance = tolerance)
            df_i = assemble_data_to_dataframe(horizon_layers, key, tolerance)
            horizons_data.append(df_i)

    df = pd.concat(horizons_data, ignore_index=True)
    file_path = os.path.join(PATH_TO_RESULTS, f'SMOSMANIA_2022_{VARIABLE}_tolerance_data.csv')
    df.to_csv(file_path, index=False)

    layerwise_subset = {'layer0':df[df['layers'] == 'layer0'], 'layer1':df[df['layers'] == 'layer1'], 'layer2':df[df['layers'] == 'layer2']}

    visualise(layerwise_subset)

