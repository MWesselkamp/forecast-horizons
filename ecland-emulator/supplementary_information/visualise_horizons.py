
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

def load_yaml(file_path):
    """Load a YAML file."""
    with open(file_path, 'r') as f:
        return yaml.load(f, Loader=yaml.UnsafeLoader)

def check_condition(condition):
    """Helper function to check if condition is met in any element.
    """
    if np.any(condition):
        return np.argmax(condition)
    else:
        return len(condition)

def ensemble_horizons(stations_dict):
    """Calculate horizons for each layer."""
    layers = {}
    for layer, scores in stations_dict.items():
        layers[f"layer{layer}"] = {
            "h_ecland": check_condition((EX_CONFIG['tolerance'] - scores["scores"]["ECLand"]) < 0),
            "h_emulator": check_condition((EX_CONFIG['tolerance'] - scores["scores"]["Emulators"]) < 0),
            "h_skillscore": check_condition((1 - scores["skill_scores"]["Emulators"]) < 0)
        }
    return layers

def assemble_data_to_dataframe(station_dict, station_name, soil_type):
    """Assemble data from station_dict into a DataFrame."""
    
    data = []
    #for file_path in yaml_files:
    #    station_name = os.path.basename(file_path).split('_')[1]  # Adjust split if station name is located differently
    #    layer_horizons = load_yaml(file_path)
        
    for layer, values in station_dict.items():
        data.append({
                "station": station_name,
                "soil_type": soil_type,
                "layer": layer,
                "h_ecland": values["h_ecland"],
                "h_emulator": values["h_emulator"],
                "h_skillscore": values["h_skillscore"]
            })

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=["station", "soil_type", "layer", "h_ecland", "h_emulator", "h_skillscore"])
    return df

def visualise(df):

    label_properties = {'weight':'bold', 'size':16}
    legend_properties = {'weight':'bold', 'size':14}
    tick_properties = {'size':16}
    linewidth = 2
    figsize = (13, 5)
    titles = ["Surface Layer", "Subsurface Layer 1", "Subsurface Layer 2"]

    colors = {
        "h_ecland": {"facecolor": "cyan", "edgecolor": "black"},
        "h_emulator": {"facecolor": "magenta", "edgecolor": "black"},
        "h_skillscore": {"facecolor": "purple", "edgecolor": "black"}
    }

    layered_data = df.groupby("layer")[["h_ecland", "h_emulator", "h_skillscore"]].apply(lambda x: x.values.tolist())

    # Define subplots
    fig, axs = plt.subplots(1, 3, figsize=figsize, sharey=True)

    for i, layer in enumerate(df['layer'].unique()):
        data = [df[df['layer'] == layer][column] for column in ["h_ecland", "h_emulator", "h_skillscore"]]
        box = axs[i].boxplot(data, labels=[r"$h_{a,ECLand}$", r"$h_{a,Emulator}$", r"$h_{r}$"], patch_artist=True)

        axs[i].set_title(titles[i], fontdict=label_properties) 
        #axs[i].set_xlabel('Type', fontdict=label_properties)
        plt.setp(axs[i].get_yticklabels(), **tick_properties)
        plt.setp(axs[i].get_xticklabels(), **tick_properties)
        if i == 0:
            axs[i].set_ylabel('Horizon [6-hrly timesteps]', fontdict=label_properties)

        for j, patch in enumerate(box['boxes']):
            column = ["h_ecland", "h_emulator", "h_skillscore"][j]
            patch.set_facecolor(colors[column]["facecolor"])
            patch.set_edgecolor(colors[column]["edgecolor"])
            patch.set_linewidth(1.5)
        for j, median in enumerate(box['medians']):
            median.set_color("orange")
            median.set_linewidth(1.8)
            median_value = median.get_ydata()[0]
            # Add the median value text above the box
            axs[i].text(j + 1, 10, f"{median_value}", 
                        ha='center', va='bottom', fontdict={'size': 14, 'weight': 'bold'})


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
    for key, item in use_stations.items():
        with open(f"ecland-emulator/results/SMOSMANIA_{key}_2022_{VARIABLE}_ensemble.yaml", 'r') as f:
            layers = yaml.load(f, Loader=yaml.UnsafeLoader)
        station_dict = ensemble_horizons(layers)
        df_i = assemble_data_to_dataframe(station_dict, key, item)
        horizons_data.append(df_i)


    df = pd.concat(horizons_data, ignore_index=True)
    file_path = os.path.join(PATH_TO_RESULTS, f'SMOSMANIA_2022_{VARIABLE}_horizons_data.csv')
    df.to_csv(file_path, index=False)

    print(df)

    visualise(df)