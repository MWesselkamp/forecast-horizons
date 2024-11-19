import os
import yaml
import json

def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def set_parameters(config, process, scenario, dir):

    scenario_config = config[scenario]
    observation_params = scenario_config['observation_params']
    initial_params = scenario_config['initial_params']
    process_config = scenario_config['processes'][process]

    true_noise = process_config['true_noise']
    initial_noise = process_config['initial_noise']

    parameters = {
        "observation_params": observation_params,
        "initial_params": initial_params,
        "true_noise": true_noise,
        "initial_noise": initial_noise
    }

    os.makedirs(dir, exist_ok=True)
    with open(os.path.join(dir, "parameters.json"), "w") as json_file:
        json.dump(parameters, json_file)

    return parameters
