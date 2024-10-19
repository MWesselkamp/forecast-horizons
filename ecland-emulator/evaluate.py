import torch
import yaml
import os
import sys
import matplotlib.pyplot as plt

parent_dir = os.path.abspath('..')
sys.path.append(parent_dir)

from models import *
from data_module import *
from forecast_module import *
from evaluation_module import *
from helpers import *
from observation_module import *
from tests.test_model import *
from visualisations import *

set_global_seed(42)

SCRIPT_DIR = os.getcwd()
sys.path.append(os.path.dirname(SCRIPT_DIR))
print(SCRIPT_DIR) 

path_to_plots = '../plots/'
path_to_results = '../results/'

network =  'soil_TERENO_ISMN_2022.nc'#'soil_SMOSMANIA_ISMN_2022.nc' # 'soil_TERENO_ISMN_2022.nc'
station = 'Gevenich'# 'Lahas'
variable = 'st'
depth = [0.05, 0.2, 0.5]

models = ['mlp', 'lstm'] # , 'xgb'

print(models)