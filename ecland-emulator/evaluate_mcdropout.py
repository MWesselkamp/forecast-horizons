import torch
import os
import sys
import matplotlib.pyplot as plt
import yaml
import argparse
from scores.probability import crps_cdf
import xarray as xr
import scipy
import pandas as pd
from joblib import Parallel, delayed

parent_dir = os.path.abspath('..')
sys.path.append(parent_dir)

from modules.data_module import *
from modules.evaluation_module import *
from modules.mcdropout_module import *
from modules.observation_module import *

from misc.helpers import *
from tests.test_model import *

set_global_seed(42)

SCRIPT_DIR = os.getcwd()
sys.path.append(os.path.dirname(SCRIPT_DIR))
print(SCRIPT_DIR) 

INITIAL_TIME = '2022-02-01T00:00:00'
VARIABLE = 'st'
if VARIABLE == 'st':
     VAR_ID = 3
else:
     VAR_ID = 0
TARG_LST = [ "swvl1", "swvl2", "swvl3", "stl1", "stl2", "stl3", "snowc"]
TARG_VARS = [ "stl1", "stl2", "stl3"]
MC_SAMPLES = 10
HORIZON = 52 # two weeks horizon
CONFIG = load_config(config_path = 'configs/mlp_emulator_node.yaml')
HPARS = load_hpars(use_model = 'ecland-emulator/mlp')

def run_analysis(station_id, Station):

    closest_grid_cell = Station.closest_indices_dict[station_id]
    ForecastModel = MCdropoutForecastModule(hpars=HPARS, config=CONFIG, closest_grid_cell = closest_grid_cell) 

    ForecastModel.initialise_dataset(INITIAL_TIME)
    _ = ForecastModel.load_model()
    _, _, _, y_prog_initial_state = ForecastModel.load_test_data()  
    matching_indices_variable = ForecastModel.match_indices(target_variables=TARG_VARS)
    matching_indices_target = ForecastModel.match_indices(target_variables=TARG_LST)

    station_data = Station.select_station_data(station_id)
    clim_mean, clim_std = Station.load_climatology(station_id=station_id, initial_time = 31*4, variable=VARIABLE)
    y_prog_initial_state[..., matching_indices_variable] = station_data[:y_prog_initial_state.shape[0]]


    initial_vector =  ForecastModel.transform_station_data(station_data = y_prog_initial_state, 
                                                        target_variable_list = TARG_LST)

    dynamic_features, dynamic_features_prediction = ForecastModel.run_forecast(initial_conditions=initial_vector,
                                                                            mc_samples=MC_SAMPLES,
                                                                            predictions_perturbation = True)
    dynamic_features = ForecastModel.backtransformation(dynamic_features)
    dynamic_features_prediction = [ForecastModel.backtransformation(dynamic_features_prediction[i]) for i in range(dynamic_features_prediction.shape[0])]
    dynamic_features_prediction = torch.stack(dynamic_features_prediction, dim=0)

    # prepare data for plots
    ensemble_pred = dynamic_features_prediction.squeeze()[:,:HORIZON,:]
    ecland_pred = dynamic_features.squeeze()[:HORIZON]
    observations = station_data.squeeze()[:HORIZON]
    clim_mean = clim_mean[:,:HORIZON]
    clim_std = clim_std[:,:HORIZON]
    doy = Station.doy_vector[:HORIZON].T

    fig, ax = plt.subplots(1,3,figsize = (12,4), sharey=False, sharex=True, constrained_layout = True)
    for layer in range(3):
        ax[layer].fill_between(doy, clim_mean[layer] + clim_std[layer], clim_mean[layer] - clim_std[layer], color = "lightblue",alpha = 0.6, label="Climatological std")
        ax[layer].plot(doy, ensemble_pred[...,layer+VAR_ID].squeeze().T, color= "lightgray", alpha = 0.7, linewidth = 0.8)
        ax[layer].plot(doy, ensemble_pred[...,layer+VAR_ID].squeeze()[0].T, color= "lightgray", alpha = 0.7, linewidth = 0.8, label="aiLand")
        ax[layer].plot(doy, clim_mean[layer], color = "blue", linewidth = 0.8, label="Climatological mean")
        ax[layer].plot(doy, ecland_pred[:,layer+VAR_ID], color = "cyan", alpha = 0.8, linewidth = 0.8, label="ecLand")
        ax[layer].plot(doy, observations[:,layer], color = "red", linewidth = 0.8, label="Ground Station")
        ax[layer].set_xlabel("Lead times")
        ax[layer].set_ylabel("Soil temperature [K]")
        ax[layer].tick_params(axis='x', rotation=45)
    ax[layer].legend(loc="upper right")
    #for a in ax:  # Loop through all axes
    #    a.tick_params(axis='x', rotation=45)
    plt.savefig(f'ecland-emulator/plots/mc_ensemble_fc_{station_id}.pdf')
    plt.show()

    print("MC ensemble forecast")
    crps_fc = np.zeros((HORIZON,3))
    crps_clim = np.zeros((HORIZON,3))

    #horiz_fc = np.zeros((MC_SAMPLES,3))
    #cfd_fc = np.zeros((MC_SAMPLES,3))

    for layer in range(3):
        for t in range(HORIZON):
            t_step =dynamic_features_prediction[...,layer + VAR_ID].squeeze().T[t].numpy()
            t_step = np.unique(np.sort(t_step, axis=0), axis = 0)

            obs = station_data[...,0].squeeze()[t]

            climatological_distr = np.random.normal(loc = clim_mean[layer,t], scale=clim_std[layer,t], size=MC_SAMPLES)
            climatological_distr = np.sort(climatological_distr, axis=0)

            fcst_cdf = scipy.stats.norm.cdf(t_step, loc=np.mean(t_step), scale = np.std(t_step))
            clim_cdf = scipy.stats.norm.cdf(climatological_distr, loc=np.mean(climatological_distr), scale = np.std(climatological_distr))

            fcst_array = xr.DataArray(coords={'stl': t_step}, data=fcst_cdf)
            clim_array = xr.DataArray(coords={'stl': climatological_distr}, data=clim_cdf)
            obs_array = xr.DataArray(obs)
            crps_fc[t, layer] = crps_cdf(fcst_array, obs_array, threshold_dim='stl').total.values.round(4)
            crps_clim[t, layer] = crps_cdf(clim_array, obs_array, threshold_dim='stl').total.values.round(4)
        #horiz_fc[:,layer] = t_step
        #cfd_fc[:,layer] = fcst_cdf

    crpss = 1 - (crps_fc/crps_clim)

    fig, ax = plt.subplots(1,3,figsize = (12,4), sharey=True, sharex=True, constrained_layout = True)
    for layer in range(3):
        ax[layer].hlines(y = 0, xmin=0, xmax=HORIZON, linestyles="--", color="black")
        ax[layer].plot(doy, crpss[:,layer].T, color= "darkblue")
        #ax[0].plot(dynamic_features[...,3].squeeze(), color = "cyan")
        ax[layer].set_ylabel("CRPSS")
        ax[layer].set_xlabel("Lead time")
        ax[layer].tick_params(axis='x', rotation=45)
    plt.savefig(f'ecland-emulator/plots/mc_ensemble_crpss_{station_id}.pdf')
    plt.show()

    #fig, ax = plt.subplots(1,3,figsize = (12,4), sharey=True, sharex=True, constrained_layout = True)
    #for layer in range(3):
    #    ax[layer].plot(horiz_fc[:,layer], cfd_fc[:,layer], color= "darkblue")
    #    #ax[0].plot(dynamic_features[...,3].squeeze(), color = "cyan")
    #    ax[layer].set_ylabel("Probability of non-exceedance")
    #    ax[layer].set_xlabel("Relative size")
    # plt.savefig(f'ecland-emulator/plots/mc_ensemble_exeedanceprob_{station_id}.pdf')
    # plt.show()

    return crpss

station_ids = ['Condom', 'Villevielle', 'LaGrandCombe', 'Narbonne', 'Urgons']
                   # 'CabrieresdAvignon', 'Savenes', 'PeyrusseGrande','Sabres', 
                   # 'Mouthoumet','Mejannes-le-Clap',  'CreondArmagnac', 'SaintFelixdeLauragais',
                   # 'Mazan-Abbaye', 'LezignanCorbieres']


Station = ObservationModule(network = 'soil_SMOSMANIA_ISMN_2022.nc', 
                                    station = station_ids ,
                                    variable = VARIABLE,
                                    depth=[0.05, 0.2, 0.3],
                                    years = [2021, 2022]) # Initialise the Observation Module with the default Station (Gevenich)
        
Station.load_station() # Load two years of station data for lookback slicing
Station.load_forcing() # Load forcing for matching data base with station data
Station.match_station_with_forcing() # Match the station with clostest grid cell and extract the index of the grid cell
Station.process_station_data()
Station.slice_station_data(lookback=0, t_0=INITIAL_TIME)

results = Parallel(n_jobs=-1)(delayed(run_analysis)(sid, Station) for sid in station_ids)

start_time = pd.Timestamp('2022-02-01T00:00:00')
time_steps = pd.date_range(start=start_time, periods=10, freq='6H')  # Adjust periods as needed
time_strings = time_steps.strftime('%Y-%m-%d %H:%M:%S').tolist()

doy = Station.doy_vector[:HORIZON]

data_np = np.stack(results)

print(doy.shape)
print(doy)

# Convert to Python datetime
doy = [pd.Timestamp(t).to_pydatetime() for t in doy]

print(data_np.shape)

fig, ax = plt.subplots(1,3,figsize = (12,4), sharey=False, sharex=True, constrained_layout = True)
for layer in range(3):
        station_mean = np.nanmean(data_np[...,layer].squeeze(), axis=0)
        station_std = np.nanstd(data_np[...,layer].squeeze(), axis=0)

        ax[layer].hlines(y = 0, xmin=0, xmax=HORIZON, linestyles="--", color="black")
        ax[layer].fill_between((station_mean+station_std).T, (station_mean-station_std).T, color="lightblue", alpha = 0.7, linewidth = 0.9, label="Station spread")
        ax[layer].plot(station_mean.T, color="blue", alpha = 0.99, label = "Station mean")
        #ax[0].plot(dynamic_features[...,3].squeeze(), color = "cyan")
        ax[layer].set_ylabel("CRPSS")
        ax[layer].set_xlabel("Lead time")
        ax[layer].tick_params(axis='x', rotation=45)
ax[layer].legend()
plt.savefig(f'ecland-emulator/plots/mc_ensemble_crpss_combined.pdf')
plt.show()