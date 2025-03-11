import torch
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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


SCRIPT_DIR = os.getcwd()
sys.path.append(os.path.dirname(SCRIPT_DIR))
print(SCRIPT_DIR) 


VARIABLE = 'sm'
if VARIABLE == 'st':
    VAR_ID = 3
    TARG_VARS = [ "stl1", "stl2", "stl3"]
    CONFIG = load_config(config_path = 'configs/mlp_emulator_node_st.yaml')
    station_ids = ['Condom' , 'Villevielle', 'LaGrandCombe', 'Narbonne', 'Urgons',
               'CabrieresdAvignon', 'Savenes', 'PeyrusseGrande','Sabres', 
                'Mouthoumet','Mejannes-le-Clap',  'CreondArmagnac', 'SaintFelixdeLauragais',
               'Mazan-Abbaye', 'LezignanCorbieres']
    HORIZON = 52 # two weeks horizon
else:
     VAR_ID = 0
     TARG_VARS = [ "swvl1", "swvl2", "swvl3"]
     CONFIG = load_config(config_path = 'configs/mlp_emulator_node_sm.yaml')
     station_ids = ['Savenes', 'Mouthoumet', 'Mazan-Abbaye', 'LezignanCorbieres', 
                        'LaGrandCombe', 'CreondArmagnac', 'Urgons', 'Condom']
     HORIZON = 52*4 # eight weeks horizon

INITIAL_TIME = '2022-02-01T00:00:00'
TARG_LST = [ "swvl1", "swvl2", "swvl3", "stl1", "stl2", "stl3", "snowc"]
MC_SAMPLES = 5000
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
    doy = pd.to_datetime(Station.doy_vector[:HORIZON].T)

    
    print("MC ensemble forecast")
    crps_fc = np.empty((HORIZON,3))
    crps_fc[:]  = np.nan
    crps_clim = np.empty((HORIZON,3))
    crps_clim[:]  = np.nan

    climatological_distr = np.random.normal(
        loc=clim_mean[:, :HORIZON], scale=clim_std[:, :HORIZON], size=(MC_SAMPLES, 3, HORIZON)
    )
    climatological_distr.sort(axis=0) 

    for layer in range(3):

        for t in range(HORIZON):

            t_step =dynamic_features_prediction[...,layer + VAR_ID].squeeze().T[t].numpy()
            t_step = np.unique(t_step, axis = 0)

            obs = station_data[...,layer].squeeze()[t]

            if np.isnan(np.asarray(obs)):
                continue # Skip CRPS computation for this time step if obs is missing

            # Precomputed climatological distribution for this layer and time step
            clim_t = climatological_distr[:, layer, t]

            fcst_cdf = scipy.stats.norm.cdf(t_step, loc=np.mean(t_step), scale=np.std(t_step))
            clim_cdf = scipy.stats.norm.cdf(clim_t, loc=np.mean(clim_t), scale=np.std(clim_t))

            crps_fc[t, layer] = crps_cdf(
                xr.DataArray(fcst_cdf, coords={'stl': t_step}),
                xr.DataArray(obs),
                threshold_dim='stl'
            ).total.values.round(4)

            crps_clim[t, layer] = crps_cdf(
                xr.DataArray(clim_cdf, coords={'stl': clim_t}),
                xr.DataArray(obs),
                threshold_dim='stl'
            ).total.values.round(4)

    crpss = np.where(
        (np.isnan(crps_fc)) | (np.isnan(crps_clim)),  
        np.nan,  # Assign NaN if either value is missing
        1 - np.divide(crps_fc, crps_clim, where=crps_clim != 0)  
    )

    ylabel = "Soil temperature [K]" if VARIABLE == 'st' else "Soil moisture [m$^{3}$/m$^{-3}$]"

    # Create figure and subplots with different heights for the two rows
    fig, ax = plt.subplots(2, 3, figsize=(12, 6), sharex=True, constrained_layout=True,
                        gridspec_kw={'height_ratios': [2, 1]})  # First row double the height

    # ---- First Row (Soil Temperature) ----
    for layer in range(3):
        ax[0, layer].fill_between(doy, clim_mean[layer] + clim_std[layer], clim_mean[layer] - clim_std[layer],
                                color="lightblue", alpha=0.6, label="Climatological std")
        ax[0, layer].plot(doy, ensemble_pred[..., layer + VAR_ID].squeeze().T, color="lightgray", alpha=0.7, linewidth=0.8)
        ax[0, layer].plot(doy, ensemble_pred[..., layer + VAR_ID].squeeze()[0].T, color="lightgray",
                        alpha=0.7, linewidth=0.8, label="aiLand")
        ax[0, layer].plot(doy, clim_mean[layer], color="blue", linewidth=0.8, label="Climatological mean")
        ax[0, layer].plot(doy, ecland_pred[:, layer + VAR_ID], color="cyan", alpha=0.8, linewidth=0.8, label="ecLand")
        ax[0, layer].plot(doy, observations[:, layer], color="red", linewidth=0.8, label="Ground Station")

        ax[0, layer].set_ylabel(ylabel)
        ax[0, layer].tick_params(axis='x', rotation=45)

    # ---- Second Row (CRPSS) ----
    for layer in range(3):
        ax[1, layer].plot(doy, crpss[:, layer].T, color="darkblue")
        ax[1, layer].hlines(y=0, xmin=doy[0], xmax=doy[-1], linestyles="--", color="black")
        ax[1, layer].set_ylabel("CRPSS")

    # Set common x-axis properties
    for layer in range(3):
        ax[1, layer].set_xlabel("Lead time")
        ax[1, layer].tick_params(axis='x', rotation=45)
        ax[1, layer].xaxis.set_major_locator(mdates.AutoDateLocator())
        ax[1, layer].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.99))

    # Ensure layout accommodates the legend
    plt.tight_layout(rect=[0, 0, 1, 0.93])  # Leaves space at top for legend


    plt.savefig(f'ecland-emulator/plots/mc_ensemble_combined_{VARIABLE}_{station_id}.pdf')
    plt.show()

    return crpss


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

doy = pd.to_datetime(Station.doy_vector[:HORIZON].T)
data_np = np.stack(results)

fig, ax = plt.subplots(1,3,figsize = (12,4), sharey=False, sharex=True, constrained_layout = True)
for layer in range(3):
        station_mean = np.nanmean(data_np[...,layer].squeeze(), axis=0)
        station_std = np.nanstd(data_np[...,layer].squeeze(), axis=0)
        ax[layer].fill_between(doy, (station_mean+station_std), (station_mean-station_std), color="lightblue", alpha = 0.7, linewidth = 0.9, label="Station spread")
        ax[layer].hlines(y = 0, xmin=doy[0], xmax=doy[-1], linestyles="--", color="black")
        ax[layer].plot(doy, station_mean, color="blue", alpha = 0.99, label = "Station mean")
        #ax[0].plot(dynamic_features[...,3].squeeze(), color = "cyan")
        ax[layer].set_ylabel("CRPSS")
        ax[layer].set_xlabel("Lead time")
        ax[layer].tick_params(axis='x', rotation=45)
        ax[layer].xaxis.set_major_locator(mdates.AutoDateLocator())
        ax[layer].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax[layer].legend()
plt.savefig(f'ecland-emulator/plots/mc_ensemble_crpss_combined_{VARIABLE}.pdf')
plt.show()