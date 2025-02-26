import torch
import os
import sys
from scores.probability import crps_cdf
import xarray as xr
import scipy
import pandas as pd
from joblib import Parallel, delayed, parallel_backend
import time

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
VARIABLE = 'sm'

if VARIABLE == 'st':
    VAR_ID = 3
    TARG_VARS = [ "stl1", "stl2", "stl3"]
    CONFIG = load_config(config_path = 'configs/mlp_emulator_node_st.yaml')
    station_ids = ['Condom' ] #, 'Villevielle', 'LaGrandCombe', 'Narbonne', 'Urgons',
             #  'CabrieresdAvignon', 'Savenes', 'PeyrusseGrande','Sabres', 
             #   'Mouthoumet','Mejannes-le-Clap',  'CreondArmagnac', 'SaintFelixdeLauragais',
             #  'Mazan-Abbaye', 'LezignanCorbieres']
    HORIZON = 52 # two weeks horizon
else:
    VAR_ID = 0
    TARG_VARS = [ "swvl1", "swvl2", "swvl3"]
    CONFIG = load_config(config_path = 'configs/mlp_emulator_node_sm.yaml')
    station_ids = [ 'Condom'] # 'Savenes', 'Mouthoumet', 'Mazan-Abbaye', 'LezignanCorbieres', 
                     #   'LaGrandCombe', 'CreondArmagnac', 'Urgons',
    HORIZON = 52*4 # two weeks horizon
     
TARG_LST = [ "swvl1", "swvl2", "swvl3", "stl1", "stl2", "stl3", "snowc"]
MC_SAMPLES = 1000
INITIAL_TIMES = 150

HPARS = load_hpars(use_model = 'ecland-emulator/mlp')

def run_analysis(station_id, Station, time_step):

    closest_grid_cell = Station.closest_indices_dict[station_id]
    ForecastModel = MCdropoutForecastModule(hpars=HPARS, config=CONFIG, closest_grid_cell = closest_grid_cell) 

    ForecastModel.initialise_dataset(TIME_STRINGS[time_step])
    _ = ForecastModel.load_model()
    _, _, _, y_prog_initial_state = ForecastModel.load_test_data()  
    matching_indices_variable = ForecastModel.match_indices(target_variables=TARG_VARS)

    station_data = Station.select_station_data(station_id)
    clim_mean, clim_std = Station.load_climatology(station_id=station_id, initial_time = (31*4) + time_step, variable=VARIABLE)
    y_prog_initial_state[..., matching_indices_variable] = station_data[:y_prog_initial_state.shape[0]]


    initial_vector =  ForecastModel.transform_station_data(station_data = y_prog_initial_state, 
                                                        target_variable_list = TARG_LST)

    dynamic_features, dynamic_features_prediction = ForecastModel.run_forecast(initial_conditions=initial_vector,
                                                                            mc_samples=MC_SAMPLES,
                                                                            predictions_perturbation = True)
    dynamic_features = ForecastModel.backtransformation(dynamic_features)
    dynamic_features_prediction = [ForecastModel.backtransformation(dynamic_features_prediction[i]) for i in range(dynamic_features_prediction.shape[0])]
    dynamic_features_prediction = torch.stack(dynamic_features_prediction, dim=0)


    clim_mean = clim_mean[:,:HORIZON]
    clim_std = clim_std[:,:HORIZON]

    crps_fc = np.empty((HORIZON,3))
    crps_fc[:]  = np.nan
    crps_clim = np.empty((HORIZON,3))
    crps_clim[:]  = np.nan

    # Vectorized sampling for climatological distribution
    climatological_distr = np.random.normal(
        loc=clim_mean[:, :HORIZON], scale=clim_std[:, :HORIZON], size=(MC_SAMPLES, 3, HORIZON)
    )
    climatological_distr.sort(axis=0) 

    for layer in range(3):

        for t in range(HORIZON):

            t_step =dynamic_features_prediction[...,layer + VAR_ID].squeeze().T[t].numpy()
            t_step = np.unique(t_step, axis = 0)

            obs = station_data[...,layer].squeeze()[t]

            if np.isnan(np.asarray(obs)).any():  
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

    # Compute CRPSS safely, avoiding division by zero and NaNs
    crpss = np.where(
        (np.isnan(crps_fc)) | (np.isnan(crps_clim)),  
        np.nan,  # Assign NaN if either value is missing
        1 - np.divide(crps_fc, crps_clim, where=crps_clim != 0)  
    )

    return crpss


start = time.time()

Station = ObservationModule(network = 'soil_SMOSMANIA_ISMN_2022.nc', 
                                    station = station_ids ,
                                    variable = VARIABLE,
                                    depth=[0.05, 0.2, 0.3],
                                    years = [2021, 2022]) # Initialise the Observation Module with the default Station (Gevenich)
        
Station.load_station() # Load two years of station data for lookback slicing
Station.load_forcing() # Load forcing for matching data base with station data
Station.match_station_with_forcing() # Match the station with clostest grid cell and extract the index of the grid cell
Station.process_station_data()

start_time = pd.Timestamp('2022-02-01T00:00:00')
time_steps = pd.date_range(start=start_time, periods=INITIAL_TIMES, freq='6H')  # Adjust periods as needed
TIME_STRINGS = time_steps.strftime('%Y-%m-%d %H:%M:%S').tolist()

def process_station(t):
    station_t = Station.get_class_copy()
    station_t.slice_station_data(lookback=0, t_0=TIME_STRINGS[t])
    results = run_analysis("Condom", station_t, t) 
    return results

final_results = Parallel(n_jobs=8, backend="loky")(delayed(process_station)(t) for t in range(len(TIME_STRINGS)))

# Flatten the nested list if needed
final_results = np.array(final_results)

# Convert to xarray DataArray:q

xr_data = xr.DataArray(
    final_results,
    dims=["t_init", "lead_time", "var"], #"station",
    coords={"t_init": TIME_STRINGS, "lead_time": np.arange(HORIZON), "var": TARG_VARS}, # "station": station_ids,
    name="forecast_horizons"
)

xr_data.to_netcdf(f"ecland-emulator/results/forecast_horizons_{VARIABLE}.nc")

print("--- %s seconds ---" % (time.time() - start))

print(f"Saved to results.")