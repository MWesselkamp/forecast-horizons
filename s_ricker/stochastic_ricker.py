"""
Steady state predictability analysis with stochastic ricker model.
author: @mariekewesselkamp
"""
import sys
print(sys.path)

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import norm
from scores.probability import crps_cdf
from joblib import Parallel, delayed, parallel_backend

np.random.seed(42)

def sample_pars(ns, r, k, sigma_N, error_size):
    """
    Generate random samples for parameters using normal distribution.
    
    Parameters:
    ns (int): Number of samples
    r (float): Mean value for r
    k (float): Mean value for k
    sigma_N (float): Mean value for sigma.N
    error_size (float): Standard deviation factor for normal distribution
    
    Returns:
    xr.Dataset: Dataset containing sampled values for r, k, and sigma_N
    """
    pars = {
        'r': ("samples", np.random.normal(r, error_size * r, ns)),
        'k': ("samples", np.random.normal(k, error_size * k, ns)),
        'sigma_N': ("samples", np.random.normal(sigma_N, error_size * sigma_N, ns))
    }
    
    return xr.Dataset(pars)

def ricker_sim(X, params):

    """
    Simulate the next population value using the Ricker model.
    
    Parameters:
    N (float): Current population size
    params (dict): Dictionary containing parameters 'r' and 'k'
    
    Returns:
    float: Next population size
    """
    
    return X*np.exp(params['r']*(1 - X/params['k']))

def observations(r, k, N_init, sigma_N, error_size, tsteps):
    """
    Generate time-series data using the Ricker model with stochasticity.
    
    Parameters:
    r (float): Growth rate parameter
    k (float): Carrying capacity
    N_init (float): Initial population size
    sigma_N (float): Observation noise
    error_size (float): Error factor for parameter variation
    tsteps (int): Number of time steps
    
    Returns:
    xr.Dataset: Dataset containing true dynamics, processed values, and noise
    """
    params_true = {'r': r, 'k': k, 'sigma_N': sigma_N}
    dyn_true = np.zeros(tsteps)
    dyn_proc = np.zeros(tsteps)
    
    dyn_true[0] = N_init
    dyn_proc[0] = np.random.normal(dyn_true[0], params_true['sigma_N'])
    
    for i in range(1, tsteps):
        dyn_true[i] = ricker_sim(dyn_true[i - 1], params_true)
        dyn_proc[i] = np.random.normal(dyn_true[i], params_true['sigma_N'])
        params_true = sample_pars(1, r, k, sigma_N, error_size)
    
    sigma_N_values = np.sqrt((dyn_true - dyn_proc) ** 2)
    
    return xr.Dataset({
        "dyn_true": ("time", dyn_true),
        "dyn_proc": ("time", dyn_proc),
        "sigma_N": ("time", sigma_N_values)
    }, coords={"time": np.arange(tsteps)})


def plot_observations(dat_train, plot=True):
    plt.plot(dat_train.coords['time'].values, dat_train["dyn_true"], color = "black", label = "True dynamic")
    plt.plot(dat_train.coords['time'].values, dat_train["dyn_proc"], color ="blue", label = "Observed dynamic")
    plt.ylabel("Relative size")
    plt.xlabel("Time")
    plt.legend()
    if plot:
        plt.show()

def plot_setup(plot=True):
    fig, ax = plt.subplots(1, 1)
    ax.fill_between(np.arange(horiz),climatological_mean+2*climatological_std, climatological_mean-2*climatological_std, color = "lightgray")
    ax.plot(np.arange(horiz), output.squeeze(), color="blue", linewidth=0.8, alpha =0.2)
    ax.plot(np.full((horiz, 1), climatological_mean), color="red", linewidth=0.9, alpha =0.8, label = 'climatological mean')
    ax.plot(np.arange(horiz), y_obs[:horiz], color="black", linewidth=0.9, alpha =0.9, label = 'observations')
    ax.set_xlabel("Time")
    ax.set_ylabel("Relative size")
    plt.legend()
    if plot:
        plt.show()

def plot_mae(plot=True):

    fig, ax = plt.subplots(1, 1)
    ax.plot(ensemble_error.transpose(), color= "lightgray")
    ax.plot(ensemble_mean_error, color= "black", label = "Ensemble average")
    ax.plot(climatological_error.transpose(), color= "red", label = "Climatological average")
    ax.set_xlabel("Time")
    ax.set_ylabel("Mean absolute error")
    plt.legend()
    if plot:
        plt.show()

# helper function
def compute_cdf(values):
    mu, sigma = norm.fit(values) # fit normal distribution to fc
    values_cdf = norm.cdf(values, loc = mu, scale = sigma)
    return values_cdf

def run_forecast(N_init, ensemble_size, time_horizon):

    params = sample_pars(ns=ensemble_size,
                        r=r,
                        k=k,
                        sigma_N=sigma_N,
                        error_size=parameter_error)

    # initialise forecast with IC uncertainty
    X = np.random.normal(loc=N_init, scale=IC_error, size = ensemble_size) 

    output = np.zeros((time_horizon, ensemble_size, 1))
    output[0,:,:] = X[:, np.newaxis]

    for t in range(1, time_horizon):

        params = sample_pars(ns=ensemble_size,
                        r=r,
                        k=k,
                        sigma_N=sigma_N,
                        error_size=parameter_error)
        X = output[t-1, :, :]
        sim = ricker_sim(X.squeeze(), params)
        output[t, :, :] = sim.values[:, np.newaxis]

    return output

def crps_over_time(time_horizon, forecast, observation):

    fc_crps = np.zeros(time_horizon)
    fc_sorted = np.sort(forecast, axis=1)

    for t in range(time_horizon):

        fc = fc_sorted[t,:,:].squeeze() # forecast at t
        fc_cdf  = compute_cdf(fc)
        observed = observation[t].values

        fc_array = xr.DataArray(coords={'rel_size': fc}, data=fc_cdf)
        obs_array = xr.DataArray(observed)

        fc_crps[t] = crps_cdf(fc_array, obs_array, threshold_dim='rel_size').total.values.round(6)

    return(fc_crps)

horiz = 50 # forecast horizon for forecast model
horiz_obs = 100 # forecast horizon for creating observational truth
clim_horiz = 1000 # forecast horizon during climatological forecast

ne = 500 # Ensemble size for climatology (use same size for forecast).

# True initial conditions
r = 0.05 # growth rate
k = 1 # carrying capacity
sigma_N = 0.001 # observation error for creating observational truth 
N_init = k # set initial conditions to carrying capacity for steady state dynamics

parameter_error = 0.04 # relative precision for scale of parameter distribution
IC_error = 0.0001 # assumed initial conditions error

# simulate observations with observation error and stochastic parameters
dat_train = observations(r = r, k=k, N_init = N_init, sigma_N = sigma_N, 
                          error_size = parameter_error, tsteps = clim_horiz) 
# plot_observations(dat_train)
y_obs = dat_train['dyn_proc'][-horiz_obs:]

N_init = y_obs[0] # initial conditions for climatological forecast
print("Initial conditions for climatological forecast: ", N_init.values)

# Create climatology with long-term simulation and error propagation in r and k and IC. 
climatology = run_forecast(N_init=y_obs[0], ensemble_size=500, time_horizon=clim_horiz)

print("Climatological mean: ", climatology.squeeze().mean())
print("Climatological SD: ", climatology.squeeze().std())

climatological_mean = climatology.squeeze().mean()
climatological_std = climatology.squeeze().std()

# use saturated climatological distribution for comparison with forecast distribution
climatology_short = climatology[-horiz:, :, :] 

# initial observation as initial conditions for forecast
# Run forecast from n = 0 over horiz with 500 members with error propagation in r and k and IC. 
output = run_forecast(N_init=y_obs[0], ensemble_size=500, time_horizon=horiz)

# MAE 
ensemble_error = (output.squeeze().transpose()-y_obs[:horiz].values) # Absolute error
ensemble_mean_error = ensemble_error.mean(axis=0) # Mean absolute error
climatological_error = (np.full((horiz), climatological_mean) - y_obs[:horiz].values) # absolute mean error

plot_setup(plot=True)
plot_mae(plot=True)

# CRPS 
# Compute the crps for all time steps

crps_fc = crps_over_time(horiz, output, y_obs)
crps_clim = crps_over_time(horiz, climatology_short, y_obs)
crpss = (1 - crps_fc/crps_clim)

fig, ax = plt.subplots(1, 1)
ax.plot(crpss)
ax.set_ylabel("CRPSS")
ax.set_xlabel("Time")
plt.show()

# Now do this over multiple time steps with different initalisation of N_init from y_obs


def compute_crpss(i):
    """Function to run forecast, compute CRPS, and return CRPSS for one iteration."""

    output = run_forecast(N_init=y_obs[i], ensemble_size=500, time_horizon=horiz_obs)
    climatology = run_forecast(N_init=y_obs[i], ensemble_size=500, time_horizon=clim_horiz)
    climatology_short = climatology[-horiz_obs:, :, :] 

    y_obs_short = y_obs[i:]

    crps_fc = crps_over_time(len(y_obs_short), output, y_obs_short)
    crps_clim = crps_over_time(len(y_obs_short), climatology_short, y_obs_short)
    return (1 - crps_fc/crps_clim)

# Run in parallel (uses all but one available CPU cores)
crpss_results = Parallel(n_jobs=-1)(delayed(compute_crpss)(i) for i in range(horiz))
# reset worker usage to 1.
with parallel_backend("loky"):
    pass

# Store results in preallocated array
shifted_matrix = np.full((horiz, 2 * horiz), np.nan)
for i in range(horiz):
    shifted_matrix[i, i:] = crpss_results[i]  # Shifted placement

print("CRPSS shifted matrix shape:", shifted_matrix.shape)
print("CRPSS shifted_matrix rows are varying initial forecast times")
print("CRPSS shifted_matrix columns are lead times")


fig, ax = plt.subplots(1, 1)
sns.heatmap(shifted_matrix, cmap="viridis", annot=False, linewidths=0.5)
plt.show()

print(np.nanmean(shifted_matrix, axis=0).shape)
predicted_lead_times = shifted_matrix[:,horiz:]

fig, ax = plt.subplots(1, 1)
#ax.fill_between(np.arange(horiz), 
#    np.mean(predicted_lead_times, axis=0)+np.std(predicted_lead_times, axis=0), 
#    np.mean(predicted_lead_times, axis=0)-np.std(predicted_lead_times, axis=0), color = "lightgray")
ax.plot(np.mean(predicted_lead_times, axis=0))
plt.show()

fig, ax = plt.subplots(1, 1)
ax.boxplot(predicted_lead_times, vert=True, patch_artist=True)
plt.show()