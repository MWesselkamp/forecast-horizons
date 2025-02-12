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

def crps_on_timestep(forecast, observation):


    fc_sorted = np.sort(forecast, axis=0)

    fc = fc_sorted.squeeze() # forecast at t
    fc_cdf  = compute_cdf(fc)
    observed = observation.values

    fc_array = xr.DataArray(coords={'rel_size': fc}, data=fc_cdf)
    obs_array = xr.DataArray(observed)

    fc_crps = crps_cdf(fc_array, obs_array, threshold_dim='rel_size').total.values.round(6)

    return(fc_crps)

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

def compute_crpss_parallel(i):
    """Function to run forecast, compute CRPS, and return CRPSS parallel for one iteration."""

    output = run_forecast(N_init=y_obs[i], ensemble_size=500, time_horizon=horiz)
    climatology = run_forecast(N_init=y_obs[i], ensemble_size=500, time_horizon=horiz)
    climatology_short = climatology[-horiz:, :, :] 

    y_obs_short = y_obs[i:(i+horiz)]

    crps_fc = crps_over_time(len(y_obs_short), output, y_obs_short)
    crps_clim = crps_over_time(len(y_obs_short), climatology_short, y_obs_short)
    return (1 - crps_fc/crps_clim)

horiz = 25 # forecast horizon for forecast model
horiz_obs = 100 # forecast horizon for creating observational truth
clim_horiz = 1000 # forecast horizon during climatological forecast

ne = 500 # Ensemble size for climatology (use same size for forecast).

# True initial conditions
r = 0.05 # growth rate
k = 1 # carrying capacity
sigma_N = 0.00 # 1 # observation error for creating observational truth 
N_init = k # set initial conditions to carrying capacity for steady state dynamics

parameter_error = 0.03 # relative precision for scale of parameter distribution
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
# Or use the estimated climatological distribution, which is then always the same
climatological_distribution = np.random.normal(loc=climatological_mean, scale=climatological_std, size=ne)

# initial observation as initial conditions for forecast
# Run forecast from n = 0 over horiz with 500 members with error propagation in r and k and IC. 
output = run_forecast(N_init=y_obs[0], ensemble_size=500, time_horizon=horiz)

# MAE 
ensemble_error = (output.squeeze().transpose()-y_obs[:horiz].values) # Absolute error
ensemble_mean_error = ensemble_error.mean(axis=0) # Mean absolute error
climatological_error = (np.full((horiz), climatological_mean) - y_obs[:horiz].values) # absolute mean error

#plot_setup(plot=True)
#plot_mae(plot=True)

# CRPS 
# Compute the crps for all time steps

crps_fc = crps_over_time(horiz, output, y_obs)
crps_clim = crps_over_time(horiz, climatology_short, y_obs)
crpss = (1 - crps_fc/crps_clim)

fig, ax = plt.subplots(1, 1)
ax.plot(crpss)
ax.set_ylabel("CRPSS")
ax.set_xlabel("Lead Time")
plt.show()

# Forecast for specific day at different horizons, i.e. from different initalisation of N_init from y_obs

crps_fc = np.zeros((horiz, 1))
crpss_list = np.zeros((horiz, 1))

observed_subset = y_obs[:(horiz+1)]
print("Observations on day: ", (horiz))
observed_fh = observed_subset[horiz] # We look only at one day at a time, here at day horiz.

 # we use the estimated climatological distribution, hence this will always be the same.
crps_clim = np.full((horiz, 1), crps_on_timestep(climatological_distribution, observed_fh))

print("Iterating over forecast times")

for i in range(horiz):

    print("Initial forecast time: ", i)
    print("Run forecast over horizon: ", (horiz-i))

    output = run_forecast(N_init=observed_subset[i], ensemble_size=500, time_horizon=(horiz-i))

    forecast_distribution = output[-1, ...] # forecast distribution at horizon for specific day

    crps_fc[i] = crps_on_timestep(forecast_distribution, observed_fh)
    crpss_list[i] = (1 - crps_fc[i]/crps_clim[i])

print("Finished iteration.")

fig, ax = plt.subplots(1, 1)
ax.plot(crpss_list[::-1])
ax.set_ylabel("CRPSS")
ax.set_xlabel("Lead time")
plt.show()

# Now forecast at all horizons over multiple time steps with different initalisation of N_init from y_obs

initial_forecast_times = 25

crps_fc = np.zeros((initial_forecast_times, horiz))
crps_clim = np.zeros((initial_forecast_times, horiz))
crpss_list = np.zeros((initial_forecast_times, horiz))


print("Iterating over forecast times")
for i in range(initial_forecast_times):

    output = run_forecast(N_init=y_obs[i], ensemble_size=500, time_horizon=horiz)
    #climatology = run_forecast(N_init=y_obs[i], ensemble_size=500, time_horizon=clim_horiz)
    #climatology_short = climatology[-horiz:, :, :] 

    y_obs_short = y_obs[i:(i+horiz)]

    crps_fc[i,:] = crps_over_time(horiz, output, y_obs_short)
    crps_clim[i,:] = crps_over_time(horiz, climatology_short, y_obs_short)
    crpss_list[i,:] = (1 - crps_fc[i,:]/crps_clim[i,:])
print("Finished iteration.")

fig, ax = plt.subplots(1, 1)
ax.plot(crpss_list.transpose(), color = "lightblue")
ax.set_ylabel("CRPSS")
ax.set_xlabel("Lead Time")
plt.show()

fig, ax = plt.subplots(1, 1)
ax.plot(np.fliplr(crpss_list).diagonal(), color = "salmon")
ax.set_ylabel("CRPSS")
ax.set_xlabel("Initial forecast time")
ax.set_title("Forecast skill at Day 50 from initial forecast times")
plt.show()

fig, ax = plt.subplots(1, 1)
sns.heatmap(crpss_list, cmap="Greys", annot=False, linewidths=0.5)
plt.show()

fig, ax = plt.subplots(1, 1)
ax.fill_between(np.arange(horiz), 
    np.mean(crpss_list, axis=0)+np.std(crpss_list, axis=0), 
    np.mean(crpss_list, axis=0)-np.std(crpss_list, axis=0), color = "lightgray")
ax.plot(np.mean(crpss_list, axis=0))
ax.set_ylabel("CRPSS")
ax.set_xlabel("Lead Time")
plt.show()

# Now shift matrix.
shift = True

if shift: 
    # Store results in preallocated array
    matrix = np.full((initial_forecast_times, 2 * horiz), np.nan)
    for i in range(initial_forecast_times):
        matrix[i, i:(i+horiz)] = crpss_list[i]  # Shifted placement
    print("CRPSS shifted matrix shape:", matrix.shape)
    print(np.nanmean(matrix, axis=0).shape)

    fig, ax = plt.subplots(1, 1)
    sns.heatmap(matrix, cmap="Greys", annot=False, linewidths=0.5)
    plt.show()

    fig, ax = plt.subplots(1, 1)
    sns.heatmap(matrix.transpose(), cmap="Greys", annot=False, linewidths=0.5)
    plt.show()

    predicted_lead_times = matrix[:, initial_forecast_times:(horiz-initial_forecast_times)]
    fig, ax = plt.subplots(1, 1)
    ax.boxplot(predicted_lead_times, vert=True, patch_artist=True)
    plt.show()

    fig, ax = plt.subplots(1, 1)
    ax.plot(np.mean(predicted_lead_times, axis=0))
    plt.show()

# Do this with parallel processing

parallel = False

if parallel:
    # Run in parallel (uses all but one available CPU cores)
    crpss_results = Parallel(n_jobs=-1)(delayed(compute_crpss_parallel)(i) for i in range(10))
    # reset worker usage to 1.
    with parallel_backend("loky"):
        pass

    matrix = np.zeros((10, horiz))
    matrix[:] = np.array(crpss_results)

    predicted_lead_times = matrix

