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


# Set global font sizes
plt.rcParams.update({
    'font.size': 14,          # Default font size
    'axes.titlesize': 14,     # Title font size
    'axes.labelsize': 14,     # X and Y label font size
    'xtick.labelsize': 12,    # X tick labels font size
    'ytick.labelsize': 12,    # Y tick labels font size
    'legend.fontsize': 12,    # Legend font size
    'figure.titlesize': 14    # Figure title font size
})


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
    plt.xlabel("Lead Time")
    plt.legend()
    if plot:
        plt.show()

def plot_setup(ax):
    ax.fill_between(np.arange(horiz),climatological_mean+2*climatological_std, climatological_mean-2*climatological_std, color = "lightgray", alpha = 0.6)
    ax.plot(np.arange(horiz), output.squeeze(), color="steelblue", linewidth=0.8, alpha =0.5)
    ax.plot(np.full((horiz, 1), climatological_mean), color="dimgray", linewidth=0.9, label = 'Climatological average')
    ax.plot(np.arange(horiz), y_obs[:horiz], color="red", linewidth=0.9, alpha =0.9, label = 'Observation')
    ax.set_xlabel("Lead Time")
    ax.set_ylabel("Relative size")
    ax.legend()

def plot_mae(ax):

    ax.fill_between(np.arange(horiz), climatological_mean_error + climatological_mean_error_spread, climatological_mean_error - climatological_mean_error_spread, color = "lightgray", alpha = 0.6)
    ax.plot(climatological_mean_error, color= "dimgray", label = "Climatological average")
    ax.fill_between(np.arange(horiz), ensemble_mean_error + ensemble_mean_error_spread, ensemble_mean_error - ensemble_mean_error_spread, color = "steelblue", alpha = 0.6)
    ax.plot(ensemble_mean_error, color= "blue", label = "Ensemble average")
    ax.set_xlabel("Lead Time")
    ax.set_ylabel("Absolute error")
    ax.legend()

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
horiz_obs = 150 # forecast horizon for creating observational truth
clim_horiz = 1000 # forecast horizon during climatological forecast

ne = 1000 # Ensemble size for climatology (use same size for forecast).

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
climatology = run_forecast(N_init=y_obs[0], ensemble_size=ne, time_horizon=clim_horiz)

print("Climatological mean: ", climatology.squeeze().mean())
print("Climatological SD: ", climatology.squeeze().std())

climatological_mean = climatology.squeeze().mean()
climatological_std = climatology.squeeze().std()

# use saturated climatological distribution for comparison with forecast distribution
climatology_short = climatology[-horiz:, :, :] 
print(climatology_short.shape)
# Use estimated climatological distribution from saturated climatology, which will always be the same
climatological_distribution = np.random.normal(loc=climatological_mean, scale=climatological_std, size=ne)
climatology_short = np.tile(climatological_distribution[:, np.newaxis], horiz).transpose()[:,:,np.newaxis]
print(climatology_short.shape)

# initial observation as initial conditions for forecast
# Run forecast from n = 0 over horiz with 500 members with error propagation in r and k and IC. 
output = run_forecast(N_init=y_obs[0], ensemble_size=500, time_horizon=horiz)

# MAE 
ensemble_error = abs(output.squeeze().transpose()-y_obs[:horiz].values) # Absolute error
ensemble_mean_error = ensemble_error.mean(axis=0) # Mean absolute error
ensemble_mean_error_spread = ensemble_error.std(axis=0) #  absolute error spread
climatological_error = abs(climatology_short.squeeze().transpose() - y_obs[:horiz].values) # absolute error
climatological_mean_error = climatological_error.mean(axis=0) # mean absolute error
climatological_mean_error_spread = climatological_error.std(axis=0) #  absolute error spread

# Spread-error
def rolling_pearson(x, y, window):
    """Computes rolling Pearson correlation over a moving window."""
    corrs = np.full(len(x), np.nan)  # Initialize array with NaNs
    for i in range(len(x) - window + 1):
        corrs[i + window - 1] = np.corrcoef(x[i:i+window], y[i:i+window])[0, 1]
    return corrs

spread_error_correlation = rolling_pearson(ensemble_mean_error, ensemble_mean_error_spread, window=8)

# CRPS :Compute the crps for all time steps
crps_fc = crps_over_time(horiz, output, y_obs)
crps_clim = crps_over_time(horiz, climatology_short, y_obs)
crpss = (1 - crps_fc/crps_clim)

fig, ax = plt.subplots(1, 1, figsize = (5, 4), constrained_layout=True)
plot_setup(ax = ax)
plt.savefig("s_ricker/plots/setup.pdf")
plt.show()

fig, ax = plt.subplots(1, 1, figsize = (5, 4), constrained_layout=True)
plot_mae(ax = ax)
plt.savefig("s_ricker/plots/mae.pdf")
plt.show()

fig, ax = plt.subplots(1, 1, figsize = (5, 4), constrained_layout=True)
ax.hlines(y=0,xmin=0,xmax=horiz, linestyles="--", colors="black")
ax.plot(crpss, color = "darkblue")
ax.set_ylabel("CRPSS")
ax.set_xlabel("Lead Time")
plt.savefig("s_ricker/plots/crps.pdf")
plt.show()

fig, ax = plt.subplots(1, 1, figsize = (5, 4), constrained_layout=True)
ax.hist(spread_error_correlation,density=True, bins=15, histtype='stepfilled',)
ax.set_xlabel("Spread-error correlation")
ax.set_ylabel("Frequency")
plt.savefig("s_ricker/plots/spreaderror.pdf")
plt.show()


fig, ax = plt.subplots(2, 1, figsize = (4, 4), constrained_layout=True)
ax[0].hist(climatological_distribution,density=True, bins='auto', histtype='stepfilled', alpha=0.5, color="lightgray", label = "Climatology")
ax[0].hist(output[1,...],density=True, bins='auto', histtype='stepfilled', alpha=0.5, color="steelblue", label = "Forecast")
ax[0].vlines(x = y_obs[1], ymin=0, ymax=280, color="red", label = "Observation")
ax[0].set_xlabel("Relative size")
ax[0].set_ylabel("Frequency")
ax[0].legend(loc="upper right")
ax[1].hist(climatological_distribution,density=True, bins='auto', histtype='stepfilled', alpha=0.5, color="lightgray", label = "Climatology")
ax[1].hist(output[-1,...],density=True, bins='auto', histtype='stepfilled', alpha=0.5, color="steelblue", label = "Forecast")
ax[1].vlines(x = y_obs[horiz], ymin=0, ymax=100, color="red", label = "Observation")
ax[1].set_xlabel("Relative size")
ax[1].set_ylabel("Frequency")
plt.savefig("s_ricker/plots/distributions_at_horizon.pdf")
plt.show()

# Forecast for specific day at different horizons, i.e. from different initalisation of N_init from y_obs
horiz = 100
days = 50

crps_fc = np.zeros((days, horiz))
crpss_list = np.zeros((days, horiz))
crps_clim = np.zeros((days, horiz))

print("Iterating over observation days.")

for day in range(days):

    observed_subset = y_obs[day:(day + horiz)]
    observed_fh = observed_subset[-1] # We look only at one day at a time, here at day horiz.

    # we use the estimated climatological distribution, hence this will always be the same on the same day.
    crps_clim[day,:] = np.full((1, horiz), crps_on_timestep(climatological_distribution, observed_fh))

    for i in range(horiz):

        output = run_forecast(N_init=observed_subset[i], ensemble_size=500, time_horizon=(horiz-i))
        #plot_setup(time_horizon=(horiz-i), observed_ts=observed_subset[i:])

        forecast_distribution = output[-1, ...] # forecast distribution at horizon for specific day

        crps_fc[day, i] = crps_on_timestep(forecast_distribution, observed_fh)
        crpss_list[day, i] = (1 - crps_fc[day, i]/crps_clim[day, i])

print("Finished iteration.")

print("crps_fc:", crps_fc.shape)
print("crps_list:", crpss_list.shape)

daily_mean = np.mean(crpss_list[:,::-1].transpose(),  axis=1)
daily_std = np.std(crpss_list[:,::-1].transpose(),  axis=1)
daily_median = np.quantile(crpss_list[:,::-1].transpose(), q = 0.5, axis=1)
daily_qupper = np.quantile(crpss_list[:,::-1].transpose(), q = 0.75, axis=1)
daily_qlower = np.quantile(crpss_list[:,::-1].transpose(), q = 0.25, axis=1)

fig, ax = plt.subplots(1, 1, figsize = (5,4), constrained_layout = True)
#ax.plot(crpss_list[:,::-1].transpose(), color = "lightgray", alpha = 0.7, linewidth = 0.8)
ax.hlines( y=0, xmin = 0, xmax=horiz, linestyles="--", color = "black")
ax.fill_between(np.arange(horiz), daily_qupper, daily_qlower, alpha = 0.6, color = "salmon", label = "Interquartile range")
ax.plot(daily_median, color = "red", label = "Median")
ax.fill_between(np.arange(horiz), daily_mean + daily_std, daily_mean - daily_std, alpha = 0.6, color = "lightblue", label = "Spread")
ax.plot(daily_mean, color = "blue", label = "Mean")
ax.set_ylabel("CRPSS")
ax.set_xlabel("Forecast horizon")
ax.set_title("Scoring distribution at horizon (n = 50 [days])")
ax.set_ylim((-1,1))
ax.legend()
plt.savefig("s_ricker/plots/leadtime-distribution.pdf")
plt.show()


forecast_limits = np.where((crpss_list[:, ::-1] < 0).any(axis=1), np.argmax(crpss_list[:, ::-1] < 0, axis=1), np.nan)
print(forecast_limits)

forecast_limits_clean = forecast_limits[~np.isnan(forecast_limits)]
forecast_limit_average = np.mean(forecast_limits_clean)
average_forecast_limit = np.argmax(daily_mean < 0)
# Compute histogram

fig, ax = plt.subplots(1, 1, figsize = (5,4), constrained_layout = True)
ax.hist(forecast_limits_clean, bins=50, color='blue', alpha=0.7, edgecolor='black')
ax.set_xlabel("Forecast limit")
ax.set_ylabel("Frequency")
fig.tight_layout()
plt.savefig("s_ricker/plots/forecast-limit-hist.pdf")
plt.show()

fig, ax = plt.subplots(1, 1, figsize = (5,4), constrained_layout = True)
sns.kdeplot(forecast_limits_clean, fill=True, color='blue', alpha=0.6, clip=(0, np.inf))
# Labels and title
ax.vlines(x = forecast_limit_average, ymin=0, ymax=0.035, color = "red", label="Forecast limit average")
ax.vlines(x = average_forecast_limit, ymin=0, ymax=0.035, color = "black", label="Average forecast limit")
ax.set_xlabel("Forecast limit")
ax.set_ylabel("Smoothed Density")
ax.legend()
plt.savefig("s_ricker/plots/forecast-limit-distribution.pdf")
plt.show()