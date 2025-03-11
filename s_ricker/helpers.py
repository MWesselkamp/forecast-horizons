import sys
print(sys.path)

import numpy as np
import xarray as xr
from scipy.stats import norm
from scores.probability import crps_cdf

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


def compute_cdf(values):
    mu, sigma = norm.fit(values) # fit normal distribution to fc
    values_cdf = norm.cdf(values, loc = mu, scale = sigma)
    return values_cdf

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