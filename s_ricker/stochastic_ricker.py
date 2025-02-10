import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

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


def plot_observations(dat_train):
    plt.plot(dat_train.coords['time'].values, dat_train["dyn_true"], color = "black", label = "True dynamic")
    plt.plot(dat_train.coords['time'].values, dat_train["dyn_proc"], color ="blue", label = "Observed dynamic")
    plt.ylabel("Relative size")
    plt.xlabel("Time")
    plt.legend()
    plt.show()

horiz = 30

# True initial conditions
r = 0.05
k = 1
sigma_N = 0.001 # observation error
N_init = k # set initial conditions to carrying capacity for steady state dynamics

parameter_error = 0.04 # precision for parameter samples
IC_error = 0.0001 # initial conditions error

dat_train = observations(r = r, k=k, N_init = N_init, sigma_N = sigma_N, 
                          error_size = parameter_error, tsteps = horiz) # simulate observations with observation error and stochastic parameters
# plot_observations(dat_train)

N_init = dat_train['dyn_proc'][0] # initial conditions for forecast
print("Initial conditions", N_init.values)

# Create climatology with long-term simulation and error propagation 

ne = 500 # Ensemble size. production run should be 200 - 5000, depending on what your computer can handle
clim_horiz = 1000 #days, forecast horizon during forecast

params = sample_pars(ns=ne,
                     r=r,
                     k=k,
                     sigma_N=sigma_N,
                     error_size=parameter_error)

X = np.random.normal(loc=N_init, scale=IC_error, size = ne)
#plt.hist(X)
#plt.show()

climatology = np.zeros((clim_horiz, ne, 1))
climatology[0,:,:] = X[:, np.newaxis]

for t in range(1, clim_horiz):

    params = sample_pars(ns=ne,
                     r=r,
                     k=k,
                     sigma_N=sigma_N,
                     error_size=parameter_error)
    X = climatology[t-1, :, :]
    sim = ricker_sim(X.squeeze(), params)
    climatology[t, :, :] = sim.values[:, np.newaxis]

# Run forecast from n = 0 over horiz

ne = 500 # Ensemble size. production run should be 200 - 5000, depending on what your computer can handle

params = sample_pars(ns=ne,
                     r=r,
                     k=k,
                     sigma_N=sigma_N,
                     error_size=parameter_error)

X = np.random.normal(loc=N_init, scale=IC_error, size = ne)
#plt.hist(X)
#plt.show()

output = np.zeros((horiz, ne, 1))
output[0,:,:] = X[:, np.newaxis]

for t in range(1, horiz):

    params = sample_pars(ns=ne,
                     r=r,
                     k=k,
                     sigma_N=sigma_N,
                     error_size=parameter_error)
    X = output[t-1, :, :]
    sim = ricker_sim(X.squeeze(), params)
    output[t, :, :] = sim.values[:, np.newaxis]

plt.plot(climatology[-horiz:, :, :].squeeze(), color="lightgray", linewidth=0.8)
plt.plot(output.squeeze(), color="blue", linewidth=0.8)
plt.xlabel("Time")
plt.ylabel("Relative size")
plt.show()