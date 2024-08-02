import numpy as np
import torch
from models import Ricker_Predation
from utils import simulate_temperature
def create_observations(years, observation_params, true_noise, full_dynamics=False):
    """
    Create observations using the Ricker Predation model.

    Parameters:
    years (int): Number of years to simulate.
    observation_params (list): Parameters for the observation model.
    true_noise (float): Noise level for the observation model.
    full_dynamics (bool): Whether to return full dynamics or split into training and testing sets.

    Returns:
    dict or tuple: If full_dynamics is True, returns the full observed dynamics and temperature.
                   Otherwise, returns a dictionary with training and testing sets, climatology, and standard deviation.
    """

    # Calculate time steps
    timesteps = 365 * years
    train_steps = 365 * (years - 1)
    test_steps = 365

    # Simulate temperature data
    temperature = simulate_temperature(timesteps=timesteps)

    # Initialize the observation model
    observation_model = Ricker_Predation(params=observation_params, noise=true_noise)

    # Generate observed dynamics
    observed_dynamics = observation_model(Temp=temperature)
    y = observed_dynamics[0, :].clone().detach().requires_grad_(True)

    # Split data into training and testing sets
    y_train, y_test = y[:train_steps], y[train_steps:]
    temp_train, temp_test = temperature[:train_steps], temperature[train_steps:]

    # Create climatology from training data
    climatology = y_train.view((years - 1), 365)
    sigma = np.std(climatology.detach().numpy(), axis=0)
    sigma_train = np.tile(sigma, reps=(years - 1))
    sigma_test = sigma

    # Return full dynamics if requested
    if full_dynamics:
        return observed_dynamics.detach().numpy(), temperature

    # Return split datasets and climatology
    return {
        'y_train': y_train,
        'y_test': y_test,
        'sigma_train': sigma_train,
        'sigma_test': sigma_test,
        'x_train': temp_train,
        'x_test': temp_test,
        'climatology': climatology
    }
