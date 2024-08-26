import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class RickerPredation(nn.Module):
    """
    A neural network model representing a predation system based on the Ricker model.
    """

    def __init__(self, initial_conditions, params, noise=None):
        super().__init__()
        self.initial_conditions = initial_conditions
        self.noise = noise
        self.model_params = self._initialize_parameters(params, noise)

    @classmethod
    def create_instance(cls, initial_conditions, params, noise=None):
        # Creating an instance of the class within the class
        return cls(initial_conditions, params, noise=None)
    def _initialize_parameters(self, params, noise):
        """Initialize model parameters with optional noise."""
        param_list = params + [noise] if noise is not None else params

        # Convert the list to a tensor and ensure it's independent with gradient tracking enabled
        param_tensor = torch.tensor(param_list, dtype=torch.double).clone().detach().requires_grad_(True)

        return torch.nn.Parameter(param_tensor)

    def forward(self, forcing):
        """Forward pass to compute the model's output based on forcing input."""
        params = self._unpack_parameters()
        return self._compute_dynamics(forcing, params)

    def _unpack_parameters(self):
        """Unpack parameters for easier readability."""
        if self.noise is not None:
            return self.model_params[:10], self.model_params[10]
        return self.model_params[:10], None

    def _compute_dynamics(self, forcing, params):
        """Compute the population dynamics."""

        (alpha1, beta1, gamma1, bx1, cx1, alpha2, beta2, gamma2, bx2, cx2), sigma = params

        forcing = forcing.squeeze()
        num_steps = len(forcing)

        species_1 = torch.full((num_steps,), self.initial_conditions['species1'](), dtype=torch.double)
        species_2 = torch.full((num_steps,), self.initial_conditions['species2'](), dtype=torch.double)

        # Stack them to create a 2D tensor
        out = torch.stack([species_1, species_2])

        for i in range(num_steps - 1):
            out[0, i + 1] = out[0, i] * torch.exp(alpha1 * (1 - beta1 * out[0, i] - gamma1 * out[1, i]
                                                            + bx1 * forcing[i] + cx1 * forcing[i] ** 2))
            out[1, i + 1] = out[1, i] * torch.exp(alpha2 * (1 - beta2 * out[1, i] - gamma2 * out[0, i]
                                                            + bx2 * forcing[i] + cx2 * forcing[i] ** 2))
            if sigma is not None:
                noise_term = sigma * torch.normal(mean=torch.tensor([0.0]), std=torch.tensor([0.1]))
                out[:, i + 1] += noise_term

        return out

    def simulate_forcing(self, timesteps, freq_s=365, phase_shift=0, add_trend=False, add_noise=False):
        """
        Simulate forcing data over a given number of timesteps.

        Parameters:
        - timesteps (int): Number of timesteps to simulate.
        - freq_s (int): Frequency scale, defaults to 365 (days in a year).
        - phase_shift (float): Phase shift to start the sine curve at a different point.
        - add_trend (bool): Whether to add a linear trend to the sine wave.
        - add_noise (bool): Whether to add noise to the sine wave.

        Returns:
        - numpy.ndarray: Simulated forcing data.
        """
        freq = timesteps / freq_s
        x = np.arange(timesteps)

        # Apply the phase shift to the sine function
        y = np.sin(2 * np.pi * freq * (x / timesteps) + phase_shift)

        if add_trend:
            y += np.linspace(0, 0.1, timesteps)

        if add_noise:
            y += np.random.normal(0, 0.08, timesteps)

        return np.round(y, 4)

    def create_observations(self, years, forcing = None, phase_shift = 0, split_data=True):
        """
        Create observations.
        """
        timesteps = 365 * years
        phase_shift = np.pi * phase_shift

        train_size = 365 * (years - 1)

        if forcing is None:
            forcing = self.simulate_forcing(timesteps=timesteps,
                                            phase_shift = phase_shift,
                                            add_noise=True)

        observation_model = self.create_instance(initial_conditions = self.initial_conditions,
                                                 params=self.model_params,
                                                 noise=self.noise)
        observed_dynamics = observation_model.forward(forcing=torch.tensor(forcing, dtype=torch.double))

        if split_data:
            return self._process_observations(observed_dynamics, forcing, train_size)
        else:
            return torch.tensor(observed_dynamics)

    def create_ensemble(self, ensemble_size, years=1, phase_shift= 0):

        ensemble = [
            self.create_observations(years=years,
                                     phase_shift=phase_shift).get('y_test') for _ in range(ensemble_size)
                    ]
        ensemble = torch.stack(ensemble)

        return ensemble

    def iterate_ensemble(self, forcing, ensemble_size, years=1, phase_shift= 0):

        self.ensemble = [
            self.create_observations(years=years,
                                     forcing = forcing,
                                     phase_shift=phase_shift,
                                     split_data=False) for _ in range(ensemble_size)
                    ]

        ensemble = torch.stack(self.ensemble)

        return ensemble

    def _process_observations(self, observed_dynamics, forcing, train_size):
        """Process and split the observed dynamics into training and testing sets."""
        y = observed_dynamics.detach().clone().requires_grad_(True)

        y_train, y_test = y[:,:train_size], y[:,train_size:]
        forcing_train, forcing_test = forcing[:train_size], forcing[train_size:]

        climatology = y_train#.view((-1, 365))
        sigma = np.std(climatology.detach().numpy())
        sigma_train = np.tile(sigma, reps=(y_train.shape[1] // 365))
        sigma_test = sigma

        return {
            'y_train': y_train,
            'y_test': y_test,
            'sigma_train': sigma_train,
            'sigma_test': sigma_test,
            'x_train': forcing_train,
            'x_test': forcing_test,
            'climatology': climatology
        }
    def plot_time_series(self, observations_dict, series_name):
        """
        Plot the specified time series from the observations dictionary.

        Parameters:
        - observations_dict: Dictionary containing time series data.
        - series_name: The key of the time series to plot (e.g., 'y_train', 'y_test', etc.).
        """
        if series_name not in observations_dict:
            raise ValueError(f"Series name '{series_name}' not found in observations.")

        series_data = observations_dict[series_name]

        if not isinstance(series_data, np.ndarray):
            series_data = series_data.detach().numpy()

        if series_data.shape[0] > 1:
            series_data = series_data.transpose()

        plt.figure(figsize=(10, 5))
        plt.plot(series_data, label=series_name)
        plt.title(f"Time Series: {series_name}")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.show()
    def __repr__(self):
        param_values = [f"{name}: {value.item()}" for name, value in zip(
            ["alpha1", "beta1", "gamma1", "bx1", "cx1", "alpha2", "beta2", "gamma2", "bx2", "cx2"],
            self.model_params[:10]
        )]
        sigma_str = f"sigma: {self.noise}" if self.noise is not None else "sigma: N/A"
        return "Model Parameters:\n" + ", ".join(param_values) + f", {sigma_str}"

class Ricker_Ensemble(nn.Module):
    """
     Single-Species extended Ricker with Ensemble prediction.
    """

    def __init__(self, params, noise=None, initial_uncertainty = None):

        super().__init__()

        if (not noise is None) & (not initial_uncertainty is None):
            self.model_params = torch.nn.Parameter(torch.tensor(params + [noise] + [initial_uncertainty], requires_grad=True, dtype=torch.double))
            self.initial_uncertainty = initial_uncertainty
            self.noise = noise
        elif (not noise is None) & (initial_uncertainty is None):
            self.model_params = torch.nn.Parameter(torch.tensor(params + [noise], requires_grad=True, dtype=torch.double))
            self.initial_uncertainty = initial_uncertainty
            self.noise = noise
        elif (noise is None) & (not initial_uncertainty is None):
            self.model_params = torch.nn.Parameter(torch.tensor(params + [initial_uncertainty], requires_grad=True, dtype=torch.double))
            self.initial_uncertainty = initial_uncertainty
            self.noise = noise
        elif (noise is None) & (initial_uncertainty is None):
            self.model_params = torch.nn.Parameter(torch.tensor(params, requires_grad=True, dtype=torch.double))
            self.noise = noise
            self.initial_uncertainty = initial_uncertainty

    def forward(self, N0, Temp, ensemble_size=15):

        if (not self.noise is None) & (not self.initial_uncertainty is None):
            alpha, beta, bx, cx, sigma, phi = self.model_params
        elif (not self.noise is None) & (self.initial_uncertainty is None):
            alpha, beta, bx, cx, sigma = self.model_params
        elif (self.noise is None) & (not self.initial_uncertainty is None):
            alpha, beta, bx, cx, phi = self.model_params
        else:
            alpha, beta, bx, cx = self.model_params

        Temp = Temp.squeeze()

        if not self.initial_uncertainty is None:
            initial = N0 + phi * torch.normal(torch.zeros((ensemble_size)), torch.repeat_interleave(torch.tensor([.1, ]), ensemble_size))
            out = torch.zeros((len(initial), len(Temp)), dtype=torch.double)
        else:
            initial = N0
            out = torch.zeros((1, len(Temp)), dtype=torch.double)

        out[:,0] = initial  # initial value

        if not self.noise is None:
            for i in range(len(Temp) - 1):
                out[:,i + 1] = out.clone()[:,i] * torch.exp(
                    alpha * (1 - beta * out.clone()[:,i] + bx * Temp[i] + cx * Temp[i] ** 2))# \
                          #   + sigma * torch.normal(mean=torch.tensor([0.0, ]), std=torch.tensor([1.0, ]))
            var = sigma * 2#torch.normal(mean=torch.tensor([0.0, ]), std=torch.tensor([1.0, ]))
            out_upper = out + torch.repeat_interleave(var, len(Temp)) #+ torch.full_like(out, var.item())
            out_lower = out - torch.repeat_interleave(var, len(Temp))  #- torch.full_like(out, var.item())

            return out, [out_upper, out_lower]

        else:
            for i in range(len(Temp) - 1):
                out[:,i + 1] = out.clone()[:,i] * torch.exp(
                    alpha * (1 - beta * out.clone()[:,i] + bx * Temp[i] + cx * Temp[i] ** 2))

            return out, None

    def get_fit(self):

        return {"alpha": self.model_params[0].item(), \
            "beta": self.model_params[1].item(), \
                "bx": self.model_params[2].item(), \
                    "cx": self.model_params[3].item(), \
               "sigma": self.noise if self.noise is None else self.model_params[4].item(), \
               "phi": self.initial_uncertainty if self.initial_uncertainty is None else (self.model_params[5].item() if self.noise is not None else self.model_params[4].item())
                }

    def forecast(self, N0, Temp, ensemble_size=15):

        if (not self.noise is None) & (not self.initial_uncertainty is None):
            alpha, beta, bx, cx, sigma, phi = self.model_params
        elif (not self.noise is None) & (self.initial_uncertainty is None):
            alpha, beta, bx, cx, sigma = self.model_params
        elif (self.noise is None) & (not self.initial_uncertainty is None):
            alpha, beta, bx, cx, phi = self.model_params
        else:
            alpha, beta, bx, cx = self.model_params

        Temp = Temp.squeeze()

        if not self.initial_uncertainty is None:
            initial = N0 + phi * torch.normal(torch.zeros((ensemble_size)),
                                              torch.repeat_interleave(torch.tensor([.1, ]), ensemble_size))
            out = torch.zeros((len(initial), len(Temp)), dtype=torch.double)
        else:
            initial = N0
            out = torch.zeros((1, len(Temp)), dtype=torch.double)

        out[:, 0] = initial  # initial value

        if not self.noise is None:
            for i in range(len(Temp) - 1):
                out[:, i + 1] = out.clone()[:, i] * torch.exp(
                    alpha * (1 - beta * out.clone()[:, i] + bx * Temp[i] + cx * Temp[i] ** 2))  \
                    + sigma * torch.normal(mean=torch.tensor([0.0, ]), std=torch.tensor([0.1, ]))

            #out = out + sigma * torch.normal(mean=torch.tensor([0.0, ]), std=torch.tensor([1.0, ]))

            return out

        else:
            for i in range(len(Temp) - 1):
                out[:, i + 1] = out.clone()[:, i] * torch.exp(
                    alpha * (1 - beta * out.clone()[:, i] + bx * Temp[i] + cx * Temp[i] ** 2))

            return out
