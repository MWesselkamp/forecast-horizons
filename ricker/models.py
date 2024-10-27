import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class RickerPredation(nn.Module):
    """
    A neural network model representing a predation system based on the Ricker model.
    """

    def __init__(self, initial_conditions, params, forcing_params, noise=None):
        super().__init__()
        self.initial_conditions = initial_conditions
        self.noise = noise
        self.model_params = self._initialize_parameters(params, noise)
        self.forcing_params = forcing_params
        self.resolution = self.forcing_params['resolution']
        self.phase_shift = self.forcing_params['phase_shift']

    @classmethod
    def create_instance(cls, initial_conditions, params, forcing_params, noise=None):
        # Creating an instance of the class within the class
        return cls(initial_conditions, params, forcing_params, noise=None)
    def _initialize_parameters(self, params, noise):
        """Initialize model parameters with optional noise."""
        if noise is not None:
            params['sigma'] = noise  # Add noise to the dictionary

        # Convert the parameter dictionary values to tensors and ensure they are independent with gradient tracking enabled
        param_dict = {key: torch.tensor(value, dtype=torch.double).clone().detach().requires_grad_(True)
                      for key, value in params.items()}

        # Create torch.nn.Parameter for each tensor
        param_dict = {key: torch.nn.Parameter(tensor) for key, tensor in param_dict.items()}

        return nn.ParameterDict(param_dict)

    def forward(self, forcing):
        """Forward pass to compute the model's output based on forcing input."""
        return self._compute_dynamics(forcing, self.model_params)

    def _compute_dynamics(self, forcing, params):
        """Compute the population dynamics."""

        # Unpack the model parameters
        alpha1 = params['alpha1']
        beta1 = params['beta1']
        gamma1 = params['gamma1']
        bx1 = params['bx1']
        cx1 = params['cx1']
        alpha2 = params['alpha2']
        beta2 = params['beta2']
        gamma2 = params['gamma2']
        bx2 = params['bx2']
        cx2 = params['cx2']
        sigma = params.get('sigma', None)  # sigma might not always be present

        sigma_forcing = 0.08

        forcing = forcing.squeeze()
        num_steps = len(forcing)

        species_1 = torch.full((num_steps,), self.initial_conditions['species1'](), dtype=torch.double)
        species_2 = torch.full((num_steps,), self.initial_conditions['species2'](), dtype=torch.double)

        # Stack them to create a 2D tensor
        out = torch.stack([species_1, species_2])

        # List to store Jacobians
        jacobians = []

        for i in range(num_steps - 1):

            # Detach the current state to ensure correct gradient computation for each timestep
            out_t = out[:, i].detach().clone().requires_grad_(True)

            z1 = torch.normal(bx1 * forcing[i] + cx1 * forcing[i] ** 2, sigma_forcing)
            z2 = torch.normal(bx2 * forcing[i] + cx2 * forcing[i] ** 2, sigma_forcing)

            out[0, i + 1] = out[0, i] * torch.exp(alpha1 * (1 - beta1 * out[0, i] - gamma1 * out[1, i] + z1))
            out[1, i + 1] = out[1, i] * torch.exp(alpha2 * (1 - beta2 * out[1, i] - gamma2 * out[0, i] + z2))
            if sigma is not None:
                out[:, i + 1] += sigma * torch.normal(mean=torch.tensor([0.0]), std=torch.tensor([0.1]))

            # Compute the Jacobian of the next state w.r.t. the current state
            jacobian = torch.autograd.functional.jacobian(
                lambda x: torch.stack([
                    x[0] * torch.exp(
                        alpha1 * (1 - beta1 * x[0] - gamma1 * x[1] + z1)),
                    x[1] * torch.exp(
                        alpha2 * (1 - beta2 * x[1] - gamma2 * x[0] + z2))
                ]), out_t)
            # Append the Jacobian to the list
            jacobians.append(jacobian)

        self.jacobians = torch.stack(jacobians)

        return out

    def simulate_forcing(self, timesteps, add_trend=False, add_noise=False):
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
        freq = timesteps / self.resolution
        x = np.arange(timesteps)

        phase_shift = np.pi * self.phase_shift

        # Apply the phase shift to the sine function
        y = np.sin(2 * np.pi * freq * (x / timesteps) + phase_shift)

        if add_trend:
            y += np.linspace(0, 0.1, timesteps)

        if add_noise:
            y += np.random.normal(0, 0.08, timesteps)

        return torch.tensor(np.round(y, 4), dtype=torch.double)

    def create_observations(self, years, forcing = None, split_data=True):
        """
        Create observations.
        """

        timesteps = self.resolution * years
        train_size = self.resolution * (years - 1)

        if forcing is None:
            forcing = self.simulate_forcing(timesteps=timesteps,
                                            add_noise=False)

        observed_dynamics = self.forward(forcing=forcing)

        if split_data:
            return self._process_observations(observed_dynamics, forcing, train_size)
        else:
            return torch.tensor(observed_dynamics)

    def create_ensemble(self, ensemble_size, forcing = None, years=1):

        ensemble = [
            self.create_observations(years=years,
                                     forcing=forcing).get('y_test') for _ in range(ensemble_size)
                    ]
        ensemble = torch.stack(ensemble)

        return ensemble

    def _process_observations(self, observed_dynamics, forcing, train_size):
        """Process and split the observed dynamics into training and testing sets."""
        y = observed_dynamics.detach().clone().requires_grad_(True)

        y_train, y_test = y[:,:train_size], y[:,train_size:]
        forcing_train, forcing_test = forcing[:train_size], forcing[train_size:]

        climatology = y_train#.view((-1, 365))
        sigma = np.std(climatology.detach().numpy())
        sigma_train = np.tile(sigma, reps=(y_train.shape[1] // self.resolution))
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

    def compute_lyapunov_exponent(self, num_timesteps = None):
        """
        Compute the Lyapunov exponent over time based on Rogers 2021.

        Parameters:
        - num_timesteps : number of timesteps to compute the lyapunovs over.

        Returns:
        - float: The Lyapunov exponent.
        """
        # Ensure the Jacobians are in double precision for better accuracy
        jacobians = self.jacobians.to(torch.float64)

        if num_timesteps is None:
            num_timesteps = jacobians.shape[0]

        # Initialize the matrix product as the identity matrix
        product_jacobian = torch.eye(2, dtype=torch.float64)

        # Compute the product of Jacobians over time
        for t in range(num_timesteps):
            product_jacobian = torch.matmul(product_jacobian, jacobians[t,...])

        # Compute the eigenvalues of the resulting product matrix
        eigenvalues, _ = torch.linalg.eig(product_jacobian)

        # Find the largest eigenvalue in magnitude and use only the real component
        largest_eigenvalue = torch.max(torch.abs(eigenvalues.real))

        # Compute the Lyapunov exponent by averaging over time.
        lyapunov_exponent = (1 / num_timesteps) * torch.log(largest_eigenvalue)

        return lyapunov_exponent.item()  # Convert to Python float

    def __repr__(self):
        param_values = [f"{name}: {value.item()}" for name, value in zip(
            ["alpha1", "beta1", "gamma1", "bx1", "cx1", "alpha2", "beta2", "gamma2", "bx2", "cx2"],
            self.model_params[:10]
        )]
        sigma_str = f"sigma: {self.noise}" if self.noise is not None else "sigma: N/A"
        return "Model Parameters:\n" + ", ".join(param_values) + f", {sigma_str}"