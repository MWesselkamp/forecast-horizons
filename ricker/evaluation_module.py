import torch
import scipy.stats as stats
import numpy as np

from utils import compute_autocorrelation

class EvaluationModule:
    def __init__(self, ensemble, climatology, config):
        """
        Initialize the EvaluationModule with a 3D tensor and an evaluation metric.

        Parameters:
        - tensor_3d (torch.Tensor): A 3-dimensional tensor (shape: [batch_size, time_steps, features]).
        """
        self.ensemble = ensemble
        self.climatology = climatology
        self.resolution = config["forcing"]["resolution"]
        self.resampling_size = config["simulation"]["resampling_size"]
        self.bootstrap_samples = config['simulation']['ensemble_size']
        self.ensemble_samples = config['simulation']['resampling_size']

    def decorrelation_coefficient(self):

        decorrelation_coefficients = []
        for species_idx in range(self.climatology.shape[0]):  # Assuming two species (0 and 1)
            decorrelation_coefficients.append(np.mean(
                [compute_autocorrelation(self.climatology[species_idx, i, :].detach().numpy()) for i in
                 range(self.climatology.shape[1])]
            ))
            print(f"Autocorrelation Species{species_idx + 1}: {decorrelation_coefficients[species_idx]:.4f}")

        return decorrelation_coefficients

    def degrees_of_freedom(self):

        decorrelation_coefficients = self.decorrelation_coefficient()
        dofs = []
        for decorr_coeff in decorrelation_coefficients:

            dofs.append( self.resolution / (
                    (1 - decorr_coeff) / (1 + decorr_coeff)) - 1 )

        return dofs

    def compute_PPP(self, X, t, sigma_c):

        if X.ndimension() != 2:
            raise ValueError("Input tensor X must be 2D with shape (N, T).")

        N, T = X.shape

        # Validate inputs
        if not (0 <= t < T):
            raise ValueError(f"Time index t={t} is out of bounds for tensor with T={T}.")
        if N <= 1:
            raise ValueError(f"Number of ensembles N={N} must be greater than 1 to avoid division by zero.")

        # Step 1: Compute the mean over N samples at time t
        mean_t = X[:, t].mean()  # Shape: scalar

        # Step 2: Compute the squared differences (X[j, t] - mean_t)^2
        diff_squared = (X[:, t] - mean_t) ** 2  # Shape: (N,)

        # Step 3: Normalize the squared differences by sigma_c^2
        normalized_diff_squared = diff_squared / (sigma_c ** 2)  # Shape: (N,)

        # Step 4: Sum the normalized squared differences across all samples
        sum_terms = normalized_diff_squared.sum()  # Scalar

        # Step 5: Compute PPP(t) using the formula with division by (N - 1)
        PPP_t = 1 - (1 / (N - 1)) * sum_terms

        return PPP_t

    def compute_bootstrap_PPP(self, X, t, sigma_c):
        """
        Compute PPP(t) based on the formula:

        PPP(t) = 1 - (1 / (N * (M - 1))) * Σ_{j=1}^{N} Σ_{i=1}^{M} [(X[j, i, t] - mean_j(t))^2 / sigma_c^2]

        Parameters:
        - X (torch.Tensor): A 3D tensor of shape (N, M, T), where
            N = number of bootstrapped ensembles (indexed by j),
            M = number of members per bootstrap ensemble (indexed by i),
            T = number of time steps.
        - t (int): The time step at which to compute PPP(t).
        - sigma_c (float or torch scalar): The climatological standard deviation used for normalization.

        Returns:
        - PPP_t (torch.Tensor): A scalar tensor representing PPP(t) at time t.

        Raises:
        - ValueError: If the time index t is out of bounds or if M <= 1 (to avoid division by zero).
        """
        if X.ndimension() == 3:
            N, M, T = X.shape
        elif X.ndimension() == 2:
            N, T = X.shape
            M = 1  # Treat as if there's only one sample (M=1)
            X = X.unsqueeze(1)  # Add the extra dimension for M to maintain consistency (shape: (N, 1, T))
        else:
            raise ValueError("Input tensor X must have either 2 or 3 dimensions.")

            # Validate inputs
        if not (0 <= t < T):
            raise ValueError(f"Time index t={t} is out of bounds for tensor with T={T}.")
        if M <= 1 and X.ndimension() == 3:  # Only check if originally 3D input
            raise ValueError(f"Number of samples per group M={M} must be greater than 1 to avoid division by zero.")

        # Step 1: Compute the mean over samples i for each group j at time t
        # Shape of mean_j_t: (N,)
        mean_j_t = X[:, :, t].mean(dim=1)

        # Step 2: Compute the squared differences (X[j, i, t] - mean_j_t[j])^2
        # Reshape mean_j_t to (N, 1) for broadcasting
        diff_squared = (X[:, :, t] - mean_j_t.unsqueeze(1)) ** 2  # Shape: (N, M)

        # Step 3: Normalize the squared differences by sigma_c^2
        normalized_diff_squared = diff_squared / (sigma_c ** 2)  # Shape: (N, M)

        # Step 4: Sum the normalized squared differences across all groups and samples
        sum_terms = normalized_diff_squared.sum()  # Scalar

        # Step 5: Compute PPP(t) using the formula
        PPP_t = 1 - (1 / (N * (M - 1))) * sum_terms

        return PPP_t

    def bootstrap_normalised_variances(self, X, t, sigma_c, reference = "ensemble_mean", seasonal_normalisation = False):

        N, M, T = X.shape
        print("X.shape: ", N, M, T)

        # Validate inputs
        if not (0 <= t < T):
            raise ValueError(f"Time index t={t} is out of bounds for tensor with T={T}.")
        if M <= 1:
            raise ValueError(f"Number of samples per group M={M} must be greater than 1 to avoid division by zero.")

        # Step 1: Compute the mean over samples i for each group j at time t
        # Shape of mean_j_t: (N,)
        if reference == "ensemble_mean":

            mean_j_t = X[:, :, t].mean(dim=1)
            # Step 2: Compute the squared differences (X[j, i, t] - mean_j_t[j])^2
            # Reshape mean_j_t to (N, 1) for broadcasting
            diff_squared = (X[:, :, t] - mean_j_t.unsqueeze(1)) ** 2  # Shape: (N, M)

        # Step 3: Normalize the squared differences by sigma_c^2
        if seasonal_normalisation:
            normalized_diff_squared = (1 / ((M - 1))) * diff_squared.sum(axis = 1) / (sigma_c[t] ** 2)  # Shape: (N, M)
        else:
            normalized_diff_squared = (1 / ((M - 1))) * diff_squared.sum(axis=1) / (sigma_c ** 2)

        return normalized_diff_squared

    def PPP_threshold(self, df1, df2, alpha = 0.05):

        """
        Computes the threshold for PPP based on the critical f-statistics, assuming an alpha level of 0.05 as default.

        Parameters:
            -df1: within group degrees of freedom of nominator
            -df2: climatological degrees of freedom, of denominator
            -alpha: significance level

        Returns:
            PPP-threshold
        """

        PPP_threshold = 1 - 1 / stats.f.ppf(1 - alpha / 2, df1, df2)
        print("PPP threshold:", PPP_threshold)
        return PPP_threshold

    def _ensemble_sample(self, dimension, ensemble_samples):
        """
        Take samples from the ensemble with replacement.

        Parameters:
        - dimension (int): The dimension from which to take samples (0, 1, or 2).
        - n_samples (int): The number of samples to take.

        Returns:
        - torch.Tensor: A tensor containing a sub sample of the ensemble.
        """
        size = self.ensemble.size(dimension)
        indices = torch.randint(0, size, (ensemble_samples,), dtype=torch.long)  # Sampling with replacement
        return torch.index_select(self.ensemble, dimension, indices)

    def bootstrap_ensemble(self):

        bootstrap = [self._ensemble_sample(dimension = 0, ensemble_samples = self.ensemble_samples) for _ in range(self.bootstrap_samples)]
        return torch.stack(bootstrap)
