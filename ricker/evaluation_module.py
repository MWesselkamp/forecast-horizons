import torch
import scipy.stats as stats
class EvaluationModule:
    def __init__(self, ensemble):
        """
        Initialize the EvaluationModule with a 3D tensor and an evaluation metric.

        Parameters:
        - tensor_3d (torch.Tensor): A 3-dimensional tensor (shape: [batch_size, time_steps, features]).
        """
        self.ensemble = ensemble

    def compute_PPP(self, X, t, sigma_c):
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
        N, M, T = X.shape

        # Validate inputs
        if not (0 <= t < T):
            raise ValueError(f"Time index t={t} is out of bounds for tensor with T={T}.")
        if M <= 1:
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
        return PPP_threshold

    def _ensemble_sample(self, dimension, ensemble_samples):
        """
        Take a samples from the ensemble.

        Parameters:
        - dimension (int): The dimension from which to take samples (0, 1, or 2).
        - n_samples (int): The number of samples to take.

        Returns:
        - torch.Tensor: A tensor containing a sub sample of the ensemble.
        """
        indices = torch.randperm(self.ensemble.size(dimension))[:ensemble_samples]
        return torch.index_select(self.ensemble, dimension, indices)
    def bootstrap_ensemble(self, bootstrap_samples, ensemble_samples):

        bootstrap = [self._ensemble_sample(dimension = 0, ensemble_samples = ensemble_samples) for _ in range(bootstrap_samples)]
        return torch.stack(bootstrap)

