import numpy as np
from data import *
class EvaluationModule:
    def __init__(self, species_data):

        self.species_data = species_data
        self.residuals()
        self.absolute_error()

        self.species_data['residuals_expect'] = self.residuals_expect
        self.species_data['residuals_upper'] = self.residuals_upper
        self.species_data['residuals_lower'] = self.residuals_lower

        self.species_data['ae_expect'] = self.ae_expect
        self.species_data['ae_upper'] = self.ae_upper
        self.species_data['ae_lower'] = self.ae_lower

    def residuals(self):
        self.residuals_expect = self.species_data['h0_ideal'].values - self.species_data['h0_predicted'].values
        self.residuals_upper = self.species_data['h0_ideal_upper'].values - self.species_data['h0_predicted'].values
        self.residuals_lower = self.species_data['h0_ideal_lower'].values - self.species_data['h0_predicted'].values
    def absolute_error(self):
        """
        The absolute error collapses deviations towards upper and lower boundary standards basically into a one directed metric.
        This means, we will not distinguish positive or negative bias when computing the horizon, but just absolute deviations.
        """
        self.ae_expect = np.abs(self.residuals_expect)
        self.ae_upper = np.abs(self.residuals_upper)
        self.ae_lower = np.abs(self.residuals_lower)

    def set_quantitative_standard(self):
        """
        This function checks if the expected distance to the reference is larger than the distance to the standard references.
        """
        self.rho_ae_upper = np.where(self.ae_expect >= self.ae_upper, self.ae_expect, np.inf)
        self.rho_ae_lower = np.where(self.ae_expect >= self.ae_lower, self.ae_expect, np.inf)

        print("Rho upper: ", self.rho_ae_upper)
        print("Rho lower: ", self.rho_ae_lower)

        horizon_upper = self.get_horizon(self.rho_ae_upper)
        horizon_lower = self.get_horizon(self.rho_ae_upper)
        print("Horizon upper ", horizon_upper, " timesteps")
        print("Horizon lower ", horizon_lower, " timesteps")

        self.rho_ae_upper = self.rho_ae_upper[horizon_upper]
        self.rho_ae_lower = self.rho_ae_lower[horizon_lower]
        print("Rho AE upper: ", self.rho_ae_upper)
        print("Rho AE lower: ", self.rho_ae_lower)

    def get_horizon(self, rho_ae):

        valid_indices = np.where(rho_ae != np.inf)[0]
        print("Valid indices: ", valid_indices)
        if valid_indices.size > 0:
            horizon = valid_indices[np.argmin(rho_ae[valid_indices])]
        else:
            horizon = 0  # Handle the case where all values are np.inf

        alt_horizon = np.argmin(rho_ae)
        print("Alternative Horizon computation ", alt_horizon)

        return horizon

    def get_horizon_trajectory(self):
        """
        We compute rho - AE, for a two-sided look up.
        We make an exception:
         if rho is inf, i.e. AE always within standards, we pick closest distance to standard as rho.
        """
        if not np.all(np.isinf(self.rho_ae_upper)):
            ae_horizon_trajectory_upper = self.rho_ae_upper - self.ae_expect
            self.species_data['ae_horizon_trajectory_upper'] = ae_horizon_trajectory_upper
        if not np.all(np.isinf(self.rho_ae_lower)):
            ae_horizon_trajectory_lower = self.rho_ae_lower - self.ae_expect
            self.species_data['ae_horizon_trajectory_lower'] = ae_horizon_trajectory_lower

        if np.all(np.isinf(self.rho_ae_upper)) and np.all(np.isinf(self.rho_ae_lower)):
            print("Make expection and use smallest distance to standard as rho.")
            threshold_alt = np.min([np.min(self.ae_upper), np.min(self.ae_lower)])
            self.species_data['ae_horizon_trajectory']  = threshold_alt - self.ae_expect
        else:
            average_columns = ['ae_horizon_trajectory_lower', 'ae_horizon_trajectory_upper']
            available_columns = [col for col in average_columns if col in self.species_data.columns]
            self.species_data['ae_horizon_trajectory'] = self.species_data[available_columns].mean(axis=1, skipna=True)

    def test_horizon_trajectory(self):
        try:
            print("Mean trajectory horizon", np.argmin(self.species_data['ae_horizon_trajectory'] > 0))
        except KeyError:
            print("No trajectory horizon")
    def get_extended_results(self):
        return self.species_data
    def get_aggregated_results(self):

        select_columns = ['age', 'stand_idx', 'species', 'species_fullname', 'ae_horizon_trajectory']
        return self.species_data[select_columns]