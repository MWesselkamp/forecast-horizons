# Assuming prepare_data and visualisations are defined in separate Python files
import matplotlib.pyplot as plt

from iLand.data import *
from iLand.evaluation_module import *
from iLand.visualisations import *
from iLand.helpers import *

dm = DataSets()
measurements, predictions_h100, predictions, measurements_subset = dm.get_data(baumarten_num = [1, 2, 3, 4, 5, 7])

# Create some plots of the idealized measurements, the predictions at age 100 and predicted timeseries.
create_dominant_heights_correlation_plot(measurements_subset, predictions_h100,
                                         "iLand/plots/dominant_heights_correlation_h100.pdf")

create_site_index_boundaries_plot(measurements, predictions, site_index = 10, species='piab',
                                  save_to="iLand/plots/site_index_forecast.pdf")

create_boundaries_scheme_plot(measurements_subset,
                                  rid_value=2,
                                  output_file="iLand/plots/yield_class_boundaries_scheme.pdf")

create_idealized_measurements_timeseries_plot(measurements_subset, predictions_h100,
                                              output_file="iLand/plots/idealized_measurements_timeseries.pdf")

create_residuals_boxplots(predictions_h100, measurements_subset,
                          output_file="iLand/plots/residuals_age100_boxplots.pdf")

create_predicted_site_indices_plot(measurements_subset,
                                   predictions,
                                   predictions_h100,
                                   output_file="iLand/plots/predicted_site_indices.pdf")

species = 'piab'
stand_idx = 2
standard = 1

piab_DM = DataModule(species = 'piab',
                     stand_idx = 2,
                     standard = 1)

piab_DM.process_data_subsets(measurements, predictions)
piab_DM.create_reference()
piab_DM.create_reference_standards()
piab_data = piab_DM.get_results_dataframe()

EM = EvaluationModule(piab_data)
EM.set_quantitative_standard()
EM.get_horizon_trajectory()
EM.test_horizon_trajectory()
piab_horizon = EM.get_extended_results()

def main(predictions, measurements, species_names, threshold=1):
    """
        Main function to process predicted and measured data for multiple species, compute residuals,
        and generate horizon trajectories. It also creates plots for each species and returns processed data.

        Parameters:
        - predictions: DataFrame containing predicted time series data.
        - measurements: DataFrame containing measured time series data.
        - species_names: List of species names to process.

        Returns:
        - horizons_df_container: List of DataFrames containing horizon trajectories for each species.
        - thresholds_container: List of DataFrames containing threshold values for each species.
        - residuals_container: List of DataFrames containing residuals for each species and stand.
    """
    thresholds_container = []
    horizons_df_container = []
    residuals_container = []

    for species_name in species_names:
        # Select the predicted and ideal time series subsets
        spec_predictions = predictions[predictions['species'] == species_name]
        spec_ideal = measurements[measurements['species'] == species_name]

        # Create vector of unique species idx
        stand_idxs = spec_predictions['rid'].unique()

        # Compile reference data frames for the species with upper and lower boundaries
        spec_reference_dataframes = pd.concat(
            [get_reference_boundaries(idx, spec_predictions, spec_ideal, threshold=threshold) for idx in stand_idxs]
        )

        # Initialize lists to store results for the current species
        absolute_error_horizon_trajectories = []
        threshold_uppers = []
        threshold_lowers = []
        threshold_alts = []

        # DataFrame to store residuals
        residuals_df = pd.DataFrame()

        # Process each stand index
        for idx in stand_idxs:
            # Calculate residuals for the current stand
            residuals_idx = create_residuals_dataframe(spec_reference_dataframes, species_name, idx)
            residuals_df = pd.concat([residuals_df, residuals_idx])

            # Calculate absolute errors for the residuals and bounds
            absolute_error = np.abs(residuals_idx['resids'])
            absolute_error_upper = np.abs(residuals_idx['resids_upper'])
            absolute_error_lower = np.abs(residuals_idx['resids_lower'])

            # Determine thresholds for upper and lower residual bounds
            threshold_upper = np.min(absolute_error[absolute_error >= absolute_error_upper]) if np.any(absolute_error >= absolute_error_upper) else np.inf
            threshold_lower = np.min(absolute_error[absolute_error >= absolute_error_lower]) if np.any(absolute_error >= absolute_error_lower) else np.inf
            threshold_uppers.append(threshold_upper)
            threshold_lowers.append(threshold_lower)

            if not np.isinf(threshold_upper):
                absolute_error_horizon_trajectories.append(threshold_upper - absolute_error)
            elif not np.isinf(threshold_lower):
                absolute_error_horizon_trajectories.append(threshold_lower - absolute_error)
            else:
                threshold_alt = np.min([np.min(absolute_error_upper), np.min(absolute_error_lower)])
                absolute_error_horizon_trajectories.append(threshold_alt - absolute_error)
                threshold_alts.append(threshold_alt)

        horizon_trajectories = np.vstack(absolute_error_horizon_trajectories)

        # Create a DataFrame for horizon means and standard deviations for the species
        horizons_df = pd.DataFrame({
            'age': spec_reference_dataframes['age'].unique(),
            'species': species_name,
            'h_means': horizon_trajectories.mean(axis=0),
            'h_sd': horizon_trajectories.std(axis=0)
        })

        thresholds_df = pd.DataFrame({
            'species': species_name,
            'rho_upper': threshold_uppers,
            'rho_lower': threshold_lowers
        })

        create_horizons_trajectories_plot(horizon_trajectories, horizons_df,
                                          output_file=f"iLand/plots/horizon_{species_name}.pdf")

        horizons_df_container.append(horizons_df)
        thresholds_container.append(thresholds_df)
        residuals_container.append(residuals_df)

    return horizons_df_container, thresholds_container, residuals_container

def simulate_thresholds(predictions, measurements, species_names, threshold=1):
    """
        Main function to process predicted and measured data for multiple species, compute residuals,
        and generate horizon trajectories. It also creates plots for each species and returns processed data.

        Parameters:
        - predictions: DataFrame containing predicted time series data.
        - measurements: DataFrame containing measured time series data.
        - species_names: List of species names to process.

        Returns:
        - horizons_df_container: List of DataFrames containing horizon trajectories for each species.
        - thresholds_container: List of DataFrames containing threshold values for each species.
        - residuals_container: List of DataFrames containing residuals for each species and stand.
    """

    horizons_df_container = []
    residuals_container = []

    for species_name in species_names:
        # Select the predicted and ideal time series subsets
        spec_predictions = predictions[predictions['species'] == species_name]
        spec_ideal = measurements[measurements['species'] == species_name]

        # Create vector of unique species idx
        stand_idxs = spec_predictions['rid'].unique()

        # Compile reference data frames for the species with upper and lower boundaries
        spec_reference_dataframes = pd.concat(
            [get_reference_boundaries(idx, spec_predictions, spec_ideal, threshold=1) for idx in stand_idxs]
        )

        # Initialize lists to store results for the current species
        absolute_error_horizon_trajectories = []
        # DataFrame to store residuals
        residuals_df = pd.DataFrame()

        # Process each stand index
        for idx in stand_idxs:
            # Calculate residuals for the current stand
            residuals_idx = create_residuals_dataframe(spec_reference_dataframes, species_name, idx)
            residuals_df = pd.concat([residuals_df, residuals_idx])

            # Calculate absolute errors for the residuals and bounds
            absolute_error = np.abs(residuals_idx['resids'])

            absolute_error_horizon_trajectories.append(threshold - absolute_error)

        horizon_trajectories = np.vstack(absolute_error_horizon_trajectories)

        # Create a DataFrame for horizon means and standard deviations for the species
        horizons_df = pd.DataFrame({
            'age': spec_reference_dataframes['age'].unique(),
            'species': species_name,
            'h_means': horizon_trajectories.mean(axis=0),
            'h_sd': horizon_trajectories.std(axis=0)
        })


        horizons_df_container.append(horizons_df)
        residuals_container.append(residuals_df)

    return horizons_df_container, residuals_container


species_names = predictions['species'].unique()[:5] # ignore pisy

horizons_df_container, thresholds_container, residuals_container = main(predictions, measurements, species_names, threshold=1)
horizons_assembled = pd.concat(horizons_df_container, ignore_index=True)
thresholds_assembled = pd.concat(thresholds_container, ignore_index=True)
residuals_assembled = pd.concat(residuals_container, ignore_index=True)

horizons_assembled = add_species_fullname(horizons_assembled)
thresholds_assembled = add_species_fullname(thresholds_assembled)
create_horizons_assembled_plot(horizons_assembled,
                                   output_file="iLand/plots/iLand_horizons_assembled.pdf")
create_thresholds_assembled_plot(thresholds_assembled,
                                     output_file="iLand/plots/threshold_distribution.pdf")
# Find ages where h_means and h_means +/- h_sd < 0 for the first time
result_df = find_horizon(horizons_assembled)
# Plot the results
plot_age_limit_by_species(result_df, output_file="iLand/plots/iLand_age_limits.pdf")

# plot_age_limit_by_species_mulitples(result_dfs, output_file="iLand/plots/iLand_age_limits_multiple_thresholds.pdf")
result_dfs = []
thresholds = [0.0, 0.5,1.0,1.5,2.0,2.5,3.0, 3.5]
for rho in thresholds:
    horizons_df_container, residuals_container = simulate_thresholds(predictions, measurements, species_names, threshold=rho)
    horizons_assembled = pd.concat(horizons_df_container, ignore_index=True)
    thresholds_assembled = pd.concat(thresholds_container, ignore_index=True)
    residuals_assembled = pd.concat(residuals_container, ignore_index=True)
    horizons_assembled = add_species_fullname(horizons_assembled)
    thresholds_assembled = add_species_fullname(thresholds_assembled)
    result_dfs.append(find_horizon(horizons_assembled))

plot_age_limit_by_species_mulitples(result_dfs,thresholds=thresholds,
                                    output_file="iLand/plots/iLand_age_limits_multiple_thresholds.pdf")



df = residuals_assembled.groupby(['species', 'age']).apply(lambda x: x.assign(
    mean_resid_upper=x['resids_upper'].mean(),
    mean_resid_lower=x['resids_lower'].mean()
)).reset_index(drop=True)

plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='age', y='resids', hue='species', estimator=None)
plt.axhline(0, color='gray', linewidth=1.0)
plt.xlabel("Age")
plt.ylabel("Residuals (Dominant height [m])")
plt.legend(title='Species', loc='upper right')
plt.title("Residuals by Age and Species")
plt.tight_layout()
plt.savefig("residuals_plot.pdf")
plt.show()


def mse(differences):
    return np.mean(differences ** 2)


def ae(differences):
    return np.abs(differences)


diffs = np.linspace(-2, 2, 100)
plt.plot(diffs, [mse(x) for x in diffs], label='MSE')
plt.title("Mean Squared Error")
plt.show()

plt.plot(residuals_assembled['resids'], [ae(x) for x in residuals_assembled['resids']], label='Absolute Error')
plt.xlim(-3, 3)
plt.ylim(0, 5)
plt.ylabel('Absolute error')
plt.show()

plt.plot(residuals_assembled['resids'], [mse(x) for x in residuals_assembled['resids']], label='Mean Squared Error')
plt.xlim(-5, 5)
plt.ylim(0, 10)
plt.ylabel('Mean squared error')
plt.show()


