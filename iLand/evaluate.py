# Assuming prepare_data and visualisations are defined in separate Python files
from iLand.data import *
from iLand.visualisations import *
from iLand.helpers import *

measurements, predictions_h100, predictions = get_data(baumarten_num = [1, 2, 3, 4, 5, 7])

# Create a subset data frame of ideal observations that matches BWI site indices
measurements_subset = pd.DataFrame()

for stand_idx in range(len(predictions_h100)):
    measurements_stand_idx = measurements[
                                 (predictions_h100['species'][stand_idx] == measurements['species']) &
                                 (predictions_h100['site_index'][stand_idx] == measurements['dGz100'])
                                 ].iloc[:, [1, 2, 5, 16]]
    measurements_stand_idx['rid'] = predictions_h100['rid'][stand_idx]
    measurements_subset = pd.concat([measurements_subset, measurements_stand_idx], ignore_index=True)

measurements_subset.to_csv("iLand/data/measurements_subset.csv", index=False)
print(measurements_subset['species'].unique())
print(len(measurements_subset['rid'].unique()))

measurements_subset = measurements_subset.sort_values(by='rid')
predictions_h100 = predictions_h100.sort_values(by='rid')

# Create some plots of the idealized measurements, the predictions at age 100 and predicted timeseries.
create_dominant_heights_correlation_plot(measurements_subset, predictions_h100,
                                         "iLand/plots/dominant_heights_correlation_h100.pdf")
create_dominant_height_deviations_plot(predictions_h100, measurements_subset,
                                       rid_value=2,
                                       output_file="iLand/plots/reconstructed_SI_boundaries.pdf")
create_site_index_boundaries_plot(measurements_subset, predictions,
                                  rid_value=2,
                                  output_file="iLand/plots/yield_class_boundaries.pdf")

create_site_index_boundaries_plot_new(measurements_subset, predictions,
                                  rid_value=2,
                                  output_file="iLand/plots/yield_class_boundaries.pdf")

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

def get_reference_boundaries(stand_idx, predicted_h0, measured_h0, threshold, spec='piab'):
    # Print stand index and species for debugging/confirmation
    print(stand_idx)
    print(spec)

    # Filter the predicted H0 DataFrame for the specified stand ID and only every fifth year
    predicted_idx = predicted_h0[(predicted_h0['rid'] == stand_idx) & (predicted_h0['age'] % 5 == 0)]

    # Determine maximum stand age up to which to look up measured site indices
    maximum_age = min(max(measured_h0['Alter'].unique()), max(predicted_idx['age'].unique()))
    predicted_idx = predicted_idx[predicted_idx['age'] <= maximum_age]

    # Determine the predicted site index
    predicted_idx_SI = predicted_idx['site_index'].unique()[0]

    # Filter the measured H0 DataFrame for the appropriate age range
    measured_h0 = measured_h0[(measured_h0['Alter'] >= 45) & (measured_h0['Alter'] <= maximum_age)]

    # Select the expected measurements based on the predicted site index
    measured_idx_expected = measured_h0[measured_h0['dGz100'] == predicted_idx_SI]

    # Determine the upper bound
    # Upper bound is the next site index, but if out of range, it remains the same
    upper_bound_SI = min(predicted_idx_SI + threshold, measured_h0['dGz100'].max())
    measured_idx_upper_bound = measured_h0[measured_h0['dGz100'] == upper_bound_SI]

    # Lower bound is the previous site index, but if out of range, it remains the same
    lower_bound_SI = max(predicted_idx_SI - threshold, measured_h0['dGz100'].min())
    measured_idx_lower_bound = measured_h0[measured_h0['dGz100'] == lower_bound_SI]

    # Create and return a DataFrame with the results
    result = pd.DataFrame({
        'rid': stand_idx,
        'age': predicted_idx['age'],
        'h0_predicted': predicted_idx['dominant_height'],
        'h0_ideal': measured_idx_expected['Ho'].values,
        'h0_ideal_upper': measured_idx_upper_bound['Ho'].values,
        'h0_ideal_lower': measured_idx_lower_bound['Ho'].values
    })

    return result



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


