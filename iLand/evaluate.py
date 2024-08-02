import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming prepare_data and visualisations are defined in separate Python files
from data import *
from visualisations import *

# Create a subset data frame of ideal observations that matches BWI site indices
measurements_subset = pd.DataFrame()
for stand_idx in range(len(predictions_h100)):
    measurements_stand_idx = measurements[
                                 (predictions_h100['species'][stand_idx] == measurements['species']) &
                                 (predictions_h100['site_index'][stand_idx] == measurements['dGz100'])
                                 ].iloc[:, [1, 2, 5, 16]]
    measurements_stand_idx['rid'] = predictions_h100['rid'][stand_idx]
    measurements_subset = pd.concat([measurements_subset, measurements_stand_idx], ignore_index=True)

measurements_subset.to_csv("measurements_subset.csv", index=False)

# Create some plots of the idealized measurements, the predictions at age 100 and predicted timeseries.
create_dominant_heights_correlation_plot(measurements_subset, predictions_h100,
                                         "plots/dominant_heights_correlation_h100.pdf")
create_dominant_height_deviations_plot(predictions_h100, measurements_subset,
                                       rid_value=2,
                                       output_file="plots/reconstructed_SI_boundaries.pdf")
create_site_index_boundaries_plot(measurements_subset, predictions,
                                  rid_value=2,
                                  output_file="plots/yield_class_boundaries.pdf")

create_idealized_measurements_timeseries_plot(boundaries, measurements_subset, predictions_h100,
                                              output_file="plots/idealized_measurements_timeseries.pdf")
create_residuals_boxplots(predictions_h100, measurements_subset,
                          output_file="plots/residuals_boxplots.pdf")
create_predicted_site_indices_plot(measurements_subset, predictions, predictions_h100)


def get_reference_boundaries(stand_idx, predicted_h0, measured_h0, threshold=1, spec='piab'):
    print(stand_idx)
    print(spec)

    # select stand ID and only every fifth year (for comparison with ideal)
    predicted_idx = predicted_h0[
        (predicted_h0['rid'] == stand_idx) & (predicted_h0['age'] % 5 == 0)
        ]

    # determine maximum stand age up to which we look up measured site indices.
    maximum_age = min(max(predicted_idx['age']), max(measured_h0['Alter'].unique()))

    predicted_idx = predicted_idx[predicted_idx['age'] <= maximum_age]
    predicted_idx_SI = predicted_idx['site_index'].unique()

    # select ideal observations only in the predicted age range
    measured_h0 = measured_h0[(measured_h0['Alter'] >= 45) & (measured_h0['Alter'] <= maximum_age)]

    # expected measurements based on predicted Site index.
    measured_idx_expected = measured_h0[measured_h0['dGz100'] == predicted_idx_SI]

    # Determine upper and lower bounds
    if predicted_idx_SI + threshold > measured_h0['dGz100'].max():
        measured_idx_upper_bound = measured_h0[measured_h0['dGz100'] == predicted_idx_SI]
    else:
        measured_idx_upper_bound = measured_h0[measured_h0['dGz100'] == predicted_idx_SI + threshold]

    if predicted_idx_SI - threshold < measured_h0['dGz100'].min():
        measured_idx_lower_bound = measured_h0[measured_h0['dGz100'] == predicted_idx_SI]
    else:
        measured_idx_lower_bound = measured_h0[measured_h0['dGz100'] == predicted_idx_SI - threshold]

    return pd.DataFrame({
        'rid': stand_idx,
        'age': predicted_idx['age'],
        'h0_predicted': predicted_idx['dominant_height'],
        'h0_ideal': measured_idx_expected['Ho'],
        'h0_ideal_upper': measured_idx_upper_bound['Ho'],
        'h0_ideal_lower': measured_idx_lower_bound['Ho']
    })


def create_residuals_dataframe(spec_reference_dataframes, idx, species_name, age):
    residuals_idx = pd.DataFrame({
        'resids': (spec_reference_dataframes['h0_ideal'][spec_reference_dataframes['rid'] == idx] -
                   spec_reference_dataframes['h0_predicted'][spec_reference_dataframes['rid'] == idx]),
        'resids_upper': (spec_reference_dataframes['h0_ideal_upper'][spec_reference_dataframes['rid'] == idx] -
                         spec_reference_dataframes['h0_predicted'][spec_reference_dataframes['rid'] == idx]),
        'resids_lower': (spec_reference_dataframes['h0_ideal_lower'][spec_reference_dataframes['rid'] == idx] -
                         spec_reference_dataframes['h0_predicted'][spec_reference_dataframes['rid'] == idx]),
        'species': species_name,
        'stand_idx': idx,
        'age': age
    })
    return residuals_idx


# We conduct this for all species:
thresholds_container = []
horizons_df_container = []
residuals_container = []
j = 0
for species_name in baumarten_char:
    # select the predicted and ideal time series subsets
    spec_predictions = predictions[predictions['species'] == species_name]
    spec_ideal = measurements[measurements['species'] == species_name]

    # Create vector of unique species idx.
    stand_idxs = spec_predictions['rid'].unique()

    # Now assemble data frames with reference ideal upper and lower boundaries.
    spec_reference_dataframes = pd.concat(
        [get_reference_boundaries(x, spec_predictions, spec_ideal) for x in stand_idxs], ignore_index=True)

    absolute_error_horizon_trajectories = []
    threshold_uppers = []
    threshold_lowers = []
    threshold_alts = []

    residuals_df = pd.DataFrame()

    for idx in stand_idxs:
        residuals_idx = create_residuals_dataframe(spec_reference_dataframes, idx, species_name,
                                                   spec_reference_dataframes['age'][
                                                       spec_reference_dataframes['rid'] == idx].values[0])
        residuals_df = pd.concat([residuals_df, residuals_idx], ignore_index=True)

        absolute_error = np.abs(residuals_idx['resids'])
        absolute_error_upper = np.abs(residuals_idx['resids_upper'])
        absolute_error_lower = np.abs(residuals_idx['resids_lower'])

        threshold_upper = absolute_error[absolute_error >= absolute_error_upper].min()
        threshold_lower = absolute_error[absolute_error >= absolute_error_lower].min()
        threshold_uppers.append(threshold_upper)
        threshold_lowers.append(threshold_lower)

        if not np.isinf(threshold_upper):
            absolute_error_horizon_trajectories.append(threshold_upper - absolute_error)
        elif not np.isinf(threshold_lower):
            absolute_error_horizon_trajectories.append(threshold_lower - absolute_error)
        else:
            threshold_alt = min(absolute_error_upper.min(), absolute_error_lower.min())
            absolute_error_horizon_trajectories.append(threshold_alt - absolute_error)
            threshold_alts.append(threshold_alt)

    horizon_trajectories = np.vstack(absolute_error_horizon_trajectories)

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
                                      output_file=f"horizon_{species_name}.pdf")

    horizons_df_container.append(horizons_df)
    thresholds_container.append(thresholds_df)
    residuals_container.append(residuals_df)

    j += 1

horizons_assembled = pd.concat(horizons_df_container, ignore_index=True)
thresholds_assembled = pd.concat(thresholds_container, ignore_index=True)
residuals_assembled = pd.concat(residuals_container, ignore_index=True)

create_horizons_assembled_plot(horizons_assembled,
                               output_file="plots/horizons_assembled.pdf")
create_thresholds_assembled_plot(thresholds_assembled,
                                 output_file="plots/threshold_distribution.pdf")

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


