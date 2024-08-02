import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def create_dominant_heights_correlation_plot(measurements_subset, predictions_h100, output_file, width=6, height=6):

    common_lim = (min(measurements_subset['Ho']), max(measurements_subset['Ho']))

    df_plot = pd.DataFrame({
        'Ho_obs': measurements_subset['Ho'][measurements_subset['Alter'] == 100],
        'Ho_preds': predictions_h100['dominant_height'],
        'species': predictions_h100['species_fullname']
    })

    plt.figure(figsize=(width, height))
    sns.scatterplot(data=df_plot, x='Ho_obs', y='Ho_preds', hue='species', style='species', size='species',
                    sizes=(20, 200))
    plt.plot(common_lim, common_lim, linestyle='dashed', color='black')  # Line with slope 1 and intercept 0
    plt.xlim(common_lim)
    plt.ylim(common_lim)
    plt.xlabel("Observed Dominant Height [m]")
    plt.ylabel("Predicted Dominant Height [m]")
    plt.legend(title='Species')
    plt.title("Dominant Heights Correlation")
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()


def create_dominant_height_deviations_plot(predictions_h100, measurements_subset, rid_value, output_file):
    plt.figure(figsize=(6, 6))

    resid = predictions_h100['dominant_height'][predictions_h100['rid'] == rid_value] - \
            measurements_subset['Ho'][(measurements_subset['rid'] == rid_value) & (measurements_subset['Alter'] == 100)]

    plt.plot(measurements_subset['Alter'][measurements_subset['rid'] == rid_value],
             measurements_subset['Ho'][measurements_subset['rid'] == rid_value],
             label='Observed Dominant Height', linewidth=2)

    plt.scatter(100, predictions_h100['dominant_height'][predictions_h100['rid'] == rid_value], color='red', s=100,
                label='Predicted Dominant Height at Age 100')

    for dGz100_value in [8, 9, 11, 12]:
        subset = measurements_subset[(measurements_subset['species'] == measurements_subset['species'][
            measurements_subset['rid'] == rid_value]) &
                                     (measurements_subset['dGz100'] == dGz100_value)]
        plt.plot(subset['Alter'], subset['Ho'], color='lightblue' if dGz100_value in [8, 12] else 'blue', linewidth=1.5)

    plt.axhline(0, linestyle='dashed', color='black')
    plt.axvline(100, linestyle='dashed', color='black')

    plt.legend(["g = 10", "Predicted H100", "g = 8/12", "g = 9/11", "t=100"])
    plt.xlabel('Time [Age]')
    plt.ylabel('Dominant height [m]')
    plt.title('Dominant Height Deviations')
    plt.savefig(output_file)
    plt.close()


def create_site_index_boundaries_plot(measurements_subset, predictions, rid_value, output_file):
    all_lines = pd.concat([
        measurements_subset[(measurements_subset['species'] == measurements_subset['species'][
            measurements_subset['rid'] == rid_value]) &
                            (measurements_subset['dGz100'] == 8)].assign(type='8'),
        measurements_subset[(measurements_subset['species'] == measurements_subset['species'][
            measurements_subset['rid'] == rid_value]) &
                            (measurements_subset['dGz100'] == 12)].assign(type='12'),
        measurements_subset[(measurements_subset['species'] == measurements_subset['species'][
            measurements_subset['rid'] == rid_value]) &
                            (measurements_subset['dGz100'] == 9)].assign(type='9'),
        measurements_subset[(measurements_subset['species'] == measurements_subset['species'][
            measurements_subset['rid'] == rid_value]) &
                            (measurements_subset['dGz100'] == 11)].assign(type='11')
    ])

    plt.figure(figsize=(7, 6))
    sns.lineplot(data=all_lines, x='Alter', y='Ho', hue='type', linewidth=0.8)
    sns.lineplot(data=measurements_subset[measurements_subset['rid'] == rid_value], x='Alter', y='Ho', color='black',
                 linewidth=2, label='g = 10')
    sns.lineplot(data=predictions[predictions['rid'] == rid_value], x='age', y='dominant_height', color='red',
                 linewidth=2, label='y_hat')
    plt.axvline(100, linestyle='dashed', color='black')

    plt.xlabel("Time [Age]")
    plt.ylabel("Dominant height [m]")
    plt.title("Site Index Boundaries")
    plt.legend(title='Yield class (g)')
    plt.savefig(output_file)
    plt.close()


def create_idealized_measurements_timeseries_plot(boundaries, measurements_subset, predictions_h100, output_file):
    boundaries = measurements_subset.groupby(['Alter', 'species']).agg(min_Ho=('Ho', 'min'),
                                                                       max_Ho=('Ho', 'max')).reset_index()

    plt.figure(figsize=(5, 7))
    sns.lineplot(data=measurements_subset, x='Alter', y='Ho', hue='species', alpha=0.5)
    for _, row in boundaries.iterrows():
        plt.fill_between([row['Alter'], row['Alter']], row['min_Ho'], row['max_Ho'], alpha=0.3)
    plt.scatter(100, predictions_h100['dominant_height'], hue=predictions_h100['species'], alpha=0.7, s=50)

    plt.xlabel("Age")
    plt.ylabel("Dominant height [m]")
    plt.title("Idealized Measurements Time Series")
    plt.savefig(output_file)
    plt.close()


def create_residuals_boxplots(predictions_h100, measurements_subset, output_file):
    stands = predictions_h100['rid'].unique()
    resids = pd.DataFrame({'stands': stands, 'species': predictions_h100['species'], 'residuals': np.nan})

    for id in range(len(stands)):
        resids['residuals'].iloc[id] = \
        predictions_h100['dominant_height'][predictions_h100['rid'] == stands[id]].values[0] - \
        measurements_subset['Ho'][
            (measurements_subset['rid'] == stands[id]) & (measurements_subset['Alter'] == 100)].values[0]

    MBE = resids.groupby('species')['residuals'].mean().reset_index()
    print(MBE)

    plt.figure(figsize=(6, 6))
    sns.boxplot(data=resids, x='species', y='residuals', palette='Set1')
    plt.axhline(0, linestyle='dashed', color='black')
    plt.xlabel("Species")
    plt.ylabel("Residuals in Dominant Height [m] at year 100")
    plt.title("Residuals Boxplots")
    plt.savefig(output_file)
    plt.close()


def create_predicted_site_indices_plot(measurements_subset, predictions, predictions_h100,
                                       output_file="plots/predicted_site_indices.pdf"):
    plt.figure(figsize=(6, 6))
    sns.lineplot(data=measurements_subset, x='Alter', y='Ho', hue='dGz100', label='Ideal')
    sns.lineplot(data=predictions, x='age', y='dominant_height', hue='rid', label='iLand')
    sns.scatterplot(data=predictions_h100, x=100, y='dominant_height', color='red', label='Predicted H100', s=100)

    plt.xlabel("Age")
    plt.ylabel("Dominant height [m]")
    plt.title("dGz100 Predicted")
    plt.savefig(output_file)
    plt.close()


def create_horizons_trajectories_plot(horizon_trajectories, horizons_df, output_file):
    plt.figure(figsize=(7, 6))
    for trajectory in horizon_trajectories.T:
        plt.plot(horizons_df['age'], trajectory, color='gray')
    plt.axhline(0, linestyle='dashed', color='black')
    plt.plot(horizons_df['age'], horizons_df['h_means'], color='blue', linewidth=2)
    plt.fill_between(horizons_df['age'], horizons_df['h_means'] + horizons_df['h_sd'],
                     horizons_df['h_means'] - horizons_df['h_sd'], color='lightblue', alpha=0.5)

    plt.xlabel("Age")
    plt.ylabel(r"AE - $\rho_{upper/lower}$")
    plt.title("Horizons Trajectories")
    plt.savefig(output_file)
    plt.close()


def create_horizons_assembled_plot(horizons_assembled, output_file):
    plt.figure(figsize=(7, 6))
    for species in horizons_assembled['species'].unique():
        subset = horizons_assembled[horizons_assembled['species'] == species]
        plt.fill_between(subset['age'], subset['h_means'] - subset['h_sd'], subset['h_means'] + subset['h_sd'],
                         alpha=0.3)
        plt.plot(subset['age'], subset['h_means'], linewidth=1.5)

    plt.axhline(0, color="black", linewidth=1.0)
    plt.xlabel("Lead time [age]")
    plt.ylabel("Error in dominant height [m]")
    plt.title("Horizons Assembled Plot")
    plt.savefig(output_file)
    plt.close()


def create_thresholds_assembled_plot(thresholds_assembled, output_file):
    thresholds_assembled_long = pd.melt(thresholds_assembled, id_vars=['species'],
                                        value_vars=thresholds_assembled.columns[1:],
                                        var_name='threshold', value_name='rho')

    custom_labels = {'rho_lower': 'Lower', 'rho_upper': 'Upper'}
    plt.figure(figsize=(9, 6))
    sns.boxplot(data=thresholds_assembled_long, x='species', y='rho', hue='species', palette='Set1', size=1.5,
                alpha=0.7)
    plt.xlabel("Species")
    plt.ylabel("Threshold (Dominant height [m])")
    plt.title("Thresholds Assembled Plot")
    plt.savefig(output_file)
    plt.close()


