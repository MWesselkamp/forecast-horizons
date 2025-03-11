import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns
import tol_colors as tc
import os

from data import *
from matplotlib import font_manager
from metrics import absolute_differences

def create_dominant_heights_correlation_plot(measurements_subset, predictions_h100, output_file, width=6, height=6):

    common_lim = (min(measurements_subset['Ho']), max(measurements_subset['Ho']))

    df_plot = pd.DataFrame({
        'Ho_obs': measurements_subset['Ho'][measurements_subset['Alter'] == 100].values,
        'Ho_preds': predictions_h100['dominant_height'].values,
        'species': predictions_h100['species_fullname'].values
    })

    plt.figure(figsize=(width, height))
    sns.scatterplot(data=df_plot, x='Ho_obs', y='Ho_preds',
                    hue='species', style='species', s = 100)
    plt.plot(common_lim, common_lim, linestyle='dashed', color='black')  # Line with slope 1 and intercept 0
    #plt.xlim(common_lim)
    #plt.ylim(common_lim)
    plt.xlabel("Observed Dominant Height [m]")
    plt.ylabel("Predicted Dominant Height [m]")
    plt.legend(title='Species')
    #plt.title("Dominant Heights Correlation")
    plt.savefig(output_file)
    plt.close()


def create_dominant_height_deviations_plot(predictions_h100, measurements_subset, rid_value, output_file):

    plt.figure(figsize=(6, 6))

    for dGz100_value in [8, 9, 11, 12]:
        subset = measurements_subset[
            (measurements_subset['species'] ==
             measurements_subset['species'][measurements_subset['rid'] == rid_value].values[0]) &
            (measurements_subset['dGz100'] == dGz100_value)
            ]
        plt.plot(subset['Alter'],
                 subset['Ho'],
                 color='lightblue' if dGz100_value in [8, 12] else 'blue',
                 linewidth=1.5,
                 label=f'g = {dGz100_value}')

    plt.plot(measurements_subset['Alter'][measurements_subset['rid'] == rid_value],
             measurements_subset['Ho'][measurements_subset['rid'] == rid_value],
             label='Observed Dominant Height', linewidth=2)

    plt.scatter(100, predictions_h100['dominant_height'][predictions_h100['rid'] == rid_value], color='red', s=100,
                label='Predicted Dominant Height at Age 100')

    plt.axhline(0, linestyle='dashed', color='black')
    plt.axvline(100, linestyle='dashed', color='black')

    plt.legend(["g = 10", "Predicted H100", "g = 8/12", "g = 9/11", "t=100"])
    plt.xlabel('Time [Age]')
    plt.ylabel('Dominant height [m]')
    plt.title('Dominant Height Deviations')
    plt.savefig(output_file)
    plt.close()

def create_dominant_heights_boxplots(predictions_h100, measurements, output_dir="iLand/plots"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create the first plot
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=predictions_h100, y='dominant_height', hue='species_fullname', palette="Set1")
    plt.xlabel('Species')
    plt.ylabel('Dominant height [m]')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "predictions_dominant_height_boxplot.pdf"))
    plt.close()

    # Create the second plot
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=measurements, y='Ho', hue='species_fullname', palette="Set1")
    plt.xlabel('Species')
    plt.ylabel('Dominant height [m]')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "measurements_dominant_height_boxplot.pdf"))
    plt.close()

    # Arrange the plots side by side and save to a PDF file
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    sns.boxplot(data=predictions_h100, y='dominant_height', hue='species_fullname', palette="Set1", ax=axes[0])
    sns.boxplot(data=measurements, y='Ho', hue='species_fullname', palette="Set1", ax=axes[1])
    for ax in axes:
        ax.set_xlabel('Species')
        ax.set_ylabel('Dominant height [m]')
        ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dominant_heights_boxplots.pdf"))
    plt.close()


def create_site_index_boundaries_plot(measurements, predictions,
                                      rho_g,
                                      #ae_ex,
                                      site_index = 10,
                                      species = 'piab',
                                      save_to  = "site_index_forecast.pdf"):

    fig, axs = plt.subplots(2, 1, figsize=(7, 6),height_ratios=[2, 1],
                             sharex=True, sharey=False)

    dm = DataManipulator(measurements, predictions, species = species)
    measurements_df = dm.subset_measurements(min_age=45, max_age=110)
    predictions_df = dm.subset_predictions(min_age=45, max_age=110, site_index = [site_index])

    measured_yield_class_rho3 = dm.select_measurement_site_index(measurements_df, site_index = [7,13])
    measured_yield_class_rho2 = dm.select_measurement_site_index(measurements_df, site_index = [8,12])
    measured_yield_class_rho1 = dm.select_measurement_site_index(measurements_df, site_index = [9,11])

    grouped_df = measured_yield_class_rho1.groupby('dGz100')
    #for name, group in grouped_df:
    #    axs[0].plot(group['Alter'], group['Ho'], color="salmon", linewidth=2)
    #grouped_df = measured_yield_class_rho1.groupby('dGz100')
    #for name, group in grouped_df:
    #    axs[0].plot(group['Alter'], group['Ho'], color="salmon", linewidth=2)
    #grouped_df = measured_yield_class_rho3.groupby('dGz100')
    for name, group in grouped_df:
        axs[0].plot(group['Alter'], group['Ho'], color="salmon", linewidth=2)
    reference_yield_class = measurements_df.query(f"dGz100 == {site_index}")
    axs[0].plot(reference_yield_class['Alter'], reference_yield_class['Ho'], color="black", label="Verification", linewidth=2)
    bias = []
    for idx in predictions_df['rid'].unique():
        predicted_yield_class = dm.select_predictions_plot(predictions_df, idx)
        predicted_yield_class_sparse = predicted_yield_class[predicted_yield_class['age'] % 5 == 0]
        bias.append(absolute_differences(predicted_yield_class_sparse['dominant_height'].values, reference_yield_class['Ho'].values))
        axs[0].plot(predicted_yield_class['age'], predicted_yield_class['dominant_height'], color="blue", linewidth=2)

    axs[0].plot([], [], color="salmon", label="Reference")
    axs[0].plot([], [], color="blue", label="Forecast")
    #axs[0].axvline(x=100, color='black', linestyle='--')
    axs[0].set_ylabel('Dominant height [m]', fontsize=16, fontweight = 'bold')
    axs[0].tick_params(axis='x', labelrotation=45, labelsize=16)
    axs[0].tick_params(axis='y', labelsize=16)
    bold_font = font_manager.FontProperties(size=14)
    axs[0].legend(title=f'Picea $\\it{{abies}}$ (k={site_index})', fontsize=14, title_fontproperties=bold_font,
                  loc = "upper left")
    axs[0].grid(False)

    #axs[1].plot(predicted_yield_class_sparse['age'], ae_ex.transpose(), color="blue", linewidth=2.5)
    axs[1].plot(predicted_yield_class_sparse['age'], rho_g.transpose(), color="salmon", linewidth=2)
    axs[1].plot(predicted_yield_class_sparse['age'], rho_g.transpose()[:,0], color="salmon", linewidth=2, label="Tolerance")
    axs[1].plot(predicted_yield_class_sparse['age'], np.array(bias).transpose(), color = "darkblue", linewidth=2)
    axs[1].plot(predicted_yield_class_sparse['age'], np.array(bias).transpose()[:,0], color = "darkblue", linewidth=2, label="Error")
    axs[1].set_xlabel('Stand Age [years]', fontsize=16, fontweight = 'bold')
    axs[1].set_ylabel('Absolute error', fontsize=16, fontweight = 'bold')
    axs[1].tick_params(axis='x', labelrotation=45, labelsize=16)
    axs[1].tick_params(axis='y', labelsize=16)
    axs[1].legend(fontsize=14, title_fontproperties=bold_font,loc = "upper left")

    fig.align_ylabels(axs)

    plt.tight_layout()
    plt.savefig(save_to, format='pdf')
    plt.close()

def create_boundaries_scheme_plot(measurements_subset, rid_value, output_file):
    # Filter and label the data based on dGz100 values
    df_8 = measurements_subset[(measurements_subset['species'] ==
                                measurements_subset['species'][measurements_subset['rid'] == rid_value].values[0]) &
                               (measurements_subset['dGz100'] == 8)].copy()
    df_8['type'] = '8'

    df_12 = measurements_subset[(measurements_subset['species'] ==
                                 measurements_subset['species'][measurements_subset['rid'] == rid_value].values[0]) &
                                (measurements_subset['dGz100'] == 12)].copy()
    df_12['type'] = '12'

    df_9 = measurements_subset[(measurements_subset['species'] ==
                                measurements_subset['species'][measurements_subset['rid'] == rid_value].values[0]) &
                               (measurements_subset['dGz100'] == 9)].copy()
    df_9['type'] = '9'

    df_11 = measurements_subset[(measurements_subset['species'] ==
                                 measurements_subset['species'][measurements_subset['rid'] == rid_value].values[0]) &
                                (measurements_subset['dGz100'] == 11)].copy()
    df_11['type'] = '11'

    # Combine all dataframes
    all_lines = pd.concat([df_8, df_12, df_9, df_11])

    plt.figure(figsize=(7, 6))
    label_mapping = {'8': f'+/-2$\\rho$', '12': 'Label2', '9': '+/-$\\rho$', '11': 'Label3'}

    # Plot each group
    sns.lineplot(data=all_lines, x='Alter', y='Ho', hue='type', linewidth=2.5,
                 palette={'8': 'lightsalmon', '12': 'lightsalmon', '9': 'red', '11': 'red'})

    # Plot the '10' line from the measurements_subset
    sns.lineplot(data=measurements_subset[measurements_subset['rid'] == rid_value], x='Alter', y='Ho', color='black',
                 linewidth=2.5, label='Reference')

    # Plot the predictions line

    # Add vertical dashed line at x = 100
    plt.axvline(x=100, color='black', linestyle='--')

    # Customize the legend and labels
    plt.xlabel('Time', fontsize=26)
    plt.ylabel('State unit', fontsize=26)
    plt.xticks([])
    plt.yticks([])

    # Customize the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    new_labels = [label_mapping.get(label, label) for label in labels]

    filtered_handles_labels = [(h, l) for h, l in zip(handles, new_labels) if l not in ['Label2', 'Label3']]
    filtered_handles, filtered_labels = zip(*filtered_handles_labels)

    plt.legend(filtered_handles, filtered_labels, loc='upper left', bbox_to_anchor=(0, 1.15), title='', ncol=3,
               title_fontsize=16, fontsize=20)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_linewidth(2)  # Increase linewidth of the x-axis spine
    ax.spines['left'].set_linewidth(2)

    # Minimal theme
    sns.set_theme(style="whitegrid")

    # Save the plot to a file
    plt.tight_layout()
    plt.savefig(output_file, format='pdf')
    plt.close()

def create_idealized_measurements_timeseries_plot(measurements_subset, predictions_h100, output_file):
    # Create the boundaries dataframe by grouping and summarizing
    boundaries = measurements_subset.groupby(['Alter', 'species']).agg(
        min_Ho=('Ho', 'min'),
        max_Ho=('Ho', 'max')
    ).reset_index()

    # Filter species list to remove "pisy"
    # measurements_subset = add_species_fullname(measurements_subset)
    species_list = measurements_subset['species'].unique()
    species_list = [species for species in species_list if species != 'pisy']
    species_names = [species for species in measurements_subset['species_fullname'].unique() if species != "Pinus \nsylvestris"]

    # Create subplots: one for each species
    fig, axes = plt.subplots(len(species_list), 1, figsize=(5, 7),
                             sharex=True, sharey=True)

    # If only one species, wrap axes in a list to make it iterable
    if len(species_list) == 1:
        axes = [axes]

    cmap = tc.tol_cmap('light')  # Extract 5 colors from the "Set1" colormap
    colors = cmap(np.linspace(0.2, 1, 5))

    # Plot for each species in its own subplot
    for i, species in enumerate(species_list):
        ax = axes[i]

        # Filter data for the current species and age between 45 and 110
        species_data = boundaries[(boundaries['species'] == species) ] #  & (boundaries['Alter'] >=45 ) & (boundaries['Alter'] <= 110)
        species_measurements = measurements_subset[(measurements_subset['species'] == species) ] #& (measurements_subset['Alter'] >= 45) & (measurements_subset['Alter'] <= 110)]
        species_predictions = predictions_h100[(predictions_h100['species'] == species)]

        # Plot the ribbon (shaded area between min_Ho and max_Ho)
        #ax.fill_between(
        #    species_data['Alter'], species_data['min_Ho'], species_data['max_Ho'],
        #    alpha=0.2, color = colors[i]
        #)

        ax.plot(
            species_measurements['Alter'], species_measurements['Ho'],
            linestyle='-', alpha=0.6, linewidth=0.6, color = colors[i]
        )

        ax.scatter(
            [100] * len(species_predictions), species_predictions['dominant_height'],
            s=50, alpha=0.6, edgecolor='black', color='white', linewidth=0.7,
        )

        ax.set_ylabel(species_names[i], fontsize=14)
        ax.yaxis.set_label_position("right")
        ax.tick_params(axis='both', which='major', labelsize=16)

    plt.xlabel("Age", fontsize=16)
    # Set shared y-axis label
    fig.text(0.01, 0.5, 'Dominant height [m]', va='center', rotation='vertical', fontsize=16)

    # Only one legend, placed on the top right
    handles, labels = ax.get_legend_handles_labels()
    #fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2, fontsize=12, title="Legend")
    plt.subplots_adjust(left=0.14, right=0.88, hspace=0.2)
    plt.savefig(output_file, format='pdf')
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
    plt.ylabel("Absolute error in Dominant Height [m]")
    plt.savefig(output_file)
    plt.close()

def create_predicted_site_indices_plot(measurements_subset, predictions, predictions_h100,
                                       output_file="predicted_site_indices.pdf"):
    plt.figure(figsize=(6, 6))

    # Plot the ideal observations
    sns.lineplot(
        data=measurements_subset,
        x='Alter', y='Ho', hue='species', style='dGz100',
        palette=['gray'], linewidth=1.5, legend=False
    )

    # Plot the iLand predictions
    sns.lineplot(
        data=predictions,
        x='age', y='dominant_height', hue='species', style='rid',
        palette=['yellow'], linewidth=1.5, legend=False
    )

    # Plot the predicted H100 points
    sns.scatterplot(
        data=predictions_h100,
        x=[100] * len(predictions_h100), y='dominant_height',
        hue='species', style='rid',
        palette=['red'], s=50, edgecolor='black', legend=False
    )

    # Customize the color legend manually
    handles = [
        plt.Line2D([0], [0], color='gray', lw=2, label='Ideal Observation'),
        plt.Line2D([0], [0], color='yellow', lw=2, label='iLand Predictions'),
        plt.Line2D([0], [0], marker='o', color='red', lw=0, label='Predicted H100', markerfacecolor='red', markersize=6)
    ]
    plt.legend(handles=handles, title='Legend')

    # Labels and title
    plt.xlabel("Age", fontsize=12)
    plt.ylabel("Dominant height [m]", fontsize=12)
    plt.title("dGz100 Predicted", fontsize=14)

    # Facet grid by species
    g = sns.FacetGrid(measurements_subset, col='species', sharex=True, sharey=True, height=6, aspect=1)
    g.map_dataframe(sns.lineplot, x='Alter', y='Ho', color='gray', linewidth=1.5, alpha=0.7)
    #g.map_dataframe(sns.lineplot, x='age', y='dominant_height', data=predictions, color='yellow', linewidth=1.5)
    #g.map_dataframe(sns.scatterplot, data=predictions_h100, x=100, y='dominant_height', color='red', s=50,
    #                edgecolor='black')

    # Adjust the layout and save the plot
    plt.tight_layout()
    plt.savefig(output_file, format='pdf')
    plt.close()

def create_horizons_trajectories_plot(horizon_trajectories, horizons_df, output_file):
    plt.figure(figsize=(7, 6))  # 7 inches wide and 6 inches high

    # Plot all horizon trajectories in gray
    for trajectory in horizon_trajectories:
        plt.plot(horizons_df['age'], trajectory, color='gray', linewidth=1, linestyle='-')

    # Plot the mean horizon trajectory in blue
    plt.plot(horizons_df['age'], horizons_df['h_means'], color='blue', linewidth=2, linestyle='-')

    # Plot the mean + standard deviation in light blue
    plt.plot(horizons_df['age'], horizons_df['h_means'] + horizons_df['h_sd'], color='lightblue', linewidth=2, linestyle='-')

    # Plot the mean - standard deviation in light blue
    plt.plot(horizons_df['age'], horizons_df['h_means'] - horizons_df['h_sd'], color='lightblue', linewidth=2, linestyle='-')

    # Add a horizontal line at y = 0
    plt.axhline(y=0, color='black', linewidth=1, linestyle='--')

    # Set the labels
    plt.xlabel("Age")
    plt.ylabel(r'AE - $\rho_{upper/lower}$')

    # Save the plot to a file
    plt.savefig(output_file, format='pdf')
    plt.close()

def create_horizons_assembled_plot(horizons_assembled, output_file):

    plt.style.use('default')
    plt.figure(figsize=(7, 6))

    # Get unique species from the DataFrame
    unique_species = sorted(horizons_assembled['species_fullname'].unique())

    # Define colors for each species using Matplotlib's colormap 'Set1'
    cmap = tc.tol_cmap('light')  # Extract 5 colors from the "Set1" colormap
    colors = cmap(np.linspace(0.2, 1, 5))

    # Plot the ribbons for standard deviations
    for i in range(5):
        subset = horizons_assembled[horizons_assembled['species_fullname'] == unique_species[i]]
        plt.fill_between(
            subset['age']-40,
            subset['h_means'] - 2*subset['h_sd'],
            subset['h_means'] + 2*subset['h_sd'],
            alpha=0.1,
            color=colors[i]
        )

    # Plot the mean lines for each species
    for i in range(5):
        species_data = horizons_assembled[horizons_assembled['species_fullname'] == unique_species[i]]
        plt.plot(species_data['age']-40, species_data['h_means'], label=unique_species[i], color=colors[i], linewidth=2)

    # Add horizontal dashed line at y = 0
    plt.axhline(y=0, color='black', linewidth=2, linestyle='--')

    # Set labels
    plt.xlabel("Lead time [years]", fontsize=18, fontweight = 'bold')
    plt.ylabel("$\mathbf{\\varrho}-$AE [m]", fontsize=18, fontweight = 'bold')

    bold_font = font_manager.FontProperties(weight='bold', size=16)
    plt.legend(fontsize=14,
               ncol = 3,
               loc='upper center',
               title_fontproperties=bold_font,
               prop={'size':14})

    # Apply minimal theme and rotate x-axis labels
    plt.xticks(rotation=45, fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylim((-8.5, 7))

    # Save the plot to a PDF file
    plt.tight_layout()
    plt.grid(False)
    plt.savefig(output_file, format='pdf')
    plt.close()

def plot_age_limit_by_species_mulitples(result_dfs, thresholds, output_file):
    """
    Plot the mean age by species for multiple result DataFrames.
    The x-axis represents the AE cut-off thresholds, and the y-axis represents the mean age.
    Each species is plotted with dots connected by lines across the datasets, with filled confidence intervals.

    Parameters:
    - result_dfs: List of result DataFrames.
    - thresholds: List of AE cut-off thresholds to label the x-axis.
    - output_file: Path to save the resulting plot as a PDF.
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    cmap = tc.tol_cmap('light')  # Extract 5 colors from the "Set1" colormap
    colors = cmap(np.linspace(0.2, 1, 5))

    species_names = sorted(result_dfs['species_fullname'].unique())

    for i in range(5):

        species_data = result_dfs.query(f"species_fullname == {repr(species_names[i])}")

        x_values =  thresholds #list(range(1, len(species_data['mean_age'].values) + 1))
        ax.hlines(y=110 - 45, xmin=min(x_values), xmax=max(x_values), linestyles="-",
                  color="black", linewidth=1.1)
        ax.hlines(y=45 -45 , xmin=min(x_values), xmax=max(x_values), linestyles="-",
                  color="black", linewidth=1.1)
        ax.vlines(x=1, ymin=45 -45 , ymax=110 - 45, linestyles="--",
                  color="black", linewidth=1.3)
        ax.plot(x_values, species_data['mean_age'].values - 45,
                color=colors[i], label=species_names[i],
                markersize=10, linewidth=2.6, alpha=0.9)
        # Add filled confidence intervals for the current species
        # ax.fill_between(x_values, lower_bounds, upper_bounds, color=colors[species_idx], alpha=0.2)

    #y_ticks = ax.get_yticks()
    #new_y_ticks = y_ticks - 45
    #ax.set_yticks(new_y_ticks)
    #ax.set_yticklabels([f'{int(yt)}' for yt in new_y_ticks])

    ax.set_xlabel('$\mathbf{\\varrho}$ [m]', fontsize=18, fontweight = 'bold')
    ax.set_ylabel('Forecast limit [years]', fontsize=18, fontweight = 'bold')
    ax.tick_params(axis='both', labelsize=18)
    #ax.legend()

    plt.tight_layout()
    plt.savefig(output_file, format='pdf')
    plt.close()