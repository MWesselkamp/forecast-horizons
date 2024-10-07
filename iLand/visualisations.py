import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

def create_site_index_boundaries_plot(measurements_subset, predictions, rid_value, output_file):
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

    # Plot each group
    sns.lineplot(data=all_lines, x='Alter', y='Ho', hue='type', linewidth=0.8,
                 palette={'8': 'lightblue', '12': 'lightblue', '9': 'blue', '11': 'blue'})

    # Plot the '10' line from the measurements_subset
    sns.lineplot(data=measurements_subset[measurements_subset['rid'] == rid_value], x='Alter', y='Ho', color='black',
                 linewidth=2, label='10')

    # Plot the predictions line
    sns.lineplot(data=predictions[predictions['rid'] == rid_value], x='age', y='dominant_height', color='red',
                 linewidth=2, label='y_hat')

    # Add vertical dashed line at x = 100
    plt.axvline(x=100, color='black', linestyle='--')

    # Customize the legend and labels
    plt.xlabel('Time [Age]', fontsize=16)
    plt.ylabel('Dominant height [m]', fontsize=16)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)

    # Customize the legend
    plt.legend(title='Yield class (g)', title_fontsize=16, fontsize=12)

    # Minimal theme
    sns.set_theme(style="whitegrid")

    # Save the plot to a file
    plt.savefig(output_file, format='pdf')
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

def create_idealized_measurements_timeseries_plot(boundaries, measurements_subset, predictions_h100, output_file):
    # Create the boundaries dataframe by grouping and summarizing
    boundaries = measurements_subset.groupby(['Alter', 'species']).agg(
        min_Ho=('Ho', 'min'),
        max_Ho=('Ho', 'max')
    ).reset_index()

    plt.figure(figsize=(5, 7))

    # Plot the ribbon (shaded area between min_Ho and max_Ho)
    for species in boundaries['species'].unique():
        species_data = boundaries[boundaries['species'] == species]
        plt.fill_between(
            species_data['Alter'], species_data['min_Ho'], species_data['max_Ho'],
            alpha=0.3, label=f'{species} range'
        )

    # Plot the line for measurements_subset
    sns.lineplot(
        data=measurements_subset,
        x='Alter', y='Ho', hue='species',
        estimator=None, units='rid', alpha=0.5, linewidth=1
    )

    # Plot the points for predictions at age 100
    sns.scatterplot(
        data=predictions_h100,
        x=[100] * len(predictions_h100), y='dominant_height',
        hue='species', style='species', s=50, alpha=0.7,
        edgecolor='black', linewidth=0.7
    )

    # Set the palette
    sns.set_palette("Set1")

    # Labels and title
    plt.xlabel("Age", fontsize=16)
    plt.ylabel("Dominant height [m]", fontsize=16)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.title("dGz100 Reconstructed Observations", fontsize=18)

    # Adjust the legend
    plt.legend(title="Species", title_fontsize=16, fontsize=12)

    # Minimal theme
    sns.despine()

    # Save the plot to a file
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
    unique_species = horizons_assembled['species_fullname'].unique()

    # Define colors for each species using Matplotlib's colormap 'Set1'
    colors = plt.get_cmap("Set1", len(unique_species))

    # Plot the ribbons for standard deviations
    for i, species in enumerate(unique_species):
        subset = horizons_assembled[horizons_assembled['species_fullname'] == species]
        plt.fill_between(
            subset['age'],
            subset['h_means'] - subset['h_sd'],
            subset['h_means'] + subset['h_sd'],
            alpha=0.1,
            color=colors(i)
        )

    # Plot the mean lines for each species
    for i, species in enumerate(unique_species):
        species_data = horizons_assembled[horizons_assembled['species_fullname'] == species]
        plt.plot(species_data['age'], species_data['h_means'], label=species, color=colors(i), linewidth=2)

    # Add horizontal dashed line at y = 0
    plt.axhline(y=0, color='black', linewidth=2, linestyle='--')

    # Set labels
    plt.xlabel("Lead time [age]", fontsize=18)
    plt.ylabel("Absolute error [m]", fontsize=18)

    # Adjust legend
    plt.legend(fontsize=14,
               ncol = 3,
               loc='upper center')

    # Apply minimal theme and rotate x-axis labels
    plt.xticks(rotation=45, fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylim((-7, 5))

    # Save the plot to a PDF file
    plt.tight_layout()
    plt.grid(False)
    plt.savefig(output_file, format='pdf')
    plt.close()

def create_thresholds_assembled_plot(thresholds_assembled, output_file):
    # Convert the DataFrame from wide to long format
    thresholds_assembled_long = thresholds_assembled.melt(
        id_vars=['species_fullname'],
        value_vars=['rho_upper', 'rho_lower'],
        var_name='threshold',
        value_name='rho'
    )

    # Custom labels for the facets
    custom_labels = {"rho_lower": "Lower", "rho_upper": "Upper"}

    # Increase the size of the plot panel to accommodate x-ticklabels
    plt.figure(figsize=(14, 14))

    # Facet by threshold
    g = sns.FacetGrid(thresholds_assembled_long, col='threshold',
                      col_wrap=2,
                      sharey=True,
                      height=4.5,  # Increase height of each facet for more space
                      aspect=1.2  # Increase aspect ratio for wider panels
    )
    g.map_dataframe(
        sns.boxplot,
        x='species_fullname', y='rho', hue='species_fullname',
        palette='Set1')

    # Adjust the transparency of the boxes
    for ax in g.axes.flat:
        for patch in ax.artists:
            patch.set_alpha(0.7)

    g.set_axis_labels("", "Threshold [m]", fontsize=16)

    # Adjust facet titles
    for ax, title in zip(g.axes.flat, thresholds_assembled_long['threshold'].unique()):
        ax.set_title(custom_labels[title], fontsize=16)

    # Customize plot appearance and adjust x-tick labels
    for ax in g.axes.flat:
        ax.tick_params(axis='x', rotation=45, labelsize=14)
        ax.tick_params(axis='y', labelsize=14)

    # Adjust the layout to fix spacing issues
    g.fig.subplots_adjust(top=0.9, bottom=0.2, left=0.1, right=0.9, hspace=0.4, wspace=0.3)

    # Align y-axis labels by ensuring consistent padding
    for ax in g.axes.flat:
        ax.yaxis.labelpad = 10

    plt.tight_layout()
    plt.savefig(output_file, format='pdf')
    plt.close()

def plot_age_limit_by_species(result_df, output_file):
    fig, ax = plt.subplots(figsize=(10, 5))

    # Loop through each row in the result DataFrame
    for index, row in result_df.iterrows():
        species = row['species_fullname']
        mean_age = row['mean_age']
        plus_sd_age = row['plus_sd_age']
        minus_sd_age = row['minus_sd_age']

        # Calculate error bars
        lower_error = mean_age - minus_sd_age if not pd.isna(minus_sd_age) else 0
        upper_error = plus_sd_age - mean_age if not pd.isna(plus_sd_age) else 0

        # Plot the mean age with error bars
        ax.errorbar(mean_age, index, xerr=[[lower_error], [upper_error]],
                    fmt='o',
                    markersize=12,  # Increase marker size
                    elinewidth=3,  # Increase error bar line width
                    capsize=5,
                    label=species)

    ax.set_yticks(range(len(result_df)))
    ax.set_yticklabels(result_df['species_fullname'], fontsize = 20)
    ax.set_xlabel('Age', fontsize = 20)
    ax.tick_params(axis='x', labelsize=20)  # Increase size of x-tick labels
    ax.tick_params(axis='y', labelsize=20)  # Increase size of y-tick labels
    plt.xlim((40, 115))
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(output_file, format='pdf')
    plt.close()
