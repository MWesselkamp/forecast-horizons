import matplotlib.pyplot as plt
import numpy as np
import torch
import os

from mpl_toolkits.axes_grid1 import make_axes_locatable

class VisualisationModule:

    def __init__(self, config):

        self.config = config

        self.label_properties = {'weight': 'bold', 'size': 16}
        self.tick_properties = {'size': 16}
        self.legend_properties = {
                "prop": {
                "size": 14
                }
            }

        self.xlabel = "Lead time [Generation]"
    def plot_losses(self, losses, loss_fun, log=True):
        if log:
            ll = np.log(torch.stack(losses).detach().numpy())
        else:
            ll = torch.stack(losses).detach().numpy()
        plt.plot(ll)
        plt.ylabel(f'{loss_fun} loss')
        plt.xlabel(f'Epoch')
        plt.savefig(os.path.join(self.path_to_plots, 'losses.pdf'))
        plt.close()

    def plot_posterior(self, df, saveto=''):

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
        axes[0, 0].hist(df["alpha"], bins=10, edgecolor='black')
        axes[0, 0].set_title("Histogram of alpha")
        axes[0, 1].hist(df["beta"], bins=10, edgecolor='black')
        axes[0, 1].set_title("Histogram of beta")
        axes[1, 0].hist(df["bx"], bins=10, edgecolor='black')
        axes[1, 0].set_title("Histogram of bx")
        axes[1, 1].hist(df["cx"], bins=10, edgecolor='black')
        axes[1, 1].set_title("Histogram of cx")
        plt.tight_layout()
        plt.show()
        plt.savefig(os.path.join(self.path_to_plots, 'posterior.pdf'))
        plt.close()

    def plot_simulations(self, observations_dict, series_name, climatology = None):
        """
        Plot the specified time series from the observations dictionary.

        Parameters:
        - observations_dict: Dictionary containing time series data.
        - series_name: The key of the time series to plot (e.g., 'y_train', 'y_test', etc.).
        """
        if series_name not in observations_dict:
            raise ValueError(f"Series name '{series_name}' not found in observations.")

        series_data = observations_dict[series_name]

        if not isinstance(series_data, np.ndarray):
            series_data = series_data.detach().numpy()

        if series_data.shape[0] > 1:
            series_data = series_data.transpose()
        labs = ["Species 1", "Species 2"]
        cols = ["blue", "magenta"]
        plt.figure(figsize=(8, 5))
        if not climatology is None:
            plt.plot(climatology, color = "lightgray", alpha = 0.8)
        for spec in range(series_data.shape[1]):
            plt.plot(series_data[:,spec], label=labs[spec], color = cols[spec])
        plt.xlabel(self.xlabel, **self.label_properties)
        plt.ylabel("Rel. Population Size", **self.label_properties)
        plt.legend(**self.legend_properties, frameon=False)
        plt.setp(plt.gca().get_xticklabels(), **self.tick_properties)
        plt.setp(plt.gca().get_xticklabels(), **self.tick_properties)

        plt.savefig(os.path.join(self.config['path_to_plots'], "plot_simulations.pdf"))
        plt.tight_layout()
        plt.show()

    def plot_ppp(self, species1_PPP, species2_PPP, PPP_threshold_species1, PPP_threshold_species2):
        plt.figure(figsize=(8, 5))
        plt.plot(species1_PPP, label="Species 1", color="blue")
        plt.plot(species2_PPP, label="Species 2", color="magenta")
        plt.hlines(PPP_threshold_species1, xmin=0, xmax=len(species1_PPP), color="darkblue", linestyles='--')
        plt.hlines(PPP_threshold_species2, xmin=0, xmax=len(species1_PPP), color="purple", linestyles='--')

        plt.xlabel(self.xlabel, **self.label_properties)
        plt.ylabel("PPP",  **self.label_properties)
        # Set font size for tick labels to 16
        #plt.xticks(fontsize=16)
        #plt.yticks(fontsize=16)
        plt.setp(plt.gca().get_xticklabels(), **self.tick_properties)
        plt.setp(plt.gca().get_yticklabels(), **self.tick_properties)

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), frameon=False,
                   **self.legend_properties)

        # Save the plot to a file
        plt.savefig(os.path.join(self.config['path_to_plots'], "ricker_forecast_horizon.pdf"))
        plt.tight_layout()
        plt.show()
    def plot_ensemble(self, ensemble):

        plt.figure(figsize=(7, 5))
        plt.plot(ensemble[:,1,:].detach().numpy().squeeze().transpose(),
                 alpha = 0.3, color = "pink")
        plt.plot(ensemble[:, 0, :].detach().numpy().squeeze().transpose(),
             alpha=0.3, color="lightblue")
        plt.plot(ensemble[:, 1, :].detach().numpy().squeeze().transpose().mean(axis=1),
                 color="magenta", label="Species 2")
        plt.plot(ensemble[:, 0, :].detach().numpy().squeeze().transpose().mean(axis=1),
                 color="blue", label="Species 1")
        plt.xlabel(self.xlabel,  **self.label_properties)
        plt.ylabel("Rel. Population Size",  **self.label_properties)
        # Set font size for tick labels to 16
        plt.setp(plt.gca().get_xticklabels(), **self.tick_properties)
        plt.setp(plt.gca().get_xticklabels(), **self.tick_properties)

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(),
                   **self.legend_properties,
                   frameon=False)

        # Save the plot to a file
        plt.savefig(os.path.join(self.config['path_to_plots'],"ensemble.pdf"))
        plt.tight_layout()
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15)
        plt.legend()
        plt.show()
    def plot_combined(self, species1_PPP, species2_PPP, PPP_threshold_species1, PPP_threshold_species2, ensemble):
        # Create a figure with two subplots arranged in two rows
        fig, ax = plt.subplots(2, 1, figsize=(6, 6),
                               sharex= True)  # 2 rows, 1 column
        labelsize = 16

        # Plot the ensemble in the second subplot
        ax[0].plot(ensemble[:, 1, :].detach().numpy().squeeze().transpose(),
                   alpha=0.3, color="pink")
        ax[0].plot(ensemble[:, 0, :].detach().numpy().squeeze().transpose(),
                   alpha=0.3, color="lightblue")
        ax[0].plot(ensemble[:, 1, :].detach().numpy().squeeze().transpose().mean(axis=1),
                   color="magenta", label="Species 2")
        ax[0].plot(ensemble[:, 0, :].detach().numpy().squeeze().transpose().mean(axis=1),
                   color="blue", label="Species 1")

        #ax[0].set_xlabel("Time", fontsize=16)
        ax[0].set_ylabel("Rel. Population Size", **self.label_properties)
        ax[0].tick_params(axis='both', which='major', labelsize=labelsize)

        handles, labels = ax[0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax[0].legend(by_label.values(), by_label.keys(),frameon=False, **self.legend_properties)

        # Plot PPP for both species in the first subplot
        ax[1].plot(species1_PPP, label="Species 1", color="blue")
        ax[1].plot(species2_PPP, label="Species 2", color="magenta")
        ax[1].hlines(PPP_threshold_species1, xmin=0, xmax=len(species1_PPP), color="darkblue", linestyles='--')
        ax[1].hlines(PPP_threshold_species2, xmin=0, xmax=len(species1_PPP), color="purple", linestyles='--')

        ax[1].set_xlabel(self.xlabel, **self.label_properties)
        ax[1].set_ylabel("PPP", **self.label_properties)
        ax[1].tick_params(axis='both', which='major', labelsize=labelsize)

        handles, labels = ax[1].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        #ax[1].legend(by_label.values(), by_label.keys(), fontsize=16)


        fig.align_ylabels(ax)  # Align y-axis labels for both subplots
        plt.subplots_adjust(left=0.15)  # Adjust left margin to ensure y-axis labels are aligned
        # Adjust layout to avoid overlap
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['path_to_plots'],"combined_plot.pdf"))
        plt.show()

    def plot_combined_2(self, species1_normalised_variances, species2_normalised_variances, species1_PPP, species2_PPP, PPP_threshold_species1, PPP_threshold_species2, time_horizon, red_noise = None):
        # Create a figure with two subplots arranged in two rows
        fig, ax = plt.subplots(2, 1, figsize=(6, 6),
                               sharex= True)  # 2 rows, 1 column

        # Plot the ensemble in the second subplot
        x = np.arange(0, time_horizon)
        ax[0].plot(x, species1_normalised_variances[:time_horizon, :], color="pink", alpha=0.7)
        ax[0].plot(x, species1_normalised_variances[:time_horizon, :].mean(axis=1), color="magenta", label  = "Species 1", alpha=0.7)
        ax[0].plot(x, species2_normalised_variances[:time_horizon, :], color="lightblue", alpha=0.7)
        ax[0].plot(x, species2_normalised_variances[:time_horizon, :].mean(axis=1), color="blue",label  = "Species 2", alpha=0.7)
        if not red_noise is None:
            ax[0].plot(x, red_noise[:time_horizon], color="red", label="red noise", alpha=0.7)
        ax[0].set_ylabel('Normalised variances', **self.label_properties)
        ax[0].tick_params(axis='both', which='major', labelsize=16)

        handles, labels = ax[0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax[0].legend(by_label.values(), by_label.keys(),frameon=False, **self.legend_properties)

        # Plot PPP for both species in the first subplot
        ax[1].plot(x, (1-species1_normalised_variances[:time_horizon, :])- PPP_threshold_species1, color="pink", alpha=0.7)
        ax[1].plot(x, (1-species1_normalised_variances[:time_horizon, :]).mean(axis=1) - PPP_threshold_species1, color="magenta", label="Species 1",
                   alpha=0.7)
        ax[1].plot(x, (1-species2_normalised_variances[:time_horizon, :]) - PPP_threshold_species2, color="lightblue", alpha=0.7)
        ax[1].plot(x, (1-species2_normalised_variances[:time_horizon, :]).mean(axis=1) - PPP_threshold_species2, color="blue", label="Species 2",
                   alpha=0.7)
        ax[1].hlines(0, xmin=0, xmax=time_horizon, colors="black", linestyles="--")
        ax[1].set_xlabel(self.xlabel, **self.label_properties)
        ax[1].set_ylabel("$\mathbf{\\varrho}-$PPP", **self.label_properties)
        ax[1].tick_params(axis='both', which='major', labelsize=16)

        fig.align_ylabels(ax)  # Align y-axis labels for both subplots
        plt.subplots_adjust(left=0.15)  # Adjust left margin to ensure y-axis labels are aligned

        plt.tight_layout()
        plt.savefig(os.path.join(self.config['path_to_plots'],"combined_plot_2.pdf"))
        plt.show()

    def plot_combined_3(self, species1_normalised_variances, species2_normalised_variances, species1_X, species2_X, PPP_threshold_species1, PPP_threshold_species2, time_horizon, red_noise = None):
        # Create a figure with two subplots arranged in two rows
        fig, ax = plt.subplots(2, 1, figsize=(6, 6),
                               sharex= True)  # 2 rows, 1 column
        spec1_mu = species1_X[:time_horizon, :].mean(axis=1)
        spec2_mu = species2_X[:time_horizon, :].mean(axis=1)
        spec1_sd = species1_X[:time_horizon, :].std(axis=1)
        spec2_sd = species2_X[:time_horizon, :].std(axis=1)

        # Plot the ensemble in the second subplot
        x = np.arange(0, time_horizon)
        ax[0].fill_between(x, spec1_mu + 2*spec1_sd, spec1_mu - 2*spec1_sd, color="pink", alpha=0.5)
        ax[0].fill_between(x, spec2_mu + 2*spec2_sd, spec2_mu - 2*spec2_sd, color="lightblue", alpha=0.8)
        ax[0].plot(x, species1_X[:time_horizon, :].mean(axis=1), color="magenta", label  = "Species 1", alpha=0.7)
        ax[0].plot(x, species2_X[:time_horizon, :].mean(axis=1), color="blue",label  = "Species 2", alpha=0.7)
        if not red_noise is None:
            ax[0].plot(x, red_noise[:time_horizon], color="red", label="red noise", alpha=0.7)
        ax[0].set_ylabel('Rel. Population Size', **self.label_properties)
        ax[0].tick_params(axis='both', which='major', labelsize=16)

        handles, labels = ax[0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax[0].legend(by_label.values(), by_label.keys(), frameon=False,**self.legend_properties)

        # Plot PPP for both species in the first subplot
        ax[1].plot(x, (1-species1_normalised_variances[:time_horizon, :])- PPP_threshold_species1, color="pink", alpha=0.7)
        ax[1].plot(x, (1-species1_normalised_variances[:time_horizon, :]).mean(axis=1) - PPP_threshold_species1,
                   color="magenta", label="Species 1", alpha=0.7)
        ax[1].plot(x, (1-species2_normalised_variances[:time_horizon, :]) - PPP_threshold_species2,
                   color="lightblue", alpha=0.7)
        ax[1].plot(x, (1-species2_normalised_variances[:time_horizon, :]).mean(axis=1) - PPP_threshold_species2,
                   color="blue", label="Species 2", alpha=0.7)
        ax[1].hlines(0, xmin=0, xmax=time_horizon, colors="black", linestyles="--")
        ax[1].set_xlabel(self.xlabel, **self.label_properties)
        ax[1].set_ylabel("$\mathbf{\\varrho}-$PPP", **self.label_properties)
        ax[1].tick_params(axis='both', which='major', labelsize=16)

        fig.align_ylabels(ax)  # Align y-axis labels for both subplots
        #plt.subplots_adjust(left=0.15)  # Adjust left margin to ensure y-axis labels are aligned

        plt.tight_layout()
        plt.savefig(os.path.join(self.config['path_to_plots'],"combined_plot_3.pdf"))
        plt.show()
    def plot_combined_4(self, observations_dict, series_name, climatology,
                        species1_normalised_variances, species2_normalised_variances,
                        PPP_threshold_species1, PPP_threshold_species2,
                        time_horizon):

        fig, ax = plt.subplots(2, 1, figsize=(6, 6),
                               sharex= True)  # 2 rows, 1 column

        series_data = observations_dict[series_name]

        if not isinstance(series_data, np.ndarray):
            series_data = series_data.detach().numpy()

        if series_data.shape[0] > 1:
            series_data = series_data.transpose()
        labs = ["Species 1", "Species 2"]
        cols = ["blue", "magenta"]

        if not climatology is None:
            ax[0].plot(climatology, color = "lightgray", alpha = 0.8)
        for spec in range(series_data.shape[1]):
            ax[0].plot(series_data[:,spec], label=labs[spec], color = cols[spec])
        ax[0].set_ylabel("Rel. Population Size", **self.label_properties)
        ax[0].legend(**self.legend_properties, frameon=False)
        ax[0].tick_params(axis='both', which='major', labelsize=16)

        handles, labels = ax[0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax[0].legend(by_label.values(), by_label.keys(),frameon=False, **self.legend_properties)

        # Plot PPP for both species in the first subplot
        x = np.arange(0, time_horizon)
        ax[1].plot(x, (1-species1_normalised_variances)- PPP_threshold_species1, color="pink", alpha=0.5)
        ax[1].plot(x, (1-species2_normalised_variances) - PPP_threshold_species2, color="lightblue", alpha=0.5)
        ax[1].plot(x, (1 - species1_normalised_variances).mean(axis=1) - PPP_threshold_species1, color="magenta",
                   label="Species 1",
                   alpha=0.8)
        ax[1].plot(x, (1-species2_normalised_variances).mean(axis=1) - PPP_threshold_species2, color="blue", label="Species 2",
                   alpha=0.8)
        ax[1].hlines(0, xmin=0, xmax=time_horizon, colors="black", linestyles="--")
        ax[1].set_xlabel(self.xlabel, **self.label_properties)
        ax[1].set_ylabel("$\mathbf{\\varrho}-$PPP", **self.label_properties)
        ax[1].tick_params(axis='both', which='major', labelsize=16)

        fig.align_ylabels(ax)  # Align y-axis labels for both subplots
        plt.subplots_adjust(left=0.15)  # Adjust left margin to ensure y-axis labels are aligned

        plt.tight_layout()
        plt.savefig(os.path.join(self.config['path_to_plots'],"combined_plot_4.pdf"))
        plt.show()

    def plot_horizon_maps(self, iterated_dynamics_species1, iterated_dynamics_species2):
        # Create subplots with equal aspect ratios
        fig, ax = plt.subplots(1, 2, figsize=(10, 8), sharey=True)  # Two subplots side by side

        # Plot heatmap for iterated_dynamics_species1
        heatmap1 = ax[0].imshow(iterated_dynamics_species1.transpose(),
                                vmin=0, origin='lower', cmap='OrRd_r', aspect='equal')
        ax[0].set_xlabel('Initial time', **self.label_properties)
        ax[0].set_ylabel(self.xlabel, **self.label_properties)
        ax[0].set_title('Species 1', **self.label_properties)

        # Plot heatmap for iterated_dynamics_species2
        heatmap2 = ax[1].imshow(iterated_dynamics_species2.transpose(),
                                vmin=0, origin='lower', cmap='OrRd_r', aspect='equal')
        ax[1].set_xlabel('Initial time', **self.label_properties)
        ax[1].set_title('Species 2', **self.label_properties)

        # Reduce horizontal space between the subplots
        plt.subplots_adjust(wspace=0.1)

        # Ensure both heatmaps are the same size by using make_axes_locatable
        cax = fig.add_axes([0.85, 0.15, 0.03, 0.4]) # Adjust this to position the colorbar
        cb = plt.colorbar(heatmap2, cax=cax)
        cb.set_label('PPP', **self.label_properties)
        cb.ax.tick_params(labelsize=16)

        # Adjust tick label fontsize for both subplots
        plt.setp(ax[0].get_xticklabels(), fontsize = 16)
        plt.setp(ax[0].get_yticklabels(), fontsize = 16)
        plt.setp(ax[1].get_xticklabels(), fontsize = 16)
        plt.setp(ax[1].get_yticklabels(), fontsize = 16)

        # Adjust layout and save the plot
        plt.tight_layout(rect=[0, 0, 0.82, 1])  # Adjust rect to leave more space for the colorbar
        plt.savefig(os.path.join(self.config['path_to_plots'],'horizon_maps.pdf'))
        plt.show()

    def plot_binary_horizon_maps(self, iterated_dynamics_species1, iterated_dynamics_species2):
        # Create subplots with equal aspect ratios
        fig, ax = plt.subplots(1, 2, figsize=(10, 8), sharey=True)  # Two subplots side by side

        # Plot heatmap for iterated_dynamics_species1
        heatmap1 = ax[0].imshow(iterated_dynamics_species1.transpose(),
                                vmin=0, origin='lower', cmap='OrRd', aspect='equal')
        ax[0].set_xlabel('Initial time', self.label_properties)
        ax[0].set_ylabel(self.xlabel, self.label_properties)
        ax[0].set_title('Species 1', self.label_properties)

        # Plot heatmap for iterated_dynamics_species2
        heatmap2 = ax[1].imshow(iterated_dynamics_species2.transpose(),
                                vmin=0, origin='lower', cmap='OrRd', aspect='equal')
        ax[1].set_xlabel('Initial time', self.label_properties)
        ax[1].set_title('Species 2', self.label_properties)

        # Reduce horizontal space between the subplots
        plt.subplots_adjust(wspace=0.1)

        # Adjust tick label fontsize for both subplots
        plt.setp(ax[0].get_xticklabels(), fontsize = 16)
        plt.setp(ax[0].get_yticklabels(), fontsize = 16)
        plt.setp(ax[1].get_xticklabels(), fontsize = 16)
        plt.setp(ax[1].get_yticklabels(), fontsize = 16)

        # Adjust layout and save the plot
        plt.tight_layout()  # Adjust rect to leave more space for the colorbar
        plt.savefig(os.path.join(self.config['path_to_plots'],'horizon_maps_binary.pdf'))
        plt.show()

    def plot_ensemble_sensitivity(self, ensemble_sizes, horizons1, horizons2, ppp_rho1, ppp_rho2):

        fig, ax1 = plt.subplots(figsize=(8, 5))  # Create a figure and axis object

        # First plot: Horizons on the primary y-axis
        ax1.plot(ensemble_sizes, horizons1, color="magenta", alpha=0.8, linewidth=1.6, label="$\widehat{h}_{Species1}$")
        ax1.plot(ensemble_sizes, horizons2, color="blue", alpha=0.8, linewidth=1.6, label="$\widehat{h}_{Species2}$")
        ax1.set_xlabel("Ensemble size", **self.label_properties)
        ax1.set_ylabel("Horizon estimate", **self.label_properties)
        ax1.legend(**self.legend_properties, frameon=False, loc="upper left")  # Adjust the location if needed
        plt.setp(ax1.get_xticklabels(), **self.tick_properties)
        plt.setp(ax1.get_yticklabels(), **self.tick_properties)

        # Second plot: PPP_Rho on the secondary y-axis
        ax2 = ax1.twinx()  # Create a twin axis sharing the same x-axis
        ax2.plot(ensemble_sizes, ppp_rho1, color="purple", linestyle="--", linewidth=1.6, label="$\\varrho_{Species1}$")
        ax2.plot(ensemble_sizes, ppp_rho2, color="darkblue", linestyle="--", linewidth=1.6,
                 label="$\\varrho_{Species2}$")
        ax2.set_ylabel("$\mathbf{\\varrho_{PPP}}$", **self.label_properties)
        plt.setp(ax2.get_yticklabels(), **self.tick_properties)

        # Combined legend for both y-axes
        # Collect labels and handles from both axes for a combined legend
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2, **self.legend_properties,
                   bbox_to_anchor=(0.7, 0.5),
                   frameon=False)

        plt.tight_layout()
        plt.savefig(os.path.join(self.config['path_to_plots'], "plot_ensemble_size_sensitivity.pdf"))
        plt.show()