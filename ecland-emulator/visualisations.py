import matplotlib.pyplot as plt
import os
import numpy as np

class VisualisationModule:

    def __init__(self, network, station, variable, maximum_leadtime, path_to_plots):

        self.network = network
        self.station = station
        self.variable = variable
        self.maximum_leadtime = maximum_leadtime
        self.path_to_plots = path_to_plots

        self.ylabel = 'Soil temperature [K]' if self.variable == 'st' else 'Soil moisture [mm]'
        self.network_name = self.network.split('_')[1]
        self.year = self.network.split('_')[2]
        self.label_properties = {'weight':'bold', 'size':16}
        self.legend_properties = {'weight':'bold', 'size':14}

    def plot_station_data(self, y_prog, station_data, matching_indices):

        fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True) 

        ax[0].plot(y_prog[:self.maximum_leadtime,:,matching_indices[0]], color="cyan", label="ECland")
        ax[0].plot(station_data[:self.maximum_leadtime,:,0], color="green", label=f"{self.station}")
        ax[0].set_title("Surface Layer", **self.label_properties)
        ax[0].set_xlabel('Lead time', **self.label_properties)
        ax[0].set_ylabel(self.ylabel, **self.label_properties)
        ax[0].legend(prop=self.legend_properties, loc = "upper right")

        ax[1].plot(y_prog[:self.maximum_leadtime,:,matching_indices[1]], color="cyan", label="ECland")
        ax[1].plot(station_data[:self.maximum_leadtime,:,1], color="green", label=f"{self.station}")
        ax[1].set_xlabel('Lead time', **self.label_properties)
        ax[1].set_title("Subsurface Layer 1", **self.label_properties)
        ax[1].legend(prop=self.legend_properties, loc = "upper right")

        ax[2].plot(y_prog[:self.maximum_leadtime,:,matching_indices[2]], color="cyan", label="ECland")
        ax[2].plot(station_data[:self.maximum_leadtime,:,2], color="green", label=f"{self.station}")  
        ax[2].set_xlabel('Lead time', **self.label_properties)
        ax[2].set_title("Subsurface Layer 2", **self.label_properties)
        ax[2].legend(prop=self.legend_properties, loc = "upper right")

        plt.tight_layout()
        plt.savefig(os.path.join(self.path_to_plots, f'{self.network_name}_{self.station}_{self.year}_{self.variable}_timeseries.pdf'))
        plt.show()

    def plot_station_data_and_forecast(self, dynamic_features_dict, dynamic_features_prediction_dict, station_data, matching_indices):

        fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True) 

        if 'mlp' in dynamic_features_prediction_dict:
            ax[0].plot(dynamic_features_prediction_dict['mlp'][:self.maximum_leadtime,:,matching_indices[0]], color="magenta", label="MLP", alpha = 0.6)
            ax[1].plot(dynamic_features_prediction_dict['mlp'][:self.maximum_leadtime,:,matching_indices[1]], color="magenta", label="MLP", alpha = 0.6)
            ax[2].plot(dynamic_features_prediction_dict['mlp'][:self.maximum_leadtime,:,matching_indices[2]], color="magenta", label="MLP", alpha = 0.6)
        if 'lstm' in dynamic_features_prediction_dict:
            ax[0].plot(dynamic_features_prediction_dict['lstm'][:self.maximum_leadtime,:,matching_indices[0]], color="purple", label="LSTM", alpha = 0.6)
            ax[1].plot(dynamic_features_prediction_dict['lstm'][:self.maximum_leadtime,:,matching_indices[1]], color="purple", label="LSTM", alpha = 0.6)
            ax[2].plot(dynamic_features_prediction_dict['lstm'][:self.maximum_leadtime,:,matching_indices[2]], color="purple", label="LSTM", alpha = 0.6)
        if 'xgb' in dynamic_features_prediction_dict:
            ax[0].plot(dynamic_features_prediction_dict['xgb'][:self.maximum_leadtime,:,matching_indices[0]], color="pink", label="XGB", alpha = 0.6)
            ax[1].plot(dynamic_features_prediction_dict['xgb'][:self.maximum_leadtime,:,matching_indices[1]], color="pink", label="XGM", alpha = 0.6)
            ax[2].plot(dynamic_features_prediction_dict['xgb'][:self.maximum_leadtime,:,matching_indices[2]], color="pink", label="XGB", alpha = 0.6)

        ax[0].plot(dynamic_features_dict['mlp'][:self.maximum_leadtime,:,matching_indices[0]], color="cyan", label="ECland", alpha = 0.6)
        ax[0].plot(station_data[:self.maximum_leadtime,:,0], color="green", label=self.station, alpha = 0.6)
        ax[0].set_xlabel('Lead time', **self.label_properties)
        ax[0].set_ylabel(self.ylabel, **self.label_properties)

        ax[1].plot(dynamic_features_dict['mlp'][:self.maximum_leadtime,:,matching_indices[1]], color="cyan", label="ECland", alpha = 0.6)
        ax[1].plot(station_data[:self.maximum_leadtime,:,1], color="green", label=self.station, alpha = 0.6)
        ax[1].set_xlabel('Lead time', **self.label_properties)

        ax[2].plot(dynamic_features_dict['mlp'][:self.maximum_leadtime,:,matching_indices[2]], color="cyan", label="ECland", alpha = 0.6)
        ax[2].plot(station_data[:self.maximum_leadtime,:,2], color="green", label=self.station, alpha = 0.6)
        ax[2].set_xlabel('Lead time', **self.label_properties)

        ax[0].set_title("Surface Layer", **self.label_properties)
        ax[0].legend(prop=self.legend_properties)
        ax[1].set_title("Subsurface Layer 1", **self.label_properties)
        ax[1].legend(prop=self.legend_properties)
        ax[2].set_title("Subsurface Layer 2", **self.label_properties)
        ax[2].legend(prop=self.legend_properties)

        plt.tight_layout()
        plt.savefig(os.path.join(self.path_to_plots, f'{self.network_name}_{self.station}_{self.year}_{self.variable}_timeseries_forecast.pdf'))
        plt.show()

    def plot_scores(self, scores_l1, scores_l2, scores_3, score, hod = None, log_y=True):

        fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True) 

        x_label = "Lead time [6-hourly]" if hod is None else "Lead time [day]"
        colors = ["cyan", "magenta", "purple", "pink"]
        i = 0
        for model, scores in scores_l1.items():
            ax[0].plot(scores, color = colors[i], alpha = 0.7, label = model)
            i += 1
        ax[0].hlines(0, xmin = 0, xmax=self.maximum_leadtime, color = "black", alpha = 0.8, linestyle = '--')
        ax[0].set_ylabel(f"{score}", **self.label_properties)
        ax[0].set_xlabel(x_label, **self.label_properties)
        ax[0].legend(prop=self.legend_properties, loc = "lower right")

        i = 0
        for model, scores in scores_l2.items():
            ax[1].plot(scores, color = colors[i], alpha = 0.7, label = model)
            i += 1
        ax[1].hlines(0, xmin = 0, xmax=self.maximum_leadtime, color = "black", alpha = 0.8, linestyle = '--')
        ax[1].set_xlabel(x_label, **self.label_properties)
        ax[1].legend(prop=self.legend_properties, loc = "lower right")

        i = 0
        for model, scores in scores_3.items():
            ax[2].plot(scores, color = colors[i], alpha = 0.7, label = model)
            i += 1
        ax[2].hlines(0, xmin = 0, xmax=self.maximum_leadtime, color = "black", alpha = 0.8, linestyle = '--')
        ax[2].set_xlabel(x_label, **self.label_properties)
        ax[2].legend(prop=self.legend_properties, loc = "lower right")

        if log_y:
            ax[0].set_yscale('log')
            ax[1].set_yscale('log')
            ax[2].set_yscale('log')
        ax[0].set_title("Surface Layer", **self.label_properties)
        ax[1].set_title("Subsurface Layer 1", **self.label_properties)
        ax[2].set_title("Subsurface Layer 2", **self.label_properties)

        plt.tight_layout()
        if hod is None:
            fig_path = os.path.join(self.path_to_plots, f'{self.network_name}_{self.station}_{self.year}_{self.variable}_scores.pdf')
        else:
            fig_path = os.path.join(self.path_to_plots, f'{self.network_name}_{self.station}_{self.year}_{self.variable}_hod{hod}_scores.pdf')
        plt.savefig(fig_path)
        plt.show()

    def plot_horizons(self, scores_l1, scores_l2, scores_3, score, threshold, hod = None, log_y=True):

        fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True) 

        x_label = "Lead time [6-hourly]" if hod is None else "Lead time [day]"
        colors = ["cyan", "magenta", "purple", "pink"]
        i = 0
        for model, scores in scores_l1.items():
            ax[0].plot(threshold - scores, color = colors[i], alpha = 0.7, label = model)
            i += 1
        ax[0].hlines(0, xmin = 0, xmax=self.maximum_leadtime, color = "black", alpha = 0.8, linestyle = '--')
        ax[0].set_ylabel(f"{score} - $\\varrho$", **self.label_properties)
        ax[0].set_xlabel(x_label, **self.label_properties)
        ax[0].legend(prop=self.legend_properties, loc = "lower right")

        i = 0
        for model, scores in scores_l2.items():
            ax[1].plot(threshold - scores, color = colors[i], alpha = 0.7, label = model)
            i += 1
        ax[1].hlines(0, xmin = 0, xmax=self.maximum_leadtime, color = "black", alpha = 0.8, linestyle = '--')
        ax[1].set_xlabel(x_label, **self.label_properties)
        ax[1].legend(prop=self.legend_properties, loc = "lower right")

        i = 0
        for model, scores in scores_3.items():
            ax[2].plot(threshold - scores, color = colors[i], alpha = 0.7, label = model)
            i += 1
        ax[2].hlines(0, xmin = 0, xmax=self.maximum_leadtime, color = "black", alpha = 0.8, linestyle = '--')
        ax[2].set_xlabel(x_label, **self.label_properties)
        ax[2].legend(prop=self.legend_properties, loc = "lower right")

        if log_y:
            ax[0].set_yscale('log')
            ax[1].set_yscale('log')
            ax[2].set_yscale('log')
        ax[0].set_title("Surface Layer", **self.label_properties)
        ax[1].set_title("Subsurface Layer 1", **self.label_properties)
        ax[2].set_title("Subsurface Layer 2", **self.label_properties)

        plt.tight_layout()
        if hod is None:
            fig_path = os.path.join(self.path_to_plots, f'{self.network_name}_{self.station}_{self.year}_{self.variable}_horizons.pdf')
        else:
            fig_path = os.path.join(self.path_to_plots, f'{self.network_name}_{self.station}_{self.year}_{self.variable}_hod{hod}_horizons.pdf')
        plt.savefig(fig_path)
        plt.show()

    def plot_skill_scores(self, skill_scores_l1, skill_scores_l2, skill_scores_l3, score, hod = None, log_y = True):

        fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True) 

        x_label = "Lead time [6-hourly]" if hod is None else "Lead time[day]"
        colors = ["magenta", "purple", "pink"]
        ax[0].axhspan(0, 1, facecolor='lightgray', alpha=0.5)  
        i = 0
        for model, scores in skill_scores_l1.items():
            ax[0].plot(scores, color = colors[i], alpha = 0.7, label = model)
            i += 1
        ax[0].hlines(1, xmin = 0, xmax=self.maximum_leadtime, color = "black", alpha = 0.8, label = "$\\rho$", linestyle = '--')
        ax[0].set_ylabel(f"{score}-SS", **self.label_properties)
        ax[0].set_xlabel(x_label, **self.label_properties)
        ax[0].legend(prop=self.legend_properties, loc = "lower right")

        ax[1].axhspan(0, 1, facecolor='lightgray', alpha=0.5) 
        i = 0
        for model, scores in skill_scores_l2.items():
            ax[1].plot(scores, color = colors[i], alpha = 0.7, label = model)
            i += 1
        ax[1].hlines(1, xmin = 0, xmax=self.maximum_leadtime, color = "black", alpha = 0.8, label = "$\\rho$", linestyle = '--')
        ax[1].set_xlabel(x_label, **self.label_properties)
        ax[1].legend(prop=self.legend_properties, loc = "lower right")

        ax[2].axhspan(0, 1, facecolor='lightgray', alpha=0.5) 
        i = 0
        for model, scores in skill_scores_l3.items():
            ax[2].plot(scores, color = colors[i], alpha = 0.7, label = model)
            i += 1
        ax[2].hlines(1, xmin = 0, xmax=self.maximum_leadtime, color = "black", alpha = 0.8, label = "$\\rho$", linestyle = '--')
        ax[2].set_xlabel(x_label, **self.label_properties)
        ax[2].legend(prop=self.legend_properties, loc = "lower right")

        if log_y:
            ax[0].set_yscale('log')
            ax[1].set_yscale('log')
            ax[2].set_yscale('log')
            
        ax[0].set_title("Surface Layer", **self.label_properties)
        ax[1].set_title("Subsurface Layer 1", **self.label_properties)
        ax[2].set_title("Subsurface Layer 2", **self.label_properties)

        plt.tight_layout()
        if hod is None:
            fig_path = os.path.join(self.path_to_plots, f'{self.network_name}_{self.station}_{self.year}_{self.variable}_skillscores.pdf')
        else:
            fig_path = os.path.join(self.path_to_plots, f'{self.network_name}_{self.station}_{self.year}_{self.variable}_hod{hod}_skillscores.pdf')
        plt.savefig(fig_path)
        plt.show()