import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

class VisualisationModule:

    def __init__(self, 
                 network, 
                 station, 
                 variable, 
                 maximum_leadtime, 
                 doy_vector,
                 evaluation, 
                 path_to_plots):

        self.network = network
        self.station = station
        self.variable = variable
        self.maximum_leadtime = maximum_leadtime
        self.doy_vector = doy_vector
        self.evaluation = evaluation
        self.path_to_plots = path_to_plots

        self.ylabel = 'Soil temperature [K]' if self.variable == 'st' else 'Soil moisture [mm]'
        self.xlabel = 'Lead time [6-hrly]'
        self.title1 = "Surface Layer"
        self.title2 = "Subsurface Layer 1"
        self.title3 = "Subsurface Layer 2"
        self.linewidth = 2
        self.figsize = (5, 13)

        self.network_name = self.network.split('_')[1]
        self.year = self.network.split('_')[2]
        self.label_properties = {'weight':'bold', 'size':16}
        self.legend_properties = {'weight':'bold', 'size':14}
        self.tick_properties = {'size':16}
        
    def plot_station_data(self, y_prog, station_data, matching_indices):

        doy_vector = self.doy_vector[:self.maximum_leadtime]
        step = len(doy_vector) // 4

        fig, ax = plt.subplots(3, 1, figsize=self.figsize, sharex=True, sharey=True) 

        ax[0].plot(doy_vector,
                   y_prog[:self.maximum_leadtime,:,matching_indices[0]], color="cyan", label="ECland")
        ax[0].plot(doy_vector,
                   station_data[:self.maximum_leadtime,:,0], color="green", label=f"{self.station}")
        #ax[0].set_xlabel(self.xlabel, **self.label_properties)
        ax[0].legend(prop=self.legend_properties, loc = "upper right")

        ax[1].plot(doy_vector,
                   y_prog[:self.maximum_leadtime,:,matching_indices[1]], color="cyan", label="ECland")
        ax[1].plot(doy_vector,
                   station_data[:self.maximum_leadtime,:,1], color="green", label=f"{self.station}")
        #ax[1].set_xlabel(self.xlabel, **self.label_properties)
        ax[1].legend(prop=self.legend_properties, loc = "upper right")

        ax[2].plot(doy_vector,
                   y_prog[:self.maximum_leadtime,:,matching_indices[2]], color="cyan", label="ECland")
        ax[2].plot(doy_vector,
                   station_data[:self.maximum_leadtime,:,2], color="green", label=f"{self.station}")  
        ax[2].set_xlabel(self.xlabel, **self.label_properties)
        ax[2].legend(prop=self.legend_properties, loc = "upper right")

        ax[0].set_title("Surface Layer", **self.label_properties)
        ax[1].set_title("Subsurface Layer 1", **self.label_properties)
        ax[2].set_title("Subsurface Layer 2", **self.label_properties)

        for a in ax:
            tick_positions = doy_vector[::step]  # Adjust frequency as needed
            a.set_xticks(tick_positions)
            a.set_xticklabels([pd.Timestamp(t).strftime('%Y-%m-%d') for t in tick_positions], rotation=25)
            a.set_ylabel(self.ylabel, **self.label_properties)
            plt.setp(a.get_yticklabels(), **self.tick_properties)
            plt.setp(a.get_xticklabels(), **self.tick_properties)


        plt.tight_layout()
        plt.savefig(os.path.join(self.path_to_plots, f'{self.network_name}_{self.station}_{self.year}_{self.variable}_timeseries.pdf'))
        plt.show()

    def plot_station_data_and_forecast(self, dynamic_features_dict, dynamic_features_prediction_dict, station_data, matching_indices):

        doy_vector = self.doy_vector[:self.maximum_leadtime]
        step = len(doy_vector) // 4

        fig, ax = plt.subplots(3, 1, figsize=self.figsize, sharex=True, sharey=True) 

        if 'mlp' in dynamic_features_prediction_dict:
            for i, a in enumerate(ax):
                ax[i].plot(doy_vector,
                           dynamic_features_prediction_dict['mlp'][:self.maximum_leadtime,:,matching_indices[i]], 
                           color="magenta", label="MLP", alpha = 0.8, linewidth = self.linewidth)
        if 'lstm' in dynamic_features_prediction_dict:
            for i, a in enumerate(ax):
                ax[i].plot(doy_vector,
                           dynamic_features_prediction_dict['lstm'][:self.maximum_leadtime,:,matching_indices[i]], 
                           color="purple", label="LSTM", alpha = 0.8, linewidth = self.linewidth)        
        if 'xgb' in dynamic_features_prediction_dict:
            for i, a in enumerate(ax):
                ax[i].plot(doy_vector,
                           dynamic_features_prediction_dict['xgb'][:self.maximum_leadtime,:,matching_indices[i]], 
                           color="pink", label="XGB", alpha = 0.9, linewidth = self.linewidth) 
                
        ax[0].plot(doy_vector, dynamic_features_dict['mlp'][:self.maximum_leadtime,:,matching_indices[0]], 
                   color="cyan", label="ECLand", alpha = 0.7, linewidth = self.linewidth)
        ax[0].plot(doy_vector, station_data[:self.maximum_leadtime,:,0], 
                   color="green", label=self.station, alpha = 0.7, linewidth = self.linewidth)
        ax[0].legend(prop=self.legend_properties, frameon=True)

        ax[1].plot(doy_vector, dynamic_features_dict['mlp'][:self.maximum_leadtime,:,matching_indices[1]], 
                   color="cyan", label="ECLand", alpha = 0.7, linewidth = self.linewidth)
        ax[1].plot(doy_vector, station_data[:self.maximum_leadtime,:,1], 
                   color="green", label=self.station, alpha = 0.7, linewidth = self.linewidth)
        
        ax[2].plot(doy_vector, dynamic_features_dict['mlp'][:self.maximum_leadtime,:,matching_indices[2]], 
                   color="cyan", label="ECLand", alpha = 0.7, linewidth = self.linewidth)
        ax[2].plot(doy_vector, station_data[:self.maximum_leadtime,:,2], 
                   color="green", label=self.station, alpha = 0.7, linewidth = self.linewidth)
        ax[2].set_xlabel(self.xlabel, **self.label_properties)

        ax[0].set_title("Surface Layer", **self.label_properties)
        ax[1].set_title("Subsurface Layer 1", **self.label_properties)
        ax[2].set_title("Subsurface Layer 2", **self.label_properties)

        for a in ax:
            tick_positions = doy_vector[::step]  # Adjust frequency as needed
            a.set_xticks(tick_positions)
            a.set_xticklabels([pd.Timestamp(t).strftime('%Y-%m-%d') for t in tick_positions], rotation=25)
            #a.set_xlabel(self.xlabel, **self.label_properties)
            a.set_ylabel(self.ylabel, **self.label_properties)
            plt.setp(a.get_yticklabels(), **self.tick_properties)
            plt.setp(a.get_xticklabels(), **self.tick_properties)

        plt.tight_layout()
        plt.savefig(os.path.join(self.path_to_plots, f'{self.network_name}_{self.station}_{self.year}_{self.variable}_timeseries_forecast.pdf'))
        plt.show()

    def plot_scores(self, scores_l1, scores_l2, scores_3, score, hod = None, log_y=True):

        if hod is None:
            doy_vector = self.doy_vector[:self.maximum_leadtime]
            step = len(doy_vector) // 4
        else:
            doy_vector = self.doy_vector[:self.maximum_leadtime*4]
            doy_vector = doy_vector[::4]

        fig, ax = plt.subplots(3, 1, figsize=self.figsize, sharex=True, sharey=True) 

        x_label = self.xlabel if hod is None else "Lead time [day]"
        colors = ["cyan", "magenta", "purple", "pink"]
        i = 0
        for model, scores in scores_l1.items():
            ax[0].plot(doy_vector, scores, color = colors[i], alpha = 0.8, label = model, linewidth = self.linewidth)
            i += 1
        ax[0].legend(prop=self.legend_properties, frameon=True)

        i = 0
        for model, scores in scores_l2.items():
            ax[1].plot(doy_vector, scores, color = colors[i], alpha = 0.8, label = model, linewidth = self.linewidth)
            i += 1

        i = 0
        for model, scores in scores_3.items():
            ax[2].plot(doy_vector, scores, color = colors[i], alpha = 0.9, label = model, linewidth = self.linewidth)
            i += 1
        ax[2].set_xlabel(x_label, **self.label_properties)
        
        ax[0].set_title("Surface Layer", **self.label_properties)
        ax[1].set_title("Subsurface Layer 1", **self.label_properties)
        ax[2].set_title("Subsurface Layer 2", **self.label_properties)

        for a in ax:
            a.hlines(0, xmin = min(doy_vector), xmax=max(doy_vector), 
                     color = "black", alpha = 0.8, linestyle = '--',
                     linewidth = self.linewidth)
            #a.set_xlabel(x_label, **self.label_properties)
            a.set_ylabel(f"{score}", **self.label_properties)
            if log_y:
                a.set_yscale('log')
            tick_positions = doy_vector[::step]  # Adjust frequency as needed
            a.set_xticks(tick_positions)
            a.set_xticklabels([pd.Timestamp(t).strftime('%Y-%m-%d') for t in tick_positions], rotation=25)
            plt.setp(a.get_yticklabels(), **self.tick_properties)
            plt.setp(a.get_xticklabels(), **self.tick_properties)
   
        plt.tight_layout()
        if hod is None:
            fig_path = os.path.join(self.path_to_plots, f'{self.network_name}_{self.station}_{self.year}_{self.variable}_{self.evaluation}_scores.pdf')
        else:
            fig_path = os.path.join(self.path_to_plots, f'{self.network_name}_{self.station}_{self.year}_{self.variable}_{self.evaluation}_hod{hod}_scores.pdf')
        plt.savefig(fig_path)
        plt.show()

    def plot_horizons(self, scores_l1, scores_l2, scores_3, score, threshold, hod = None, log_y=True):

        if hod is None:
            doy_vector = self.doy_vector[:self.maximum_leadtime]
            step = len(doy_vector) // 4
        else:
            doy_vector = self.doy_vector[:self.maximum_leadtime*4]
            doy_vector = doy_vector[::4]

        fig, ax = plt.subplots(3, 1, figsize=self.figsize, sharex=True, sharey=True) 

        x_label = self.xlabel if hod is None else "Lead time [day]"
        colors = ["cyan", "magenta", "purple", "pink"]
        i = 0
        for model, scores in scores_l1.items():
            ax[0].plot(doy_vector, threshold - scores, color = colors[i], alpha = 0.8, label = model, linewidth = self.linewidth)
            i += 1
        ax[0].legend(prop=self.legend_properties, frameon=True)

        i = 0
        for model, scores in scores_l2.items():
            ax[1].plot(doy_vector, threshold - scores, color = colors[i], alpha = 0.8, label = model, linewidth = self.linewidth)
            i += 1

        i = 0
        for model, scores in scores_3.items():
            ax[2].plot(doy_vector, threshold - scores, color = colors[i], alpha = 0.8, label = model, linewidth = self.linewidth)
            i += 1
        ax[2].set_xlabel(x_label, **self.label_properties)

        ax[0].set_title("Surface Layer", **self.label_properties)
        ax[1].set_title("Subsurface Layer 1", **self.label_properties)
        ax[2].set_title("Subsurface Layer 2", **self.label_properties)

        for a in ax:
            a.hlines(0, xmin = min(doy_vector), xmax=max(doy_vector), 
                     color = "black", alpha = 0.8, linestyle = '--',
                     linewidth = self.linewidth)
            #a.set_xlabel(x_label, **self.label_properties)
            a.set_ylabel(f"{score} - $\\varrho$", **self.label_properties)

            #a.legend(prop=self.legend_properties, frameon=False)
            if log_y:
                a.set_yscale('log')
            tick_positions = doy_vector[::step]  # Adjust frequency as needed
            a.set_xticks(tick_positions)
            a.set_xticklabels([pd.Timestamp(t).strftime('%Y-%m-%d') for t in tick_positions], rotation=25)

            plt.setp(a.get_yticklabels(), **self.tick_properties)
            plt.setp(a.get_xticklabels(), **self.tick_properties)

        plt.tight_layout()
        if hod is None:
            fig_path = os.path.join(self.path_to_plots, f'{self.network_name}_{self.station}_{self.year}_{self.variable}_{self.evaluation}_horizons.pdf')
        else:
            fig_path = os.path.join(self.path_to_plots, f'{self.network_name}_{self.station}_{self.year}_{self.variable}_{self.evaluation}_hod{hod}_horizons.pdf')
        plt.savefig(fig_path)
        plt.show()

    def plot_skill_scores(self, skill_scores_l1, skill_scores_l2, skill_scores_l3, score, hod = None, log_y = True, sharey = True, invert = False):

        if hod is None:
            doy_vector = self.doy_vector[:self.maximum_leadtime]
            step = len(doy_vector) // 4
        else:
            doy_vector = self.doy_vector[:self.maximum_leadtime*4]
            doy_vector = doy_vector[::4]
        
        fig, ax = plt.subplots(3, 1, figsize=self.figsize, sharex=True, sharey=sharey) 

        x_label = self.xlabel if hod is None else "Lead time[day]"
        colors = ["magenta", "purple", "pink"]

        ax[0].axhspan(0, 1, facecolor='lightgray', alpha=0.5)  
        i = 0
        for model, scores in skill_scores_l1.items():
            if invert:
                scores = 1-scores
            ax[0].plot(doy_vector, scores, color = colors[i], alpha = 0.8, label = model, linewidth = self.linewidth)
            i += 1
        #ax[0].set_ylabel(f"{score}-SS", **self.label_properties)
        ax[0].legend(prop=self.legend_properties, frameon=True)

        ax[1].axhspan(0, 1, facecolor='lightgray', alpha=0.5) 
        i = 0
        for model, scores in skill_scores_l2.items():
            if invert:
                scores = 1-scores
            ax[1].plot(doy_vector, scores, color = colors[i], alpha = 0.8, label = model, linewidth = self.linewidth)
            i += 1

        ax[2].axhspan(0, 1, facecolor='lightgray', alpha=0.5) 
        i = 0
        for model, scores in skill_scores_l3.items():
            if invert:
                scores = 1-scores
            ax[2].plot(doy_vector, scores, color = colors[i], alpha = 0.8, label = model, linewidth = self.linewidth)
            i += 1
        ax[2].set_xlabel(x_label, **self.label_properties)
            
        ax[0].set_title("Surface Layer", **self.label_properties)
        ax[1].set_title("Subsurface Layer 1", **self.label_properties)
        ax[2].set_title("Subsurface Layer 2", **self.label_properties)

        for a in ax:
            a.hlines(0, xmin = min(doy_vector), xmax=max(doy_vector), 
                     color = "black", alpha = 0.8, linestyle = '--', linewidth = self.linewidth)
            #a.set_xlabel(x_label, **self.label_properties)
            a.set_ylabel(f"{score}-SS", **self.label_properties)
            a.set_ylim(-1,1)

            #a.legend(prop=self.legend_properties, frameon=False)
            if log_y:
                a.set_yscale('log')
            tick_positions = doy_vector[::step]  # Adjust frequency as needed
            a.set_xticks(tick_positions)
            a.set_xticklabels([pd.Timestamp(t).strftime('%Y-%m-%d') for t in tick_positions], rotation=25)

            plt.setp(a.get_yticklabels(), **self.tick_properties)
            plt.setp(a.get_xticklabels(), **self.tick_properties)

        plt.tight_layout()
        if hod is None:
            fig_path = os.path.join(self.path_to_plots, f'{self.network_name}_{self.station}_{self.year}_{self.variable}_{self.evaluation}_skillscores.pdf')
        else:
            fig_path = os.path.join(self.path_to_plots, f'{self.network_name}_{self.station}_{self.year}_{self.variable}_{self.evaluation}_hod{hod}_skillscores.pdf')
        plt.savefig(fig_path)
        plt.show()