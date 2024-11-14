import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

class VisualisationClass:

    def __init__(self, 
                 network, 
                 station, 
                 variable, 
                 maximum_leadtime, 
                 score,
                 doy_vector,
                 evaluation, 
                 path_to_plots):

        self.network = network
        self.station = station
        self.variable = variable
        self.score = score
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

class VisualisationSingle(VisualisationClass):
        
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

    def plot_scores(self, scores_l1, scores_l2, scores_3, hod = None, log_y=True):

        if hod is None:
            doy_vector = self.doy_vector[:self.maximum_leadtime]
            step = len(doy_vector) // 4
        else:
            doy_vector = self.doy_vector[:self.maximum_leadtime*4]
            doy_vector = doy_vector[::4]
            step = len(doy_vector) // 4

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
            a.set_ylabel(f"{self.score}", **self.label_properties)
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

    def plot_horizons(self, scores_l1, scores_l2, scores_l3, threshold,
                      scores_l1_std = None, scores_l2_std = None, scores_l3_std = None, hod = None, log_y=True):

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
            shifted_score = threshold - scores
            if (scores_l1_std is not None) and (model in scores_l1_std.keys()):
                scores_std = scores_l1_std[model]
                upper = shifted_score + 4.3*scores_std
                lower = shifted_score - 4.3*scores_std
                ax[0].fill_between(doy_vector, upper, lower, color = colors[i], alpha = 0.2)

            ax[0].plot(doy_vector, shifted_score, color = colors[i], alpha = 0.8, label = model, linewidth = self.linewidth)
            i += 1
        ax[0].legend(prop=self.legend_properties, frameon=True)

        i = 0
        for model, scores in scores_l2.items():
            shifted_score = threshold - scores
            if (scores_l2_std is not None) and (model in scores_l2_std.keys()):
                scores_std = scores_l2_std[model]
                upper = shifted_score + 4.3*scores_std
                lower = shifted_score - 4.3*scores_std
                ax[1].fill_between(doy_vector, upper, lower, color = colors[i], alpha = 0.2)

            ax[1].plot(doy_vector, threshold - scores, color = colors[i], alpha = 0.8, label = model, linewidth = self.linewidth)
            i += 1

        i = 0
        for model, scores in scores_l3.items():
            shifted_score = threshold - scores
            if (scores_l3_std is not None) and (model in scores_l3_std.keys()):
                scores_std = scores_l3_std[model]
                upper = shifted_score + 4.3*scores_std
                lower = shifted_score - 4.3*scores_std
                ax[2].fill_between(doy_vector, upper, lower, color = colors[i], alpha = 0.2)
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
            a.set_ylabel(f"$\\varrho-${self.score}", **self.label_properties)

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

    def plot_skill_scores(self, skill_scores_l1, skill_scores_l2, skill_scores_l3, hod = None, log_y = True, sharey = True, invert = False):

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
            a.set_ylabel(f"{self.score}-SS", **self.label_properties)
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

    def plot_initial_state_perturbation(self, ensemble_dict, fc_numerical, observations, ens_size):
        
        doy_vector = self.doy_vector[:self.maximum_leadtime]
        step = len(doy_vector) // 4

        fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True) 
        
        ax[0].set_title("Surface Layer", **self.label_properties)
        ax[1].set_title("Subsurface Layer 1", **self.label_properties)
        ax[2].set_title("Subsurface Layer 2", **self.label_properties)

        colors = ["purple", "purple", "purple"]
        j=0
        for key, ensemble_prediction in ensemble_dict.items():
            for i in range(ens_size):
                ax[0].plot(doy_vector, ensemble_prediction[i, :,3], color=colors[j], alpha = 0.3)
            j+=1
        ax[0].plot(doy_vector, ensemble_prediction[...,3].mean(axis=0), color="magenta", alpha = 0.7,label="AILand Ensemble Mean", linewidth = self.linewidth)
        ax[0].plot(doy_vector, fc_numerical[:,3], color="cyan", label="ECLand", alpha = 0.7, linewidth = self.linewidth)
        ax[0].plot(doy_vector, observations[:,:,0], color="lime", label=self.station, alpha = 0.7, linewidth = self.linewidth)

        j=0
        for key, ensemble_prediction in ensemble_dict.items():
            for i in range(ens_size):
                ax[1].plot(doy_vector, ensemble_prediction[i, :,4], color=colors[j],  alpha = 0.3)
            j+=1
        ax[1].plot(doy_vector, ensemble_prediction[...,4].mean(axis=0), color="magenta", alpha = 0.7,label="AILand Ensemble Mean", linewidth = self.linewidth)
        ax[1].plot(doy_vector, fc_numerical[:,4], color="cyan", label="ECLand", alpha = 0.7, linewidth = self.linewidth)
        ax[1].plot(doy_vector, observations[:,:,1], color="lime", label=self.station, alpha = 0.7, linewidth = self.linewidth)
        ax[1].legend(prop=self.legend_properties, frameon=True)

        j=0
        for key, ensemble_prediction in ensemble_dict.items():
            for i in range(ens_size):
                ax[2].plot(doy_vector, ensemble_prediction[i, :,5], color=colors[j], alpha = 0.3)
            j+=1
        ax[2].plot(doy_vector, ensemble_prediction[...,5].mean(axis=0), color="magenta", alpha = 0.7,label="AILand Ensemble Mean", linewidth = self.linewidth)
        ax[2].plot(doy_vector, fc_numerical[:,5], color="cyan", label="ECLand", alpha = 0.7, linewidth = self.linewidth)
        ax[2].plot(doy_vector, observations[:,:,2], color="lime", label=self.station, alpha = 0.7, linewidth = self.linewidth) 

        for a in ax:
            a.set_xlabel(self.xlabel, **self.label_properties)
            a.set_ylabel(f"{self.ylabel}", **self.label_properties)
            tick_positions = doy_vector[::step]  # Adjust frequency as needed
            a.set_xticks(tick_positions)
            a.set_xticklabels([pd.Timestamp(t).strftime('%Y-%m-%d') for t in tick_positions], rotation=25)

            plt.setp(a.get_yticklabels(), **self.tick_properties)
            plt.setp(a.get_xticklabels(), **self.tick_properties)

        plt.tight_layout()
        fig_path = os.path.join(self.path_to_plots, f'{self.network_name}_{self.station}_{self.year}_{self.variable}_{self.evaluation}_ICperturbed.pdf')
        plt.savefig(fig_path)
        plt.show()


class VisualisationMany(VisualisationClass):

    def layer_forecast(self, layer):

        observations = []
        fc_numerical = []
        fc_emulators = []

        for station_name, station in self.forecast_dict.items():

            observations.append(station[layer]['observations'])
            fc_numerical.append(station[layer]['fc_numerical'])
            fc_emulators.append(station[layer]['fc_emulators'])

        observations = np.array(observations).squeeze()
        fc_numerical = np.array(fc_numerical).squeeze()
        fc_emulators = np.array(fc_emulators).squeeze()

        obs_mean = np.nanquantile(observations, 0.5, axis=0, interpolation='linear')
        num_mean = np.nanquantile(fc_numerical, 0.5, axis=0, interpolation='linear')
        em_ens_mean = np.nanquantile(fc_emulators, 0.5, axis=1, interpolation='linear') # model ensemble std for 3 models only
        em_mean = np.nanquantile(em_ens_mean, 0.5, axis=1, interpolation='linear') # total mean

        obs_std_upper = np.nanquantile(observations, 0.975, axis=0, interpolation='linear')
        obs_std_lower = np.nanquantile(observations, 0.025, axis=0, interpolation='linear')
        num_std_upper = np.nanquantile(fc_numerical, 0.975, axis=0, interpolation='linear')
        num_std_lower = np.nanquantile(fc_numerical, 0.025, axis=0, interpolation='linear')
        em_ens_std_upper = np.nanquantile(fc_emulators, 0.975, axis=1, interpolation='linear')
        em_ens_std_lower = np.nanquantile(fc_emulators, 0.025, axis=1, interpolation='linear')
        em_std_upper = np.nanquantile(em_ens_mean, 0.975, axis=1, interpolation='linear')
        em_std_lower = np.nanquantile(em_ens_mean, 0.025, axis=1, interpolation='linear')

        #obs_mean = np.nanmean(observations, axis=0).transpose()
        #obs_std = np.nanstd(observations, axis=0).transpose()
        #num_mean = np.nanmean(fc_numerical, axis=0).transpose()
        #num_std = np.nanstd(fc_numerical, axis=0).transpose()
        #em_ens_mean = np.nanmean(fc_emulators, axis=1).transpose()
        #em_ens_std = 4.3 * np.nanstd(fc_emulators, axis=1).transpose() / np.sqrt(3) # model ensemble std for 3 models only
        #em_mean = np.nanmean(em_ens_mean, axis=1) # total mean
        #em_std = np.sqrt((np.nan_to_num(em_ens_std**2).sum(axis=1)) / em_ens_std.shape[1]) # propagate uncertainty from ens_std.

        return (obs_mean, obs_std, num_mean, num_std, em_mean, em_std)

    def layer_statistics(self, layer):
        
        scores_numerical = []
        scores_emulators = []
        scores_dispersion = []
        skill_scores = []

        # Collect data for each station
        for station_name, station in self.stations_dict.items():
            scores_numerical.append(station[layer]['scores']['ECLand'])
            scores_emulators.append(station[layer]['scores']['Emulators'])
            scores_dispersion.append(station[layer]['scores_dispersion']['Emulators'])
            skill_scores.append(station[layer]['skill_scores']['Emulators'])

        # Convert lists to arrays
        scores_numerical = np.array(scores_numerical)
        scores_emulators = np.array(scores_emulators)
        scores_dispersion = np.array(scores_dispersion)
        skill_scores = np.array(skill_scores)

        # Check for NaNs in the input data
        if np.isnan(scores_numerical).any() or np.isnan(scores_emulators).any() or np.isnan(skill_scores).any():
            print("Warning: NaNs found in the input data.")

        # Sort arrays for consistency in results
        skill_scores = np.sort(skill_scores, axis=0)
        scores_emulators = np.sort(scores_emulators, axis=0)
        scores_numerical = np.sort(scores_numerical, axis=0)

        # Use np.nanquantile to ignore NaNs in quantile computations
        skill_scores_mean = np.nanquantile(skill_scores, 0.5, axis=0, interpolation='linear')
        scores_mean_emulators = np.nanquantile(scores_emulators, 0.5, axis=0, interpolation='linear')
        scores_mean_numerical = np.nanquantile(scores_numerical, 0.5, axis=0, interpolation='linear')

        skill_scores_upper = np.nanquantile(skill_scores, 0.975, axis=0, interpolation='linear')
        skill_scores_lower = np.nanquantile(skill_scores, 0.025, axis=0, interpolation='linear')
        scores_numerical_upper = np.nanquantile(scores_numerical, 0.975, axis=0, interpolation='linear')
        scores_numerical_lower = np.nanquantile(scores_numerical, 0.025, axis=0, interpolation='linear')
        scores_emulators_upper = np.nanquantile(scores_emulators, 0.975, axis=0, interpolation='linear')
        scores_emulators_lower = np.nanquantile(scores_emulators, 0.025, axis=0, interpolation='linear')

        # Uncomment if using dispersion for further calculations
        # scores_upper_preds = scores_emulators_upper + 2 * np.nanmean(scores_dispersion, axis=0)
        # scores_lower_preds = scores_emulators_lower - 2 * np.nanmean(scores_dispersion, axis=0)

        return (scores_mean_numerical, scores_numerical_upper, scores_numerical_lower,
            scores_mean_emulators, scores_emulators_upper, scores_emulators_lower,
            skill_scores_mean, skill_scores_upper, skill_scores_lower)
    
    def assemble_scores(self, stations_dict):

        self.stations_dict = stations_dict
        statistics = ["scores_mean_numerical", "scores_numerical_upper", "scores_numerical_lower",
            "scores_mean_emulators", "scores_emulators_upper", "scores_emulators_lower",
            "skill_scores_mean", "skill_scores_upper", "skill_scores_lower"]
        self.scores_l1 = dict(zip(statistics, self.layer_statistics('layer0')))
        self.scores_l2 = dict(zip(statistics, self.layer_statistics('layer1')))
        self.scores_l3 = dict(zip(statistics, self.layer_statistics('layer2')))

    def assemble_forecasts(self, forecast_dict):

        self.forecast_dict = forecast_dict
        fcs = ["obs_mean", "obs_std", "num_mean", "num_std", "em_mean", "em_std"]
        self.fc_l1 = dict(zip(fcs, self.layer_forecast('layer0')))
        self.fc_l2 = dict(zip(fcs, self.layer_forecast('layer1')))
        self.fc_l3 = dict(zip(fcs, self.layer_forecast('layer2')))

    def layer_horizons(self, threshold):    
        
        h_skill_mean = np.argmax( (1 - self.scores_l1["skill_scores_mean"]) < 0)
        h_numerical_mean = np.argmax( (threshold - self.scores_l1["scores_mean_numerical"]) < 0)
        h_emulator_mean = np.argmax( (threshold - self.scores_l1["scores_mean_emulators"]) < 0)
        
    
    def plot_forecasts(self):

        doy_vector = self.doy_vector[:self.maximum_leadtime]
        step = len(doy_vector) // 4

        fig, ax = plt.subplots(3, 1, figsize=self.figsize, sharex=True, sharey=True) 

        ax[0].set_title("Surface Layer", **self.label_properties)
        ax[1].set_title("Subsurface Layer 1", **self.label_properties)
        ax[2].set_title("Subsurface Layer 2", **self.label_properties)

        ax[0].fill_between(doy_vector, 
                           self.fc_l1["obs_mean"] - 2*self.fc_l1["obs_std"],
                           self.fc_l1["obs_mean"] + 2*self.fc_l1["obs_std"], 
                           color = "green", alpha =0.3)
        ax[0].fill_between(doy_vector, 
                           self.fc_l1["num_mean"] - 2*self.fc_l1["num_std"],
                           self.fc_l1["num_mean"] + 2*self.fc_l1["num_std"], 
                           color = "cyan", alpha =0.3)
        ax[0].fill_between(doy_vector, 
                           self.fc_l1["em_mean"] - 2*self.fc_l1["em_std"],
                           self.fc_l1["em_mean"] + 2*self.fc_l1["em_std"], 
                           color = "purple", alpha =0.3)
        ax[0].set_ylabel(f"{self.ylabel}", **self.label_properties)
        ax[0].plot(doy_vector, self.fc_l1["obs_mean"], color = "lime", label = "SMOSMANIA")
        ax[0].plot(doy_vector, self.fc_l1["num_mean"], color = "cyan", label = "ECLand")
        ax[0].plot(doy_vector, self.fc_l1["em_mean"], color = "magenta", label = "AILand")

        ax[1].fill_between(doy_vector, 
                           self.fc_l2["obs_mean"] - 2*self.fc_l2["obs_std"],
                           self.fc_l2["obs_mean"] + 2*self.fc_l2["obs_std"], 
                           color = "green", alpha =0.3)
        ax[1].fill_between(doy_vector, 
                           self.fc_l2["num_mean"] - 2*self.fc_l2["num_std"],
                           self.fc_l2["num_mean"] + 2*self.fc_l2["num_std"], 
                           color = "cyan", alpha =0.3)
        ax[1].set_ylabel(f"{self.ylabel}", **self.label_properties)
        ax[1].fill_between(doy_vector, 
                           self.fc_l2["em_mean"] - 2*self.fc_l2["em_std"],
                           self.fc_l2["em_mean"] + 2*self.fc_l2["em_std"], 
                           color = "purple", alpha =0.3)
        ax[1].plot(doy_vector, self.fc_l2["obs_mean"], color = "lime", label = "SMOSMANIA")
        ax[1].plot(doy_vector, self.fc_l2["num_mean"], color = "cyan", label = "ECLand")
        ax[1].plot(doy_vector, self.fc_l2["em_mean"], color = "magenta", label = "AILand")

        ax[2].fill_between(doy_vector, 
                       self.fc_l3["obs_mean"] - 2*self.fc_l3["obs_std"],
                           self.fc_l3["obs_mean"] + 2*self.fc_l3["obs_std"], 
                           color = "green", alpha =0.3)
        ax[2].fill_between(doy_vector, 
                           self.fc_l3["num_mean"] - 2*self.fc_l3["num_std"],
                           self.fc_l3["num_mean"] + 2*self.fc_l3["num_std"], 
                           color = "cyan", alpha =0.3)
        ax[2].fill_between(doy_vector, 
                           self.fc_l3["em_mean"] - 2*self.fc_l3["em_std"],
                           self.fc_l3["em_mean"] + 2*self.fc_l3["em_std"], 
                           color = "purple", alpha =0.3)
        ax[2].plot(doy_vector, self.fc_l3["obs_mean"], color = "lime", label = "SMOSMANIA")
        ax[2].plot(doy_vector, self.fc_l3["num_mean"], color = "cyan", label = "ECLand")
        ax[2].plot(doy_vector, self.fc_l3["em_mean"], color = "magenta", label = "AILand")
        ax[2].set_ylabel(f"{self.ylabel}", **self.label_properties)
        ax[2].legend(prop=self.legend_properties, frameon=True)
        ax[2].set_xlabel(self.xlabel, **self.label_properties)

        for a in ax:
            
            tick_positions = doy_vector[::step]  # Adjust frequency as needed
            a.set_xticks(tick_positions)
            a.set_xticklabels([pd.Timestamp(t).strftime('%Y-%m-%d') for t in tick_positions], rotation=25)
            plt.setp(a.get_yticklabels(), **self.tick_properties)
            plt.setp(a.get_xticklabels(), **self.tick_properties)
   
        plt.tight_layout()
        fig_path = os.path.join(self.path_to_plots, f'{self.network_name}_{self.station}_{self.year}_{self.variable}_{self.evaluation}_fc.pdf')
        plt.savefig(fig_path)
        plt.show()

    def plot_scores(self):

        doy_vector = self.doy_vector[:self.maximum_leadtime]
        step = len(doy_vector) // 4

        fig, ax = plt.subplots(3, 1, figsize=self.figsize, sharex=True, sharey=True) 


        ax[0].set_title("Surface Layer", **self.label_properties)
        ax[1].set_title("Subsurface Layer 1", **self.label_properties)
        ax[2].set_title("Subsurface Layer 2", **self.label_properties)

        ax[0].fill_between(doy_vector, 
                           self.scores_l1["scores_emulators_lower"], 
                           self.scores_l1["scores_emulators_upper"], 
                           alpha = 0.3,
                           color = "magenta")
        #ax[0].fill_between(np.arange(len(self.scores_l1["scores_mean_emulators"])), 
        #                   self.scores_l1["scores_emulators_lower"], 
        #                   self.scores_l1["scores_emulators_upper"], 
        #                   alpha = 0.3,
        #                   color = "blue")
        ax[0].fill_between(doy_vector, 
                           self.scores_l1["scores_numerical_lower"], 
                           self.scores_l1["scores_numerical_upper"], 
                           alpha = 0.3,
                           color = "cyan")
        ax[0].plot(doy_vector, self.scores_l1["scores_mean_emulators"], color="magenta", label="Emulator")
        ax[0].plot(doy_vector, self.scores_l1["scores_mean_numerical"], color="cyan", label="ECLand")

        ax[1].fill_between(doy_vector, 
                           self.scores_l2["scores_emulators_lower"], 
                           self.scores_l2["scores_emulators_upper"], 
                           alpha = 0.3,
                           color = "magenta")
        ax[1].fill_between(doy_vector, 
                           self.scores_l2["scores_numerical_lower"], 
                           self.scores_l2["scores_numerical_upper"], 
                           alpha = 0.3,
                           color = "cyan")
        ax[1].plot(doy_vector, self.scores_l2["scores_mean_emulators"], color="magenta", label="Emulator")
        ax[1].plot(doy_vector, self.scores_l2["scores_mean_numerical"], color="cyan", label="ECLand")
        ax[1].legend(prop=self.legend_properties, frameon=True)

        ax[2].fill_between(doy_vector, 
                           self.scores_l3["scores_emulators_lower"], 
                           self.scores_l3["scores_emulators_upper"], 
                           alpha = 0.3,
                           color = "magenta")
        ax[2].fill_between(doy_vector, 
                           self.scores_l3["scores_numerical_lower"], 
                           self.scores_l3["scores_numerical_upper"], 
                           alpha = 0.3,
                           color = "cyan")
        ax[2].plot(doy_vector, self.scores_l3["scores_mean_emulators"], color="magenta", label="Emulator")
        ax[2].plot(doy_vector, self.scores_l3["scores_mean_numerical"], color="cyan", label="ECLand")
        #ax[2].legend(prop=self.legend_properties, frameon=True)
        ax[2].set_xlabel(self.xlabel, **self.label_properties)

        for a in ax:
            a.hlines(0, xmin = min(doy_vector), xmax=max(doy_vector), 
                     color = "black", alpha = 0.8, linestyle = '--',
                     linewidth = self.linewidth)
            a.set_ylabel(f"{self.ylabel}", **self.label_properties)
            tick_positions = doy_vector[::step]  # Adjust frequency as needed
            a.set_xticks(tick_positions)
            a.set_xticklabels([pd.Timestamp(t).strftime('%Y-%m-%d') for t in tick_positions], rotation=25)
            plt.setp(a.get_yticklabels(), **self.tick_properties)
            plt.setp(a.get_xticklabels(), **self.tick_properties)
   
        plt.tight_layout()
        fig_path = os.path.join(self.path_to_plots, f'{self.network_name}_{self.station}_{self.year}_{self.variable}_{self.evaluation}_scores.pdf')
        plt.savefig(fig_path)
        plt.show()

    def plot_horizons(self, threshold):

        doy_vector = self.doy_vector[:self.maximum_leadtime]
        step = len(doy_vector) // 4

        fig, ax = plt.subplots(3, 1, figsize=self.figsize, sharex=True, sharey=True) 

        ax[0].set_title("Surface Layer", **self.label_properties)
        ax[1].set_title("Subsurface Layer 1", **self.label_properties)
        ax[2].set_title("Subsurface Layer 2", **self.label_properties)

        ax[0].fill_between(doy_vector, 
                           threshold - self.scores_l1["scores_emulators_lower"], 
                           threshold - self.scores_l1["scores_emulators_upper"], 
                           alpha = 0.3,
                           color = "magenta")
        #ax[0].fill_between(np.arange(len(self.scores_l1["scores_mean_emulators"])), 
        #                   self.scores_l1["scores_emulators_lower"], 
        #                   self.scores_l1["scores_emulators_upper"], 
        #                   alpha = 0.3,
        #                   color = "blue")
        ax[0].fill_between(doy_vector, 
                           threshold - self.scores_l1["scores_numerical_lower"], 
                           threshold - self.scores_l1["scores_numerical_upper"], 
                           alpha = 0.3,
                           color = "cyan")
        ax[0].plot(doy_vector, threshold -self.scores_l1["scores_mean_emulators"], color="magenta", label="Emulator")
        ax[0].plot(doy_vector, threshold -self.scores_l1["scores_mean_numerical"], color="cyan", label="ECLand")

        ax[1].fill_between(doy_vector, 
                           threshold - self.scores_l2["scores_emulators_lower"], 
                           threshold - self.scores_l2["scores_emulators_upper"], 
                           alpha = 0.3,
                           color = "magenta")
        ax[1].fill_between(doy_vector, 
                           threshold - self.scores_l2["scores_numerical_lower"], 
                           threshold - self.scores_l2["scores_numerical_upper"], 
                           alpha = 0.3,
                           color = "cyan")
        ax[1].plot(doy_vector, threshold -self.scores_l2["scores_mean_emulators"], color="magenta", label="Emulator")
        ax[1].plot(doy_vector, threshold -self.scores_l2["scores_mean_numerical"], color="cyan", label="ECLand")
        ax[1].legend(prop=self.legend_properties, frameon=True)

        ax[2].fill_between(doy_vector, 
                           threshold - self.scores_l3["scores_emulators_lower"], 
                           threshold - self.scores_l3["scores_emulators_upper"], 
                           alpha = 0.3,
                           color = "magenta")
        ax[2].fill_between(doy_vector, 
                           threshold - self.scores_l3["scores_numerical_lower"], 
                           threshold - self.scores_l3["scores_numerical_upper"], 
                           alpha = 0.3,
                           color = "cyan")
        ax[2].plot(doy_vector, threshold - self.scores_l3["scores_mean_emulators"], color="magenta", label="Emulator")
        ax[2].plot(doy_vector, threshold - self.scores_l3["scores_mean_numerical"], color="cyan", label="ECLand")
        #ax[2].legend(prop=self.legend_properties, frameon=True)
        ax[2].set_xlabel(self.xlabel, **self.label_properties)

        for a in ax:
            a.hlines(0, xmin = min(doy_vector), xmax=max(doy_vector), 
                     color = "black", alpha = 0.8, linestyle = '--',
                     linewidth = self.linewidth)
            a.set_ylabel(f"$\\varrho-${self.score}", **self.label_properties)
            tick_positions = doy_vector[::step]  # Adjust frequency as needed
            a.set_xticks(tick_positions)
            a.set_xticklabels([pd.Timestamp(t).strftime('%Y-%m-%d') for t in tick_positions], rotation=25)
            plt.setp(a.get_yticklabels(), **self.tick_properties)
            plt.setp(a.get_xticklabels(), **self.tick_properties)
   
        plt.tight_layout()
        fig_path = os.path.join(self.path_to_plots, f'{self.network_name}_{self.station}_{self.year}_{self.variable}_{self.evaluation}_horizons.pdf')
        plt.savefig(fig_path)
        plt.show()

    def plot_skill_scores(self):

        doy_vector = self.doy_vector[:self.maximum_leadtime]
        step = len(doy_vector) // 4

        fig, ax = plt.subplots(3, 1, figsize=self.figsize, sharex=True, sharey=True) 

        ax[0].set_title("Surface Layer", **self.label_properties)
        ax[1].set_title("Subsurface Layer 1", **self.label_properties)
        ax[2].set_title("Subsurface Layer 2", **self.label_properties)

        ax[0].axhspan(0, 1, facecolor='lightgray', alpha=0.5)  
        ax[0].fill_between(doy_vector, 
                           1- self.scores_l1["skill_scores_upper"], 
                           1- self.scores_l1["skill_scores_lower"], 
                           alpha = 0.3,
                           color = "blue")
        ax[0].plot(doy_vector, 1- self.scores_l1["skill_scores_mean"], color="darkblue", label="SMOSMANIA")

        ax[1].axhspan(0, 1, facecolor='lightgray', alpha=0.5)  
        ax[1].fill_between(doy_vector, 
                           1-self.scores_l2["skill_scores_upper"], 
                           1-self.scores_l2["skill_scores_lower"], 
                           alpha = 0.3,
                           color = "blue")
        ax[1].plot(doy_vector, 1-self.scores_l2["skill_scores_mean"], color="darkblue", label="SMOSMANIA")
        #ax[1].legend(prop=self.legend_properties, frameon=True)

        ax[2].axhspan(0, 1, facecolor='lightgray', alpha=0.5)  
        ax[2].fill_between(doy_vector, 
                           1- self.scores_l3["skill_scores_upper"], 
                           1- self.scores_l3["skill_scores_lower"], 
                           alpha = 0.3,
                           color = "blue")
        ax[2].plot(doy_vector, 1-self.scores_l3["skill_scores_mean"], color="darkblue", label="SMOSMANIA")
        #ax[2].legend(prop=self.legend_properties, frameon=True)
        ax[2].set_xlabel(self.xlabel, **self.label_properties)

        for a in ax:
            a.hlines(0, xmin = min(doy_vector), xmax=max(doy_vector), 
                     color = "black", alpha = 0.8, linestyle = '--',
                     linewidth = self.linewidth)
            a.set_ylabel(f"{self.score}-SS", **self.label_properties)
            a.set_ylim(-1,1)
            tick_positions = doy_vector[::step]  # Adjust frequency as needed
            a.set_xticks(tick_positions)
            a.set_xticklabels([pd.Timestamp(t).strftime('%Y-%m-%d') for t in tick_positions], rotation=25)
            plt.setp(a.get_yticklabels(), **self.tick_properties)
            plt.setp(a.get_xticklabels(), **self.tick_properties)
   
        plt.tight_layout()
        fig_path = os.path.join(self.path_to_plots, f'{self.network_name}_{self.station}_{self.year}_{self.variable}_{self.evaluation}_skillscores.pdf')
        plt.savefig(fig_path)
        plt.show()