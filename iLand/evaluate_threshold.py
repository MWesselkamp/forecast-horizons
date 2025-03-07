# Assuming prepare_data and visualisations are defined in separate Python files
import matplotlib.pyplot as plt
import numpy as np

from data import *
from evaluation_module import *
from visualisations import *
from helpers import *

dm = DataSets()
measurements, predictions_h100, predictions, measurements_subset = dm.get_data(baumarten_num = [1, 2, 3, 4, 5, 7])
def main(species, plot_idx, threshold):

    piab_DM = DataModule(species=species,
                         stand_idx=plot_idx,
                         standard=1)

    piab_DM.process_data_subsets(measurements, predictions)
    piab_DM.create_reference()
    piab_DM.create_reference_standards()
    piab_data = piab_DM.get_results_dataframe()

    EM = EvaluationModule(piab_data)
    ae_horizon_trajectory = threshold - EM.ae_expect
    return piab_data, ae_horizon_trajectory

species_names = predictions.species.unique()[:5] # ignore pisy


results = pd.DataFrame()
thresholds = np.linspace(0, 4, 20)
for threshold in thresholds:
    horizons_assemble_df = pd.DataFrame()
    for species in species_names:

        horizons_trajectories = []
        plot_indices = predictions.query(f"species == '{species}'").rid.unique()

        for plot_idx in plot_indices:

            dataset, ae_horizon_trajectory = main(species, plot_idx, threshold=threshold)
            horizons_trajectories.append(ae_horizon_trajectory)

        horizons_df = pd.DataFrame({
                    'age': dataset.age.unique(),
                    'species': species,
                    'species_fullname': dataset.species_fullname.unique().item(),
                    'h_means': np.array(horizons_trajectories).mean(axis= 0),
                    'h_sd': np.array(horizons_trajectories).std(axis= 0)
                    })

        horizons_assemble_df = pd.concat([horizons_assemble_df, horizons_df])

    results = pd.concat([results, find_horizon(horizons_assemble_df)])

plot_age_limit_by_species_mulitples(results,
                                    thresholds= thresholds,
                                    output_file="iLand/plots/iLand_threshold_horizons.pdf")
