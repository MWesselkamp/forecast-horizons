# Assuming prepare_data and visualisations are defined in separate Python files
import matplotlib.pyplot as plt
import numpy as np

from data import *
from evaluation_module import *
from visualisations import *
from helpers import *

dm = DataSets()
measurements, predictions_h100, predictions, measurements_subset = dm.get_data(baumarten_num = [1, 2, 3, 4, 5, 7])
def main(species, plot_idx, standard):
    piab_DM = DataModule(species=species,
                         stand_idx=plot_idx,
                         standard=standard)

    piab_DM.process_data_subsets(measurements, predictions)
    piab_DM.create_reference()
    piab_DM.create_reference_standards()
    piab_data = piab_DM.get_results_dataframe()

    EM = EvaluationModule(piab_data)
    EM.set_quantitative_standard()
    EM.get_horizon_trajectory()
    EM.test_horizon_trajectory()
    # piab_extended_results = EM.get_extended_results()

    return EM.get_aggregated_results()

species_names = predictions.species.unique()[:5] # ignore pisy
standard = 1
horizons_assemble_df = pd.DataFrame()

for species in species_names:

    horizons_species = pd.DataFrame()
    horizons_trajectories = []
    plot_indices = predictions.query(f"species == '{species}'").rid.unique()

    for plot_idx in plot_indices:

        piab_aggregated_results = main(species, plot_idx, standard)
        horizons_species = pd.concat([horizons_species, piab_aggregated_results])
        horizons_trajectories.append(piab_aggregated_results['ae_horizon_trajectory'])

    horizons_df = pd.DataFrame({
                'age': piab_aggregated_results.age.unique(),
                'species': species,
                'species_fullname': piab_aggregated_results.species_fullname.unique().item(),
                'h_means': np.array(horizons_trajectories).mean(axis= 0),
                'h_sd': np.array(horizons_trajectories).std(axis= 0)
                })

    horizons_assemble_df = pd.concat([horizons_assemble_df, horizons_df])

create_horizons_assembled_plot(horizons_assemble_df,
                               output_file="iLand/plots/iLand_horizons_assembled_trial.pdf")

