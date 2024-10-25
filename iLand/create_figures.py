import matplotlib.pyplot as plt
import numpy as np

from iLand.data import *
from iLand.evaluation_module import *
from iLand.visualisations import *
from iLand.helpers import *

dm = DataSets()
measurements, predictions_h100, predictions, measurements_subset = dm.get_data(baumarten_num = [1, 2, 3, 4, 5, 7])

species = 'piab'
visualisation_subset = predictions.query("species == 'piab' & site_index == 10")
visualisation_subset_m = measurements.query("species == 'piab' & dGz100 == 10")
ae_rhos= []
for ridx in visualisation_subset.rid.unique():
    piab_DM = DataModule(species=species,
                        stand_idx=ridx,
                        standard=1)

    piab_DM.process_data_subsets(measurements, predictions)
    piab_DM.create_reference()
    piab_DM.create_reference_standards()
    piab_data = piab_DM.get_results_dataframe()
    EM = EvaluationModule(piab_data)
    species_data = EM.get_extended_results()
    select_columns = ['ae_lower', 'ae_upper']
    ae_rhos.append(species_data[select_columns].mean(axis = 1, skipna=True ))

# Create some plots of the idealized measurements, the predictions at age 100 and predicted timeseries.
create_dominant_heights_correlation_plot(measurements_subset, predictions_h100,
                                         "iLand/plots/dominant_heights_correlation_h100.pdf")

create_site_index_boundaries_plot(measurements, predictions,
                                  rho_g= np.array(ae_rhos),
                                  site_index = 10,
                                  species='piab',
                                  save_to="iLand/plots/site_index_forecast.pdf")

create_boundaries_scheme_plot(measurements_subset,
                                  rid_value=2,
                                  output_file="iLand/plots/yield_class_boundaries_scheme.pdf")

create_idealized_measurements_timeseries_plot(measurements_subset, predictions_h100,
                                              output_file="iLand/plots/idealized_measurements_timeseries.pdf")

create_residuals_boxplots(predictions_h100, measurements_subset,
                          output_file="iLand/plots/residuals_age100_boxplots.pdf")

create_predicted_site_indices_plot(measurements_subset,
                                   predictions,
                                   predictions_h100,
                                   output_file="iLand/plots/predicted_site_indices.pdf")