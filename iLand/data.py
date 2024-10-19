import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

class DataSets:

    def __init__(self):

        self.measurements, self.predictions_h100, self.predictions = self.read_data()
    def read_data(self):

        measurements = pd.read_csv("iLand/data/site_index_dat.csv", sep=",")
        predictions_h100 = pd.read_csv("iLand/data/stadtwald_testing_results_h100.txt", sep="\s+")
        predictions = pd.read_csv("iLand/data/stadtwald_testing_results_SI_time_series.txt", sep="\s+")

        return measurements, predictions_h100, predictions

    def inspect_data(self):

        print("Insepcting data.")
        print(self.measurements.describe())
        print(self.measurements.head())
        print(self.predictions.describe())
        print(self.predictions_h100.describe())
        print(self.predictions_h100['species'].value_counts())

    def add_species_fullname(self, data_frame):
        print("Add full species name to data_frame.")
        species_map = {
            "piab": "Picea \nabies",
            "abal": "Abies \nalba",
            "psme": "Pseudotsuga \nmenziesii",
            "pisy": "Pinus \nsylvestris",
            "lade": "Larix \ndecidua",
            "fasy": "Fagus \nsylvatica"
        }
        data_frame['species_fullname'] = data_frame['species'].map(species_map)
        return data_frame

    def _select_species_with_index(self, baumarten_num = [1, 2, 3, 4, 5, 7]):
        print("Select species with index.")
        self.measurements = self.measurements[self.measurements['BArt'].isin(baumarten_num)]
    def match_name_with_index(self, baumarten_num):
        print("Matching name with index.")
        species_map_num_to_char = {
            1: "piab",
            2: "abal",
            3: "psme",
            4: "pisy",
            5: "lade",
            7: "fasy"
        }

        self._select_species_with_index(baumarten_num)
        self.measurements.loc[:, 'species'] = self.measurements['BArt'].map(species_map_num_to_char)#

    def create_ideal_observations(self):
        print("Create a data frame of ideal observations that matches BWI site indices.")
        measurements_subset = pd.DataFrame()

        for stand_idx in range(len(self.predictions_h100)):
            measurements_stand_idx = self.measurements[
                                         (self.predictions_h100['species'][stand_idx] == self.measurements['species']) &
                                         (self.predictions_h100['site_index'][stand_idx] == self.measurements['dGz100'])
                                         ].iloc[:, [1, 2, 5, 16]]
            measurements_stand_idx['rid'] = self.predictions_h100['rid'][stand_idx]
            measurements_subset = pd.concat([measurements_subset, measurements_stand_idx], ignore_index=True)

        measurements_subset.to_csv("iLand/data/measurements_subset.csv", index=False)

        print("Species in data set: ", measurements_subset['species'].unique())
        print("Number of plots in data set: ", len(measurements_subset['rid'].unique()))

        return measurements_subset
    def get_data(self, baumarten_num = [1, 2, 3, 4, 5, 7]):

        self.inspect_data()
        self.match_name_with_index(baumarten_num)

        self.predictions_h100 = self.add_species_fullname(self.predictions_h100)
        self.measurements = self.add_species_fullname(self.measurements)
        self.measurements['new_index'] = self.measurements.apply(lambda row: f"{row['BArt']}_{row['dGz100']}", axis=1)

        # create_dominant_heights_boxplots(self.predictions_h100, self.measurements)
        self.measurements_subset = self.create_ideal_observations()

        self.measurements_subset = self.measurements_subset.sort_values(by='rid')
        self.predictions_h100 = self.predictions_h100.sort_values(by='rid')



        return self.measurements, self.predictions_h100, self.predictions, self.measurements_subset


class DataManipulator:
    def __init__(self, measurements, predictions, species):

        self.species = species

        self.measurements = self.species_subsets(measurements)
        self.predictions = self.species_subsets(predictions)

    def species_subsets(self, dataframe):
        query_string = f"species == '{self.species}'"
        return dataframe.query(query_string)
    def subset_measurements(self, min_age=40, max_age=115, new_index=None):

        query_string = f"species == '{self.species}' & Alter >= {min_age} & Alter <= {max_age}"
        if new_index is not None:
            query_string += f" & new_index == '{new_index}'"

        # make species dataframe subset a class attribute and save possible yield classes
        self.measurements_subset = self.measurements.query(query_string)
        self.site_indices = self.measurements_subset.dGz100.unique()

        return self.measurements_subset

    def subset_predictions(self, min_age=40, max_age=115, plot_index=None, site_index = None):

        query_string = f"species == '{self.species}' & age >= {min_age} & age <= {max_age}"
        if plot_index is not None:
            query_string += f" & rid == '{plot_index}'"
        if site_index is not None:
            dGz100_query = " | ".join([f"site_index == {val}" for val in site_index])
            query_string += f" & ({dGz100_query})"

        self.predictions_subset = self.predictions.query(query_string)

        return self.predictions_subset

    def select_measurement_site_index(self, measurements_subset, site_index):
        query_string = " | ".join([f"dGz100 == {val}" for val in site_index])
        return measurements_subset.query(query_string)

    def select_predictions_plot(self, predictions_subset, plot_index):
        query_string =  f"rid == {plot_index}"
        print(f"Selected species == {predictions_subset.query(query_string).species.unique()}")
        print(f"Plot index rid == {plot_index}")
        print(f"Site index == {predictions_subset.query(query_string).site_index.unique()}")

        return predictions_subset.query(query_string)
    def get_site_index(self, predictions_subset):
        return predictions_subset.site_index.unique()
    def filter_predictions_years(self, predictions_subset):
        predictions_subset_sparse = predictions_subset[predictions_subset['age'] % 5 == 0]
        return predictions_subset_sparse

class DataModule:

    def __init__(self, species='piab', stand_idx=2, standard=1):

        self.species = species
        self.stand_idx = stand_idx
        self.standard = standard

    def process_data_subsets(self, measurements, predictions):
        print("Process data with DataManipulator.")

        self.dm = DataManipulator(measurements, predictions, species=self.species)

        maximum_age = min(max(self.dm.measurements['Alter'].unique()), max(self.dm.predictions['age'].unique()))
        minimum_age = max(min(self.dm.measurements['Alter'].unique()), min(self.dm.predictions['age'].unique()))
        print("Minimum age: ", minimum_age)
        print("Maximum age: ", maximum_age)
        predictions_df = self.dm.subset_predictions(min_age=minimum_age,
                                               max_age=maximum_age)
        predictions_df_idx = self.dm.select_predictions_plot(predictions_df, plot_index=self.stand_idx)
        self.predictions_df_idx_sparse = self.dm.filter_predictions_years(predictions_df_idx)
        self.idx_site_index = self.dm.get_site_index(self.predictions_df_idx_sparse)

        self.measurement_df = self.dm.subset_measurements(min_age=minimum_age,
                                                max_age=maximum_age)
    def create_reference(self):
        print("Select reference measurement based on predicted site index.")
        self.reference_measurement = self.dm.select_measurement_site_index(self.measurement_df, site_index=self.idx_site_index)
    def create_reference_standards(self):
        print("Select reference standard based on upper and lower bounds, derived from predicted site index.")
        print("self.idx_site_index site index: ", self.idx_site_index[0] )
        lower_bound_site_index = max(self.idx_site_index[0] - self.standard, self.dm.site_indices.min())
        print("Lower bound site index: ", lower_bound_site_index)
        upper_bound_site_index = min(self.idx_site_index[0] + self.standard, self.dm.site_indices.max())
        print("Upper bound site index: ", upper_bound_site_index)
        self.reference_lower_bound = self.dm.select_measurement_site_index(self.measurement_df, site_index=[lower_bound_site_index])
        self.reference_upper_bound = self.dm.select_measurement_site_index(self.measurement_df, site_index=[upper_bound_site_index])
    def get_results_dataframe(self):

        result = pd.DataFrame({
            'species':self.species,
            'species_fullname': self.reference_measurement.species_fullname.unique()[0],
            'stand_idx': self.stand_idx,
            'age': self.predictions_df_idx_sparse['age'],
            'h0_predicted': self.predictions_df_idx_sparse['dominant_height'],
            'h0_ideal': self.reference_measurement['Ho'].values,
            'h0_ideal_upper': self.reference_upper_bound['Ho'].values,
            'h0_ideal_lower': self.reference_lower_bound['Ho'].values
        })

        return result
