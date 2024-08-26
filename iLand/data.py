import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Function to read data
def read_data():
    measurements = pd.read_csv("./data/site_index_dat.csv", sep=",")
    predictions_h100 = pd.read_csv("iLand/data/stadtwald_testing_results_h100.txt", sep="\s+")
    predictions = pd.read_csv("iLand/data/stadtwald_testing_results_SI_time_series.txt", sep="\s+")
    return measurements, predictions_h100, predictions

# Function to inspect data
def inspect_data(measurements, predictions_h100, predictions):
    print(measurements.describe())
    print(measurements.head())
    print(predictions.describe())
    print(predictions_h100.describe())
    print(predictions_h100['species'].value_counts())

# Function to add species full names
def add_species_fullname(predictions_h100):
    species_map = {
        "piab": "Picea \nabies",
        "abal": "Abies \nalba",
        "psme": "Pseudotsuga \nmenziesii",
        "pisy": "Pinus \nsylvestris",
        "lade": "Larix \ndecidua",
        "fasy": "Fagus \nsylvatica"
    }
    predictions_h100['species_fullname'] = predictions_h100['species'].map(species_map)
    return predictions_h100

# Function to subset and add species information to measurements
def process_measurements(measurements):
    baumarten_num = [1, 2, 3, 4, 5, 7]
    species_map_num_to_char = {
        1: "piab",
        2: "abal",
        3: "psme",
        4: "pisy",
        5: "lade",
        7: "fasy"
    }
    species_map_num_to_fullname = {
        1: "Picea \nabies",
        2: "Abies \nalba",
        3: "Pseudotsuga \nmenziesii",
        4: "Pinus \nsylvestris",
        5: "Larix \ndecidua",
        7: "Fagus \nsylvatica"
    }

    measurements = measurements[measurements['BArt'].isin(baumarten_num)]
    measurements = measurements.copy()
    measurements.loc[:, 'species'] = measurements['BArt'].map(species_map_num_to_char)
    measurements.loc[:, 'species_fullname'] = measurements['BArt'].map(species_map_num_to_fullname)

    return measurements

# Function to create and save plots
def create_and_save_plots(predictions_h100, measurements, output_dir="plots"):
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

# Main function to run the script
def get_data():

    measurements, predictions_h100, predictions = read_data()
    inspect_data(measurements, predictions_h100, predictions)
    predictions_h100 = add_species_fullname(predictions_h100)
    measurements = process_measurements(measurements)
    create_and_save_plots(predictions_h100, measurements)

    return measurements, predictions_h100, predictions

#if __name__ == "__main__":
#
#    measurements, predictions_h100, predictions = get_data()
