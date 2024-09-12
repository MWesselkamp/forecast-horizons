import pandas as pd
def create_residuals_dataframe(spec_reference_dataframes, species_name, idx):
    """
        Create a DataFrame containing residuals between ideal and predicted values,
        along with additional metadata for a specific species and stand index.

        Parameters:
        - spec_reference_dataframes: DataFrame containing reference data with predictions and ideal values.
        - species_name: The name of the species (string) to be included in the result.
        - idx: The unique identifier for the stand to filter the data (rid).

        Returns:
        - residuals_df: A DataFrame with the calculated residuals and corresponding metadata.
    """
    # Filter the dataframe to include only the rows for the specified stand (rid == idx)
    filtered_df = spec_reference_dataframes.loc[spec_reference_dataframes['rid'] == idx]

    # Calculate residuals: difference between ideal and predicted values, including upper and lower bounds
    # Also, include species name, stand index, and the age of the stand for context
    residuals_df = pd.DataFrame({
        'resids': filtered_df['h0_ideal'].values - filtered_df['h0_predicted'].values,
        'resids_upper': filtered_df['h0_ideal_upper'].values - filtered_df['h0_predicted'].values,
        'resids_lower': filtered_df['h0_ideal_lower'].values - filtered_df['h0_predicted'].values,
        'species': species_name,
        'stand_idx': idx,
        'age': filtered_df['age'].values
    })

    return residuals_df


def find_horizon(df):
    results = []

    # Iterate over each species_fullname
    for species, group in df.groupby('species_fullname'):
        # Sort by age just in case
        group_sorted = group.sort_values(by='age')

        # Find the age where h_means < 0 for the first time, with fallback to 110 if not found
        mean_age = group_sorted.loc[group_sorted['h_means'] < 0, 'age'].min()
        mean_age = mean_age if pd.notna(mean_age) else 110

        # Find the age where h_means + h_sd < 0 for the first time, with fallback to 110 if not found
        plus_sd_age = group_sorted.loc[(group_sorted['h_means'] + group_sorted['h_sd']) < 0, 'age'].min()
        plus_sd_age = plus_sd_age if pd.notna(plus_sd_age) else 110

        # Find the age where h_means - h_sd < 0 for the first time, with fallback to 110 if not found
        minus_sd_age = group_sorted.loc[(group_sorted['h_means'] - group_sorted['h_sd']) < 0, 'age'].min()
        minus_sd_age = minus_sd_age if pd.notna(minus_sd_age) else 110

        # Append results as a dictionary
        results.append({
            'species_fullname': species,
            'mean_age': mean_age,
            'plus_sd_age': plus_sd_age,
            'minus_sd_age': minus_sd_age
        })

    # Convert results to a DataFrame
    result_df = pd.DataFrame(results)
    return result_df