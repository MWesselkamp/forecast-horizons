import pandas as pd
def find_horizon(df):
    results = []

    for species, group in df.groupby('species_fullname'):

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

        results.append({
            'species_fullname': species,
            'mean_age': mean_age,
            'plus_sd_age': plus_sd_age,
            'minus_sd_age': minus_sd_age
        })

    result_df = pd.DataFrame(results)
    return result_df