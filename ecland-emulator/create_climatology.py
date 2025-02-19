import xarray as xr
import pandas as pd
import glob
import netCDF4 as nc

file_path = "/perm/dadf/HSAF_validation/in_situ_data/pre_processed_data/ismn_nc/soil_SMOSMANIA_ISMN_2022.nc"

file_pattern = "/perm/dadf/HSAF_validation/in_situ_data/pre_processed_data/ismn_nc/soil_SMOSMANIA_ISMN_*.nc" 

file_list = glob.glob(file_pattern)

create_combined = False

if create_combined:
# Open datasets one by one, ensuring they are closed properly
    datasets = []
    for file in file_list:
        try:
            print("Open file.")
            ds = xr.open_dataset(file)  # Open file
            datasets.append(ds)  # Store it
        except Exception as e:
            print(f"Skipping file {file} due to error: {e}")

    # Merge datasets if any are successfully opened
    if datasets:
        ds_combined = xr.concat(datasets, dim="time")
        print("Successfully combined datasets.")

        # Save to NetCDF or perform computations
        ds_combined.to_netcdf("combined_output.nc")

        # Close all datasets
        for ds in datasets:
            ds.close()
        ds_combined.close()
    else:
        print("No valid datasets found.")

ds = xr.open_dataset("combined_output.nc") 
ds = ds.sel(station_id = ['Condom', 'Villevielle', 'LaGrandCombe', 'Narbonne', 'Urgons',
                    'CabrieresdAvignon', 'Savenes', 'PeyrusseGrande','Sabres', 
                    'Mouthoumet','Mejannes-le-Clap',  'CreondArmagnac', 'SaintFelixdeLauragais',
                    'Mazan-Abbaye', 'LezignanCorbieres'], depth = [0.05, 0.2, 0.3])

print(ds)

ds = ds.sortby("time")
# Select only required variables
selected_vars = ["sm", "st"]
ds = ds[selected_vars]
ds["st"] = ds["st"] + 273.15 # convert to kelvin


ds["time"] = ds["time"].dt.floor("6h")
# Resample to 6-hour intervals, keeping NaN where no data exists
ds_resampled = ds.resample(time="6h", closed="left", label="left").mean()

# Extract day of year and hour, ensuring they are NumPy arrays
day_of_year = ds["time"].dt.dayofyear.data  # Convert to NumPy array
hour = ds["time"].dt.hour.data  # Convert to NumPy array

# Assign as new coordinates
ds = ds.assign_coords(day_of_year=("time", day_of_year), hour=("time", hour))

ds_agg = ds.groupby(["day_of_year", "hour"]).mean(dim="time")
ds_agg.to_netcdf("day_time_hourly_avg.nc")

ds_agg = ds.groupby(["day_of_year", "hour"]).std(dim="time")
ds_agg.to_netcdf("day_time_hourly_std.nc")

print(ds_agg)