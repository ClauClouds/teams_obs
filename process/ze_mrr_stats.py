""""
code to calculate, for the two sites, the mean ze profile for the entire campaign, 
 for convective days and for MOBL-T days. 

"""



from datetime import date
import site
from turtle import mode
from unittest import skip

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.projections.polar import PolarAxes
from readers.MRR import read_MRR
from readers.data_info import PLOT_SITES_NAMES, MWR_SITES_NAMES, site_lats, site_lons
from readers.MWR import read_MWR_flags, read_iwv_elev
from figures.plot_settings import VAR_DICT
import numpy as np
import pandas as pd
import xarray as xr
from figures.utils import calc_iwv_deviation, create_site_inset, find_all_files_for_site, plot_iwv_ring_on_map, read_file_list_for_mode
from readers.data_info import orography_path, iop_conv_days, iop_MoBL_T_days, hours_diurnal_cycle_calc, azimuth_bins
import os
import pdb
import progressbar


def main():

    plotting = False # set to True to make a plot for each file with 2 subplots: one with the ze time height and one with the mean ze profile
    # Anchor time bins to a common reference date so they can be compared with the
    # time-of-day coordinates assigned below.
    time_bins = pd.to_datetime(
        [f"2000-01-01 {hour}" for hour in hours_diurnal_cycle_calc] + ["2000-01-02 00:00"]
    )

    sites = MWR_SITES_NAMES
    # remove bolzano from the sites list because there are no MRR data for this site
    sites = {key: value for key, value in sites.items() if key != "bolzano"}

    # loop on the sites
    for site_name in sites:

        print(f"Processing site: {site_name}")

         # MRR path root
        mrr_path_root = f"/data/obs/campaigns/teamx/{site_name}/mrr/l1/2025/"
        mrr_string = "mrr_improtoo_"
        file_ending = ".nc.gz"

        # read all filenames for the site
        file_list, n_files = find_all_files_for_site(mrr_path_root, mrr_string, site_name, file_ending)
        print(f"Found {n_files} MRR files for site {site_name}")

        # find files in file_list with dates from iop_conv_days and iop_MoBL_T_days
        file_list_conv = [file for file in file_list if any(day in file for day in iop_conv_days)]
        file_list_mobl_t = [file for file in file_list if any(day in file for day in iop_MoBL_T_days)]
        print(f"Found {len(file_list_conv)} files for convective days for site {site_name}")
        print(f"Found {len(file_list_mobl_t)} files for MOBL-T days for site {site_name}")

        # call functions to calculate ze profiles for all files, convective days files, and MOBL-T days files
        ds_ze_all = calc_ze_profiles(file_list, time_bins, site_name, "all", plotting=plotting)
        print(f"Calculated mean Ze profile for all days for site {site_name}")
        ds_ze_convective = calc_ze_profiles(file_list_conv, time_bins, site_name, "convective", plotting=plotting)
        print(f"Calculated mean Ze profile for convective days for site {site_name}")
        ds_ze_mobl_t = calc_ze_profiles(file_list_mobl_t, time_bins, site_name, "mobl_t", plotting=plotting)
        print(f"Calculated mean Ze profile for MOBL-T days for site {site_name}")

        print(f"Finished processing site: {site_name}")


def calc_ze_profiles(file_list, time_bins, site_name, mode, plotting=False):
    
    # loop on the files found with progressbar 
    ze_mean = np.zeros((len(file_list), 50)) # initialize array to store ze profiles for all files, with 31 being the number of height levels in the MRR data
    ze_interval_mean = np.zeros((len(file_list), len(hours_diurnal_cycle_calc), 50)) # initialize array to store ze profiles for all files, with 31 being the number of height levels in the MRR data
    ind_file = 0 # initialize index to keep track of the file number for storing ze profiles in the ze_mean array
    for file in progressbar.progressbar(file_list):

        print(f"Processing file: {file}")

        # extract date from the filename of the form 20250830_teamx_coll_mrr_raw.txt.gz
        date_str = file.split("_")[0].split("/")[-1]

        # if date is 20250516, then skip it
        if date_str == "20250516":
            print(f"Skipping file {file} due to known issues with MRR data on this day.")
            continue

        #date = date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))
        # read the MRR data
        ds_mrr = read_MRR(file, True, None)
        if ds_mrr is None:
            print(f"Skipping file {file} due to issues with MRR data that prevent interpolation on a common height grid.")
            continue
        # read rain flag from MWR radiometer data  
        ds_rain = read_MWR_flags(site_name, date_str)
        if ds_rain is None:
            print(f"Skipping file {file} due to missing MWR rain flag data.")
            continue
        # resmaple ds_rain on the time resolution of ds_mrr
        ds_rain_reindex = ds_rain.reindex(time=ds_mrr.time, method="nearest", tolerance="60s")
        # merge the two datasets on the time dimension
        ds_merged = xr.merge([ds_mrr, ds_rain_reindex])

        # drop time stamps that do not exactly match between the two datasets (i.e., where rain flag is NaN because there was no matching time stamp in the rain dataset)
        ds_merged = ds_merged.dropna(dim="time", subset=["rain"])

        # select only the data points where the rain flag is 1 ß(indicating rain)
        ind_rainy = np.where(ds_merged.rain == 1)[0]
        ds_rainy = ds_merged.isel(time=ind_rainy, drop=True)

        if len(ds_rainy.time) == 0:
            print(f"No rainy time steps found in file {file}. Skipping to next file.")
            continue

        # Replace the original date with a common reference date while keeping
        # the time of day, so all rainy profiles can be grouped into the same
        # diurnal-cycle bins.
        ds_rainy = ds_rainy.assign_coords(
            time=pd.to_datetime(ds_rainy.time.dt.strftime("2000-01-01 %H:%M:%S").values)
        )
        
        # calculate the mean ze profile for the rainy time steps and store it in the ze_mean array
        ze_mean[ind_file, :] = ds_rainy.Ze.mean(dim="time", skipna=True).values
        ze_interval_mean[ind_file, :, :] = ds_rainy.groupby_bins("time", time_bins).mean(dim="time", skipna=True).Ze.values

        # make a plot for each file with 2 subplots: one with the ze time height and one with the mean ze profile
        if plotting:
            plt.figure(figsize=(12, 6))
            # share y axis between the two subplots
            plt.subplot(1, 2, 1).sharey(plt.subplot(1, 2, 2))
            plt.subplot(1, 2, 1)
            plt.pcolormesh(ds_rainy.time, 
                        ds_rainy.height, 
                        ds_rainy.Ze.T, 
                        cmap="viridis", vmin=-10, vmax=30)
            plt.title(f"Ze time-height for file {file}")
            plt.subplot(1, 2, 2)
            plt.plot(ds_rainy.Ze.mean(dim="time", skipna=True), ds_rainy.height, color="blue")
            plt.title(f"Mean Ze profile for file {file}")

            plt.tight_layout()
            plt.savefig(f"ze_profiles_{site_name}_{date_str}_{mode}.png")
            plt.close()

            pdb.set_trace()

        ind_file += 1

    # calculate mean over the number of days


    ze_mean_profile = np.nanmean(ze_mean, axis=0)
    ze_std = np.nanstd(ze_mean, axis=0)
    ze_hourly_mean = np.nanmean(ze_interval_mean, axis=0)
    ze_hourly_std = np.nanstd(ze_interval_mean, axis=0)

    # plot mean profile
    plt.figure()
    plt.plot(ze_mean_profile, 
             ds_rainy.height,
               color="blue")
    plt.fill_betweenx(ds_rainy.height, 
                      ze_mean_profile - ze_std, 
                      ze_mean_profile + ze_std, 
                      color="blue", alpha=0.3)
    plt.title(f"Mean Ze profile for site {site_name} and mode {mode}")
    plt.xlabel("Ze (dBZ)")
    plt.ylabel("Height (m)")
    plt.savefig(f"mean_ze_profile_{site_name}_{mode}.png")
    plt.close()

    # plot all hourly mean profiles in one plot with colors assigned based on time of day
    plt.figure(figsize=(8, 6))
    for i in range(len(hours_diurnal_cycle_calc)):
        plt.plot(ze_hourly_mean[i, :], 
                 ds_rainy.height, 
                 color=plt.cm.viridis(i / len(hours_diurnal_cycle_calc)), 
                 label=f"{hours_diurnal_cycle_calc[i]}:00")
    plt.title(f"Mean Ze profiles for different times of day for site {site_name} and mode {mode}")
    plt.xlabel("Ze (dBZ)")
    plt.ylabel("Height (m)")        
    plt.legend(title="Time of day")
    plt.savefig(f"mean_ze_profiles_by_time_{site_name}_{mode}.png")
    plt.close() 
    pdb.set_trace()

    # store ze profiles in a dataset and save it as a netcdf file
    ds_output = xr.Dataset(
        data_vars={
            "ze_mean": (["height"], ze_mean_profile),
            "ze_hourly_mean": (["time_bin", "height"], ze_hourly_mean), 
            "ze_std": (["height"], ze_std),
            "ze_hourly_std": (["time_bin", "height"], ze_hourly_std)
        },
        coords={
            "height": (["height"], ds_rainy.height.values),
            "time_bin": time_bins[:-1]  # Exclude the last bin edge for labeling
        }
    )   

    output_path = f"/home/cacquist/Documents/GitHub/EXPATS/teams_obs/data/mrr_ze_profiles/"
    os.makedirs(output_path, exist_ok=True)
    output_file = f"{output_path}mrr_ze_profiles_{site_name}_{mode}.nc"
    ds_output.to_netcdf(output_file)

    print(f"Saved mean Ze profiles for site {site_name} to {output_file}")
    return ds_output
        
if __name__ == "__main__":
    main()
