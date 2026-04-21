"""
code to calculate rain occurrence from MWR rain flag. 
The code also calculates rain occurrence for convective days and MOBL-T days separately, and stores the results in a netcdf file.
The ncdf file contains:
- rain occurrence for each time bin and each file
- rain counts for each time bin and each file
- total counts for each time bin and each file

""""

from fileinput import filename
import site
from turtle import mode

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER

from matplotlib.colors import BoundaryNorm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.projections.polar import PolarAxes
from readers.data_info import PLOT_SITES_NAMES, MWR_SITES_NAMES, site_lats, site_lons
from readers.MWR import read_MWR_flags, read_iwv_elev
from figures.plot_settings import VAR_DICT
import numpy as np
import pandas as pd
import re
import xarray as xr
from figures.utils import find_all_files_for_site, plot_teamx_sites, read_file_list_for_mode
from readers.data_info import orography_path, iop_conv_days, iop_MoBL_T_days, hours_diurnal_cycle_calc, azimuth_bins, fci_path, coords_file_path, domain
import os
import pdb
import progressbar

  

def main():

    for site in MWR_SITES_NAMES:
        print(f"Processing site: {site}")   
        site_instr = MWR_SITES_NAMES[site]

        # find all MWR files for the site
        mrr_path_root = f"/data/obs/campaigns/teamx/{site}/{site_instr}/actris/level1/2025/"
        mrr_string = "MWR_1C01_"
        file_ending = ".nc"

        # define time bins for rain occurrence calculation
        time_bins = pd.date_range(start="2000-01-01 00:00:00", end="2000-01-02 00:00:00", freq="2h")


        file_list, n_files = find_all_files_for_site(mrr_path_root, mrr_string, site, file_ending)
        print(f"Found {n_files} MWR files for site {site}")
        print(file_list)
        pdb.set_trace()

        # find files in file_list with dates from iop_conv_days and iop_MoBL_T_days
        file_list_conv = [file for file in file_list if any(day in file for day in iop_conv_days)]
        file_list_mobl_t = [file for file in file_list if any(day in file for day in iop_MoBL_T_days)]
        print(f"Found {len(file_list_conv)} files for convective days for site {site}")
        print(f"Found {len(file_list_mobl_t)} files for MOBL-T days for site {site}")

        # calculate rain occurrence for all files, convective days files, and MOBL-T days files
        calc_rain_occurrence(None, time_bins, n_files, file_list, site, "diurnal")
        calc_rain_occurrence(None, time_bins, len(file_list_conv), file_list_conv, site, "convective")
        calc_rain_occurrence(None, time_bins, len(file_list_mobl_t), file_list_mobl_t, site, "mobl_t")

def calc_rain_occurrence(ds_rain, time_bins, n_files, file_list, site, mode):
    # define matrix where to store rain occurrence for each time bin for all files
    rain_occ_diurnal = np.zeros((len(time_bins)-1, n_files))
    rain_counts = np.zeros((len(time_bins)-1, n_files))
    total_counts = np.zeros((len(time_bins)-1, n_files))


    # add progressbar for the loop over the files

    for ind_file, filename in progressbar.progressbar(enumerate(file_list), max_value=n_files):

        date_str = filename.split("/")[-1].split('.')[0][-8:] # extract the date from the filename
        ds_rain = read_MWR_flags(site, date_str)

        # redefine time 
        ds_rain = ds_rain.assign_coords(
            time=pd.to_datetime(ds_rain.time.dt.strftime("2000-01-01 %H:%M:%S").values)
        )
        
        # group by time bins and count how many time stamps we have rain and how many time stamps are there in the interval
        rain_time_stamps = ds_rain.groupby_bins("time", time_bins).sum("time") # count how many time stamps we have rain in each time bin
        total_time_stamps = ds_rain.groupby_bins("time", time_bins).count("time") # count how many time stamps we have in total in each time bin

        print(f"Rain time stamps for site {site}: {rain_time_stamps.rain.values}")
        print(f"Total time stamps for site {site}: {total_time_stamps.rain.values}")

        rain_occurrence = rain_time_stamps.rain.values / total_time_stamps.rain.values
        print(f"Rain occurrence for site {site}: {rain_occurrence}")

        # store rain occurrence, rain counts, and total counts for each time bin and each file in the respective matrices
        rain_occ_diurnal[:, ind_file] = rain_occurrence
        rain_counts[:, ind_file] = rain_time_stamps.rain.values
        total_counts[:, ind_file] = total_time_stamps.rain.values

    # store data in ncdf
    ds_out = xr.Dataset(
        data_vars={
            "rain_occ_diurnal": (("time_bin", "file"), rain_occ_diurnal),
            "rain_counts": (("time_bin", "file"), rain_counts),
            "total_counts": (("time_bin", "file"), total_counts)
        },
        coords={
            "time_bin": time_bins[:-1],
            "file": np.arange(n_files)
        }
    )
    output_path = f"data/mrr_ze_profiles/"
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, f"{site}_{mode}_rain_occurrence_v2.nc")
    ds_out.to_netcdf(output_file)
    print(f"Saved {mode} rain occurrence data for site {site} to {output_file}")

if __name__ == "__main__":
    main()  