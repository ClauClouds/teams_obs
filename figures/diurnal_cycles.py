"""
codes to plot diurnal cycle of the IWV on convective days and MOBL thermal days for the three sites of the campaign. 
The diurnal cycle is plotted for each day with a superthin line transparency and then
 the mean and standard deviation of the IWV values among all days for each time stamp as
   a thicker line with shaded area for the standard deviation.

    
"""
from readers.data_info import PLOT_SITES_NAMES, MWR_SITES_NAMES, iop_conv_days, iop_MoBL_T_days
from readers.txt import read_txt_file
from readers.MWR import read_lwp_iwv
import matplotlib.pyplot as plt
from figures.plotting import plot_diurnal_cycle_single_site, plot_hourly_percentiles
import numpy as np   
import pandas as pd
import xarray as xr
import pdb
import progressbar
import os

from figures.utils import find_all_files_for_site

def main():
    
    # select which group of days to plot the diurnal cycle for
    day_type =  "all"# "MOBL_T"# "convective" # or  "MOBL_T"
    var_type = "lwp" # "lwp" or "iwv"

    for site in PLOT_SITES_NAMES.keys():

        # read list of days for the convective days and the MOBL thermal days or for all days
        if day_type == "convective":
            days = iop_conv_days
        elif day_type == "MOBL_T":
            days = iop_MoBL_T_days
        else:
            # find list of all LWP files for the site and extract the days from the filenames
            path_root = f"/data/obs/campaigns/teamx/{site}/{MWR_SITES_NAMES[site]}/actris/level2/2025/" #/data/campaigns/teamx/collalbo/kithat/actris/level2/2025/05/18
            filename_string = "MWR_single_"
            file_ending = ".nc"
            file_list, n_files = find_all_files_for_site(path_root, filename_string, site, file_ending)
            # /data/obs/campaigns/teamx/bolzano/hatpro/actris/level2/2025/05/24/MWR_single_bolzano_20250524.nc
            days = [file.split("/")[-1].split("_")[-1][:-3] for file in file_list]
            print(f"Found {n_files} files for site {site} and variable {var_type}. Extracted days: {days}")
            
        print(f"Processing site: {site}")

        ds_type = []
        days_list = []
        # loop on days from the selected group of days with progressbar evolving with the loop
        for day in progressbar.progressbar(days):

            day_string = day
            days_list.append(day_string)

            # identify path to the MWR data of the date
            path_file_sel = f"/data/obs/campaigns/teamx/{site}/{MWR_SITES_NAMES[site]}/actris/level2/{day_string[:4]}/{day_string[4:6]}/{day_string[6:8]}/"
            # read iwv data for the site and the selected days
            ds_iwv = read_lwp_iwv(site, day, var_type, path_file_sel)

            # add day and site as attributes to the dataset
            ds_iwv.attrs['day'] = day_string
            ds_iwv.attrs['site'] = site

            # Use a common reference date so all days can be interpolated on the same diurnal cycle.
            time_of_day = pd.to_datetime(ds_iwv.time.values).strftime('2000-01-01 %H:%M:%S')
            ds_iwv = ds_iwv.assign_coords(time=pd.to_datetime(time_of_day))

            # append dataset of the day to the type dataset
            ds_iwv = ds_iwv.drop_vars(["elevation", "azimuth"], errors="ignore")
            ds_type.append(ds_iwv)

        print(f"Finished processing site: {site}")

        # store ds_type for the site in a xarray dataset and save it as a netcdf file
        ds_type_all = xr.concat(ds_type, dim="days")
        ds_type_all.to_netcdf(f"data/{site}_{day_type}_{var_type}_diurnal_cycle.nc")
        print(f"Saved diurnal cycle dataset for site {site} and day type {day_type} and variable {var_type} as netcdf file.")

        # plot diurnal cycle of iwv for the site and the selected days
        #if var_type == "iwv":
        #    plot_diurnal_cycle_single_site(site, ds_type, day_type, var_type)  # (site, date, var, path_root)
        #elif var_type == "lwp":
        #    plot_hourly_percentiles(site, ds_type, day_type, var_type)  # (site, date, var, path_root)

    
if __name__ == "__main__":
    main()