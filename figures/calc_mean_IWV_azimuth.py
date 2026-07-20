"""
Code to calculate the mean IWV in a given azimuth direction for the entire campaign for each site, at 30 degrees elevation
Then, it plots a map of the area with the wind roses on showing the mean IWV in each azimuth direction for each site, 
and saves the figure in the plots folder. 
"""
import os
import pdb
import numpy as np
import progressbar 
import xarray as xr
import matplotlib.pyplot as plt
from figures.utils import find_all_files_for_site
from readers.MWR import read_iwv_elev
from readers.data_info import site_names, site_lats, site_lons, hours_diurnal_cycle_calc, azimuth_bins, MWR_SITES_NAMES, PLOT_SITES_NAMES
from readers.data_info import domain_expats
from readers.data_info import orography_path
from readers.data_info import iop_conv_days, iop_MoBL_T_days
from readers.data_info import path_pattern_classification, pattern_legend
from readers.data_info import domain_German_flood, domain_joyce, domain_ACTA, domain_TEAMX
from readers.data_info import pattern_legend
from readers.data_info import MWR_SITES_NAMES
from readers.data_info import PLOT_SITES_NAMES      
from readers.data_info import hours_diurnal_cycle_calc, azimuth_bins

def main():

    site_names = ["lagonero", "collalbo", "bolzano"]

    # loop over sites and calculate mean IWV in each azimuth bin for the entire campaign at 30 degrees elevation
    for site_name in site_names:

        print(f"Calculating mean IWV in azimuth bins for site {site_name} at 30 degrees elevation.")

        # load the MWR data for the site
        path_root = f"/data/obs/campaigns/teamx/{site_name}/{MWR_SITES_NAMES[site_name]}/actris/level2/"

        # find all files in the path_root directory and subdirectories
        mwr_string = "MWR_single"
        file_ending = ".nc"

        # read all filenames for the site
        file_list, n_files = find_all_files_for_site(path_root, mwr_string, site_name, file_ending)
        print(f"Found {n_files} MRR files for site {site_name}")

        # extra date from filename '/data/obs/campaigns/teamx/lagonero/tophat/actris/level2/2025/09/08/MWR_single_lagonero_20250908.nc',
        dates = [f.split("/")[-1].split("_")[-1].split(".")[0] for f in file_list]

        ds_mean_all = []
        # progressbar for loop over dates
        for date in progressbar.progressbar(dates, max_value=len(dates)):

            day_string = date
            elev_sel = 30
            yy = day_string[:4]
            mm = day_string[4:6]
            dd = day_string[6:8]
            path_file = path_root + f"{yy}/{mm}/{dd}/"
            # read the MWR data for the site, day and elevation
            ds_site = read_iwv_elev(site_name, day_string, 'iwv', elev_sel, path_file)
            pdb.set_trace()
            # calculate daily mean IWV in each azimuth bin
            ds_mean = ds_site.groupby_bins("azimuth_angle", azimuth_bins, labels=azimuth_bins[:-1]).mean(dim="time")   
            print(np.shape(ds_mean.iwv.values))
            ds_mean_all.append(ds_mean) 

        # concatenate the mean IWV for all days and calculate the overall mean for the campaign
        ds_mean_all_concat = xr.concat(ds_mean_all, dim="azimuth_bin")
        ds_mean_campaign = ds_mean_all_concat.groupby("azimuth_bin").mean(dim="azimuth_bin")      
        print(ds_mean_campaign)
        pdb.set_trace() 
if __name__ == "__main__":
    main()