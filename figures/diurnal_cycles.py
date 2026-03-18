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
from plotting import plot_diurnal_cycle_single_site, plot_hourly_percentiles
import numpy as np   
import pandas as pd
import xarray as xr
import pdb

def main():
    
    # select which group of days to plot the diurnal cycle for
    day_type =  "convective" # or  "MOBL_T"
    var_type = "lwp" # "lwp" or "iwv"

    # read list of days for the convective days and the MOBL thermal days
    if day_type == "convective":
        days = iop_conv_days
    elif day_type == "MOBL_T":
        days = iop_MoBL_T_days
    else:
        raise ValueError("day_type must be either 'convective' or 'MOBL_T'")
    

    # make a plot for each site
    for site in PLOT_SITES_NAMES.keys():

        ds_type = []
        # loop on days from the selected group of days
        for i, day in enumerate(days):

            day_string = day

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
            ds_type.append(ds_iwv)
        
        # plot diurnal cycle of iwv for the site and the selected days
        if var_type == "iwv":
            plot_diurnal_cycle_single_site(site, ds_type, day_type, var_type)  # (site, date, var, path_root)
        elif var_type == "lwp":
            plot_hourly_percentiles(site, ds_type, day_type, var_type)  # (site, date, var, path_root)

    
if __name__ == "__main__":
    main()