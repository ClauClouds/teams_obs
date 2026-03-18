"""
code to calculate mean IWV diurnal cycle over the entire campaign.
Then, for every day of one of the categories 'convective' or 'mobl-t', 
it calculates the anomaly compared to the mean diurnal cycle.
It then derives the mean anomaly diurnal cycle for the convective days 
and for the MOBL-T days and plots them together in one plot.
"""
from readers.data_info import PLOT_SITES_NAMES, MWR_SITES_NAMES, iop_conv_days, iop_MoBL_T_days
from readers.MWR import read_lwp_iwv, read_all_data_for_campaign
import matplotlib.pyplot as plt
from plotting import plot_diurnal_cycle_single_site, plot_hourly_percentiles, plot_mean_dc
import numpy as np   
import pandas as pd
import xarray as xr
import pdb

def main():

    # select which group of days to plot the diurnal cycle for
    day_type =  "convective" # or  "MOBL_T"
    var_type = "iwv" # "lwp" or "iwv"


    # read list of days for the convective days and the MOBL thermal days
    if day_type == "convective":
        days = iop_conv_days
    elif day_type == "MOBL_T":
        days = iop_MoBL_T_days
    else:
        raise ValueError("day_type must be either 'convective' or 'MOBL_T'")
    

    # select site for the plot
    site = "bolzano" # specify site to plot ("bolzano", "collalbo", or "lagonero")

    # derive matrix containing data
    var_matrix, time_dc = read_all_data_for_campaign(site, var_type)
    print(var_matrix)

    # calculate mean and standard deviation for each time step of the diurnal cycle
    ds_mean_dc = np.nanmean(var_matrix, axis=0)
    ds_std_dc = np.nanstd(var_matrix, axis=0)

    # plot diurnal cycle time series
    plot_mean_dc(site, time_dc, ds_mean_dc, ds_std_dc, var_type)

    # calculate anomalies for each day of the selected type (convective or MOBL-T) compared to the mean diurnal cycle
    anomalies = var_matrix - ds_mean_dc

if __name__ == "__main__":
    main()