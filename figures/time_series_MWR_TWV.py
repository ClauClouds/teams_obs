"""
This code generates a plot of the time series of the three sites of MWR radiometer observations.
It then extracts the closest pixel to the location of the MWR from the FCI TWV and then plots the time series of the FCI TWV.
Both data time series with uncertainties are plotted together for comparison.

"""

import os
import pdb
import sys

import matplotlib.pyplot as plt
import xarray as xr

from process.read_and_resample_to_fci import read_and_resample_to_fci
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from readers.data_info import path_twv, path_MWR, site_names, site_lats, site_lons
from readers.MWR import read_lwp_iwv
from process.find_closest_pixel import find_closest_pixel
from figures.plotting import plot_time_series


def main():

    # date of the comparison
    date = "20250622"

    # select the closest pixel to the location of the MWR from the FCI TWV
    # reading fci twv for all the selected sites and then storing in list of datasets
    fci_twv = find_closest_pixel(path_twv, site_lats, site_lons, site_names)
    
    # read the MWR data for the three sites and collect them in a list of datasets
    mwr_twv = read_and_resample_to_fci(site_names, date, path_MWR)


    # plot the time series of the FCI TWV and MWR IWV for the three sites
    plot_time_series(fci_twv, mwr_twv, site_names, date)

if __name__ == "__main__":
    main()


