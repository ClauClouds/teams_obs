"""
code to calculate spatial distribution of clouds based on the IR brightness temperature (BT) measured by the FCI instrument on board the MTG satellite
and, during daytime, on the visible reflectance (VIS) measured by the same instrument. 
The code calculates the mean BT and VIS for each time interval from hours_diurnal_cycle_calc of the day and for each pixel of the FCI grid, 
then plots the mean diurnal cycle of BT and VIS for each pixel in a map.
"""

from fileinput import filename
import site
from turtle import mode

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.projections.polar import PolarAxes
from readers.data_info import PLOT_SITES_NAMES, MWR_SITES_NAMES, site_lats, site_lons
from readers.MWR import read_iwv_elev
from figures.plot_settings import VAR_DICT
import numpy as np
import pandas as pd
import re
import xarray as xr
from figures.utils import find_all_files_for_site
from readers.data_info import orography_path, iop_conv_days, iop_MoBL_T_days, hours_diurnal_cycle_calc, azimuth_bins, fci_path
import os
import pdb


def read_fci_dataset(file_list, var):
    """Open and combine FCI files without requiring dask.
    Parameters:
    file_list (list): List of file paths to the FCI data files.
    var (str): Variable to extract from the FCI files, e.g., "BT
    Returns:
    xarray.Dataset: Combined dataset containing the specified variable from all FCI files.
    time_dc (numpy array): Array of datetime values corresponding to the time steps of the diurnal cycle.
    """

    # add bar to show progress of reading files using bar from progress package
    from progress.bar import Bar
    bar = Bar('Reading FCI files', max=len(file_list))

    first_ds = xr.open_dataset(file_list[0])
    if var not in first_ds:
        first_ds.close()
        raise ValueError(f"Variable {var} not found in {file_list[0]}.")

    if first_ds[var].ndim != 3:
        first_ds.close()
        raise ValueError(
            f"Variable {var} in {file_list[0]} has shape {first_ds[var].shape}; expected 3 dimensions (time, y, x)."
        )

    n_time_per_file, dim_y, dim_x = first_ds[var].shape
    first_ds.close()

    # define empty matrix to store the data at each time step
    var_matrix = np.empty((len(file_list) * n_time_per_file, dim_y, dim_x))

    # define time array of 10 min resolution
    time_dc = np.empty(len(file_list) * n_time_per_file, dtype='datetime64[ns]')

    for i, file_path in enumerate(file_list):
        print(f"Reading file {file_path}...")

        ds = xr.open_dataset(file_path)
        
        # extract time from the file name and convert to datetime format
        basename = os.path.basename(file_path)
        time_match = re.search(r"(\d{8})", basename)
        if time_match is None:
            ds.close()
            raise ValueError(f"Could not extract an 8-digit date from filename {basename}.")

        time_str = time_match.group(1)
        day_start = np.datetime64(f"{time_str[:4]}-{time_str[4:6]}-{time_str[6:8]}T00:00:00")
        time_dc[i*n_time_per_file:(i+1)*n_time_per_file] = np.array([
            day_start + np.timedelta64(10 * j, 'm') for j in range(n_time_per_file)
        ])
        
        # extract variable of interest (BT or VIS) and store in the matrix
        var_values = ds[var].values
        if var_values.shape != (n_time_per_file, dim_y, dim_x):
            ds.close()
            raise ValueError(
                f"Variable {var} in {file_path} has shape {var_values.shape}; expected {(n_time_per_file, dim_y, dim_x)}."
            )

        var_matrix[i*n_time_per_file:(i+1)*n_time_per_file, :, :] = var_values
        ds.close()  # close the dataset after reading to free up memory
        bar.next()
    bar.finish()
    return var_matrix, time_dc


def main():

    # read all IR FCI data for the campaign and calculate mean diurnal cycle of BT and VIS for each pixel of the FCI grid
    mode = "ir_105" # or "VIS_06"
    file_list, n_files = find_all_files_for_site(fci_path, mode, "expats")
    if n_files == 0:
        raise ValueError(f"No FCI files found for mode {mode} under {fci_path}.")

    file_list = sorted(file_list)
    print(file_list)

    # read all files in one xarray dataset without relying on dask
    var_matrix, time_dc = read_fci_dataset(file_list, mode)
    print(f"Shape of the combined FCI dataset: {var_matrix.shape}")

    pdb.set_trace()

if __name__ == "__main__":
    main()
