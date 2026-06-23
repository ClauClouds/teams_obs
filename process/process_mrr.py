"""
This code should read the MRR files from IMPROtoo and from Metek and:
- remove interferences between 7 UTC and 15:30 UTC.
- understand why some rainy profiles do not show up in the quicklooks
- store corrected data in a ncdf file for publication and for further analysis.

input:
- MRR files from IMPROtoo and from Metek located in /data/campaigns/teamx/lagonero/mrr/l1/
"""
from datetime import time
import site
from turtle import mode

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from scipy.ndimage import label
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.projections.polar import PolarAxes
from readers.data_info import PLOT_SITES_NAMES, MWR_SITES_NAMES, site_lats, site_lons
from readers.MWR import read_iwv_elev
from figures.plot_settings import VAR_DICT
import numpy as np
import pandas as pd
import xarray as xr
from figures.utils import find_closest_dc_value, read_file_list_for_mode, calculate_mean_anomaly_for_time_selection, plot_mean_azimuth_ring, get_shared_colorbar_limits, get_regular_integer_colorbar_spec
from readers.data_info import orography_path, iop_conv_days, iop_MoBL_T_days, hours_diurnal_cycle_calc, azimuth_bins
import os
import pdb

def main():

    # define the sites and the path to the MRR data
    sites = ['lagonero', 'collalbo']
    site_selected = "lagonero"
    path_mrr = f"/data/campaigns/teamx/{site_selected}/mrr/l1/"

    # time stamps to plot to understand the interferences
    time_stamps = ["20250701T14:40:00","20250701T17:45:00"]
    date_selected = time_stamps[0][:8]

    # read the MRR data for the selected site and day and store in a dataset
    ds_mrr = read_mrr_data(path_mrr, site_selected, date_selected)

    # apply the function to filter interference to every time step in the dataset
    ds_mrr_filtered = filter_interference_in_mrr(ds_mrr)
    print(ds_mrr_filtered)
    pdb.set_trace()

    # plot radar reflectivity as a function of height for the entire day
    Ze = ds_mrr_filtered.Ze.values
    time = ds_mrr_filtered.time.values
    height = ds_mrr_filtered.range.values

    # select only ze values with interf_flag equal to False (coherent echo)
    Ze = np.where(ds_mrr_filtered.interf_flag.values == False, Ze, np.nan)

    plt.figure(figsize=(10, 6))
    plt.pcolormesh(time, height, Ze.T, shading='auto', cmap='viridis')
    plt.colorbar(label='Radar Reflectivity [dBZ]')
    plt.ylabel('Height [m]')
    plt.title('MRR Radar Reflectivity')
    plt.legend()

    # save figure
    plt.savefig(f"plots/mrr_reflectivity_no_int_{date_selected}.png")
    print("MRR data filtered for interference and vertical continuity. Plots saved.")

    # plot height spectrogram at selected time stamps
    for time_stamp in time_stamps:
        plot_mrr_spectrogram(ds_mrr_filtered.sel(time=time_stamp, method='nearest'))

def filter_interference_in_mrr(ds_mrr):
    """
    the function is filtering the interfence patterns in the Doppler spectra height spectrograms by applying 2 methods:
    1) it exploits connected component size: counts the number of connected 
    areas and then it compareded with a given threshold to decide 
    if the echo is fragmented (interference) when too many separated areas exist or coherent (precipitation) 
    when only a few separated areas in the Doppler height spectrogram exist
    2) it then check for vertical continuity of the ze profile using the function vertical_continuity_filter:
     if the Ze echo is fragmented but it shows vertical continuity, 
    it is likely to be a precipitation event and not an interference pattern. 
    In this case, the function does not filter the echo and it returns the original dataset. 
    If the echo is fragmented and it does not show vertical continuity, 
    it is likely to be an interference pattern and the function returns
     a dataset with eta equal to nan everywhere.

    The code then returns:
    - the original dataset without filtering if the echo is coherent
    - a dataset with eta equal to nan everywhere if the echo is fragmented (interference)   
    input:
    - ds_mrr: dataset containing the MRR data for the selected site and day
    output:
    - ds_mrr_filtered: dataset containing the MRR data for the selected site and day after filtering the interference
    """
    # define cable car starting and ending clock times
    start_hour = 7
    start_minute = 0
    end_hour = 15
    end_minute = 30

    # get the processed day from the dataset itself
    day_string = pd.to_datetime(ds_mrr.time.values[0]).strftime("%Y-%m-%d")

    # build full timestamps for that day
    cable_car_start = np.datetime64(f"{day_string}T{start_hour:02d}:{start_minute:02d}:00")
    cable_car_end = np.datetime64(f"{day_string}T{end_hour:02d}:{end_minute:02d}:00")

    print(f"Filtering interference in MRR data between {cable_car_start} and {cable_car_end}")

    # Keep the full day dataset and only modify the cable car time window.
    ds_mrr_filtered = ds_mrr.copy()
    ds_mrr_filtered = ds_mrr_filtered.assign(
        interf_flag=(("time", "range"), np.full((ds_mrr_filtered.sizes["time"], ds_mrr_filtered.sizes["range"]), False, dtype=bool))
    )

    ds_mrr_window = ds_mrr_filtered.sel(time=slice(cable_car_start, cable_car_end))

    # Apply the interference filter only inside the cable car time window.
    for time_value in ds_mrr_window.time.values:
        ds_mrr_sel = ds_mrr_filtered.sel(time=time_value)
        eta_dbz = ds_mrr_sel.eta.values
        eta_dbz_filtered, interf_flag = filter_interference(eta_dbz)

        ds_mrr_filtered["eta"].loc[dict(time=time_value)] = eta_dbz_filtered
        ds_mrr_filtered["interf_flag"].loc[dict(time=time_value)] = np.full(
            ds_mrr_filtered.sizes["range"],
            interf_flag,
            dtype=bool,
        )

    # apply vertical continuity filter to the Ze profiles in the dataset
    Ze = ds_mrr_filtered.Ze.values
    Ze_filtered, vertical_filter_flag = vertical_continuity_filter(Ze, 
                                                            ze_thr=-5, 
                                                            min_depth_bins=5, 
                                                            min_continuity=0.4, 
                                                            min_largest_fraction=0.5)
    
    ds_mrr_filtered.Ze.values = Ze_filtered  
    # update the filter flag in the dataset to indicate if the echo is fragmented (interference) or coherent (precipitation) after applying the vertical continuity filter
    ds_mrr_filtered.interf_flag.values = np.where(vertical_filter_flag, True, ds_mrr_filtered.interf_flag.values)
    return ds_mrr_filtered


def filter_interference(eta_dbz):
    """
    the function is filtering the interfence patterns in the Doppler spectra by exploiting
     connected component size: counts the number of connected areas and then it compared
    with a given threshold to decide if the echo is fragmented (interference) or coherent (precipitation)
    The code then returns:
    - the original dataset without filtering if the echo is coherent
    - a dataset with eta equal to nan everywhere if the echo is fragmented (interference)
    - a flag to indicate if the echo is fragmented or coherent

    input:
    - eta_dbz: array containing the profile of Doppler spectras in dBZ (doppler, height) etxracted at a 
    given time step from the MRR dataset
    output:
    - eta_dbz_filtered: array (doppler, height) containing the Doppler spectra in dBZ after filtering the interference
    - interf_flag: boolean flag indicating if the echo is fragmented (True) or coherent (False)
    """
    from scipy.ndimage import label

    # define the bottom of the mask for the interference patterns in the Doppler spectra
    mask = eta_dbz > 3
    
    # apply connected component labeling to the mask
    labels, nlab = label(mask)
    #print(f"Number of connected components: {nlab}")

    # set a threshold for the size of the connected components to be considered as interference patterns
    size_threshold = 5
    
    # initialize variable to store the size of the largest connected component
    largest_size = 0

    # check on the number of connected components
    if nlab > size_threshold:
        #print("fragmented echo", "largest connected component size: ", largest_size, "threshold: ", size_threshold)
        # return eta equal to nan everywhere 
        return np.full_like(eta_dbz, np.nan), True
    else:
        #print("coherent echo", "largest connected component size: ", largest_size, "threshold: ", size_threshold)
        # return the original dataset without filtering
        return eta_dbz, False



def vertical_continuity_filter(Ze, ze_thr=-5, min_depth_bins=5,
                               min_continuity=0.5,
                               min_largest_fraction=0.6):
    """
    This function applies a vertical continuity filter to the Ze profiles in the MRR dataset.
    The filter is based on the idea that precipitation echoes should show vertical continuity in the Ze profiles
    while interference patterns are often fragmented and do not show vertical continuity.
    The function identifies the largest connected component of Ze values above a given threshold and calculates the vertical
    continuity and the fraction of valid bins in the largest connected component. 
    If the largest connected component does not meet the criteria for vertical 
    continuity and fraction of valid bins, the function returns a dataset with Ze
     equal to nan everywhere. Otherwise, it keeps only the largest vertically 
     coherent part of the Ze profile and sets the rest to nan.

    Ze shape: (time, height)
    input:
    - Ze: array containing the Ze profiles in dBZ (time, height) extracted from the MRR dataset
    - ze_thr: threshold for Ze values to be considered as valid (default: -5 dBZ)

    - min_depth_bins: minimum number of bins in the largest connected component to
     be considered as a valid echo (default: 6 bins), increase it to make it more stringent, decrease it to make it less stringent
     (considering the MRR vertical resolution of 100 m, 6 bins correspond to a vertical extent of 600 m, which is a reasonable threshold for precipitation echoes)

    - min_continuity: minimum vertical continuity (fraction of valid bins in the largest connected component) 
    to be considered as a valid echo (default: 0.6), Measures how filled the vertical column is. Increase it to make it more stringent, decrease it to make it less stringent.
    
    - min_largest_fraction: minimum fraction of valid bins in the largest connected component 
    compared to all valid bins to be considered as a valid echo (default: 0.7), Increase it to make it more stringent (ex: 0.9 means
    at least 90% of all detected signal must belong to the single connected vertical structure),
     decrease it to make it less stringent.
    output:
    - Ze_out: array containing the Ze profiles in dBZ after applying the vertical continuity filter
    - a flag to indicate if the echo is fragmented or coherent

    """
    # copy the input Ze array to avoid modifying the original data
    Ze_out = Ze.copy()
    # create a flag of the same dimensions of Ze
    vertical_filter_flag = np.full(
        (Ze.shape[0], Ze.shape[1]),
        False,
        dtype=bool
    )
    # loop over the time dimension and apply the vertical continuity filter to each profile
    for it in range(Ze.shape[0]):
        # create a boolean array to identify valid Ze values above the threshold
        valid = np.isfinite(Ze[it, :]) & (Ze[it, :] > ze_thr)
        # if there are no valid Ze values, skip the profile
        if valid.sum() == 0:
            continue
        # apply connected component labeling to the valid Ze values
        labels, ncomp = label(valid)
        # calculate the size of each connected component
        sizes = np.array([
            np.sum(labels == i)
            for i in range(1, ncomp + 1)
        ])
        # identify the largest connected component
        largest_label = np.argmax(sizes) + 1
        largest = labels == largest_label
        # calculate the number of bins in the largest connected component
        inds = np.where(largest)[0]
        depth_bins = inds.max() - inds.min() + 1
        # calculate the vertical continuity and the fraction of valid bins in the largest connected component
        continuity = largest.sum() / depth_bins
        largest_fraction = largest.sum() / valid.sum()

        # reject fragmented profiles
        if (
            depth_bins < min_depth_bins or
            continuity < min_continuity or
            largest_fraction < min_largest_fraction
        ):
            Ze_out[it, :] = np.nan
            vertical_filter_flag[it, :] = True
        else:
            # keep only the largest vertically coherent part
            Ze_out[it, ~largest] = np.nan
            vertical_filter_flag[it, :] = False

    return Ze_out, vertical_filter_flag


def plot_mrr_spectrogram(ds_mrr):
    """
    code to plot the MRR height spectrogram for the selected time stamp
    input:
    - ds_mrr: dataset containing the MRR data for the selected site and time stamp
    output:
    - plot of the MRR spectrogram for the selected time stamp   
    """
    # extract the variables of interest from the dataset
    time = ds_mrr.time.values
    time_label = pd.to_datetime(time).strftime('%Y%m%dT%H%M%S')
    height = ds_mrr.height.values
    eta = ds_mrr.eta.values
    velocity = ds_mrr.velocity.values
    eta_noDA = ds_mrr.eta_noDA.values
    velocity_noDA = ds_mrr.velocity_noDA.values
    ze = ds_mrr.Ze.values
    ze_noDA = ds_mrr.Ze_noDA.values

    # plot the spectrogram
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(velocity, height, eta, shading='auto')
    plt.colorbar(label='Eta [dBZ]')
    plt.xlabel('Doppler velocity [m/s]')
    plt.ylabel('Height [m]')
    plt.title('MRR height Spectrogram')
    # save figure
    plt.savefig(f"plots/mrr_spectrogram_{time_label}.png")

    # plot the spectrogram
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(velocity_noDA, height, eta_noDA, shading='auto')
    plt.colorbar(label='Eta [dBZ]')
    plt.xlabel('Doppler velocity [m/s]')
    plt.ylabel('Height [m]')
    plt.title('MRR height Spectrogram')
    # save figure
    plt.savefig(f"plots/mrr_spectrogram_noDA_{time_label}.png")

    # plot radar reflectivity as a function of height
    # select only ze values with interf_flag equal to False (coherent echo)
    ze = np.where(ds_mrr.interf_flag.values == False, ze, np.nan)
    ze_noDA = np.where(ds_mrr.interf_flag.values == False, ze_noDA, np.nan)
    
    plt.figure(figsize=(10, 6))
    plt.plot(ze, height, marker='x', label='With DA')
    plt.plot(ze_noDA, height, marker='o', label='Without DA')
    plt.xlabel('Radar Reflectivity [dBZ]')
    plt.ylabel('Height [m]')
    plt.title('MRR Radar Reflectivity')
    plt.legend()
    # save figure
    plt.savefig(f"plots/mrr_reflectivity_{time_label}.png")



def read_mrr_data(path_mrr, site_selected, date_selected):
    """
    code to read the MRR data for the selected site and day and store in a dataset
    input:
    - path_mrr: path to the MRR data
    - site_selected: name of the selected site
    - date_selected: date of the selected data
    output:
    - ds_mrr: dataset containing the MRR data for the selected site

    dependencies:
    - find_file_mrr: function to find the file of the day and site selected
    - filter_interference: function to filter interference in the MRR data between 7 UTC and 15:30 UTC
    """
    # find the file of the day and site selected
    file_mrr = find_file_mrr(path_mrr, site_selected, date_selected)

    # unzip file it format ending is gz
    if file_mrr.endswith('nc.gz'):
        import gzip
        import shutil
        # extract only filename from the file path
        filename = file_mrr.split("/")[-1]

        # define the path for the unzipped file
        unzipped_file_path = f"{filename[:-3]}"  # Remove the .gz extension
        
        # unzip the file
        with gzip.open(file_mrr, 'rb') as f_in:
            with open(unzipped_file_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        file_mrr = unzipped_file_path

    # read the NetCDF file using xarray
    ds_mrr = xr.open_dataset(file_mrr)

    # remove the unzipped file to save space
    if os.path.exists(file_mrr) and file_mrr.endswith('.nc'):
        os.remove(file_mrr) 
        
    return ds_mrr

def find_file_mrr(path_mrr, site_selected, date_selected):
    """
    code to find the file of the day and site selected
    input:  
    - path_mrr: path to the MRR data
    - site_selected: name of the selected site
    - date_selected: date of the selected data
    output:
    - file_mrr: path to the MRR file for the selected site and day
    """
    if site_selected == "lagonero":
        site_selected = "lago"


    # build path with the date
    path_mrr = os.path.join(path_mrr, date_selected[:4], date_selected[4:6], date_selected[6:8])
    # list all files in the path
    files = os.listdir(path_mrr)

    # find the file of the day and site selected
    for file in files:
        if "improtoo" in file:
            if file.endswith("nc.gz"):
                print("Found file: ", file)
                file_mrr = os.path.join(path_mrr, file)
                return file_mrr

    raise FileNotFoundError(f"No MRR file found for site {site_selected} and date {date_selected} in path {path_mrr}")




if __name__ == "__main__":
    main()

