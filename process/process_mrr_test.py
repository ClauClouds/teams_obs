"""
This code should read the MRR files from IMPROtoo and from Metek and:
- remove interferences between 7 UTC and 15:30 UTC.
- understand why some rainy profiles do not show up in the quicklooks
- store corrected data in a ncdf file for publication and for further analysis.

input:
- MRR files from IMPROtoo and from Metek located in /data/campaigns/teamx/lagonero/mrr/l1/

which envitronment to use:
- source .teams_venv/bin/activate
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
from readers.MWR import read_MWR_flags

def main():

    # set in input which filters to remove we want to apply:
    filter_RR_on = True  # if True, apply the rain filtering based on the MWR flags
    filter_spectra_connectivity_on = True  # if True, apply the filtering based on the connected component size in the Doppler spectra
    filter_Ze_vertical_continuity_on = False  # if True, apply the filtering based on the vertical continuity of the Ze profiles
    filter_horizontal_band_filter_on = False

    # define the sites and the path to the MRR data and MWR data
    sites = ['lagonero', 'collalbo']
    site_selected = "lagonero"
    path_mrr = f"/data/campaigns/teamx/{site_selected}/mrr/l1/"

    # time stamps to plot to understand the interferences
    time_stamps = ["20250705T12:45:00","20250705T12:50:00"]
    date_selected = time_stamps[0][:8]

    # read the MRR data for the selected site and day and store in a dataset
    ds_mrr = read_mrr_data(path_mrr, site_selected, date_selected)

    # Convert the MRR height field from time-range to a fixed vertical coordinate.
    # This keeps Ze as a clean time x range matrix and prevents height from
    # being treated as a profile variable during rain masking.
    if "height" in ds_mrr and ds_mrr["height"].dims == ("time", "range"):
        height_1d = ds_mrr["height"].median(dim="time", skipna=True)
        ds_mrr = ds_mrr.drop_vars("height").assign_coords(height=("range", height_1d.values))

    os.makedirs("plots", exist_ok=True)

    # if MWR flag file exists, then select only the time stamps where rain flag is true
    if find_MRR_flag(site_selected, date_selected) and filter_RR_on: 

        # read the MWR flags for the selected site and day and store in a dataset
        ds_mwr = read_MWR_flags(site_selected, date_selected)

        # Align categorical MWR flags to the MRR timestamps without numeric interpolation.
        ds_mwr_interp = ds_mwr.reindex(time=ds_mrr.time, method='nearest')

        # create a boolean rain flag on the MRR time grid
        rain_flag = xr.DataArray(
            ds_mwr_interp.rain.values.astype(bool),
            coords={"time": ds_mrr.time},
            dims=("time",),
        )

        # check interference patterns in the MRR data for the selected site and day
        #calculate_interference_patterns(ds_mrr, date_selected, rain_flag)

        # Set all non-rain measurement profiles to NaN, but keep coordinate-like
        # variables intact. In these MRR files, height can be time-dependent, so
        # masking every variable with a time dimension would create NaN plot axes.
        vars_to_keep = {"height", "range", "lat", "lon", "latitude", "longitude", "altitude", "time"}
        for var_name in ds_mrr.data_vars:
            if "time" in ds_mrr[var_name].dims and var_name not in vars_to_keep:
                ds_mrr[var_name] = ds_mrr[var_name].where(rain_flag)

        # plot time height Ze for the entire day
        plot_time_height_Ze(ds_mrr, date_selected, time_stamps, "rain_flag_filtered")

    # plot spectrogram for the selected time stamps to understand the interferences
    for time_stamp in time_stamps:
        ds_mrr_sel = ds_mrr.sel(time=time_stamp)
        print(ds_mrr_sel)
        plot_mrr_spectrogram(ds_mrr_sel)

        # for each time stamps, set to nan all variable at range gates where 

    # set up new interference filtering 



def calculate_interference_patterns(ds_mrr, date_selected, rain_flag):
    """
    function to plot reflectivity profiles of MRR data and mean doppler velocity profiles for the MRR data non rainy during interfercene

    input:
    - ds_mrr: dataset containing the MRR data for the selected site and day
    - date_selected: date of the selected data
    - rain_flag: boolean array indicating if there is rain or not for the selected site and day
    output:
    - plot of the reflectivity profiles of MRR data and mean doppler velocity profiles for the MRR data non rainy during interfercene
    """ 

    # select all time stamps where there is no rain in the time interval in which cable car is running (7 UTC to 15:30 UTC)
    cable_car_start = np.datetime64(f"{date_selected[:4]}-{date_selected[4:6]}-{date_selected[6:8]}T07:00:00")
    cable_car_end = np.datetime64(f"{date_selected[:4]}-{date_selected[4:6]}-{date_selected[6:8]}T15:30:00")
    time_mask = (ds_mrr.time.values >= cable_car_start) & (ds_mrr.time.values <= cable_car_end)

    ds_mrr_filtered = ds_mrr.sel(time=time_mask)

    # now filter the dataset to keep only the time stamps where there is no rain
    ds_mrr_filtered = ds_mrr_filtered.where(~rain_flag, drop=True)
    
    # plot all Ze profiles in a plot
    plt.figure(figsize=(10, 6))
    for time_value in ds_mrr_filtered.time.values:
        ds_mrr_sel = ds_mrr_filtered.sel(time=time_value)

        # set superthin lines for the Ze profiles
        plt.plot(ds_mrr_sel.Ze.values, ds_mrr_sel.height.values, alpha=0.5, linewidth=0.5, color='blue')
    plt.xlabel('Radar Reflectivity [dBZ]')
    plt.ylabel('Height [m]')
    plt.title(f'MRR Radar Reflectivity Profiles for {date_selected} (non-rain)')
    plt.show()
    # save figure
    plt.savefig(f"plots/mrr_interf_ze_profiles_{date_selected}.png")


    # plot all mean doppler velocity profiles in a plot
    plt.figure(figsize=(10, 6))
    for time_value in ds_mrr_filtered.time.values:
        ds_mrr_sel = ds_mrr_filtered.sel(time=time_value)

        # set superthin lines for the mean doppler velocity profiles
        plt.plot(ds_mrr_sel.W.values, ds_mrr_sel.height.values, alpha=0.5, linewidth=0.5, color='red')
    plt.xlabel('Mean Doppler Velocity [m/s]')
    plt.ylabel('Height [m]')
    plt.title(f'MRR Mean Doppler Velocity Profiles for {date_selected} (non-rain)')
    plt.show()
    # save figure
    plt.savefig(f"plots/mrr_interf_mean_doppler_velocity_profiles_{date_selected}.png")

    # calculate Vd(h+1)-Vd(h) for increasing heights
    Vd_diff = np.abs(np.diff(ds_mrr_filtered.W.values, axis=1))


    # plot a distribution of the Vd(h+1)-Vd(h) values for all time stamps and heights
    plt.figure(figsize=(10, 6))
    plt.hist(np.abs(Vd_diff.flatten()), bins=100, color='green', alpha=0.7) 
    plt.xlabel('Vd(h+1) - Vd(h) [m/s]')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Vd(h+1) - Vd(h) for {date_selected} (non-rain)')


    # save figure
    plt.savefig(f"plots/mrr_interf_mean_doppler_velocity_diff_distribution_{date_selected}.png")

    # print smallest Vd_diff value found
    print(f"Smallest Vd(h+1) - Vd(h) value found: {np.nanmin(Vd_diff)} m/s")

    return print(f"MRR data for {date_selected} plotted and saved.")


def plot_time_height_Ze(ds_mrr, date_selected, time_stamps, info_output):
    """
    Plot time-height radar reflectivity (Ze) for the entire day.

    Parameters:
    ds_mrr : xarray.Dataset
        The MRR dataset.
    date_selected : str
        The selected date in the format 'YYYYMMDD'.
    time_stamps : list of str
        List of time stamps to for spectrogram plotting in the format 'YYYYMMDDTHHMMSS'.
    info_output : str
        Additional information for the output filename about input data or filtering applied.
    """

    # plot radar reflectivity as a function of height and time for the entire day
    Ze = ds_mrr.Ze.values
    time = ds_mrr.time.values
    height = ds_mrr.height.values

    plt.figure(figsize=(10, 6))
    plt.pcolormesh(time, height, Ze.T, shading='auto', cmap='viridis')
    plt.colorbar(label='Radar Reflectivity [dBZ]')
    plt.ylabel('Height [m]')

    # format xaxis with time stamps as HH:MM
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
    plt.gcf().autofmt_xdate()
    plt.title('MRR Radar Reflectivity')
    plt.legend()

    # plot vertical lines for the time stamps selected
    for time_stamp in time_stamps:
        time_stamp_dt = pd.to_datetime(time_stamp)
        plt.axvline(x=time_stamp_dt, color='r', linestyle='--', label=f'Selected Time: {time_stamp_dt.strftime("%H:%M")}')  
    
    # set ylim min at the first height with Ze values non Nan
    min_height_idx = np.where(np.any(np.isfinite(Ze), axis=0))[0][0]
    plt.ylim(height[min_height_idx], height.max())

    # save figure
    plt.savefig(f"plots/mrr_reflectivity_{info_output}_{date_selected}.png")
    return print(f"MRR data for {info_output} on {date_selected} plotted and saved.")

def filter_interference_in_mrr(ds_mrr, filter_horizontal_band_filter_on, filter_spectra_connectivity_on=True, filter_Ze_vertical_continuity_on=True, ):
    """
    the function is filtering the interfence patterns in the Doppler spectra height spectrograms by applying 2 methods:
    1) it exploits connected component size: counts the number of connected 
    areas and then it compareded with a given threshold to decide 
    if the echo is fragmented (interference) when too many separated areas exist or coherent (precipitation) 
    when only a few separated areas in the Doppler height spectrogram exist
    2) it then checks the smoothness of the Ze profile using the function
    vertical_continuity_filter: if adjacent valid Ze bins show an abrupt jump
    larger than a chosen threshold, the profile is flagged as interference.
    Otherwise, the original Ze profile is retained.

    The code then returns:
    - the original dataset without filtering if the echo is coherent
    - a dataset with eta equal to nan everywhere if the echo is fragmented (interference)   
    input:
    - ds_mrr: dataset containing the MRR data for the selected site and day
    output:
    - ds_mrr_filtered: dataset containing the MRR data for the selected site 
    and day after filtering the interference. It contains a new
     variable interf_flag which is True if the echo is fragmented (interference) 
     and False if the echo is coherent

    """
    # define cable car starting and ending clock times in UTC
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
    if filter_spectra_connectivity_on:
        # loop over the time dimension and apply the interference filter to each profile
        for time_value in ds_mrr_window.time.values:

            # select the dataset for the current time step
            ds_mrr_sel = ds_mrr_filtered.sel(time=time_value)

            # extract the eta and Ze profiles for the current time step
            eta_dbz = ds_mrr_sel.eta.values
            ze_profile = ds_mrr_sel.Ze.values

            # filter the eta and Ze profiles using the filter_interference function
            eta_dbz_filtered, interf_flag = filter_interference(eta_dbz)

            # create a boolean mask to identify valid eta values above the threshold
            valid_height = np.any(
                np.isfinite(eta_dbz_filtered) &
                (eta_dbz_filtered > 3.0),
                axis=1
            )

            # create a filtered Ze profile by setting invalid heights to NaN
            ze_profile_filtered = ze_profile.copy() 
            ze_profile_filtered[~valid_height] = np.nan

            # update the dataset with the filtered eta and Ze profiles and the interference flag
            ds_mrr_filtered["eta"].loc[dict(time=time_value)] = eta_dbz_filtered
            ds_mrr_filtered["Ze"].loc[dict(time=time_value)] = ze_profile_filtered
            ds_mrr_filtered["interf_flag"].loc[dict(time=time_value)] = ~valid_height


    # Apply horizontal band filter to the Ze profiles in the datasetZe = ds_mrr_filtered.Ze.values
    if filter_horizontal_band_filter_on:

        # apply horizontal band filter to the Ze profiles in the dataset
        Ze = ds_mrr_filtered.Ze.values

        # create a boolean mask to select the time steps within the cable car time window
        time_mask = (
            (ds_mrr_filtered.time.values >= cable_car_start) &
            (ds_mrr_filtered.time.values <= cable_car_end)
        )

        # extract the height values from the dataset
        height = ds_mrr_filtered.height.values
        # apply the horizontal_component_filter function to the Ze profiles within the cable car time window
        tmp_flag = horizontal_component_filter(
            Ze[time_mask, :],
            height,
            ze_thr=-5,
            min_time_extent=15,
            max_height_extent_m=600,
            min_aspect_ratio=6,
            min_height_m=10,
        )
        
        # update the interf_flag variable in the dataset to indicate if the echo is fragmented (interference) or coherent (precipitation) after applying the horizontal band filter
        ds_mrr_filtered.interf_flag.values[time_mask, :] |= tmp_flag

    # apply vertical continuity filter to the Ze profiles in the dataset
    if filter_Ze_vertical_continuity_on:
        Ze = ds_mrr_filtered.Ze.values
        Ze_filtered, vertical_filter_flag = vertical_continuity_filter(Ze, 
                                                                ze_thr=-5, 
                                                                max_vertical_jump_db=8.0)
        
        ds_mrr_filtered.Ze.values = Ze_filtered  

        # update the filter flag in the dataset to indicate if the echo is fragmented (interference) or coherent (precipitation) after applying the vertical continuity filter
        ds_mrr_filtered.interf_flag.values = np.where(vertical_filter_flag, True, ds_mrr_filtered.interf_flag.values)
    
    return ds_mrr_filtered


def vertical_continuity_filter(Ze, ze_thr=-5, max_vertical_jump_db=8.0):
    """
    Filter Ze profiles using only a vertical jump criterion.

    A profile is kept when the Ze signal changes smoothly with height and is
    rejected when two adjacent valid range bins differ by more than
    max_vertical_jump_db.

    Ze shape: (time, height)
    input:
    - Ze: array containing the Ze profiles in dBZ (time, height) extracted from the MRR dataset
    - ze_thr: threshold for Ze values to be considered as valid (default: -5 dBZ)
    - max_vertical_jump_db: maximum allowed absolute Ze jump between two adjacent
    valid bins. Increase it to tolerate sharper vertical
    gradients, decrease it to reject profiles with abrupt bin-to-bin changes.
    output:
        - Ze_out: array containing the Ze profiles in dBZ after applying the jump-based filter
        - vertical_filter_flag: boolean array that is True where the profile is rejected
            because it contains a vertical Ze jump larger than the threshold

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
        profile = Ze[it, :]
        # create a boolean array to identify valid Ze values above the threshold
        valid = np.isfinite(profile) & (profile > ze_thr)
        # if there are no valid Ze values, skip the profile
        if valid.sum() == 0:
            continue

        valid_inds = np.where(valid)[0]
        adjacent_mask = np.diff(valid_inds) == 1
        adjacent_jumps = np.abs(np.diff(profile[valid_inds]))[adjacent_mask]
        max_vertical_jump = adjacent_jumps.max() if adjacent_jumps.size else 0.0

        # reject profiles with abrupt Ze jumps along adjacent valid bins
        if max_vertical_jump > max_vertical_jump_db:
            Ze_out[it, :] = np.nan
            vertical_filter_flag[it, :] = True

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
    ze = ds_mrr.Ze.values
    ze_noDA = ds_mrr.Ze_noDA.values
    velocity_noDA = ds_mrr.velocity_noDA.values
    vd_noDA = ds_mrr.W_noDA.values
    vd = ds_mrr.W.values


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


    # plot the radar reflectivity profiles
    plt.figure(figsize=(10, 6))
    plt.plot(ze, height, marker='x', label='With DA')
    plt.plot(ze_noDA, height, marker='o', label='Without DA')
    plt.xlabel('Radar Reflectivity [dBZ]')
    plt.ylabel('Height [m]')
    plt.title('MRR Radar Reflectivity')
    plt.legend()
    # save figure
    plt.savefig(f"plots/mrr_reflectivity_{time_label}.png")

    # plot the mean Doppler velocity profiles
    plt.figure(figsize=(10, 6))
    plt.plot(vd, height, marker='x', label='With DA')
    plt.plot(vd_noDA, height, marker='o', label='Without DA')
    plt.xlabel('Mean Doppler Velocity [m/s]')
    plt.ylabel('Height [m]')
    plt.title('MRR Mean Doppler Velocity')
    plt.legend()
    # save figure
    plt.savefig(f"plots/mrr_mean_doppler_velocity_{time_label}.png")

    #plot the difference between consecutive values at increasing heights for the mean Doppler velocity profiles
    vd_diff = np.abs(np.diff(vd))
    plt.figure(figsize=(10, 6))
    plt.plot(vd_diff, height[:-1], marker='x', label='With DA')
    plt.xlabel('Vd(h+1) - Vd(h) [m/s]')
    plt.ylabel('Height [m]')
    plt.title(f'profile of  Vd(h+1) - Vd(h) for {time_label}')

    # plot a vertical line at x = 0.1
    plt.axvline(x=0.075, color='r', linestyle='--', label='Threshold: 0.1')

    # save figure
    plt.savefig(f"plots/mrr_mean_doppler_velocity_diff_profile_{time_label}.png")

def filter_interference(
    eta_dbz,
    eta_thr=3.0,
    min_component_size=20,
    min_vertical_extent=4,
):
    """
    function to filter the interference in the Doppler spectra height spectrograms by applying a connected component size criterion.
    The function counts the number of connected areas in the eta_dbz array and then compares it 
    with a given threshold to decide if the echo is fragmented (interference) when too many separated areas
    exist or coherent (precipitation) when only a few separated areas in the Doppler height spectrogram exist.
    input:
    - eta_dbz: 2D array of Doppler spectra [dBZ], shape: (height, velocity)
    - eta_thr: threshold for eta values to be considered as valid (default: 3.0 dBZ)
    - min_component_size: minimum number of connected components to consider the echo as coherent (default: 20)
    - min_vertical_extent: minimum vertical extent of the connected components to consider the echo as coherent (default: 4)
    output:
    - eta_out: 2D array of Doppler spectra [dBZ] after filtering    
    Removes only small/shallow spectral components, not the whole profile.
    - removed_any: boolean flag indicating if any connected components were removed (True) or not (False)   
    """
    # copy the input eta_dbz array to avoid modifying the original data
    eta_out = eta_dbz.copy()

    # create a boolean mask to identify valid eta values above the threshold
    mask = np.isfinite(eta_dbz) & (eta_dbz > eta_thr)

    # label connected components in the mask and get the number of labels
    labels, nlab = label(mask)

    # if there are no connected components, return the original eta_dbz array and False
    if nlab == 0:
        return eta_out, False

    # loop over the connected components and remove those that are too small or too shallow
    removed_any = False
    for ilab in range(1, nlab + 1):
        # create a boolean mask for the current connected component
        component = labels == ilab
        # calculate the size of the connected component
        size = component.sum()
        # calculate the vertical extent of the connected component
        z_idx, v_idx = np.where(component)
        vertical_extent = z_idx.max() - z_idx.min() + 1

        # determine if the connected component should be removed based on its size and vertical extent
        remove_component = (
            size < min_component_size
            or vertical_extent < min_vertical_extent
        )
        
        # if the connected component should be removed, set the corresponding values in eta_out to NaN and update the removed_any flag
        if remove_component:
            eta_out[component] = np.nan
            removed_any = True

    return eta_out, removed_any

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

from scipy.ndimage import label, find_objects

def horizontal_component_filter(
    Ze,
    height,
    ze_thr=-5,
    min_time_extent=20,
    max_height_extent_m=600,
    min_aspect_ratio=8,
    min_height_m=8000,
):
    """
    Flags long, thin, high-altitude horizontal structures in Ze(time, height).

    Ze shape: time, height
    height shape: height
    """

    flag = np.full_like(Ze, False, dtype=bool)

    valid = np.isfinite(Ze) & (Ze > ze_thr)
    valid[:, height < min_height_m] = False

    structure = np.ones((3, 3), dtype=bool)
    labels, nlab = label(valid, structure=structure)
    objects = find_objects(labels)

    for ilab, slc in enumerate(objects, start=1):
        if slc is None:
            continue

        t_slice, z_slice = slc

        time_extent = t_slice.stop - t_slice.start
        z_inds = np.arange(z_slice.start, z_slice.stop)

        height_extent_m = height[z_inds].max() - height[z_inds].min()
        aspect_ratio = time_extent / max(len(z_inds), 1)

        is_horizontal_band = (
            time_extent >= min_time_extent
            and height_extent_m <= max_height_extent_m
            and aspect_ratio >= min_aspect_ratio
        )

        if is_horizontal_band:
            flag[labels == ilab] = True

    return flag


def find_MRR_flag(site_selected, date):
    """
    code to find the MWR flag file for the selected site and day
    input:
    - site_selected: name of the selected site
    - date: date of the selected data
    output:
    - True if the MWR flag file exists, False otherwise
    """
    if site_selected == 'collalbo':
        instr = 'kithat'
    elif site_selected == 'lagonero':
        instr = 'tophat'
    elif site_selected == 'bolzano':
        instr = 'hatpro'

    path_root = '/data/obs/campaigns/teamx/' + site_selected + '/'+ instr +'/actris/level1/'
    
    # read yy, mm, dd from date
    yy = date[:4]
    mm = date[4:6]
    dd = date[6:8]
    
    path_global = path_root + yy + '/' + mm + '/' + dd + '/'
    filename = path_global + 'MWR_1C01_'+site_selected+'_'+date+'.nc'
    
    # check if the file exists
    if os.path.exists(filename):
        return True
    else:
        print(f"No MWR flag file found for site {site_selected} and date {date} in path {path_global}")
        return False



def detect_velocity_plateaus(
    vd: np.ndarray,
    *,
    tolerance: float = 0.15,
    min_gates: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect nearly constant VD over consecutive range gates.

    Two adjacent finite gates are connected when their absolute VD difference is
    no larger than ``tolerance``.  All gates in a connected run are flagged when
    the run contains at least ``min_gates`` gates.

    Parameters
    ----------
    vd:
        One-dimensional mean Doppler velocity profile, ordered by height.
    tolerance:
        Maximum difference between adjacent gates, in the same units as ``vd``
        (normally m/s).
    min_gates:
        Minimum number of consecutive gates in a plateau.

    Returns
    -------
    mask, step:
        Boolean plateau mask and absolute adjacent-gate VD difference.  The
        first value of ``step`` is NaN because it has no preceding gate.
    """
    vd = np.asarray(vd, dtype=float)
    if vd.ndim != 1:
        raise ValueError("vd must be a one-dimensional profile")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    if min_gates < 2:
        raise ValueError("min_gates must be at least 2")

    step = np.full(vd.size, np.nan)
    if vd.size > 1:
        step[1:] = np.abs(np.diff(vd))

    adjacent_match = (
        np.isfinite(vd[:-1])
        & np.isfinite(vd[1:])
        & (np.abs(np.diff(vd)) <= tolerance)
    )
    matched_edges = _mark_true_runs(adjacent_match, min_gates - 1)

    mask = np.zeros(vd.size, dtype=bool)
    edge_indices = np.flatnonzero(matched_edges)
    mask[edge_indices] = True
    mask[edge_indices + 1] = True
    return mask, step


def detect_ze_zigzags(
    ze: np.ndarray,
    *,
    min_step: float = 1.0,
    min_turns: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detect repeated alternating changes in a Ze profile.

    A turn occurs at gate ``h`` if the slopes on its two sides have opposite
    signs and both changes have magnitude at least ``min_step``.  Consecutive
    turns are retained when there are at least ``min_turns`` of them.  The mask
    includes the end gates participating in the zigzag.

    Parameters
    ----------
    ze:
        One-dimensional reflectivity profile in dBZ, ordered by height.
    min_step:
        Minimum absolute reflectivity change between neighboring gates in dBZ.
        This prevents tiny noise fluctuations from being called a zigzag.
    min_turns:
        Minimum number of consecutive direction changes.  Two turns correspond
        to a pattern involving at least four gates.

    Returns
    -------
    mask, step, turn:
        Boolean zigzag mask, signed adjacent-gate Ze difference, and Boolean
        mask of the central turning gates.
        turn
    """
    ze = np.asarray(ze, dtype=float)
    if ze.ndim != 1:
        raise ValueError("ze must be a one-dimensional profile")
    if min_step < 0:
        raise ValueError("min_step must be non-negative")
    if min_turns < 1:
        raise ValueError("min_turns must be at least 1")

    step = np.full(ze.size, np.nan)
    if ze.size > 1:
        step[1:] = np.diff(ze)

    turn = np.zeros(ze.size, dtype=bool)
    if ze.size >= 3:
        left = np.diff(ze)[:-1]
        right = np.diff(ze)[1:]
        finite = np.isfinite(ze[:-2]) & np.isfinite(ze[1:-1]) & np.isfinite(ze[2:])
        turn[1:-1] = (
            finite
            & (left * right < 0.0)
            & (np.abs(left) >= min_step)
            & (np.abs(right) >= min_step)
        )

    retained_turns = _mark_true_runs(turn, min_turns)
    mask = retained_turns.copy()
    indices = np.flatnonzero(retained_turns)
    mask[np.maximum(indices - 1, 0)] = True
    mask[np.minimum(indices + 1, ze.size - 1)] = True
    return mask, step, retained_turns


if __name__ == "__main__":
    main()

