"""functions useful to derive stats and calculate plots 

"""
from readers.data_info import MWR_SITES_NAMES, PLOT_SITES_NAMES
import matplotlib.pyplot as plt
from figures.plot_settings import VAR_DICT
import numpy as np   
import os
import pandas as pd
import subprocess
import xarray as xr
import pdb
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.projections.polar import PolarAxes
from readers.data_info import PLOT_SITES_NAMES, MWR_SITES_NAMES, site_lats, site_lons
from figures.plot_settings import VAR_DICT
import numpy as np
import pandas as pd
import xarray as xr
from readers.data_info import orography_path



def get_scan_ids(ds_iwv_elev, max_time_gap_seconds=300):
    """Label each sample with its azimuth scan id."""
    time_values = pd.to_datetime(ds_iwv_elev.time.values)
    azimuth_values = ds_iwv_elev.azimuth_angle.values

    if len(time_values) == 0:
        return np.array([], dtype=int)

    time_diffs = np.diff(time_values.values) / np.timedelta64(1, 's')
    azimuth_diffs = np.diff(azimuth_values)
    scan_start = np.r_[True, (time_diffs > max_time_gap_seconds) | (azimuth_diffs < 0)]

    return np.cumsum(scan_start) - 1


def calc_iwv_deviation(ds_iwv_elev):

    """
    Calculate the deviation of IWV from the mean over the azimuth scan.
    For every full scan, compute the mean IWV of that scan, then subtract that scan mean from each measurement in the same scan.
    So the output tells you:
    How much each individual IWV value is above or below the average IWV of its own scan.
    input:
    ds_iwv_elev: xarray dataset containing the IWV measurements at different azimuth angles and times for one elevation angle.
    Returns:
    xarray DataArray containing the deviation of IWV from the mean over the azimuth scan.
    """

    # find time intervals in which azimuth changes from 360 to 0, which indicates the start of a new azimuth scan
    scan_ids = get_scan_ids(ds_iwv_elev) # assigns each measurement to a scan number

    # calculate mean IWV for each scan
    mean_iwv_per_scan = pd.DataFrame({'iwv': ds_iwv_elev.iwv.values, 'scan_id': scan_ids}).groupby('scan_id')['iwv'].mean().values # mean IWV for each scan

    # calculate deviation from mean for each scan
    iwv_deviation = ds_iwv_elev.iwv.values - np.repeat(mean_iwv_per_scan, np.bincount(scan_ids)) # repeats each scan mean so it matches the original measurement length.

    return xr.DataArray(iwv_deviation, coords=ds_iwv_elev.iwv.coords, dims=ds_iwv_elev.iwv.dims)




def extract_closest_scan(ds_iwv_elev, time_sel, max_time_gap_seconds=300):
    """Return the full azimuth scan closest to the requested time."""
    time_values = pd.to_datetime(ds_iwv_elev.time.values) # Converts time into pandas datetime values, so time differences can be computed easily
    azimuth_values = ds_iwv_elev.azimuth_angle.values # Extracts the azimuth angle for each measurement.

    time_diffs = np.diff(time_values.values) / np.timedelta64(1, 's') # Computes the time gap in seconds between each pair of consecutive measurements.
    azimuth_diffs = np.diff(azimuth_values) # Computes how much azimuth changes from one measurement to the next.

    """Identifies the start of each azimuth scan based on either 
    a large time gap (greater than max_time_gap_seconds) 
    or a reset in azimuth (azimuth_diffs < 0)."""
    scan_start = np.r_[True, (time_diffs > max_time_gap_seconds) | (azimuth_diffs < 0)]
    scan_ids = np.cumsum(scan_start) - 1

    scan_mid_times = pd.Series(time_values).groupby(scan_ids).median()
    scan_index = np.abs(scan_mid_times - pd.Timestamp(time_sel)).argmin()

    return ds_iwv_elev.isel(time=scan_ids == scan_index)



def extract_scan_by_id(ds_iwv_elev, scan_ids, scan_id):
    """Return one full azimuth scan by its id."""
    return ds_iwv_elev.isel(time=scan_ids == scan_id)


def aggregate_scan_by_azimuth(ds_scan):
    """Average repeated beams at the same azimuth within one scan."""
    df_data = {
        'azimuth': ds_scan.azimuth_angle.values,
        'iwv': ds_scan.iwv.values,
    }
    if 'IWV_deviation' in ds_scan:
        df_data['IWV_deviation'] = ds_scan.IWV_deviation.values

    df_scan = pd.DataFrame(df_data)
    df_scan = df_scan.groupby('azimuth', as_index=False).mean().sort_values('azimuth')

    return df_scan


def get_shared_colorbar_limits(dataset_paths, data_var, var_plot, symmetric=False, scale_factor=1.0):
    """Compute one colorbar range across multiple datasets."""
    value_mins = []
    value_maxs = []

    for dataset_path in dataset_paths:
        if not os.path.exists(dataset_path):
            continue

        with xr.open_dataset(dataset_path) as dataset:
            if data_var not in dataset:
                continue

            values = dataset[data_var].values

        if values.size == 0 or np.all(np.isnan(values)):
            continue

        value_mins.append(float(np.nanmin(values)))
        value_maxs.append(float(np.nanmax(values)))

    if not value_mins or not value_maxs:
        colorbar_vmin = float(VAR_DICT[var_plot]['vmin'])
        colorbar_vmax = float(VAR_DICT[var_plot]['vmax'])
        return float(np.floor(colorbar_vmin * scale_factor)), float(np.ceil(colorbar_vmax * scale_factor))

    if symmetric:
        colorbar_limit = max(abs(min(value_mins)), abs(max(value_maxs)))
        if np.isclose(colorbar_limit, 0.0):
            colorbar_limit = 0.5
        colorbar_limit *= scale_factor
        colorbar_limit = float(np.ceil(colorbar_limit))
        return -colorbar_limit, colorbar_limit

    colorbar_vmin = float(min(value_mins))
    colorbar_vmax = float(max(value_maxs))
    if np.isclose(colorbar_vmin, colorbar_vmax):
        colorbar_vmin -= 0.5
        colorbar_vmax += 0.5

    center = 0.5 * (colorbar_vmin + colorbar_vmax)
    half_range = 0.5 * (colorbar_vmax - colorbar_vmin) * scale_factor
    colorbar_vmin = center - half_range
    colorbar_vmax = center + half_range

    return float(np.floor(colorbar_vmin)), float(np.ceil(colorbar_vmax))


def get_regular_integer_colorbar_spec(colorbar_vmin, colorbar_vmax, max_tick_count=6):
    """Return integer colorbar limits and regularly spaced integer ticks."""
    colorbar_vmin = int(np.floor(colorbar_vmin))
    colorbar_vmax = int(np.ceil(colorbar_vmax))

    if colorbar_vmin == colorbar_vmax:
        return colorbar_vmin, colorbar_vmax, np.array([colorbar_vmin], dtype=int)

    tick_step = max(1, int(np.ceil((colorbar_vmax - colorbar_vmin) / max(max_tick_count - 1, 1))))
    adjusted_vmin = int(np.floor(colorbar_vmin / tick_step) * tick_step)
    adjusted_vmax = int(np.ceil(colorbar_vmax / tick_step) * tick_step)
    ticks = np.arange(adjusted_vmin, adjusted_vmax + tick_step, tick_step, dtype=int)

    return adjusted_vmin, adjusted_vmax, ticks


def azimuth_to_edges(azimuth_deg):
    """Convert azimuth centers to angular bin edges in radians."""
    if len(azimuth_deg) == 1:
        width_deg = 5.0
        return np.deg2rad(np.array([azimuth_deg[0] - width_deg / 2, azimuth_deg[0] + width_deg / 2]))

    midpoint_edges = 0.5 * (azimuth_deg[:-1] + azimuth_deg[1:])
    first_edge = azimuth_deg[0] - 0.5 * (azimuth_deg[1] - azimuth_deg[0])
    last_edge = azimuth_deg[-1] + 0.5 * (azimuth_deg[-1] - azimuth_deg[-2])
    return np.deg2rad(np.r_[first_edge, midpoint_edges, last_edge])


def plot_iwv_azimuth_ring(
    ds_scan,
    site,
    elev_sel,
    var_plot='iwv',
    ax=None,
    add_colorbar=True,
    show_title=True,
    show_direction_labels=True,
):
    """Plot IWV as colored azimuth sectors extending from the origin."""
    df_scan = aggregate_scan_by_azimuth(ds_scan)

    azimuth_deg = df_scan['azimuth'].to_numpy()
    values = df_scan[var_plot].to_numpy()
    theta_edges = azimuth_to_edges(azimuth_deg)
    inner_radius = 0.0
    outer_radius = 1.0

    cmap = VAR_DICT[var_plot]['cmap']
    vmin = VAR_DICT[var_plot]['vmin']
    vmax = VAR_DICT[var_plot]['vmax']
    color_step = VAR_DICT[var_plot].get('color_step', 0.5)
    tick_step = VAR_DICT[var_plot].get('tick_step', 2.0)
    color_bounds = np.arange(vmin, vmax + color_step, color_step)
    colorbar_ticks = np.arange(vmin, vmax + tick_step, tick_step)
    num_bins = max(1, len(color_bounds) - 1)
    cmap = cmap.resampled(num_bins)
    norm = plt.matplotlib.colors.BoundaryNorm(color_bounds, cmap.N, clip=True)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
    else:
        fig = ax.figure

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    theta_grid, radius_grid = np.meshgrid(theta_edges, np.array([inner_radius, outer_radius]))
    mesh = ax.pcolormesh(
        theta_grid,
        radius_grid,
        values[np.newaxis, :],
        cmap=cmap,
        norm=norm,
        shading='flat',
        edgecolors='white',
        linewidth=1.0,
    )

    # plot a black circle representingt he outer radius of the plot
    ax.plot(np.linspace(0, 2 * np.pi, 100), np.full(100, outer_radius), color='black', linewidth=1.5)
    ax.set_ylim(0, outer_radius)
    ax.set_yticks([])
    ax.set_xticks(np.deg2rad(np.arange(0, 360, 45)))
    if show_direction_labels:
        ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
    else:
        ax.set_xticklabels([])
    ax.grid(color='0.8', linewidth=0.8)
    ax.spines['polar'].set_visible(False)

    if show_title:
        scan_start = pd.to_datetime(ds_scan.time.values.min()).strftime('%H:%M:%S')
        scan_end = pd.to_datetime(ds_scan.time.values.max()).strftime('%H:%M:%S')
        ax.set_title(
            f"{PLOT_SITES_NAMES[site]} {VAR_DICT[var_plot]['label']} by azimuth\n"
            f"{pd.to_datetime(ds_scan.time.values[0]).strftime('%Y-%m-%d')} {scan_start}-{scan_end} UTC, elevation {elev_sel}°",
            fontsize=16,
            pad=24,
        )

    if add_colorbar:
        cbar = fig.colorbar(mesh, ax=ax, pad=0.12, shrink=0.8)
        cbar.set_label(VAR_DICT[var_plot]['label'], fontsize=12)
        cbar.set_ticks(colorbar_ticks)
        cbar.ax.tick_params(labelsize=10)

    return fig, ax, mesh


def create_site_inset(main_ax, lon, lat, size_deg):
    """Create a polar inset axis centered on a site location."""
    return inset_axes(
        main_ax,
        width="100%",
        height="100%",
        bbox_to_anchor=(
            lon - size_deg / 2,
            lat - size_deg / 2,
            size_deg,
            size_deg,
        ),
        bbox_transform=main_ax.transData,
        axes_class=PolarAxes,
    )


def aggregate_scan_by_azimuth(ds_scan):
    """Average repeated beams at the same azimuth within one scan."""
    df_data = {
        'azimuth': ds_scan.azimuth_angle.values,
        'iwv': ds_scan.iwv.values,
    }
    if 'IWV_deviation' in ds_scan:
        df_data['IWV_deviation'] = ds_scan.IWV_deviation.values

    df_scan = pd.DataFrame(df_data)
    return df_scan.groupby('azimuth', as_index=False).mean().sort_values('azimuth')


def azimuth_to_edges(azimuth_deg):
    """Convert azimuth centers to angular bin edges in radians."""
    if len(azimuth_deg) == 1:
        width_deg = 5.0
        return np.deg2rad(np.array([azimuth_deg[0] - width_deg / 2, azimuth_deg[0] + width_deg / 2]))

    midpoint_edges = 0.5 * (azimuth_deg[:-1] + azimuth_deg[1:])
    first_edge = azimuth_deg[0] - 0.5 * (azimuth_deg[1] - azimuth_deg[0])
    last_edge = azimuth_deg[-1] + 0.5 * (azimuth_deg[-1] - azimuth_deg[-2])
    return np.deg2rad(np.r_[first_edge, midpoint_edges, last_edge])


def plot_map_azimuth_ring(ax, ds_scan, site, elev_sel, var_plot):
    """Draw one azimuth-ring plot on an existing polar inset axis."""

    df_scan = aggregate_scan_by_azimuth(ds_scan)
    azimuth_deg = df_scan['azimuth'].to_numpy()
    values = df_scan[var_plot].to_numpy()
    theta_edges = azimuth_to_edges(azimuth_deg)

    cmap = VAR_DICT[var_plot]['cmap']
    vmin = VAR_DICT[var_plot]['vmin']
    vmax = VAR_DICT[var_plot]['vmax']
    color_step = VAR_DICT[var_plot].get('color_step', 0.5)
    color_bounds = np.arange(vmin, vmax + color_step, color_step)
    num_bins = max(1, len(color_bounds) - 1)
    cmap = cmap.resampled(num_bins)
    norm = plt.matplotlib.colors.BoundaryNorm(color_bounds, cmap.N, clip=True)

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    theta_grid, radius_grid = np.meshgrid(theta_edges, np.array([0.0, 1.0]))
    mesh = ax.pcolormesh(
        theta_grid,
        radius_grid,
        values[np.newaxis, :],
        cmap=cmap,
        norm=norm,
        shading='flat',
        edgecolors='white',
        linewidth=0.6,
    )

    ax.plot(np.linspace(0, 2 * np.pi, 100), np.ones(100), color='black', linewidth=1.0)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([])
    ax.set_xticks(np.deg2rad(np.arange(0, 360, 90)))
    ax.set_xticklabels([])
    ax.grid(color='0.8', linewidth=0.6)
    ax.spines['polar'].set_visible(False)
    ax.text(-0.12, 0.5, PLOT_SITES_NAMES[site], transform=ax.transAxes, ha='right', va='center', fontsize=8)

    return mesh


def plot_iwv_ring_on_map(ax, site, ds_iwv_elev, day_string, elev_sel, time_sel, var_plot, update_limits=True):
    """Draw one azimuth-ring scan for a site on an existing map inset axis."""
    if ds_iwv_elev.sizes.get('time', 0) == 0:
        ax.set_axis_off()
        ax.text(0.5, 0.55, 'No\ndata', transform=ax.transAxes, ha='center', va='center', fontsize=8)
        ax.text(-0.12, 0.5, PLOT_SITES_NAMES[site], transform=ax.transAxes, ha='right', va='center', fontsize=8)
        return None

    # add max and min IWV values to the VAR_DICT to set the color scale of the plot, based on the values in the dataset
    if update_limits and var_plot == 'iwv':
        iwv_min = ds_iwv_elev.iwv.min().item()
        iwv_max = ds_iwv_elev.iwv.max().item()
        VAR_DICT['iwv']['vmin'] = np.floor(iwv_min / 2) * 2
        VAR_DICT['iwv']['vmax'] = np.ceil(iwv_max / 2) * 2

    # add max deviation value to the VAR_DICT for the IWV deviation plot, based on the mean absolute deviation in the dataset, to set a symmetric color scale around zero
    if update_limits and 'IWV_deviation' in ds_iwv_elev:
        iwv_dev_max = np.nanmean(np.abs(ds_iwv_elev.IWV_deviation.values))
        VAR_DICT['IWV_deviation']['vmin'] = -np.ceil(iwv_dev_max / 2) * 2
        VAR_DICT['IWV_deviation']['vmax'] = np.ceil(iwv_dev_max / 2) * 2

    ds_scan = extract_closest_scan(ds_iwv_elev, time_sel)
    return plot_map_azimuth_ring(ax, ds_scan, site, elev_sel, var_plot)





def make_video_from_frames(input_framerate, output_prefix, output_dir, target_duration_seconds=55):
    """
    code to create a video with specific duration from the input selected images, 
    which are then removed after the video is created to save space. 
    The video is saved in the same output directory as the images.
    input parameters:
    - input_framerate: the framerate to use for the video, which is calculated based on the number of images and the target duration.
    - output_prefix: the prefix of the image files to include in the video.
    - output_dir: the directory where the images are saved and where the video will be saved.
    - target_duration_seconds: the desired duration of the output video in seconds 
    (default is 55 seconds to keep the total video duration below one minute).   
    output:
    - a video file in mp4 format with the specified duration, created from the input images,
      and the input images are removed after the video is created.

    """
    # Keep the total video duration below one minute.
    output_framerate = input_framerate
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(input_framerate),
            "-pattern_type",
            "glob",
            "-i",
            f"{output_dir}/{output_prefix}_*.png",
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v",
            "libx264",
            "-r",
            str(output_framerate),
            "-pix_fmt",
            "yuv420p",
            f"{output_dir}/{output_prefix}.mp4",
        ],
        check=True,
    )

    # remove the generated images after creating the video
    for file in os.listdir(output_dir):
        if file.startswith(output_prefix) and file.endswith(".png"):
            os.remove(os.path.join(output_dir, file))
    return None


def find_all_files_for_site(path_root, filename_string, site_name, file_ending=".nc"):
    """
    function to find all files witha given string in all subdirectories of a given root path

    Args:
        path_root (str): root path to search for files
        filename_string (str): string to search for in filenames
        site_name (str): name of the site for which files are being searched, used for printing the number of files found

    Returns:
        list: list of file paths that match the search criteria
        int: number of files found
    """

    all_files = []
    for root, dirs, files in os.walk(path_root):
        for file in files:
            if file.endswith(file_ending) and filename_string in file:
                all_files.append(os.path.join(root, file))

    return all_files, len(all_files)




def read_file_list_for_mode(path_root, site_name, mode, iop_conv_days, iop_MoBL_T_days):
    """
    code to read the list of files for a given site and a given mode,
      which can be "diurnal_cycle", "convective_days" or "MOBL_T_days".    

    Args:
        path_root (str): root path where the files are stored
        site_name (str): name of the site for which files are being searched
        mode (str): mode for which files are being searched, can be "diurnal_cycle", "convective_days" or "MOBL_T_days"
        iop_conv_days (list): list of convective days to filter files for the "convective_days" mode
        iop_MoBL_T_days (list): list of MoBL_T days to filter files for the "MOBL_T_days" mode
    Raises:
        ValueError: if the mode is not one of the expected values

    Returns:
        list: list of file paths that match the search criteria
        int: number of files found
    Dependencies:
        find_all_files_for_site: function to find all files with a given string in all sub
        directories of a given root path, used to find the initial list of files before filtering by mode   
    """
    if mode == "diurnal_cycle":
        file_found_list, N_stat = find_all_files_for_site(path_root, f"MWR_single_{site_name}_", site_name)
    elif mode == "convective_days":
        # read all files
        file_found_list, N_stat = find_all_files_for_site(path_root, f"MWR_single_{site_name}_", site_name)
        # select from file_found_list only the files corresponding to the convective days defined in data_info.py
        file_found_list = [file for file in file_found_list if any(day in file for day in iop_conv_days)]
    elif mode == "MOBL_T_days":
        # read all files
        file_found_list, N_stat = find_all_files_for_site(path_root, f"MWR_single_{site_name}_", site_name)
        # select from file_found_list only the files corresponding to the MoBL_T days defined in data_info.py
        file_found_list = [file for file in file_found_list if any(day in file for day in iop_MoBL_T_days)]
    else:
        raise ValueError("mode must be either 'diurnal_cycle', 'convective_days' or 'MOBL_T_days'")
    N_stat = len(file_found_list)
    print(f"Found {N_stat} files for site {site_name} and mode {mode} in path {path_root}.")

    return file_found_list, N_stat



def find_closest_dc_value(hour_sel, azimuth_sel, hours_diurnal_cycle_calc, azimuth_bins, dc_var):
    """script to identify the closest diurnal cycle value for the given time step and azimuth angle values, to calculate the anomaly as the deviation from the diurnal cycle mean value at the same time step and azimuth values

    Args:
        hour_sel (_type_): selected hour of the time step to find the closest diurnal cycle value for
        azimuth_sel (_type_): azimuth angle value of the time step to find the closest diurnal cycle value for
        time_intervals_hours (_type_): array of hours corresponding to the time intervals of the diurnal cycle file
        azimuth_bins (_type_): array of azimuth bin edges used in the diurnal cycle file
        dc_var (_type_): array of diurnal cycle values corresponding to the time intervals and azimuth bins of the diurnal cycle file
    """

    # Convert configured hour strings like "06:00" to integer hours.
    time_intervals_hours = pd.to_datetime(hours_diurnal_cycle_calc, format="%H:%M").hour.to_numpy()

    # find which time_interval_hours values enclose the selected time_sel hour 
    dc_hour_sel = np.max(time_intervals_hours[time_intervals_hours <= hour_sel])

    # find index of the time_intervals_hours value corresponding to min_hour_dc
    dc_hour_sel_index = np.where(time_intervals_hours == dc_hour_sel)[0][0]

    wrapped_azimuth = np.mod(azimuth_sel, 360.0)
    available_azimuth_bins = min(len(azimuth_bins) - 1, dc_var.shape[1])
    effective_azimuth_bins = np.asarray(azimuth_bins[:available_azimuth_bins + 1])
    dc_az_bin_index = np.searchsorted(effective_azimuth_bins, wrapped_azimuth, side="right") - 1
    dc_az_bin_index = np.clip(dc_az_bin_index, 0, available_azimuth_bins - 1)
        
    # read dc values for the enclosing diurnal-cycle hour and azimuth bin
    dc_sel_value = dc_var[dc_hour_sel_index, dc_az_bin_index]
    return dc_sel_value


def calculate_mean_anomaly_for_time_selection(ds_site, day, hours_diurnal_cycle_calc, azimuth_bins, var_plot):
    """
    function to calculate the mean anomaly for each time selection of the diurnal cycle and each azimuth bin, 
    to be used in the maps of anomalies for the convective days and MoBL_T days modes.
    Args:
        ds_site (_type_): xarray dataset containing the data for one site and one day, with the anomaly variable already calculated as the deviation from the diurnal cycle mean value at the same time step and azimuth values
        day (_type_): string representing the day in the format "YYYYMMDD"
        hours_diurnal_cycle_calc (_type_): list of hours corresponding to the time intervals of the diurnal cycle calculation
        azimuth_bins (_type_): array of azimuth bin edges used in the diurnal cycle calculation
        var_plot (_type_): string representing the variable to plot, which is used to select the correct variable from the ds_site dataset
          for the calculation of the mean anomaly values
    Returns:
        np.ndarray: matrix of mean anomaly values for each time selection of the diurnal cycle and 
        each azimuth bin of the selected day, with shape (number of time selections, number of azimuth bins - 1)
    """
    
    # define time intervals for the diurnal cycle calculation as strings in the format "YYYY-MM-DDTHH:MM:SS" for each day and each hour of the diurnal cycle calculation
    interval_starts = [f"{day[:4]}-{day[4:6]}-{day[6:8]}T{hour}:00" for hour in hours_diurnal_cycle_calc]
    next_day = pd.Timestamp(day) + pd.Timedelta(days=1)
    interval_ends = interval_starts[1:] + [f"{next_day.strftime('%Y-%m-%d')}T00:00:00"]

    
    anomaly_dc = np.empty((len(hours_diurnal_cycle_calc), len(azimuth_bins)-1)) # matrix to collect the mean values for each time selection of the diurnal cycle for each day and each azimuth bin
    anomaly_dc.fill(np.nan) # initialize the anomaly matrix with nans
    # set hours to plot and time steps array for averaging over intervals  6,8,10,12,14,16
    for i, time_sel in enumerate(hours_diurnal_cycle_calc):

        # slice dataset for the time selection 
        ds_time_sel = ds_site.sel(time=slice(interval_starts[i], interval_ends[i]))


        if ds_time_sel.sizes.get('time', 0) == 0:
            print(
                f"No data available for time selection {interval_starts[i]}-{interval_ends[i]} "
                f"for day {day}. Leaving that interval as NaN."
            )
            continue

        # group by azimuth angle intervals of 20 degrees and calculate mean over the time selection for each azimuth angle
        try:
            # select for azimuth in the range 0-360 and group by azimuth angle intervals of 20 degrees and calculate mean over the time selection for each azimuth angle
            anomaly_dc[i, :] = ds_time_sel.groupby_bins("azimuth_angle", azimuth_bins).mean(dim="time", skipna=True)["anomaly"].values
        except Exception as e:
            print(f"Error calculating mean for time selection {interval_starts[i]}-{interval_ends[i]} for day {day}: {e}")
            print(f"Skipping time selection {time_sel} for day {day}.")
            print("-------------------------------- --------------------------------")
            continue
    return anomaly_dc



def plot_mean_azimuth_ring(ax, azimuth_edges_deg, values, var_plot, vmin=None, vmax=None, n_color_bins=20):
    """Plot one mean azimuth ring from pre-aggregated values."""
    if var_plot == "anomalies":
        cmap = VAR_DICT["IWV_deviation"]['cmap']
        if vmin is None:
            vmin = VAR_DICT["IWV_deviation"]['vmin']
        if vmax is None:
            vmax = VAR_DICT["IWV_deviation"]['vmax'] 
    else:
        cmap = VAR_DICT[var_plot]['cmap']

        if vmin is None:
            vmin = VAR_DICT[var_plot]['vmin']
        if vmax is None:
            vmax = VAR_DICT[var_plot]['vmax'] 

    color_bounds = np.linspace(vmin, vmax, n_color_bins + 1)
    cmap = cmap.resampled(n_color_bins)
    norm = plt.matplotlib.colors.BoundaryNorm(color_bounds, cmap.N, clip=True)

    theta_edges = np.deg2rad(azimuth_edges_deg)
    theta_grid, radius_grid = np.meshgrid(theta_edges, np.array([0.0, 1.0]))

    mesh = ax.pcolormesh(
        theta_grid,
        radius_grid,
        values[np.newaxis, :],
        cmap=cmap,
        norm=norm,
        shading='flat',
        edgecolors='white',
        linewidth=0.6,
    )

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.plot(np.linspace(0, 2 * np.pi, 100), np.ones(100), color='black', linewidth=1.0)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([])
    ax.set_xticks(np.deg2rad(np.arange(0, 360, 90)))
    ax.set_xticklabels(['N', 'E', 'S', 'W'])
    ax.grid(color='0.8', linewidth=0.6)
    ax.spines['polar'].set_visible(False)

    return mesh





def plot_teamx_sites(ax, color, symbol_size, label_fontsize=5):
    """
    Function to plot instrument positions on a map for the quicklook browser.
    Parameters
    ----------
    ax : cartopy axes
        The axes on which to plot the instrument positions.
    color : str
        The color of the instrument positions.
    symbol_size : int
        The size of the symbols used to represent the instrument positions.
    Returns
    -------
        None
        """       
    #add instrument positions for the quicklook browser
    # define a dictionary with coordinates of the sites, site names
    dict_towns = {
        #'Branzol': [46.4031302, 11.32243],
        #'Brixen': [46.71042, 11.65246],
        #'Dornacherof':[46.49978, 11.43554], 
        #'Ehrenburg': [46.79559, 11.83649],
        #'Felthuner hutte': [46.60479, 11.45674],
        #'Garganzone':[46.58495, 11.20144], 
        'Klobenstein':[46.53965, 11.45832], 
        #'Meran':[46.67114, 11.15257], 
        #'Naturns':[46.64995, 11.00418], 
        #'Plose': [46.69555, 11.73333],
        #'Rittenhorn': [46.61499, 11.46083],
        #'Sarnthein': [46.6427, 11.35729],
        'Schwarzseespitze': [46.59605, 11.45255], 
        #'Sterzing':[46.89633, 11.43214],
        #'St Martin':[46.78353, 11.22874]}
        'Bozen':[46.49829, 11.35479]}
    # loop on dictionary items for plotting scatter points
    for site, coords in dict_towns.items():
        lat, lon = coords
        ax.scatter(lon, lat, marker='x', color=color, s=symbol_size, transform=ccrs.PlateCarree())
        ax.text(
            lon + 0.01,
            lat - 0.01,
            site,
            color=color,
            transform=ccrs.PlateCarree(),
            ha='left',
            va='top',
            fontsize=label_fontsize,
        )
        

    return