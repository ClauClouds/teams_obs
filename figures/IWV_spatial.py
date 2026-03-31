"""
code to plot spatial distribution of IWV as a function of time for each site for convective and MOBL_T days.
We plot a round circle with directions on any radius indicating the IWV at that time of the day. We have one plot for each time step
and we plot a video by merging the time steps together.

"""

from readers.data_info import MWR_SITES_NAMES, PLOT_SITES_NAMES
from readers.MWR import read_iwv_elev
import matplotlib.pyplot as plt
from figures.plot_settings import VAR_DICT
import numpy as np   
import os
import pandas as pd
import subprocess
import xarray as xr
import pdb

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


def get_scan_ids(ds_iwv_elev, max_time_gap_seconds=300):
    """Label each sample with its azimuth scan id."""
    time_values = pd.to_datetime(ds_iwv_elev.time.values)
    azimuth_values = ds_iwv_elev.azimuth_angle.values

    time_diffs = np.diff(time_values.values) / np.timedelta64(1, 's')
    azimuth_diffs = np.diff(azimuth_values)
    scan_start = np.r_[True, (time_diffs > max_time_gap_seconds) | (azimuth_diffs < 0)]

    return np.cumsum(scan_start) - 1


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

def main():

    day_string = "20250625"
    site = "lagonero"
    elev_sel = 30
    var2plot= "IWV_deviation" # "iwv" or "IWV_deviation"

    # read MWR data for spatial IWV distributions
    path_root = f"/data/obs/campaigns/teamx/{site}/{MWR_SITES_NAMES[site]}/actris/level2/{day_string[:4]}/{day_string[4:6]}/{day_string[6:8]}/"
    ds_iwv_elev = read_iwv_elev(site, day_string, "iwv", elev_sel, path_root)

    # add variable for deviation from mean IWV over the azimuth scan
    IWV_deviation = calc_iwv_deviation(ds_iwv_elev)
    # add deviation to 
    ds_iwv_elev['IWV_deviation'] = IWV_deviation

    output_dir = "plots/IWV_spatial"
    output_prefix = f"{var2plot}_spatial_{site}_{day_string}_{elev_sel}deg"
    os.makedirs(output_dir, exist_ok=True)

    # identify max and min IWV values during the day to set the color scale of the plots
    iwv_min = ds_iwv_elev.iwv.min().item()
    iwv_max = ds_iwv_elev.iwv.max().item()
    iwv_dev_max = np.nanmean(np.abs(ds_iwv_elev.IWV_deviation.values))

    VAR_DICT['iwv']['vmin'] = np.floor(iwv_min / 2) * 2
    VAR_DICT['iwv']['vmax'] = np.ceil(iwv_max / 2) * 2   
    VAR_DICT['IWV_deviation']['vmin'] = -np.ceil(iwv_dev_max / 2) * 2
    VAR_DICT['IWV_deviation']['vmax'] = np.ceil(iwv_dev_max / 2) * 2

    # Generate one frame per azimuth scan rather than one frame per sample.
    scan_ids = get_scan_ids(ds_iwv_elev)
    unique_scan_ids = np.unique(scan_ids)
    for scan_id in unique_scan_ids:
        ds_scan = extract_scan_by_id(ds_iwv_elev, scan_ids, scan_id)
        scan_time = pd.to_datetime(ds_scan.time.values[0])

        fig, _, _ = plot_iwv_azimuth_ring(ds_scan, site, elev_sel, var2plot)
        fig.savefig(f"{output_dir}/{output_prefix}_{scan_time.strftime('%Y%m%d_%H%M%S')}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    # Keep the total video duration below one minute.
    target_duration_seconds = 55
    input_framerate = max(1, int(np.ceil(len(unique_scan_ids) / target_duration_seconds)))
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
        

if __name__ == "__main__":
    main()


