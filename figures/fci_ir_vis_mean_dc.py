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
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import BoundaryNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.projections.polar import PolarAxes
from readers.data_info import PLOT_SITES_NAMES, MWR_SITES_NAMES, site_lats, site_lons
from readers.MWR import read_iwv_elev
from figures.plot_settings import VAR_DICT
import numpy as np
import pandas as pd
import re
import xarray as xr
from figures.utils import find_all_files_for_site, plot_teamx_sites
from readers.data_info import orography_path, iop_conv_days, iop_MoBL_T_days, hours_diurnal_cycle_calc, azimuth_bins, fci_path, coords_file_path, domain
import os
import pdb
import cmcrameri.cm as cm

def get_hour_interval_bounds(hour_labels):
    """Build half-open time intervals from successive hour labels."""

    interval_starts = pd.to_datetime(hour_labels, format="%H:%M")
    if len(interval_starts) < 2:
        raise ValueError("At least two hour labels are required to build averaging intervals.")

    interval_ends = list(interval_starts[1:])
    last_step = interval_starts[-1] - interval_starts[-2]
    interval_ends.append(interval_starts[-1] + last_step)
    return interval_starts, pd.DatetimeIndex(interval_ends)


def compute_mean_diurnal_cycle_for_selected_hours(file_list, var, hour_labels):
    """Compute the mean diurnal cycle for intervals between successive hour labels."""

    interval_starts, interval_ends = get_hour_interval_bounds(hour_labels)
    start_minutes = (interval_starts.hour * 60 + interval_starts.minute).to_numpy()
    end_minutes = (interval_ends.hour * 60 + interval_ends.minute).to_numpy()
    is_ir_variable = var.startswith("ir_")
    cloud_bt_threshold = 240.0

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
    y_coords = first_ds[var].coords.get("y")
    x_coords = first_ds[var].coords.get("x")
    first_ds.close()

    hourly_sum = np.zeros((len(interval_starts), dim_y, dim_x), dtype=np.float64)
    hourly_count = np.zeros((len(interval_starts), dim_y, dim_x), dtype=np.int32)
    if is_ir_variable:
        hourly_cloud_count = np.zeros((len(interval_starts), dim_y, dim_x), dtype=np.int32)
        hourly_total_count = np.zeros((len(interval_starts), dim_y, dim_x), dtype=np.int32)

    for file_path in file_list:
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
        time_values = np.array([
            day_start + np.timedelta64(10 * j, 'm') for j in range(n_time_per_file)
        ])
        time_index = pd.to_datetime(time_values)
        time_minutes = (time_index.hour * 60 + time_index.minute).to_numpy()
        
        raw_values = ds[var].values.astype(np.float32, copy=False)
        if raw_values.shape != (n_time_per_file, dim_y, dim_x):
            ds.close()
            raise ValueError(
                f"Variable {var} in {file_path} has shape {raw_values.shape}; expected {(n_time_per_file, dim_y, dim_x)}."
            )

        if is_ir_variable:
            cloud_mask = np.isfinite(raw_values) & (raw_values < cloud_bt_threshold)
            var_values = np.where(cloud_mask, raw_values, np.nan)
        else:
            cloud_mask = None
            var_values = raw_values

        for output_index, (start_minute, end_minute) in enumerate(zip(start_minutes, end_minutes)):
            if end_minute > start_minute:
                interval_mask = (time_minutes >= start_minute) & (time_minutes < end_minute)
            else:
                interval_mask = (time_minutes >= start_minute) | (time_minutes < end_minute)

            if not np.any(interval_mask):
                continue

            hour_values = var_values[interval_mask, :, :]
            valid_mask = np.isfinite(hour_values)
            hourly_sum[output_index, :, :] += np.where(valid_mask, hour_values, 0.0).sum(axis=0)
            hourly_count[output_index, :, :] += valid_mask.sum(axis=0, dtype=np.int32)

            if is_ir_variable:
                hourly_cloud_count[output_index, :, :] += cloud_mask[interval_mask, :, :].sum(axis=0, dtype=np.int32)
                hourly_total_count[output_index, :, :] += np.isfinite(raw_values[interval_mask, :, :]).sum(axis=0, dtype=np.int32)

        ds.close()

    mean_diurnal_cycle = np.full(hourly_sum.shape, np.nan, dtype=np.float32)
    np.divide(
        hourly_sum,
        hourly_count,
        out=mean_diurnal_cycle,
        where=hourly_count > 0,
    )

    coords = {"hour": hour_labels}
    if y_coords is not None:
        coords["y"] = y_coords
    else:
        coords["y"] = np.arange(dim_y)
    if x_coords is not None:
        coords["x"] = x_coords
    else:
        coords["x"] = np.arange(dim_x)

    data_vars = {
        f"mean_{var}": (("hour", "y", "x"), mean_diurnal_cycle),
        f"count_{var}": (("hour", "y", "x"), hourly_count),
    }

    if is_ir_variable:
        cloud_fraction = np.full(hourly_cloud_count.shape, np.nan, dtype=np.float32)
        np.divide(
            hourly_cloud_count,
            hourly_total_count,
            out=cloud_fraction,
            where=hourly_total_count > 0,
        )
        data_vars[f"cloud_count_{var}"] = (("hour", "y", "x"), hourly_cloud_count)
        data_vars[f"total_count_{var}"] = (("hour", "y", "x"), hourly_total_count)
        data_vars[f"cloud_fraction_{var}"] = (("hour", "y", "x"), cloud_fraction)

    dataset = xr.Dataset(data_vars, coords=coords)
    dataset.attrs["aggregation_window"] = "interval_between_successive_hour_labels"
    dataset.attrs["interval_labels"] = ",".join(hour_labels)
    return dataset




def load_coords(channel: str):
    """
    Load data for one day for a specific channel.
    """
    coord_file = f"{coords_file_path}/{channel}_original_coords.nc"
    ds_coords = xr.open_dataset(coord_file)
    
    return ds_coords

def main():

    def format_hour_interval(hour_label):
        start_time = pd.to_datetime(hour_label, format="%H:%M")
        interval_starts, interval_ends = get_hour_interval_bounds(hours_diurnal_cycle_calc)
        start_index = hours_diurnal_cycle_calc.index(hour_label)
        end_time = interval_ends[start_index]
        return f"{start_time.strftime('%H:%M')}-{end_time.strftime('%H:%M')} UTC"

    def style_map_axes(ax):
        min_lon, max_lon, min_lat, max_lat = domain
        ax.coastlines(resolution='10m', color='orangered', linestyle=':', linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linestyle=':', color='orangered', linewidth=0.5)
        ax.contour(
            ds_orography['lons'].values,
            ds_orography['lats'].values,
            ds_orography['orography'].values,
            levels=contour_levels,
            colors='dimgray',
            linewidths=0.7,
            linestyles=contour_linestyles,
            transform=ccrs.PlateCarree(),
        )
        plot_teamx_sites(ax, color='black', symbol_size=10, label_fontsize=city_label_fontsize)
        ax.set_extent(domain, crs=ccrs.PlateCarree())

        gridlines = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            linewidth=0.5,
            color='0.5',
            alpha=0.6,
            linestyle=':',
            x_inline=False,
            y_inline=False,
        )
        gridlines.top_labels = False
        gridlines.right_labels = False
        gridlines.xlocator = mticker.FixedLocator(np.arange(10.75, 12.01, 0.25))
        gridlines.ylocator = mticker.FixedLocator(np.arange(46.4, 47.21, 0.2))
        gridlines.xlabel_style = {'size': map_tick_fontsize}
        gridlines.ylabel_style = {'size': map_tick_fontsize}
        gridlines.xformatter = LONGITUDE_FORMATTER
        gridlines.yformatter = LATITUDE_FORMATTER

    # if nc file with mean diurnal cycle already exists, read it and skip the calculation
    mode = "ir_105" #"VIS_06" # or "ir_105
    output_path = f"data/diurnal_cycle/mean_diurnal_cycle_{mode}.nc"
    recompute_diurnal_cycle = True
    if os.path.exists(output_path):
        print(f"Mean diurnal cycle dataset for {mode} already exists. Reading from file...")
        mean_diurnal_cycle_ds = xr.open_dataset(output_path)
        recompute_diurnal_cycle = (
            mean_diurnal_cycle_ds.attrs.get("aggregation_window")
            != "interval_between_successive_hour_labels"
        )
        if recompute_diurnal_cycle:
            mean_diurnal_cycle_ds.close()
            print("Existing diurnal cycle dataset uses outdated hourly aggregation. Recomputing...")
        else:
            print(f"Shape of the mean diurnal cycle dataset: {mean_diurnal_cycle_ds[f'mean_{mode}'].shape}")

    if recompute_diurnal_cycle:
        # read all IR FCI data for the campaign and calculate mean diurnal cycle of BT and VIS for each pixel of the FCI grid
        file_list, n_files = find_all_files_for_site(fci_path, mode, "expats")
        if n_files == 0:
            raise ValueError(f"No FCI files found for mode {mode} under {fci_path}.")

        file_list = sorted(file_list)
        print(file_list)

        mean_diurnal_cycle_ds = compute_mean_diurnal_cycle_for_selected_hours(
            file_list,
            mode,
            hours_diurnal_cycle_calc,
        )
        print(f"Shape of the mean diurnal cycle dataset: {mean_diurnal_cycle_ds[f'mean_{mode}'].shape}")

        # if path does not exist, create it
        if not os.path.exists("data/diurnal_cycle/"):
            os.makedirs("data/diurnal_cycle/")
        mean_diurnal_cycle_ds.to_netcdf(output_path)
        print(f"Saved mean diurnal cycle dataset to {output_path}")


    print("plotting mean diurnal cycle maps...")
    CMAP1 = cm.batlow_r
    n_hours = len(hours_diurnal_cycle_calc)
    n_cols = 3
    n_rows = (n_hours + n_cols - 1) // n_cols
    title_fontsize = 16
    map_tick_fontsize = 12
    colorbar_tick_fontsize = 16
    colorbar_label_fontsize = 18
    city_label_fontsize = 10
    cloud_fraction_levels = np.arange(0.0, 0.401, 0.05)
    cloud_fraction_norm = BoundaryNorm(cloud_fraction_levels, ncolors=plt.get_cmap("Blues").N, clip=True)

    # reading orography data for plotting
    ds_orography = xr.open_dataset( orography_path)

    # reading coordinates for the channel to plot the maps with correct lat and lon values
    coords_ds = xr.open_dataset(f"{coords_file_path}/{mode}_original_coords.nc")#load_coords(channel)
    lat = coords_ds['latitude'].values
    lon = coords_ds['longitude'].values
    contour_levels = [1000, 2000, 3000]
    contour_linestyles = [':', '--', '-']

    # plot  
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows), subplot_kw={"projection": ccrs.PlateCarree()})
    axes = axes.flatten()
    img = None

    for i, hour in enumerate(hours_diurnal_cycle_calc):

        ax = axes[i]
        mean_var = mean_diurnal_cycle_ds[f"mean_{mode}"].sel(hour=hour)
        ax.set_title(format_hour_interval(hour), fontsize=title_fontsize)
        ax.tick_params(labelsize=map_tick_fontsize)
        
        img = ax.pcolormesh(lon, lat, mean_var.values, transform=ccrs.PlateCarree(), cmap=CMAP1, vmin=210, vmax=270)
        style_map_axes(ax)

    fig.tight_layout(rect=[0.0, 0.0, 0.88, 1.0])

    if img is not None:
        cax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
        cbar = fig.colorbar(img, cax=cax, orientation='vertical')
        cbar.ax.tick_params(labelsize=colorbar_tick_fontsize)
        cbar.set_label("BT 10.5 micron (K)", fontsize=colorbar_label_fontsize)
        
    
    for ax in axes[n_hours:]:
        ax.set_axis_off()

    ############################################
    
    fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows), subplot_kw={"projection": ccrs.PlateCarree()})
    axes2 = axes2.flatten()
    img = None
    for i, hour in enumerate(hours_diurnal_cycle_calc):
        ax = axes2[i]
        cloud_fraction = mean_diurnal_cycle_ds[f"cloud_fraction_{mode}"].sel(hour=hour)
        ax.set_title(format_hour_interval(hour), fontsize=title_fontsize)
        ax.tick_params(labelsize=map_tick_fontsize)
        
        img = ax.pcolormesh(
            lon,
            lat,
            cloud_fraction.values,
            transform=ccrs.PlateCarree(),
            cmap="Blues",
            norm=cloud_fraction_norm,
        )
        style_map_axes(ax)
        
    for ax in axes2[n_hours:]:
        ax.set_axis_off()

    fig2.tight_layout(rect=[0.0, 0.0, 0.88, 1.0])

    if img is not None:
        cax = fig2.add_axes([0.90, 0.15, 0.02, 0.70])
        cbar = fig2.colorbar(
            img,
            cax=cax,
            orientation='vertical',
            boundaries=cloud_fraction_levels,
            ticks=cloud_fraction_levels,
            spacing='proportional',
        )
        cbar.ax.tick_params(labelsize=colorbar_tick_fontsize)
        cbar.set_label("High Cloud fraction (BT <240 K)", fontsize=colorbar_label_fontsize)


    # save figures to file
    # create path if it does not exist
    if not os.path.exists("plots/maps/"):
        os.makedirs("plots/maps/")
    fig.savefig(f"plots/maps/fci_mean_diurnal_cycle_{mode}.png", dpi=300)
    fig2.savefig(f"plots/maps/fci_cloud_fraction_{mode}.png", dpi=300)
    coords_ds.close()
    ds_orography.close()

if __name__ == "__main__":
    main()
