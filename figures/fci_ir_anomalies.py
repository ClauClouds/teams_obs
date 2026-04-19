"""
code to plot cloud fraction anomalies of convective days and MoBL_T days with respect to hourly mean diurnal cycle 
"""

from fileinput import filename
import site
from turtle import mode

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER

from matplotlib.colors import BoundaryNorm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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

from figures.fci_ir_vis_mean_dc import compute_mean_diurnal_cycle_for_selected_hours, get_hour_interval_bounds


def main():

    def format_hour_interval(hour_label):
        start_time = pd.to_datetime(hour_label, format="%H:%M")
        interval_starts, interval_ends = get_hour_interval_bounds(hours_diurnal_cycle_calc)
        start_index = hours_diurnal_cycle_calc.index(hour_label)
        end_time = interval_ends[start_index]
        return f"{start_time.strftime('%H:%M')}-{end_time.strftime('%H:%M')} UTC"

    def style_map_axes(ax):
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

    # read all IR FCI data for the campaign and calculate mean diurnal cycle of BT and VIS for each pixel of the FCI grid
    mode = "ir_105" #"VIS_06" # or "ir_105" # or 

    # check if mean diurnal cycle for convective days and MoBL_T days has already been calculated and saved to file, if yes read it, if not calculate it and save it to file    
    if os.path.exists(f"data/diurnal_cycle/mean_diurnal_cycle_convective_days_{mode}.nc") and os.path.exists(f"data/diurnal_cycle/mean_diurnal_cycle_MoBL_T_days_{mode}.nc"):
        mean_diurnal_cycle_convective = xr.open_dataset(f"data/diurnal_cycle/mean_diurnal_cycle_convective_days_{mode}.nc")
        mean_diurnal_cycle_MoBL_T = xr.open_dataset(f"data/diurnal_cycle/mean_diurnal_cycle_MoBL_T_days_{mode}.nc")
        print("Loaded mean diurnal cycle datasets from file.")
    else:
        file_list, n_files = find_all_files_for_site(fci_path, mode, "expats")

        # find in file_list all the files that correspond to the convective days and MoBL_T days
        convective_files = [f for f in file_list if any(day in f for day in iop_conv_days)]
        MoBL_T_files = [f for f in file_list if any(day in f for day in iop_MoBL_T_days)]

        mean_diurnal_cycle_convective = compute_mean_diurnal_cycle_for_selected_hours(convective_files, mode, hours_diurnal_cycle_calc)
        mean_diurnal_cycle_MoBL_T = compute_mean_diurnal_cycle_for_selected_hours(MoBL_T_files, mode, hours_diurnal_cycle_calc)

        # if path does not exist, create it
        if not os.path.exists("data/diurnal_cycle/"):
            os.makedirs("data/diurnal_cycle/")
        output_path_cv = f"data/diurnal_cycle/mean_diurnal_cycle_convective_days_{mode}.nc"
        output_path_MoBL_T = f"data/diurnal_cycle/mean_diurnal_cycle_MoBL_T_days_{mode}.nc"
        mean_diurnal_cycle_convective.to_netcdf(output_path_cv)
        mean_diurnal_cycle_MoBL_T.to_netcdf(output_path_MoBL_T)
        print(f"Saved mean diurnal cycle dataset for convective days to {output_path_cv}")
        print(f"Saved mean diurnal cycle dataset for MoBL_T days to {output_path_MoBL_T}")

    
    import cmcrameri.cm as cm   
    # plot the mean diurnal cycle for convective days exactly as in fci_ir_vis_mean_dc.py but with the mean diurnal cycle calculated only for the selected hours and with the same colorbar limits for convective and MoBL_T days to be able to compare them
    # set colorbar limits to be the same for convective and MoBL_T days
    
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
        mean_var = mean_diurnal_cycle_convective[f"mean_{mode}"].sel(hour=hour)
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
        cloud_fraction = mean_diurnal_cycle_convective[f"cloud_fraction_{mode}"].sel(hour=hour)
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
    fig.savefig(f"plots/maps/fci_mean_diurnal_cycle_convective_days.png", dpi=300)
    fig2.savefig(f"plots/maps/fci_cloud_fraction_convective_days.png", dpi=300)
    coords_ds.close()
    ds_orography.close()


    # do the same plots for MoBL_T days
    # plot
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows), subplot_kw={"projection": ccrs.PlateCarree()})
    axes = axes.flatten()
    img = None
    for i, hour in enumerate(hours_diurnal_cycle_calc):

        ax = axes[i]
        mean_var = mean_diurnal_cycle_MoBL_T[f"mean_{mode}"].sel(hour=hour)
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
        cloud_fraction = mean_diurnal_cycle_MoBL_T[f"cloud_fraction_{mode}"].sel(hour=hour)
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
    if not os.path.exists("plots/maps/"):
        os.makedirs("plots/maps/")
    fig.savefig(f"plots/maps/fci_mean_diurnal_cycle_MoBL_T_days.png", dpi=300)
    fig2.savefig(f"plots/maps/fci_cloud_fraction_MoBL_T_days.png", dpi=300)
    coords_ds.close()
    ds_orography.close()    

if __name__ == "__main__":
    main()