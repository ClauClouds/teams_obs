"""code to calculate for the group of days of interest (conv or MOBL-T) 
the mean IWV deviation over specific time intervals profided by users
"""

import site

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
import xarray as xr
from figures.utils import calc_iwv_deviation, create_site_inset, plot_iwv_ring_on_map
from readers.data_info import orography_path, iop_conv_days, iop_MoBL_T_days, hours_diurnal_cycle_calc
import os
import pdb

def build_colorbar_ticks(var_plot, max_labels=None, exact_labels=False):
    """Build readable colorbar ticks for the selected variable."""
    tick_step = VAR_DICT[var_plot].get('tick_step', 2.0)

    if exact_labels and max_labels is not None and max_labels > 1:
        return np.linspace(
            VAR_DICT[var_plot]['vmin'],
            VAR_DICT[var_plot]['vmax'],
            max_labels,
        )

    ticks = np.arange(
        VAR_DICT[var_plot]['vmin'],
        VAR_DICT[var_plot]['vmax'] + 0.5 * tick_step,
        tick_step,
    )

    if max_labels is not None and len(ticks) > max_labels:
        stride = int(np.ceil((len(ticks) - 1) / (max_labels - 1)))
        ticks = ticks[::stride]
        if not np.isclose(ticks[-1], VAR_DICT[var_plot]['vmax']):
            ticks = np.r_[ticks, VAR_DICT[var_plot]['vmax']]

    return ticks

def main():

    # define the time and variable to plot
    elev_sel = 30 # other options are 10, 20, 30,
    var_plot = 'IWV_deviation' # other options are 'IWV_deviation' 
    Transparent_flag = False # whether to save the figure with transparent background (True) or white background (False)

    # select days of IOP to plot from convective days and MoBL_T days lists defined in data_info.py
    days = iop_conv_days # iop_MoBL_T_days
    days_string = "convective_days"

    output_file = f"plots/maps/maps_{var_plot}_{days_string}_panel_{elev_sel}.png"
    if os.path.exists(output_file):            
        print(f"File {output_file} already exists. Skipping day {days_string}.")
    else:
        print(f"Processing days {days_string}...")

    # list to collect datasets for each day of the selected ones
    ds_diurnal_cycle = []
    for day in days:
   
        print(f"Processing day {day}...") 

        # set hours to plot and time steps array for single plotting without averages 6,8,10,12,14,16
        time_selections = [f"{day[:4]}-{day[4:6]}-{day[6:8]}T{hour}:00" for hour in hours_diurnal_cycle_calc]
        print("**************************")
        print(time_selections)

        # avoid processing 20250722 for now because of data issues
        if day == "20250722":
            print(f"Skipping day {day} due to data issues.")
            continue

        # read the data and calculate the IWV deviation mean over the selected time intervals for each site for this day

    
        site_names, site_datasets = prepare_site_datasets(day, elev_sel, var_plot, time_selections, site)

        # plot all selected hours in a single 3x2 figure for each variable.
        print(f"Plotting 3x2 panel of spatial distribution of {var_plot} for {day_string}...")

        plot_spatial_iwv_distribution_panel(day_string, elev_sel, var_plot, time_selections, Transparent_flag)




def prepare_site_datasets(day_string, elev_sel, var_plot, time_selections, site):
    """Read site datasets and define one shared normalization for the chosen variable.
    This function reads the IWV data for each site, calculates the deviation if needed, 
    and determines the min and max values across all sites to set a common color scale for the plots.
    parameters:
    - day_string: string in the format 'YYYYMMDD' representing the day to plot
    - elev_sel: elevation angle to select for the MWR data (e.g., 10, 20, or 30 degrees)
    - var_plot: variable to plot, either 'iwv' or 'IWV_deviation'
    - time_selections: list of time strings in the format 'YYYY-MM-DDTHH:MM:SS' to select for plotting (e.g., hourly times)
    returns:
    - site_names: list of site names
    - site_datasets: list of xarray datasets for each site, containing the selected variable and time dimension 

    """
    
    site_value_mins = []
    site_value_maxs = []
    site_datasets = []
    iwv_tick_step = VAR_DICT['iwv'].get('tick_step', 5.0)
    iwv_dev_tick_step = VAR_DICT['IWV_deviation'].get('tick_step', 1.0)

    # loop on the three site
    for site_name in site_names:
        path_root = f"/data/obs/campaigns/teamx/{site_name}/{MWR_SITES_NAMES[site_name]}/actris/level2/{day_string[:4]}/{day_string[4:6]}/{day_string[6:8]}/"
        ds_site = read_iwv_elev(site_name, day_string, 'iwv', elev_sel, path_root)
        site_datasets.append(ds_site)

        if ds_site.sizes.get('time', 0) == 0:
            print(f"No valid IWV data for {site_name} on {day_string} at {elev_sel} deg. Leaving inset empty.")
            continue

        if var_plot == 'IWV_deviation':
            ds_site['IWV_deviation'] = calc_iwv_deviation(ds_site)

        if plot_type == "hourly_mean":
            site_values = ds_site[var_plot].resample(time='1h').mean().values
        elif plot_type == "time_steps":
            site_values = ds_site[var_plot].values
        else:
            raise ValueError("plot_type must be either 'hourly_mean' or 'time_steps')")

        if var_plot == 'IWV_deviation':
            site_abs_max = np.nanmax(np.abs(site_values))
            site_abs_max = max(site_abs_max, 4.0)
            site_value_mins.append(-np.ceil(site_abs_max / iwv_dev_tick_step) * iwv_dev_tick_step)
            site_value_maxs.append(np.ceil(site_abs_max / iwv_dev_tick_step) * iwv_dev_tick_step)
        elif var_plot == 'iwv':
            site_value_mins.append(np.floor(np.nanmin(site_values) / iwv_tick_step) * iwv_tick_step)
            site_value_maxs.append(np.ceil(np.nanmax(site_values) / iwv_tick_step) * iwv_tick_step)
        else:
            raise ValueError("var_plot must be either 'iwv' or 'IWV_deviation'")

    if site_value_mins and site_value_maxs:
        VAR_DICT[var_plot]['vmin'] = float(np.nanmin(site_value_mins))
        VAR_DICT[var_plot]['vmax'] = float(np.nanmax(site_value_maxs))

    return site_names, site_datasets


def configure_map_axis(main_ax, domain_ACTA, ds_orography):
    """Configure one map axis and draw the orography background."""
    main_ax.coastlines(resolution='10m', color='0.35', linestyle=':', linewidth=0.5)
    main_ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='0.35', linewidth=0.5)
    main_ax.set_extent(domain_ACTA, crs=ccrs.PlateCarree())

    return main_ax.pcolormesh(
        ds_orography['lons'].values,
        ds_orography['lats'].values,
        ds_orography['orography'].values,
        cmap='Greys',
        vmin=0,
        vmax=2500,
        shading='auto',
        alpha=0.95,
        transform=ccrs.PlateCarree(),
        zorder=1,
    )


def plot_spatial_iwv_distribution_panel(day_string, elev_sel, var_plot, time_selections, Transparent_flag=False):
    """Create a 3x2 panel of hourly spatial distributions with shared figure colorbars."""
    site_names, site_datasets = prepare_site_datasets(day_string, elev_sel, var_plot, plot_type='hourly_mean')

    lon_margin = 0.07
    lat_margin = 0.06
    domain_ACTA = [
        min(site_lons) - lon_margin,
        max(site_lons) + lon_margin,
        min(site_lats) - lat_margin,
        max(site_lats) + lat_margin,
    ]

    ds_orography = xr.open_dataset(orography_path)
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 11.5),
        subplot_kw={'projection': ccrs.PlateCarree()},
    )
    fig.suptitle(
        pd.Timestamp(time_selections[0]).strftime('%Y-%m-%d'),
        fontsize=24,
        y=0.975,
    )
    fig.subplots_adjust(left=0.035, right=0.98, bottom=0.14, top=0.88, wspace=0.035, hspace=0.16)

    inset_size_deg = 0.05
    terrain = None
    mesh = None

    for index, (ax, time_sel) in enumerate(zip(axes.flat, time_selections)):
        terrain = configure_map_axis(ax, domain_ACTA, ds_orography)
        ax.set_title(pd.Timestamp(time_sel).strftime('%H:%M UTC'), fontsize=20, pad=10)
        ax.set_xticks([])
        ax.set_yticks([])

        wrax_bolzano = create_site_inset(ax, site_lons[0], site_lats[0], inset_size_deg)
        wrax_collalbo = create_site_inset(ax, site_lons[1], site_lats[1], inset_size_deg)
        wrax_lagonero = create_site_inset(ax, site_lons[2], site_lats[2], inset_size_deg)

        for inset_ax, site_name, ds_site in zip(
            [wrax_bolzano, wrax_collalbo, wrax_lagonero],
            site_names,
            site_datasets,
        ):
            current_mesh = plot_iwv_ring_on_map(
                inset_ax,
                site_name,
                ds_site,
                day_string,
                elev_sel,
                time_sel,
                var_plot=var_plot,
                update_limits=False,
            )
            if mesh is None and current_mesh is not None:
                mesh = current_mesh

    if mesh is None:
        raise ValueError(f"No valid site data available to draw {var_plot} for {day_string}.")

    cbar_y0 = 0.055
    cbar_height = 0.022
    cbar_width = 0.32
    cbar_gap = 0.08
    cbar_left = 0.12

    cax_var = fig.add_axes([cbar_left, cbar_y0, cbar_width, cbar_height])
    cbar = fig.colorbar(mesh, cax=cax_var, orientation='horizontal')
    cbar.set_label(VAR_DICT[var_plot]['label'], fontsize=20)
    max_labels = 7 if var_plot == 'iwv' else None
    colorbar_ticks = build_colorbar_ticks(
        var_plot,
        max_labels=max_labels,
        exact_labels=(var_plot == 'iwv'),
    )
    cbar.set_ticks(colorbar_ticks)
    cbar.ax.tick_params(labelsize=18)

    cax_orog = fig.add_axes([cbar_left + cbar_width + cbar_gap, cbar_y0, cbar_width, cbar_height])
    cbar_orog = fig.colorbar(terrain, cax=cax_orog, orientation='horizontal')
    cbar_orog.set_label('Orography [m]', fontsize=20)
    cbar_orog.ax.tick_params(labelsize=18)

    fig.savefig(
        f"plots/maps/maps_{var_plot}_{day_string}_panel_{elev_sel}.png",
        dpi=300,
        bbox_inches='tight',
        transparent=Transparent_flag,
    )
    plt.close(fig)

def plot_spatial_iwv_distribution(day_string, elev_sel, var_plot, time_sel, plot_type, Transparent_flag=False):
    """Create a single map for one selected time."""
    site_names, site_datasets = prepare_site_datasets(day_string, elev_sel, var_plot, plot_type)

    """create a map for all the convective dates averaged together"""
    site_names, site_datasets = prepare_site_datasets(day_string, elev_sel, var_plot, plot_type='time_steps')

    lat_bolzano, lon_bolzano = site_lats[0], site_lons[0]
    lat_collalbo, lon_collalbo = site_lats[1], site_lons[1]
    lat_lagonero, lon_lagonero = site_lats[2], site_lons[2]

    lon_margin = 0.07
    lat_margin = 0.06
    domain_ACTA = [
        min(site_lons) - lon_margin,
        max(site_lons) + lon_margin,
        min(site_lats) - lat_margin,
        max(site_lats) + lat_margin,
    ]

    fig = plt.figure(figsize=(8, 6))
    main_ax = plt.axes(projection=ccrs.PlateCarree())
    main_ax.set_title("", fontsize=16)

    ds_orography = xr.open_dataset(orography_path)
    terrain = configure_map_axis(main_ax, domain_ACTA, ds_orography)

    cax_orog = fig.add_axes([0.18, 0.06, 0.40, 0.02])
    cbar_orog = fig.colorbar(terrain, cax=cax_orog, orientation='horizontal')
    cbar_orog.set_label('Orography [m]', fontsize=10)
    cbar_orog.ax.tick_params(labelsize=9)

    inset_size_deg = 0.05
    wrax_bolzano = create_site_inset(main_ax, lon_bolzano, lat_bolzano, inset_size_deg)
    wrax_collalbo = create_site_inset(main_ax, lon_collalbo, lat_collalbo, inset_size_deg)
    wrax_lagonero = create_site_inset(main_ax, lon_lagonero, lat_lagonero, inset_size_deg)

    plot_iwv_ring_on_map(wrax_bolzano, site_names[0], site_datasets[0], day_string, elev_sel, time_sel, var_plot=var_plot, update_limits=False)
    plot_iwv_ring_on_map(wrax_collalbo, site_names[1], site_datasets[1], day_string, elev_sel, time_sel, var_plot=var_plot, update_limits=False)
    mesh = plot_iwv_ring_on_map(wrax_lagonero, site_names[2], site_datasets[2], day_string, elev_sel, time_sel, var_plot=var_plot, update_limits=False)

    cbar = fig.colorbar(mesh, ax=main_ax, pad=0.01, shrink=0.68, fraction=0.035)
    cbar.set_label(VAR_DICT[var_plot]['label'], fontsize=10)
    max_labels = 7 if var_plot == 'iwv' else None
    colorbar_ticks = build_colorbar_ticks(
        var_plot,
        max_labels=max_labels,
        exact_labels=(var_plot == 'iwv'),
    )
    cbar.set_ticks(colorbar_ticks)
    cbar.ax.tick_params(labelsize=9)

    timestamp_label = pd.Timestamp(time_sel).strftime('%Y-%m-%d %H:%M UTC')
    main_ax.text(
        0.02,
        0.02,
        timestamp_label,
        transform=main_ax.transAxes,
        fontsize=9,
        ha='left',
        va='bottom',
        bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.8, 'boxstyle': 'round,pad=0.2'},
    )

    fig.tight_layout(rect=[0.03, 0.08, 0.90, 0.98])

    fig.savefig(
        f"plots/maps/maps_{var_plot}_{day_string}_{time_sel}_{elev_sel}.png",
        dpi=300,
        bbox_inches='tight',
        transparent=Transparent_flag,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()

