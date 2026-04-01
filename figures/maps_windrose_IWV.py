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
from readers.data_info import orography_path


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
    var_plot_names = ['iwv', 'IWV_deviation'] # other options are 'IWV_deviation' 
    Transparent_flag = False # whether to save the figure with transparent background (True) or white background (False)
    plot_type = "hourly_mean" # "hourly_mean" or "time_steps"

    # select day to plot
    day_string = "20250625"
    # set hours to plot and time steps array for single plotting without averages
    hours = ["06:00", "09:00", "12:00", "15:00", "18:00", "20:00"]
    time_steps = [f"{day_string[:4]}-{day_string[4:6]}-{day_string[6:8]}T08:15:00",
                    f"{day_string[:4]}-{day_string[4:6]}-{day_string[6:8]}T12:15:00",
                    f"{day_string[:4]}-{day_string[4:6]}-{day_string[6:8]}T16:15:00",
                    f"{day_string[:4]}-{day_string[4:6]}-{day_string[6:8]}T20:15:00"]

    if plot_type == "hourly_mean":
        
        # plot all selected hours in a single 3x2 figure for each variable.
        for var_plot in var_plot_names:
            print(f"Plotting 3x2 panel of spatial distribution of {var_plot} for {day_string}...")
            time_selections = [
                f"{day_string[:4]}-{day_string[4:6]}-{day_string[6:8]}T{hour}:00"
                for hour in hours
            ]
            plot_spatial_iwv_distribution_panel(day_string, elev_sel, var_plot, time_selections, Transparent_flag)


    elif plot_type == "time_steps":


        # loop on time steps to plot spatial distribution of IWV values and IWV deviation at each time step without averaging.
        for var_plot in var_plot_names:
            for time_sel in time_steps:
                plot_spatial_iwv_distribution(day_string, elev_sel, var_plot, time_sel, plot_type, Transparent_flag)


    else:
        raise ValueError("plot_type must be either 'hourly_mean' or 'time_steps')")






def prepare_site_datasets(day_string, elev_sel, var_plot, plot_type):
    """Read site datasets and define one shared normalization for the chosen variable."""
    site_names = ['bolzano', 'collalbo', 'lagonero']
    site_value_mins = []
    site_value_maxs = []
    site_datasets = []

    iwv_tick_step = VAR_DICT['iwv'].get('tick_step', 5.0)
    iwv_dev_tick_step = VAR_DICT['IWV_deviation'].get('tick_step', 1.0)

    for site_name in site_names:
        path_root = f"/data/obs/campaigns/teamx/{site_name}/{MWR_SITES_NAMES[site_name]}/actris/level2/{day_string[:4]}/{day_string[4:6]}/{day_string[6:8]}/"
        ds_site = read_iwv_elev(site_name, day_string, 'iwv', elev_sel, path_root)

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

        site_datasets.append(ds_site)

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
        figsize=(15, 10.5),
        subplot_kw={'projection': ccrs.PlateCarree()},
    )
    fig.suptitle(
        pd.Timestamp(time_selections[0]).strftime('%Y-%m-%d'),
        fontsize=24,
        y=0.95,
    )
    fig.subplots_adjust(left=0.035, right=0.84, bottom=0.045, top=0.91, wspace=0.035, hspace=0.07)

    inset_size_deg = 0.05
    terrain = None
    mesh = None

    for index, (ax, time_sel) in enumerate(zip(axes.flat, time_selections)):
        terrain = configure_map_axis(ax, domain_ACTA, ds_orography)
        ax.set_title(pd.Timestamp(time_sel).strftime('%H:%M UTC'), fontsize=20)
        ax.set_xticks([])
        ax.set_yticks([])

        wrax_bolzano = create_site_inset(ax, site_lons[0], site_lats[0], inset_size_deg)
        wrax_collalbo = create_site_inset(ax, site_lons[1], site_lats[1], inset_size_deg)
        wrax_lagonero = create_site_inset(ax, site_lons[2], site_lats[2], inset_size_deg)

        plot_iwv_ring_on_map(wrax_bolzano, site_names[0], site_datasets[0], day_string, elev_sel, time_sel, var_plot=var_plot, update_limits=False)
        plot_iwv_ring_on_map(wrax_collalbo, site_names[1], site_datasets[1], day_string, elev_sel, time_sel, var_plot=var_plot, update_limits=False)
        mesh = plot_iwv_ring_on_map(wrax_lagonero, site_names[2], site_datasets[2], day_string, elev_sel, time_sel, var_plot=var_plot, update_limits=False)

    top_row_boxes = [ax.get_position() for ax in axes[0, :]]
    bottom_row_boxes = [ax.get_position() for ax in axes[1, :]]
    cbar_x0 = max(box.x1 for box in top_row_boxes) + 0.01
    cbar_width = 0.012

    top_row_y0 = min(box.y0 for box in top_row_boxes)
    top_row_y1 = max(box.y1 for box in top_row_boxes)
    bottom_row_y0 = min(box.y0 for box in bottom_row_boxes)
    bottom_row_y1 = max(box.y1 for box in bottom_row_boxes)

    cax_var = fig.add_axes([cbar_x0, top_row_y0, cbar_width, top_row_y1 - top_row_y0])
    cbar = fig.colorbar(mesh, cax=cax_var)
    cbar.set_label(VAR_DICT[var_plot]['label'], fontsize=20)
    max_labels = 7 if var_plot == 'iwv' else None
    colorbar_ticks = build_colorbar_ticks(
        var_plot,
        max_labels=max_labels,
        exact_labels=(var_plot == 'iwv'),
    )
    cbar.set_ticks(colorbar_ticks)
    cbar.ax.tick_params(labelsize=18)

    cax_orog = fig.add_axes([cbar_x0, bottom_row_y0, cbar_width, bottom_row_y1 - bottom_row_y0])
    cbar_orog = fig.colorbar(terrain, cax=cax_orog)
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

