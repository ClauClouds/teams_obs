import cartopy.crs as ccrs
import cartopy.feature as cfeature

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.projections.polar import PolarAxes
from readers.data_info import PLOT_SITES_NAMES, MWR_SITES_NAMES, site_lats, site_lons
from readers.MWR import read_iwv_elev
from figures.IWV_spatial import calc_iwv_deviation, extract_closest_scan
from figures.plot_settings import VAR_DICT
import numpy as np
import pandas as pd
import xarray as xr


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


def plot_map_azimuth_ring(ax, ds_scan, site, elev_sel, var_plot='IWV_deviation'):
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


def plot_iwv_ring_on_map(ax, site, day_string, elev_sel, time_sel, var_plot='IWV_deviation'):
    """Draw one azimuth-ring scan for a site on an existing map inset axis."""
    path_root = f"/data/obs/campaigns/teamx/{site}/{MWR_SITES_NAMES[site]}/actris/level2/{day_string[:4]}/{day_string[4:6]}/{day_string[6:8]}/"
    ds_iwv_elev = read_iwv_elev(site, day_string, 'iwv', elev_sel, path_root)

    if var_plot == 'IWV_deviation':
        ds_iwv_elev['IWV_deviation'] = calc_iwv_deviation(ds_iwv_elev)

    iwv_min = ds_iwv_elev.iwv.min().item()
    iwv_max = ds_iwv_elev.iwv.max().item()
    VAR_DICT['iwv']['vmin'] = np.floor(iwv_min / 2) * 2
    VAR_DICT['iwv']['vmax'] = np.ceil(iwv_max / 2) * 2

    if 'IWV_deviation' in ds_iwv_elev:
        iwv_dev_max = np.nanmean(np.abs(ds_iwv_elev.IWV_deviation.values))
        VAR_DICT['IWV_deviation']['vmin'] = -np.ceil(iwv_dev_max / 2) * 2
        VAR_DICT['IWV_deviation']['vmax'] = np.ceil(iwv_dev_max / 2) * 2

    ds_scan = extract_closest_scan(ds_iwv_elev, time_sel)
    return plot_map_azimuth_ring(ax, ds_scan, site, elev_sel, var_plot=var_plot)


def main():
    day_string = "20250625"
    elev_sel = 30
    time_sel = f"{day_string[:4]}-{day_string[4:6]}-{day_string[6:8]}T12:00:00"
    var_plot = 'IWV_deviation'
    orography_path = "/home/cacquist/Documents/GitHub/EXPATS/orography_expats_high_res.nc"

    # Coordinates of the station we were measuring windspeed
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

    # plot map of the teamx domain
    fig = plt.figure(figsize=(8, 6))
    # Plot data
    main_ax = plt.axes(projection=ccrs.PlateCarree())
    main_ax.set_title("TeamX domain", fontsize=16)
    # Add features
    main_ax.coastlines(resolution='10m', color='0.35', linestyle=':', linewidth=0.5)
    main_ax.add_feature(cfeature.BORDERS, linestyle=':', color='0.35', linewidth=0.5)
    main_ax.set_extent(domain_ACTA, crs=ccrs.PlateCarree())

    # plot high-resolution orography as the map background
    ds_orography = xr.open_dataset(orography_path)
    terrain = main_ax.pcolormesh(
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

    inset_size_deg = 0.05
    wrax_bolzano = create_site_inset(main_ax, lon_bolzano, lat_bolzano, inset_size_deg)
    wrax_collalbo = create_site_inset(main_ax, lon_collalbo, lat_collalbo, inset_size_deg)
    wrax_lagonero = create_site_inset(main_ax, lon_lagonero, lat_lagonero, inset_size_deg)

    plot_iwv_ring_on_map(wrax_bolzano, 'bolzano', day_string, elev_sel, time_sel, var_plot=var_plot)
    plot_iwv_ring_on_map(wrax_collalbo, 'collalbo', day_string, elev_sel, time_sel, var_plot=var_plot)
    mesh = plot_iwv_ring_on_map(wrax_lagonero, 'lagonero', day_string, elev_sel, time_sel, var_plot=var_plot)

    cbar = fig.colorbar(mesh, ax=main_ax, pad=0.02, shrink=0.75)
    cbar.set_label(VAR_DICT[var_plot]['label'], fontsize=10)
    tick_step = VAR_DICT[var_plot].get('tick_step', 2.0)
    cbar.set_ticks(np.arange(VAR_DICT[var_plot]['vmin'], VAR_DICT[var_plot]['vmax'] + tick_step, tick_step))
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

    savefig = True
    if savefig:
        fig.savefig("plots/map_windrose_IWV.png", dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    main()

