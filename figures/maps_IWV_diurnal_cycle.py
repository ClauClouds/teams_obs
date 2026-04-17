
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
import xarray as xr
from figures.utils import calc_iwv_deviation, create_site_inset, plot_iwv_ring_on_map, read_file_list_for_mode
from readers.data_info import orography_path, iop_conv_days, iop_MoBL_T_days, hours_diurnal_cycle_calc, azimuth_bins
import os
import pdb


def plot_mean_azimuth_ring(ax, azimuth_edges_deg, values, var_plot, vmin=None, vmax=None, n_color_bins=20):
    """Plot one mean azimuth ring from pre-aggregated values."""
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



def main():

    elevs = [10, 20, 30] # elevation angles to plot the diurnal cycle for, other options are 10, 20, 30,
    vars2plot = ["iwv", "IWV_deviation"] # other options are "lwp" and "IWV_deviation"
    sites = ["lagonero", "collalbo", "bolzano"] # sites to plot the insets for, other options are "bolzano" and "collalbo"
    
    # loop on elevation angles, variables to plot and sites
    for elev_sel in elevs:
        for var_plot in vars2plot:
            for site_name in sites:

                print(f"Processing site {site_name} for variable {var_plot} at elevation {elev_sel}°...")
                print("----------------------------------------------------------------")
                # define the time and variable to plot
                mode =   "MOBL_T_days" #"diurnal_cycle" #"convective_days" # or#depending on the selected days list
                interval_start_hours = hours_diurnal_cycle_calc
                panel_title_fontsize = 16
                direction_label_fontsize = 16
                colorbar_label_fontsize = 16
                colorbar_tick_fontsize = 14
                suptitle_fontsize = 20

                output_file_nc = f"data/diurnal_cycle/mean_{mode}_{var_plot}_{site_name}_elev_{elev_sel}.nc"
                figure_file = f"plots/dc_spatial_{mode}_{var_plot}_{site_name}_elev_{elev_sel}.png"

                if os.path.exists(output_file_nc):            
                    print(f"File {output_file_nc} already exists. Skipping")
                    print("----------------------------------------------------------------")

                else:
                    print(f"Processing {mode}...")
                    print("----------------------------------------------------------------")

                    # find all available files for that site to calculate the diurnal cycle: from the path root list all the files with string 
                    path_root = f"/data/obs/campaigns/teamx/{site_name}/{MWR_SITES_NAMES[site_name]}/actris/level2/"

                    # read file list    
                    file_found_list, N_stat = read_file_list_for_mode(path_root, site_name, mode, iop_conv_days, iop_MoBL_T_days)
                    if N_stat == 0:
                        print(f"No files found for site {site_name}, variable {var_plot}, elevation {elev_sel}°. Skipping combination.")
                        print("----------------------------------------------------------------")
                        continue
                    print(f"Found {N_stat} files for site {site_name} to process for {mode}.")
                    print("Processing files...")

                    # create empty matrices of nans to collect the mean values for each time selection of the diurnal cycle for each day
                    var_dc_matrix = np.empty((len(file_found_list), len(interval_start_hours), len(azimuth_bins)-1)) # matrix to collect the mean values for each time selection of the diurnal cycle for each day and each azimuth bin
                    var_dc_matrix.fill(np.nan) # matrix to collect the mean values for each time selection of the diurnal cycle for each day
                    counts_days = 0 # counter for the number of days with data available for the selected
                    valid_days = []

                    for i_file, file_full_path in enumerate(file_found_list):
                        
                        # extract day from file name of the type '/data/obs/campaigns/teamx/lagonero/tophat/actris/level2/2025/09/09/MWR_single_lagonero_20250909.nc'
                        day = file_full_path.split("/")[-1].split(".")[0].split("_")[-1]

                        print(f"Processing day {day}...") 

                        # set hours to plot and time steps array for single plotting without averages 6,8,10,12,14,16
                        interval_starts = [f"{day[:4]}-{day[4:6]}-{day[6:8]}T{hour}:00" for hour in interval_start_hours]
                        next_day = pd.Timestamp(day) + pd.Timedelta(days=1)
                        interval_ends = interval_starts[1:] + [f"{next_day.strftime('%Y-%m-%d')}T00:00:00"]
                        print("**************************")

                        # split path from filename
                        path_file_sel = "/".join(file_full_path.split("/")[:-1]) + "/"
                        filename = file_full_path.split("/")[-1]

                        # read the dataset for the day and the selected variable and elevation
                        ds_site = read_iwv_elev(site_name, day, var_plot, elev_sel, path_file_sel)

                        if len(ds_site.time.values) == 0:
                            print(f"No data available for day {day} at elevation {elev_sel} for site {site_name}. Skipping.")
                            print("-------------------------------- --------------------------------")

                            continue    
                        
                        day_index = counts_days

                        # calculate means over the selected time selections for the diurnal cycle
                        for i, time_sel in enumerate(interval_starts):

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
                                var_dc_matrix[day_index, i, :] = ds_time_sel.groupby_bins("azimuth_angle", azimuth_bins).mean(dim="time", skipna=True)[var_plot].values
                            except Exception as e:
                                print(f"Error calculating mean for time selection {interval_starts[i]}-{interval_ends[i]} for day {day}: {e}")
                                print(f"Skipping time selection {time_sel} for day {day}.")
                                print("-------------------------------- --------------------------------")
                                continue
                        
                        valid_days.append(day)
                        counts_days += 1
                        print("----------------------------------------------------------------")

                    if counts_days == 0:
                        print(
                            f"No valid data available to calculate {mode} for site {site_name}, "
                            f"variable {var_plot}, elevation {elev_sel}°. Skipping combination."
                        )
                        print("----------------------------------------------------------------")
                        continue

                    print(counts_days)
                    valid_var_dc_matrix = var_dc_matrix[:counts_days]

                    # calculate mean over the valid days for each time selection of the diurnal cycle and each azimuth bin
                    valid_counts = np.sum(~np.isnan(valid_var_dc_matrix), axis=0)
                    mean_dc_matrix = np.full(valid_counts.shape, np.nan)
                    np.divide(
                        np.nansum(valid_var_dc_matrix, axis=0),
                        valid_counts,
                        out=mean_dc_matrix,
                        where=valid_counts > 0,
                    )


                    # store matrix of mean values in xarray dataset
                    ds_diurnal_cycle = xr.Dataset(
                        {
                            var_plot: (['day', 'time_selection', 'azimuth_bin'], valid_var_dc_matrix),
                            f"mean_{var_plot}": (['time_selection', 'azimuth_bin'], mean_dc_matrix),
                            f"count_{var_plot}": (['time_selection', 'azimuth_bin'], valid_counts.astype(int)),
                        },      
                        coords={
                            'day': valid_days,
                            'time_selection': interval_start_hours,
                            'azimuth_bin': [f"{azimuth_bins[i]}-{azimuth_bins[i+1]}" for i in range(len(azimuth_bins)-1)]
                        }
                    )   

                    # creat output directory if it does not exist
                    output_dir = "data/diurnal_cycle/"
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    # save dataset of the diurnal cycle mean values for each day and each azimuth bin in a netcdf file
                    ds_diurnal_cycle.to_netcdf(output_file_nc)
                    print(f"Saved diurnal cycle dataset to {output_file_nc}")


                # if ncdf file has been stored, plot a figure with 9 subplots each containing a windrose map of the mean values for each 
                # time selection of the diurnal cycle and each azimuth bin, with a title indicating the time selection 
                # and the number of valid days used to calculate the mean for that time selection

                if not os.path.exists(output_file_nc):
                    print(
                        f"Output dataset {output_file_nc} is not available for site {site_name}, "
                        f"variable {var_plot}, elevation {elev_sel}°. Skipping plot."
                    )
                    print("----------------------------------------------------------------")
                    continue

                print(f"Plotting figure {figure_file}...")
                ds_diurnal_cycle = xr.open_dataset(output_file_nc)
                os.makedirs(os.path.dirname(figure_file), exist_ok=True)

                # set color scale from the plotted mean values only
                mean_values_all = ds_diurnal_cycle[f"mean_{var_plot}"].values
                if np.all(np.isnan(mean_values_all)):
                    print(
                        f"All plotted mean values are NaN for site {site_name}, variable {var_plot}, "
                        f"elevation {elev_sel}°. Skipping plot."
                    )
                    ds_diurnal_cycle.close()
                    print("----------------------------------------------------------------")
                    continue

                colorbar_vmin = float(np.nanmin(mean_values_all))
                colorbar_vmax = float(np.nanmax(mean_values_all))
                if np.isclose(colorbar_vmin, colorbar_vmax):
                    colorbar_vmin -= 0.5
                    colorbar_vmax += 0.5

                # plot figure with 9 subplots each containing a windrose map of the mean values for each time selection of the diurnal cycle and each azimuth bin, with a title indicating the time selection and the number of valid days used to calculate the mean for that time selection
                fig, axes = plt.subplots(3, 3, figsize=(15, 15), subplot_kw={'projection': 'polar'})
                axes = axes.flatten()
                mesh = None
                azimuth_edges = azimuth_bins

                for i, time_sel in enumerate(interval_start_hours):
                    ax = axes[i]
                    mean_values = ds_diurnal_cycle[f"mean_{var_plot}"].sel(time_selection=time_sel).values
                    count_values = ds_diurnal_cycle[f"count_{var_plot}"].sel(time_selection=time_sel).values
                    ax.tick_params(axis='x', labelsize=direction_label_fontsize)

                    mesh = plot_mean_azimuth_ring(
                        ax,
                        azimuth_edges,
                        mean_values,
                        var_plot,
                        vmin=colorbar_vmin,
                        vmax=colorbar_vmax,
                    )

                    positive_counts = count_values[count_values > 0]
                    if len(positive_counts) == 0:
                        count_label = "N=0"
                    elif np.all(positive_counts == positive_counts[0]):
                        count_label = f"N={int(positive_counts[0])}"
                    else:
                        count_label = f"N={int(np.nanmin(positive_counts))}-{int(np.nanmax(positive_counts))}"

                    ax.set_title(f"{time_sel} UTC\n{count_label}", fontsize=panel_title_fontsize)

                for ax in axes[len(hours_diurnal_cycle_calc[:-1]):]:
                    ax.set_axis_off()


                fig.subplots_adjust(left=0.06, right=0.86, bottom=0.06, top=0.90, wspace=0.28, hspace=0.30)

                if mesh is not None:
                    cax = fig.add_axes([0.89, 0.18, 0.025, 0.62])
                    cbar = fig.colorbar(mesh, cax=cax, orientation='vertical')
                    cbar.set_label(VAR_DICT[var_plot]['label'], fontsize=colorbar_label_fontsize)
                    colorbar_ticks = np.linspace(colorbar_vmin, colorbar_vmax, 6)
                    cbar.set_ticks(colorbar_ticks)
                    cbar.ax.tick_params(labelsize=colorbar_tick_fontsize)

                plt.suptitle(f"Mean {VAR_DICT[var_plot]['label']} at {PLOT_SITES_NAMES[site_name]} - Elevation Scan {elev_sel}° - {mode.replace('_', ' ').title()}", fontsize=suptitle_fontsize)
                plt.savefig(figure_file)
                plt.close(fig)
                ds_diurnal_cycle.close()
                print(f"Saved figure to {figure_file}")



if __name__ == "__main__":
    main()