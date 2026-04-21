"""
Plot the anomalies of variables for convective days and moblt days with respect to diurnal cycle patterns

"""



from datetime import time
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
from figures.utils import find_closest_dc_value, read_file_list_for_mode, calculate_mean_anomaly_for_time_selection, plot_mean_azimuth_ring, get_shared_colorbar_limits, get_regular_integer_colorbar_spec
from readers.data_info import orography_path, iop_conv_days, iop_MoBL_T_days, hours_diurnal_cycle_calc, azimuth_bins
import os
import pdb

def main():

    elevs = [30]#[10, 20, 30] # elevation angles to plot the diurnal cycle for, other options are 10, 20, 30,
    vars2plot = ["iwv", "IWV_deviation"] # other options are "lwp" and "IWV_deviation"
    sites = ["lagonero", "collalbo", "bolzano"] # sites to plot the insets for, other options are "bolzano" and "collalbo"
    mode = "convective_days" # other option is ""MOBL_T_days"# "
    iwv_colorbar_scale_factor = 1.25
    dc_hours = np.array([pd.to_datetime(hour, format="%H:%M").hour for hour in hours_diurnal_cycle_calc[3:]]) # array of the hours of the diurnal cycle to plot
    # loop on elevation angles, variables to plot and sites
    for elev_sel in elevs:
        for var_plot in vars2plot:
            for site_name in sites:
                
                # check if anomaly file already exists, if yes skip the calculation
                output_path = f"/home/cacquist/Documents/GitHub/EXPATS/teams_obs/data/anomalies/mean_anomaly_{mode}_{var_plot}_{site_name}_elev_{elev_sel}.nc"
                if os.path.exists(output_path):
                    print(f"Anomaly file already exists for site {site_name}, variable {var_plot}, elevation {elev_sel}°. Skipping combination.")
                    print("----------------------------------------------------------------")
                    continue
                
                #  read file list    
                path_root = f"/data/obs/campaigns/teamx/{site_name}/{MWR_SITES_NAMES[site_name]}/actris/level2/"
                file_found_list, N_stat = read_file_list_for_mode(path_root, site_name, mode, iop_conv_days, iop_MoBL_T_days)

                if N_stat == 0:
                    print(f"No files found for site {site_name}, variable {var_plot}, elevation {elev_sel}°. Skipping combination.")
                    print("----------------------------------------------------------------")
                    continue
                print("Processing files...")
            

                # read the file with the mean diurnal cycle of the campaign for the selected variable, angle, and site
                try:
                    dc_data = xr.open_dataset(f"/home/cacquist/Documents/GitHub/EXPATS/teams_obs/data/diurnal_cycle/mean_diurnal_cycle_{var_plot}_{site_name}_elev_{elev_sel}.nc") 
                except FileNotFoundError:
                    print(f"Mean diurnal cycle file not found for site {site_name}, variable {var_plot}, elevation {elev_sel}°. Skipping combination.")
                    print("----------------------------------------------------------------")
                    continue

                # read mean anomalies
                var_name = "mean_"+var_plot
                dc_var = dc_data[var_name].values

                # loop on all iop days of the mode and plot the anomalies for each day
                # create empty matrices of nans to collect the mean values for each time selection of the diurnal cycle for each day
                anomaly_dc_matrix = np.empty((len(file_found_list), len(hours_diurnal_cycle_calc[3:]), len(azimuth_bins)-1)) # matrix to collect the mean values for each time selection of the diurnal cycle for each day and each azimuth bin
                anomaly_dc_matrix.fill(np.nan) # matrix to collect the mean values for each time selection of the diurnal cycle for each day

                # loop on iop mode days 
                for i_file, file_full_path in enumerate(file_found_list):
                    
                    # extract day from file name of the type '/data/obs/campaigns/teamx/lagonero/tophat/actris/level2/2025/09/09/MWR_single_lagonero_20250909.nc'
                    day = file_full_path.split("/")[-1].split(".")[0].split("_")[-1]

                    # split path from filename
                    path_file_sel = "/".join(file_full_path.split("/")[:-1]) + "/"
                    filename = file_full_path.split("/")[-1]

                    # read the dataset for the day and the selected variable and elevation
                    ds_site = read_iwv_elev(site_name, day, var_plot, elev_sel, path_file_sel)

                    # loop on time stamps of the day 
                    anomaly = np.empty(ds_site.time.size) # array to collect the anomaly values for each time step of the file
                    anomaly.fill(np.nan) # initialize the anomaly array with nans
                    for i_time, time_sel in enumerate(ds_site.time.values):
                        
                        # skip all time stamps below the hour of the first hour of the diurnal cycle    
                        if pd.to_datetime(time_sel).hour < dc_hours[0]:
                            print(f"Skipping time step {time_sel} because it is below the first hour of the diurnal cycle ({hours_diurnal_cycle_calc[0]}h).")
                            continue

                        # read the variable values for the selected time step and calculate the deviation from the diurnal cycle mean value at the same time step and azimuth values
                        ds_sel = ds_site.sel(time=time_sel)
                        var_sel = ds_sel[var_plot].values
                        azimuth_sel = ds_sel.azimuth_angle.values
                        hour_sel = pd.to_datetime(time_sel).hour

                        # find the closest diurnal cycle hour and azimuth bin to the selected time step and azimuth value
                        dc_sel_value = find_closest_dc_value(hour_sel, azimuth_sel, hours_diurnal_cycle_calc[3:], azimuth_bins, dc_var)

                        # calculate anomaly as the deviation from the diurnal cycle mean value at the same time step and azimuth values
                        anomaly[i_time] = var_sel - dc_sel_value # calculate the anomaly as the deviation from the diurnal cycle mean value at the same time step and azimuth values

                    # add the anomaly array to the ds_site dataset as a new variable
                    ds_site["anomaly"] = (("time"), anomaly)    

                    # calculate mean anomaly over the selected time selections and azimuth bins
                    anomaly_dc_matrix[i_file, :, :] = calculate_mean_anomaly_for_time_selection(ds_site, day, hours_diurnal_cycle_calc[3:], azimuth_bins, var_plot) # calculate the mean anomaly for each time selection of the diurnal cycle and each azimuth bin and add it to the matrix
                mean_anomaly = np.nanmean(anomaly_dc_matrix, axis=0)
                count_days = np.sum(~np.isnan(anomaly_dc_matrix), axis=0).astype(int)
                print(
                    f"Finished {site_name}, {var_plot}, {elev_sel}deg. "
                    f"Mean anomaly matrix shape: {mean_anomaly.shape}"
                )

                # store mean anomaly in a new netcdf file
                ds_mean_anomaly = xr.Dataset(
                    {
                        "mean_anomaly": (("hour", "azimuth_bin"), mean_anomaly),
                        "count_days": (("hour", "azimuth_bin"), count_days),
                    },  
                    coords={
                        "hour": hours_diurnal_cycle_calc[3:],
                        "azimuth_bin": azimuth_bins[:-1] + np.diff(azimuth_bins)/2 # calculate the center of the azimuth bins
                    }
                )
                output_path = f"/home/cacquist/Documents/GitHub/EXPATS/teams_obs/data/anomalies/mean_anomaly_{mode}_{var_plot}_{site_name}_elev_{elev_sel}.nc"
                # create output directory if it does not exist
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                ds_mean_anomaly.to_netcdf(output_path)
                print(f"Saved anomaly dataset to {output_path}")

    for var_plot in vars2plot:
        for site_name in sites:
            anomaly_paths = [
                f"/home/cacquist/Documents/GitHub/EXPATS/teams_obs/data/anomalies/mean_anomaly_{mode}_{var_plot}_{site_name}_elev_{elev_sel}.nc"
                for elev_sel in elevs
            ]
            colorbar_vmin, colorbar_vmax = get_shared_colorbar_limits(
                anomaly_paths,
                "mean_anomaly",
                var_plot,
                symmetric=True,
                scale_factor=iwv_colorbar_scale_factor if var_plot == "iwv" else 1.0,
            )
            colorbar_vmin, colorbar_vmax, colorbar_ticks = get_regular_integer_colorbar_spec(
                colorbar_vmin,
                colorbar_vmax,
            )

            for elev_sel in elevs:
                input_path = f"/home/cacquist/Documents/GitHub/EXPATS/teams_obs/data/anomalies/mean_anomaly_{mode}_{var_plot}_{site_name}_elev_{elev_sel}.nc"
                if not os.path.exists(input_path):
                    print(f"Mean anomaly file not found for site {site_name}, variable {var_plot}, elevation {elev_sel}°. Skipping plotting.")
                    print("----------------------------------------------------------------")
                    continue

                print(f"Plotting mean anomaly for site {site_name}, variable {var_plot}, elevation {elev_sel}°.")
                figure_file = f"plots/poster_plots/anomalies_spatial_{mode}_{var_plot}_{site_name}_elev_{elev_sel}.png"
                plot_output_path = f"/home/cacquist/Documents/GitHub/EXPATS/teams_obs/" + figure_file
                os.makedirs(os.path.dirname(plot_output_path), exist_ok=True)

                ds_mean_anomaly = xr.open_dataset(input_path)
                panel_title_fontsize = 22
                direction_label_fontsize = 22
                colorbar_label_fontsize = 22
                colorbar_tick_fontsize = 22
                suptitle_fontsize = 22

                fig, axes = plt.subplots(3, 3, figsize=(15, 15), subplot_kw={'projection': 'polar'})
                axes = axes.flatten()
                mesh = None
                azimuth_edges = azimuth_bins

                for i, time_sel in enumerate(hours_diurnal_cycle_calc[3:]):
                    ax = axes[i]
                    mean_values = ds_mean_anomaly.mean_anomaly.sel(hour=time_sel).values
                    ax.tick_params(axis='x', labelsize=direction_label_fontsize)

                    mesh = plot_mean_azimuth_ring(
                        ax,
                        azimuth_edges,
                        mean_values,
                        "anomalies",
                        vmin=colorbar_vmin,
                        vmax=colorbar_vmax,
                    )

                    # position the title at the left of the subplot in line with N direction
                    ax.set_title(f"{time_sel} UTC", fontsize=panel_title_fontsize, loc='left')

                for ax in axes[len(hours_diurnal_cycle_calc[3:]):]:
                    ax.set_axis_off()

                fig.subplots_adjust(left=0.06, right=0.86, bottom=0.06, top=0.90, wspace=0.28, hspace=0.30)

                if mesh is not None:
                    cax = fig.add_axes([0.89, 0.18, 0.025, 0.62])
                    cbar = fig.colorbar(mesh, cax=cax, orientation='vertical')
                    cbar.set_label(VAR_DICT[var_plot]['label'], fontsize=colorbar_label_fontsize)
                    cbar.set_ticks(colorbar_ticks)
                    cbar.ax.tick_params(labelsize=colorbar_tick_fontsize)

                # in bold, add a suptitle with the variable name, site name, elevation angle and mode of days plotted
                plt.suptitle(f"Mean anomaly of {VAR_DICT[var_plot]['label']} - {PLOT_SITES_NAMES[site_name]} - {elev_sel}° elev - {mode.replace('_', ' ').title()}", fontsize=suptitle_fontsize, fontweight='bold')
                plt.savefig(plot_output_path)
                plt.close(fig)
                ds_mean_anomaly.close()
                print(f"Saved figure to {plot_output_path}")
                plt.figure(figsize=(10, 6))

if __name__ == "__main__":
    main()
