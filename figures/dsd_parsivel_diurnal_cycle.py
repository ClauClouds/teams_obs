"""
With this code, we calculate the mean dsd from all days of the campaign that are available in the parsivel data, and plot the diurnal cycle of the mean dsd for each day. We also plot the diurnal cycle of the mean dsd for each day separately, to see if there are any differences between the days.
we derive the mean dsd over the 2h intervals defined in data_info.py, and plot the diurnal cycle of the mean dsd for all days one in each suplot of the time interval.
We then overplot the convective mean DSD and the MOBL_T dsd over the same time intervals, to see if there are any differences between the convective and non-convective days.
"""

from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
import xarray as xr
import site
from turtle import mode

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.projections.polar import PolarAxes
from readers.parsivel import read_parsivel
from readers.data_info import PLOT_SITES_NAMES, MWR_SITES_NAMES, site_lats, site_lons
from readers.MWR import read_iwv_elev
from figures.plot_settings import VAR_DICT
import numpy as np
import pandas as pd
import xarray as xr
from figures.utils import calc_iwv_deviation, create_site_inset, find_all_files_for_site, plot_iwv_ring_on_map, read_file_list_for_mode
from readers.data_info import orography_path, iop_conv_days, iop_MoBL_T_days, hours_diurnal_cycle_calc, azimuth_bins
import os
import pdb


def smooth_dsd_profile(profile, window_size=3):
    if window_size <= 1:
        return np.asarray(profile, dtype=float)

    kernel = np.ones(window_size, dtype=float)
    profile = np.asarray(profile, dtype=float)
    valid_mask = np.isfinite(profile)

    summed_values = np.convolve(np.where(valid_mask, profile, 0.0), kernel, mode="same")
    valid_counts = np.convolve(valid_mask.astype(float), kernel, mode="same")

    smoothed_profile = np.full(profile.shape, np.nan, dtype=float)
    np.divide(summed_values, valid_counts, out=smoothed_profile, where=valid_counts > 0)
    return smoothed_profile


def main():

    # colors for locations from matplotlib colors
    col_LN = "mediumblue"
    col_CB = "purple"
    col_BZ = "orchid"


    def format_hour_interval(hour_labels, index):
        start_time = pd.to_datetime(hour_labels[index], format="%H:%M")
        if index < len(hour_labels) - 1:
            end_time = pd.to_datetime(hour_labels[index + 1], format="%H:%M")
        else:
            step = pd.to_datetime(hour_labels[-1], format="%H:%M") - pd.to_datetime(hour_labels[-2], format="%H:%M")
            end_time = pd.to_datetime(hour_labels[-1], format="%H:%M") + step
        return f"{start_time.strftime('%H:%M')}-{end_time.strftime('%H:%M')}"
        

    sites = ['Schwartzseespitze', 'Klobenstein', 'Bozen'] # select site to plot
    dsd_smoothing_window = 3
    parsivel_patterns = {
        'Schwartzseespitze': 'PARS2020M_Schwarzseespitze',
        'Klobenstein': 'PARS2020L_Klobenstein',
        'Bozen': 'PARS2020A_Bozen',
    }
    site_colors = {
        'Schwartzseespitze': col_LN,
        'Klobenstein': col_CB,
        'Bozen': col_BZ,
    }
    site_counts = {}
    for site in sites:
            
        parsivel_path= f"/data/campaigns/teamx/kit_data/Parsivel_netcdf/"
        print(f"Processing site: {site}")

        # Use the site-specific instrument code and on-disk site spelling.
        file_list, n_files = find_all_files_for_site(
            parsivel_path,
            parsivel_patterns[site],
            site,
            ".nc.gz",
        )
        
        print(f"Found {n_files} parsivel files for site {site}.")
        if n_files == 0:
            raise ValueError(f"No Parsivel files found for site {site} under {parsivel_path}.")

        print(f"First file: {file_list[0]}, {site}")

        first_valid_ds = None
        for filename in file_list:
            candidate_ds = read_parsivel(filename)
            if candidate_ds is None or 'number_concentration' not in candidate_ds:
                if candidate_ds is not None:
                    candidate_ds.close()
                continue
            first_valid_ds = candidate_ds
            break

        if first_valid_ds is None:
            raise ValueError(f"Could not read any valid Parsivel dataset for site {site}.")

        n_diameter_bins = int(first_valid_ds.number_concentration.shape[-1])
        if 'diameter' in first_valid_ds.coords:
            diameter_values = first_valid_ds['diameter'].values
        else:
            diameter_values = np.arange(n_diameter_bins)

        if len(diameter_values) == n_diameter_bins + 1:
            diameter_plot = diameter_values[:-1]
        else:
            diameter_plot = diameter_values[:n_diameter_bins]

        first_valid_ds.close()

        # loop on files to read parsivel data and store data in
        dsd_matrix = np.full((n_files, len(hours_diurnal_cycle_calc[3:]), n_diameter_bins), np.nan)
        dsd_matrix_conv = np.full((n_files, len(hours_diurnal_cycle_calc[3:]), n_diameter_bins), np.nan)
        dsd_matrix_MoBL_T = np.full((n_files, len(hours_diurnal_cycle_calc[3:]), n_diameter_bins), np.nan)
        
        # loop on all files to read parsivel data and store mean dsd for each time interval in the corresponding array
        for i_file, filename in enumerate(file_list):

            print(f"Reading file {i_file+1}/{n_files}: {filename}")
            ds = read_parsivel(filename)
            if ds is None:
                print(f"Could not read file {filename}, skipping.")
                continue
            
            # check if it is raining at all during the day, if not we skip the file
            rr = ds.rr.values # rain rate in mm/h
            if np.all(rr == 0):

                print(f"No rain during the day in file {filename}, skipping.")
                ds.close()
                continue

            else: 

                print(f"Rain detected during the day in file {filename}, processing.")

                # extract day string from filename 
                day = filename.split("/")[-1].split(".")[0].split("_")[0]

                # define time intervals for the diurnal cycle calculation as strings in the format "YYYY-MM-DDTHH:MM:SS" for each day and each hour of the diurnal cycle calculation
                interval_starts = [f"{day[:4]}-{day[4:6]}-{day[6:8]}T{hour}:00" for hour in hours_diurnal_cycle_calc[3:]]
                next_day = pd.Timestamp(day) + pd.Timedelta(days=1)
                interval_ends = interval_starts[1:] + [f"{next_day.strftime('%Y-%m-%d')}T00:00:00"]
                
                # calculate mean dsd for each time interval and store it in the corresponding array
                dsd_mean = np.full((len(interval_starts), n_diameter_bins), np.nan)
                dsd_mean_conv = np.full((len(interval_starts), n_diameter_bins), np.nan)
                dsd_mean_MoBL_T = np.full((len(interval_starts), n_diameter_bins), np.nan)
                
                # loop on time intervals
                for i, time_sel in enumerate(hours_diurnal_cycle_calc[3:]):

                    # slice dataset for the time selection 
                    ds_time_sel = ds.sel(time=slice(interval_starts[i], interval_ends[i]))
                    if ds_time_sel.sizes.get('time', 0) == 0:
                        continue

                    # The stored DSD is in log10 units, so convert back to linear space before averaging.
                    number_concentration_linear = xr.where(
                        np.isfinite(ds_time_sel.number_concentration),
                        10.0 ** ds_time_sel.number_concentration,
                        np.nan,
                    )
                    mean_dsd = number_concentration_linear.mean(dim="time", skipna=True)
                    dsd_mean[i, :] = mean_dsd.values

                    # if day is in one of the lists, we store it in the corresponding array
                    if day in iop_conv_days:
                        dsd_mean_conv[i, :] = mean_dsd.values
                    if day in iop_MoBL_T_days:
                        dsd_mean_MoBL_T[i, :] = mean_dsd.values 

                # store mean dsd for the day in the corresponding array
                dsd_matrix[i_file, :, :] = dsd_mean
                dsd_matrix_conv[i_file, :, :] = dsd_mean_conv
                dsd_matrix_MoBL_T[i_file, :, :] = dsd_mean_MoBL_T
                ds.close()

        if np.all(np.isnan(dsd_matrix)):
            raise ValueError(f"No rainy Parsivel files with valid DSD data found for site {site}.")
        
        # calculate mean dsd over all days for each time interval
        dsd_mean_all = np.nanmean(dsd_matrix, axis=0)
        dsd_mean_conv_all = np.nanmean(dsd_matrix_conv, axis=0)
        dsd_mean_MoBL_T_all = np.nanmean(dsd_matrix_MoBL_T, axis=0)

        site_counts[site] = {
            "all": int(np.sum(np.any(np.isfinite(dsd_matrix), axis=(1, 2)))),
            "convective": int(np.sum(np.any(np.isfinite(dsd_matrix_conv), axis=(1, 2)))),
            "MoBL_T": int(np.sum(np.any(np.isfinite(dsd_matrix_MoBL_T), axis=(1, 2)))),
        }

        # store data in xarray and save to file ncdf
        dsd_mean_all_xr = xr.DataArray(dsd_mean_all, coords=[hours_diurnal_cycle_calc[3:], diameter_plot], dims=["time_interval", "diameter"])
        dsd_mean_conv_all_xr = xr.DataArray(dsd_mean_conv_all, coords=[hours_diurnal_cycle_calc[3:], diameter_plot], dims=["time_interval", "diameter"])
        dsd_mean_MoBL_T_all_xr = xr.DataArray(dsd_mean_MoBL_T_all, coords=[hours_diurnal_cycle_calc[3:], diameter_plot], dims=["time_interval", "diameter"])
        ds_out = xr.Dataset({
            "dsd_mean_all": dsd_mean_all_xr,
            "dsd_mean_conv_all": dsd_mean_conv_all_xr,
            "dsd_mean_MoBL_T_all": dsd_mean_MoBL_T_all_xr
        })
        output_dir = "data/diurnal_cycle/"
        os.makedirs(output_dir, exist_ok=True)
        ds_out.to_netcdf(os.path.join(output_dir, f"dsd_diurnal_cycle_dsd_{site}.nc"))


    # read data from both files
    ds_lagonero = xr.open_dataset("data/diurnal_cycle/dsd_diurnal_cycle_dsd_Schwartzseespitze.nc")
    ds_collalbo = xr.open_dataset("data/diurnal_cycle/dsd_diurnal_cycle_dsd_Klobenstein.nc")
    ds_bolzano = xr.open_dataset("data/diurnal_cycle/dsd_diurnal_cycle_dsd_Bozen.nc")

    # plot dsds for each time interval in a subplot, with the convective and MoBL_T mean dsds overplotted for both sites together
    # distinguish sites from linestile solid or dashed
    # set x and y axis to log scale, and set limits to the same for all subplots
    title_fontsize = 14
    axis_label_fontsize = 14
    tick_label_fontsize = 14
    legend_fontsize = 13
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10), sharex=True, sharey=True)
    for i, hour in enumerate(hours_diurnal_cycle_calc[3:]):

        ax = axes[i//3, i%3]

        for ds, site_key, site_name in zip(
            [ds_lagonero, ds_collalbo, ds_bolzano],
            ["Schwartzseespitze", "Klobenstein", "Bozen"],
            ["Schwartzseespitze", "Klobenstein", "Bozen"],
        ):
            site_count = site_counts[site_key]
            site_color = site_colors[site_key]
            dsd_all_smoothed = smooth_dsd_profile(ds.dsd_mean_all[i, :].values, dsd_smoothing_window)
            dsd_conv_smoothed = smooth_dsd_profile(ds.dsd_mean_conv_all[i, :].values, dsd_smoothing_window)
            dsd_mobl_t_smoothed = smooth_dsd_profile(ds.dsd_mean_MoBL_T_all[i, :].values, dsd_smoothing_window)
            line_all, = ax.plot(
                diameter_plot,
                dsd_all_smoothed,
                label=f"{site_name} - All days (N={site_count['all']})",
                color=site_color,
                linewidth=4,
                linestyle="solid",
            )
            line_conv, = ax.plot(
                diameter_plot,
                dsd_conv_smoothed,
                label=f"{site_name} - Convective (N={site_count['convective']})",
                color=site_color,
                linewidth=4,
                linestyle="dashed",
            )
            line_mobl_t, = ax.plot(
                diameter_plot,
                dsd_mobl_t_smoothed,
                label=f"{site_name} - MoBL_T (N={site_count['MoBL_T']})",
                color=site_color,
                linewidth=4,
                linestyle="dotted",
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"Time interval: {format_hour_interval(hours_diurnal_cycle_calc[3:], i)}", fontsize=title_fontsize)
        if i // 3 == 2:
            ax.set_xlabel("Diameter (mm)", fontsize=axis_label_fontsize)
        else:
            ax.set_xlabel("")
        if i % 3 == 0:
            ax.set_ylabel("Num conc [1/m^3/mm]", fontsize=axis_label_fontsize)
        else:
            ax.set_ylabel("")
        ax.tick_params(axis="both", labelsize=tick_label_fontsize, width=1.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.5)
        ax.spines["bottom"].set_linewidth(1.5)

    site_legend_handles = [
        Line2D(
            [0],
            [0],
            color=site_colors[site_name],
            linewidth=4,
            linestyle="solid",
            label=f"{site_name} (All: N={site_counts[site_name]['all']})",
        )
        for site_name in ["Schwartzseespitze", "Klobenstein", "Bozen"]
    ]
    category_legend_handles = [
        Line2D([0], [0], color="black", linewidth=4, linestyle="solid", label="All days"),
        Line2D([0], [0], color="black", linewidth=4, linestyle="dashed", label="Convective"),
        Line2D([0], [0], color="black", linewidth=4, linestyle="dotted", label="MoBL_T"),
    ]

    fig.legend(
        handles=site_legend_handles,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.055),
        frameon=False,
        fontsize=legend_fontsize,
        columnspacing=1.6,
        handlelength=2.8,
    )
    fig.legend(
        handles=category_legend_handles,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.02),
        frameon=False,
        fontsize=legend_fontsize,
        columnspacing=1.6,
        handlelength=2.8,
    )

    plt.tight_layout(rect=[0.0, 0.14, 1.0, 1.0])
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"dsd_diurnal_cycle.png"), dpi=300)
    
    
if __name__ == "__main__":
    main()
