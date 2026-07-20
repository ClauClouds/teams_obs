"""
code to time series of rain occurrence for the three sites and for all data, for convective days, and for MOBL-T days. 
Returns:
    it produces two plots:
    - one with the percentage of rain occurrence (rainy time stamps / total time stamps in each 2h time bin) 
    for the three sites, for all data, for convective days, and for MOBL-T days. 
    - one with the mean rain duration over 2 h time bins (only considering rainy time stamps)
      for the three sites, for all data, for convective days, and for MOBL-T days.
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
from readers.MWR import read_MWR_flags, read_iwv_elev
from figures.plot_settings import VAR_DICT
import numpy as np
import pandas as pd
import re
import xarray as xr
from figures.utils import find_all_files_for_site, plot_teamx_sites
from readers.data_info import orography_path, iop_conv_days, iop_MoBL_T_days, hours_diurnal_cycle_calc, azimuth_bins, fci_path, coords_file_path, domain
import os
import pdb
import progressbar



def main():

    # read input files
    ds_collalbo = xr.open_dataset("data/mrr_ze_profiles/collalbo_diurnal_rain_occurrence_v2.nc")
    ds_lagonero = xr.open_dataset("data/mrr_ze_profiles/lagonero_diurnal_rain_occurrence_v2.nc")
    ds_bolzano = xr.open_dataset("data/mrr_ze_profiles/bolzano_diurnal_rain_occurrence_v2.nc")
    ds_collalbo_conv = xr.open_dataset("data/mrr_ze_profiles/collalbo_convective_rain_occurrence_v2.nc")
    ds_lagonero_conv = xr.open_dataset("data/mrr_ze_profiles/lagonero_convective_rain_occurrence_v2.nc")
    ds_bolzano_conv = xr.open_dataset("data/mrr_ze_profiles/bolzano_convective_rain_occurrence_v2.nc")
    ds_collalbo_mobl_t = xr.open_dataset("data/mrr_ze_profiles/collalbo_mobl_t_rain_occurrence_v2.nc")
    ds_lagonero_mobl_t = xr.open_dataset("data/mrr_ze_profiles/lagonero_mobl_t_rain_occurrence_v2.nc")
    ds_bolzano_mobl_t = xr.open_dataset("data/mrr_ze_profiles/bolzano_mobl_t_rain_occurrence_v2.nc")
    all_case_counts = {
        "collalbo": ds_collalbo.sizes["file"],
        "lagonero": ds_lagonero.sizes["file"],
        "bolzano": ds_bolzano.sizes["file"],
    }

    dict_rain_ratio, dict_mean = calc_rain_ratio(ds_collalbo, ds_lagonero, ds_bolzano)
    dict_rain_ratio_convective, dict_mean_convective = calc_rain_ratio(ds_collalbo_conv, ds_lagonero_conv, ds_bolzano_conv)
    dict_rain_ratio_mobl_t, dict_mean_mobl_t = calc_rain_ratio(ds_collalbo_mobl_t, ds_lagonero_mobl_t, ds_bolzano_mobl_t)

    filename_out = "/home/cacquist/Documents/GitHub/EXPATS/teams_obs/plots/rain_ratio_diurnal_cycle_all_sites_v3.png"
    ylabel="Rain probability [%] \n" \
    "(rainy time stamps / total time stamps \n" \
    " in each 2h time bin)"
    ylim = 60
    plot_rain_occ_diurnal(dict_rain_ratio,dict_rain_ratio_convective, dict_rain_ratio_mobl_t, pd.to_datetime(ds_collalbo.time_bin.values), filename_out, ylabel, ylim, all_case_counts)


    # read mean rain occurrence for each site and call plotting function to plot diurnal cycle of rain occurrence for the two sites
    dict_diurnal = {
        "collalbo": ds_collalbo,
        "lagonero": ds_lagonero,
        "bolzano": ds_bolzano
    }
    dict_convective = {
        "collalbo": ds_collalbo_conv,
        "lagonero": ds_lagonero_conv,
        "bolzano": ds_bolzano_conv
    }
    dict_mobl_t = {
        "collalbo": ds_collalbo_mobl_t,
        "lagonero": ds_lagonero_mobl_t,
        "bolzano": ds_bolzano_mobl_t
    }

    rain_occ_diurnal = read_rain_data(dict_diurnal)
    rain_occ_convective = read_rain_data(dict_convective)
    rain_occ_mobl_t = read_rain_data(dict_mobl_t)

    filename_out = "/home/cacquist/Documents/GitHub/EXPATS/teams_obs/plots/rain_occ_diurnal_cycle_all_sites_v2.png"
    ylabel2="Mean rain duration over 2 h [min] \n" \
    "(only rainy time stamps)"
    ylim=125
    plot_rain_occ_diurnal(rain_occ_diurnal, rain_occ_convective, rain_occ_mobl_t, pd.to_datetime(ds_collalbo.time_bin.values), filename_out, ylabel2, ylim, all_case_counts)



def calc_rain_ratio(ds_collalbo, ds_lagonero, ds_bolzano):

    # calculate mean of the datasets over the file dimension
    ds_collalbo_mean = ds_collalbo.mean(dim="file", skipna=True)
    ds_lagonero_mean = ds_lagonero.mean(dim="file", skipna=True)
    ds_bolzano_mean = ds_bolzano.mean(dim="file", skipna=True)

    # for each site, calculate rain counts / total counts 
    rain_ratio_collalbo = ds_collalbo_mean.rain_counts.values / ds_collalbo_mean.total_counts.values
    rain_ratio_lagonero = ds_lagonero_mean.rain_counts.values / ds_lagonero_mean.total_counts.values
    rain_ratio_bolzano = ds_bolzano_mean.rain_counts.values / ds_bolzano_mean.total_counts.values

    # converting values in percentage
    dict_rain_ratio = {
        "collalbo": rain_ratio_collalbo*100,
        "lagonero": rain_ratio_lagonero*100,
        "bolzano": rain_ratio_bolzano*100
    }

    dict_mean = {
        "collalbo": ds_collalbo_mean,
        "lagonero": ds_lagonero_mean,
        "bolzano": ds_bolzano_mean
    }
    return dict_rain_ratio, dict_mean

def read_rain_data(dict_mean):

    # read only rain occurrence for each site
    rain_occ_collalbo = dict_mean["collalbo"].rain_occ_diurnal.values
    rain_occ_lagonero = dict_mean["lagonero"].rain_occ_diurnal.values
    rain_occ_bolzano = dict_mean["bolzano"].rain_occ_diurnal.values
    # subsitute all zero with nans in the rain occurrence arrays
    rain_occ_collalbo[rain_occ_collalbo == 0] = np.nan
    rain_occ_lagonero[rain_occ_lagonero == 0] = np.nan
    rain_occ_bolzano[rain_occ_bolzano == 0] = np.nan

    # calculate mean rain occurrence for each time bin for each site
    mean_rain_occ_collalbo = np.nanmean(rain_occ_collalbo, axis=1)
    mean_rain_occ_lagonero = np.nanmean(rain_occ_lagonero, axis=1)
    mean_rain_occ_bolzano = np.nanmean(rain_occ_bolzano, axis=1)    

    # converting in minutes of rain occurrence over 2h time bins
    rain_occ_diurnal = {
        "collalbo": mean_rain_occ_collalbo*120,
        "lagonero": mean_rain_occ_lagonero*120,
        "bolzano": mean_rain_occ_bolzano*120
    }
    return rain_occ_diurnal
def plot_rain_occ_diurnal(rain_occ_diurnal, rain_occ_convective, rain_occ_mobl_t, time_bins, filename_out, ylabel, ylim, all_case_counts):

    font_size = 20


    # colors for locations from matplotlib colors
    col_LN = "mediumblue"
    col_CB = "purple"
    col_BZ = "orchid"


    # plot diurnal cycle of rain occurrence for the two sites
    colors = {"collalbo": col_CB, "lagonero": col_LN, "bolzano": col_BZ}
    # set all fonts to larger size for better visibility for all text, including axis labels, ticks, and legend
    plt.rcParams.update(
        {
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": font_size,
        }
    )
    fig, axes = plt.subplots(figsize=(15, 8))



    
    hour_ticks = time_bins.hour

    for site in rain_occ_diurnal:
        print(rain_occ_diurnal[site])
        axes.plot(hour_ticks, 
                 rain_occ_diurnal[site], 
                 label=f"{PLOT_SITES_NAMES[site]} - all days (N={all_case_counts[site]})", color=colors[site], linewidth=4)
        axes.plot(hour_ticks,
                 rain_occ_convective[site],
                 label=f"{PLOT_SITES_NAMES[site]} - convective days", color=colors[site], linewidth=4, linestyle="--")
        axes.plot(hour_ticks,
                 rain_occ_mobl_t[site],
                 label=f"{PLOT_SITES_NAMES[site]} - MOBL-T days", color=colors[site], linewidth=4, linestyle=":")
        
    axes.set_xticks(hour_ticks)
    axes.set_xticklabels([f"{hour:02d}:00" for hour in hour_ticks])
    axes.tick_params(axis="both", labelsize=font_size)
    axes.set_xlabel("Time UTC [hh:mm]", fontsize=font_size)
    axes.set_xlim(6, 22)
    axes.set_ylim(0, ylim)
    axes.set_ylabel(ylabel, fontsize=font_size)
    axes.set_title("Rainy cases: percentage of rain occurrence", fontsize=25, loc='left')

    # put the legend outside the plot horizontally below the x axis
    axes.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        fontsize=font_size,
    )


    # remove upper and right spines
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    # make remaining spines thicker
    axes.spines["bottom"].set_linewidth(1.5)
    axes.spines["left"].set_linewidth(1.5)
    # add grid lines
    axes.grid(color="grey", alpha=0.5, linestyle="--")

    fig.subplots_adjust(bottom=0.28)
    fig.tight_layout()
    fig.savefig(
        filename_out,
        dpi=300,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    main()
