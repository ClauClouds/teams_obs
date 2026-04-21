"""
code to plot diurnal cyle of rain occurrence for the two sites, and radar reflectivity mean profiles for
- all days
- convective days
- MOBL-T days
and diurnal cycle of radar reflectivity profiles for the same categories, for both sites.
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
from matplotlib.lines import Line2D
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

    # Anchor time bins to a common reference date so they can be compared with the
    # time-of-day coordinates assigned below.
    time_bins = pd.to_datetime(
        [f"2000-01-01 {hour}" for hour in hours_diurnal_cycle_calc] + ["2000-01-02 00:00"]
    )

    # read data from mrr profiles
    ds_collalbo = xr.open_dataset("data/mrr_ze_profiles/mrr_ze_profiles_collalbo.nc")
    ds_collalbo_convective = xr.open_dataset("data/mrr_ze_profiles/mrr_ze_profiles_collalbo_convective.nc")
    ds_collalbo_mobl_t = xr.open_dataset("data/mrr_ze_profiles/mrr_ze_profiles_collalbo_mobl_t.nc")

    ds_lagonero = xr.open_dataset("data/mrr_ze_profiles/mrr_ze_profiles_lagonero.nc")  
    ds_lagonero_convective = xr.open_dataset("data/mrr_ze_profiles/mrr_ze_profiles_lagonero_convective.nc")
    ds_lagonero_mobl_t = xr.open_dataset("data/mrr_ze_profiles/mrr_ze_profiles_lagonero_mobl_t.nc")

    print("Data loaded successfully.")

    # figure with 2 subplots for the two sites with mean ze profile for all days, convective days and MOBL-T days,
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=False)

    ds_arr = [ds_collalbo, ds_lagonero]
    # set to nan where ze_mean is 0, because this means that there was no valid data for that height bin, and we don't want to plot it as 0 ze
    for ds in ds_arr:
        ds["ze_mean"] = ds["ze_mean"].where(ds["ze_mean"] != 0, np.nan) 

    ds_arr_convective = [ds_collalbo_convective, ds_lagonero_convective]
    ds_arr_mobl_t = [ds_collalbo_mobl_t, ds_lagonero_mobl_t]
    ground_height = [1192, 2066]
    
    for ax, ds, ds_convective, ds_mobl_t, gh in zip(axes, ds_arr, ds_arr_convective, ds_arr_mobl_t, ground_height):
        # plot mean ze profiles for each site
        ax.plot(ds["ze_mean"], 
                    ds["height"]+gh,
                    label="All days", 
                    color="grey", 
                    linestyle="-", 
                    linewidth=4)
        ax.plot(ds_convective["ze_mean"], 
                    ds_convective["height"]+gh,
                    label="Convective days",
                    color="orange",
                    linestyle="-",
                    linewidth=4)
        ax.plot(ds_mobl_t["ze_mean"], 
                    ds_mobl_t["height"]+gh,
                    label="MOBL-T days",
                    color="green",
                    linestyle="-",
                    linewidth=4)
        # remove spines and make the remaining thicker
        ax.spines["left"].set_linewidth(1.5)
        ax.spines["bottom"].set_linewidth(1.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # set ticks labels font size
        ax.tick_params(axis="both", which="major", labelsize=18)

        ax.set_xlabel("Mean Ze [dBZ]", fontsize=20)
    axes[0].set_ylabel("Height [m]", fontsize=20)
    axes[0].set_title("Klobenstein", fontsize=20)
    axes[1].set_title("Schwartzseespitze", fontsize=20)
    axes[0].set_xlim(-10, 20)
    axes[1].set_xlim(0, 25)
    axes[0].set_ylim(ground_height[0], 5000)
    axes[1].set_ylim(ground_height[1], 5000)

    legend_handles = [
        Line2D([0], [0], color="grey", linestyle="-", linewidth=4, label="All days"),
        Line2D([0], [0], color="orange", linestyle="-", linewidth=4, label="Convective days"),
        Line2D([0], [0], color="green", linestyle="-", linewidth=4, label="MOBL-T days"),
    ]

    # Put one figure-level legend centered below both subplots.
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.04),
        frameon=False,
        fontsize=14,
        ncol=3,
    )
    fig.subplots_adjust(bottom=0.16, wspace=0.25)

    # save figure
    output_path = f"/home/cacquist/Documents/GitHub/EXPATS/teams_obs/plots/ze_mean_profile.png"
    plt.savefig(output_path, bbox_inches="tight")
    # set 


if __name__ == "__main__":
    main()

