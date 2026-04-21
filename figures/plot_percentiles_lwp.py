"""
code to plot hourly percentiles for LWP for all data, only convective days and only MOBL-T days
for each site
"""



from datetime import datetime
import pdb
import pandas as pd
import numpy as np
import cmcrameri as cmc
import matplotlib.pyplot as plt
from figures.plot_settings import *
import matplotlib.dates as mdates
from readers.data_info import *
from figures.plot_settings import VAR_DICT
import xarray as xr

def main():



    # specify for which site to do the plot
    sites = ["lagonero", "collalbo", "bolzano"]
    site_names = ["Schwartzseespitze", "Klobenstein", "Bozen"]

    for site, site_name in zip(sites, site_names):
        print(f"Processing site: {site_name}")
        # read files for that site
        lwp_all = xr.open_dataset(f"data/{site}_all_lwp_diurnal_cycle.nc")
        lwp_convective = xr.open_dataset(f"data/{site}_convective_lwp_diurnal_cycle.nc")
        lwp_mobl_t = xr.open_dataset(f"data/{site}_MOBL_T_lwp_diurnal_cycle.nc")

        # calculate LWP matrix
        print("Calculating LWP matrix for all days...")
        lwp_matrix_all = calc_lwp_matrix(lwp_all)
        print("Calculating LWP matrix for convective days...")
        lwp_matrix_convective = calc_lwp_matrix(lwp_convective)
        print("Calculating LWP matrix for MOBL-T days...")
        lwp_matrix_mobl_t = calc_lwp_matrix(lwp_mobl_t)

        # calculate boxplot stats
        print("Calculating boxplot stats for all days...")
        boxplot_stats_all, median_values_all, hour_positions_all = calc_boxplot_stats(lwp_matrix_all)
        print("Calculating boxplot stats for convective days...")
        boxplot_stats_convective, median_values_convective, hour_positions_convective = calc_boxplot_stats(lwp_matrix_convective)
        print("Calculating boxplot stats for MOBL-T days...")
        boxplot_stats_mobl_t, median_values_mobl_t, hour_positions_mobl_t = calc_boxplot_stats(lwp_matrix_mobl_t)

        # make figure where for each hour we have the three boxplots for all days, convective days and MOBL-T days
        print("Plotting boxplots...")
        fig, ax = plt.subplots(figsize=(12, 6))
        # enlarge figure fonts, label fonts, axis ticks fonts, and legend fonts
        plt.rcParams.update({'font.size': 14})
        # enlarge x and y tick fonts
        font_size = 16
        ax.tick_params(axis='x', labelsize=font_size)
        ax.tick_params(axis='y', labelsize=font_size)
        ax.bxp(boxplot_stats_all, 
            positions=hour_positions_all,
            widths=0.2, 
            showfliers=False, 
            patch_artist=True, 
            boxprops=dict(facecolor="grey", color="grey"), 
            label="All days",
            medianprops=dict(color="black"))
        
        ax.bxp(boxplot_stats_convective, 
            positions=np.array(hour_positions_convective)+0.25, 
            widths=0.2, 
            showfliers=False, 
            patch_artist=True, 
            boxprops=dict(facecolor="orange", color="orange"), 
            label="Convective days",
            medianprops=dict(color="black"))
        ax.bxp(boxplot_stats_mobl_t, 
            positions=np.array(hour_positions_mobl_t)+0.5, 
            widths=0.2, 
            showfliers=False, 
            patch_artist=True, 
            boxprops=dict(facecolor="green", color="green"), 
            label="MOBL-T days",
            medianprops=dict(color="black"))
        
        ax.set_xticks(np.arange(24))
        # format x axis to show hours in hh:mm format using mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.set_xticklabels([f"{i:02d}:00" for i in range(24)])
        # orient x axis labels at 45 degrees
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_xlabel("Time UTC [hh:mm]", fontsize=font_size)    
        ax.set_ylabel("LWP [gm-2]", fontsize=font_size)
        ax.set_title(f"LWP hourly percentiles: {site_name}", fontsize=font_size, loc="left")
        # set y axis in log
        ax.set_ylim(0, 200)

        # position the legend outside the plot below the x axis: write in the legend Convective, MOBL_T and all with the corresponding colors
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=False, fontsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.grid(color='grey', alpha=0.5, linestyle='--')
        plt.tight_layout()
        plt.savefig(f"plots/{site}_lwp_hourly_percentiles.png", dpi=300)
        plt.close()



def calc_lwp_matrix(ds):

    N_days = len(ds.days.values)
    lwp_matrix = np.zeros((N_days, 24*60*60//3)) # matrix to store lwp values for all days and hours
    d_cycle_start = datetime(year=2000, month=1, day=1, hour=0, minute=0, second=0)
    d_cycle_end = datetime(year=2000, month=1, day=1, hour=23, minute=59, second=59)
    time_cycle = pd.date_range(start=d_cycle_start, end=d_cycle_end, freq='3s')

    # loop on days to be plotted
    for i_day, day in enumerate(ds.days.values):
        
        # reading data from single day dataset
        ds_day = ds.sel(days=day)

        # resample on 3 secondly time steps with linear interpolation to have the same time stamps for all days (and then calculate mean and std)
        ds_day = ds_day.interp(time=time_cycle)

        # search small nan gaps (less than 10 min) and fill them with linear interpolation to avoid having too many nans in the time series of the days (which would lead to having few values to calculate mean and std among days for each time step)
        ds_day = ds_day.interpolate_na(dim='time', method='linear', limit=10*60//3)

        lwp_matrix[i_day, :] = ds_day['lwp'].values

    return lwp_matrix


def calc_boxplot_stats(lwp_matrix):

    # Calculate percentile-based boxplot statistics for each hour.
    # q1, med and q3 define the box; whislo and whishi define the whiskers.
    boxplot_stats = []
    median_values = []
    hour_positions = []
    for i in range(24):
        hour_values = lwp_matrix[:, i*60*60//3:(i+1)*60*60//3].reshape(-1)
        hour_values = hour_values[~np.isnan(hour_values)]

        if hour_values.size == 0:
            continue

        median_value = np.nanpercentile(hour_values, 50)

        boxplot_stats.append(
            {
                "label": f"{i:02d}:00",
                "whislo": np.nanpercentile(hour_values, 10),
                "q1": np.nanpercentile(hour_values, 25),
                "med": median_value,
                "q3": np.nanpercentile(hour_values, 75),
                "whishi": np.nanpercentile(hour_values, 90),
                "fliers": [],
            }
        )
        median_values.append(median_value)
        hour_positions.append(i)

    return boxplot_stats, median_values, hour_positions

if __name__ == "__main__":
    main()