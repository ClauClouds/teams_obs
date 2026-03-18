""""
plotting functions for the figures of the paper

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
# select three equidistant colors from the crameri color palette cork

def select_descrete_colors(n_colors):
    """
    Select n equidistant colors from a Crameri color palette.

    Args:
        n_colors (int): Number of colors to select.
    Dependencies:
        CMAP (matplotlib.colors.Colormap): Crameri color palette to use for selecting colors.

    Returns:
        list: List of selected colors.
    """
    # Get the colormap
    cmap = CMAP

    # Get n_colors discrete colors
    colors = [cmap(i / (n_colors - 1)) for i in range(n_colors)]    
    return colors

def plot_time_series(fci_twv, mwr_twv, site_names, date):
    
    """
    function to plot the time series of the FCI TWV and MWR IWV for the three sites

    Args:
        fci_twv_bz(xarray.DataArray): FCI TWV values for the BZ site.
        fci_twv_cb(xarray.DataArray): FCI TWV values for the CB site.
        fci_twv_lg(xarray.DataArray): FCI TWV values for the LG site.
        iwv_bz(xarray.DataArray): MWR IWV values for the BZ site.
        iwv_cb(xarray.DataArray): MWR IWV values for the CB site.
        iwv_lg(xarray.DataArray): MWR IWV values for the LG site.
        site_names (list): list of names of the sites
        date (str): date in "yyyymmdd" format to plot in the title of the figure
    dependencies:
    PLOT_SITES_NAMES (dict): dictionary with the names of the sites to plot as keys and the names to show in the plot as values
    Returns:
    """
    # select colors for the FCI and MWR
    colors = ["orange", "green"]

    # plot the same as before but with all sites in the same plot (differentiate with line style)
    fig2 = plt.figure(figsize=(15, 10))
    ax2 = fig2.add_subplot(1, 1, 1)

    for i in range(len(site_names)):

        ax2.plot(fci_twv[i].time.values, 
                fci_twv[i].twv.values, 
                label=f"{PLOT_SITES_NAMES[site_names[i]]} FCI TWV", 
                color=colors[0],
                linestyle=["-", "--", ":"][i],
                linewidth=4)
        # plot uncertainty as shaded area around the line
        ax2.fill_between(fci_twv[i].time.values, 
                         fci_twv[i].twv.values - fci_twv[i].twv_uncertainty.values,
                         fci_twv[i].twv.values + fci_twv[i].twv_uncertainty.values,
                         color=colors[0], alpha=0.3)
        
        ax2.plot(mwr_twv[i].time.values,
                mwr_twv[i].iwv.values, 
                label=f"{PLOT_SITES_NAMES[site_names[i]]} MWR IWV", 
                color=colors[1],
                linestyle=["-", "--", ":"][i],
                linewidth=4)  
        # plot uncertainty as shaded area around the line
        ax2.fill_between(mwr_twv[i].time.values,
                         mwr_twv[i].iwv.values - mwr_twv[i].iwv_std.values,
                         mwr_twv[i].iwv.values + mwr_twv[i].iwv_std.values,
                         color=colors[1], alpha=0.3)
         
    # position the legend outside the plot on the bottom
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), frameon=False, ncol=3)   
    # set legend font size
    legend = ax2.get_legend()
    legend.get_frame().set_linewidth(0.0)
    legend.get_frame().set_alpha(0.0)
    for text in legend.get_texts():
        text.set_fontsize(18)

    # add title in bold
    ax2.set_title(f"Case study: {date}", fontsize=20, fontweight='bold')
    ax2.set_xlabel("Time UTC [hh:mm]", fontsize=20)
    ax2.set_ylabel("Water Vapor [kg/m2]", fontsize=20)
    starttime = datetime(year=int(date[:4]), month=int(date[4:6]), day=int(date[6:8]), hour=6, minute=0)
    endtime = datetime(year=int(date[:4]), month=int(date[4:6]), day=int(date[6:8]), hour=17, minute=0)
    ax2.set_xlim([starttime, endtime])
    ax2.set_ylim([5, 27])
    ax2.tick_params(axis='both', which='major', labelsize=20)

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_linewidth(1.5) 
    ax2.spines['left'].set_linewidth(1.5)
    ax2.grid(color='grey', alpha=0.5, linestyle='--')
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    # add minor ticks every 30 min 
    ax2.xaxis.set_minor_locator(mdates.MinuteLocator(interval=30))

    # increase tick lenght of minor ticks and major ticks
    ax2.tick_params(axis='x', which='minor', length=3)
    ax2.tick_params(axis='x', which='major', length=7)



    fig2.tight_layout()
    fig2.savefig("plots/time_series_MWR_TWV_comparison_all_sites.png", dpi=300)   
 
    return None    


def plot_diurnal_cycle_single_site(site, ds_type, day_type, var_type):
    """
    plot diurnal cycle of the IWV for the group of days in ds_type
    The diurnal cycle is plotted for each day with a superthin line transparency and then
    the mean and standard deviation of the IWV values among all days for each time stamp as
   a thicker line with shaded area for the standard deviation.
   input:
    site: str, name of the site
    ds_type: list of xarray datasets, each containing data for a single day
    day_type: str, type of day ("convective" or "MOBL_T")
    var_type: str, variable to plot ("lwp" or "iwv")
    dependencies:
    PLOT_SITES_NAMES (dict): dictionary with the names of the sites to plot as keys and the names to show in the plot as values
    Returns:
    None, saves the plot in the "plots" directory with name "diurnal_cycle_{var_type}_{site}_{day_type}.png"
    """

    # plot a figure with the characteristics specified above
    fig2 = plt.figure(figsize=(15, 10))
    ax2 = fig2.add_subplot(1, 1, 1)
    N_days = len(ds_type)

    # generate matrix where to store iwv for all days and a time dimension for the entire day (e.g. from 00:00 to 23:59 with 3 secondly time steps) to calculate mean and std among days for each time step
    iwv_matrix = np.zeros((N_days, 24*60*60//3))
    d_cycle_start = datetime(year=2000, month=1, day=1, hour=0, minute=0, second=0)
    d_cycle_end = datetime(year=2000, month=1, day=1, hour=23, minute=59, second=59)
    time_cycle = pd.date_range(start=d_cycle_start, end=d_cycle_end, freq='3s')

    # loop on days to be plotted
    for i_day in range(len(ds_type)):

        # reading data from single day dataset
        ds_day = ds_type[i_day]

        # Use a common reference date so all days can be interpolated on the same diurnal cycle.
        time_of_day = pd.to_datetime(ds_day.time.values).strftime('2000-01-01 %H:%M:%S')
        ds_day = ds_day.assign_coords(time=pd.to_datetime(time_of_day))

        # resample on 3 secondly time steps with linear interpolation to have the same time stamps for all days (and then calculate mean and std)
        ds_day = ds_day.interp(time=time_cycle)

        # search small nan gaps (less than 10 min) and fill them with linear interpolation to avoid having too many nans in the time series of the days (which would lead to having few values to calculate mean and std among days for each time step)
        ds_day = ds_day.interpolate_na(dim='time', method='linear', limit=10*60//3)

        time = ds_day.time.values
        var_values = ds_day[var_type].values
        iwv_matrix[i_day, :] = var_values
        # plot diurnal cycle of the day with superthin line and transparency
        ax2.plot(time_cycle, 
                 var_values, 
                 color="blue", 
                 alpha=0.3,
                 linewidth=0.5)
        
    # calculate mean and standard deviation of the IWV values among all days for each time stamp
    # if values different from nan in the time stamps are less than 3, set mean and std to nan to avoid plotting mean and std for time stamps with few values among days

    n_values = np.sum(~np.isnan(iwv_matrix), axis=0)
    iwv_matrix[:, n_values < 4] = np.nan
    iwv_mean = np.nanmedian(iwv_matrix, axis=0)
    iwv_std = np.nanstd(iwv_matrix, axis=0)

    # plot mean diurnal cycle with thicker line and shaded area for the standard deviation
    ax2.plot(time_cycle, 
             iwv_mean, 
             color="blue", 
             label=f"{PLOT_SITES_NAMES[site]} Mean Diurnal Cycle", 
             linewidth=3)
    
    ax2.fill_between(time_cycle, 
                    iwv_mean - iwv_std, 
                    iwv_mean + iwv_std, 
                    color="blue",
                    alpha=0.3, 
                    label=f"{PLOT_SITES_NAMES[site]} Std Dev")
    
    # add title in bold
    ax2.set_title(f"Diurnal Cycle of {var_type.upper()} for {PLOT_SITES_NAMES[site]} - {day_type} Days", fontsize=20, fontweight='bold')
    ax2.set_xlabel("Time UTC [hh:mm]", fontsize=20)
    ax2.set_ylabel("IWV [kg/m2]", fontsize=20)
    ax2.tick_params(axis='both', which='major', labelsize=20) 
     
    # set x limits from 00:00 to 23:59
    starttime = datetime(year=2000, month=1, day=1, hour=0, minute=0)
    endtime = datetime(year=2000, month=1, day=1, hour=23, minute=59)
    ax2.set_xlim([starttime, endtime])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_linewidth(1.5)
    ax2.spines['left'].set_linewidth(1.5)
    ax2.grid(color='grey', alpha=0.5, linestyle='--')
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    # add minor ticks every 30 min
    ax2.xaxis.set_minor_locator(mdates.MinuteLocator(interval=30))
    # increase tick lenght of minor ticks and major ticks
    ax2.tick_params(axis='x', which='minor', length=3)
    ax2.tick_params(axis='x', which='major', length=7)
    fig2.tight_layout()
    fig2.savefig(f"plots/diurnal_cycle_{var_type}_{site}_{day_type}.png", dpi=300)   
    return None 



def plot_hourly_percentiles(site, ds_type, day_type, var_type):
    """
    function to calculate hourly percentiles and plot one percentile-based box for each hour

    Args:
        site (string): str, name of the site
        ds_type (list of xarray datasets): ist of xarray datasets, each containing data for a single day
        day_type (string): str, type of day ("convective" or "MOBL_T")
        var_type (string): str, variable to plot ("lwp" or "iwv")
    """




    N_days = len(ds_type)
    lwp_matrix = np.zeros((N_days, 24*60*60//3)) # matrix to store lwp values for all days and hours
    d_cycle_start = datetime(year=2000, month=1, day=1, hour=0, minute=0, second=0)
    d_cycle_end = datetime(year=2000, month=1, day=1, hour=23, minute=59, second=59)
    time_cycle = pd.date_range(start=d_cycle_start, end=d_cycle_end, freq='3s')

    # loop on days to be plotted
    for i_day in range(len(ds_type)):

        # reading data from single day dataset
        ds_day = ds_type[i_day]

        # resample on 3 secondly time steps with linear interpolation to have the same time stamps for all days (and then calculate mean and std)
        ds_day = ds_day.interp(time=time_cycle)

        # search small nan gaps (less than 10 min) and fill them with linear interpolation to avoid having too many nans in the time series of the days (which would lead to having few values to calculate mean and std among days for each time step)
        ds_day = ds_day.interpolate_na(dim='time', method='linear', limit=10*60//3)

        var_values = ds_day[var_type].values
        lwp_matrix[i_day, :] = var_values

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
        hour_positions.append(len(hour_positions))

    # plot a figure with the percentiles as wiskers plot
    fig2 = plt.figure(figsize=(15, 10))
    ax2 = fig2.add_subplot(1, 1, 1)
    # plot y axis in log scale
    #ßax2.set_yscale('log')
    # Plot one percentile-based box for each hour.
    # fill box plot in light orange and median line in orange thicker than the box edges
    ax2.bxp(boxplot_stats, 
            positions=hour_positions, 
            widths=0.6, 
            showfliers=True, 
            patch_artist=True, 
            boxprops=dict(facecolor='lightorange', color='orange'), 
            medianprops=dict(color='orange', linewidth=2), 
            whiskerprops=dict(color='orange', linewidth=1.5),
            capprops=dict(color='orange', linewidth=1.5))
    
    # set ylim
    ax2.set_ylim([0, 100])
    ax2.plot(hour_positions, median_values, color="black", linewidth=2, marker="o")
    # add title in bold
    ax2.set_title(f"Hourly Percentiles of {var_type.upper()} for {PLOT_SITES_NAMES[site]} - {day_type} Days", fontsize=20, fontweight='bold')
    ax2.set_xlabel("Hour of the Day [UTC]", fontsize=20)
    ax2.set_ylabel(f"{var_type.upper()} [gm-2]", fontsize=20)
    ax2.set_xticks(hour_positions)
    ax2.set_xticklabels([stats["label"] for stats in boxplot_stats], rotation=45)
    ax2.tick_params(axis='both', which='major', labelsize=20)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_linewidth(1.5)
    ax2.spines['left'].set_linewidth(1.5)
    ax2.grid(color='grey', alpha=0.5, linestyle='--')
    fig2.tight_layout()
    fig2.savefig(f"plots/hourly_percentiles_{var_type}_{site}_{day_type}.png", dpi=300) 

    return None


def plot_mean_dc(site, time, ds_mean, ds_std, var_type):
    """
    plot diurnal cycle of the selected variable for the group of days in ds_type
    with a thick line for the mean and a shaded area for the standard deviation
    input:
    site: str, name of the site 
        ds_mean: xarray dataset containing the mean diurnal cycle of the variable among the days in ds_type
        ds_std: xarray dataset containing the standard deviation of the diurnal cycle of the variable among the days in ds_type
        var_type: str, variable to plot ("lwp" or "iwv")
    dependencies:
        PLOT_SITES_NAMES (dict): dictionary with the names of the sites to plot as keys and the names to show in the plot as values
    Returns:
        None, saves the plot in the "plots" directory with name "diurnal_cycle_{var_type}_{site}_{day_type}.png"
    """
    
    # plot a figure with the characteristics specified above
    fig2 = plt.figure(figsize=(15, 10))
    ax2 = fig2.add_subplot(1, 1, 1)
    # set all fonts to size 20
    plt.rcParams.update({'font.size': 20})

    # plot mean ds_mean with thicker line and shaded area for the standard deviation ds_std
    ax2.plot(time, 
             ds_mean, 
             color="black",
            label=f"Diurnal Cycle",
            linewidth=3)    
    ax2.fill_between(time, 
                    ds_mean - ds_std, 
                    ds_mean + ds_std, 
                    color="black",
                    alpha=0.3, 
                    label=f"{PLOT_SITES_NAMES[site]} Std Dev")        
    
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_linewidth(1.5)
    ax2.spines['left'].set_linewidth(1.5)
    ax2.grid(color='grey', alpha=0.5, linestyle='--')
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    # add minor ticks every 30 min
    ax2.xaxis.set_minor_locator(mdates.MinuteLocator(interval=30))
    ax2.set_xlim([datetime(year=2000, month=1, day=1, hour=0, minute=0), datetime(year=2000, month=1, day=1, hour=23, minute=59)])
    # set tick labels size to 20
    ax2.tick_params(axis='both', which='major', labelsize=20)
    # increase tick lenght of minor ticks and major ticks
    ax2.tick_params(axis='x', which='minor', length=3)
    ax2.tick_params(axis='x', which='major', length=7)
    ax2.set_xlabel("Time UTC [hh:mm]", fontsize=20)
    ax2.set_ylabel("IWV [kg/m2]", fontsize=20)
    ax2.set_title(f"{PLOT_SITES_NAMES[site]}", fontsize=20, fontweight='bold')
    fig2.tight_layout()
    fig2.savefig(f"plots/diurnal_cycle_mean_{var_type}_{site}.png", dpi=300) 
    
    return None