""""
plotting functions for the figures of the paper

"""


from datetime import datetime
import pdb
import pandas as pd

import cmcrameri as cmc
import matplotlib.pyplot as plt
from figures.plot_settings import *
import matplotlib.dates as mdates
from readers.data_info import *

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