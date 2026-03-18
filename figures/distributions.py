"""
codes to plot distributions of IWV and LWP for convective and MOBL-T conditions.
We want two subplots, one with IWV distribution for convective days in orange and for MOBL-T days in gree
and one with LWP distribution for convective days in orange and for MOBL-T days in blue.
Plot are for a given site to be specified in input
"""
from readers.data_info import PLOT_SITES_NAMES, MWR_SITES_NAMES, iop_conv_days, iop_MoBL_T_days
from readers.txt import read_txt_file
from readers.MWR import read_lwp_iwv
import matplotlib.pyplot as plt
from plotting import plot_diurnal_cycle_single_site, plot_hourly_percentiles
import numpy as np   
import pandas as pd
import xarray as xr
import pdb

def main():
    
    # select for which site you want to do the comparison of the distributions of IWV and LWP for convective and MOBL-T conditions
    site = "lagonero" # specify site to plot ("bolzano", "collalbo", or "lagonero")

    # read LWP and IWV for convective days
    IWV_conv = []
    LWP_conv = []
    for i_day, day in enumerate(iop_conv_days):
        day_string = day
        path_file_sel = f"/data/obs/campaigns/teamx/{site}/{MWR_SITES_NAMES[site]}/actris/level2/{day_string[:4]}/{day_string[4:6]}/{day_string[6:8]}/"
        ds_iwv = read_lwp_iwv(site, day, "iwv", path_file_sel)
        ds_lwp = read_lwp_iwv(site, day, "lwp", path_file_sel)

        # append only IWV and LWP values to the respective lists
        IWV_conv.append(ds_iwv.iwv.values)
        LWP_conv.append(ds_lwp.lwp.values)
    
    # read LWP and IWV for MOBL-T days
    IWV_MOBL_T = []
    LWP_MOBL_T = []
    for i_day, day in enumerate(iop_MoBL_T_days):
        day_string = day
        path_file_sel = f"/data/obs/campaigns/teamx/{site}/{MWR_SITES_NAMES[site]}/actris/level2/{day_string[:4]}/{day_string[4:6]}/{day_string[6:8]}/"
        ds_iwv = read_lwp_iwv(site, day, "iwv", path_file_sel)
        ds_lwp = read_lwp_iwv(site, day, "lwp", path_file_sel)

        # append only IWV and LWP values to the respective lists
        IWV_MOBL_T.append(ds_iwv.iwv.values)
        LWP_MOBL_T.append(ds_lwp.lwp.values)

    # convert lists of arrays to single lists of values
    IWV_conv = [item for sublist in IWV_conv for item in sublist]
    LWP_conv = [item for sublist in LWP_conv for item in sublist]
    IWV_MOBL_T = [item for sublist in IWV_MOBL_T for item in sublist]
    LWP_MOBL_T = [item for sublist in LWP_MOBL_T for item in sublist]

    # remove nans from the lists
    IWV_conv = [x for x in IWV_conv if not np.isnan(x)]
    LWP_conv = [x for x in LWP_conv if not np.isnan(x)]
    IWV_MOBL_T = [x for x in IWV_MOBL_T if not np.isnan(x)]
    LWP_MOBL_T = [x for x in LWP_MOBL_T if not np.isnan(x)] 

    # remove LWP values equal or smaller than 20 g/m^2 to focus on the distribution of higher LWP values (we want to exclude clear-sky and very thin clouds)
    LWP_conv = [x for x in LWP_conv if x > 20]
    LWP_MOBL_T = [x for x in LWP_MOBL_T if x > 20]

    # make the plots of the distributions of IWV and LWP for convective and MOBL-T conditions
    plt.figure(figsize=(12, 6))
    fig = plt.gcf()
    # set all font size to 20ß
    plt.rcParams.update({'font.size': 18})
    plt.subplot(1, 2, 1)
    plt.hist(IWV_conv, 
             bins=20, 
             alpha=0.5, 
             density=True,
             label="Convective days", 
             color="orange")
    plt.hist(IWV_conv,
            bins=20,
            density=True,
            histtype="step",
            color="black",
            linewidth=2
        )
    plt.hist(IWV_MOBL_T, 
             bins=20, 
             alpha=0.5, 
             density=True,
             label="MOBL-T days", 
             color="green")
    plt.hist(IWV_MOBL_T,
            bins=20,
            density=True,
            histtype="step",
            color="black",
            linewidth=2
        )
    plt.xlabel("IWV [mm]")
    plt.ylabel("Normalized frequency")
    plt.title(f"IWV distribution for {PLOT_SITES_NAMES[site]}")
    # put legend outside the plot under the x axis
    ax1 = plt.gca()
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_linewidth(1.5)
    ax1.spines['left'].set_linewidth(1.5)
    ax1.grid(color='grey', alpha=0.5, linestyle='--')

    plt.subplot(1, 2, 2)
    # plot normalized histogram line thick 
    plt.hist(LWP_conv, 
             bins=50, 
             alpha=0.5, 
             label="Convective days", 
             color="orange", 
             density=True,
             linewidth=2.)
    plt.hist(LWP_conv,
            bins=50,
            density=True,
            histtype="step",
            color="black",
            linewidth=2
        )
    plt.hist(LWP_MOBL_T, 
             bins=50, 
             alpha=0.5, 
             label="MOBL-T days", 
             color="green", 
             density=True,
             linewidth=2.)    
    plt.hist(LWP_MOBL_T,
            bins=50,
            density=True,
            histtype="step",
            color="black",
            linewidth=2
        )
    
    # set log scale on x
    #plt.xscale("log")
    plt.xlabel("LWP [g/m^2]")
    plt.ylabel("Normalized frequency")
    plt.title(f"LWP distribution for {PLOT_SITES_NAMES[site]}")
    # set x lim 
    plt.xlim(0, 800)
    ax2 = plt.gca()
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_linewidth(1.5)
    ax2.spines['left'].set_linewidth(1.5)
    ax2.grid(color='grey', alpha=0.5, linestyle='--')

    # plot only one legend for both subplots in the center of the figure under the x axis
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=2)

    plt.tight_layout(rect=[0, 0.08, 1, 1])



    # save figure
    plt.savefig(f"plots/distributions_IWV_LWP_{site}.png", dpi=300) 
    







if __name__ == "__main__":
    main()  