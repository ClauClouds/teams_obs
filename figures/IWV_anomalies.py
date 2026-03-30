"""
plot IWV anomalies for convective and MOBL-T days for the three sites by reading 
"""
import pdb

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from readers.data_info import PLOT_SITES_NAMES, MWR_SITES_NAMES
import matplotlib.dates as mdates


def main():

    iwv_bz_conv = xr.open_dataset("/home/cacquist/Documents/GitHub/EXPATS/teams_obs/ncdf_anomalies/mean_anomaly_convective_iwv_bolzano.nc")
    iwv_cb_conv = xr.open_dataset("/home/cacquist/Documents/GitHub/EXPATS/teams_obs/ncdf_anomalies/mean_anomaly_convective_iwv_collalbo.nc")
    iwv_lg_conv = xr.open_dataset("/home/cacquist/Documents/GitHub/EXPATS/teams_obs/ncdf_anomalies/mean_anomaly_convective_iwv_lagonero.nc")
    iwv_bz_MOBL_T = xr.open_dataset("/home/cacquist/Documents/GitHub/EXPATS/teams_obs/ncdf_anomalies/mean_anomaly_MOBL_T_iwv_bolzano.nc")
    iwv_cb_MOBL_T = xr.open_dataset("/home/cacquist/Documents/GitHub/EXPATS/teams_obs/ncdf_anomalies/mean_anomaly_MOBL_T_iwv_collalbo.nc")
    iwv_lg_MOBL_T = xr.open_dataset("/home/cacquist/Documents/GitHub/EXPATS/teams_obs/ncdf_anomalies/mean_anomaly_MOBL_T_iwv_lagonero.nc")
    
    # plot figure
    fig2, ax2 = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # first subplot: bolzano
    ax2[0].plot(iwv_bz_conv.time, iwv_bz_conv.mean_anomaly, label="Convective", color="orange", linewidth=4)
    ax2[0].plot(iwv_bz_MOBL_T.time, iwv_bz_MOBL_T.mean_anomaly, label="MOBL-T", color="green", linewidth=4)

    ax2[1].plot(iwv_cb_conv.time, iwv_cb_conv.mean_anomaly, label="Convective", color="orange", linewidth=4)
    ax2[1].plot(iwv_cb_MOBL_T.time, iwv_cb_MOBL_T.mean_anomaly, label="MOBL-T", color="green", linewidth=4)

    ax2[2].plot(iwv_lg_conv.time, iwv_lg_conv.mean_anomaly, label="Convective", color="orange", linewidth=4)
    ax2[2].plot(iwv_lg_MOBL_T.time, iwv_lg_MOBL_T.mean_anomaly, label="MOBL-T", color="green", linewidth=4)

    titles = ["Bolzano", "Collalbo", "Lagonero"]
    elevations = [262, 1560, 2060]
    for index, axis in enumerate(ax2):
        axis.axhline(0, color='black', linestyle='--', linewidth=0.5)
        axis.set_title(f"{titles[index]}, ({elevations[index]} [m])", fontsize=20)
        axis.set_xlim([np.datetime64("2000-01-01T00:10:30"), np.datetime64("2000-01-01T23:59:30")])
        axis.set_ylim([-2, 8])
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
        axis.spines['bottom'].set_linewidth(1.5)
        axis.spines['left'].set_linewidth(1.5)
        axis.grid(color='grey', alpha=0.5, linestyle='--')
        axis.xaxis.set_major_locator(mdates.HourLocator(interval=3))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        axis.xaxis.set_minor_locator(mdates.MinuteLocator(interval=30))
        axis.tick_params(axis='both', which='major', labelsize=20)
        axis.tick_params(axis='x', which='minor', length=3)
        axis.tick_params(axis='x', which='major', length=7)
        axis.set_ylabel("IWV anomaly [kgm$^{-2}$]", fontsize=20)

    ax2[2].set_xlabel("Time of day (HH:MM) [UTC]", fontsize=20)
    # position legend outside the plot under the x axis
    ax2[2].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, fontsize=20)
    fig2.tight_layout()
    fig2.savefig("plots/IWV_anomalies_convective.png", dpi=300)   


if __name__ == "__main__":
    main()