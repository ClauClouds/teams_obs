"""
code to plot spatial distribution of IWV as a function of time for each site for convective and MOBL_T days.
We plot a round circle with directions on any radius indicating the IWV at that time of the day. We have one plot for each time step
and we plot a video by merging the time steps together.

"""

from readers.data_info import MWR_SITES_NAMES, PLOT_SITES_NAMES
from readers.MWR import read_iwv_elev
import matplotlib.pyplot as plt
from figures.plot_settings import VAR_DICT
import numpy as np   
import os
import pandas as pd
import subprocess
import xarray as xr
import pdb

from figures.utils import get_scan_ids, extract_scan_by_id, plot_iwv_azimuth_ring, calc_iwv_deviation, make_video_from_frames


def main():

    day_string = "20250625"
    site = "lagonero"
    elev_sel = 30
    var2plot= "IWV_deviation" # "iwv" or "IWV_deviation"

    # read MWR data for spatial IWV distributions
    path_root = f"/data/obs/campaigns/teamx/{site}/{MWR_SITES_NAMES[site]}/actris/level2/{day_string[:4]}/{day_string[4:6]}/{day_string[6:8]}/"
    ds_iwv_elev = read_iwv_elev(site, day_string, "iwv", elev_sel, path_root)

    # add variable for deviation from mean IWV over the azimuth scan
    IWV_deviation = calc_iwv_deviation(ds_iwv_elev)
    # add deviation to 
    ds_iwv_elev['IWV_deviation'] = IWV_deviation

    output_dir = "plots/IWV_spatial"
    output_prefix = f"{var2plot}_spatial_{site}_{day_string}_{elev_sel}deg"
    os.makedirs(output_dir, exist_ok=True)

    # identify max and min IWV values during the day to set the color scale of the plots
    iwv_min = ds_iwv_elev.iwv.min().item()
    iwv_max = ds_iwv_elev.iwv.max().item()
    iwv_dev_max = np.nanmean(np.abs(ds_iwv_elev.IWV_deviation.values))

    VAR_DICT['iwv']['vmin'] = np.floor(iwv_min / 2) * 2
    VAR_DICT['iwv']['vmax'] = np.ceil(iwv_max / 2) * 2   
    VAR_DICT['IWV_deviation']['vmin'] = -np.ceil(iwv_dev_max / 2) * 2
    VAR_DICT['IWV_deviation']['vmax'] = np.ceil(iwv_dev_max / 2) * 2

    # Generate one frame per azimuth scan rather than one frame per sample.
    scan_ids = get_scan_ids(ds_iwv_elev)
    unique_scan_ids = np.unique(scan_ids)
    for scan_id in unique_scan_ids:
        ds_scan = extract_scan_by_id(ds_iwv_elev, scan_ids, scan_id)
        scan_time = pd.to_datetime(ds_scan.time.values[0])

        fig, _, _ = plot_iwv_azimuth_ring(ds_scan, site, elev_sel, var2plot)
        fig.savefig(f"{output_dir}/{output_prefix}_{scan_time.strftime('%Y%m%d_%H%M%S')}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    # define param for the video creation
    target_duration_seconds = 55
    input_framerate = max(1, int(np.ceil(len(unique_scan_ids) / target_duration_seconds)))
    make_video_from_frames(input_framerate, output_prefix, output_dir, target_duration_seconds)



if __name__ == "__main__":
    main()


