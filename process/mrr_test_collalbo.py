"""
This code should read the MRR files from IMPROtoo and from Metek and:
- remove interferences between 7 UTC and 15:30 UTC.
- understand why some rainy profiles do not show up in the quicklooks
- store corrected data in a ncdf file for publication and for further analysis.

input:
- MRR files from IMPROtoo and from Metek located in /data/campaigns/teamx/lagonero/mrr/l1/

which envitronment to use:
- source .teams_venv/bin/activate
"""
from datetime import time
import site
from turtle import mode

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from scipy.ndimage import label
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
from readers.MWR import read_MWR_flags
from mrr_io import find_MRR_flag, read_mrr_data, save_filtered_mrr_dataset
from mrr_config import load_mrr_interference_config
from process.mrr_plots import plot_time_height_Ze

def main():

    # load config
    config = load_mrr_interference_config(args.config)

    # set in input which filters to remove we want to apply:
    filter_RR_on = True  # if True, apply the rain filtering based on the MWR flags
    filter_spectra_connectivity_on = True  # if True, apply the filtering based on the connected component size in the Doppler spectra
    filter_Ze_vertical_continuity_on = False  # if True, apply the filtering based on the vertical continuity of the Ze profiles
    filter_horizontal_band_filter_on = False

    # define the sites and the path to the MRR data and MWR data
    sites = ['lagonero', 'collalbo']
    site_selected = "collalbo"
    path_mrr = f"/data/campaigns/teamx/{site_selected}/mrr/l1/"

    # time stamps to plot to understand the interferences
    time_stamps = ["20250828T03:00:00","20250828T15:00:00"]
    date_selected = time_stamps[0][:8]

    # read the MRR data for the selected site and day and store in a dataset
    ds_mrr = read_mrr_data(path_mrr, site_selected, date_selected)

    # Convert the MRR height field from time-range to a fixed vertical coordinate.
    # This keeps Ze as a clean time x range matrix and prevents height from
    # being treated as a profile variable during rain masking.
    if "height" in ds_mrr and ds_mrr["height"].dims == ("time", "range"):
        height_1d = ds_mrr["height"].median(dim="time", skipna=True)
        ds_mrr = ds_mrr.drop_vars("height").assign_coords(height=("range", height_1d.values))

    os.makedirs("plots", exist_ok=True)

    # read the MWR flags for the selected site and day and store in a dataset
    ds_mwr = read_MWR_flags(site_selected, date_selected)
    print(f"Applying MRR interference filtering for site {site_selected} on {date_selected}")
    use_mwr_rain_flag = True
    mwr_flag_exists = find_MRR_flag(site_selected, date_selected) if use_mwr_rain_flag else False

    if use_mwr_rain_flag and mwr_flag_exists:
        
        # read the MWR flags for the selected site and day and store in a dataset
        ds_mwr = read_MWR_flags(site_selected, date_selected)

        # Align categorical MWR flags to the MRR timestamps without numeric
        # interpolation, and report the actual nearest-neighbor time offsets.
        mwr_time_index = pd.DatetimeIndex(ds_mwr.time.values)
        mrr_time_index = pd.DatetimeIndex(ds_mrr.time.values)
        nearest_mwr_index = mwr_time_index.get_indexer(mrr_time_index, method="nearest")
        valid_nearest = nearest_mwr_index >= 0
        nearest_offsets = pd.Series(
            np.abs(mwr_time_index[nearest_mwr_index[valid_nearest]] - mrr_time_index[valid_nearest])
        )
        if not nearest_offsets.empty:
            print(
                "MWR/MRR nearest-time offsets: "
                f"median={nearest_offsets.median()}, "
                f"95%={nearest_offsets.quantile(0.95)}, "
                f"max={nearest_offsets.max()}"
            )

        mwr_reindex_tolerance = pd.Timedelta(config.mwr_reindex_tolerance)
        rain_extension = pd.Timedelta(config.rain_extension)
        ds_mwr_interp = ds_mwr.reindex(
            time=ds_mrr.time,
            method="nearest",
            tolerance=mwr_reindex_tolerance,
        )

        unmatched_mwr_flags = int(ds_mwr_interp.rain.isnull().sum())
        if unmatched_mwr_flags:
            print(
                f"MWR rain flag unmatched for {unmatched_mwr_flags} MRR profiles "
                f"using tolerance {mwr_reindex_tolerance}. Treating them as not rainy."
            )

        rain_flag_raw = ds_mwr_interp.rain.fillna(0).astype(bool)
        mrr_time_step = pd.Series(mrr_time_index).diff().dropna().median()
        if pd.isna(mrr_time_step) or mrr_time_step <= pd.Timedelta(0):
            rain_extension_steps = 0
        else:
            rain_extension_steps = int(np.ceil(rain_extension / mrr_time_step))

        rain_flag = rain_flag_raw.rolling(
            time=2 * rain_extension_steps + 1,
            center=True,
            min_periods=1,
        ).max().astype(bool)

        print(
            "MWR rain-flag expansion on MRR grid: "
            f"raw={int(rain_flag_raw.sum())} profiles, "
            f"expanded={int(rain_flag.sum())} profiles, "
            f"buffer=+/-{rain_extension}."
        )

    # plot the MRR data for the selected site 
    plot_time_height_Ze(ds_mrr, date_selected, info_output="test_only_rain", time_stamps=time_stamps)
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MRR interference filtering")
    parser.add_argument(
        "--config",
        type=str,
        default="process/remove_interfence_mrr_config.yaml",
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()

    main()