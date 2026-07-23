""""
code to plot a figure containing a profile of Ze, vd, sw, sk, rain rate for the time step
with their calculated uncertainties and percentiles. The time step is selected by the user in the main function.
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



def main():

    # read MRR file for the selected site and day 
