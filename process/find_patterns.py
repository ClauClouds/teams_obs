"""
With this code, we just calculate how many times during the months of the campaign we have each pattenr
and what are the most recurring patterns for the convective days and the MOBL days
"""


import pdb

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from figures.distr_patterns import plot_distr_patterns_IOP_days, plot_distr_patterns_teamx
from readers.txt import read_txt_file
from readers.data_info import path_pattern_classification, pattern_legend, iop_conv_days, iop_MoBL_T_days


def main():

    # read classification file
    ds_patterns = read_txt_file(path_pattern_classification)
    
    # find patterns for convective days and MOBL days
    patterns_conv_days = ds_patterns.sel(time=iop_conv_days)
    patterns_MoBL_T_days = ds_patterns.sel(time=iop_MoBL_T_days)

    # plot distribution of patterns for convective days and MOBL days
    plot_distr_patterns_IOP_days(patterns_conv_days, patterns_MoBL_T_days)

    # plot monthly distribution for the months of the campaign
    plot_distr_patterns_teamx(ds_patterns)

if __name__ == "__main__":
    main()