"""

function to read FCI TWV data from netCDF files.
"""

import glob
import pdb

from readers.data_info import path_twv, site_names
import xarray as xr
import numpy as np  
from datetime import datetime

def read_fci_twv(path_fcitwv):

    """
    function to read all fci twv files at once and return a ds xarray dataset
    input:
    path_fcitwv: path to the directory where the fci twv files are stored
    output:
    ds_fci_twv: xarray dataset containing the fci twv data

    """

    # read all files in the directory
    file_list = sorted(glob.glob(path_fcitwv + "*.nc"))

    # construct time array from file name string fci_alps_yyyymmdd_hhmmss.nc
    time_array = []
    #/Users/claudia/Documents/Data/TWV_MTG_Alps/fci_alps_20250622_0600.nc
    for file in file_list:
        print(file)
        year = file.split("/")[-1].split(".")[0].split("_")[-2][:4]
        month = file.split("/")[-1].split(".")[0].split("_")[-2][4:6]
        day = file.split("/")[-1].split(".")[0].split("_")[-2][6:8]
        hour = file.split("/")[-1].split(".")[0].split("_")[-1][:2]
        minute = file.split("/")[-1].split(".")[0].split("_")[-1][2:4]        # construct datetime string in the format "yyyymmdd_hhmmss"
        dt = datetime(year=int(year), month=int(month), day=int(day), hour=int(hour), minute=int(minute))
        # convert to datetime object        
        time_array.append(dt)
    
    nc_list = [xr.open_dataset(file) for file in file_list]
    # loop on files to read them 
    ds_fci_twv = xr.concat(nc_list, dim="time") 

    # assign time arrat as time coordinate to the dataset
    ds_fci_twv = ds_fci_twv.assign_coords(time=time_array)

    print("FCI TWV data read successfully")

    return ds_fci_twv