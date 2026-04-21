

import os

import xarray as xr
import gzip
import shutil
import pdb
import numpy as np
import matplotlib.pyplot as plt

def read_MRR(file_path, interp_keyword, height_grid=None):
    """
    function to unzip, read and return MRR data from a NetCDF file.
    I also interpolates the profiles on a common height grid, if specified.

    Args:
        file_path (string): full string to nc.gz file
        interp_keyword (string): keyword for interpolation method (True or false)
        height_grid (array-like, optional): height grid for interpolation

    Returns:
        ds: xarray dataset
    Returns none if the file has changing range heights for different time steps, which means that interpolation cannot be done on a common height grid.
    """
    # unzip the file if it is gzipped
    if file_path.endswith('.gz'):

        # extract only filename from the file path
        filename = file_path.split("/")[-1]


        unzipped_file_path = f"/home/cacquist/{filename[:-3]}"  # Remove the .gz extension
        with gzip.open(file_path, 'rb') as f_in:
            with open(unzipped_file_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        file_path = unzipped_file_path

    # read the NetCDF file using xarray
    try:

        ds = xr.open_dataset(file_path)
        dim_time = len(ds.time.values)

        # remove the unzipped file to save space
        if file_path.endswith('.nc'):
            os.remove(file_path)

        # if interpolation is requested, interpolate the profiles on the specified height grid
        if interp_keyword:

            print("Interpolation requested. Interpolating profiles on the specified height grid.")
            if height_grid is None:
                print("No height grid provided for interpolation. Using default height grid.")

                # check if height variable (time, range) is the same for all time steps
                height_var = ds.height.values[0,:]
                if not np.all([np.array_equal(height_var, ds.height.isel(time=i).values) for i in range(dim_time)]):
                    print("Height variable is not the same for all time steps. skip the file")
                    return None
                else:
                    # change values of the range coordinate to the height variable, so that it can be used for interpolation
                    height_var = ds.height.values[0,:]
                    # drop height variable
                    ds = ds.drop_vars("height")
                    # rename range coordinate to height and assign height variable as coordinate
                    ds = ds.rename({"range": "height"})
                    ds = ds.assign_coords(height=("height", height_var))

                    # define height grid for interpolation if not provided as dim_time, dim_height
                    height_arr = np.arange(0, 5000, 100)  # Example height grid from 0 to 5000 m with 100 m intervals
                    # interpolate on new height grid
                    ds_interp = ds.interp(height=height_arr, method="linear")

            return ds_interp
        
        return ds
        
    except Exception as e:
        
        print(f"Error reading MRR data from {file_path}: {e}")
        return None
