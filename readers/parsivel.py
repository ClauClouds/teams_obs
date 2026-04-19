import os

import xarray as xr
import gzip
import shutil


def read_parsivel(file_path):
    """
    function to unzip, read and return parsivel data from a NetCDF file.

    Args:
        file_path (string): full string to nc.gz file

    Returns:
        ds: xarray dataset
    """
    # unzip the file if it is gzipped
    if file_path.endswith('.gz'):

        # check if there is already an unzipped version of the file, if yes, use it, if not, unzip the file
        unzipped_file_path = file_path[:-3]  # Remove the .gz extension
        if not os.path.exists(unzipped_file_path):
            with gzip.open(file_path, 'rb') as f_in:
                with open(unzipped_file_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

        file_path = unzipped_file_path
        
    # read the NetCDF file using xarray
    try:
        ds = xr.open_dataset(file_path)
        # remove the unzipped file after reading
        if file_path.endswith('.nc'):
            os.remove(file_path)
        return ds
    except Exception as e:
        print(f"Error reading parsivel data from {file_path}: {e}")
        return None

