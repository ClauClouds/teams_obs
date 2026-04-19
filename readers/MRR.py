

import xarray as xr
import gzip
import shutil


def read_MWR(file_path):
    """
    function to unzip, read and return MWR data from a NetCDF file.

    Args:
        file_path (string): full string to nc.gz file

    Returns:
        ds: xarray dataset
    """
    # unzip the file if it is gzipped
    if file_path.endswith('.gz'):

        unzipped_file_path = file_path[:-3]  # Remove the .gz extension
        with gzip.open(file_path, 'rb') as f_in:
            with open(unzipped_file_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        file_path = unzipped_file_path
    # read the NetCDF file using xarray
    try:
        ds = xr.open_dataset(file_path)
        return ds
    except Exception as e:
        print(f"Error reading MWR data from {file_path}: {e}")
        return None
