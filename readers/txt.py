"""
reader for txt file containing mesoscale pattern classification

"""



from turtle import pd
import xarray as xr
from datetime import datetime

def read_txt_file(file_name):
    """
    Reads a txt file and returns an xarray dataset with year, month, day, and pattern classification.

    Parameters:
    filename (str): The path to the txt file.

    Returns:
    xarray.Dataset: An xarray dataset with year, month, day, and pattern classification.
    """

    # initialize empty lists to store the data
    timestamps = []
    patterns = []
    # open the txt file and read the lines
    with open(file_name, 'r') as file:
        lines = file.readlines()
        # loop through the lines and extract the data
        for line in lines:
            if line[0:4] == "YYYY":
                continue
            # split the line into components
            components = line.split()
            # append the year, month, day, and pattern classification to the respective lists
            year = int(components[0])
            month = int(components[1])
            day = int(components[2])
            pattern = components[3]
            timestamps.append(datetime(year, month, day))
            patterns.append(pattern)

    
    # construct xarray dataset from the lists
    dataset = xr.Dataset(
        data_vars={
            'pattern': (('time'), patterns)
        },
        coords={
            'time': timestamps}, 
    )

    # select dataset for the campaign duration(17th May 2025 - 09 September 2025)
    dataset = dataset.sel(time=slice('2025-05-17', '2025-09-09'))
    return dataset


