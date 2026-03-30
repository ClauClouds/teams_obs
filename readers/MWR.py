"""
scripts to read MWR data

"""

from datetime import datetime
import gzip
import shutil
import site
import xarray as xr
import numpy as np
import pandas as pd
import pdb
from progress.bar import Bar
from readers.data_info import MWR_SITES_NAMES


def read_MWR_flags(site, date):
    """
    code to read MWR radiometer flags for a given site 

    Args:
        site (str): site name
            Options: 'lagonero', 'collalbo' 
        date (str): date in 'yyyymmdd' format
        
    Returns:
        flags (dict): dictionary containing flags for the site
            - 'rain': boolean, True if rain is detected, False otherwise
            - 'snow': boolean, True if snow is detected, False otherwise
            - 'clouds': boolean, True if clouds are detected, False otherwise
    """
    if site == 'collalbo':
        instr = 'kithat'
    elif site == 'lagonero':
        instr = 'tophat'
    elif site == 'bolzano':
        instr = 'hatpro'

    path_root = '/data/obs/campaigns/teamx/' + site + '/'+ instr +'/actris/level1/'
    
    
    # read yy, mm, dd from date
    yy = date[:4]
    mm = date[4:6]
    dd = date[6:8]
    
    path_global = path_root + yy + '/' + mm + '/' + dd + '/'
    filename = path_global + 'MWR_1C01_'+site+'_'+date+'.nc'
    
    ds = xr.open_dataset(filename)
    
    # read rain rate variable
    rain_rate = ds['rainfall_rate'].values
    
    # select time stamps whe
    
    # create a rain flag: when quality flag == 6 set rain flag to 1, otherwise set to 0
    rain_flag = np.zeros(len(rain_rate), dtype=int)
    ind_rain = np.where(rain_rate > 0.)[0]
    rain_flag[ind_rain] = 1

    time = ds.time.values
    
    # create an output dataset with time as coordinate
    ds = xr.Dataset(
        {
            'rain': (['time'], rain_flag),
        },
        coords={
            'time': time
        }
    )
    
    return ds


def read_lwp_iwv(site, date, var, path_root):
    """
    function to read LWP or IWV values from level 2 MWR radiometer data postprocessed 
    following actris scripts from
    Tobias Marke

    Args:
        site (str): site name
            Options: 'lagonero', 'collalbo', 'bolzano'
        date (str): date in 'yyyymmdd' format
        var (str): variable to read, options are 'lwp' or 'iwv'
        path_root (str): path to the directory where the MWR data is stored

    Returns:
        xarray.DataArray: Liquid Water Path (LWP) or Integrated Water Vapor (IWV) for 
        the given site and date in gm-2 (LWP) or kgm-2 (IWV)

    """

    if site == 'collalbo':
        instr = 'kithat'
    elif site == 'lagonero':
        instr = 'tophat'
    elif site == 'bolzano':
        instr = 'hatpro'
    
    # read yy, mm, dd from date
    yy = date[:4]
    mm = date[4:6]
    dd = date[6:8]

    # set filename
    filename = f"{path_root}MWR_single_{site}_{yy}{mm}{dd}.nc"

    # read dataset
    ds = xr.open_dataset(filename) 

    # read lwp data 
    if var == 'lwp':

        lwp = ds['lwp'].values * 1000 # LWP in gm-2
        lwp_flag = ds['lwp_quality_flag'].values
        elev = ds['elevation_angle'].values
        units = 'gm-2'
        # select only lwp values which have flag = 0
        var_out = np.where((lwp_flag == 0) * (elev > 80.), lwp, np.nan) 

        # offset correction for lagonero to avoid negative lwp values
        ds_offset = read_offset_correction_lwp(site, date, path_root)

        # add offset correction to lwp values
        var_out = var_out + ds_offset.offset_correction.values  
        
    # read iwv data
    elif var == 'iwv':
        iwv = ds['iwv'].values # kgm-2
        iwv_flag = ds['iwv_quality_flag'].values
        elev = ds['elevation_angle'].values   
        units = 'kgm-2'
        # select only iwv values which have flag = 0
        var_out = np.where((iwv_flag == 0) * (elev > 80.), iwv, np.nan) 

    # build output dataset with lwp or iwv, time and site
    ds_out = xr.Dataset(
        {
            var: (['time'], var_out),
        },
        coords={
            'time': ds.time.values,
            'site': site,
            'units': units
        }
    )


    return ds_out


def read_offset_correction_lwp(site, day, path_root):

    """
    function to read the offset correction csv file for LWP for the given site and date from the MWR data

    Args:
        site (str): site name   
            Options: 'lagonero', 'collalbo', 'bolzano'  
        date (str): date in 'yyyymmdd' format
        path_root (str): path to the directory where the MWR data is stored

    Returns:
        float: offset correction value to apply to the LWP values for the given site and date
    """
    
    if site == 'collalbo':
        instr = 'kithat'
    elif site == 'lagonero':
        instr = 'tophat'
    elif site == 'bolzano':
        instr = 'hatpro'
    
    # construct path 
    yy = '2025'
    path_csv = f"/data/obs/campaigns/teamx/{site}/{instr}/actris/level2/{yy}/"
    # set filename
    filename = f"{path_csv}{site}_lwp_offset_2025.csv"

    # read csv file and extract the offset correction value for the date
    df = pd.read_csv(filename)

    # extract from date the month and the day since date is mm-dd 
    times_string = df['date'].values

    # loop on time stamps
    time_arr = []
    offset_arr = []

    for i in range(len(times_string)):

        # extract mm and dd from times_string elements, since they are in the format mm-dd  
        mm, dd = times_string[i].split('-')

        # construct datetime from yy, mm and dd
        time = datetime(int(yy), int(mm), int(dd))
        time_arr.append(time)

        offset_arr.append(df['offset'].values[i])    
    
    # store arrays in xarray dataset
    ds_offset = xr.Dataset(
        {
            'offset_correction': (['time'], offset_arr),
        },
        coords={
            'time': time_arr
        }
    )

    # select the offset correction value for the date of interest
    try:
        ds_offset_sel = ds_offset.sel(time=datetime.strptime(day, "%Y%m%d"))
    except KeyError:
        print(f"No offset correction value found for date {day}. Setting offset correction to 0.")
        ds_offset_sel = xr.Dataset(
            {
                'offset_correction': (['time'], [0]),
            },
            coords={
                'time': [datetime.strptime(day, "%Y%m%d")]
            }
        )

    return ds_offset_sel


def read_all_data_for_campaign(site, var_type):
    """
    function to read all MWR data for the entire campaign . Interpolate them on a 3 s time resolution 
    and store them in a matrix which is returned as output. 
    This is useful to calculate the mean diurnal cycle and the standard deviation among days 
    for each time step of the diurnal cycle.
    input:
        site (str): site name   
            Options: 'lagonero', 'collalbo', 'bolzano'  
        var_type (str): variable to read, options are 'lwp' or 'iwv'    
    output:
        all_files (list): list of all files read for the campaign
        var_matrix (numpy array): matrix containing the variable values for all days and time steps, with shape (N_days, N_time_steps)
    """
    
    # path to files
    path_file_sel = f"/data/obs/campaigns/teamx/{site}/{MWR_SITES_NAMES[site]}/actris/level2/2025/"

    # list all .nc files in all the directories under the path containing the string MWR_single
    import os 
    all_files = []
    for root, dirs, files in os.walk(path_file_sel):
        for file in files:
            if file.endswith(".nc") and "MWR_single" in file:
                all_files.append(os.path.join(root, file))
    
    # loop on all the files to read the data
    N_days = len(all_files)
    var_matrix = np.empty((N_days, 24*60*60//3)) # matrix to store lwp values for all days and hours)) # initialize matrix to store the data, with shape (N_days, 24) since we want to store the diurnal cycle of each day
    
    # define time array with 3s resolution for plotting diurnal cycle
    d_cycle_start = datetime(year=2000, month=1, day=1, hour=0, minute=0, second=0)
    d_cycle_end = datetime(year=2000, month=1, day=1, hour=23, minute=59, second=59)
    time_cycle = pd.date_range(start=d_cycle_start, end=d_cycle_end, freq='3s')


    # loop on days to be plotted
    # add progress bar to the loop using bar
    with Bar("Reading MWR files", max=len(all_files)) as bar:
        for i_day in range(len(all_files)):

            # remove filename from the file path to get the path to the directory where the file is stored
            dir_path = os.path.dirname(all_files[i_day])+'/'
            # extract day string from file path
            day_string = all_files[i_day].split('/')[-1].split('_')[-1][:-3] # extract the date string from the file name, which is in the format MWR_single_{site}_{yyyymmdd}.nc
            
            print(f" - day {day_string}...")

            # reading data from single day dataset
            ds_day = read_lwp_iwv(site, day_string, var_type, dir_path)

            # Use a common reference date so all days can be interpolated on the same diurnal cycle.
            time_of_day = pd.to_datetime(ds_day.time.values).strftime('2000-01-01 %H:%M:%S')
            ds_day = ds_day.assign_coords(time=pd.to_datetime(time_of_day))

            # resample on 3 secondly time steps with linear interpolation to have the same time stamps for all days (and then calculate mean and std)
            ds_day = ds_day.interp(time=time_cycle)

            # search small nan gaps (less than 10 min) and fill them with linear interpolation to avoid having too many nans in the time series of the days (which would lead to having few values to calculate mean and std among days for each time step)
            ds_day = ds_day.interpolate_na(dim='time', method='linear', limit=10*60//3)

            var_values = ds_day[var_type].values
            var_matrix[i_day, :] = var_values
            bar.next()


    return var_matrix, time_cycle


