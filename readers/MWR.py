"""
scripts to read MWR data

"""

import gzip
import shutil
import xarray as xr
import numpy as np

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
        var_out = np.where((lwp_flag == 0) * (elev > 89.5), lwp, np.nan) 
        
    # read iwv data
    elif var == 'iwv':
        iwv = ds['iwv'].values # kgm-2
        iwv_flag = ds['iwv_quality_flag'].values
        elev = ds['elevation_angle'].values   
        units = 'kgm-2'
        # select only iwv values which have flag = 0
        var_out = np.where((iwv_flag == 0) * (elev > 89.5), iwv, np.nan) 

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