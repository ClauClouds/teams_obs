

import pdb
import site
import numpy as np
import xarray as xr
from readers.FCI_TWV import read_fci_twv



def haversine(lat1, lon1, lat2, lon2):
    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    R = 6371  # Earth radius in kilometers
    return R * c



def find_closest_pixel(path_twv, site_lats, site_lons, site_names):
    """
    function to find the closest pixel to the location of the MWR from the FCI TWV
    for the three sites

    Args:
        ds_fci_twv (xarray.Dataset): FCI TWV dataset
        site_lats (list): list of latitudes of the sites
        site_lons (list): list of longitudes of the sites
        site_names (list): list of names of the sites
    Returns:
        list: List of xarray.Dataset objects containing FCI TWV values for
        the closest pixel to the location of the MWR for each site

    Dependency:
        read_fci_twv function to read the FCI TWV data from the netCDF files
    """

    # read the FCI TWV data
    ds_fci_twv = read_fci_twv(path_twv)

    # read values of lat, lon and twv from the dataset
    lat = ds_fci_twv.lat.values[0,:,:]
    lon = ds_fci_twv.lon.values[0,:,:]
    twv = ds_fci_twv.tcwv.values
    twv_uncertainty = ds_fci_twv.tcwv_uncertainty.values

    # find the value of lat and lon closest to the site location for each site
    twv_list = []
    for i_site, site_lat in enumerate(site_lats):

        # read site lon
        site_lon = site_lons[i_site]

        # find indeces the minimum difference between lat and site_lat and between lon and site_lon 
        dist = haversine(lat, lon, site_lat, site_lon)
        min_idx = np.unravel_index(np.argmin(dist), lat.shape)
        closest_lat = lat[min_idx[0], min_idx[1]]
        closest_lon = lon[min_idx[0], min_idx[1]]
        print(f"site lat and lon are {site_lat} and {site_lon}")
        print(f"closest lat and lon are {closest_lat} and {closest_lon}")

        twv_closest = twv[:, min_idx[0], min_idx[1]]  
        twv_uncertainty_closest = twv_uncertainty[:, min_idx[0], min_idx[1]]  
        print(f"twv closest pixel timeseries is {twv_closest}")

        # store info in xarray dataset
        ds = xr.Dataset(
            {
                'twv': (['time'], twv_closest),
                'twv_uncertainty': (['time'], twv_uncertainty_closest)  
            },
            coords={
                'time': ds_fci_twv.time.values
            },
            attrs={
                'site_lat': site_lat,
                'site_lon': site_lon,
                'lat_closest': closest_lat,
                'lon_closest': closest_lon,
                "units": "kg/m2", 
                "site_name": site_names[i_site]
            }
        )
        twv_list.append(ds)


    return twv_list
