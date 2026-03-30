"""
code to calculate mean IWV diurnal cycle over the entire campaign.
Then, for every day of one of the categories 'convective' or 'mobl-t', 
it calculates the anomaly compared to the mean diurnal cycle.
It then derives the mean anomaly diurnal cycle for the convective days 
and for the MOBL-T days and plots them together in one plot.

to call, first activate the conda environment
source .teams_venv/bin/activate
 with the required packages, then run:
python figures/anomalies.py



"""
from readers.data_info import PLOT_SITES_NAMES, MWR_SITES_NAMES, iop_conv_days, iop_MoBL_T_days
from readers.MWR import read_lwp_iwv, read_all_data_for_campaign
import matplotlib.pyplot as plt
from figures.plotting import plot_diurnal_cycle_single_site, plot_hourly_percentiles, plot_mean_dc
import numpy as np   
import pandas as pd
import xarray as xr
import pdb
from datetime import datetime

def main():

    # select if plotting is on or off
    plot_on = False # set to False to turn off plotting and just calculate anomalies
    types_days = ["convective", "MOBL_T"]
    sites = ["lagonero", "collalbo", "bolzano"]

    # loop on types of days and sites to calculate anomalies for each combination
    for day_type in types_days:
        for site in sites: 

            # select which group of days to plot the diurnal cycle for
            var_type = "iwv" # "lwp" or "iwv"

            # read list of days for the convective days and the MOBL thermal days
            if day_type == "convective":
                days = iop_conv_days
            elif day_type == "MOBL_T":
                days = iop_MoBL_T_days
            else:
                raise ValueError("day_type must be either 'convective' or 'MOBL_T'")
            
            # derive matrix containing data
            var_matrix, time_dc = read_all_data_for_campaign(site, var_type)

            # calculate mean and standard deviation for each time step of the diurnal cycle
            ds_mean_dc = np.nanmean(var_matrix, axis=0)
            ds_std_dc = np.nanstd(var_matrix, axis=0)

            # plot diurnal cycle time series
            if plot_on:
                plot_mean_dc(site, time_dc, ds_mean_dc, ds_std_dc, var_type)

            # calculate anomalies for each day of the selected type (convective or MOBL-T) compared to the mean diurnal cycle
            ds_anomalies = []
            for i, day in enumerate(days):

                # read yy and month from day string
                yy = day[0:4]
                mm = day[4:6]
                dd = day[6:8]

                # find right path to the selected file
                path_file_sel = f"/data/obs/campaigns/teamx/{site}/{MWR_SITES_NAMES[site]}/actris/level2/{yy}/{mm}/{dd}/"
                # read data for the day with the specific variable type (LWP or IWV)
                ds_day = read_lwp_iwv(site, day, var_type, path_file_sel)

            # Use a common reference date so all days can be interpolated on the same diurnal cycle.
                time_of_day = pd.to_datetime(ds_day.time.values).strftime('2000-01-01 %H:%M:%S')
                ds_day = ds_day.assign_coords(time=pd.to_datetime(time_of_day))

                # resample on 3 secondly time steps with linear interpolation to have the same time stamps for all days (and then calculate mean and std)
                ds_day = ds_day.interp(time=time_dc)

                # search small nan gaps (less than 10 min) and fill them with linear interpolation to avoid having too many nans in the time series of the days (which would lead to having few values to calculate mean and std among days for each time step)
                ds_day_interp = ds_day.interpolate_na(dim='time', method='linear', limit=10*60//3)
                
                # calculate anomaly compared to mean diurnal cycle
                var_day = ds_day_interp[var_type].values
                anomaly_day = var_day - ds_mean_dc

                # construct fake time array for the day (assuming the same time steps as the mean diurnal cycle)
                
                # store anomaly in xarray dataset
                ds_anomaly_day = xr.Dataset(
                    {
                        'anomaly': (['time'], anomaly_day),
                    },      
                    coords={
                        'time': ds_day_interp.time.values
                    }
                ).expand_dims(day=[pd.Timestamp(day)])

                ds_anomalies.append(ds_anomaly_day)
            
            # concatenate all anomaly datasets into one dataset
            ds_anomalies_all = xr.concat(ds_anomalies, dim='day')

            # calculatte mean anomaly for the selected day type (convective or MOBL-T)
            mean_anomaly = ds_anomalies_all.mean(dim='day', skipna=True)
            std_anomaly = ds_anomalies_all.std(dim='day', skipna=True)
            time_anomaly = ds_anomalies_all.time.values

            # store mean_anomaly in xarray dataset
            ds_mean_anomaly = xr.Dataset(
                {
                    'mean_anomaly': (['time'], mean_anomaly.anomaly.values),
                    'std_anomaly': (['time'], std_anomaly.anomaly.values)
                },  
                coords={
                    'time': time_anomaly
                }
            )
            # save to ncdf
            ds_mean_anomaly.to_netcdf(f"/home/cacquist/Documents/GitHub/EXPATS/teams_obs/ncdf_anomalies/mean_anomaly_{day_type}_{var_type}_{site}.nc") 
            
        

if __name__ == "__main__":
    main()