
from readers.MWR import read_lwp_iwv
import pdb
import xarray as xr

def read_and_resample_to_fci(site_names, date, path_MWR):
    """
    function to read the MWR data for the three sites and resample it to the same time resolution as the FCI TWV data

    Args:
        site_names (list): list of names of the sites
        date (str): date in "yyyymmdd" format to read the MWR data for
        path_MWR (str): path to the directory where the MWR data is stored
    Returns:
        list: List of xarray.Dataset objects containing MWR data for each site
    """

    mwr_twv = [read_lwp_iwv(site, date, "iwv", path_MWR) for site in site_names]

    # calculate std over 10 min interval for each of the datasets in mwr_twv and store it in a new variable in the dataset
    mwr_twv_fci = []
    for i, ds in enumerate(mwr_twv):

        pdb.set_trace()
        # add std to the dataset as a new variable
        time = ds['time'].resample(time="10min").mean()
        iwv_std = ds['iwv'].resample(time="10min").std().values
        iwv = ds['iwv'].resample(time="10min").mean().values

        # define a dataset with the resampled iwv and its std and the time coordinate
        ds_fci = xr.Dataset(
            {
                'iwv': (['time'], iwv),
                'iwv_std': (['time'], iwv_std),
            },  
            coords={
                'time': time.values,
            },
            attrs={
                'site': site_names[i],
            }
        )
        mwr_twv_fci.append(ds_fci)


    return mwr_twv_fci