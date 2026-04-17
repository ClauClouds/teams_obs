"""
file containing the paths to the data files, directories, and output paths for the plots.
"""
import numpy as np


# TWV data from MTG FCI provided by Cintia Carbajßl
path_twv = "/Users/claudia/Documents/Data/TWV_MTG_Alps/"

# MWR radiometer path
path_MWR = "/Users/claudia/Documents/Data/MWR_TWV_comparison/"

# orography path
orography_path = "/home/cacquist/Documents/GitHub/EXPATS/orography_expats_high_res.nc"

# sites for comparison of TWV from MTG FCI and MWR radiometer
site_names = ["bolzano", "collalbo", "lagonero"]

# name strings to plot for the sites
PLOT_SITES_NAMES = {
    "bolzano": "Bozen",
    "collalbo": "Klobenstein",
    "lagonero": "Schwartzseespitze"}

MWR_SITES_NAMES = {
    "bolzano": "hatpro",
    "collalbo": "kithat",
    "lagonero": "tophat"}

# lats and lons of the sites
site_lats = [46.49067, 46.53965, 46.59605]
site_lons = [11.33982, 11.45832, 11.45255]


# definition of domains of interest
domain_German_flood =  [ 5.,    9.,    48.,   52.  ] # minlon, maxlon, minlat, maxlat
domain_expats       =  [ 5.,   16.,    42.,   51.5 ] # minlon, maxlon, minlat, maxlat
domain_joyce        =  [ 6.,   6.5,    50.8,  51.3 ] # minlon, maxlon, minlat, maxlat   
domain_ACTA         =  [ 10.73,12.0,   46.3,  47.2 ] # minlon, maxlon, minlat, maxlat
domain_TEAMX        =  [ 9.9,  12.7,   45.5,  47.4 ] # minlon, maxlon, minlat, maxlat


# IOP convective days

iop_conv_days = [ "20250625","20250630", "20250701", "20250705", "20250719", "20250722", "20250723", "20250724"]
iop_MoBL_T_days = ["20250624", "20250628", "20250711", "20250712", "20250715", "20250716", "20250718", "20250720"]


# mesoscale pattern classification from txt file
path_pattern_classification = "/home/cacquist/Documents/GitHub/EXPATS/teams_obs/ERA5_pseudoPCs_labels_noreg_xr_4025.txt"


# pattern legend dictionary (from Ilaria's notes)
pattern_legend = {
    "0": "0: Scandinavian Trough (ScTr)",
    "1": "1: European Blocking (EuBL)",
    "2": "2: Atlantic Ridge (AR)",
    "3": "3: Greenland Blocking (GrBL)",
    "4": "4: Scandinavian Blocking (ScBL)",
    "5": "5: Zonal Regime (ZO)",
    "6": "6: Atlantic Trough (ATr)",
    "7": "7: No Regime (NoReg) or transition",
}


# hourly intervals for calculating the diurnal cycle of LWP and IWV
# set hours to plot and time steps array for single plotting without averages 6,8,10,12,14,16
hours_diurnal_cycle_calc = ["06:00", "08:00", "10:00", "12:00", "14:00", "16:00", "18:00", "20:00", "22:00"]
azimuth_bins = np.arange(0, 360, 20) # azimuth angle bins of 20 degrees for calculating mean IWV over the azimuth scan for each time selection of the diurnal cycle


# path fci data
fci_path = "/data/trade_pc/mtg/fci/processed/no_parallax/original_grid/2025"