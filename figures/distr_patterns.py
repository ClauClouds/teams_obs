"""
scripts to plot distributions for the diffeents mesoscale patterns during the campaign and for the convective days and the MOBL days
author: Claudia Acquistapace
date: 2025-03-16
contact: claudia.acquistapace-at_unipd.it


"""
from readers.data_info import path_pattern_classification, pattern_legend, iop_conv_days, iop_MoBL_T_days
from readers.txt import read_txt_file
import matplotlib.pyplot as plt
import pandas as pd

def plot_distr_patterns_IOP_days(patterns_conv_days, patterns_MoBL_T_days):
    """
    Plots the distribution of mesoscale patterns for the convective days and the MOBL days.

    Parameters:
    patterns_conv_days (xarray.Dataset): An xarray dataset with the pattern classification for the convective days.
    patterns_MoBL_T_days (xarray.Dataset): An xarray dataset with the pattern classification for the MOBL days.

    Returns:
    None: The function saves the plot as a png file.
    """

    # count the number of occurrences of each pattern for the convective days and the MOBL days
    pattern_conv = patterns_conv_days['pattern'].values
    pattern_MoBL_T = patterns_MoBL_T_days['pattern'].values

    pattern_ids = [str(i) for i in range(len(pattern_legend))]
    distribution_conv = pd.Series(pattern_conv).value_counts().reindex(pattern_ids, fill_value=0)
    distribution_MoBL_T = pd.Series(pattern_MoBL_T).value_counts().reindex(pattern_ids, fill_value=0)

    # normalize distribution over the total numer of days for the convective days and the MOBL days
    distribution_conv = distribution_conv / len(patterns_conv_days['time'])
    distribution_MoBL_T = distribution_MoBL_T / len(patterns_MoBL_T_days['time'])

    # plot the distribution of patterns for the convective days and the MOBL days
    fig, ax = plt.subplots(figsize=(15, 8))
    x_positions = list(range(len(pattern_ids)))
    bar_width = 0.4
    ax.bar(
        [x - bar_width / 2 for x in x_positions],
        distribution_conv.values,
        width=bar_width,
        label='Convective ',
        alpha=0.7,
    )
    ax.bar(
        [x + bar_width / 2 for x in x_positions],
        distribution_MoBL_T.values,
        width=bar_width,
        label='MOBL thermal',
        alpha=0.7,
    )
    ax.set_xticks(x_positions)
    ax.set_xticklabels([pattern_legend[pattern_id] for pattern_id in pattern_ids], rotation=45, ha='right')
    ax.set_ylabel('Number of Occurrences')
    ax.set_title('Distribution of Mesoscale Patterns for Convective Days and MOBL Days')
    ax.legend(frameon=False)
    # set font size of legend
    ax.legend(fontsize=14)
    # remove top and right spines and make the other two thicker
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    # increase font size of the labels and title
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.set_title('Distribution of Mesoscale Patterns for Convective Days and MOBL Days', fontsize=18)
    ax.set_ylabel('Normalized Occurrences', fontsize=15) 
    plt.tight_layout()
    plt.savefig('plots/distribution_patterns_IOP_days.png')

    return None


def plot_distr_patterns_teamx(ds_patterns):
    """
    Plots the distribution of mesoscale patterns for the months of the campaign.

    Parameters:
    ds_patterns (xarray.Dataset): An xarray dataset with the pattern classification for the months of the campaign.

    Returns:
    None: The function saves the plot as a png file.
    """

    # make one plot with all months for the campaign 
    
    # plot the distribution of patterns for the month
    fig, ax = plt.subplots(figsize=(15, 8))
    pattern_ids = [str(i) for i in range(len(pattern_legend))]
    x_positions = list(range(len(pattern_ids)))

    months_labels = ['May', 'June', 'July', 'August', 'September']
    months = ['05', '06', '07', '08', '09']
    bar_width = 0.8 / len(months)

    # loop on the months (05, 06, 07, 08, 09) 
    for i_month, month in enumerate(months):

        # select the patterns for the month
        patterns_month = ds_patterns.sel(time=ds_patterns['time.month'] == int(month))['pattern'].values
        month_label = months_labels[i_month]

        # count the number of occurrences of each pattern for the month

        distribution_month = pd.Series(patterns_month).value_counts().reindex(pattern_ids, fill_value=0)

        # normalize distribution over the total number of days for the month
        distribution_month = distribution_month / len(patterns_month)
        month_offset = (i_month - (len(months) - 1) / 2) * bar_width
        ax.bar(
            [x + month_offset for x in x_positions],
            distribution_month.values,
            width=bar_width,
            label=month_label,
            alpha=0.7,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([pattern_legend[pattern_id] for pattern_id in pattern_ids], rotation=45, ha='right')

    # set font size of title and labels
    ax.set_title('Distribution of Mesoscale Patterns During the Campaign', fontsize=18)
    ax.set_ylabel('Normalized Occurrences', fontsize=15)
    ax.legend(frameon=False, fontsize=14)
    # remove top and right spines and make the other two thicker
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    # increase font size of x and y ticks
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    # increase tick lenght of minor ticks and major ticks
    ax.tick_params(axis='x', which='minor', length=3)
    ax.tick_params(axis='x', which='major', length=7)
    ax.tick_params(axis='y', which='minor', length=3)
    ax.tick_params(axis='y', which='major', length=7)

    plt.tight_layout()
    plt.savefig(f'plots/distribution_patterns_month.png')
    return None
