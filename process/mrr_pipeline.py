"""Campaign-level MRR postprocessing pipeline.

This module runs one-day or campaign-time window MRR postprocessing using the
interference filtering helpers from ``remove_interfence_mrr``. The active site,
dates, interference window, and site-specific filter switches are read from
``process/remove_interfence_mrr_config.yaml``.

Common processing applied to configured sites:
- read the MRR level-1 daily file for the selected site and date;
- preserve time-dependent MRR height so each profile keeps the height vector
  valid at its timestamp;
- read the MWR rain flag, reindex it to MRR time with the configured tolerance,
  and expand rainy periods by the configured rain-extension buffer;
- mask non-rain MRR profiles inside the configured interference window unless
  they are protected by the deep-continuous-Ze or lower-continuous-Ze tests;
- optionally calculate MRR moment and rain-rate uncertainties;
- optionally save diagnostic before/after Ze quicklooks and the processed
  NetCDF dataset.

Lagonero processing:
- uses the Lagonero interference window from the YAML, currently the cable-car
  period;
- keeps ``keep_lowest_connected_component: true``. After upper-interference
  detection, detached upper Ze/W components are removed by keeping only the
  lowest connected rain column. This preserves the existing Lagonero setup.

Collalbo processing:
- uses the Collalbo interference window from the YAML;
- protects profiles whose echo reaches the highest range gate and extends
  downward by at least ``top_rooted_min_vertical_extent``;
- removes short detached upper Ze fragments above lower-rooted profiles using
  ``remove_short_detached_ze_columns``;
- uses the MWR rain flag, but sets
  ``apply_upper_interference_masking: false`` and
  ``keep_lowest_connected_component: false`` to avoid over-chopping rooted,
  vertically gappy rain profiles. Elevated-only profile removal remains active
  except for profiles protected by the top-rooted check.

How to run:
source .teams_venv/bin/activate
python process/mrr_pipeline.py --config process/remove_interfence_mrr_config.yaml
nohup python process/mrr_pipeline.py --config process/remove_interfence_mrr_config.yaml > logs/log_collalbo_20250516_20250909.log 2>&1 & 
PID 187196
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from readers.MWR import read_MWR_flags

try:
    from process.mrr_uncertainty import mrr_moment_uncertainty
except ModuleNotFoundError:
    from mrr_uncertainty import mrr_moment_uncertainty

try:
    from process.mrr_config import load_mrr_interference_config
    from process.mrr_interference import (
        PROFILE_MASK_SKIP_VARS,
        _mark_true_runs,
        apply_range_gate_mask_to_profile,
        calculate_mean_interference_vertical_extent,
        check_profile,
        keep_lowest_connected_ze_component,
        mask_short_detached_ze_columns,
        mask_upper_interference,
        mrr_has_continuous_ze_starting_below_height,
        mrr_has_deep_continuous_ze,
        mrr_has_lower_echo,
        mrr_has_top_rooted_ze_extent,
        mrr_is_elevated_only_ze_profile,
    )
    from process.mrr_io import find_MRR_flag, read_mrr_data, save_filtered_mrr_dataset
    from process.mrr_plots import plot_time_height_Ze
except ModuleNotFoundError:
    from mrr_config import load_mrr_interference_config
    from mrr_interference import (
        PROFILE_MASK_SKIP_VARS,
        _mark_true_runs,
        apply_range_gate_mask_to_profile,
        calculate_mean_interference_vertical_extent,
        check_profile,
        keep_lowest_connected_ze_component,
        mask_short_detached_ze_columns,
        mask_upper_interference,
        mrr_has_continuous_ze_starting_below_height,
        mrr_has_deep_continuous_ze,
        mrr_has_lower_echo,
        mrr_has_top_rooted_ze_extent,
        mrr_is_elevated_only_ze_profile,
    )
    from mrr_io import find_MRR_flag, read_mrr_data, save_filtered_mrr_dataset
    from mrr_plots import plot_time_height_Ze


def _height_profile_for_time(ds: xr.Dataset, time_stamp) -> np.ndarray:
    """Return the height vector that belongs to one MRR profile."""
    height = ds["height"]
    if "time" in height.dims:
        return height.sel(time=time_stamp).values
    return height.values


def average_mrr_dataset_over_time(ds: xr.Dataset, frequency: str | None) -> xr.Dataset:
    """Average time-dependent MRR variables over the requested frequency."""
    if not frequency:
        return ds

    bool_vars = [
        name
        for name, data_array in ds.data_vars.items()
        if "time" in data_array.dims and np.issubdtype(data_array.dtype, np.bool_)
    ]
    ze_vars = ["Ze"] if "Ze" in ds and "time" in ds["Ze"].dims else []
    numeric_vars = [
        name
        for name, data_array in ds.data_vars.items()
        if (
            "time" in data_array.dims
            and name not in bool_vars
            and name not in ze_vars
            and np.issubdtype(data_array.dtype, np.number)
        )
    ]
    static_vars = [
        name
        for name, data_array in ds.data_vars.items()
        if "time" not in data_array.dims
    ]

    pieces = []
    if numeric_vars:
        pieces.append(
            ds[numeric_vars]
            .resample(time=frequency, label="right", closed="right")
            .mean(dim="time", skipna=True, keep_attrs=True)
        )
    if ze_vars:
        ze_attrs = ds["Ze"].attrs.copy()
        ze_linear = 10.0 ** (ds["Ze"] / 10.0)
        ze_linear_mean = ze_linear.resample(
            time=frequency, label="right", closed="right"
        ).mean(dim="time", skipna=True, keep_attrs=True)
        ze_mean = 10.0 * np.log10(ze_linear_mean.where(ze_linear_mean > 0.0))
        ze_mean.name = "Ze"
        ze_mean.attrs.update(ze_attrs)
        ze_mean.attrs["postprocessing_time_average"] = (
            "Averaged in linear reflectivity units, then converted back to dBZ."
        )
        pieces.append(ze_mean.to_dataset())
    if bool_vars:
        bool_ds = (
            ds[bool_vars]
            .astype(int)
            .resample(time=frequency, label="right", closed="right")
            .max(dim="time", skipna=True, keep_attrs=True)
            .astype(bool)
        )
        pieces.append(bool_ds)
    if static_vars:
        pieces.append(ds[static_vars])

    if not pieces:
        return ds

    averaged = xr.merge(pieces, compat="override")
    averaged.attrs.update(ds.attrs)
    averaged.attrs["postprocessing_final_time_average"] = frequency
    averaged.attrs["postprocessing_final_time_average_Ze_rule"] = (
        "Ze is converted from dBZ to linear units before averaging, then "
        "converted back to dBZ."
    )
    averaged.attrs["postprocessing_final_time_average_bool_rule"] = (
        "Boolean time-dependent variables are true when any source profile "
        "inside the averaging bin is true."
    )
    return averaged


@dataclass(frozen=True)
class ProcessedMRRDay:
    """Result returned by the daily MRR postprocessing pipeline."""

    dataset: xr.Dataset
    output_file: Path | None
    output_files: list[Path]


def process_mrr_day(config, *, date_selected: str | None = None, make_plots: bool = True, save_output: bool | None = None) -> ProcessedMRRDay:
    """Read, filter, annotate, and optionally save one MRR day."""

    # Read the configuration values for the selected site and day
    sites = config.sites
    site_selected = config.site_selected
    path_mrr = config.path_mrr
    time_stamps = config.time_stamps
    date_selected = date_selected or time_stamps[0][:8]

    # Read the filtering parameters from the config file
    filter_RR_on = config.filter_RR_on
    min_interference_extent_factor = config.min_interference_extent_factor
    min_interference_layer_fraction = config.min_interference_layer_fraction
    interference_min_lowest_echo_height = config.interference_min_lowest_echo_height
    min_interference_time_profiles = config.min_interference_time_profiles
    min_lower_echo_time_profiles = config.min_lower_echo_time_profiles
    min_elevated_connected_gates = config.min_elevated_connected_gates
    min_lower_echo_connected_gates = config.min_lower_echo_connected_gates
    max_interference_missing_gates = config.max_interference_missing_gates
    max_rain_column_missing_gates = config.max_rain_column_missing_gates
    protect_top_rooted_profiles = config.protect_top_rooted_profiles
    top_rooted_min_vertical_extent = config.top_rooted_min_vertical_extent
    keep_lowest_connected_component = config.keep_lowest_connected_component
    apply_upper_interference_masking = config.apply_upper_interference_masking
    remove_short_detached_ze_columns = config.remove_short_detached_ze_columns
    max_detached_ze_column_vertical_extent = config.max_detached_ze_column_vertical_extent
    min_detached_ze_column_base_height = config.min_detached_ze_column_base_height
    lower_echo_height_limit = config.lower_echo_height_limit
    min_lower_echo_peak_ze = config.min_lower_echo_peak_ze
    min_lower_continuous_ze_gates = config.min_lower_continuous_ze_gates

    # read the MRR data for the selected site and day and store in a dataset
    print(f"Reading MRR data for site {site_selected} on {date_selected} from {path_mrr}")
    ds_mrr = read_mrr_data(path_mrr, site_selected, date_selected)


    if config.campaign_start_datetime or config.campaign_end_datetime:
        start_time = pd.to_datetime(config.campaign_start_datetime) if config.campaign_start_datetime else None
        end_time = pd.to_datetime(config.campaign_end_datetime) if config.campaign_end_datetime else None
        ds_mrr = ds_mrr.sel(time=slice(start_time, end_time))
        if ds_mrr.sizes.get("time", 0) == 0:
            raise ValueError(
                f"No MRR profiles remain for {date_selected} after applying campaign time limits."
            )

    # Keep the original MRR height field. From 2025-05-26 onward the height
    # range changed at both sites, so height can be time-dependent and must
    # remain available as the per-timestamp height vector in the output.
    if "height" in ds_mrr:
        if "height" in ds_mrr.coords and "time" in ds_mrr["height"].dims:
            ds_mrr = ds_mrr.reset_coords("height")
        ds_mrr["height"].attrs.setdefault(
            "postprocessing_note",
            "Preserved from input so each timestamp retains its own height vector.",
        )

    os.makedirs("plots", exist_ok=True)

    # If interference filtering is enabled, read the MWR rain flag and apply the
    # MRR interference filtering logic to the Ze and W fields.
    if config.remove_interference and filter_RR_on:

        print(f"Applying MRR interference filtering for site {site_selected} on {date_selected}")
        use_mwr_rain_flag = config.use_mwr_rain_flag
        mwr_flag_exists = find_MRR_flag(site_selected, date_selected) if use_mwr_rain_flag else False

        if use_mwr_rain_flag and mwr_flag_exists:
            # read the MWR flags for the selected site and day and store in a dataset
            ds_mwr = read_MWR_flags(site_selected, date_selected)

            # Align categorical MWR flags to the MRR timestamps without numeric
            # interpolation, and report the actual nearest-neighbor time offsets.
            mwr_time_index = pd.DatetimeIndex(ds_mwr.time.values)
            mrr_time_index = pd.DatetimeIndex(ds_mrr.time.values)
            nearest_mwr_index = mwr_time_index.get_indexer(mrr_time_index, method="nearest")
            valid_nearest = nearest_mwr_index >= 0
            nearest_offsets = pd.Series(
                np.abs(mwr_time_index[nearest_mwr_index[valid_nearest]] - mrr_time_index[valid_nearest])
            )
            if not nearest_offsets.empty:
                print(
                    "MWR/MRR nearest-time offsets: "
                    f"median={nearest_offsets.median()}, "
                    f"95%={nearest_offsets.quantile(0.95)}, "
                    f"max={nearest_offsets.max()}"
                )

            mwr_reindex_tolerance = pd.Timedelta(config.mwr_reindex_tolerance)
            rain_extension = pd.Timedelta(config.rain_extension)
            ds_mwr_interp = ds_mwr.reindex(
                time=ds_mrr.time,
                method="nearest",
                tolerance=mwr_reindex_tolerance,
            )

            unmatched_mwr_flags = int(ds_mwr_interp.rain.isnull().sum())
            if unmatched_mwr_flags:
                print(
                    f"MWR rain flag unmatched for {unmatched_mwr_flags} MRR profiles "
                    f"using tolerance {mwr_reindex_tolerance}. Treating them as not rainy."
                )

            rain_flag_raw = ds_mwr_interp.rain.fillna(0).astype(bool)
            mrr_time_step = pd.Series(mrr_time_index).diff().dropna().median()
            if pd.isna(mrr_time_step) or mrr_time_step <= pd.Timedelta(0):
                rain_extension_steps = 0
            else:
                rain_extension_steps = int(np.ceil(rain_extension / mrr_time_step))

            rain_flag = rain_flag_raw.rolling(
                time=2 * rain_extension_steps + 1,
                center=True,
                min_periods=1,
            ).max().astype(bool)

            print(
                "MWR rain-flag expansion on MRR grid: "
                f"raw={int(rain_flag_raw.sum())} profiles, "
                f"expanded={int(rain_flag.sum())} profiles, "
                f"buffer=+/-{rain_extension}."
            )
        elif use_mwr_rain_flag and not config.allow_missing_mwr_rain_flag:
            raise FileNotFoundError(
                f"MWR rain flag is required by the config but is not available for "
                f"{site_selected} on {date_selected}."
            )
        else:
            reason = "disabled in config" if not use_mwr_rain_flag else "not available"
            print(
                f"MWR rain flag {reason}; using an all-false rain flag and running "
                "MRR-structure-based interference filtering."
            )
            rain_flag = xr.DataArray(
                np.zeros(ds_mrr.sizes["time"], dtype=bool),
                coords={"time": ds_mrr.time},
                dims=("time",),
            )

        # plot the Ze time height plot for the day before filtering
        if make_plots:
            plot_time_height_Ze(
                ds_mrr,
                date_selected,
                info_output="pipeline_before_filtering",
                time_stamps=time_stamps,
            )

        # add rain flag to the MRR dataset
        ds_mrr = ds_mrr.assign(rain_flag=rain_flag)

        interference_start = config.interference_start
        interference_end = config.interference_end

        profile_times = pd.to_datetime(ds_mrr.time.values).time
        interference_window = xr.DataArray(
            [interference_start <= profile_time <= interference_end for profile_time in profile_times],
            coords={"time": ds_mrr.time},
            dims=("time",),
        )

        mean_interference_vertical_extent = calculate_mean_interference_vertical_extent(
            ds_mrr["Ze"].where(interference_window).values,
            ds_mrr["height"].values,
            min_lowest_echo_height=interference_min_lowest_echo_height,
            min_connected_gates=min_elevated_connected_gates,
        )
        if np.isfinite(mean_interference_vertical_extent):
            print(
                "Mean interference vertical extent for "
                f"{date_selected}: {mean_interference_vertical_extent:.1f} m "
                f"from profiles with lowest finite Ze above {interference_min_lowest_echo_height:.0f} m."
            )
        else:
            mean_interference_vertical_extent = 1500.0
            print(
                "No elevated-only profiles found for mean interference vertical extent; "
                f"using fallback {mean_interference_vertical_extent:.1f} m."
            )

        min_interference_vertical_extent = (
            min_interference_extent_factor * mean_interference_vertical_extent
        )
        print(
            "Elevated-profile keep criterion: "
            f"continuous Ze extent >= {min_interference_vertical_extent:.1f} m "
            f"and continuity fraction >= {min_interference_layer_fraction:.2f}, "
            f"persisting for at least {min_interference_time_profiles} consecutive profiles, "
            f"with connected Ze segments of at least {min_elevated_connected_gates} gates "
            f"and at most {max_interference_missing_gates} missing gates inside the kept layer."
        )
        print(
            "Lower-rain keep criterion: "
            f"lower echo must persist for at least {min_lower_echo_time_profiles} consecutive profiles "
            f"with at least {min_lower_echo_connected_gates} connected gates below {lower_echo_height_limit:.0f} m "
            f"and peak Ze >= {min_lower_echo_peak_ze:.1f} dBZ."
        )
        print(
            "Rain-column connectivity criterion: "
            f"detached upper fragments are removed when separated from the lowest rain column by more than {max_rain_column_missing_gates} missing gates."
        )
        print(
            "Upper-profile routing criterion: "
            f"process profiles with upper-interference masking when rain flag is on or a continuous Ze segment of at least {min_lower_continuous_ze_gates} gates starts below {lower_echo_height_limit:.0f} m."
        )

        # Apply the MWR rain flag as a prefilter inside the interference
        # window. Profiles outside this window remain unchanged. Protect deep,
        # vertically continuous Ze columns so tall cloud or precipitation
        # columns are not removed only because the surface/radiometer rain flag
        # is false.
        deep_continuous_ze = xr.DataArray(
            [
                mrr_has_deep_continuous_ze(
                    ds_mrr["Ze"].sel(time=time_stamp).values,
                    _height_profile_for_time(ds_mrr, time_stamp),
                    ze_min=-5.0,
                    min_vertical_extent_m=min_interference_vertical_extent,
                    min_layer_fraction=min_interference_layer_fraction,
                    max_missing_gates=max_interference_missing_gates,
                )
                for time_stamp in ds_mrr.time.values
            ],
            coords={"time": ds_mrr.time},
            dims=("time",),
        )
        persistent_deep_continuous_ze = xr.DataArray(
            _mark_true_runs(
                (deep_continuous_ze & interference_window).values,
                min_interference_time_profiles,
            ),
            coords={"time": ds_mrr.time},
            dims=("time",),
        )
        elevated_only_ze = xr.DataArray(
            [
                mrr_is_elevated_only_ze_profile(
                    ds_mrr["Ze"].sel(time=time_stamp).values,
                    _height_profile_for_time(ds_mrr, time_stamp),
                    min_lowest_echo_height=interference_min_lowest_echo_height,
                    ze_min=-5.0,
                    min_connected_gates=min_elevated_connected_gates,
                )
                for time_stamp in ds_mrr.time.values
            ],
            coords={"time": ds_mrr.time},
            dims=("time",),
        )
        lower_echo_mask = xr.DataArray(
            [
                mrr_has_lower_echo(
                    ds_mrr["Ze"].sel(time=time_stamp).values,
                    ds_mrr["W"].sel(time=time_stamp).values,
                    _height_profile_for_time(ds_mrr, time_stamp),
                    search_below_height=lower_echo_height_limit,
                    min_connected_gates=min_lower_echo_connected_gates,
                    ze_min=-10,
                    min_peak_ze=min_lower_echo_peak_ze,
                )
                for time_stamp in ds_mrr.time.values
            ],
            coords={"time": ds_mrr.time},
            dims=("time",),
        )
        lower_continuous_ze_mask = xr.DataArray(
            [
                mrr_has_continuous_ze_starting_below_height(
                    ds_mrr["Ze"].sel(time=time_stamp).values,
                    _height_profile_for_time(ds_mrr, time_stamp),
                    start_below_height=lower_echo_height_limit,
                    ze_min=-10,
                    min_connected_gates=min_lower_continuous_ze_gates,
                    max_missing_gates=max_rain_column_missing_gates,
                )
                for time_stamp in ds_mrr.time.values
            ],
            coords={"time": ds_mrr.time},
            dims=("time",),
        )
        top_rooted_ze = xr.DataArray(
            [
                protect_top_rooted_profiles
                and mrr_has_top_rooted_ze_extent(
                    ds_mrr["Ze"].sel(time=time_stamp).values,
                    _height_profile_for_time(ds_mrr, time_stamp),
                    min_vertical_extent_m=top_rooted_min_vertical_extent,
                    ze_min=-10,
                    max_missing_gates=max_interference_missing_gates,
                )
                for time_stamp in ds_mrr.time.values
            ],
            coords={"time": ds_mrr.time},
            dims=("time",),
        )
        persistent_lower_continuous_ze = xr.DataArray(
            _mark_true_runs(
                (lower_continuous_ze_mask & interference_window).values,
                min_lower_echo_time_profiles,
            ),
            coords={"time": ds_mrr.time},
            dims=("time",),
        )
        persistent_lower_echo = xr.DataArray(
            _mark_true_runs(
                (lower_echo_mask & interference_window).values,
                min_lower_echo_time_profiles,
            ),
            coords={"time": ds_mrr.time},
            dims=("time",),
        )
        top_rooted_profiles = int((top_rooted_ze & interference_window).sum())
        if top_rooted_profiles:
            print(
                f"Protected {top_rooted_profiles} top-rooted profiles because Ze reaches "
                f"the highest range gate and extends downward for at least "
                f"{top_rooted_min_vertical_extent:.0f} m."
            )

        protected_profiles = int((persistent_deep_continuous_ze & ~rain_flag).sum())
        if protected_profiles:
            print(
                f"Protected {protected_profiles} non-rain profiles from MWR prefilter "
                "because Ze has a deep, mostly continuous finite layer that persists in time."
            )
        removed_elevated_profiles = int(
            (elevated_only_ze & ~persistent_deep_continuous_ze & ~top_rooted_ze & interference_window).sum()
        )
        if removed_elevated_profiles:
            print(
                f"Flagged {removed_elevated_profiles} elevated-only profiles inside the interference window "
                "for removal because their Ze extent is too short, not continuous enough, or they do not persist in time."
            )

        for var_name in ds_mrr.data_vars:
            if "time" in ds_mrr[var_name].dims and var_name not in PROFILE_MASK_SKIP_VARS:
                ds_mrr[var_name] = ds_mrr[var_name].where(
                    rain_flag | persistent_deep_continuous_ze | persistent_lower_continuous_ze | top_rooted_ze | ~interference_window
                )

        # loop on time stamps to filter interference:
        for time_stamp in ds_mrr.time.values:

            # read Ze and vd profiles for the selected time stamp and the corresponding rain flag
            ze_profile = ds_mrr["Ze"].sel(time=time_stamp).values
            vd_profile = ds_mrr["W"].sel(time=time_stamp).values
            mwr_rain_flag = bool(ds_mrr["rain_flag"].sel(time=time_stamp).values)
            height_profile = _height_profile_for_time(ds_mrr, time_stamp)

            # Inside the interference time window, first check whether the
            # profile contains a connected lower echo that looks like real rain.
            # If it does, preserve the lower part and only look for
            # interference-like structure aloft.
            if interference_start <= pd.to_datetime(time_stamp).time() <= interference_end:
                profile_persistent_deep_continuous_ze = bool(
                    persistent_deep_continuous_ze.sel(time=time_stamp).values
                )
                profile_elevated_only_ze = bool(
                    elevated_only_ze.sel(time=time_stamp).values
                )
                profile_top_rooted_ze = bool(
                    top_rooted_ze.sel(time=time_stamp).values
                )

                # Remove elevated-only profiles unless their Ze layer is deep
                # enough and sufficiently continuous compared with the typical
                # daily interference extent.
                if (
                    profile_elevated_only_ze
                    and not profile_persistent_deep_continuous_ze
                    and not profile_top_rooted_ze
                ):
                    apply_range_gate_mask_to_profile(
                        ds_mrr,
                        time_stamp,
                        np.ones_like(height_profile, dtype=bool),
                    )
                    continue

                if remove_short_detached_ze_columns:
                    short_detached_mask = mask_short_detached_ze_columns(
                        ze_profile,
                        height_profile,
                        lower_root_max_height=lower_echo_height_limit,
                        min_lower_connected_gates=min_lower_continuous_ze_gates,
                        max_missing_gates=max_rain_column_missing_gates,
                        max_detached_vertical_extent_m=max_detached_ze_column_vertical_extent,
                        min_detached_base_height_m=min_detached_ze_column_base_height,
                        ze_min=-10,
                    )
                    if short_detached_mask.any():
                        apply_range_gate_mask_to_profile(ds_mrr, time_stamp, short_detached_mask)
                        ze_profile = ds_mrr["Ze"].sel(time=time_stamp).values
                        vd_profile = ds_mrr["W"].sel(time=time_stamp).values

                if not apply_upper_interference_masking:
                    continue

                lower_echo = bool(lower_echo_mask.sel(time=time_stamp).values)
                profile_persistent_lower_echo = bool(
                    persistent_lower_echo.sel(time=time_stamp).values
                )
                profile_persistent_lower_continuous_ze = bool(
                    persistent_lower_continuous_ze.sel(time=time_stamp).values
                )
                process_upper_profile = bool(
                    mwr_rain_flag or profile_persistent_lower_continuous_ze
                )
                
                # If the MWR rain flag is true and a lower echo exists, treat
                # the profile as rain-plus-possible-interference and only mask
                # suspicious gates above the protected lower region.
                if process_upper_profile and (lower_echo or profile_persistent_lower_echo):
                    result = mask_upper_interference(
                        ze=ze_profile,
                        vd=vd_profile,
                        height=height_profile,
                        combine="or",
                        mask_all_above=False,
                        vd_tolerance=0.075,
                        plateau_min_gates=5,
                        ze_min_step=2.0,
                        interference_above_height=1500.0,
                        evidence_window=5,
                        min_evidence_gates=3,
                    )

                    if keep_lowest_connected_component:
                        ze_filtered, vd_filtered = keep_lowest_connected_ze_component(
                            result.ze_filtered,
                            result.vd_filtered,
                            ze_min=-10,
                            max_missing_gates=max_rain_column_missing_gates,
                        )
                    else:
                        ze_filtered = result.ze_filtered
                        vd_filtered = result.vd_filtered
                    
                    removed_gate_mask = (
                        (np.isfinite(ze_profile) & ~np.isfinite(ze_filtered))
                        | (np.isfinite(vd_profile) & ~np.isfinite(vd_filtered))
                    )
                    apply_range_gate_mask_to_profile(ds_mrr, time_stamp, removed_gate_mask)
                
                
                elif process_upper_profile:
                    result = mask_upper_interference(
                        ze=ze_profile,
                        vd=vd_profile,
                        height=height_profile,
                        combine="or",
                        mask_all_above=False,
                        vd_tolerance=0.075,
                        plateau_min_gates=5,
                        ze_min_step=2.0,
                        interference_above_height=2500.0,
                        evidence_window=5,
                        min_evidence_gates=3,
                    )

                    if keep_lowest_connected_component:
                        ze_filtered, vd_filtered = keep_lowest_connected_ze_component(
                            result.ze_filtered,
                            result.vd_filtered,
                            ze_min=-10,
                            max_missing_gates=max_rain_column_missing_gates,
                        )
                    else:
                        ze_filtered = result.ze_filtered
                        vd_filtered = result.vd_filtered

                    removed_gate_mask = (
                        (np.isfinite(ze_profile) & ~np.isfinite(ze_filtered))
                        | (np.isfinite(vd_profile) & ~np.isfinite(vd_filtered))
                    )
                    apply_range_gate_mask_to_profile(ds_mrr, time_stamp, removed_gate_mask)

                elif mwr_rain_flag and (not lower_echo or not profile_persistent_lower_echo):
                    if not profile_persistent_deep_continuous_ze:
                        apply_range_gate_mask_to_profile(
                            ds_mrr,
                            time_stamp,
                            np.ones_like(height_profile, dtype=bool),
                        )
                        continue

                    print(
                        f"Time {time_stamp} is rainy but has no connected lower echo. "
                        "Checking the full profile for interference-like mid/upper-level structure."
                    )
                    detection = check_profile(
                        ze_profile, 
                        vd_profile, 
                        vd_tolerance=0.075, 
                        plateau_min_gates=3, 
                        ze_min_step=2.0, 
                        zigzag_min_turns=2, 
                        combine='or')
                    
                    plateau_mask = detection.velocity_plateau
                    zigzag_mask = detection.ze_zigzag
                    interference_mask = detection.combined

                    Ze_filtered = np.asarray(ze_profile, dtype=float).copy()
                    Ze_filtered[interference_mask] = np.nan
                    vd_filtered = np.asarray(vd_profile, dtype=float).copy()
                    vd_filtered[interference_mask] = np.nan

                    apply_range_gate_mask_to_profile(ds_mrr, time_stamp, interference_mask)
                else:
                    continue  # No interference, no filtering to apply

                    
        ds_mrr = ds_mrr.assign(
            interference_window=interference_window,
            deep_continuous_ze=deep_continuous_ze,
            persistent_deep_continuous_ze=persistent_deep_continuous_ze,
            elevated_only_ze=elevated_only_ze,
            lower_echo_mask=lower_echo_mask,
            lower_continuous_ze_mask=lower_continuous_ze_mask,
            persistent_lower_continuous_ze=persistent_lower_continuous_ze,
            persistent_lower_echo=persistent_lower_echo,
            top_rooted_ze=top_rooted_ze,
        )

        # plot the Ze time height plot for the day after filtering
        if make_plots:
            plot_time_height_Ze(
                ds_mrr,
                date_selected,
                info_output="pipeline_after_filtering",
                time_stamps=time_stamps,
            )

    if config.calculate_uncertainty:
        print(f"Calculating MRR uncertainty for site {site_selected} on {date_selected}")
        ds_unc = mrr_moment_uncertainty(
            ds_mrr["eta"],
            noise_std=ds_mrr["etaNoiseStd"] if "etaNoiseStd" in ds_mrr else None,
            velocity_dim="velocity",
            n_realizations=300,
            calculate_rain_rate=True,
            vertical_air_velocity=0.0,
            vertical_air_velocity_uncertainty=0.2,
            rain_rate_relative_model_uncertainty=0.10,
            calibration_uncertainty_db=1.0,
            random_seed=42,
        )
        ds_unc = ds_unc[
            [
                "ze_random_uncertainty",
                "ze_p16",
                "ze_p84",
                "ze_db_random_uncertainty",
                "ze_db_total_uncertainty",
                "ze_db_p16",
                "ze_db_p84",
                "rain_rate_random_uncertainty",
                "rain_rate_calibration_uncertainty",
                "rain_rate_total_uncertainty",
                "rain_rate_p16",
                "rain_rate_p50",
                "rain_rate_p84",
                "rain_rate_relative_uncertainty",
                "rain_rate_total_relative_uncertainty",
                "mean_velocity_random_uncertainty",
                "spectral_width_random_uncertainty",
                "skewness_random_uncertainty",
                "kurtosis_random_uncertainty",
            ]
        ]
        ds_mrr = xr.merge([ds_mrr, ds_unc], compat="override")
        ds_mrr.attrs.update(ds_unc.attrs)
        ds_mrr.attrs["postprocessing_uncertainty_status"] = "calculated"
        ds_mrr.attrs["postprocessing_uncertainty_variables"] = ",".join(ds_unc.data_vars)

    if save_output is None:
        save_output = config.save_filtered_dataset

    output_file = None
    output_files: list[Path] = []
    ds_return = ds_mrr
    if save_output:
        for time_res in config.time_resolutions:
            if time_res == "1min":
                ds_to_save = ds_mrr
                final_time_average = None
            elif time_res in {"5min", "5_min"}:
                final_time_average = "5min"
                print(f"Averaging processed MRR dataset over {final_time_average} before saving")
                ds_to_save = average_mrr_dataset_over_time(ds_mrr, final_time_average)
                if make_plots:
                    plot_time_height_Ze(
                        ds_to_save,
                        date_selected,
                        info_output="pipeline_after_averaging",
                        time_stamps=time_stamps,
                    )
            else:
                raise ValueError(f"Unsupported output time resolution: {time_res}")

            output_config = replace(
                config,
                time_res=time_res,
                final_time_average=final_time_average,
            )
            print(
                f"Saving {time_res} filtered MRR dataset for site "
                f"{site_selected} on {date_selected}"
            )
            saved_file = save_filtered_mrr_dataset(
                ds_to_save,
                config=output_config,
                date_selected=date_selected,
            )
            output_files.append(saved_file)
            output_file = saved_file
            ds_return = ds_to_save

    print(f"----------------- processing {date_selected} for site {site_selected} completed -----------------")
    return ProcessedMRRDay(dataset=ds_return, output_file=output_file, output_files=output_files)


def get_configured_dates(config) -> list[str]:
    """Return YYYYMMDD dates requested by the config."""
    if config.campaign_dates:
        return list(config.campaign_dates)
    if config.campaign_start_date and config.campaign_end_date:
        date_index = pd.date_range(
            pd.to_datetime(config.campaign_start_date),
            pd.to_datetime(config.campaign_end_date),
            freq="D",
        )
        return [date.strftime("%Y%m%d") for date in date_index]
    return [config.time_stamps[0][:8]]


def run_pipeline(config, *, make_plots: bool = True) -> list[ProcessedMRRDay]:
    """Run the configured daily or campaign MRR pipeline."""
    results = []
    for date_selected in get_configured_dates(config):
        print(f"Processing MRR day {date_selected} for site {config.site_selected}")
        results.append(
            process_mrr_day(
                config,
                date_selected=date_selected,
                make_plots=make_plots,
            )
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MRR postprocessing pipeline.")
    parser.add_argument(
        "--config",
        default=Path(__file__).with_name("remove_interfence_mrr_config.yaml"),
        type=Path,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip before/after diagnostic plots.",
    )
    args = parser.parse_args()
    config = load_mrr_interference_config(args.config)
    run_pipeline(config, make_plots=config.make_plots and not args.no_plots)


if __name__ == "__main__":
    main()
