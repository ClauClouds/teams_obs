"""Campaign-level MRR postprocessing pipeline.

This module orchestrates one-day and campaign-style processing using the
interference filtering helpers from ``remove_interfence_mrr``.  The scientific
filtering logic is intentionally unchanged from the original script; this file
only makes it reusable from code and future campaign loops.


how to run:
source .teams_venv/bin/activate
python process/mrr_pipeline.py --config process/remove_interfence_mrr_config.yaml


pid: 2979193

"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
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
        mask_upper_interference,
        mrr_has_continuous_ze_starting_below_height,
        mrr_has_deep_continuous_ze,
        mrr_has_lower_echo,
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
        mask_upper_interference,
        mrr_has_continuous_ze_starting_below_height,
        mrr_has_deep_continuous_ze,
        mrr_has_lower_echo,
        mrr_is_elevated_only_ze_profile,
    )
    from mrr_io import find_MRR_flag, read_mrr_data, save_filtered_mrr_dataset
    from mrr_plots import plot_time_height_Ze



@dataclass(frozen=True)
class ProcessedMRRDay:
    """Result returned by the daily MRR postprocessing pipeline."""

    dataset: xr.Dataset
    output_file: Path | None


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

    # Convert the MRR height field from time-range to a fixed vertical coordinate.
    # This keeps Ze as a clean time x range matrix and prevents height from
    # being treated as a profile variable during rain masking.
    if "height" in ds_mrr and ds_mrr["height"].dims == ("time", "range"):
        height_1d = ds_mrr["height"].median(dim="time", skipna=True)
        ds_mrr = ds_mrr.drop_vars("height").assign_coords(height=("range", height_1d.values))

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
            plot_time_height_Ze(ds_mrr, date_selected, info_output="before_filtering", time_stamps=time_stamps)

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
        height_for_extent = ds_mrr["height"].values
        deep_continuous_ze = xr.DataArray(
            [
                mrr_has_deep_continuous_ze(
                    ds_mrr["Ze"].sel(time=time_stamp).values,
                    height_for_extent,
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
                    height_for_extent,
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
                    height_for_extent,
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
                    height_for_extent,
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
        protected_profiles = int((persistent_deep_continuous_ze & ~rain_flag).sum())
        if protected_profiles:
            print(
                f"Protected {protected_profiles} non-rain profiles from MWR prefilter "
                "because Ze has a deep, mostly continuous finite layer that persists in time."
            )
        removed_elevated_profiles = int(
            (elevated_only_ze & ~persistent_deep_continuous_ze & interference_window).sum()
        )
        if removed_elevated_profiles:
            print(
                f"Flagged {removed_elevated_profiles} elevated-only profiles inside the interference window "
                "for removal because their Ze extent is too short, not continuous enough, or they do not persist in time."
            )

        for var_name in ds_mrr.data_vars:
            if "time" in ds_mrr[var_name].dims and var_name not in PROFILE_MASK_SKIP_VARS:
                ds_mrr[var_name] = ds_mrr[var_name].where(
                    rain_flag | persistent_deep_continuous_ze | persistent_lower_continuous_ze | ~interference_window
                )

        # loop on time stamps to filter interference:
        for time_stamp in ds_mrr.time.values:

            # read Ze and vd profiles for the selected time stamp and the corresponding rain flag
            ze_profile = ds_mrr["Ze"].sel(time=time_stamp).values
            vd_profile = ds_mrr["W"].sel(time=time_stamp).values
            mwr_rain_flag = bool(ds_mrr["rain_flag"].sel(time=time_stamp).values)
            if "time" in ds_mrr["height"].dims:
                height_profile = ds_mrr["height"].sel(time=time_stamp).values
            else:
                height_profile = ds_mrr["height"].values

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

                # Remove elevated-only profiles unless their Ze layer is deep
                # enough and sufficiently continuous compared with the typical
                # daily interference extent.
                if profile_elevated_only_ze and not profile_persistent_deep_continuous_ze:
                    apply_range_gate_mask_to_profile(
                        ds_mrr,
                        time_stamp,
                        np.ones_like(height_profile, dtype=bool),
                    )
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

                    ze_connected, vd_connected = keep_lowest_connected_ze_component(
                        result.ze_filtered,
                        result.vd_filtered,
                        ze_min=-10,
                        max_missing_gates=max_rain_column_missing_gates,
                    )

                    ze_filtered = ze_connected
                    vd_filtered = vd_connected
                    
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

                    ze_filtered, vd_filtered = keep_lowest_connected_ze_component(
                        result.ze_filtered,
                        result.vd_filtered,
                        ze_min=-10,
                        max_missing_gates=max_rain_column_missing_gates,
                    )

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
        )

        # plot the Ze time height plot for the day after filtering
        if make_plots:
            plot_time_height_Ze(ds_mrr, date_selected, info_output="after_filtering", time_stamps=time_stamps)

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
    if save_output:
        print(f"Saving filtered MRR dataset for site {site_selected} on {date_selected}")
        output_file = save_filtered_mrr_dataset(ds_mrr, config=config, date_selected=date_selected)

    print(f"----------------- processing {date_selected} for site {site_selected} completed -----------------")
    return ProcessedMRRDay(dataset=ds_mrr, output_file=output_file)


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
