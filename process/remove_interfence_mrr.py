"""MRR interference filtering driven by rain context and Ze/W profile structure.

This module implements a multi-stage filtering workflow for MRR profiles during
the known cable-car interference period at Lagonero. The code reads one MRR
day, aligns the radiometer rain flag to the MRR time grid, classifies each
profile according to the vertical structure of Ze and W, and then masks only
the parts of the column that are most likely contaminated by interference.

Overview of the filtering sequence
----------------------------------
1. Read one day of MRR data and preserve the original height field, including
    time-dependent height vectors on the time-range grid.

2. Read MWR rain flags, align them to the MRR timestamps with nearest-neighbor
    matching, and expand them in time by a configurable buffer. This produces a
    rain context on the MRR grid even when the two instruments do not sample at
    exactly the same timestamps.

3. Restrict the filtering to the known interference window. Profiles outside
    that time interval are left unchanged.

4. Estimate a typical daily interference depth from elevated-only Ze profiles.
    A profile contributes to this daily statistic when its first connected Ze
    segment starts above a configurable height threshold. The mean vertical
    extent of those elevated profiles is then used as a reference depth for the
    later elevated-profile screening.

5. Identify elevated-only Ze profiles and keep only the ones that are deep,
    vertically continuous, and persistent in time. Elevated-only profiles that
    are too shallow, fragmented, or isolated in time are removed entirely.

6. Identify profiles that contain a plausible lower rain column below a chosen
    height threshold. This lower-column test is height-based rather than gate-
    based: it requires connected Ze/W gates below the threshold and can also
    require a minimum peak Ze there. A second routing rule allows a profile to
    enter the upper-profile interference treatment when it either has the MWR
    rain flag or contains a continuous Ze segment starting below that same
    lower-column height.

7. For profiles routed into upper-profile processing, detect interference-like
    structure aloft using two signatures:
    - mean Doppler velocity (W/VD) that is nearly constant over several gates,
    - radar reflectivity (Ze) with repeated gate-to-gate zigzag structure.

8. Apply upper-profile masking only above a configurable height. After the
    shape-based interference mask is computed, keep the lowest connected rain
    column and remove detached upper fragments that are no longer connected to
    the lower echo.

9. Write the filtered Ze and W fields back into the dataset by replacing the
    rejected gates with NaN. The script also produces before/after time-height
    plots to inspect the result.

Interpretation of the main profile classes
------------------------------------------
- Elevated-only interference candidates:
  profiles whose first connected Ze segment starts above the lower-column
  threshold. These are compared against the daily mean interference extent and
  removed unless they are sufficiently deep, continuous, and temporally
  persistent.

- Lower-column profiles:
  profiles with connected Ze/W structure below the lower-column threshold.
  These are treated as plausible rain columns and are routed to the upper-
  profile interference filter so that only the contaminated upper part is
  removed.

- Detached upper fragments:
  finite Ze/W structures above the lower rain column that are separated from
  the lowest connected component. These are treated as interference and are
  removed after upper-profile processing.

The helper functions in this file return Boolean masks and diagnostics, but the
main execution block applies the selected filters directly by writing NaNs back
into the dataset variables.

Note: filtering currently valid for data from 20250626 onwards. Before different ranges setup, need to change threhsolds

how to run:
source .teams_venv/bin/activate
.teams_venv/bin/python -m py_compile process/remove_interfence_mrr.py
python process/remove_interfence_mrr.py --config process/remove_interfence_mrr_config.yaml
pid collalbo run : 160968

"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import ast
import numpy as np
from datetime import time
import site
from turtle import mode

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from scipy.ndimage import label
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.projections.polar import PolarAxes
from readers.data_info import PLOT_SITES_NAMES, MWR_SITES_NAMES, site_lats, site_lons
from readers.MWR import read_iwv_elev
from figures.plot_settings import VAR_DICT
import numpy as np
import pandas as pd
import xarray as xr
from figures.utils import find_closest_dc_value, read_file_list_for_mode, calculate_mean_anomaly_for_time_selection, plot_mean_azimuth_ring, get_shared_colorbar_limits, get_regular_integer_colorbar_spec
from readers.data_info import orography_path, iop_conv_days, iop_MoBL_T_days, hours_diurnal_cycle_calc, azimuth_bins
import os
import pdb
from readers.MWR import read_MWR_flags





@dataclass(frozen=True)
class ProfileInterferenceResult:
    """Masks and diagnostics returned by :func:`check_profile`."""

    velocity_plateau: np.ndarray
    ze_zigzag: np.ndarray
    combined: np.ndarray
    vd_step: np.ndarray
    ze_step: np.ndarray
    ze_turn: np.ndarray


@dataclass(frozen=True)
class UpperInterferenceResult:
    """Output from :func:`mask_upper_interference`."""

    ze_filtered: np.ndarray
    vd_filtered: np.ndarray
    mask: np.ndarray
    evidence: np.ndarray
    cutoff_index: int | None
    cutoff_height: float | None


@dataclass(frozen=True)
class MRRInterferenceConfig:
    """Runtime settings for the MRR interference filtering script."""

    sites: list[str]
    site_selected: str
    path_mrr: str
    time_stamps: list[str]
    remove_interference: bool
    filter_RR_on: bool
    make_plots: bool
    min_interference_extent_factor: float
    min_interference_layer_fraction: float
    interference_min_lowest_echo_height: float
    min_interference_time_profiles: int
    min_lower_echo_time_profiles: int
    min_elevated_connected_gates: int
    min_lower_echo_connected_gates: int
    max_interference_missing_gates: int
    max_rain_column_missing_gates: int
    protect_top_rooted_profiles: bool
    top_rooted_min_vertical_extent: float
    keep_lowest_connected_component: bool
    apply_upper_interference_masking: bool
    remove_short_detached_ze_columns: bool
    max_detached_ze_column_vertical_extent: float
    min_detached_ze_column_base_height: float
    lower_echo_height_limit: float
    min_lower_echo_peak_ze: float
    min_lower_continuous_ze_gates: int
    use_mwr_rain_flag: bool
    allow_missing_mwr_rain_flag: bool
    mwr_reindex_tolerance: str
    rain_extension: str
    interference_start: time
    interference_end: time
    save_filtered_dataset: bool
    output_dir: str
    output_file_template: str
    output_compression_level: int
    output_overwrite: bool
    time_res: str
    time_resolutions: list[str]
    final_time_average: str | None
    calculate_uncertainty: bool
    campaign_start_date: str | None
    campaign_end_date: str | None
    campaign_start_datetime: str | None
    campaign_end_datetime: str | None
    campaign_dates: list[str]
    config_path: str


def _strip_yaml_comment(line: str) -> str:
    in_single_quote = False
    in_double_quote = False
    for index, char in enumerate(line):
        if char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
        elif char == '\"' and not in_single_quote:
            in_double_quote = not in_double_quote
        elif char == "#" and not in_single_quote and not in_double_quote:
            return line[:index]
    return line


def _parse_yaml_scalar(value: str):
    value = value.strip()
    lower_value = value.lower()
    if lower_value == "true":
        return True
    if lower_value == "false":
        return False
    if lower_value in {"null", "none", "~"}:
        return None
    if value.startswith(("'", '\"', "[")):
        return ast.literal_eval(value)
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _read_simple_yaml(config_path: Path) -> dict:
    config: dict = {}
    stack: list[tuple[int, dict]] = [(-1, config)]

    for line_number, raw_line in enumerate(config_path.read_text().splitlines(), start=1):
        line = _strip_yaml_comment(raw_line).rstrip()
        if not line.strip():
            continue

        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        key, separator, value = stripped.partition(":")
        if not separator:
            raise ValueError(f"Invalid YAML line {line_number} in {config_path}: {raw_line}")

        while indent <= stack[-1][0]:
            stack.pop()

        current_mapping = stack[-1][1]
        key = key.strip()
        value = value.strip()
        if value:
            current_mapping[key] = _parse_yaml_scalar(value)
        else:
            nested_mapping: dict = {}
            current_mapping[key] = nested_mapping
            stack.append((indent, nested_mapping))

    return config


def load_mrr_interference_config(config_path: str | Path) -> MRRInterferenceConfig:
    """Read the YAML config used by this script."""
    config_path = Path(config_path)
    raw_config = _read_simple_yaml(config_path)

    campaign_config = raw_config.get("campaign", {})
    output_config = raw_config.get("output", {})
    processing_config = raw_config.get("processing", {})

    sites_config = raw_config.get("sites")
    site_selected = campaign_config.get("site")
    if site_selected is None:
        dataset_config = raw_config.get("datasets", raw_config)
        site_selected = dataset_config["site_selected"]
    else:
        dataset_config = raw_config.get("datasets", {})

    if isinstance(sites_config, dict) and site_selected in sites_config:
        site_config = sites_config[site_selected]
        sites = list(sites_config)
        path_mrr = site_config.get("path_mrr")
        if path_mrr is None:
            path_mrr = site_config["path_mrr_template"].format(
                site=site_selected,
                site_selected=site_selected,
            )
        filtering_config = site_config.get("filtering", {})
        rain_flag_config = site_config.get("rain_flag_config", {})
        window_config = site_config.get("interference_window") or site_config.get("interference_windows")
    else:
        dataset_config = raw_config.get("datasets", raw_config)
        sites = list(dataset_config["sites"])
        path_mrr = dataset_config.get("path_mrr")
        if path_mrr is None:
            path_mrr = dataset_config["path_mrr_template"].format(site_selected=site_selected)
        filtering_config = raw_config.get(f"filtering_setup_{site_selected}")
        if filtering_config is None:
            filtering_config = raw_config.get("filtering_setup", raw_config)
        rain_flag_config = raw_config.get("rain_flag_config", raw_config)
        window_config = raw_config.get("interference_windows", {}).get(site_selected)

    if not window_config:
        raise ValueError(
            f"No interference window configured for site '{site_selected}' in {config_path}"
        )

    time_stamps = list(raw_config.get("time_stamps", []))

    raw_time_res = processing_config.get("time_res")
    if raw_time_res is None:
        final_time_average_config = processing_config.get("final_time_average", "5min")
        raw_time_res = "1min" if final_time_average_config is None else final_time_average_config

    requested_time_res = raw_time_res if isinstance(raw_time_res, list) else [raw_time_res]
    time_resolutions = []
    for requested in requested_time_res:
        normalized_time_res = str(requested).lower().replace(" ", "")
        if normalized_time_res in {"1min", "1_min", "1t"}:
            time_resolutions.append("1min")
        elif normalized_time_res in {"5min", "5_min", "5t"}:
            time_resolutions.append("5min")
        else:
            raise ValueError(
                f"Unsupported processing.time_res '{requested}'. Use '1min' or '5min'."
            )
    time_resolutions = list(dict.fromkeys(time_resolutions))
    time_res = time_resolutions[0]
    final_time_average = None if time_res == "1min" else "5min"

    return MRRInterferenceConfig(
        sites=sites,
        site_selected=site_selected,
        path_mrr=path_mrr,
        time_stamps=time_stamps,
        remove_interference=bool(processing_config.get("remove_interference", True)),
        filter_RR_on=bool(filtering_config["filter_RR_on"]),
        make_plots=bool(processing_config.get("make_plots", True)),
        min_interference_extent_factor=float(filtering_config["min_interference_extent_factor"]),
        min_interference_layer_fraction=float(filtering_config["min_interference_layer_fraction"]),
        interference_min_lowest_echo_height=float(filtering_config["interference_min_lowest_echo_height"]),
        min_interference_time_profiles=int(filtering_config["min_interference_time_profiles"]),
        min_lower_echo_time_profiles=int(filtering_config["min_lower_echo_time_profiles"]),
        min_elevated_connected_gates=int(filtering_config["min_elevated_connected_gates"]),
        min_lower_echo_connected_gates=int(filtering_config["min_lower_echo_connected_gates"]),
        max_interference_missing_gates=int(filtering_config["max_interference_missing_gates"]),
        max_rain_column_missing_gates=int(filtering_config["max_rain_column_missing_gates"]),
        protect_top_rooted_profiles=bool(filtering_config.get("protect_top_rooted_profiles", False)),
        top_rooted_min_vertical_extent=float(filtering_config.get("top_rooted_min_vertical_extent", 1000.0)),
        keep_lowest_connected_component=bool(filtering_config.get("keep_lowest_connected_component", True)),
        apply_upper_interference_masking=bool(filtering_config.get("apply_upper_interference_masking", True)),
        remove_short_detached_ze_columns=bool(filtering_config.get("remove_short_detached_ze_columns", False)),
        max_detached_ze_column_vertical_extent=float(filtering_config.get("max_detached_ze_column_vertical_extent", 800.0)),
        min_detached_ze_column_base_height=float(filtering_config.get("min_detached_ze_column_base_height", 1500.0)),
        lower_echo_height_limit=float(filtering_config["lower_echo_height_limit"]),
        min_lower_echo_peak_ze=float(filtering_config["min_lower_echo_peak_ze"]),
        min_lower_continuous_ze_gates=int(filtering_config["min_lower_continuous_ze_gates"]),
        use_mwr_rain_flag=bool(rain_flag_config.get("use_mwr_rain_flag", True)),
        allow_missing_mwr_rain_flag=bool(rain_flag_config.get("allow_missing_mwr_rain_flag", True)),
        mwr_reindex_tolerance=rain_flag_config["mwr_reindex_tolerance"],
        rain_extension=rain_flag_config["rain_extension"],
        interference_start=time.fromisoformat(window_config["start"]),
        interference_end=time.fromisoformat(window_config["end"]),
        save_filtered_dataset=bool(output_config.get("save_filtered_dataset", True)),
        output_dir=output_config.get("output_dir", "data/mrr_filtered"),
        output_file_template=output_config.get(
            "output_filename",
            output_config.get(
                "output_file_template",
                "{date_selected}_{site_selected}_mrr_interference_filtered.nc",
            ),
        ),
        output_compression_level=int(output_config.get("compression_level", 9)),
        output_overwrite=bool(output_config.get("overwrite", True)),
        time_res=time_res,
        time_resolutions=time_resolutions,
        final_time_average=final_time_average,
        calculate_uncertainty=bool(processing_config.get("calculate_uncertainty", False)),
        campaign_start_date=campaign_config.get("start_date"),
        campaign_end_date=campaign_config.get("end_date"),
        campaign_start_datetime=campaign_config.get("start_datetime"),
        campaign_end_datetime=campaign_config.get("end_datetime"),
        campaign_dates=list(campaign_config.get("dates", [])),
        config_path=str(config_path),
    )


PROFILE_MASK_SKIP_VARS = {
    "height",
    "range",
    "lat",
    "lon",
    "latitude",
    "longitude",
    "altitude",
    "time",
    "rain_flag",
}


def apply_range_gate_mask_to_profile(
    ds: xr.Dataset,
    time_stamp,
    gate_mask: np.ndarray,
    *,
    skip_vars: set[str] = PROFILE_MASK_SKIP_VARS,
) -> None:
    """Mask selected range gates for all time/range data variables in-place."""
    gate_mask = np.asarray(gate_mask, dtype=bool)

    for var_name, data_array in ds.data_vars.items():
        if var_name in skip_vars:
            continue
        if "time" not in data_array.dims or "range" not in data_array.dims:
            continue

        profile = data_array.sel(time=time_stamp)
        if gate_mask.size != profile.sizes["range"]:
            raise ValueError(
                f"Gate mask has {gate_mask.size} gates, but {var_name} has "
                f"{profile.sizes['range']} range gates."
            )

        keep_gates = xr.DataArray(
            ~gate_mask,
            coords={"range": profile["range"]},
            dims=("range",),
        )
        ds[var_name].loc[dict(time=time_stamp)] = profile.where(keep_gates)


def add_postprocessing_metadata(
    ds: xr.Dataset,
    *,
    config: MRRInterferenceConfig,
    date_selected: str,
) -> xr.Dataset:
    """Add postprocessing metadata to the filtered output dataset."""
    ds = ds.copy()
    timestamp_utc = pd.Timestamp.now(tz="UTC").isoformat()
    history_entry = (
        f"{timestamp_utc}: MRR interference postprocessing applied with "
        f"process/remove_interfence_mrr.py using config {config.config_path}."
    )
    previous_history = ds.attrs.get("history")
    ds.attrs["history"] = (
        f"{previous_history}\n{history_entry}" if previous_history else history_entry
    )
    ds.attrs.update(
        {
            "postprocessing_name": "MRR interference filtering",
            "postprocessing_description": (
                "Interference-like MRR range gates were identified from Ze/W profile "
                "structure and masked across all data variables with time and range dimensions."
            ),
            "postprocessing_script": "https://github.com/ClauClouds/teams_obs/blob/main/process/remove_interfence_mrr.py",
            "postprocessing_config": config.config_path,
            "postprocessing_site": config.site_selected,
            "postprocessing_date": date_selected,
            "postprocessing_time_stamps": ",".join(config.time_stamps),
            "postprocessing_filter_RR_on": str(config.filter_RR_on),
            "postprocessing_use_mwr_rain_flag": str(config.use_mwr_rain_flag),
            "postprocessing_allow_missing_mwr_rain_flag": str(config.allow_missing_mwr_rain_flag),
            "postprocessing_interference_window": (
                f"{config.interference_start.isoformat()}-{config.interference_end.isoformat()}"
            ),
            "postprocessing_min_interference_extent_factor": config.min_interference_extent_factor,
            "postprocessing_min_interference_layer_fraction": config.min_interference_layer_fraction,
            "postprocessing_interference_min_lowest_echo_height_m": (
                config.interference_min_lowest_echo_height
            ),
            "postprocessing_min_interference_time_profiles": config.min_interference_time_profiles,
            "postprocessing_min_lower_echo_time_profiles": config.min_lower_echo_time_profiles,
            "postprocessing_min_elevated_connected_gates": config.min_elevated_connected_gates,
            "postprocessing_min_lower_echo_connected_gates": config.min_lower_echo_connected_gates,
            "postprocessing_max_interference_missing_gates": config.max_interference_missing_gates,
            "postprocessing_max_rain_column_missing_gates": config.max_rain_column_missing_gates,
            "postprocessing_protect_top_rooted_profiles": str(config.protect_top_rooted_profiles),
            "postprocessing_top_rooted_min_vertical_extent_m": config.top_rooted_min_vertical_extent,
            "postprocessing_keep_lowest_connected_component": str(config.keep_lowest_connected_component),
            "postprocessing_apply_upper_interference_masking": str(config.apply_upper_interference_masking),
            "postprocessing_remove_short_detached_ze_columns": str(config.remove_short_detached_ze_columns),
            "postprocessing_max_detached_ze_column_vertical_extent_m": config.max_detached_ze_column_vertical_extent,
            "postprocessing_min_detached_ze_column_base_height_m": config.min_detached_ze_column_base_height,
            "postprocessing_lower_echo_height_limit_m": config.lower_echo_height_limit,
            "postprocessing_min_lower_echo_peak_ze_dbz": config.min_lower_echo_peak_ze,
            "postprocessing_min_lower_continuous_ze_gates": config.min_lower_continuous_ze_gates,
            "postprocessing_mwr_reindex_tolerance": config.mwr_reindex_tolerance,
            "postprocessing_rain_extension": config.rain_extension,
            "postprocessing_masked_variable_rule": (
                "All data variables with both time and range dimensions were masked; "
                f"skipped variables: {','.join(sorted(PROFILE_MASK_SKIP_VARS))}."
            ),
            "postprocessing_time_res": config.time_res,
            "postprocessing_requested_time_resolutions": ",".join(config.time_resolutions),
            "postprocessing_final_time_average": str(config.final_time_average),
            "postprocessing_output_compression": (
                f"NETCDF4 zlib complevel={config.output_compression_level} shuffle=True"
            ),
        }
    )
    return ds


def _compressed_netcdf_encoding(ds: xr.Dataset, compression_level: int) -> dict:
    compression_level = int(np.clip(compression_level, 0, 9))
    encoding = {}
    for var_name, variable in ds.variables.items():
        if not variable.dims:
            continue
        if variable.dtype.kind in {"O", "U", "S"}:
            continue
        encoding[var_name] = {
            "zlib": True,
            "complevel": compression_level,
            "shuffle": True,
        }
    return encoding


def save_filtered_mrr_dataset(
    ds: xr.Dataset,
    *,
    config: MRRInterferenceConfig,
    date_selected: str,
) -> Path:
    time_resolution = config.time_res
    output_dir = Path(
        config.output_dir.format(
            site=config.site_selected,
            site_selected=config.site_selected,
            date_selected=date_selected,
            datetime=date_selected,
            time_resolution=time_resolution,
            time_res=config.time_res,
            final_time_average=time_resolution,
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / config.output_file_template.format(
        site=config.site_selected,
        site_selected=config.site_selected,
        date_selected=date_selected,
        datetime=date_selected,
        time_resolution=time_resolution,
        time_res=config.time_res,
        final_time_average=time_resolution,
    )

    ds = add_postprocessing_metadata(ds, config=config, date_selected=date_selected)
    if output_file.exists() and not config.output_overwrite:
        print(f"Output file exists and overwrite is disabled; keeping {output_file}")
        return output_file

    encoding = _compressed_netcdf_encoding(ds, config.output_compression_level)
    ds.to_netcdf(
        output_file,
        engine="netcdf4",
        format="NETCDF4",
        encoding=encoding,
    )
    print(f"Saved filtered MRR dataset to {output_file}")
    return output_file


def _mark_true_runs(condition: np.ndarray, min_length: int) -> np.ndarray:
    """Keep complete runs of True values having at least ``min_length`` items."""
    condition = np.asarray(condition, dtype=bool)
    result = np.zeros(condition.size, dtype=bool)

    start = None
    for index, value in enumerate(np.r_[condition, False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if index - start >= min_length:
                result[start:index] = True
            start = None

    return result


def mrr_has_lower_echo(
    ze: np.ndarray,
    vd: np.ndarray,
    height: np.ndarray,
    *,
    search_below_height: float = 1500.0,
    min_connected_gates: int = 3,
    ze_min: float | None = None,
    min_peak_ze: float | None = None,
    usable: np.ndarray | None = None,
) -> bool:
    """Check for a vertically connected MRR echo near the profile bottom.

    A gate contains an echo when both Ze and VD are finite and, if ``ze_min``
    is supplied, Ze is at least that threshold. The function returns True if
    at least ``min_connected_gates`` consecutive echo gates occur below
    ``search_below_height``. An optional ``min_peak_ze`` can be used to reject
    weak connected structures that do not look like a convincing lower rain
    column.

    Parameters
    ----------
    ze, vd:
        One-dimensional profiles ordered from lowest to highest range gate.
    height:
        One-dimensional height coordinate matching ``ze`` and ``vd``.
    search_below_height:
        Only gates strictly below this height are considered part of the lower
        echo test.
    min_connected_gates:
        Required number of consecutive gates containing an echo.
    ze_min:
        Optional minimum reflectivity in dBZ.  Leave as None when the MRR
        product already represents no-signal gates as NaN.  Set a value when
        weak/noise values remain finite in the product.
    min_peak_ze:
        Optional minimum peak Ze among the connected lower-echo gates.
    usable:
        Optional Boolean instrument-gate mask.  Use this to exclude permanently
        unusable gates such as the radar blind zone.

    Notes
    -----
    This tests vertical connectivity only; it does not prove that the echo is
    precipitation.  It is intended to be combined with a rain flag and the
    known interference time interval.
    """

    ze = np.asarray(ze, dtype=float)
    vd = np.asarray(vd, dtype=float)
    height = np.asarray(height, dtype=float)
    if ze.ndim != 1 or vd.ndim != 1 or height.ndim != 1:
        raise ValueError("ze, vd and height must be one-dimensional arrays")
    if ze.shape != vd.shape or ze.shape != height.shape:
        raise ValueError("ze, vd and height must have the same shape")
    if not np.isfinite(search_below_height):
        raise ValueError("search_below_height must be finite")
    if min_connected_gates < 1:
        raise ValueError("min_connected_gates must be at least 1")

    if usable is None:
        usable = np.ones(ze.size, dtype=bool)
    else:
        usable = np.asarray(usable, dtype=bool)
        if usable.shape != ze.shape:
            raise ValueError("usable must have the same shape as ze and vd")

    lower_region = usable & np.isfinite(height) & (height < search_below_height)
    search_indices = np.flatnonzero(lower_region)
    if search_indices.size < min_connected_gates:
        return False

    echo = np.isfinite(ze) & np.isfinite(vd) & lower_region
    if ze_min is not None:
        echo &= ze >= ze_min

    run_length = 0
    previous_index = None
    peak_ze_in_run = -np.inf
    for index in search_indices:
        # Even after unusable gates are excluded, do not connect observations
        # separated by a missing range gate in the original vertical grid.
        consecutive_height = previous_index is not None and index == previous_index + 1
        if echo[index]:
            if consecutive_height:
                run_length += 1
                peak_ze_in_run = max(peak_ze_in_run, ze[index])
            else:
                run_length = 1
                peak_ze_in_run = ze[index]
            if run_length >= min_connected_gates and (
                min_peak_ze is None or peak_ze_in_run >= min_peak_ze
            ):
                return True
        else:
            run_length = 0
            peak_ze_in_run = -np.inf
        previous_index = index

    return False


def mrr_has_deep_continuous_ze(
    ze: np.ndarray,
    height: np.ndarray,
    *,
    ze_min: float | None = None,
    min_vertical_extent_m: float = 1500.0,
    min_layer_fraction: float = 0.95,
    max_missing_gates: int = 1,
) -> bool:
    """Check whether one Ze profile contains a deep, mostly continuous layer.

    The reference vertical extent is computed from the profile itself: the
    height of the lowest finite Ze gate to the height of the highest finite Ze
    gate at that timestamp. This protects deep cloud columns without assuming
    that the relevant layer always starts at a fixed height.
    """
    ze = np.asarray(ze, dtype=float)
    height = np.asarray(height, dtype=float)
    if ze.ndim != 1 or height.ndim != 1 or ze.shape != height.shape:
        raise ValueError("ze and height must be one-dimensional arrays of equal length")
    if not 0 < min_layer_fraction <= 1:
        raise ValueError("min_layer_fraction must be in the interval (0, 1]")
    if max_missing_gates < 0:
        raise ValueError("max_missing_gates must be non-negative")

    echo = np.isfinite(ze) & np.isfinite(height)
    if ze_min is not None:
        echo &= ze >= ze_min

    echo_indices = np.flatnonzero(echo)
    if echo_indices.size < 2:
        return False

    profile_extent = float(height[echo_indices[-1]] - height[echo_indices[0]])
    if profile_extent < min_vertical_extent_m:
        return False

    required_extent = profile_extent * min_layer_fraction
    best_extent = 0.0
    run_start = echo_indices[0]
    previous = echo_indices[0]
    for index in echo_indices[1:]:
        gap = index - previous - 1
        if gap <= max_missing_gates:
            previous = index
            continue
        best_extent = max(best_extent, float(height[previous] - height[run_start]))
        run_start = index
        previous = index
    best_extent = max(best_extent, float(height[previous] - height[run_start]))

    return best_extent >= required_extent


def mrr_has_top_rooted_ze_extent(
    ze: np.ndarray,
    height: np.ndarray,
    *,
    min_vertical_extent_m: float = 1000.0,
    ze_min: float | None = None,
    max_missing_gates: int = 0,
) -> bool:
    """Return True when echo starts at the highest range gate and extends downward.

    The highest finite height gate must contain finite Ze. The connected echo
    layer is then followed downward, allowing up to ``max_missing_gates``
    missing gates. The profile is protected when that top-rooted layer spans at
    least ``min_vertical_extent_m``.
    """
    ze = np.asarray(ze, dtype=float)
    height = np.asarray(height, dtype=float)
    if ze.ndim != 1 or height.ndim != 1 or ze.shape != height.shape:
        raise ValueError("ze and height must be one-dimensional arrays of equal length")
    if min_vertical_extent_m < 0:
        raise ValueError("min_vertical_extent_m must be non-negative")
    if max_missing_gates < 0:
        raise ValueError("max_missing_gates must be non-negative")

    finite_height_indices = np.flatnonzero(np.isfinite(height))
    if finite_height_indices.size == 0:
        return False

    top_index = int(finite_height_indices[-1])
    echo = np.isfinite(ze) & np.isfinite(height)
    if ze_min is not None:
        echo &= ze >= ze_min
    if not echo[top_index]:
        return False

    bottom_index = top_index
    missing_run = 0
    for index in range(top_index - 1, -1, -1):
        if not np.isfinite(height[index]):
            continue
        if echo[index]:
            bottom_index = index
            missing_run = 0
            continue
        missing_run += 1
        if missing_run > max_missing_gates:
            break

    return float(height[top_index] - height[bottom_index]) >= min_vertical_extent_m


def calculate_mean_interference_vertical_extent(
    ze: np.ndarray,
    height: np.ndarray,
    *,
    min_lowest_echo_height: float = 1400.0,
    min_connected_gates: int = 4,
) -> float:
    """Mean vertical extent of elevated-only Ze profiles for one day.

    A profile contributes when the lowest gate of its first connected Ze
    segment is above ``min_lowest_echo_height``. Single stray gates below the
    main elevated layer are ignored by requiring at least
    ``min_connected_gates`` consecutive finite Ze gates. ``height`` may be a
    fixed ``(range,)`` vector or a time-dependent ``(time, range)`` array.
    """
    ze = np.asarray(ze, dtype=float)
    height = np.asarray(height, dtype=float)
    if ze.ndim != 2:
        raise ValueError("ze must have shape (time, range)")
    if height.ndim == 1:
        if height.size != ze.shape[1]:
            raise ValueError("height must have length ze.shape[1]")
    elif height.ndim == 2:
        if height.shape != ze.shape:
            raise ValueError("2-D height must have the same shape as ze")
    else:
        raise ValueError("height must have shape (range,) or (time, range)")
    if min_connected_gates < 1:
        raise ValueError("min_connected_gates must be at least 1")

    extents = []
    for profile_index, profile in enumerate(ze):
        profile_height = height if height.ndim == 1 else height[profile_index]
        finite_indices = np.flatnonzero(np.isfinite(profile) & np.isfinite(profile_height))
        if finite_indices.size < min_connected_gates:
            continue

        connected_edges = np.diff(finite_indices) == 1
        connected_gate_mask = np.zeros(finite_indices.size, dtype=bool)
        if finite_indices.size == 1:
            connected_gate_mask[0] = min_connected_gates == 1
        else:
            connected_edge_runs = _mark_true_runs(connected_edges, min_connected_gates - 1)
            edge_indices = np.flatnonzero(connected_edge_runs)
            connected_gate_mask[edge_indices] = True
            connected_gate_mask[edge_indices + 1] = True

        connected_indices = finite_indices[connected_gate_mask]
        if connected_indices.size < min_connected_gates:
            continue

        lowest_height = float(profile_height[connected_indices[0]])
        if lowest_height <= min_lowest_echo_height:
            continue
        highest_height = float(profile_height[connected_indices[-1]])
        extent = highest_height - lowest_height
        if extent > 0:
            extents.append(extent)

    if not extents:
        return np.nan
    return float(np.nanmean(extents))


def mrr_is_elevated_only_ze_profile(
    ze: np.ndarray,
    height: np.ndarray,
    *,
    min_lowest_echo_height: float = 1400.0,
    ze_min: float | None = None,
    min_connected_gates: int = 4,
) -> bool:
    """Return True when the lowest finite Ze gate is above a height threshold.

    This identifies profiles whose first connected Ze segment is elevated above
    the lower atmosphere and therefore can be compared against the typical
    daily interference vertical extent. Single stray low gates are ignored by
    requiring at least ``min_connected_gates`` consecutive finite Ze gates.
    """
    ze = np.asarray(ze, dtype=float)
    height = np.asarray(height, dtype=float)
    if ze.ndim != 1 or height.ndim != 1 or ze.shape != height.shape:
        raise ValueError("ze and height must be one-dimensional arrays of equal length")
    if min_connected_gates < 1:
        raise ValueError("min_connected_gates must be at least 1")

    finite = np.isfinite(ze) & np.isfinite(height)
    if ze_min is not None:
        finite &= ze >= ze_min

    finite_indices = np.flatnonzero(finite)
    if finite_indices.size < min_connected_gates:
        return False

    if min_connected_gates == 1:
        first_connected_index = finite_indices[0]
    else:
        connected_edges = np.diff(finite_indices) == 1
        connected_edge_runs = _mark_true_runs(connected_edges, min_connected_gates - 1)
        if not connected_edge_runs.any():
            return False
        first_connected_edge = int(np.flatnonzero(connected_edge_runs)[0])
        first_connected_index = finite_indices[first_connected_edge]

    return float(height[first_connected_index]) > min_lowest_echo_height


def mrr_has_continuous_ze_starting_below_height(
    ze: np.ndarray,
    height: np.ndarray,
    *,
    start_below_height: float = 1500.0,
    ze_min: float | None = None,
    min_connected_gates: int = 4,
    max_missing_gates: int = 0,
) -> bool:
    """Return True when the first connected Ze segment starts below a height.

    The profile is treated as a plausible rain column when its first connected
    Ze segment has at least ``min_connected_gates`` gates and begins below
    ``start_below_height``.
    """
    ze = np.asarray(ze, dtype=float)
    height = np.asarray(height, dtype=float)
    if ze.ndim != 1 or height.ndim != 1 or ze.shape != height.shape:
        raise ValueError("ze and height must be one-dimensional arrays of equal length")
    if min_connected_gates < 1:
        raise ValueError("min_connected_gates must be at least 1")
    if max_missing_gates < 0:
        raise ValueError("max_missing_gates must be non-negative")

    valid = np.isfinite(ze) & np.isfinite(height)
    if ze_min is not None:
        valid &= ze >= ze_min

    valid_indices = np.flatnonzero(valid)
    if valid_indices.size < min_connected_gates:
        return False

    run_start = valid_indices[0]
    previous = valid_indices[0]
    run_length = 1
    for index in valid_indices[1:]:
        gap = index - previous - 1
        if gap <= max_missing_gates:
            run_length += 1
        else:
            if run_length >= min_connected_gates:
                return float(height[run_start]) < start_below_height
            run_start = index
            run_length = 1
        previous = index

    if run_length >= min_connected_gates:
        return float(height[run_start]) < start_below_height
    return False


def keep_lowest_connected_ze_component(
    ze: np.ndarray,
    vd: np.ndarray,
    *,
    ze_min: float | None = None,
    max_missing_gates: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep only the lowest connected Ze/VD component in one profile.

    This is intended for rainy profiles in the interference window: when a real
    lower echo exists, detached upper fragments are treated as interference and
    removed even if the shape-based detector did not flag them.
    """
    ze = np.asarray(ze, dtype=float)
    vd = np.asarray(vd, dtype=float)
    if ze.ndim != 1 or vd.ndim != 1 or ze.shape != vd.shape:
        raise ValueError("ze and vd must be one-dimensional arrays of equal length")
    if max_missing_gates < 0:
        raise ValueError("max_missing_gates must be non-negative")

    finite = np.isfinite(ze) & np.isfinite(vd)
    if ze_min is not None:
        finite &= ze >= ze_min

    finite_indices = np.flatnonzero(finite)
    if finite_indices.size == 0:
        return ze.copy(), vd.copy()

    keep = np.zeros(ze.size, dtype=bool)
    keep[finite_indices[0]] = True
    previous = finite_indices[0]
    for index in finite_indices[1:]:
        gap = index - previous - 1
        if gap > max_missing_gates:
            break
        keep[index] = True
        previous = index

    ze_out = ze.copy()
    vd_out = vd.copy()
    ze_out[~keep] = np.nan
    vd_out[~keep] = np.nan
    return ze_out, vd_out


def mask_short_detached_ze_columns(
    ze: np.ndarray,
    height: np.ndarray,
    *,
    lower_root_max_height: float = 1500.0,
    min_lower_connected_gates: int = 4,
    max_missing_gates: int = 0,
    max_detached_vertical_extent_m: float = 800.0,
    min_detached_base_height_m: float = 1500.0,
    ze_min: float | None = None,
) -> np.ndarray:
    """Mask short detached Ze components above a lower-rooted profile.

    This targets small upper fragments that sit above a lower echo. The lower
    component must start below ``lower_root_max_height`` and have at least
    ``min_lower_connected_gates`` gates. Detached components whose base is at or
    above ``min_detached_base_height_m`` and whose vertical extent is no larger
    than ``max_detached_vertical_extent_m`` are marked for removal.
    """
    ze = np.asarray(ze, dtype=float)
    height = np.asarray(height, dtype=float)
    if ze.ndim != 1 or height.ndim != 1 or ze.shape != height.shape:
        raise ValueError("ze and height must be one-dimensional arrays of equal length")
    if min_lower_connected_gates < 1:
        raise ValueError("min_lower_connected_gates must be at least 1")
    if max_missing_gates < 0:
        raise ValueError("max_missing_gates must be non-negative")
    if max_detached_vertical_extent_m < 0:
        raise ValueError("max_detached_vertical_extent_m must be non-negative")

    valid = np.isfinite(ze) & np.isfinite(height)
    if ze_min is not None:
        valid &= ze >= ze_min

    valid_indices = np.flatnonzero(valid)
    mask = np.zeros(ze.size, dtype=bool)
    if valid_indices.size == 0:
        return mask

    components: list[tuple[int, int]] = []
    start = int(valid_indices[0])
    previous = int(valid_indices[0])
    for raw_index in valid_indices[1:]:
        index = int(raw_index)
        gap = index - previous - 1
        if gap <= max_missing_gates:
            previous = index
            continue
        components.append((start, previous))
        start = index
        previous = index
    components.append((start, previous))

    lower_component_found = False
    for start, end in components:
        gate_count = end - start + 1
        if float(height[start]) < lower_root_max_height and gate_count >= min_lower_connected_gates:
            lower_component_found = True
            break
    if not lower_component_found:
        return mask

    for start, end in components:
        component_base_height = float(height[start])
        if component_base_height < min_detached_base_height_m:
            continue
        component_extent = float(height[end] - height[start])
        if component_extent <= max_detached_vertical_extent_m:
            mask[start : end + 1] = True

    return mask


def detect_velocity_plateaus(
    vd: np.ndarray,
    *,
    tolerance: float = 0.075,
    min_gates: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect nearly constant VD over consecutive range gates.

    Two adjacent finite gates are connected when their absolute VD difference is
    no larger than ``tolerance``.  All gates in a connected run are flagged when
    the run contains at least ``min_gates`` gates.

    Parameters
    ----------
    vd:
        One-dimensional mean Doppler velocity profile, ordered by height.
    tolerance:
        Maximum difference between adjacent gates, in the same units as ``vd``
        (normally m/s).
    min_gates:
        Minimum number of consecutive gates in a plateau.

    Returns
    -------
    mask, step:
        Boolean plateau mask and absolute adjacent-gate VD difference.  The
        first value of ``step`` is NaN because it has no preceding gate.
    """
    vd = np.asarray(vd, dtype=float)
    if vd.ndim != 1:
        raise ValueError("vd must be a one-dimensional profile")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    if min_gates < 2:
        raise ValueError("min_gates must be at least 2")

    step = np.full(vd.size, np.nan)
    if vd.size > 1:
        step[1:] = np.abs(np.diff(vd))

    adjacent_match = (
        np.isfinite(vd[:-1])
        & np.isfinite(vd[1:])
        & (np.abs(np.diff(vd)) <= tolerance)
    )
    matched_edges = _mark_true_runs(adjacent_match, min_gates - 1)

    mask = np.zeros(vd.size, dtype=bool)
    edge_indices = np.flatnonzero(matched_edges)
    mask[edge_indices] = True
    mask[edge_indices + 1] = True
    return mask, step


def detect_ze_zigzags(
    ze: np.ndarray,
    *,
    min_step: float = 2.0,
    min_turns: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detect repeated alternating changes in a Ze profile.

    A turn occurs at gate ``h`` if the slopes on its two sides have opposite
    signs and both changes have magnitude at least ``min_step``.  Consecutive
    turns are retained when there are at least ``min_turns`` of them.  The mask
    includes the end gates participating in the zigzag.

    Parameters
    ----------
    ze:
        One-dimensional reflectivity profile in dBZ, ordered by height.
    min_step:
        Minimum absolute reflectivity change between neighboring gates in dBZ.
        This prevents tiny noise fluctuations from being called a zigzag.
    min_turns:
        Minimum number of consecutive direction changes.  Two turns correspond
        to a pattern involving at least four gates.

    Returns
    -------
    mask, step, turn:
        Boolean zigzag mask, signed adjacent-gate Ze difference, and Boolean
        mask of the central turning gates.
    """
    ze = np.asarray(ze, dtype=float)
    if ze.ndim != 1:
        raise ValueError("ze must be a one-dimensional profile")
    if min_step < 0:
        raise ValueError("min_step must be non-negative")
    if min_turns < 1:
        raise ValueError("min_turns must be at least 1")

    step = np.full(ze.size, np.nan)
    if ze.size > 1:
        step[1:] = np.diff(ze)

    turn = np.zeros(ze.size, dtype=bool)
    if ze.size >= 3:
        left = np.diff(ze)[:-1]
        right = np.diff(ze)[1:]
        finite = np.isfinite(ze[:-2]) & np.isfinite(ze[1:-1]) & np.isfinite(ze[2:])
        turn[1:-1] = (
            finite
            & (left * right < 0.0)
            & (np.abs(left) >= min_step)
            & (np.abs(right) >= min_step)
        )

    retained_turns = _mark_true_runs(turn, min_turns)
    mask = retained_turns.copy()
    indices = np.flatnonzero(retained_turns)
    mask[np.maximum(indices - 1, 0)] = True
    mask[np.minimum(indices + 1, ze.size - 1)] = True
    return mask, step, retained_turns


def check_profile(
    ze: np.ndarray,
    vd: np.ndarray,
    *,
    vd_tolerance: float = 0.075,
    plateau_min_gates: int = 3,
    ze_min_step: float = 2.0,
    zigzag_min_turns: int = 2,
    combine: str = "or",
) -> ProfileInterferenceResult:
    """Check one Ze/VD profile for interference-like vertical structure.

    ``combine='and'`` flags only gates where the two expanded pattern masks
    overlap.  ``combine='or'`` flags either signature and is more sensitive but
    less specific.
    """
    ze = np.asarray(ze, dtype=float)
    vd = np.asarray(vd, dtype=float)
    if ze.shape != vd.shape:
        raise ValueError("ze and vd must have the same shape")

    plateau, vd_step = detect_velocity_plateaus(
        vd, tolerance=vd_tolerance, min_gates=plateau_min_gates
    )
    zigzag, ze_step, ze_turn = detect_ze_zigzags(
        ze, min_step=ze_min_step, min_turns=zigzag_min_turns
    )

    if combine == "and":
        combined = plateau & zigzag
    elif combine == "or":
        combined = plateau | zigzag
    else:
        raise ValueError("combine must be 'and' or 'or'")

    combined &= np.isfinite(ze) & np.isfinite(vd)
    return ProfileInterferenceResult(
        velocity_plateau=plateau,
        ze_zigzag=zigzag,
        combined=combined,
        vd_step=vd_step,
        ze_step=ze_step,
        ze_turn=ze_turn,
    )



def mask_upper_interference(
    ze: np.ndarray,
    vd: np.ndarray,
    height: np.ndarray | None = None,
    *,
    min_lower_gates: int = 3,
    evidence_window: int = 4,
    min_evidence_gates: int = 2,
    mask_all_above: bool = True,
    vd_tolerance: float = 0.15,
    plateau_min_gates: int = 4,
    ze_min_step: float = 1.0,
    zigzag_min_turns: int = 2,
    combine: str = "and",
    interference_above_height: float | None = None,
) -> UpperInterferenceResult:
    """Preserve lower rain and mask interference in the upper profile.

    The input must be ordered from the lowest to the highest range gate.  The
    function searches upward, after ``min_lower_gates`` protected lower gates,
    for the first window containing sustained interference evidence. With
    ``combine='and'``, both the VD plateau and Ze zigzag are required; with
    ``combine='or'``, either signature is sufficient. Its first evidence gate
    becomes the interference cutoff.

    By default, every finite observation from the cutoff upward is masked. This
    is appropriate when independent knowledge says that the upper portion is
    entirely interference. Set ``mask_all_above=False`` to mask only gates that
    belong to the selected combined evidence mask, which is more conservative
    but may leave some weak interference unmasked.

    If ``interference_above_height`` is supplied, gates at or below that height
    are protected and cannot provide detection evidence or be masked.

    This routine deliberately does not infer whether the profile is rainy.  It
    should be called only for profiles already classified as precipitation plus
    interference (for example using a rain flag and the cable-car time window).
    """
    ze = np.asarray(ze, dtype=float)
    vd = np.asarray(vd, dtype=float)
    if ze.ndim != 1 or vd.ndim != 1 or ze.shape != vd.shape:
        raise ValueError("ze and vd must be one-dimensional arrays of equal length")
    if min_lower_gates < 0:
        raise ValueError("min_lower_gates must be non-negative")
    if evidence_window < 1:
        raise ValueError("evidence_window must be at least 1")
    if not 1 <= min_evidence_gates <= evidence_window:
        raise ValueError("min_evidence_gates must be between 1 and evidence_window")

    if interference_above_height is not None and height is None:
        raise ValueError(
            "height is required when interference_above_height is supplied"
        )

    if height is not None:
        height = np.asarray(height, dtype=float)
        if height.shape != ze.shape:
            raise ValueError("height must have the same shape as ze and vd")
        finite_height = height[np.isfinite(height)]
        if finite_height.size > 1 and np.any(np.diff(finite_height) <= 0):
            raise ValueError("height must increase from one range gate to the next")

    detection = check_profile(
        ze,
        vd,
        vd_tolerance=vd_tolerance,
        plateau_min_gates=plateau_min_gates,
        ze_min_step=ze_min_step,
        zigzag_min_turns=zigzag_min_turns,
        combine=combine,
    )
    evidence = detection.combined.copy()
    eligible = np.ones(ze.size, dtype=bool)
    if interference_above_height is not None:
        eligible = np.isfinite(height) & (height > interference_above_height)
        evidence &= eligible

    cutoff = None
    last_start = ze.size - evidence_window
    for start in range(min_lower_gates, last_start + 1):
        if not eligible[start]:
            continue
        window = evidence[start : start + evidence_window]
        if np.count_nonzero(window) >= min_evidence_gates:
            # Start at the first actual evidence gate, not necessarily at the
            # beginning of the search window.
            cutoff = start + int(np.flatnonzero(window)[0])
            break

    mask = np.zeros(ze.size, dtype=bool)
    if cutoff is not None:
        if mask_all_above:
            mask[cutoff:] = eligible[cutoff:] & (
                np.isfinite(ze[cutoff:]) | np.isfinite(vd[cutoff:])
            )
        else:
            mask[cutoff:] = evidence[cutoff:]

    ze_filtered = ze.copy()
    vd_filtered = vd.copy()
    ze_filtered[mask] = np.nan
    vd_filtered[mask] = np.nan

    cutoff_height = None
    if cutoff is not None and height is not None and np.isfinite(height[cutoff]):
        cutoff_height = float(height[cutoff])

    return UpperInterferenceResult(
        ze_filtered=ze_filtered,
        vd_filtered=vd_filtered,
        mask=mask,
        evidence=evidence,
        cutoff_index=cutoff,
        cutoff_height=cutoff_height,
    )


def plot_time_height_Ze(ds_mrr, date_selected, info_output, time_stamps):
    """
    Plot time-height radar reflectivity (Ze) for the entire day.

    Parameters:
    ds_mrr : xarray.Dataset
        The MRR dataset.
    date_selected : str
        The selected date in the format 'YYYYMMDD'.
    time_stamps : list of str
        List of time stamps to for spectrogram plotting in the format 'YYYYMMDDTHHMMSS'.
    info_output : str
        Additional information for the output filename about input data or filtering applied.
    """

    # plot radar reflectivity as a function of height and time for the entire day
    Ze = ds_mrr.Ze.values
    time = ds_mrr.time.values
    height = ds_mrr.height.values

    plt.figure(figsize=(10, 6))
    finite_ze = np.isfinite(Ze)
    if height.ndim == 2:
        time_grid = np.broadcast_to(time[:, np.newaxis], Ze.shape)
        mesh = plt.pcolormesh(time_grid, height, Ze, shading='auto', cmap='viridis')
    else:
        mesh = plt.pcolormesh(time, height, Ze.T, shading='auto', cmap='viridis')
    plt.colorbar(mesh, label='Radar Reflectivity [dBZ]')
    if finite_ze.any():
        # Set colorbar limits to the min and max finite Ze values.
        mesh.set_clim(np.nanmin(Ze), np.nanmax(Ze))
    else:
        print(
            f"No finite Ze values available for {info_output} on {date_selected}; "
            "saving an empty reflectivity quicklook."
        )

    plt.ylabel('Height [m]')
    # format xaxis with time stamps as HH:MM
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
    plt.gcf().autofmt_xdate()
    #plt.title('MRR Radar Reflectivity')

    # plot vertical lines for the time stamps selected
    #for time_stamp in time_stamps:
    #    time_stamp_dt = pd.to_datetime(time_stamp)
    #    plt.axvline(x=time_stamp_dt, color='r', linestyle='--', label=f'Selected Time: {time_stamp_dt.strftime("%H:%M")}')  

    #if time_stamps:
    #    plt.legend()


    # Show the full instrument height range, even when lower gates contain no
    # finite Ze after filtering.
    plt.ylim(np.nanmin(height), np.nanmax(height))

    # add xlabel as time in UTC
    plt.xlabel('Time [UTC]', fontsize=14)

    # remove the top and right spines of the plot
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    # make the left and bottom spines thicker
    plt.gca().spines['left'].set_linewidth(1.5)
    plt.gca().spines['bottom'].set_linewidth(1.5)

    # enlange the font size of the ticks
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    #enlarge all fonts
    plt.rcParams.update({'font.size': 14})

    # save figure
    plt.savefig(f"plots/mrr_reflectivity_{info_output}_{date_selected}.png")
    return print(f"MRR data for {info_output} on {date_selected} plotted and saved.")


def read_mrr_data(path_mrr, site_selected, date_selected):
    """
    code to read the MRR data for the selected site and day and store in a dataset
    input:
    - path_mrr: path to the MRR data
    - site_selected: name of the selected site
    - date_selected: date of the selected data
    output:
    - ds_mrr: dataset containing the MRR data for the selected site

    dependencies:
    - find_file_mrr: function to find the file of the day and site selected
    - filter_interference: function to filter interference in the MRR data between 7 UTC and 15:30 UTC
    """
    # find the file of the day and site selected
    file_mrr = find_file_mrr(path_mrr, site_selected, date_selected)

    # unzip file it format ending is gz
    if file_mrr.endswith('nc.gz'):
        import gzip
        import shutil
        # extract only filename from the file path
        filename = file_mrr.split("/")[-1]

        # define the path for the unzipped file
        unzipped_file_path = f"{filename[:-3]}"  # Remove the .gz extension
        
        # unzip the file
        with gzip.open(file_mrr, 'rb') as f_in:
            with open(unzipped_file_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        file_mrr = unzipped_file_path

    # read the NetCDF file using xarray
    ds_mrr = xr.open_dataset(file_mrr)

    # remove the unzipped file to save space
    if os.path.exists(file_mrr) and file_mrr.endswith('.nc'):
        os.remove(file_mrr) 
        
    return ds_mrr


def find_file_mrr(path_mrr, site_selected, date_selected):
    """
    code to find the file of the day and site selected
    input:  
    - path_mrr: path to the MRR data
    - site_selected: name of the selected site
    - date_selected: date of the selected data
    output:
    - file_mrr: path to the MRR file for the selected site and day
    """
    if site_selected == "lagonero":
        site_selected = "lago"


    # build path with the date
    path_mrr = os.path.join(path_mrr, date_selected[:4], date_selected[4:6], date_selected[6:8])
    # list all files in the path
    files = os.listdir(path_mrr)

    # find the file of the day and site selected
    for file in files:
        if "improtoo" in file:
            if file.endswith("nc.gz"):
                print("Found file: ", file)
                file_mrr = os.path.join(path_mrr, file)
                return file_mrr

    raise FileNotFoundError(f"No MRR file found for site {site_selected} and date {date_selected} in path {path_mrr}")



def find_MRR_flag(site_selected, date):
    """
    code to find the MWR flag file for the selected site and day
    input:
    - site_selected: name of the selected site
    - date: date of the selected data
    output:
    - True if the MWR flag file exists, False otherwise
    """
    if site_selected == 'collalbo':
        instr = 'kithat'
    elif site_selected == 'lagonero':
        instr = 'tophat'
    elif site_selected == 'bolzano':
        instr = 'hatpro'

    path_root = '/data/obs/campaigns/teamx/' + site_selected + '/'+ instr +'/actris/level1/'
    
    # read yy, mm, dd from date
    yy = date[:4]
    mm = date[4:6]
    dd = date[6:8]
    
    path_global = path_root + yy + '/' + mm + '/' + dd + '/'
    filename = path_global + 'MWR_1C01_'+site_selected+'_'+date+'.nc'
    
    # check if the file exists
    if os.path.exists(filename):
        return True
    else:
        print(f"No MWR flag file found for site {site_selected} and date {date} in path {path_global}")
        return False



if __name__ == "__main__":

    try:
        from process.mrr_pipeline import main
    except ModuleNotFoundError:
        from mrr_pipeline import main

    main()
