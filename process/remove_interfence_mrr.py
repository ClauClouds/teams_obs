"""MRR interference filtering driven by rain context and Ze/W profile structure.

This module implements a multi-stage filtering workflow for MRR profiles during
the known cable-car interference period at Lagonero. The code reads one MRR
day, aligns the radiometer rain flag to the MRR time grid, classifies each
profile according to the vertical structure of Ze and W, and then masks only
the parts of the column that are most likely contaminated by interference.

Overview of the filtering sequence
----------------------------------
1. Read one day of MRR data and normalize height to a fixed one-dimensional
    range coordinate when the input stores height on the time-range grid.

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

"""

from __future__ import annotations

from dataclasses import dataclass
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
    ``min_connected_gates`` consecutive finite Ze gates.
    """
    ze = np.asarray(ze, dtype=float)
    height = np.asarray(height, dtype=float)
    if ze.ndim != 2:
        raise ValueError("ze must have shape (time, range)")
    if height.ndim != 1 or height.size != ze.shape[1]:
        raise ValueError("height must be one-dimensional with length ze.shape[1]")
    if min_connected_gates < 1:
        raise ValueError("min_connected_gates must be at least 1")

    extents = []
    for profile in ze:
        finite_indices = np.flatnonzero(np.isfinite(profile) & np.isfinite(height))
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

        lowest_height = float(height[connected_indices[0]])
        if lowest_height <= min_lowest_echo_height:
            continue
        highest_height = float(height[connected_indices[-1]])
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
    plt.pcolormesh(time, height, Ze.T, shading='auto', cmap='viridis')
    plt.colorbar(label='Radar Reflectivity [dBZ]')
    plt.ylabel('Height [m]')

    # format xaxis with time stamps as HH:MM
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
    plt.gcf().autofmt_xdate()
    plt.title('MRR Radar Reflectivity')

    # plot vertical lines for the time stamps selected
    for time_stamp in time_stamps:
        time_stamp_dt = pd.to_datetime(time_stamp)
        plt.axvline(x=time_stamp_dt, color='r', linestyle='--', label=f'Selected Time: {time_stamp_dt.strftime("%H:%M")}')  

    if time_stamps:
        plt.legend()
    
    # set ylim min at the first height with Ze values non Nan
    min_height_idx = np.where(np.any(np.isfinite(Ze), axis=0))[0][0]
    plt.ylim(height[min_height_idx], height.max())

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

    # define the sites and the path to the MRR data and MWR data
    sites = ['lagonero', 'collalbo']
    site_selected = "lagonero"
    path_mrr = f"/data/campaigns/teamx/{site_selected}/mrr/l1/"

    # time stamps to plot to understand the interferences
    time_stamps = ["20250706T12:45:00","20250706T12:50:00"]
    date_selected = time_stamps[0][:8]

    filter_RR_on = True  # if True, then filter the MRR data based on the MWR rain flag
    min_interference_extent_factor = 1.0
    min_interference_layer_fraction = 0.95
    interference_min_lowest_echo_height = 1500.0
    min_interference_time_profiles = 2
    min_lower_echo_time_profiles = 2
    min_elevated_connected_gates = 4
    min_lower_echo_connected_gates = 4
    max_interference_missing_gates = 0
    max_rain_column_missing_gates = 0
    lower_echo_height_limit = 1500.0
    min_lower_echo_peak_ze = -5.0
    min_lower_continuous_ze_gates = 4

    # read the MRR data for the selected site and day and store in a dataset
    ds_mrr = read_mrr_data(path_mrr, site_selected, date_selected)


    # Convert the MRR height field from time-range to a fixed vertical coordinate.
    # This keeps Ze as a clean time x range matrix and prevents height from
    # being treated as a profile variable during rain masking.
    if "height" in ds_mrr and ds_mrr["height"].dims == ("time", "range"):
        height_1d = ds_mrr["height"].median(dim="time", skipna=True)
        ds_mrr = ds_mrr.drop_vars("height").assign_coords(height=("range", height_1d.values))

    os.makedirs("plots", exist_ok=True)

    # if MWR flag file exists, then select only the time stamps where rain flag is true
    if find_MRR_flag(site_selected, date_selected) and filter_RR_on: 

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

        mwr_reindex_tolerance = pd.Timedelta("2min")
        rain_extension = pd.Timedelta("3min")
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

        # plot the Ze time height plot for the day before filtering
        plot_time_height_Ze(ds_mrr, date_selected, info_output="before_filtering", time_stamps=time_stamps)

        # add rain flag to the MRR dataset
        ds_mrr = ds_mrr.assign(rain_flag=rain_flag)

        # define interference time window for the selected site
        if site_selected == "lagonero":
            interference_start = time(6, 30, 0)
            interference_end = time(16, 0, 0)

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

        vars_to_keep = {"height", "range", "lat", "lon", "latitude", "longitude", "altitude", "time", "rain_flag"}
        for var_name in ds_mrr.data_vars:
            if "time" in ds_mrr[var_name].dims and var_name not in vars_to_keep:
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
                    ds_mrr["Ze"].loc[dict(time=time_stamp)] = np.nan
                    ds_mrr["W"].loc[dict(time=time_stamp)] = np.nan
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
                    
                    # update the Ze and vd profiles with the filtered values
                    ds_mrr["Ze"].loc[dict(time=time_stamp)] = ze_filtered
                    ds_mrr["W"].loc[dict(time=time_stamp)] = vd_filtered
                
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

                    ds_mrr["Ze"].loc[dict(time=time_stamp)] = ze_filtered
                    ds_mrr["W"].loc[dict(time=time_stamp)] = vd_filtered

                elif mwr_rain_flag and (not lower_echo or not profile_persistent_lower_echo):
                    if not profile_persistent_deep_continuous_ze:
                        ds_mrr["Ze"].loc[dict(time=time_stamp)] = np.nan
                        ds_mrr["W"].loc[dict(time=time_stamp)] = np.nan
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

                    # update the Ze and vd profiles with the filtered values
                    ds_mrr["Ze"].loc[dict(time=time_stamp)] = Ze_filtered
                    ds_mrr["W"].loc[dict(time=time_stamp)] = vd_filtered
                else:
                    continue  # No interference, no filtering to apply

                    
        # plot the Ze time height plot for the day after filtering
        plot_time_height_Ze(ds_mrr, date_selected, info_output="after_filtering", time_stamps=time_stamps)

    