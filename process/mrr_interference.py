"""Interference detection and masking helpers for MRR profiles."""

from __future__ import annotations

try:
    from process.remove_interfence_mrr import (
        PROFILE_MASK_SKIP_VARS,
        ProfileInterferenceResult,
        UpperInterferenceResult,
        _mark_true_runs,
        apply_range_gate_mask_to_profile,
        calculate_mean_interference_vertical_extent,
        check_profile,
        detect_velocity_plateaus,
        detect_ze_zigzags,
        keep_lowest_connected_ze_component,
        mask_upper_interference,
        mrr_has_continuous_ze_starting_below_height,
        mrr_has_deep_continuous_ze,
        mrr_has_lower_echo,
        mrr_has_top_rooted_ze_extent,
        mrr_is_elevated_only_ze_profile,
    )
except ModuleNotFoundError:
    from remove_interfence_mrr import (
        PROFILE_MASK_SKIP_VARS,
        ProfileInterferenceResult,
        UpperInterferenceResult,
        _mark_true_runs,
        apply_range_gate_mask_to_profile,
        calculate_mean_interference_vertical_extent,
        check_profile,
        detect_velocity_plateaus,
        detect_ze_zigzags,
        keep_lowest_connected_ze_component,
        mask_upper_interference,
        mrr_has_continuous_ze_starting_below_height,
        mrr_has_deep_continuous_ze,
        mrr_has_lower_echo,
        mrr_has_top_rooted_ze_extent,
        mrr_is_elevated_only_ze_profile,
    )

__all__ = [
    "PROFILE_MASK_SKIP_VARS",
    "ProfileInterferenceResult",
    "UpperInterferenceResult",
    "_mark_true_runs",
    "apply_range_gate_mask_to_profile",
    "calculate_mean_interference_vertical_extent",
    "check_profile",
    "detect_velocity_plateaus",
    "detect_ze_zigzags",
    "keep_lowest_connected_ze_component",
    "mask_upper_interference",
    "mrr_has_continuous_ze_starting_below_height",
    "mrr_has_deep_continuous_ze",
    "mrr_has_lower_echo",
    "mrr_has_top_rooted_ze_extent",
    "mrr_is_elevated_only_ze_profile",
]
