"""Configuration loading for the MRR postprocessing pipeline."""

from __future__ import annotations

try:
    from process.remove_interfence_mrr import MRRInterferenceConfig, load_mrr_interference_config
except ModuleNotFoundError:
    from remove_interfence_mrr import MRRInterferenceConfig, load_mrr_interference_config

__all__ = ["MRRInterferenceConfig", "load_mrr_interference_config"]
