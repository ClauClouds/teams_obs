"""Monte-Carlo uncertainty propagation for Micro Rain Radar Doppler moments.

The input spectrum must be in *linear* units and have dimensions such as
(`time`, `height`, `velocity`).  It can be spectral reflectivity (preferred) or
another spectral quantity proportional to reflectivity.  In the latter case,
set ``reflectivity_scale`` so that

    Z_linear = reflectivity_scale * integral(spectrum dV).

Random spectral uncertainty is propagated independently at every time-height
pixel.  Calibration uncertainty is reported separately because it is normally
systematic, not an independent error for every pixel.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import xarray as xr


MOMENT_NAMES = ("ze", "mean_velocity", "spectral_width", "skewness", "kurtosis")


def atlas_terminal_velocity_to_diameter(terminal_velocity: np.ndarray) -> np.ndarray:
    """Invert the Atlas-type rain-drop fall-speed relation.

    Uses ``v_t = 9.65 - 10.3 exp(-0.6 D)``, with ``v_t`` in m s-1 and drop
    diameter ``D`` in mm. Values outside the invertible liquid-drop interval
    are returned as NaN.
    """
    vt = np.asarray(terminal_velocity, dtype=float)
    valid = (vt > 0.0) & (vt < 9.65)
    ratio = np.where(valid, (9.65 - vt) / 10.3, np.nan)
    diameter = -np.log(ratio) / 0.6
    return np.where((diameter > 0.0) & np.isfinite(diameter), diameter, np.nan)


def _rain_rate_from_spectrum(
    samples: np.ndarray,
    measured_velocity: np.ndarray,
    dv: np.ndarray,
    vertical_air_velocity: np.ndarray,
) -> np.ndarray:
    """Rain rate from spectral reflectivity, in mm h-1.

    ``samples`` must be spectral reflectivity in mm6 m-3 (m s-1)-1.
    Radar velocity is positive downward. Vertical air velocity is positive
    upward, hence terminal fall speed is v_t = v_measured + w_air.
    """
    vt = measured_velocity + vertical_air_velocity[..., None]
    diameter = atlas_terminal_velocity_to_diameter(vt)
    z_bin = samples * dv
    # Since Z_bin = D^6 N(D)dD and
    # R = 6*pi*1e-4 integral[v_t D^3 N(D)dD].
    contribution = np.where(
        np.isfinite(diameter), vt * z_bin / diameter**3, 0.0
    )
    rain_rate = 6.0e-4 * np.pi * np.sum(contribution, axis=-1)
    valid = np.any((samples > 0) & np.isfinite(diameter), axis=-1)
    return np.where(valid, rain_rate, np.nan)


def estimate_noise_std(
    spectrum: xr.DataArray,
    velocity_dim: str = "velocity",
    lowest_fraction: float = 0.20,
) -> xr.DataArray:
    """Estimate per-spectrum, per-bin random noise using low-power bins.

    This is a fallback estimator.  A noise uncertainty produced by the actual
    spectral processor (for example, from noise-only bins or an HS estimator)
    is preferable.

    The returned value has all input dimensions except ``velocity_dim`` and is
    interpreted as the 1-sigma uncertainty of each spectral bin.
    """
    if not 0.05 <= lowest_fraction <= 0.5:
        raise ValueError("lowest_fraction must be between 0.05 and 0.5")

    axis = spectrum.get_axis_num(velocity_dim)
    values = np.asarray(spectrum.values, dtype=float)
    n_low = max(4, int(np.ceil(values.shape[axis] * lowest_fraction)))
    sorted_values = np.sort(values, axis=axis)
    low = np.take(sorted_values, np.arange(n_low), axis=axis)

    median = np.nanmedian(low, axis=axis, keepdims=True)
    mad = np.nanmedian(np.abs(low - median), axis=axis)
    sigma = 1.4826 * mad

    # MAD can be zero in quantised or already thresholded spectra.
    fallback = np.nanstd(low, axis=axis, ddof=1)
    sigma = np.where((sigma > 0) & np.isfinite(sigma), sigma, fallback)

    dims = tuple(d for d in spectrum.dims if d != velocity_dim)
    coords = {d: spectrum.coords[d] for d in dims if d in spectrum.coords}
    return xr.DataArray(sigma, dims=dims, coords=coords, name="spectral_noise_std")


def _bin_widths(velocity: np.ndarray) -> np.ndarray:
    """Return integration widths for possibly nonuniform velocity bins."""
    velocity = np.asarray(velocity, dtype=float)
    if velocity.ndim != 1 or velocity.size < 2:
        raise ValueError("velocity coordinate must be one-dimensional with >=2 bins")
    if not np.all(np.isfinite(velocity)):
        raise ValueError("velocity coordinate contains non-finite values")
    edges = np.empty(velocity.size + 1)
    edges[1:-1] = 0.5 * (velocity[:-1] + velocity[1:])
    edges[0] = velocity[0] - 0.5 * (velocity[1] - velocity[0])
    edges[-1] = velocity[-1] + 0.5 * (velocity[-1] - velocity[-2])
    widths = np.abs(np.diff(edges))
    if np.any(widths <= 0):
        raise ValueError("velocity bins must be strictly monotonic")
    return widths


def _moments(samples: np.ndarray, velocity: np.ndarray, dv: np.ndarray, scale: float):
    """Moments for array shaped (realisation, ..., velocity)."""
    weights = samples * dv
    m0 = np.sum(weights, axis=-1)
    valid = m0 > 0

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.sum(weights * velocity, axis=-1) / m0
        delta = velocity - mean[..., None]
        variance = np.sum(weights * delta**2, axis=-1) / m0
        width = np.sqrt(np.maximum(variance, 0.0))
        skewness = np.sum(weights * delta**3, axis=-1) / (m0 * width**3)
        kurtosis = np.sum(weights * delta**4, axis=-1) / (m0 * width**4)

    outputs = [scale * m0, mean, width, skewness, kurtosis]
    return [np.where(valid, x, np.nan) for x in outputs]


def mrr_moment_uncertainty(
    spectrum: xr.DataArray,
    *,
    velocity: Optional[xr.DataArray] = None,
    noise_std: Optional[xr.DataArray] = None,
    velocity_dim: str = "velocity",
    n_realizations: int = 300,
    reflectivity_scale: float = 1.0,
    min_snr_linear: Optional[float] = None,
    spectral_threshold_sigma: float = 1.0,
    min_valid_bins: int = 2,
    calibration_uncertainty_db: float = 0.0,
    attenuation_uncertainty_db: Optional[xr.DataArray] = None,
    calculate_rain_rate: bool = False,
    vertical_air_velocity: float | xr.DataArray = 0.0,
    vertical_air_velocity_uncertainty: float | xr.DataArray = 0.0,
    rain_rate_relative_model_uncertainty: float = 0.0,
    lowest_noise_fraction: float = 0.20,
    random_seed: Optional[int] = None,
    chunk_size: int = 32,
) -> xr.Dataset:
    """Propagate Doppler-spectrum uncertainty into radar moments.

    Parameters
    ----------
    spectrum
        Linear, preferably noise-subtracted Doppler spectrum.  Negative and
        non-finite values are treated as invalid/zero during Monte Carlo runs.
    velocity
        One-dimensional Doppler velocity. Defaults to ``spectrum[velocity_dim]``.
    noise_std
        1-sigma random uncertainty of a spectral bin. It may be scalar, have
        dimensions (time, height), or have the full spectrum dimensions.
        If omitted, :func:`estimate_noise_std` is used.
    reflectivity_scale
        Multiplicative factor converting integrated spectrum to linear Z.
        Use 1 when the spectrum is spectral reflectivity per (m/s).
    min_snr_linear
        Optional integrated signal-to-noise threshold. Realizations below it
        are set to NaN. Noise power is approximated as sum(noise_std * dv).
    spectral_threshold_sigma
        Bins whose nominal signal is below this multiple of ``noise_std`` are
        excluded from all realizations. This avoids positive bias caused by
        clipping perturbed noise-only bins at zero. Set to 0 only if the input
        already contains a reliable spectral signal mask.
    calibration_uncertainty_db
        Systematic 1-sigma reflectivity calibration uncertainty in dB.
    attenuation_uncertainty_db
        Optional 1-sigma attenuation-correction uncertainty in dB, broadcastable
        to the spectrum without its velocity dimension.
    calculate_rain_rate
        Calculate spectral-DSD rain rate and its Monte Carlo uncertainty.
        This requires ``spectrum`` to be spectral reflectivity in
        mm6 m-3 (m s-1)-1 and velocity to be positive downward.
    vertical_air_velocity
        Estimated vertical air velocity in m s-1, positive upward. May be a
        scalar or DataArray broadcastable to (time, height).
    vertical_air_velocity_uncertainty
        Its 1-sigma uncertainty in m s-1. A single perturbation is drawn per
        spectrum and realization, so the shift is correlated across velocity
        bins as it should be.
    rain_rate_relative_model_uncertainty
        Optional fractional 1-sigma uncertainty representing residual fall-
        speed/scattering/DSD-model uncertainty (e.g. 0.10 for 10 percent).

    Returns
    -------
    xr.Dataset
        For every moment: nominal value, MC standard uncertainty and p16/p84.
        Reflectivity is returned in linear units and dBZ. Random and systematic
        dBZ uncertainty components are kept separately.

    Notes
    -----
    Perturbations are Gaussian, suitable when each stored spectrum is an average
    of many independent spectra. If the number of incoherent averages is known
    and small, a gamma/chi-square power model is more physically appropriate.
    """
    if velocity_dim not in spectrum.dims:
        raise ValueError(f"{velocity_dim!r} is not a dimension of spectrum")
    if n_realizations < 20:
        raise ValueError("Use at least 20 Monte Carlo realizations")
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    if rain_rate_relative_model_uncertainty < 0:
        raise ValueError("rain_rate_relative_model_uncertainty must be >= 0")

    spectrum = spectrum.transpose(..., velocity_dim)
    velocity = spectrum[velocity_dim] if velocity is None else velocity
    v = np.asarray(velocity.values, dtype=float)
    dv = _bin_widths(v)

    if noise_std is None:
        noise_std = estimate_noise_std(spectrum, velocity_dim, lowest_noise_fraction)
    noise_b, spectrum_b = xr.broadcast(noise_std, spectrum)
    noise_b = noise_b.transpose(*spectrum.dims)

    signal = np.where(np.isfinite(spectrum_b.values), spectrum_b.values, 0.0)
    signal = np.maximum(signal, 0.0)
    sigma = np.where(np.isfinite(noise_b.values), noise_b.values, 0.0)
    sigma = np.maximum(sigma, 0.0)
    active_bins = signal > spectral_threshold_sigma * sigma
    signal = np.where(active_bins, signal, 0.0)

    base_dims = spectrum.dims[:-1]
    base_shape = signal.shape[:-1]
    base_coords = {d: spectrum.coords[d] for d in base_dims if d in spectrum.coords}
    rng = np.random.default_rng(random_seed)

    template = spectrum.isel({velocity_dim: 0}, drop=True)
    w_input = (
        vertical_air_velocity
        if isinstance(vertical_air_velocity, xr.DataArray)
        else xr.DataArray(vertical_air_velocity)
    )
    w_std_input = (
        vertical_air_velocity_uncertainty
        if isinstance(vertical_air_velocity_uncertainty, xr.DataArray)
        else xr.DataArray(vertical_air_velocity_uncertainty)
    )
    w_mean_da, _ = xr.broadcast(w_input, template)
    w_std_da, _ = xr.broadcast(w_std_input, template)
    w_mean = np.asarray(w_mean_da.transpose(*base_dims).values, dtype=float)
    w_std = np.asarray(w_std_da.transpose(*base_dims).values, dtype=float)
    if np.any(w_std < 0):
        raise ValueError("vertical_air_velocity_uncertainty cannot be negative")

    nominal = _moments(signal[None, ...], v, dv, reflectivity_scale)
    nominal = [x[0] for x in nominal]

    # Store realizations one moment at a time; chunking limits temporary spectra.
    mc = [np.full((n_realizations,) + base_shape, np.nan) for _ in MOMENT_NAMES]
    rain_mc = (
        np.full((n_realizations,) + base_shape, np.nan)
        if calculate_rain_rate
        else None
    )
    rain_nominal = (
        _rain_rate_from_spectrum(signal[None, ...], v, dv, w_mean[None, ...])[0]
        if calculate_rain_rate
        else None
    )
    integrated_noise = np.sum(sigma * dv, axis=-1)

    for start in range(0, n_realizations, chunk_size):
        stop = min(start + chunk_size, n_realizations)
        perturbed = signal[None, ...] + rng.normal(
            loc=0.0, scale=sigma[None, ...], size=(stop - start,) + signal.shape
        )
        perturbed = np.where(active_bins[None, ...], np.maximum(perturbed, 0.0), 0.0)

        if min_valid_bins > 0:
            enough = np.sum(perturbed > 0, axis=-1) >= min_valid_bins
            perturbed = np.where(enough[..., None], perturbed, 0.0)

        vals = _moments(perturbed, v, dv, reflectivity_scale)
        if min_snr_linear is not None:
            integrated_signal = vals[0] / reflectivity_scale
            snr_ok = integrated_signal >= min_snr_linear * integrated_noise
            vals = [np.where(snr_ok, x, np.nan) for x in vals]
        for target, value in zip(mc, vals):
            target[start:stop] = value

        if calculate_rain_rate:
            w_sample = w_mean[None, ...] + rng.normal(
                0.0, w_std[None, ...], size=(stop - start,) + base_shape
            )
            rr = _rain_rate_from_spectrum(perturbed, v, dv, w_sample)
            if rain_rate_relative_model_uncertainty > 0:
                model_factor = rng.normal(
                    1.0,
                    rain_rate_relative_model_uncertainty,
                    size=rr.shape,
                )
                rr = rr * np.maximum(model_factor, 0.0)
            rain_mc[start:stop] = rr

    ds = xr.Dataset(coords=base_coords)
    for name, nominal_value, realizations in zip(MOMENT_NAMES, nominal, mc):
        ds[name] = xr.DataArray(nominal_value, dims=base_dims, coords=base_coords)
        ds[f"{name}_random_uncertainty"] = xr.DataArray(
            np.nanstd(realizations, axis=0, ddof=1), dims=base_dims, coords=base_coords
        )
        ds[f"{name}_p16"] = xr.DataArray(
            np.nanpercentile(realizations, 16, axis=0), dims=base_dims, coords=base_coords
        )
        ds[f"{name}_p84"] = xr.DataArray(
            np.nanpercentile(realizations, 84, axis=0), dims=base_dims, coords=base_coords
        )

    # Convert every MC reflectivity realization to dBZ before calculating its
    # uncertainty; this is safer near the detection limit than linearisation.
    ze_mc = mc[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        ze_db_mc = 10.0 * np.log10(ze_mc)
        ze_db = 10.0 * np.log10(ds["ze"])
    ds["ze_db"] = ze_db
    ds["ze_db_random_uncertainty"] = xr.DataArray(
        np.nanstd(ze_db_mc, axis=0, ddof=1), dims=base_dims, coords=base_coords
    )
    ds["ze_db_p16"] = xr.DataArray(
        np.nanpercentile(ze_db_mc, 16, axis=0), dims=base_dims, coords=base_coords
    )
    ds["ze_db_p84"] = xr.DataArray(
        np.nanpercentile(ze_db_mc, 84, axis=0), dims=base_dims, coords=base_coords
    )

    systematic_sq = calibration_uncertainty_db**2
    if attenuation_uncertainty_db is not None:
        attenuation_uncertainty_db, _ = xr.broadcast(attenuation_uncertainty_db, ds["ze_db"])
        ds["ze_db_attenuation_uncertainty"] = attenuation_uncertainty_db
        systematic_sq = systematic_sq + attenuation_uncertainty_db**2
    ds["ze_db_total_uncertainty"] = np.sqrt(
        ds["ze_db_random_uncertainty"] ** 2 + systematic_sq
    )

    if calculate_rain_rate:
        ds["rain_rate"] = xr.DataArray(
            rain_nominal, dims=base_dims, coords=base_coords
        )
        ds["rain_rate_random_uncertainty"] = xr.DataArray(
            np.nanstd(rain_mc, axis=0, ddof=1), dims=base_dims, coords=base_coords
        )
        ds["rain_rate_p16"] = xr.DataArray(
            np.nanpercentile(rain_mc, 16, axis=0), dims=base_dims, coords=base_coords
        )
        ds["rain_rate_p50"] = xr.DataArray(
            np.nanpercentile(rain_mc, 50, axis=0), dims=base_dims, coords=base_coords
        )
        ds["rain_rate_p84"] = xr.DataArray(
            np.nanpercentile(rain_mc, 84, axis=0), dims=base_dims, coords=base_coords
        )
        ds["rain_rate_relative_uncertainty"] = (
            ds["rain_rate_random_uncertainty"] / ds["rain_rate"]
        )
        db_to_fraction = np.log(10.0) / 10.0
        ds["rain_rate_calibration_uncertainty"] = (
            ds["rain_rate"] * db_to_fraction * calibration_uncertainty_db
        )
        rain_systematic_sq = ds["rain_rate_calibration_uncertainty"] ** 2
        if attenuation_uncertainty_db is not None:
            ds["rain_rate_attenuation_uncertainty"] = (
                ds["rain_rate"]
                * db_to_fraction
                * ds["ze_db_attenuation_uncertainty"]
            )
            rain_systematic_sq = (
                rain_systematic_sq
                + ds["rain_rate_attenuation_uncertainty"] ** 2
            )
        ds["rain_rate_total_uncertainty"] = np.sqrt(
            ds["rain_rate_random_uncertainty"] ** 2 + rain_systematic_sq
        )
        ds["rain_rate_total_relative_uncertainty"] = (
            ds["rain_rate_total_uncertainty"] / ds["rain_rate"]
        )
        for name in (
            "rain_rate",
            "rain_rate_random_uncertainty",
            "rain_rate_calibration_uncertainty",
            "rain_rate_total_uncertainty",
            "rain_rate_p16",
            "rain_rate_p50",
            "rain_rate_p84",
        ):
            ds[name].attrs["units"] = "mm h-1"
        if "rain_rate_attenuation_uncertainty" in ds:
            ds["rain_rate_attenuation_uncertainty"].attrs["units"] = "mm h-1"
        ds["rain_rate_relative_uncertainty"].attrs["units"] = "1"
        ds["rain_rate_total_relative_uncertainty"].attrs["units"] = "1"

    ds["ze"].attrs.update(units="mm6 m-3", long_name="equivalent reflectivity factor")
    ds["ze_db"].attrs.update(units="dBZ", long_name="equivalent reflectivity factor")
    ds["mean_velocity"].attrs["units"] = str(velocity.attrs.get("units", "m s-1"))
    ds["spectral_width"].attrs["units"] = str(velocity.attrs.get("units", "m s-1"))
    ds.attrs.update(
        uncertainty_method="Monte Carlo Gaussian spectral perturbation",
        n_realizations=n_realizations,
        calibration_uncertainty_db=calibration_uncertainty_db,
        rain_rate_method=(
            "spectral DSD; Atlas terminal-velocity relation"
            if calculate_rain_rate
            else "not calculated"
        ),
        random_seed="None" if random_seed is None else random_seed,
    )
    return ds


if __name__ == "__main__":
    # Minimal synthetic example.
    velocity = xr.DataArray(
        np.linspace(-2, 8, 192),
        dims="velocity",
        coords={"velocity": np.linspace(-2, 8, 192)},
        attrs={"units": "m s-1"},
    )
    peak = np.exp(-0.5 * ((velocity - 3.0) / 0.5) ** 2)
    eta = peak.expand_dims(time=[0, 1], height=[100, 200, 300]).transpose(
        "time", "height", "velocity"
    )
    noise = xr.full_like(eta.isel(velocity=0, drop=True), 0.01)
    result = mrr_moment_uncertainty(
        eta,
        noise_std=noise,
        n_realizations=200,
        calibration_uncertainty_db=1.0,
        calculate_rain_rate=True,
        vertical_air_velocity=0.0,
        vertical_air_velocity_uncertainty=0.2,
        rain_rate_relative_model_uncertainty=0.10,
        random_seed=42,
    )
    print(result)

