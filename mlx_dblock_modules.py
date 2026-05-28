from __future__ import annotations

from statistics import NormalDist

import mlx.core as mx
import numpy as np


_NORMAL = NormalDist()


def get_block_sigmas(
    num_layers: int,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    p_mean: float = -1.2,
    p_std: float = 1.2,
) -> list[float]:
    cdf_min = _NORMAL.cdf((np.log(sigma_min) - p_mean) / p_std)
    cdf_max = _NORMAL.cdf((np.log(sigma_max) - p_mean) / p_std)
    block_sigmas = []
    for i in range(num_layers + 1):
        p = cdf_min + (cdf_max - cdf_min) * (i / num_layers)
        sigma = np.exp(p_mean + p_std * _NORMAL.inv_cdf(p))
        block_sigmas.append(float(sigma))
    return block_sigmas


def get_discrete_sigmas(
    num_steps: int,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    p_mean: float = -1.2,
    p_std: float = 1.2,
    dblock: bool = False,
) -> mx.array:
    if not dblock:
        ramp = np.linspace(0, 1, num_steps, dtype=np.float32)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return mx.array(sigmas, dtype=mx.float32)

    log_sigma_min = np.log(sigma_min)
    log_sigma_max = np.log(sigma_max)
    cdf_min = _NORMAL.cdf((log_sigma_min - p_mean) / p_std)
    cdf_max = _NORMAL.cdf((log_sigma_max - p_mean) / p_std)
    cdf_points = np.linspace(cdf_min, cdf_max, num_steps, dtype=np.float64)
    sigmas = np.exp([p_mean + p_std * _NORMAL.inv_cdf(float(p)) for p in cdf_points])
    sigmas = np.flip(sigmas).astype(np.float32)
    return mx.array(sigmas, dtype=mx.float32)
