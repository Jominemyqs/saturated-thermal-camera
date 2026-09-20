from __future__ import annotations

import numpy as np


def gelman_rubin(draw_chains: np.ndarray) -> np.ndarray:
    """Classical per-target R-hat for equally sized independent chains."""
    values = np.asarray(draw_chains, dtype=float)
    if values.ndim != 3 or values.shape[0] < 2 or values.shape[1] < 2:
        raise ValueError("R-hat requires shape (chain, draw, target)")
    n_draws = values.shape[1]
    within = np.mean(np.var(values, axis=1, ddof=1), axis=0)
    between = n_draws * np.var(np.mean(values, axis=1), axis=0, ddof=1)
    variance = ((n_draws - 1.0) / n_draws) * within + between / n_draws
    return np.sqrt(
        np.divide(
            variance,
            within,
            out=np.full_like(variance, np.nan),
            where=within > 1e-14,
        )
    )


def chain_diagnostics(
    draw_chains: np.ndarray,
    *,
    selected_mask: np.ndarray | None = None,
) -> dict[str, float]:
    values = np.asarray(draw_chains, dtype=float)
    pooled_mean = np.mean(values, axis=(0, 1))
    chain_means = np.mean(values, axis=1)
    chain_rmse = np.sqrt(np.mean((chain_means - pooled_mean[None, :]) ** 2, axis=1))
    rhat = gelman_rubin(values)
    output = {
        "chain_mean_max_rmse_from_pooled_K": float(np.max(chain_rmse)),
        "rhat_median": float(np.nanmedian(rhat)),
        "rhat_max": float(np.nanmax(rhat)),
        "rhat_over_1_05_fraction": float(np.nanmean(rhat > 1.05)),
    }
    if selected_mask is not None:
        selected = np.asarray(selected_mask, dtype=bool).reshape(-1)
        if len(selected) != values.shape[2] or not np.any(selected):
            raise ValueError("Selected mask must choose targets from the chain")
        selected_rhat = rhat[selected]
        output.update(
            {
                "selected_rhat_median": float(np.nanmedian(selected_rhat)),
                "selected_rhat_max": float(np.nanmax(selected_rhat)),
                "selected_rhat_over_1_05_fraction": float(
                    np.nanmean(selected_rhat > 1.05)
                ),
            }
        )
    return output
