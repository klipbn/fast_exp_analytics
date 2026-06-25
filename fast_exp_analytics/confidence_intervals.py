from __future__ import annotations

import numpy as np
from scipy import stats


def _finite_values(values) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def _welch_df(variance_control, n_control, variance_experiment, n_experiment) -> float:
    numerator = (variance_control + variance_experiment) ** 2
    denominator = (
        variance_control**2 / (n_control - 1) + variance_experiment**2 / (n_experiment - 1)
    )
    if denominator <= 0:
        return np.nan
    return float(numerator / denominator)


def _mean_difference_ci(control, experiment, alpha: float) -> tuple[float, float]:
    control = _finite_values(control)
    experiment = _finite_values(experiment)
    if len(control) < 2 or len(experiment) < 2:
        return np.nan, np.nan

    delta = float(experiment.mean() - control.mean())
    variance_control = float(control.var(ddof=1) / len(control))
    variance_experiment = float(experiment.var(ddof=1) / len(experiment))
    se = float(np.sqrt(variance_control + variance_experiment))
    if se == 0:
        return delta, delta

    df = _welch_df(variance_control, len(control), variance_experiment, len(experiment))
    if not np.isfinite(df) or df <= 0:
        return np.nan, np.nan
    margin = float(stats.t.ppf(1 - alpha / 2, df) * se)
    return delta - margin, delta + margin


def welch_mean_difference_ci(control, experiment, alpha: float = 0.05) -> tuple[float, float]:
    """Two-sided Welch confidence interval for ``mean(experiment) - mean(control)``."""
    return _mean_difference_ci(control, experiment, alpha)


def proportion_difference_ci(
    control_success: int,
    control_observations: int,
    experiment_success: int,
    experiment_observations: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Wald z-interval for ``p(experiment) - p(control)``."""
    if control_observations <= 0 or experiment_observations <= 0:
        return np.nan, np.nan

    p_control = control_success / control_observations
    p_experiment = experiment_success / experiment_observations
    delta = float(p_experiment - p_control)
    se = float(
        np.sqrt(
            p_control * (1 - p_control) / control_observations
            + p_experiment * (1 - p_experiment) / experiment_observations
        )
    )
    if se == 0:
        return delta, delta

    margin = float(stats.norm.ppf(1 - alpha / 2) * se)
    return delta - margin, delta + margin


def metric_confidence_interval(
    control,
    experiment,
    metric_type: str,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Return a CI for an experiment metric prepared by ``_prep_metric_pair_frames``."""
    if metric_type in {"additive", "average"}:
        return welch_mean_difference_ci(control["value"], experiment["value"], alpha)

    if metric_type == "ratio":
        # Welch interval over linearized values, scaled back to ratio units
        # via the control group denominator mean.
        den_mean = _finite_values(control["den"])
        if len(den_mean) == 0:
            return np.nan, np.nan
        den_mean = float(np.mean(den_mean))
        if den_mean == 0:
            return np.nan, np.nan
        lower, upper = welch_mean_difference_ci(control["value"], experiment["value"], alpha)
        if not (np.isfinite(lower) and np.isfinite(upper)):
            return np.nan, np.nan
        return float(lower / den_mean), float(upper / den_mean)

    if metric_type == "share":
        control_valid = control["den"] > 0
        experiment_valid = experiment["den"] > 0
        control_observations = int(control_valid.sum())
        experiment_observations = int(experiment_valid.sum())
        control_success = int((control_valid & (control["num"] > 0)).sum())
        experiment_success = int((experiment_valid & (experiment["num"] > 0)).sum())
        return proportion_difference_ci(
            control_success,
            control_observations,
            experiment_success,
            experiment_observations,
            alpha,
        )

    # median and unknown types: no CI yet
    return np.nan, np.nan
