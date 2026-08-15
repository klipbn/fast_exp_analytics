from __future__ import annotations

import warnings

import numpy as np
from scipy import stats
from statsmodels.stats.proportion import confint_proportions_2indep


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


def ratio_difference_ci(
    control_num,
    control_den,
    experiment_num,
    experiment_den,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Delta-method CI for the difference of two ratios of sums."""
    control_num = np.asarray(control_num, dtype=float)
    control_den = np.asarray(control_den, dtype=float)
    experiment_num = np.asarray(experiment_num, dtype=float)
    experiment_den = np.asarray(experiment_den, dtype=float)

    control_mask = np.isfinite(control_num) & np.isfinite(control_den)
    experiment_mask = np.isfinite(experiment_num) & np.isfinite(experiment_den)
    control_num, control_den = control_num[control_mask], control_den[control_mask]
    experiment_num, experiment_den = experiment_num[experiment_mask], experiment_den[experiment_mask]
    if len(control_num) < 2 or len(experiment_num) < 2:
        return np.nan, np.nan

    control_den_mean = float(control_den.mean())
    experiment_den_mean = float(experiment_den.mean())
    if control_den_mean == 0 or experiment_den_mean == 0:
        return np.nan, np.nan

    control_ratio = float(control_num.mean() / control_den_mean)
    experiment_ratio = float(experiment_num.mean() / experiment_den_mean)
    control_influence = control_num - control_ratio * control_den
    experiment_influence = experiment_num - experiment_ratio * experiment_den
    variance_control = float(control_influence.var(ddof=1) / (len(control_num) * control_den_mean**2))
    variance_experiment = float(
        experiment_influence.var(ddof=1) / (len(experiment_num) * experiment_den_mean**2)
    )
    delta = experiment_ratio - control_ratio
    se = float(np.sqrt(variance_control + variance_experiment))
    if se == 0:
        return delta, delta

    df = _welch_df(
        variance_control, len(control_num), variance_experiment, len(experiment_num)
    )
    if not np.isfinite(df) or df <= 0:
        return np.nan, np.nan
    margin = float(stats.t.ppf(1 - alpha / 2, df) * se)
    return delta - margin, delta + margin


def proportion_difference_ci(
    control_success: int,
    control_observations: int,
    experiment_success: int,
    experiment_observations: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Newcombe-Wilson CI for ``p(experiment) - p(control)``."""
    if control_observations <= 0 or experiment_observations <= 0:
        return np.nan, np.nan
    lower, upper = confint_proportions_2indep(
        experiment_success,
        experiment_observations,
        control_success,
        control_observations,
        compare="diff",
        method="newcomb",
        alpha=alpha,
    )
    return float(lower), float(upper)


def median_difference_ci(
    control,
    experiment,
    alpha: float = 0.05,
    n_resamples: int = 10_000,
    random_state: int | None = 0,
) -> tuple[float, float]:
    """BCa bootstrap CI for ``median(experiment) - median(control)``."""
    control = _finite_values(control)
    experiment = _finite_values(experiment)
    if len(control) < 2 or len(experiment) < 2 or n_resamples < 1:
        return np.nan, np.nan

    delta = float(np.median(experiment) - np.median(control))
    if np.all(control == control[0]) and np.all(experiment == experiment[0]):
        return delta, delta

    def statistic(control_sample, experiment_sample, axis):
        return np.median(experiment_sample, axis=axis) - np.median(control_sample, axis=axis)

    kwargs = {
        "n_resamples": n_resamples,
        "confidence_level": 1 - alpha,
        "method": "BCa",
        "random_state": random_state,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=RuntimeWarning)
        try:
            result = stats.bootstrap((control, experiment), statistic, **kwargs)
        except (RuntimeWarning, ValueError):
            kwargs["method"] = "percentile"
            result = stats.bootstrap((control, experiment), statistic, **kwargs)
    return float(result.confidence_interval.low), float(result.confidence_interval.high)


def metric_confidence_interval(
    control,
    experiment,
    metric_type: str,
    alpha: float = 0.05,
    bootstrap_resamples: int = 10_000,
    random_state: int | None = 0,
) -> tuple[float, float]:
    """Return a CI for an experiment metric prepared by ``_prep_metric_pair_frames``."""
    if metric_type in {"additive", "average"}:
        return welch_mean_difference_ci(control["value"], experiment["value"], alpha)
    if metric_type == "ratio":
        return ratio_difference_ci(
            control["num"], control["den"], experiment["num"], experiment["den"], alpha
        )
    if metric_type == "share":
        control_valid = (control["den"] > 0) & control["num"].notna()
        experiment_valid = (experiment["den"] > 0) & experiment["num"].notna()
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
    if metric_type == "median":
        return median_difference_ci(
            control["value"],
            experiment["value"],
            alpha,
            bootstrap_resamples,
            random_state,
        )
    return np.nan, np.nan
