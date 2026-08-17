import numpy as np
import pytest
from scipy import stats
from statsmodels.stats.proportion import confint_proportions_2indep

from fast_exp_analytics.confidence_intervals import (
    median_difference_ci,
    proportion_difference_ci,
    ratio_difference_ci,
    welch_mean_difference_ci,
)


def test_welch_mean_difference_ci_matches_manual_formula():
    control = np.array([1.0, 2.0, 5.0, 8.0])
    experiment = np.array([3.0, 6.0, 7.0, 12.0, 15.0])

    lower, upper = welch_mean_difference_ci(control, experiment, alpha=0.05)

    variance_control = control.var(ddof=1) / len(control)
    variance_experiment = experiment.var(ddof=1) / len(experiment)
    se = np.sqrt(variance_control + variance_experiment)
    df = (variance_control + variance_experiment) ** 2 / (
        variance_control**2 / (len(control) - 1)
        + variance_experiment**2 / (len(experiment) - 1)
    )
    delta = experiment.mean() - control.mean()
    margin = stats.t.ppf(0.975, df) * se

    assert lower == pytest.approx(delta - margin)
    assert upper == pytest.approx(delta + margin)


def test_ratio_difference_ci_uses_each_group_denominator():
    control_num = np.array([8.0, 20.0, 35.0, 40.0])
    control_den = np.array([10.0, 20.0, 40.0, 50.0])
    experiment_num = np.array([12.0, 20.0, 50.0, 100.0])
    experiment_den = np.array([20.0, 20.0, 50.0, 100.0])

    lower, upper = ratio_difference_ci(
        control_num, control_den, experiment_num, experiment_den, alpha=0.05
    )

    ratio_control = control_num.mean() / control_den.mean()
    ratio_experiment = experiment_num.mean() / experiment_den.mean()
    influence_control = control_num - ratio_control * control_den
    influence_experiment = experiment_num - ratio_experiment * experiment_den
    variance_control = influence_control.var(ddof=1) / (
        len(control_num) * control_den.mean() ** 2
    )
    variance_experiment = influence_experiment.var(ddof=1) / (
        len(experiment_num) * experiment_den.mean() ** 2
    )
    se = np.sqrt(variance_control + variance_experiment)
    df = (variance_control + variance_experiment) ** 2 / (
        variance_control**2 / (len(control_num) - 1)
        + variance_experiment**2 / (len(experiment_num) - 1)
    )
    delta = ratio_experiment - ratio_control
    margin = stats.t.ppf(0.975, df) * se

    assert lower == pytest.approx(delta - margin)
    assert upper == pytest.approx(delta + margin)


def test_proportion_difference_ci_uses_newcombe_interval():
    lower, upper = proportion_difference_ci(
        control_success=0,
        control_observations=10,
        experiment_success=8,
        experiment_observations=10,
        alpha=0.05,
    )

    expected_lower, expected_upper = confint_proportions_2indep(
        8, 10, 0, 10, compare="diff", method="newcomb", alpha=0.05
    )

    assert lower == pytest.approx(expected_lower)
    assert upper == pytest.approx(expected_upper)


def test_median_difference_ci_is_reproducible_and_contains_estimate():
    control = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    experiment = np.array([3.0, 4.0, 5.0, 6.0, 7.0])

    first = median_difference_ci(control, experiment, alpha=0.05, n_resamples=1_000, random_state=7)
    second = median_difference_ci(control, experiment, alpha=0.05, n_resamples=1_000, random_state=7)

    assert first == pytest.approx(second)
    assert first[0] <= np.median(experiment) - np.median(control) <= first[1]


def test_constant_samples_return_point_interval():
    assert welch_mean_difference_ci([2, 2], [5, 5]) == pytest.approx((3.0, 3.0))
    assert ratio_difference_ci([2, 2], [4, 4], [5, 5], [10, 10]) == pytest.approx((0.0, 0.0))


def test_reversing_groups_reverses_interval_sign():
    control = np.array([1.0, 2.0, 4.0, 5.0])
    experiment = np.array([3.0, 6.0, 7.0, 9.0])

    lower, upper = welch_mean_difference_ci(control, experiment, alpha=0.05)
    reversed_lower, reversed_upper = welch_mean_difference_ci(experiment, control, alpha=0.05)

    assert reversed_lower == pytest.approx(-upper)
    assert reversed_upper == pytest.approx(-lower)
