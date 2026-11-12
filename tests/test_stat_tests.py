import numpy as np
import pytest
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

from fast_exp_analytics import effect_sizes as es
from fast_exp_analytics import stat_tests as st
from fast_exp_analytics.confidence_intervals import ratio_difference_ci
from fast_exp_analytics.multiple_testing import adjust_pvalues


@pytest.fixture()
def samples():
    rng = np.random.default_rng(1)
    return rng.exponential(10, 400), rng.exponential(11.5, 380)


def test_welch_matches_scipy_and_orders_effect_as_experiment_minus_control(samples):
    control, experiment = samples
    res = st.welch_ttest(control, experiment)
    ref = stats.ttest_ind(experiment, control, equal_var=False)

    assert res.p_value == pytest.approx(ref.pvalue)
    assert res.statistic == pytest.approx(ref.statistic)
    assert res.estimate == pytest.approx(experiment.mean() - control.mean())
    assert res.ci_lower < res.estimate < res.ci_upper


def test_student_and_one_sided_alternatives_match_scipy(samples):
    control, experiment = samples
    assert st.student_ttest(control, experiment).p_value == pytest.approx(
        stats.ttest_ind(experiment, control).pvalue
    )
    greater = st.welch_ttest(control, experiment, alternative="greater").p_value
    less = st.welch_ttest(control, experiment, alternative="less").p_value
    assert greater == pytest.approx(
        stats.ttest_ind(experiment, control, equal_var=False, alternative="greater").pvalue
    )
    assert greater + less == pytest.approx(1.0)


def test_yuen_matches_scipy_trimmed_ttest(samples):
    control, experiment = samples
    res = st.yuen_trimmed_ttest(control, experiment, trim=0.1)
    ref = stats.ttest_ind(experiment, control, equal_var=False, trim=0.1)

    assert res.p_value == pytest.approx(ref.pvalue)
    assert res.statistic == pytest.approx(ref.statistic)
    assert res.estimand == "trimmed_mean_diff"


def test_rank_tests_match_scipy(samples):
    control, experiment = samples
    mw = st.mann_whitney_test(control, experiment)
    bm = st.brunner_munzel_test(control, experiment)

    assert mw.p_value == pytest.approx(stats.mannwhitneyu(control, experiment).pvalue)
    assert bm.p_value == pytest.approx(stats.brunnermunzel(control, experiment).pvalue)
    assert mw.estimate == pytest.approx(bm.estimate)
    assert mw.estimate == pytest.approx(es.vargha_delaney_a(control, experiment))
    assert bm.ci_lower < bm.estimate < bm.ci_upper
    assert mw.extra["rank_biserial"] == pytest.approx(es.cliffs_delta(control, experiment))


def test_bootstrap_and_permutation_agree_with_welch_on_large_samples(samples):
    control, experiment = samples
    welch = st.welch_ttest(control, experiment)
    boot = st.bootstrap_test(control, experiment, n_resamples=4000, random_state=3)
    perm = st.permutation_test(control, experiment, n_resamples=4000, random_state=3)

    assert boot.ci_lower == pytest.approx(welch.ci_lower, abs=0.25)
    assert boot.ci_upper == pytest.approx(welch.ci_upper, abs=0.25)
    assert boot.p_value == pytest.approx(welch.p_value, abs=0.02)
    assert perm.p_value == pytest.approx(welch.p_value, abs=0.02)


def test_resampling_tests_are_reproducible_and_support_median_and_quantile(samples):
    control, experiment = samples
    first = st.bootstrap_test(control, experiment, statistic="median", n_resamples=500, random_state=7)
    second = st.bootstrap_test(control, experiment, statistic="median", n_resamples=500, random_state=7)
    p90 = st.quantile_test(control, experiment, q=0.9, n_resamples=500)

    assert first == second
    assert first.estimand == "median_diff"
    assert p90.estimate == pytest.approx(np.quantile(experiment, 0.9) - np.quantile(control, 0.9))


def test_permutation_p_value_never_zero():
    res = st.permutation_test(np.zeros(50), np.ones(50) * 100, n_resamples=200)
    assert res.p_value == pytest.approx(1 / 201)


def test_proportion_ztest_matches_statsmodels_and_methods_are_consistent():
    z = st.proportion_test(30, 200, 45, 210, method="ztest")
    ref_stat, ref_p = proportions_ztest([45, 30], [210, 200])

    assert z.p_value == pytest.approx(ref_p)
    assert z.statistic == pytest.approx(ref_stat)
    assert st.proportion_test(30, 200, 45, 210, method="chi2").p_value == pytest.approx(ref_p)
    assert st.proportion_test(30, 200, 45, 210, method="fisher").p_value == pytest.approx(
        stats.fisher_exact([[45, 165], [30, 170]])[1]
    )
    assert z.estimate == pytest.approx(45 / 210 - 30 / 200)
    assert z.extra["risk_ratio"] == pytest.approx((45 / 210) / (30 / 200))
    assert z.ci_lower < z.estimate < z.ci_upper


def test_proportion_test_edge_cases():
    assert np.isnan(st.proportion_test(0, 100, 0, 100).p_value)
    with pytest.raises(ValueError):
        st.proportion_test(10, 5_000, 12, 5_000, method="boschloo")
    with pytest.raises(ValueError):
        st.proportion_test(10, 100, 12, 100, method="chi2", alternative="greater")


def test_ratio_delta_test_ci_matches_existing_delta_method():
    rng = np.random.default_rng(4)
    c_den, e_den = rng.poisson(100, 600) + 1.0, rng.poisson(100, 600) + 1.0
    c_num, e_num = rng.binomial(c_den.astype(int), 0.05), rng.binomial(e_den.astype(int), 0.056)
    res = st.ratio_delta_test(c_num, c_den, e_num, e_den)
    lin = st.ratio_linearization_test(c_num, c_den, e_num, e_den)
    boot = st.ratio_bootstrap_test(c_num, c_den, e_num, e_den, n_resamples=3000)

    assert (res.ci_lower, res.ci_upper) == pytest.approx(
        ratio_difference_ci(c_num, c_den, e_num, e_den)
    )
    assert res.estimate == pytest.approx(e_num.sum() / e_den.sum() - c_num.sum() / c_den.sum())
    assert lin.p_value == pytest.approx(res.p_value, rel=0.2)
    assert boot.ci_lower == pytest.approx(res.ci_lower, abs=5e-4)
    assert boot.ci_upper == pytest.approx(res.ci_upper, abs=5e-4)


def test_paired_tests():
    rng = np.random.default_rng(2)
    before = rng.normal(10, 2, 80)
    after = before + rng.normal(0.5, 1, 80)

    assert st.paired_ttest(before, after).p_value == pytest.approx(
        stats.ttest_rel(after, before).pvalue
    )
    assert st.wilcoxon_signed_rank_test(before, after).p_value == pytest.approx(
        stats.wilcoxon(after - before).pvalue
    )
    assert st.sign_test(before, after).p_value < 0.05
    assert st.mcnemar_test(5, 19).p_value == pytest.approx(stats.binomtest(19, 24, 0.5).pvalue)
    assert st.mcnemar_test(50, 80).p_value == pytest.approx(stats.chi2.sf(29**2 / 130, 1))


def test_tost_and_noninferiority():
    rng = np.random.default_rng(0)
    control, same = rng.normal(100, 10, 5000), rng.normal(100, 10, 5000)
    worse = rng.normal(95, 10, 5000)

    assert st.tost_equivalence_test(control, same, -1.0, 1.0).p_value < 0.05
    assert st.tost_equivalence_test(control, worse, -1.0, 1.0).p_value > 0.5
    assert st.noninferiority_test(control, same, margin=1.0).p_value < 0.05
    assert st.noninferiority_test(control, worse, margin=1.0).p_value > 0.5


def test_degenerate_inputs_return_nan_not_errors():
    for fn in (st.welch_ttest, st.student_ttest, st.yuen_trimmed_ttest, st.brunner_munzel_test):
        assert np.isnan(fn([1.0], [2.0, 3.0]).p_value)
    assert np.isnan(st.welch_ttest([5, 5, 5], [5, 5, 5]).p_value)
    assert np.isnan(st.mann_whitney_test([], [1, 2]).p_value)


def test_available_tests_and_dispatcher_validation():
    assert "welch" in st.available_tests("average")
    assert "dunn" not in st.available_tests("average")
    assert "dunn" in st.available_tests("median", multi_group=True)
    assert set(st.available_tests()) == {"additive", "average", "ratio", "share", "median"}
    with pytest.raises(ValueError):
        st.available_tests("unknown")


def test_effect_sizes_and_relative_lift_ci():
    rng = np.random.default_rng(5)
    control, experiment = rng.normal(0, 1, 5000), rng.normal(0.5, 1, 5000)

    assert es.cohens_d(control, experiment) == pytest.approx(0.5, abs=0.06)
    assert es.hedges_g(control, experiment) < es.cohens_d(control, experiment)
    assert es.cohens_h(0.5, 0.5) == 0
    assert es.cliffs_delta([1, 2, 3], [4, 5, 6]) == 1.0

    delta = es.relative_lift_ci(10, 0.04, 11, 0.04, method="delta")
    fieller = es.relative_lift_ci(10, 0.04, 11, 0.04, method="fieller")
    assert delta[0] < 0.1 < delta[1]
    assert fieller == pytest.approx(delta, abs=0.01)
    assert np.isnan(es.relative_lift_ci(0.1, 1.0, 1.0, 1.0, method="fieller")).all()


def test_adjust_pvalues_methods_and_nan_handling():
    p = np.array([0.01, np.nan, 0.04, 0.03])
    holm, reject = adjust_pvalues(p, "holm")
    bh, _ = adjust_pvalues(p, "bh")
    raw, _ = adjust_pvalues(p, None)

    assert holm[0] == pytest.approx(0.03)
    assert np.isnan(holm[1]) and not reject[1]
    assert (bh[[0, 2, 3]] <= holm[[0, 2, 3]] + 1e-12).all()
    assert raw[0] == 0.01
    with pytest.raises(ValueError):
        adjust_pvalues(p, "no_such_method")
