"""Тесты диагностики, снижения дисперсии, sequential, байеса и мощности."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from fast_exp_analytics import bayesian, diagnostics, power, sequential
from fast_exp_analytics import variance_reduction as vr
from fast_exp_analytics.stat_tests import welch_ttest

# ----------------------------------------------------------------------------- diagnostics


def test_srm_check_detects_mismatch_and_respects_expected_shares():
    ok = diagnostics.srm_check({"A": 5000, "B": 5050})
    bad = diagnostics.srm_check({"A": 5000, "B": 5600})
    unequal = diagnostics.srm_check({"A": 9000, "B": 1010}, expected_shares={"A": 0.9, "B": 0.1})

    assert not ok["is_srm"] and bad["is_srm"] and not unequal["is_srm"]
    assert ok["p_value"] == pytest.approx(stats.chisquare([5000, 5050]).pvalue)
    with pytest.raises(ValueError):
        diagnostics.srm_check({"A": 10, "B": 10}, expected_shares={"A": 1.0})


def test_srm_check_df_counts_unique_units_and_overlap_check():
    df = pd.DataFrame(
        {"user_id": [1, 1, 2, 3, 4, 4], "exp_group": ["A", "A", "A", "B", "B", "A"]}
    )
    srm = diagnostics.srm_check_df(df)
    overlap = diagnostics.check_unit_overlap(df)

    assert srm["observed"] == {"A": 3, "B": 2}
    assert overlap["n_units_in_multiple_groups"] == 1
    assert overlap["n_duplicated_rows"] == 1
    assert not overlap["is_ok"]


def test_describe_groups_and_normality_report():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "exp_group": ["A"] * 500 + ["B"] * 500,
            "amount": np.concatenate([np.zeros(300), rng.lognormal(3, 1.5, 700)]),
        }
    )
    table = diagnostics.describe_groups(df, "amount")
    report = diagnostics.normality_report(df["amount"])

    assert table.loc["A", "share_zeros"] == pytest.approx(0.6)
    assert 0 < table.loc["B", "top1pct_share"] < 1
    assert report["skewness"] > 1
    assert report["required_n_for_clt"] == np.ceil(355 * report["skewness"] ** 2)
    assert report["shapiro_p_value"] < 0.001


def test_recommend_test_rules():
    rng = np.random.default_rng(1)
    normal = rng.normal(10, 2, 2000)
    heavy = rng.lognormal(0, 2.5, 500)

    assert diagnostics.recommend_test(normal, normal + 0.1)["test"] == "welch"
    assert diagnostics.recommend_test(heavy, heavy * 1.1)["test"] == "bootstrap"
    assert diagnostics.recommend_test(normal[:10], normal[10:20])["test"] == "permutation"
    assert diagnostics.recommend_test([0] * 99 + [1], [0] * 100, "share")["test"] == "fisher"
    assert diagnostics.recommend_test([0, 1] * 500, [0, 1] * 500, "share")["test"] == "ztest"
    assert diagnostics.recommend_test(normal, normal * 3, "median")["test"] == "brunner_munzel"
    assert diagnostics.recommend_test(normal, normal, "ratio")["test"] == "delta"


def test_simulate_aa_controls_false_positives_and_measures_power():
    rng = np.random.default_rng(2)
    values = rng.lognormal(0, 1, 3000)
    aa = diagnostics.simulate_aa(values, "welch", n_simulations=400, random_state=1)
    ab = diagnostics.simulate_aa(values, "welch", n_simulations=200, effect_rel=0.3, random_state=1)
    custom = diagnostics.simulate_aa(
        values, lambda c, e: welch_ttest(c, e).p_value, n_simulations=50, random_state=1
    )

    assert aa["is_fpr_ok"]
    assert 0.02 < aa["rejection_rate"] < 0.09
    assert ab["rejection_rate"] > 0.9
    assert "is_fpr_ok" not in ab
    assert len(custom["p_values"]) == 50


# ----------------------------------------------------------------------------- variance reduction


def test_cuped_reduces_variance_and_keeps_effect_unbiased():
    rng = np.random.default_rng(3)
    pre = rng.normal(100, 20, 4000)
    control = pre[:2000] + rng.normal(0, 5, 2000)
    experiment = pre[2000:] + rng.normal(2.0, 5, 2000)
    adj_c, adj_e, info = vr.cuped_adjust(control, pre[:2000], experiment, pre[2000:])

    assert info["variance_reduction"] > 0.9
    assert info["theta"] == pytest.approx(1.0, abs=0.05)
    assert adj_e.mean() - adj_c.mean() == pytest.approx(2.0, abs=0.5)
    assert welch_ttest(adj_c, adj_e).extra["se"] < 0.4 * welch_ttest(control, experiment).extra["se"]


def test_cuped_missing_covariate_is_left_unadjusted_and_useless_covariate_is_harmless():
    rng = np.random.default_rng(4)
    control, experiment = rng.normal(0, 1, 200), rng.normal(0, 1, 200)
    cov_c, cov_e = rng.normal(0, 1, 200), rng.normal(0, 1, 200)
    cov_c[:50] = np.nan
    adj_c, adj_e, info = vr.cuped_adjust(control, cov_c, experiment, cov_e)

    assert len(adj_c) == 200 and np.isfinite(adj_c).all()
    assert abs(info["variance_reduction"]) < 0.05
    all_nan = vr.cuped_adjust(control, np.full(200, np.nan), experiment, np.full(200, np.nan))
    assert np.array_equal(all_nan[0], control) and all_nan[2]["theta"] == 0.0


def test_add_cuped_column_multi_group():
    rng = np.random.default_rng(5)
    pre = rng.normal(50, 10, 900)
    df = pd.DataFrame(
        {"exp_group": np.repeat(["A", "B", "C"], 300), "pre": pre, "y": pre + rng.normal(0, 2, 900)}
    )
    out = vr.add_cuped_column(df, "y", "pre")

    assert out["y_cuped"].var() < 0.2 * out["y"].var()
    assert out["y_cuped"].mean() == pytest.approx(out["y"].mean(), abs=1e-9)


def test_winsorize_uses_pooled_threshold():
    values = np.arange(1, 101, dtype=float)
    clipped, lower, upper = vr.winsorize_values(values, upper_quantile=0.9, lower_quantile=0.1)
    df = pd.DataFrame({"g": ["A"] * 50 + ["B"] * 50, "x": values})
    out = vr.winsorize_columns(df, ["x"], upper_quantile=0.9)

    assert clipped.max() == upper == pytest.approx(np.quantile(values, 0.9))
    assert clipped.min() == lower
    assert out["x"].max() == pytest.approx(upper)
    assert out.loc[out["g"] == "A", "x"].max() == 50  # порог общий, а не по группе


def test_post_stratification_removes_between_strata_variance():
    rng = np.random.default_rng(6)
    n = 4000
    strata = rng.choice(["small", "big"], n)
    groups = rng.choice(["A", "B"], n)
    values = np.where(strata == "big", 1000.0, 10.0) + rng.normal(0, 5, n) + (groups == "B") * 2.0
    res = vr.post_stratified_difference(values, groups, strata, "A", "B")
    naive = welch_ttest(values[groups == "A"], values[groups == "B"])

    assert res["estimate"] == pytest.approx(2.0, abs=0.6)
    assert res["se"] < 0.1 * naive.extra["se"]
    assert res["n_strata_used"] == 2 and res["p_value"] < 0.001


# ----------------------------------------------------------------------------- sequential


def test_group_sequential_boundaries_match_reference_tables():
    looks = [0.2, 0.4, 0.6, 0.8, 1.0]
    obf = sequential.group_sequential_boundaries(looks, alpha=0.05, spending="obf")
    pocock = sequential.group_sequential_boundaries(looks, alpha=0.05, spending="pocock")

    # Lan-DeMets, двусторонний alpha=0.05, 5 равных просмотров (ldbounds / gsDesign)
    assert obf["z_boundary"].tolist() == pytest.approx([4.877, 3.357, 2.680, 2.290, 2.031], abs=2e-3)
    assert pocock["z_boundary"].tolist() == pytest.approx(
        [2.438, 2.427, 2.410, 2.397, 2.386], abs=2e-3
    )
    assert obf["alpha_spent"].iloc[-1] == pytest.approx(0.05)
    single = sequential.group_sequential_boundaries([1.0])
    assert single["z_boundary"].iloc[0] == pytest.approx(stats.norm.ppf(0.975))
    with pytest.raises(ValueError):
        sequential.group_sequential_boundaries([0.5, 0.4, 1.0])


def test_group_sequential_test_marks_first_crossing():
    res = sequential.group_sequential_test([1.0, 2.9, 3.5], [0.25, 0.5, 0.75, 1.0])

    assert res["reject"].tolist() == [False, False, True]
    assert res["stop"].tolist() == [False, False, True]


def test_msprt_controls_type_one_error_under_continuous_peeking():
    rng = np.random.default_rng(7)
    looks = np.arange(50, 3001, 50)
    n_sim, rejected, naive = 600, 0, 0
    for _ in range(n_sim):
        x = rng.normal(0, 1, (2, 3000))
        means = np.cumsum(x, axis=1)[:, looks - 1] / looks
        delta, se = means[1] - means[0], np.sqrt(2 / looks)
        rejected += sequential.msprt_sequence(delta, se, tau=0.1)["reject"].iloc[-1]
        naive += bool((np.abs(delta / se) > 1.96).any())

    assert rejected / n_sim <= 0.05
    assert naive / n_sim > 0.2


def test_msprt_detects_real_effect_and_ci_is_consistent_with_p_value():
    looks = np.arange(100, 5001, 100)
    se = np.sqrt(2 / looks)
    seq = sequential.msprt_sequence(np.full(len(looks), 0.2), se, tau=0.2)

    assert seq["stop"].sum() == 1
    stop = seq[seq["stop"]].iloc[0]
    assert stop["ci_lower"] > 0
    before = seq[seq["look"] < stop["look"]]
    assert (before["ci_lower"] <= 0).all()
    low, high = sequential.always_valid_ci(0.2, 0.05, tau=0.2)
    fixed = stats.norm.ppf(0.975) * 0.05
    assert high - low > 2 * fixed  # плата за право подглядывать
    assert sequential.always_valid_p_value(0.0, 0.05, tau=0.2) == 1.0


# ----------------------------------------------------------------------------- bayes и мощность


def test_bayes_proportions_and_means():
    res = bayesian.bayes_proportions(100, 1000, 130, 1000)
    same = bayesian.bayes_proportions(100, 1000, 100, 1000)
    rng = np.random.default_rng(8)
    means = bayesian.bayes_means(rng.normal(0, 1, 2000), rng.normal(0.2, 1, 2000))

    assert res["prob_experiment_better"] > 0.95
    assert res["expected_loss_experiment"] < res["expected_loss_control"]
    assert same["prob_experiment_better"] == pytest.approx(0.5, abs=0.02)
    assert res["diff_ci_lower"] < 0.03 < res["diff_ci_upper"]
    assert means["prob_experiment_better"] > 0.99
    assert means["expected_loss_experiment"] < means["expected_loss_control"]


def test_sample_size_mde_and_power_are_mutually_consistent():
    plan = power.sample_size_means(std=10, mde_abs=1.0)
    n = plan["n_control"]

    assert n == 1570
    assert power.mde_means(10, n, n) == pytest.approx(1.0, abs=1e-3)
    assert power.power_means(1.0, 10, n, n) == pytest.approx(0.8, abs=1e-3)

    prop = power.sample_size_proportions(0.1, mde_rel=0.1)
    assert prop["n_control"] == 14751
    assert power.power_proportions(0.1, 0.11, 14751, 14751) == pytest.approx(0.8, abs=2e-3)


def test_sample_size_options():
    base = power.sample_size_means(10, 1.0)["n_total"]

    assert power.sample_size_means(10, 1.0, ratio=3.0)["n_total"] > base
    assert power.sample_size_means(10, 1.0, n_comparisons=3)["n_total"] > base
    assert power.sample_size_means(10, 1.0, alternative="one-sided")["n_total"] < base
    assert power.cuped_sample_size_factor(0.7) == pytest.approx(0.51)
    assert power.optimal_control_share(1) == pytest.approx(0.5)
    assert power.optimal_control_share(4) == pytest.approx(1 / 3)
    with pytest.raises(ValueError):
        power.sample_size_proportions(0.1, mde_abs=0.01, mde_rel=0.1)


def test_ratio_metric_std_feeds_sample_size():
    rng = np.random.default_rng(9)
    den = rng.poisson(100, 5000) + 1.0
    num = rng.binomial(den.astype(int), 0.05).astype(float)
    ratio, std = power.ratio_metric_std(num, den)

    assert ratio == pytest.approx(0.05, abs=0.002)
    assert std == pytest.approx(np.sqrt(0.05 * 0.95 / 100), rel=0.1)
