import numpy as np
import pytest
from scipy import stats

from fast_exp_analytics import multi_group as mg


@pytest.fixture()
def groups():
    rng = np.random.default_rng(3)
    return {
        "A": rng.exponential(10, 300),
        "B": rng.exponential(11, 280),
        "C": rng.exponential(14, 320),
    }


def test_omnibus_tests_match_scipy(groups):
    arrays = list(groups.values())

    assert mg.anova_test(groups)["p_value"] == pytest.approx(stats.f_oneway(*arrays).pvalue)
    assert mg.kruskal_wallis_test(groups)["p_value"] == pytest.approx(stats.kruskal(*arrays).pvalue)
    assert mg.welch_anova_test(groups)["p_value"] == pytest.approx(
        stats.alexandergovern(*arrays).pvalue, rel=0.5
    )
    assert 0 <= mg.anova_test(groups)["effect_size"] <= 1
    assert mg.omnibus_test(groups, "kruskal")["test"] == "kruskal"


def test_welch_anova_for_two_groups_equals_welch_ttest(groups):
    two = {k: groups[k] for k in ("A", "B")}
    ref = stats.ttest_ind(two["A"], two["B"], equal_var=False).pvalue

    assert mg.welch_anova_test(two)["p_value"] == pytest.approx(ref)


def test_chi2_homogeneity_matches_scipy():
    res = mg.chi2_homogeneity_test({"A": 30, "B": 45, "C": 38}, {"A": 200, "B": 210, "C": 190})
    ref = stats.chi2_contingency([[30, 45, 38], [170, 165, 152]], correction=False)

    assert res["p_value"] == pytest.approx(ref[1])
    assert res["df"] == 2


def test_dunn_z_matches_manual_formula_with_ties():
    groups = {"A": [1, 1, 2, 3, 3, 3], "B": [2, 3, 4, 4, 5, 5], "C": [4, 5, 5, 6, 7, 7]}
    table = mg.dunn_test(groups, p_adjust=None)

    pooled = np.concatenate([groups[k] for k in groups])
    ranks = stats.rankdata(pooled)
    n = len(pooled)
    _, counts = np.unique(pooled, return_counts=True)
    variance = n * (n + 1) / 12 - np.sum(counts**3 - counts) / (12 * (n - 1))
    z_ab = (ranks[6:12].mean() - ranks[:6].mean()) / np.sqrt(variance * (1 / 6 + 1 / 6))

    row = table[(table["group_base"] == "A") & (table["group_exp"] == "B")].iloc[0]
    assert row["statistic"] == pytest.approx(z_ab)
    assert row["p_value"] == pytest.approx(2 * stats.norm.sf(abs(z_ab)))
    assert (table["p_value_adj"] == table["p_value"]).all()


def test_dunn_holm_adjustment_and_preplanned_pairs(groups):
    full = mg.dunn_test(groups, p_adjust="holm")
    control_only = mg.dunn_test(groups, p_adjust="holm", pairs=[("A", "B"), ("A", "C")])

    assert len(full) == 3 and len(control_only) == 2
    assert (full["p_value_adj"] >= full["p_value"] - 1e-12).all()
    ac_full = full[(full["group_exp"] == "C") & (full["group_base"] == "A")].iloc[0]
    ac_ctrl = control_only[control_only["group_exp"] == "C"].iloc[0]
    assert ac_ctrl["p_value"] == pytest.approx(ac_full["p_value"])
    assert ac_ctrl["p_value_adj"] <= ac_full["p_value_adj"] + 1e-12
    with pytest.raises(ValueError):
        mg.dunn_test(groups, pairs=[("A", "Z")])


def test_tukey_games_howell_dunnett(groups):
    tukey = mg.tukey_hsd_test(groups)
    gh = mg.games_howell_test(groups)
    dunnett = mg.dunnett_test(groups)

    ref = stats.tukey_hsd(*groups.values())
    assert tukey["p_value"].tolist() == pytest.approx(
        [ref.pvalue[0, 1], ref.pvalue[0, 2], ref.pvalue[1, 2]]
    )
    for table in (tukey, gh):
        assert len(table) == 3
        assert (table["ci_lower"] < table["estimate"]).all()
        assert (table["estimate"] < table["ci_upper"]).all()
    assert gh["estimate"].iloc[0] == pytest.approx(groups["B"].mean() - groups["A"].mean())
    assert set(dunnett["group_base"]) == {"A"} and len(dunnett) == 2
    # сравнения только с контролем мощнее всех попарных
    assert dunnett["p_value"].iloc[1] <= tukey["p_value"].iloc[1] + 1e-9


def test_games_howell_two_groups_equals_welch(groups):
    two = {k: groups[k] for k in ("A", "C")}
    ref = stats.ttest_ind(two["A"], two["C"], equal_var=False).pvalue

    assert mg.games_howell_test(two)["p_value"].iloc[0] == pytest.approx(ref, rel=1e-6)


def test_levene_and_friedman(groups):
    assert mg.levene_test(groups)["test"] == "brown_forsythe"
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, 60)
    res = mg.friedman_test({"t1": base, "t2": base + 0.5, "t3": base + 1.0})
    assert res["p_value"] < 0.001
    with pytest.raises(ValueError):
        mg.friedman_test({"t1": base, "t2": base})
