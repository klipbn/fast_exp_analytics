"""Интеграция новых критериев, CUPED и винзоризации в run_ab_test / run_abc_test."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from openpyxl import load_workbook
from scipy import stats

from fast_exp_analytics import (
    EXTENDED_COLUMNS,
    export_ab_results_to_excel,
    export_abc_results_to_excel,
    run_ab_test,
    run_abc_test,
    run_multigroup_test,
    run_omnibus_test,
)
from fast_exp_analytics.ab import AGGREGATE_COLUMN_NAMES_AB
from fast_exp_analytics.abc import AGGREGATE_COLUMN_NAMES_ABC
from fast_exp_analytics.multi_group import dunn_test, tukey_hsd_test

DATES = ("2026-01-01", "2026-01-14")
BASE_COLUMNS = ["desc", "type", "num", "den", "direction"]


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    n = 3000
    group = rng.choice(["A", "B", "C"], n)
    pre = rng.lognormal(3, 1, n)
    lift = np.select([group == "B", group == "C"], [1.08, 1.0], 1.0)
    amount = (pre * rng.lognormal(0, 0.3, n)) * lift
    shows = rng.poisson(200, n) + 1
    return pd.DataFrame(
        {
            "user_id": np.arange(n),
            "exp_group": group,
            "amount": amount,
            "amount_pre": pre,
            "shows": shows,
            "clicks": rng.binomial(shows, np.where(group == "B", 0.055, 0.05)),
            "is_paid": rng.binomial(1, np.where(group == "B", 0.24, 0.2), n),
            "is_in_exp": 1,
        }
    )


def _metrics(rows, extra_columns=()):
    columns = BASE_COLUMNS + list(extra_columns)
    return pd.DataFrame([r[1:] for r in rows], index=[r[0] for r in rows], columns=columns)


def test_default_schema_is_unchanged_and_extended_appends_columns(data):
    metrics = _metrics([["amount", "Списания", "average", "amount", "is_in_exp", "positive"]])
    base = run_ab_test(data, metrics, *DATES)
    extended = run_ab_test(data, metrics, *DATES, extended=True)

    assert list(base.columns) == AGGREGATE_COLUMN_NAMES_AB
    assert list(extended.columns) == AGGREGATE_COLUMN_NAMES_AB + EXTENDED_COLUMNS
    pd.testing.assert_frame_equal(base, extended[AGGREGATE_COLUMN_NAMES_AB])
    row = extended.iloc[0]
    a = data.loc[data["exp_group"] == "A", "amount"]
    b = data.loc[data["exp_group"] == "B", "amount"]
    assert row["test_name"] == "welch"
    assert row["statistic"] == pytest.approx(stats.ttest_ind(b, a, equal_var=False).statistic)
    assert row["effect_size_name"] == "cohens_d"
    assert row["rel_ci_lower"] < row["rel_delta"] < row["rel_ci_upper"]


def test_explicit_default_test_name_equals_no_test_column(data):
    base = _metrics([["ctr", "CTR", "ratio", "clicks", "shows", "positive"]])
    named = _metrics(
        [["ctr", "CTR", "ratio", "clicks", "shows", "positive", "linearization"]], ["test"]
    )

    pd.testing.assert_frame_equal(run_ab_test(data, base, *DATES), run_ab_test(data, named, *DATES))


@pytest.mark.parametrize(
    "metric_type,num,den,test",
    [
        ("average", "amount", "is_in_exp", "student"),
        ("average", "amount", "is_in_exp", "yuen"),
        ("average", "amount", "is_in_exp", "mannwhitney"),
        ("average", "amount", "is_in_exp", "brunner_munzel"),
        ("average", "amount", "is_in_exp", "permutation"),
        ("additive", "amount", "amount", "bootstrap"),
        ("median", "amount", "is_in_exp", "bootstrap"),
        ("median", "amount", "is_in_exp", "mood"),
        ("ratio", "clicks", "shows", "delta"),
        ("ratio", "clicks", "shows", "bootstrap"),
        ("share", "is_paid", "is_in_exp", "chi2"),
        ("share", "is_paid", "is_in_exp", "fisher"),
        ("share", "is_paid", "is_in_exp", "g_test"),
    ],
)
def test_every_alternative_test_runs_in_ab_pipeline(data, metric_type, num, den, test):
    metrics = _metrics([["m", "Метрика", metric_type, num, den, "positive", test]], ["test"])
    row = run_ab_test(data, metrics, *DATES, ci_bootstrap_resamples=300, extended=True).iloc[0]

    assert row["test_name"] == test
    assert 0 <= row["p_value"] <= 1
    assert row["ci_lower"] <= row["ci_upper"]
    assert row["result"] in {"positive", "negative", "neutral"}


def test_alternative_test_changes_p_value_but_not_point_estimates(data):
    welch = _metrics([["m", "Списания", "average", "amount", "is_in_exp", "positive"]])
    mw = _metrics(
        [["m", "Списания", "average", "amount", "is_in_exp", "positive", "mannwhitney"]], ["test"]
    )
    r_welch, r_mw = run_ab_test(data, welch, *DATES).iloc[0], run_ab_test(data, mw, *DATES).iloc[0]
    a = data.loc[data["exp_group"] == "A", "amount"]
    b = data.loc[data["exp_group"] == "B", "amount"]

    assert r_mw["p_value"] == pytest.approx(stats.mannwhitneyu(a, b).pvalue)
    assert r_mw["p_value"] != r_welch["p_value"]
    assert r_mw["abs_delta"] == r_welch["abs_delta"]
    # Манн—Уитни не оценивает разницу средних — CI остаётся Welch-интервалом
    assert (r_mw["ci_lower"], r_mw["ci_upper"]) == (r_welch["ci_lower"], r_welch["ci_upper"])


def test_cuped_narrows_ci_and_keeps_raw_point_estimates(data):
    raw = _metrics([["m", "Списания", "average", "amount", "is_in_exp", "positive"]])
    cuped = _metrics(
        [["m", "Списания", "average", "amount", "is_in_exp", "positive", "amount_pre"]],
        ["covariate"],
    )
    r_raw = run_ab_test(data, raw, *DATES).iloc[0]
    r_cuped = run_ab_test(data, cuped, *DATES, extended=True).iloc[0]

    assert r_cuped["value_base"] == r_raw["value_base"]
    assert r_cuped["abs_delta"] == r_raw["abs_delta"]
    assert r_cuped["cuped_variance_reduction"] > 0.8
    assert (r_cuped["ci_upper"] - r_cuped["ci_lower"]) < 0.5 * (r_raw["ci_upper"] - r_raw["ci_lower"])
    assert r_cuped["mde"] < 0.5 * r_raw["mde"]
    assert r_cuped["ci_lower"] < r_cuped["delta_cuped"] < r_cuped["ci_upper"]
    assert r_cuped["p_value"] < r_raw["p_value"]


def test_winsorize_caps_values_with_pooled_threshold(data):
    spiked = data.copy()
    spiked.loc[spiked.index[0], "amount"] = 1e9
    raw = _metrics([["m", "Списания", "average", "amount", "is_in_exp", "positive"]])
    capped = _metrics(
        [["m", "Списания", "average", "amount", "is_in_exp", "positive", 0.99]], ["winsorize"]
    )
    r_raw = run_ab_test(spiked, raw, *DATES).iloc[0]
    r_cap = run_ab_test(spiked, capped, *DATES, extended=True).iloc[0]
    pooled = spiked.loc[spiked["exp_group"].isin(["A", "B"]), "amount"]

    assert r_cap["winsorize_threshold"] == pytest.approx(np.quantile(pooled, 0.99))
    assert max(r_cap["value_base"], r_cap["value_exp"]) < r_cap["winsorize_threshold"]
    assert (r_cap["ci_upper"] - r_cap["ci_lower"]) < 0.01 * (r_raw["ci_upper"] - r_raw["ci_lower"])


def test_metrics_adjust_method_adds_adjusted_columns(data):
    metrics = _metrics(
        [
            ["amount", "Списания", "average", "amount", "is_in_exp", "positive"],
            ["ctr", "CTR", "ratio", "clicks", "shows", "positive"],
            ["paid", "Платящие", "share", "is_paid", "is_in_exp", "negative"],
        ]
    )
    res = run_ab_test(data, metrics, *DATES, metrics_adjust_method="holm")

    assert list(res.columns) == AGGREGATE_COLUMN_NAMES_AB + ["p_value_adj", "result_adj"]
    assert (res["p_value_adj"] >= res["p_value"] - 1e-12).all()
    assert res["p_value_adj"].min() == pytest.approx(res["p_value"].min() * 3)
    paid = res[res["metric_name"] == "Платящие"].iloc[0]
    assert paid["abs_delta"] > 0 and paid["result_adj"] in {"negative", "neutral"}


def test_invalid_options_raise_clear_errors(data):
    with pytest.raises(ValueError, match="not supported for type"):
        run_ab_test(
            data,
            _metrics([["m", "M", "share", "is_paid", "is_in_exp", "positive", "welch"]], ["test"]),
            *DATES,
        )
    with pytest.raises(ValueError, match="3\\+ групп"):
        run_ab_test(
            data,
            _metrics([["m", "M", "average", "amount", "is_in_exp", "positive", "dunn"]], ["test"]),
            *DATES,
        )
    with pytest.raises(ValueError, match="covariate"):
        run_ab_test(
            data,
            _metrics([["m", "M", "ratio", "clicks", "shows", "positive", "amount_pre"]], ["covariate"]),
            *DATES,
        )
    with pytest.raises(ValueError, match="missing covariate column"):
        run_ab_test(
            data,
            _metrics([["m", "M", "average", "amount", "is_in_exp", "positive", "nope"]], ["covariate"]),
            *DATES,
        )
    with pytest.raises(ValueError, match="quantile"):
        run_ab_test(
            data,
            _metrics([["m", "M", "average", "amount", "is_in_exp", "positive", 1.5]], ["winsorize"]),
            *DATES,
        )


def test_empty_option_cells_fall_back_to_defaults(data):
    base = _metrics([["m", "Списания", "average", "amount", "is_in_exp", "positive"]])
    empty = _metrics(
        [["m", "Списания", "average", "amount", "is_in_exp", "positive", None, np.nan, ""]],
        ["test", "covariate", "winsorize"],
    )
    empty["winsorize"] = np.nan

    pd.testing.assert_frame_equal(run_ab_test(data, base, *DATES), run_ab_test(data, empty, *DATES))


# ----------------------------------------------------------------------------- 3+ групп


def test_dunn_in_abc_pipeline_matches_standalone_dunn_with_holm(data):
    metrics = _metrics(
        [["m", "Списания", "median", "amount", "is_in_exp", "positive", "dunn"]], ["test"]
    )
    res = run_abc_test(data, metrics, *DATES, ci_bootstrap_resamples=200, extended=True)
    groups = {g: data.loc[data["exp_group"] == g, "amount"].to_numpy() for g in "ABC"}
    ref = dunn_test(groups, p_adjust="holm")

    assert list(res.columns) == AGGREGATE_COLUMN_NAMES_ABC + EXTENDED_COLUMNS
    assert res["p_value"].tolist() == pytest.approx(ref["p_value"].tolist())
    assert res["p_value_adj"].tolist() == pytest.approx(ref["p_value_adj"].tolist())
    assert (res["test_name"] == "dunn").all()


def test_tukey_in_abc_pipeline_is_not_adjusted_twice(data):
    metrics = _metrics(
        [["m", "Списания", "average", "amount", "is_in_exp", "positive", "tukey"]], ["test"]
    )
    res = run_abc_test(data, metrics, *DATES)
    groups = {g: data.loc[data["exp_group"] == g, "amount"].to_numpy() for g in "ABC"}
    ref = tukey_hsd_test(groups)

    assert res["p_value"].tolist() == pytest.approx(ref["p_value"].tolist())
    assert res["p_value_adj"].tolist() == pytest.approx(ref["p_value"].tolist())
    assert res["ci_lower"].tolist() == pytest.approx(ref["ci_lower"].tolist())


def test_dunnett_leaves_non_control_pairs_without_p_value(data):
    metrics = _metrics(
        [["m", "Списания", "average", "amount", "is_in_exp", "positive", "dunnett"]], ["test"]
    )
    res = run_abc_test(data, metrics, *DATES).set_index("pair")

    assert res.loc[["A_vs_B", "A_vs_C"], "p_value"].notna().all()
    assert np.isnan(res.loc["B_vs_C", "p_value"])
    assert res.loc["B_vs_C", "result_adj"] == "neutral"


def test_pairwise_options_work_in_abc_and_adjust_alias(data):
    metrics = _metrics(
        [
            ["a", "Списания", "average", "amount", "is_in_exp", "positive", "games_howell", None, None],
            ["b", "Списания CUPED", "average", "amount", "is_in_exp", "positive", None, "amount_pre", 0.99],
            ["c", "CTR", "ratio", "clicks", "shows", "positive", "delta", None, None],
        ],
        ["test", "covariate", "winsorize"],
    )
    res = run_abc_test(data, metrics, *DATES, pvalue_adjust_method="bh", extended=True)
    cuped = res[res["metric_name"] == "Списания CUPED"]

    assert len(res) == 9
    assert cuped["cuped_variance_reduction"].min() > 0.7
    assert cuped["winsorize_threshold"].nunique() == 1  # порог общий для всех пар
    assert (res["p_value_adj"] >= res["p_value"] - 1e-12).all()


def test_run_multigroup_test_handles_custom_groups_and_control(data):
    five = data.copy()
    five["exp_group"] = np.random.default_rng(0).choice(list("vwxyz"), len(five))
    metrics = _metrics([["m", "Списания", "average", "amount", "is_in_exp", "positive"]])
    all_pairs = run_multigroup_test(five, metrics, *DATES)
    control_only = run_multigroup_test(five, metrics, *DATES, control="x", include_all_pairs=False)

    assert len(all_pairs) == 10
    assert list(all_pairs.columns) == AGGREGATE_COLUMN_NAMES_ABC
    assert set(control_only["group_base"]) == {"x"} and len(control_only) == 4
    with pytest.raises(ValueError):
        run_multigroup_test(five, metrics, *DATES, control="nope")


def test_run_omnibus_test_per_metric_type(data):
    metrics = _metrics(
        [
            ["amount", "Списания", "average", "amount", "is_in_exp", "positive"],
            ["med", "Медиана", "median", "amount", "is_in_exp", "positive"],
            ["ctr", "CTR", "ratio", "clicks", "shows", "positive"],
            ["paid", "Платящие", "share", "is_paid", "is_in_exp", "positive"],
        ]
    )
    res = run_omnibus_test(data, metrics).set_index("metric_name")
    groups = [data.loc[data["exp_group"] == g, "amount"] for g in "ABC"]

    assert res["test"].tolist() == ["welch_anova", "kruskal", "welch_anova", "chi2"]
    assert res.loc["Медиана", "p_value"] == pytest.approx(stats.kruskal(*groups).pvalue)
    assert res.loc["CTR", "is_significant"]
    assert (res["n_groups"] == 3).all()
    anova = run_omnibus_test(data, metrics, numeric_test="anova").set_index("metric_name")
    assert anova.loc["Списания", "p_value"] == pytest.approx(stats.f_oneway(*groups).pvalue)


def test_extended_results_export_to_excel(tmp_path: Path, data):
    metrics = _metrics(
        [["m", "Списания", "average", "amount", "is_in_exp", "positive", "amount_pre"]],
        ["covariate"],
    )
    ab = run_ab_test(data, metrics, *DATES, extended=True, metrics_adjust_method="holm")
    abc = run_abc_test(data, metrics, *DATES, extended=True)
    ab_path = export_ab_results_to_excel(ab, tmp_path / "ab.xlsx")
    abc_path = export_abc_results_to_excel(abc, tmp_path / "abc.xlsx")

    for path in (ab_path, abc_path):
        workbook = load_workbook(path)
        header_values = {
            cell.value for sheet in workbook.worksheets for row in sheet.iter_rows(max_row=3) for cell in row
        }
        assert {"test_name", "rel_ci_lower", "delta_cuped"} <= header_values


# ----------------------------------------------------------------------------- исправления прежнего поведения


def _unequal_groups_frame() -> pd.DataFrame:
    rng = np.random.default_rng(21)
    n_a, n_b, n_c = 3000, 1000, 2000
    group = np.array(["A"] * n_a + ["B"] * n_b + ["C"] * n_c)
    value = rng.normal(100, 10, n_a + n_b + n_c) + np.where(group == "B", 5.0, 0.0)
    return pd.DataFrame({"user_id": np.arange(len(group)), "exp_group": group, "value": value})


def test_additive_delta_is_scaled_to_control_size_with_unequal_groups():
    df = _unequal_groups_frame()
    metrics = _metrics([["m", "Сумма", "additive", "value", "value", "positive"]])
    row = run_ab_test(df, metrics, *DATES).iloc[0]
    a = df.loc[df["exp_group"] == "A", "value"]
    b = df.loc[df["exp_group"] == "B", "value"]

    # группа B втрое меньше: сырая разница сумм отрицательна, хотя среднее выросло
    assert row["value_exp"] - row["value_base"] < 0
    assert row["value_base"] == pytest.approx(a.sum()) and row["value_exp"] == pytest.approx(b.sum())
    assert row["rel_delta"] == pytest.approx(b.mean() / a.mean() - 1)
    assert row["abs_delta"] == pytest.approx((b.mean() - a.mean()) * len(a))
    assert row["result"] == "positive"

    abc_row = run_abc_test(df, metrics, *DATES).set_index("pair").loc["A_vs_B"]
    assert abc_row["rel_delta"] == pytest.approx(row["rel_delta"])
    assert abc_row["result_adj"] == "positive"


def test_additive_delta_unchanged_for_equal_groups():
    df = _unequal_groups_frame().groupby("exp_group").head(1000)
    metrics = _metrics([["m", "Сумма", "additive", "value", "value", "positive"]])
    row = run_ab_test(df, metrics, *DATES).iloc[0]

    assert row["abs_delta"] == pytest.approx(row["value_exp"] - row["value_base"])


def test_ab_metric_type_and_direction_are_case_insensitive(data):
    lower = _metrics([["m", "Списания", "average", "amount", "is_in_exp", "positive"]])
    upper = _metrics([["m", "Списания", "Average", "amount", "is_in_exp", "Positive"]])
    r_lower, r_upper = run_ab_test(data, lower, *DATES), run_ab_test(data, upper, *DATES)

    assert r_upper["p_value"].notna().all()
    pd.testing.assert_frame_equal(r_lower, r_upper)


def test_power_now_is_exported_as_fraction_percent(tmp_path: Path, data):
    metrics = _metrics([["m", "Списания", "average", "amount", "is_in_exp", "positive"]])
    paths = [
        export_ab_results_to_excel(run_ab_test(data, metrics, *DATES), tmp_path / "ab.xlsx"),
        export_abc_results_to_excel(run_abc_test(data, metrics, *DATES), tmp_path / "abc.xlsx"),
    ]
    checked = 0
    for path in paths:
        for sheet in load_workbook(path).worksheets:
            for header_row in sheet.iter_rows(max_row=3):
                for cell in header_row:
                    if cell.value == "power_now":
                        value_cell = sheet.cell(cell.row + 1, cell.column)
                        assert 0 <= value_cell.value <= 1
                        assert value_cell.number_format == "0.0%"
                        checked += 1
    assert checked >= 2


def test_style_table_ab_renders_with_adjusted_columns(data):
    pytest.importorskip("jinja2")
    pytest.importorskip("matplotlib")
    from fast_exp_analytics import style_table_ab

    metrics = _metrics([["m", "Списания", "average", "amount", "is_in_exp", "positive"]])
    plain = style_table_ab(run_ab_test(data, metrics, *DATES), comment="t")
    adjusted = style_table_ab(run_ab_test(data, metrics, *DATES, metrics_adjust_method="holm"))

    assert "Списания" in plain.to_html()
    assert "result_adj" in adjusted.to_html()


def test_synthetic_dataset_has_pre_period_covariate_independent_of_group():
    from fast_exp_analytics import make_synthetic_abc_dataset

    df = make_synthetic_abc_dataset(n=20_000, seed=3)
    base = df.loc[df["exp_group"] == "A"]
    means = df.groupby("exp_group")["amount_pre"].mean()

    assert np.corrcoef(base["amount"], base["amount_pre"])[0, 1] > 0.8
    assert means.max() / means.min() < 1.15  # эффект группы B в ковариату не попадает


def test_zero_delta_with_significant_p_value_is_neutral_in_abc():
    rng = np.random.default_rng(5)
    rows = []
    for group, spread in (("A", 1.0), ("B", 30.0), ("C", 1.0)):
        # медианы равны нулю во всех группах, но распределения разные → ранговый тест значим
        values = np.concatenate([np.zeros(600), rng.exponential(spread, 400) * (1 if group != "B" else 5)])
        rows.append(pd.DataFrame({"exp_group": group, "value": values, "den": 1}))
    df = pd.concat(rows, ignore_index=True)
    df["user_id"] = np.arange(len(df))
    metrics = _metrics([["m", "Медиана", "median", "value", "den", "positive"]])
    row = run_abc_test(df, metrics, *DATES, ci_bootstrap_resamples=100).set_index("pair").loc["A_vs_B"]

    assert row["abs_delta"] == 0 and row["p_value"] < 0.05
    assert row["result"] == "neutral" and row["result_adj"] == "neutral"
