from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook

from fast_exp_analytics import (
    build_abcd_chat_message,
    export_abcd_results_to_excel,
    make_synthetic_abcd_dataset,
    run_ab_test,
    run_abcd_test,
    style_table_abcd,
)
from fast_exp_analytics.abc import AGGREGATE_COLUMN_NAMES_ABCD

ABCD_PAIRS = ("A_vs_B", "A_vs_C", "A_vs_D", "B_vs_C", "B_vs_D", "C_vs_D")


def _small_abcd_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    for group_index, group in enumerate(("A", "B", "C", "D")):
        for i in range(60):
            rows.append(
                {
                    "user_id": f"{group}-{i}",
                    "exp_group": group,
                    "value": float(rng.normal(10 + group_index * 2.0, 3.0)),
                    "denominator": 1,
                }
            )
    return pd.DataFrame(rows)


def _metrics_config() -> pd.DataFrame:
    return pd.DataFrame(
        [["Среднее", "average", "value", "denominator", "positive"]],
        index=["average"],
        columns=["desc", "type", "num", "den", "direction"],
    )


def test_abcd_returns_all_six_pairs_per_metric():
    result = run_abcd_test(_small_abcd_dataset(), _metrics_config(), "2026-01-01", "2026-01-07")

    assert set(result["pair"]) == set(ABCD_PAIRS)
    assert len(result) == 6
    assert set(result["group_exp"]) == {"B", "C", "D"}
    assert (result["group_base"] != result["group_exp"]).all()


def test_abcd_columns_match_aggregate_schema():
    result = run_abcd_test(_small_abcd_dataset(), _metrics_config(), "2026-01-01", "2026-01-07")

    assert list(result.columns) == AGGREGATE_COLUMN_NAMES_ABCD


def test_abcd_include_all_pairs_false_keeps_control_pairs_only():
    result = run_abcd_test(
        _small_abcd_dataset(), _metrics_config(), "2026-01-01", "2026-01-07", include_all_pairs=False
    )

    assert set(result["pair"]) == {"A_vs_B", "A_vs_C", "A_vs_D"}


def test_abcd_holm_adjustment_does_not_shrink_pvalues():
    result = run_abcd_test(_small_abcd_dataset(), _metrics_config(), "2026-01-01", "2026-01-07")

    assert result["p_value_adj"].notna().all()
    assert (result["p_value_adj"] >= result["p_value"] - 1e-12).all()


def test_abcd_simultaneous_ci_wider_than_ab():
    df = _small_abcd_dataset()
    df_ab = df[df["exp_group"].isin(["A", "B"])]
    metrics = _metrics_config()

    ab_row = run_ab_test(df_ab, metrics, "2026-01-01", "2026-01-07").iloc[0]
    abcd_result = run_abcd_test(df, metrics, "2026-01-01", "2026-01-07")
    abcd_ab = abcd_result.loc[abcd_result["pair"] == "A_vs_B"].iloc[0]

    ab_width = ab_row["ci_upper"] - ab_row["ci_lower"]
    abcd_width = abcd_ab["ci_upper"] - abcd_ab["ci_lower"]

    assert abcd_width > ab_width


def test_abcd_excel_export_has_sheet_per_pair(tmp_path: Path):
    result = run_abcd_test(_small_abcd_dataset(), _metrics_config(), "2026-01-01", "2026-01-07")
    output_path = tmp_path / "abcd.xlsx"

    export_abcd_results_to_excel(
        result,
        output_path,
        experiment_desc="Тест ABCD",
        exp_id=42,
        date_from="2026-01-01",
        date_to="2026-01-07",
    )

    wb = load_workbook(output_path)
    expected_sheets = {"summary", "all_metrics"} | set(ABCD_PAIRS)
    assert expected_sheets.issubset(set(wb.sheetnames))
    ws_all = wb["all_metrics"]
    assert ws_all.cell(row=1, column=1).value == "All ABCD metrics - Тест ABCD"


def test_abcd_chat_message_lists_all_pairs():
    result = run_abcd_test(_small_abcd_dataset(), _metrics_config(), "2026-01-01", "2026-01-07")

    msg = build_abcd_chat_message(
        result,
        experiment_desc="Тест ABCD",
        exp_id=42,
        date_from="2026-01-01",
        date_to="2026-01-07",
        dashboard_url="https://example.com",
    )

    assert "<b>ABCD тест: Тест ABCD</b>" in msg
    for pair in ABCD_PAIRS:
        assert pair.replace("_vs_", " vs ") in msg


def test_abcd_style_table_returns_styler():
    result = run_abcd_test(_small_abcd_dataset(), _metrics_config(), "2026-01-01", "2026-01-07")
    styler = style_table_abcd(result, caption="ABCD")

    assert hasattr(styler, "to_html")


def test_synthetic_abcd_dataset_covers_four_groups():
    df = make_synthetic_abcd_dataset(n=5_000, seed=3)

    assert set(df["exp_group"]) == {"A", "B", "C", "D"}
    assert len(df) == 5_000
    b_mean = df.loc[df["exp_group"] == "B", "amount"].mean()
    d_mean = df.loc[df["exp_group"] == "D", "amount"].mean()
    a_mean = df.loc[df["exp_group"] == "A", "amount"].mean()
    assert b_mean > a_mean
    assert d_mean > a_mean


def test_abcd_runs_on_synthetic_dataset():
    from fast_exp_analytics import default_metrics_config

    df = make_synthetic_abcd_dataset(n=10_000, seed=11)
    metrics = default_metrics_config()
    result = run_abcd_test(
        df, metrics, "2026-01-01", "2026-01-14", ci_bootstrap_resamples=200, ci_random_state=1
    )

    assert set(result["pair"]) == set(ABCD_PAIRS)
    assert result["p_value_adj"].notna().all()
