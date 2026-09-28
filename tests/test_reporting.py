import pandas as pd

from fast_exp_analytics import (
    build_ab_chat_message,
    build_abc_chat_message,
    build_dashboard_url_ab,
    make_synthetic_abc_dataset,
    run_ab_test,
    run_abc_test,
)


def _ab_result() -> pd.DataFrame:
    df = make_synthetic_abc_dataset(n=3_000, seed=42)
    df_ab = df[df["exp_group"] != "C"]
    metrics = pd.DataFrame.from_dict(
        {
            "shows": ["Показы", "additive", "shows", "shows", "positive"],
            "ctr": ["CTR", "ratio", "clicks", "shows", "positive"],
            "cr_created": ["Конверсия в создавших кампанию", "share", "is_create_ad", "is_in_exp", "positive"],
        },
        orient="index",
        columns=["desc", "type", "num", "den", "direction"],
    )
    return run_ab_test(df_ab, metrics, "2026-03-17", "2026-03-24")


def _abc_result() -> pd.DataFrame:
    df = make_synthetic_abc_dataset(n=3_000, seed=42)
    metrics = pd.DataFrame.from_dict(
        {"shows": ["Показы", "additive", "shows", "shows", "positive"]},
        orient="index",
        columns=["desc", "type", "num", "den", "direction"],
    )
    return run_abc_test(df, metrics, "2026-03-17", "2026-03-24")


def test_ab_chat_message_contains_header_and_metric_lines():
    msg = build_ab_chat_message(
        df_result=_ab_result(),
        experiment_desc="Test experiment",
        exp_id=4242,
        date_from="2026-03-17",
        date_to="2026-03-24",
        dashboard_url="https://example.com/dashboard",
    )

    assert "AB тест: Test experiment" in msg
    assert "Эксперимент:</b> 4242" in msg
    assert "Период:</b> 2026-03-17 - 2026-03-24 (8 д.)" in msg
    assert "Показы" in msg
    assert "https://example.com/dashboard" in msg


def test_ab_chat_message_handles_empty_result():
    empty = pd.DataFrame(
        columns=["metric_name", "metric_type", "value_base", "value_exp", "rel_delta", "p_value"]
    )
    msg = build_ab_chat_message(
        df_result=empty,
        experiment_desc="Test experiment",
        exp_id=1,
        date_from="2026-03-17",
        date_to="2026-03-24",
        dashboard_url="https://example.com/dashboard",
    )

    assert "Нет данных для отображения" in msg


def test_dashboard_url_ab_encodes_params():
    url = build_dashboard_url_ab(
        base_url="https://example.com/dashboards/ab",
        date_from="2026-03-17",
        date_to="2026-03-24",
        exp_id=4242,
        extra_params={"p_platform": "all", "p_packages": ["total", "mobile"]},
    )

    assert url.startswith("https://example.com/dashboards/ab?")
    assert "p_exp_id=4242" in url
    assert "p_platform=all" in url
    assert "p_packages=" in url


def test_abc_chat_message_covers_all_pairs():
    msg = build_abc_chat_message(
        df_result=_abc_result(),
        experiment_desc="Test experiment",
        exp_id=4242,
        date_from="2026-03-17",
        date_to="2026-03-24",
        dashboard_url="https://example.com/dashboard",
    )

    assert "ABC тест: Test experiment" in msg
    for pair in ("A vs B", "A vs C", "B vs C"):
        assert pair in msg
    assert "Holm" in msg
