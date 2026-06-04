from __future__ import annotations

import html
import math
from urllib.parse import quote, urlencode

import numpy as np
import pandas as pd

METRIC_TYPE_LABELS = {
    "additive": "абс.",
    "ratio": "отн.",
    "share": "%",
    "average": "ср.",
    "median": "med",
}


def _is_nan(x) -> bool:
    try:
        if x is None:
            return True
        if isinstance(x, (float, int, np.number)):
            return math.isnan(float(x)) or math.isinf(float(x))
        return bool(pd.isna(x))
    except Exception:
        return True


def _fmt_metric_type(metric_type: str) -> str:
    if not metric_type:
        return ""
    return METRIC_TYPE_LABELS.get(str(metric_type).lower(), str(metric_type).lower())


def _fmt_float(x, decimals: int = 2) -> str:
    if _is_nan(x):
        return "н/д"
    return f"{float(x):.{decimals}f}".replace(".", ",")


def _fmt_pct(x) -> str:
    if _is_nan(x):
        return "н/д"
    return f"{float(x):.1f}%".replace(".", ",")


def _fmt_share01_to_pct(x, decimals: int = 2) -> str:
    if _is_nan(x):
        return "н/д"
    return f"{float(x) * 100:.{decimals}f}%".replace(".", ",")


def _fmt_p(p) -> str:
    if _is_nan(p):
        return "н/д"
    p = float(p)
    if p < 0.0001:
        return "&lt;0,0001"
    return f"{p:.4f}".replace(".", ",")


def _fmt_compact(x) -> str:
    if _is_nan(x):
        return "н/д"

    x = float(x)
    sgn = "-" if x < 0 else ""
    x = abs(x)

    if x >= 1e9:
        return f"{sgn}{(x / 1e9):.2f}B".replace(".", ",")
    if x >= 1e6:
        return f"{sgn}{(x / 1e6):.2f}M".replace(".", ",")
    if x >= 1e3:
        return f"{sgn}{(x / 1e3):.2f}K".replace(".", ",")
    return f"{sgn}{x:.4f}".replace(".", ",")


def _fmt_value_by_type(x, metric_type: str, metric_name: str | None = None) -> str:
    metric_type = (metric_type or "").lower()
    metric_name = (metric_name or "").upper()

    if metric_name == "CTR":
        return _fmt_share01_to_pct(x, decimals=3)

    if metric_type == "share":
        return _fmt_share01_to_pct(x, decimals=2)

    if metric_type == "additive":
        return _fmt_compact(x)

    return _fmt_float(x, 2)


def _fmt_days_more_human(x) -> str:
    if _is_nan(x):
        return "н/д"

    try:
        v = float(x)
        if math.isinf(v):
            return "н/д"
        if v <= 0:
            return "0д"
        if v <= 60:
            return f"{int(round(v))}д"
        if v <= 365:
            months = int(round(v / 30))
            return f"~{months} мес"
        return "слишком долго"
    except Exception:
        return "н/д"


def build_dashboard_url_ab(
    *,
    base_url: str,
    date_from: str,
    date_to: str,
    exp_id: str | int,
    extra_params: dict[str, str | int | float | list | tuple | None] | None = None,
) -> str:
    params: dict[str, str | int | float | list | tuple | None] = {
        "p_date_start": date_from,
        "p_date_end": date_to,
        "p_exp_id": exp_id,
        "x-horizon-role-mode": "true",
    }
    if extra_params:
        params.update(extra_params)

    prepared = {}
    for key, value in params.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            prepared[key] = str(list(value))
        else:
            prepared[key] = str(value)

    return f"{base_url}?{urlencode(prepared, quote_via=quote, safe='')}"


def _result_icon(result: str) -> str:
    result = str(result).lower()
    if result == "positive":
        return "🟢"
    if result == "negative":
        return "🔴"
    return "⚪"


def _safe_rel_delta_pct(row: pd.Series) -> float:
    rd = row.get("rel_delta")
    return float("nan") if _is_nan(rd) else float(rd) * 100


def _icon_for_ab_row(
    row: pd.Series,
    *,
    alpha: float = 0.05,
    near_sig_p: float = 0.10,
    big_delta_pct_default: float = 7.0,
    big_delta_pct_share: float = 3.0,
    big_delta_pct_price: float = 7.0,
) -> str:
    """
    Icon logic aligned with the ABC message:
    - 🟢 / 🔴: significant positive / negative result;
    - 🟠: close to significance or a large observed effect;
    - ⚪: neutral.
    """
    result = str(row.get("result", "neutral")).lower()
    p = row.get("p_value")
    rd_pct = _safe_rel_delta_pct(row)

    metric_type = str(row.get("metric_type", "")).lower()
    metric_name = str(row.get("metric_name", "")).upper()

    # If result was computed with a p-value, keep the legacy behavior:
    # positive/negative without a p-value is still treated as significant.
    sig = _is_nan(p) or float(p) < alpha

    if sig and result == "positive":
        return "🟢"
    if sig and result == "negative":
        return "🔴"

    thr = big_delta_pct_default
    if metric_type == "share":
        thr = big_delta_pct_share
    if metric_name in {"CPA", "CPC", "CPM"}:
        thr = big_delta_pct_price

    near_sig = (not _is_nan(p)) and (alpha <= float(p) <= near_sig_p)
    big_delta = (not _is_nan(rd_pct)) and abs(rd_pct) >= thr

    if near_sig or big_delta:
        return "🟠"

    return "⚪"


def build_ab_chat_message(
    df_result: pd.DataFrame,
    *,
    experiment_desc: str,
    exp_id: str | int,
    date_from: str,
    date_to: str,
    dashboard_url: str,
    alpha: float = 0.05,
    key_metrics: tuple[str, ...] = (
        "Списания",
        "Списания в день",
        "Списания в день (ср.)",
        "Кампании",
        "Создавшие кампанию",
        "Начавшие тратить",
        "Конверсия в начавших тратить",
        "Конверсия в создавших кампанию",
        "Цели",
        "CPA",
        "CTR",
        "Goals per day",
        "Amount per day",
    ),
    max_metrics_per_pair: int | None = 5,
    max_colored_extra_per_pair: int | None = 3,
    max_metrics: int | None = None,
    show_days_more: bool = True,
    sort_by: str = "abs_rel_delta",
) -> str:
    """
    Build an HTML chat message summarizing an A/B test.

    Metric selection logic:
    1. Key metrics are printed first in the order given by ``key_metrics``,
       at most ``max_metrics_per_pair``.
    2. Then notable extra metrics with 🟢/🔴/🟠 icons are appended,
       at most ``max_colored_extra_per_pair``.
    3. ``max_metrics`` is kept for backward compatibility: when passed,
       it overrides ``max_metrics_per_pair``.
    """
    df = df_result.copy()

    # Backward compatibility with the legacy parameter.
    if max_metrics is not None:
        max_metrics_per_pair = max_metrics

    # Period length in days, inclusive.
    period_days_text = ""
    try:
        dt_from = pd.to_datetime(date_from).date()
        dt_to = pd.to_datetime(date_to).date()
        days_cnt = (dt_to - dt_from).days + 1
        if days_cnt > 0:
            period_days_text = f" ({days_cnt} д.)"
    except Exception:
        period_days_text = ""

    period_text = f"{html.escape(str(date_from))} - {html.escape(str(date_to))}{period_days_text}"

    if df.empty:
        return (
            f"<b>AB тест: {html.escape(str(experiment_desc))}</b>\n"
            f"<b>Эксперимент:</b> {html.escape(str(exp_id))}\n"
            f"<b>Период:</b> {period_text}\n\n"
            f"Нет данных для отображения.\n\n"
            f"🔎 Подробнее дашборд:\n{dashboard_url}"
        )

    for col in [
        "p_value",
        "rel_delta",
        "value_base",
        "value_exp",
        "days_more_if_same_delta",
        "power_now",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Guard against missing optional columns.
    if "p_value" not in df.columns:
        df["p_value"] = np.nan
    if "rel_delta" not in df.columns:
        df["rel_delta"] = np.nan
    if "metric_name" not in df.columns:
        df["metric_name"] = "metric"
    if "metric_type" not in df.columns:
        df["metric_type"] = ""

    key_pos = {metric: i for i, metric in enumerate(key_metrics)}

    df["__key_rank"] = df["metric_name"].map(lambda x: key_pos.get(x, 999))
    df["__p_eff"] = df["p_value"]
    df["__rd_pct"] = df["rel_delta"] * 100
    df["__abs_rel_delta"] = df["rel_delta"].abs()
    df["__icon"] = df.apply(lambda r: _icon_for_ab_row(r, alpha=alpha), axis=1)

    def _sort_metrics(src: pd.DataFrame) -> pd.DataFrame:
        if src.empty:
            return src

        if sort_by == "p_value":
            return src.sort_values(
                ["__p_eff", "__abs_rel_delta", "__key_rank"],
                ascending=[True, False, True],
                na_position="last",
            )

        return src.sort_values(
            ["__abs_rel_delta", "__p_eff", "__key_rank"],
            ascending=[False, True, True],
            na_position="last",
        )

    def render_metric_line(row: pd.Series) -> str:
        metric_name = html.escape(str(row.get("metric_name", "metric")))
        metric_type = _fmt_metric_type(str(row.get("metric_type", "")))

        base = _fmt_value_by_type(
            row.get("value_base"),
            str(row.get("metric_type", "")),
            str(row.get("metric_name", "")),
        )
        exp = _fmt_value_by_type(
            row.get("value_exp"),
            str(row.get("metric_type", "")),
            str(row.get("metric_name", "")),
        )
        delta = _fmt_pct(row.get("__rd_pct", np.nan))

        extras = [f"p={_fmt_p(row.get('__p_eff'))}"]
        if show_days_more:
            dmore = _fmt_days_more_human(row.get("days_more_if_same_delta"))
            if dmore not in {"н/д", "слишком долго"}:
                extras.append(f"ещё~{dmore}")

        metric_type_part = f" ({metric_type})" if metric_type else ""

        return (
            f"{row.get('__icon', _result_icon(row.get('result')))} "
            f"{metric_name}{metric_type_part}: "
            f"{base} → {exp} | Δ {delta} | "
            + " | ".join(extras)
        )

    lines = [
        f"<b>AB тест: {html.escape(str(experiment_desc))}</b>",
        f"<b>Эксперимент:</b> {html.escape(str(exp_id))}",
        f"<b>Период:</b> {period_text}",
        "",
    ]

    selected_metric_names: set[str] = set()
    metric_lines: list[str] = []

    key_sub = df[df["metric_name"].isin(key_metrics)].copy()
    key_sub = key_sub.sort_values(
        ["__key_rank", "__p_eff", "__abs_rel_delta"],
        ascending=[True, True, False],
        na_position="last",
    )

    if max_metrics_per_pair is not None:
        key_sub = key_sub.head(max_metrics_per_pair)

    for _, row in key_sub.iterrows():
        metric_lines.append(render_metric_line(row))
        selected_metric_names.add(str(row.get("metric_name")))

    extra = df[
        (~df["metric_name"].astype(str).isin(selected_metric_names))
        & (df["__icon"].isin(["🟢", "🔴", "🟠"]))
    ].copy()

    extra = _sort_metrics(extra)

    if max_colored_extra_per_pair is not None:
        extra = extra.head(max_colored_extra_per_pair)

    for _, row in extra.iterrows():
        metric_lines.append(render_metric_line(row))
        selected_metric_names.add(str(row.get("metric_name")))

    # If no key metrics matched and there are no notable extras,
    # fall back to the legacy "most notable metrics" logic so the message is never empty.
    if not metric_lines:
        fallback = _sort_metrics(df.copy())
        if max_metrics_per_pair is not None:
            fallback = fallback.head(max_metrics_per_pair)

        for _, row in fallback.iterrows():
            metric_lines.append(render_metric_line(row))

    if not metric_lines:
        metric_lines.append("⚪ Нет данных по метрикам для отображения")

    lines.extend(metric_lines)
    lines.extend(["", "", f"🔎 Подробнее дашборд:\n{dashboard_url}"])

    return "\n".join(lines)
