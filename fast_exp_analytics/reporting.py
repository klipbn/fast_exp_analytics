from __future__ import annotations

import html
import math
from urllib.parse import quote, urlencode

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
        return x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))
    except Exception:
        return True

def _fmt_metric_type(metric_type: str) -> str:
    return METRIC_TYPE_LABELS.get(str(metric_type).lower(), str(metric_type).lower())

def _fmt_float(x, decimals: int = 2) -> str:
    return "н/д" if _is_nan(x) else f"{float(x):.{decimals}f}".replace(".", ",")

def _fmt_pct(x) -> str:
    return "н/д" if _is_nan(x) else f"{float(x):.1f}%".replace(".", ",")

def _fmt_share01_to_pct(x, decimals: int = 2) -> str:
    return "н/д" if _is_nan(x) else f"{float(x) * 100:.{decimals}f}%".replace(".", ",")

def _fmt_p(p) -> str:
    if _is_nan(p):
        return "н/д"
    p = float(p)
    return "<0,0001" if p < 0.0001 else f"{p:.4f}".replace(".", ",")

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
    v = float(x)
    if math.isinf(v):
        return "н/д"
    if v <= 0:
        return "0д"
    if v <= 60:
        return f"{int(round(v))}д"
    if v <= 365:
        return f"~{int(round(v / 30))} мес"
    return "слишком долго"

def _pair_label(pair: str) -> str:
    return {"A_vs_B": "A vs B", "A_vs_C": "A vs C", "B_vs_C": "B vs C"}.get(pair, str(pair))

def _safe_rel_delta_pct(row: pd.Series) -> float:
    rd = row.get("rel_delta")
    return float("nan") if _is_nan(rd) else float(rd) * 100

def _pick_effect_p(row: pd.Series, use_adjusted: bool = True):
    return row.get("p_value_adj") if use_adjusted and "p_value_adj" in row.index else row.get("p_value")

def _pick_result(row: pd.Series, use_adjusted: bool = True):
    return str(row.get("result_adj" if use_adjusted and "result_adj" in row.index else "result", "neutral")).lower()

def _icon_for_row(row: pd.Series, *, alpha: float, use_adjusted: bool = True, near_sig_p: float = 0.10, big_delta_pct_default: float = 7.0, big_delta_pct_share: float = 3.0, big_delta_pct_price: float = 7.0):
    res = _pick_result(row, use_adjusted=use_adjusted)
    p = _pick_effect_p(row, use_adjusted=use_adjusted)
    rd_pct = _safe_rel_delta_pct(row)
    metric_type = str(row.get("metric_type", "")).lower()
    metric_name = str(row.get("metric_name", "")).upper()
    sig = (not _is_nan(p)) and (float(p) < alpha)
    thr = big_delta_pct_default
    if metric_type == "share":
        thr = big_delta_pct_share
    if metric_name in {"CPA", "CPC", "CPM"}:
        thr = big_delta_pct_price
    big_delta = (not _is_nan(rd_pct)) and abs(rd_pct) >= thr
    near_sig = (not _is_nan(p)) and (alpha <= float(p) <= near_sig_p)
    if sig and res == "positive":
        return "🟢"
    if sig and res == "negative":
        return "🔴"
    if near_sig or big_delta:
        return "🟠"
    return "⚪"

def build_dashboard_url_abc(*, base_url: str, params: dict[str, str | int | list | tuple | None] | None = None) -> str:
    if not params:
        return base_url
    prepared = {}
    for key, value in params.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            prepared[key] = str(list(value))
        else:
            prepared[key] = str(value)
    return f"{base_url}?{urlencode(prepared, quote_via=quote, safe='')}"

def build_abc_chat_message(
    df_result: pd.DataFrame,
    *,
    experiment_desc: str,
    exp_id: str | int,
    date_from: str,
    date_to: str,
    dashboard_url: str,
    alpha: float = 0.05,
    p_adjust_method: str = "Holm",
    use_adjusted: bool = True,
    pair_order: tuple[str, ...] = ("A_vs_B", "A_vs_C", "B_vs_C"),
    key_metrics: tuple[str, ...] = (
        "Списания",
        "Цели",
        "CPA",
        "CTR",
        "Конверсия в начавших тратить",
        "Конверсия в создавших кампанию",
        "Goals per day",
        "Amount per day",
    ),
    max_metrics_per_pair: int = 5,
    max_colored_extra_per_pair: int | None = 3,
    show_days_more: bool = True,
) -> str:
    df = df_result.copy()

    # считаем длительность периода включительно
    period_days_text = ""
    try:
        dt_from = pd.to_datetime(date_from).date()
        dt_to = pd.to_datetime(date_to).date()
        days_cnt = (dt_to - dt_from).days + 1
        if days_cnt > 0:
            period_days_text = f" ({days_cnt} д.)"
    except Exception:
        period_days_text = ""

    period_text = f"{html.escape(date_from)} - {html.escape(date_to)}{period_days_text}"

    if df.empty:
        return (
            f"<b>ABC тест: {html.escape(str(experiment_desc))}</b>\n"
            f"<b>Эксперимент:</b> {html.escape(str(exp_id))}\n"
            f"<b>Период:</b> {period_text}\n\n"
            f"Нет данных для отображения.\n\n🔎 Подробнее\n{dashboard_url}"
        )

    for col in [
        "p_value",
        "p_value_adj",
        "rel_delta",
        "value_base",
        "value_exp",
        "days_more_if_same_delta",
        "power_now",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    pair_rank = {p: i for i, p in enumerate(pair_order)}
    key_pos = {m: i for i, m in enumerate(key_metrics)}
    df["__pair_rank"] = df["pair"].map(lambda x: pair_rank.get(x, 999))
    df["__key_rank"] = df["metric_name"].map(lambda x: key_pos.get(x, 999))
    df["__p_eff"] = df["p_value_adj"] if (use_adjusted and "p_value_adj" in df.columns) else df["p_value"]
    df["__rd_pct"] = df["rel_delta"] * 100
    df["__icon"] = df.apply(lambda r: _icon_for_row(r, alpha=alpha, use_adjusted=use_adjusted), axis=1)

    msg_lines = [
        f"<b>ABC тест: {html.escape(str(experiment_desc))}</b>",
        f"<b>Эксперимент:</b> {html.escape(str(exp_id))}",
        f"<b>Период:</b> {period_text}",
        f"<b>Поправка:</b> {html.escape(str(p_adjust_method))}" if use_adjusted else "",
        "",
        "<b>Кратко по парам:</b>",
    ]
    msg_lines = [x for x in msg_lines if x != ""]

    for pair in pair_order:
        sub = df[df["pair"] == pair].copy()
        if sub.empty:
            continue

        icons = sub["__icon"]
        cnt_pos = int((icons == "🟢").sum())
        cnt_neg = int((icons == "🔴").sum())
        cnt_warn = int((icons == "🟠").sum())

        sig_sub = sub[
            sub.apply(
                lambda r: _pick_result(r, use_adjusted=use_adjusted) in {"positive", "negative"},
                axis=1,
            )
        ]

        if sig_sub.empty:
            near_sub = sub.sort_values(
                ["__p_eff", "__key_rank", "__rd_pct"],
                ascending=[True, True, False],
            )
            highlights = []
            for _, r in near_sub.head(2).iterrows():
                if pd.notna(r["__rd_pct"]):
                    highlights.append(f"{r['metric_name']} ({_fmt_pct(r['__rd_pct'])})")

            if highlights:
                msg_lines.append(
                    f"• <b>{_pair_label(pair)}</b> - значимых отличий нет; ближе всего: "
                    + ", ".join(map(html.escape, highlights))
                )
            else:
                msg_lines.append(f"• <b>{_pair_label(pair)}</b> - значимых отличий нет")
        else:
            sig_sub = sig_sub.sort_values(["__p_eff", "__rd_pct"], ascending=[True, False])
            highs = []
            for _, r in sig_sub.head(2).iterrows():
                arrow = "лучше" if _pick_result(r, use_adjusted=use_adjusted) == "positive" else "хуже"
                highs.append(f"{r['metric_name']} ({arrow}, {_fmt_pct(r['__rd_pct'])})")

            msg_lines.append(
                f"• <b>{_pair_label(pair)}</b> - 🟢 {cnt_pos} / 🔴 {cnt_neg} / 🟠 {cnt_warn}; "
                + ", ".join(map(html.escape, highs))
            )

    def render_metric_line(r: pd.Series) -> str:
        extras = [f"p_adj={_fmt_p(r['__p_eff'])}" if use_adjusted else f"p={_fmt_p(r['__p_eff'])}"]

        if show_days_more:
            dmore_s = _fmt_days_more_human(r.get("days_more_if_same_delta"))
            if dmore_s not in {"н/д", "слишком долго"}:
                extras.append(f"ещё~{dmore_s}")

        return (
            f"{r['__icon']} {html.escape(str(r['metric_name']))} "
            f"({_fmt_metric_type(str(r.get('metric_type', '')))}): "
            f"{_fmt_value_by_type(r.get('value_base'), str(r.get('metric_type', '')), str(r.get('metric_name', '')))} "
            f"→ "
            f"{_fmt_value_by_type(r.get('value_exp'), str(r.get('metric_type', '')), str(r.get('metric_name', '')))} "
            f"| Δ {_fmt_pct(r['__rd_pct'])} | "
            + " | ".join(extras)
        )

    for pair in pair_order:
        sub = df[df["pair"] == pair].copy()
        if sub.empty:
            continue

        msg_lines.extend(["", f"<b>{_pair_label(pair)}</b>"])

        key_sub = sub.sort_values(
            ["__key_rank", "__p_eff", "__rd_pct"],
            ascending=[True, True, False],
        )

        already = set()
        lines = []

        for _, r in key_sub[key_sub["metric_name"].isin(key_metrics)].head(max_metrics_per_pair).iterrows():
            lines.append(render_metric_line(r))
            already.add(str(r["metric_name"]))

        extra = sub[
            (~sub["metric_name"].isin(already))
            & (sub["__icon"].isin(["🟢", "🔴", "🟠"]))
        ].copy()

        extra = extra.sort_values(["__p_eff", "__rd_pct"], ascending=[True, False])

        if max_colored_extra_per_pair is not None:
            extra = extra.head(max_colored_extra_per_pair)

        for _, r in extra.iterrows():
            lines.append(render_metric_line(r))

        if not lines:
            lines.append("⚪ Нет заметных отличий по ключевым метрикам")

        msg_lines.extend(lines)

    msg_lines.extend(["", "🔎 Подробнее", dashboard_url])

    msg = "\n".join(msg_lines)
    msg = msg.replace("p_adj=<", "p_adj=&lt;").replace("p=<", "p=&lt;")

    return msg
