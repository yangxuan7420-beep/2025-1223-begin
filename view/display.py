from __future__ import annotations

"""展示层模块：负责在 Streamlit 页面渲染指标、图表与风险状态。"""

import re
from typing import Any, Mapping

import pandas as pd
import streamlit as st


def _format_value(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "NaN"
    return f"{value:,.2f}" if isinstance(value, (int, float)) else str(value)


def render_overview(base_metrics: Mapping[str, Any], derived_metrics: Mapping[str, Any]) -> None:
    st.header("关键指标总览")
    base_cols = st.columns(len(base_metrics) or 1)
    for idx, (key, value) in enumerate(base_metrics.items()):
        with base_cols[idx % len(base_cols)]:
            st.metric(label=key, value=_format_value(value))

    derived_cols = st.columns(2)
    with derived_cols[0]:
        st.subheader("自由现金流")
        st.metric(label="自由现金流", value=_format_value(derived_metrics.get("自由现金流")))
        fcf_reason = derived_metrics.get("自由现金流缺失原因")
        if fcf_reason:
            st.caption(f"提示：{fcf_reason}")
    with derived_cols[1]:
        st.subheader("DSRI")
        st.metric(label="DSRI", value=_format_value(derived_metrics.get("DSRI")))


def render_charts(yoy_trends: Mapping[str, Any]) -> None:
    st.header("图表展示")
    revenue_series = yoy_trends.get("营业收入", pd.Series(dtype=float))
    profit_series = yoy_trends.get("利润", pd.Series(dtype=float))

    charts = st.columns(2)
    with charts[0]:
        st.subheader("营业收入同比趋势")
        if revenue_series.empty:
            st.info("缺少营业收入同比数据，无法绘制趋势。")
        else:
            st.line_chart(revenue_series)
    with charts[1]:
        st.subheader("利润同比趋势")
        if profit_series.empty:
            st.info("缺少利润同比数据，无法绘制趋势。")
        else:
            st.line_chart(profit_series)


def render_risk_status(risk_matrix: Mapping[str, Any]) -> None:
    st.header("风险状态展示")
    if not risk_matrix:
        st.error("缺少风险矩阵，无法展示。")
        st.stop()

    def _status_icon(value: Any) -> str:
        if value is True:
            return "✅"
        if value is False:
            return "⬜"
        return "—"

    def _status_label(value: Any) -> str:
        if value is True:
            return "触发异常"
        if value is False:
            return "未触发"
        return "不适用"

    def _with_sign(percent: str) -> str:
        if percent.startswith("-"):
            return percent
        return f"+{percent}"

    def _humanize_reason(reason: str, context: str) -> str:
        trimmed = reason.strip()

        if trimmed.startswith("missing_metric:"):
            detail = trimmed.split(":", 1)[1].strip()
            return f"相关财务科目在报表中缺失，无法评估（{detail}）"

        if trimmed == "not_applicable":
            return "当前未定义适用于该指标的风险规则"
        if trimmed == "missing_metric":
            return "相关财务科目在报表中缺失，无法评估"
        if trimmed == "insufficient_periods":
            return "可用年份不足，无法进行同比判断"
        if trimmed == "cannot_compute":
            return "指标计算条件不满足"
        if trimmed == "division_by_zero":
            return "同比基期为 0，无法计算变化率"

        yoy_match = re.match(r"(.+?) YoY=([+-]?\d+)% > ([+-]?\d+)%", trimmed)
        if yoy_match:
            label, yoy_percent, threshold = yoy_match.groups()
            friendly_percent = _with_sign(yoy_percent if yoy_percent.endswith("%") else f"{yoy_percent}%")
            threshold_display = threshold if threshold.endswith("%") else f"{threshold}%"
            return f"同比序列中检测到异常：{label} 年同比 {friendly_percent}（阈值 {threshold_display}）"

        if trimmed.startswith("FCF="):
            return f"自由现金流为 {trimmed.split('=', 1)[1]}，低于正常经营区间"

        dsri_match = re.match(r"DSRI=([\d\.]+) > ([\d\.]+)", trimmed)
        if dsri_match:
            value, threshold = dsri_match.groups()
            return f"派生指标异常：DSRI={value}，高于阈值 {threshold}"

        if context == "yoy":
            return f"同比分析提示：{trimmed}"
        return trimmed

    def _explanations(info: Mapping[str, Any], context: str) -> list[str]:
        value = info.get("value")
        reasons = info.get("reasons") or []

        if value is None:
            mapped = [_humanize_reason(reason, context) for reason in reasons or ["cannot_compute"]]
            return list(dict.fromkeys(mapped))

        if value is False:
            if not reasons:
                return ["未检测到异常的同比变化" if context == "yoy" else "派生指标结果处于正常范围内"]
            return [_humanize_reason(reason, context) for reason in reasons]

        humanized = [_humanize_reason(reason, context) for reason in reasons]
        return humanized or ["触发异常，请查看相关数据"]

    rows: list[dict[str, Any]] = []
    for metric_key, payload in risk_matrix.items():
        display_name = payload.get("display_name", metric_key)
        yoy_info = payload.get("yoy_anomaly", {})
        derived_info = payload.get("derived_anomaly", {})
        rows.append(
            {
                "指标": display_name,
                "同比异常": f"{_status_icon(yoy_info.get('value'))} {_status_label(yoy_info.get('value'))}",
                "派生异常": f"{_status_icon(derived_info.get('value'))} {_status_label(derived_info.get('value'))}",
                "同比说明": "；".join(_explanations(yoy_info, "yoy")),
                "派生说明": "；".join(_explanations(derived_info, "derived")),
            }
        )

    if not rows:
        st.error("缺少风险矩阵，无法展示。")
        st.stop()

    flag_df = pd.DataFrame(rows).set_index("指标")
    st.caption("✅ 触发异常；⬜ 未触发；— 无法评估/不适用。")
    st.dataframe(flag_df)
