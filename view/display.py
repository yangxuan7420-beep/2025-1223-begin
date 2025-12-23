from __future__ import annotations

from pathlib import Path
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


def render_risk_status(flags: Mapping[str, Any]) -> None:
    st.header("风险状态展示")
    yoy_flags = flags.get("同比异常", {})
    derived_flags = flags.get("派生异常", {})

    flag_df = (
        pd.DataFrame({"同比异常": yoy_flags, "派生异常": derived_flags})
        if yoy_flags or derived_flags
        else pd.DataFrame()
    )

    if flag_df.empty:
        st.error("缺少风险标记，无法展示。")
        st.stop()

    st.dataframe(flag_df)


def render_mapping_info(mapping_info: Mapping[str, Any]) -> None:
    st.header("口径信息")
    profit_mode = mapping_info.get("profit_mode")
    profit_source_key = mapping_info.get("profit_source_key")
    selected_mapping = mapping_info.get("selected_mapping", {})

    info_cols = st.columns(2)
    with info_cols[0]:
        st.metric("利润口径", profit_mode or "缺失")
    with info_cols[1]:
        st.metric("利润字段", profit_source_key or "未命中")

    if not selected_mapping:
        st.info("尚未选择口径映射。")
        return

    rows: list[dict[str, str]] = []
    for metric, choice in selected_mapping.items():
        if not choice:
            rows.append({"指标": metric, "科目": "未选择", "来源表": ""})
            continue
        table_name = choice.get("table_name") or ""
        rows.append(
            {
                "指标": metric,
                "科目": str(choice.get("raw_name") or "未选择"),
                "来源表": Path(table_name).name if table_name else "",
            }
        )

    if rows:
        st.dataframe(pd.DataFrame(rows).set_index("指标"))
