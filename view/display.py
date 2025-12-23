from __future__ import annotations

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
    with derived_cols[1]:
        st.subheader("DSRI")
        st.metric(label="DSRI", value=_format_value(derived_metrics.get("DSRI")))


def render_charts(yoy_changes: Mapping[str, Any]) -> None:
    st.header("图表展示")
    if not yoy_changes:
        st.error("缺少同比变动率数据，无法绘制图表。")
        st.stop()

    yoy_series = pd.Series(yoy_changes)
    revenue_series = yoy_series.filter(regex="营业收入")
    profit_series = yoy_series.filter(regex="归母净利润")

    charts = st.columns(2)
    with charts[0]:
        st.subheader("营业收入同比变动率")
        st.bar_chart(revenue_series)
    with charts[1]:
        st.subheader("归母净利润同比变动率")
        st.bar_chart(profit_series)


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
