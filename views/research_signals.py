from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import streamlit as st


Signal = dict[str, Any]


def _normalize_flag(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _latest_non_nan(series: pd.Series | None) -> Any:
    if series is None or not isinstance(series, pd.Series) or series.empty:
        return None
    ordered = series.dropna()
    if ordered.empty:
        return None
    return ordered.iloc[-1]


def _series_sign(series: pd.Series | None) -> int | None:
    latest = _latest_non_nan(series)
    if latest is None:
        return None
    if latest > 0:
        return 1
    if latest < 0:
        return -1
    return 0


def _has_direction_mismatch(series_a: pd.Series | None, series_b: pd.Series | None) -> bool:
    sign_a = _series_sign(series_a)
    sign_b = _series_sign(series_b)
    if sign_a is None or sign_b is None:
        return False
    return sign_a != sign_b


def _collect_signatures(series_list: Iterable[pd.Series | None]) -> set[int]:
    signs: set[int] = set()
    for series in series_list:
        sign = _series_sign(series)
        if sign is not None:
            signs.add(sign)
    return signs


def _build_signal(
    *,
    signal_id: str,
    title: str,
    level: str,
    summary: str,
    evidence: Sequence[Mapping[str, str]],
    copy_text: Sequence[str],
) -> Signal:
    return {
        "id": signal_id,
        "title": title,
        "level": level,
        "summary": summary,
        "evidence": list(evidence),
        "copy_text": list(copy_text),
    }


def _profit_quality_signal(logic_output: Mapping[str, Any]) -> Signal:
    derived = logic_output.get("派生指标", {})
    yoy_trends = derived.get("同比趋势", {})
    risk_matrix = logic_output.get("风险矩阵", {})

    profit_trend = yoy_trends.get("利润", pd.Series(dtype=float))
    cfo_trend = yoy_trends.get("经营现金流", pd.Series(dtype=float))

    profit_flag = _normalize_flag(risk_matrix.get("profit", {}).get("yoy_anomaly", {}).get("value"))
    cfo_flag = _normalize_flag(risk_matrix.get("operating_cf", {}).get("yoy_anomaly", {}).get("value"))

    divergent_flags = (profit_flag is True and cfo_flag is False) or (profit_flag is False and cfo_flag is True)
    trend_mismatch = _has_direction_mismatch(profit_trend, cfo_trend)

    if divergent_flags:
        level = "high"
    elif trend_mismatch:
        level = "medium"
    else:
        level = "low"

    summary = (
        "利润与经营现金流的同比走势出现分化，需进一步核查盈利质量。"
        if level != "low"
        else "利润与经营现金流同比表现大体一致，暂未发现明显背离。"
    )

    evidence = [
        {"type": "chart", "key": "利润", "caption": "利润同比趋势"},
        {"type": "chart", "key": "经营现金流", "caption": "经营现金流同比趋势"},
    ]

    copy_text = [
        "关注利润与经营现金流的节奏是否一致，确认利润质量是否受应收或一次性因素影响。",
        "结合现金流入结构与利润来源，核查盈利的可持续性及会计估计变动的影响。",
    ]

    return _build_signal(
        signal_id="profit_quality",
        title="盈利质量（Profit Quality）",
        level=level,
        summary=summary,
        evidence=evidence,
        copy_text=copy_text,
    )


def _financial_structure_signal(logic_output: Mapping[str, Any]) -> Signal:
    derived = logic_output.get("派生指标", {})
    yoy_trends = derived.get("同比趋势", {})
    flags = logic_output.get("异常标记", {})
    yoy_flags = flags.get("同比异常", {}) if isinstance(flags, Mapping) else {}

    debt_flag = _normalize_flag(yoy_flags.get("有息负债"))

    level = "high" if debt_flag else "low"

    summary = (
        "有息负债出现显著波动，需关注融资用途与资本结构调整。"
        if level == "high"
        else "当前有息负债节奏平稳，暂未观察到异常波动。"
    )

    evidence = [
        {"type": "chart", "key": "有息负债", "caption": "有息负债同比趋势"},
    ]

    copy_text = [
        "梳理新增有息负债的资金用途，评估对资本结构与财务费用的潜在影响。",
        "结合偿债计划与再融资安排，关注债务滚动压力及可能的再融资风险。",
    ]

    return _build_signal(
        signal_id="financial_structure",
        title="财务结构（Leverage / Debt）",
        level=level,
        summary=summary,
        evidence=evidence,
        copy_text=copy_text,
    )


def _operating_stability_signal(logic_output: Mapping[str, Any]) -> Signal:
    derived = logic_output.get("派生指标", {})
    yoy_trends = derived.get("同比趋势", {})
    flags = logic_output.get("异常标记", {})
    yoy_flags = flags.get("同比异常", {}) if isinstance(flags, Mapping) else {}

    revenue_trend = yoy_trends.get("营业收入", pd.Series(dtype=float))
    profit_trend = yoy_trends.get("利润", pd.Series(dtype=float))
    cfo_trend = yoy_trends.get("经营现金流", pd.Series(dtype=float))

    volatility_flags = [
        _normalize_flag(yoy_flags.get("营业收入")),
        _normalize_flag(yoy_flags.get("利润")),
        _normalize_flag(yoy_flags.get("经营现金流")),
    ]
    has_volatility = any(flag is True for flag in volatility_flags)

    trend_signs = _collect_signatures([revenue_trend, profit_trend, cfo_trend])
    has_divergence = len(trend_signs - {0}) > 1

    if has_volatility or has_divergence:
        level = "medium"
        summary = "收入、利润与经营现金流的节奏存在分化，需关注经营结构变化。"
    else:
        level = "low"
        summary = "核心经营指标同比走势整体平稳，暂未见显著波动。"

    evidence = [
        {"type": "chart", "key": "营业收入", "caption": "营业收入同比趋势"},
        {"type": "chart", "key": "利润", "caption": "利润同比趋势"},
        {"type": "chart", "key": "经营现金流", "caption": "经营现金流同比趋势"},
    ]

    copy_text = [
        "对比收入、利润与经营现金流的节奏，识别是否存在结构性变化或一次性扰动。",
        "如观察到分化，进一步拆解业务条线与成本费用，确认波动来源及可持续性。",
    ]

    return _build_signal(
        signal_id="operating_stability",
        title="经营稳定性（Operating Stability）",
        level=level,
        summary=summary,
        evidence=evidence,
        copy_text=copy_text,
    )


def generate_research_signals(logic_output: Mapping[str, Any]) -> list[Signal]:
    signals = [
        _profit_quality_signal(logic_output),
        _financial_structure_signal(logic_output),
        _operating_stability_signal(logic_output),
    ]
    return signals


def _resolve_series(logic_output: Mapping[str, Any], key: str) -> pd.Series:
    derived = logic_output.get("派生指标", {})
    yoy_trends = derived.get("同比趋势", {})
    series = yoy_trends.get(key)
    if isinstance(series, pd.Series):
        return series
    return pd.Series(dtype=float)


def _resolve_metric(logic_output: Mapping[str, Any], key: str) -> Any:
    base = logic_output.get("基础指标", {})
    if isinstance(base, Mapping):
        return base.get(key)
    return None


def _render_level_badge(level: str) -> None:
    colors = {"high": "red", "medium": "orange", "low": "green"}
    color = colors.get(level, "gray")
    st.markdown(f"<span style='color:{color}; font-weight:bold;'>Level: {level}</span>", unsafe_allow_html=True)


def _render_evidence_block(logic_output: Mapping[str, Any], evidence: Sequence[Mapping[str, str]]) -> None:
    if not evidence:
        st.caption("暂无证据可展示。")
        return

    for item in evidence:
        evidence_type = item.get("type")
        caption = item.get("caption") or ""
        key = item.get("key") or ""

        if evidence_type == "chart":
            series = _resolve_series(logic_output, key)
            st.caption(caption)
            if series is None or series.dropna().empty:
                st.info("缺少可绘制的数据。")
            else:
                st.line_chart(series)
        elif evidence_type == "metric":
            value = _resolve_metric(logic_output, key)
            st.metric(label=caption or key, value=value if value is not None else "NaN")
        else:
            st.caption(caption or "未指定的证据类型")


def render_research_signals(logic_output: Mapping[str, Any]) -> None:
    st.header("研究提示（Research Signals）")
    signals = generate_research_signals(logic_output)

    for signal in signals:
        with st.container():
            st.subheader(signal["title"])
            _render_level_badge(signal["level"])
            st.write(signal["summary"])

            with st.expander("查看证据", expanded=False):
                _render_evidence_block(logic_output, signal.get("evidence", []))

            with st.expander("研报可用表述", expanded=False):
                for text in signal.get("copy_text", []):
                    st.markdown(f"- {text}")
