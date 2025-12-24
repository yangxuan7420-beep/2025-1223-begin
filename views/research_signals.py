from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import streamlit as st

from logic.factor_lab import Factor, Relation, Scenario, assumption_positive, sensitivity_analysis


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
    quant_type: str,
    title: str,
    level: str,
    summary: str,
    evidence: Sequence[Mapping[str, str]],
    copy_text: Sequence[str],
) -> Signal:
    return {
        "id": signal_id,
        "quant_type": quant_type,
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
        quant_type="profit_quality",
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
        quant_type="financial_structure",
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
        quant_type="operating_stability",
        title="经营稳定性（Operating Stability）",
        level=level,
        summary=summary,
        evidence=evidence,
        copy_text=copy_text,
    )


_DEFAULT_INTERPRETATIONS: dict[str, Any] = {
    "profit_quality": {
        "generic": {
            "interpretation": "当利润与经营现金流节奏出现分化时，可能意味着利润兑现质量需要进一步核查。",
            "risk_implication": "建议对比利润与现金流来源结构，核查应收、预付、一次性项目对利润的影响。",
        },
        "industry": {
            "快消/啤酒": {
                "interpretation": "快消行业常见先款后货与预收款特征，现金流表现通常领先于利润。需区分经营改善与一次性扰动。",
                "risk_implication": "可重点核查合同负债、渠道回款与费用投放节奏，判断利润改善是否可持续。",
            }
        },
    },
    "financial_structure": {
        "generic": {
            "interpretation": "有息负债的节奏若出现显著变化，可能意味着融资结构或资金用途发生调整，需要关注资本结构质量。",
            "risk_implication": "建议梳理新增或减少的有息负债来源与用途，核查融资成本与偿债安排对未来现金流的影响。",
        },
        "industry": {
            "快消/啤酒": {
                "interpretation": "快消企业多采用渠道预收与经营性现金流覆盖营运资金，债务上升往往对应扩产或渠道投放节奏变化。",
                "risk_implication": "可结合渠道库存与产能扩张计划，核查债务用途与销售投放节奏是否匹配，关注再融资需求。",
            }
        },
    },
    "operating_stability": {
        "generic": {
            "interpretation": "收入、利润与经营现金流的走势若出现分化，可能存在经营结构变化或一次性因素扰动，需要进一步拆解。",
            "risk_implication": "建议拆分业务条线与成本费用，对比现金流与利润节奏，核查是否存在渠道变动、订单节奏或费用扰动。",
        },
        "industry": {
            "快消/啤酒": {
                "interpretation": "快消行业旺季与费用投放节奏明显，收入与现金流可能阶段性错位，需要结合渠道库存与促销节奏解读。",
                "risk_implication": "可核查渠道库存、终端动销与费用投放节奏，判断波动是否来源于季节性或渠道结构调整。",
            }
        },
    },
}


_INDUSTRY_OPTIONS = ["通用（不选行业）", "快消/啤酒", "制造/周期", "光伏/新能源", "其他"]


_FALLBACK_WARNING = "行业解释资源缺失或格式不正确，已降级为通用提示。"

_MINIMAL_GENERIC_FALLBACK = {
    "interpretation": "通用解释暂未配置，请结合核心指标核查。",
    "risk_implication": "建议关注关键假设与数据来源，进一步核查相关披露。",
}


@lru_cache(maxsize=1)
def _load_industry_interpretations() -> tuple[dict[str, Any], bool]:
    resource_path = Path(__file__).resolve().parent.parent / "resources" / "industry_interpretations.json"
    if resource_path.exists():
        try:
            with resource_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data, False
        except Exception:
            pass
    return _DEFAULT_INTERPRETATIONS, True


def _should_apply_industry(industry: str | None) -> bool:
    return industry not in (None, "通用（不选行业）", "其他")


def _resolve_interpretations(
    signal_id: str, selected_industry: str
) -> tuple[str, str | None, str, str | None, bool]:
    templates, load_failed = _load_industry_interpretations()
    resource_issue = load_failed or not isinstance(templates, Mapping)

    if not isinstance(templates, Mapping):
        templates = _DEFAULT_INTERPRETATIONS

    signal_templates = templates.get(signal_id, {}) if isinstance(templates, Mapping) else {}
    if not isinstance(signal_templates, Mapping):
        resource_issue = True
        signal_templates = _DEFAULT_INTERPRETATIONS.get(signal_id, {})

    fallback_generic = _DEFAULT_INTERPRETATIONS.get(signal_id, {}).get("generic", {})
    generic_template = signal_templates.get("generic", {}) if isinstance(signal_templates, Mapping) else {}
    if not isinstance(generic_template, Mapping):
        resource_issue = True
        generic_template = {}

    generic_interpretation = generic_template.get("interpretation") if generic_template else None
    if not generic_interpretation:
        resource_issue = True if not fallback_generic else resource_issue
        generic_interpretation = (
            fallback_generic.get("interpretation")
            if isinstance(fallback_generic, Mapping)
            else None
        ) or _MINIMAL_GENERIC_FALLBACK["interpretation"]

    generic_risk = generic_template.get("risk_implication") if generic_template else None
    if not generic_risk:
        resource_issue = True if not fallback_generic else resource_issue
        generic_risk = (
            fallback_generic.get("risk_implication")
            if isinstance(fallback_generic, Mapping)
            else None
        ) or _MINIMAL_GENERIC_FALLBACK["risk_implication"]

    industry_interpretation = None
    industry_risk = None
    if _should_apply_industry(selected_industry) and isinstance(signal_templates, Mapping):
        industry_templates = signal_templates.get("industry") if isinstance(signal_templates, Mapping) else {}
        if not isinstance(industry_templates, Mapping):
            resource_issue = True
            industry_templates = {}
        industry_match = industry_templates.get(selected_industry) if isinstance(industry_templates, Mapping) else None
        if isinstance(industry_match, Mapping):
            industry_interpretation = industry_match.get("interpretation") or None
            industry_risk = industry_match.get("risk_implication") or None

    return generic_interpretation, industry_interpretation, generic_risk, industry_risk, resource_issue


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


_FACTOR_LAB_CONFIG: dict[str, dict[str, Any]] = {
    "profit_quality": {
        "columns": ("经营现金流", "利润"),
        "factor": Factor(
            name="CashProfitYoYRatio",
            description="经营现金流同比对利润同比的支撑程度",
            relation=Relation(operator="Ratio", operands=("经营现金流", "利润")),
            assumptions=(
                assumption_positive("利润", assumption_id="P1", description="利润同比为正"),
                assumption_positive("经营现金流", assumption_id="P2", description="经营现金流同比为正"),
            ),
        ),
        "scenarios": (
            Scenario(id="S0", title="无约束", enabled_assumptions=()),
            Scenario(id="S1", title="利润同比为正", enabled_assumptions=("P1",)),
            Scenario(
                id="S2",
                title="利润同比为正 + 经营现金流同比为正",
                enabled_assumptions=("P1", "P2"),
            ),
        ),
    },
    "financial_structure": {
        "columns": ("有息负债", "货币资金"),
        "factor": Factor(
            name="DebtCashYoYRatio",
            description="有息负债同比相对于货币资金同比的变化",
            relation=Relation(operator="Ratio", operands=("有息负债", "货币资金")),
            assumptions=(
                assumption_positive("有息负债", assumption_id="F1", description="有息负债同比为正"),
                assumption_positive("货币资金", assumption_id="F2", description="货币资金同比为正"),
            ),
        ),
        "scenarios": (
            Scenario(id="S0", title="无约束", enabled_assumptions=()),
            Scenario(id="S1", title="有息负债同比为正", enabled_assumptions=("F1",)),
            Scenario(
                id="S2",
                title="有息负债同比为正 + 货币资金同比为正",
                enabled_assumptions=("F1", "F2"),
            ),
        ),
    },
    "operating_stability": {
        "columns": ("营业收入", "利润"),
        "factor": Factor(
            name="ProfitRevenueYoYRatio",
            description="利润同比相对于收入同比的变化",
            relation=Relation(operator="Ratio", operands=("利润", "营业收入")),
            assumptions=(
                assumption_positive("营业收入", assumption_id="O1", description="营业收入同比为正"),
                assumption_positive("利润", assumption_id="O2", description="利润同比为正"),
            ),
        ),
        "scenarios": (
            Scenario(id="S0", title="无约束", enabled_assumptions=()),
            Scenario(id="S1", title="营业收入同比为正", enabled_assumptions=("O1",)),
            Scenario(
                id="S2",
                title="营业收入同比为正 + 利润同比为正",
                enabled_assumptions=("O1", "O2"),
            ),
        ),
    },
}

_QUANT_CONCLUSION_TEMPLATES: dict[str, dict[str, str]] = {
    "profit_quality": {
        "cross_section": "从现金与利润一致性的角度看，{base}",
        "relative_position": "现金兑现质量的相对位置判断：{base}",
        "sensitivity": "在现金流约束下的敏感性：{base}",
    },
    "financial_structure": {
        "cross_section": "从杠杆与偿债约束的角度看，{base}",
        "relative_position": "资本结构安全边际的相对位置：{base}",
        "sensitivity": "在债务约束变化下的敏感性：{base}",
    },
    "operating_stability": {
        "cross_section": "从收入与利润稳定性的角度看，{base}",
        "relative_position": "经营波动水平的相对位置：{base}",
        "sensitivity": "在经营节奏变化下的敏感性：{base}",
    },
}


def _build_factor_lab_frame(logic_output: Mapping[str, Any], columns: Sequence[str]) -> pd.DataFrame:
    derived = logic_output.get("派生指标", {})
    yoy_trends = derived.get("同比趋势", {})
    series_frames = []
    for column in columns:
        series = yoy_trends.get(column)
        if not isinstance(series, pd.Series):
            series = pd.Series(dtype=float)
        frame = series.rename(column).to_frame()
        series_frames.append(frame)

    combined = pd.concat(series_frames, axis=1)
    combined.index.name = "year"
    combined = combined.reset_index()
    combined["year"] = pd.to_numeric(combined["year"], errors="coerce")
    combined["entity"] = "当前标的"
    return combined


def _resolve_factor_lab_years(data: pd.DataFrame) -> list[int]:
    if "year" not in data.columns:
        return []
    years = pd.to_numeric(data["year"], errors="coerce").dropna().astype(int)
    return sorted(years.unique().tolist())


def _infer_cross_section_conclusion(summary: Mapping[str, Any]) -> str:
    coverage = summary.get("coverage", {}) if isinstance(summary, Mapping) else {}
    normal_count = int(coverage.get("normal", 0) or 0)
    total_count = int(coverage.get("total", 0) or 0)

    if normal_count < 5:
        return "当前样本规模较小，该判断在横截面上的区分度有限。"
    if total_count >= 10:
        return "该判断在横截面上存在一定分化，具备区分意义。"
    return "该判断在横截面上存在一定分化，但仍需结合更多样本验证。"


def _infer_relative_position(summary: Mapping[str, Any], entity_name: str = "当前标的") -> str:
    ranking = summary.get("ranking", []) if isinstance(summary, Mapping) else []
    if not ranking:
        return "暂无当前公司的有效排序位置。"

    rank_frame = pd.DataFrame(ranking)
    if rank_frame.empty or entity_name not in rank_frame.get("entity", []):
        return "暂无当前公司的有效排序位置。"

    rank_frame = rank_frame.reset_index(drop=True)
    position_idx = int(rank_frame.index[rank_frame["entity"] == entity_name][0])
    total = len(rank_frame)
    if total <= 1:
        return "当前公司位于样本中位附近。"

    percentile = position_idx / (total - 1)
    if percentile <= 0.33:
        position = "高位"
    elif percentile >= 0.66:
        position = "低位"
    else:
        position = "中位"

    if percentile <= 0.1:
        hint = "，接近样本高位"
    elif percentile >= 0.9:
        hint = "，接近样本低位"
    else:
        hint = ""

    return f"当前公司位于样本{position}{hint}。"


def _infer_sensitivity_conclusion(comparisons: Mapping[str, Any], latest_year: int | None) -> str:
    if not comparisons or latest_year is None:
        return "暂无可用的情景对比结果，无法判断条件敏感性。"

    stable = 0
    moderate = 0
    volatile = 0

    for comparison in comparisons.values():
        ranking_change = comparison.get("ranking_change", {})
        year_result = ranking_change.get(latest_year, {}) if isinstance(ranking_change, Mapping) else {}
        spearman = year_result.get("spearman")
        if spearman is None or pd.isna(spearman):
            moderate += 1
            continue
        if spearman >= 0.7:
            stable += 1
        elif spearman >= 0.4:
            moderate += 1
        else:
            volatile += 1

    if volatile > 0:
        return "不同条件下结果波动较大，需谨慎使用该判断。"
    if stable > 0 and moderate == 0:
        return "不同条件下排序整体稳定，结论一致性较高。"
    return "不同条件下结论大体一致，但仍存在一定波动。"


def _resolve_quant_conclusions(
    quant_type: str,
    summary: Mapping[str, Any] | None,
    comparisons: Mapping[str, Any],
    latest_year: int | None,
) -> tuple[str, str, str]:
    if summary:
        cross_section = _infer_cross_section_conclusion(summary)
        relative_position = _infer_relative_position(summary)
    else:
        cross_section = "暂无可用的横截面摘要，区分度判断有限。"
        relative_position = "暂无可用的排序信息，无法判断当前公司的相对位置。"

    sensitivity = _infer_sensitivity_conclusion(comparisons, latest_year)
    templates = _QUANT_CONCLUSION_TEMPLATES.get(quant_type)
    if not templates:
        return cross_section, relative_position, sensitivity

    return (
        templates["cross_section"].format(base=cross_section),
        templates["relative_position"].format(base=relative_position),
        templates["sensitivity"].format(base=sensitivity),
    )


def _render_factor_lab_section(
    signal_id: str, quant_type: str, logic_output: Mapping[str, Any]
) -> None:
    config = _FACTOR_LAB_CONFIG.get(quant_type)
    if not config:
        st.info("当前信号暂无量化验证配置。")
        return

    st.caption("量化验证仅用于研究探索，不影响原有研究结论与风险提示。")

    if "factor_lab_results" not in st.session_state:
        st.session_state["factor_lab_results"] = {}

    run_key = f"factor_lab_run_{signal_id}"
    if st.button("运行量化验证", key=run_key):
        data = _build_factor_lab_frame(logic_output, config["columns"])
        years = _resolve_factor_lab_years(data)
        if data.empty or not years:
            st.session_state["factor_lab_results"][signal_id] = {
                "error": "缺少可用的时间序列数据，无法执行量化验证。",
            }
        else:
            results = sensitivity_analysis(
                data,
                base_factor=config["factor"],
                scenarios=config["scenarios"],
                group_key="entity",
                time_key="year",
                years=years,
            )
            st.session_state["factor_lab_results"][signal_id] = {
                "results": results,
                "years": years,
            }

    stored = st.session_state.get("factor_lab_results", {}).get(signal_id)
    if not stored:
        return

    if stored.get("error"):
        st.info(stored["error"])
        return

    results = stored.get("results", {})
    years = stored.get("years", [])
    scenarios = config["scenarios"]
    base_id = scenarios[0].id if scenarios else None
    latest_year = max(years) if years else None
    summary = None
    if base_id and latest_year is not None:
        summary = (
            results.get("scenarios", {})
            .get(base_id, {})
            .get("summaries", {})
            .get(latest_year)
        )
    comparisons = results.get("comparisons", {})
    cross_section, relative_position, sensitivity = _resolve_quant_conclusions(
        quant_type, summary, comparisons, latest_year
    )
    st.markdown(f"- 横截面结论：{cross_section}")
    st.markdown(f"- 当前公司相对位置：{relative_position}")
    st.markdown(f"- 条件敏感性结论：{sensitivity}")


def render_research_signals(logic_output: Mapping[str, Any]) -> None:
    st.header("研究提示（Research Signals）")

    if "selected_industry" not in st.session_state:
        st.session_state["selected_industry"] = _INDUSTRY_OPTIONS[0]

    selector_col, _ = st.columns([2, 3])
    with selector_col:
        st.selectbox(
            "行业（可选）：",
            _INDUSTRY_OPTIONS,
            index=_INDUSTRY_OPTIONS.index(st.session_state["selected_industry"]),
            key="selected_industry",
        )

    selected_industry = st.session_state["selected_industry"]

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

            generic_interp, industry_interp, generic_risk, industry_risk, resource_issue = _resolve_interpretations(
                signal["id"], selected_industry
            )

            with st.expander("行业解释（可选增强）", expanded=False):
                if resource_issue:
                    st.warning(_FALLBACK_WARNING)
                st.markdown(f"**通用解释：** {generic_interp}")
                if industry_interp:
                    st.markdown(f"**{selected_industry} 增强：** {industry_interp}")
                elif _should_apply_industry(selected_industry):
                    st.info("暂无该行业的增强解释，已展示通用解释。")
                else:
                    st.caption("当前未选择行业增强，已展示通用解释。")

            with st.expander("风险含义（研究提示）", expanded=False):
                if resource_issue:
                    st.warning(_FALLBACK_WARNING)
                st.markdown(f"**通用提示：** {generic_risk}")
                if industry_risk and _should_apply_industry(selected_industry):
                    st.markdown(f"**{selected_industry} 增强提示：** {industry_risk}")
                elif _should_apply_industry(selected_industry):
                    st.info("暂无该行业的增强解释，已展示通用解释。")
                else:
                    st.caption("当前未选择行业增强，已展示通用提示。")

            with st.expander("研究判断的适用边界（量化辅助）", expanded=False):
                _render_factor_lab_section(
                    signal["id"], signal.get("quant_type", signal["id"]), logic_output
                )
