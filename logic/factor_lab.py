from __future__ import annotations

"""量化因子实验模块：构建关系、假设并输出量化验证结果。"""

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd


STATUS_NORMAL = "normal"
STATUS_ABNORMAL = "abnormal"
STATUS_UNASSESSABLE = "unassessable"


@dataclass(frozen=True)
class Relation:
    operator: str
    operands: tuple[str, ...]

    def to_dict(self) -> dict:
        return {"operator": self.operator, "operands": list(self.operands)}


@dataclass(frozen=True)
class Assumption:
    id: str
    description: str
    formula: str
    evaluator: Callable[[pd.DataFrame, str, str], pd.Series]

    def to_dict(self) -> dict:
        return {"id": self.id, "description": self.description, "formula": self.formula}


@dataclass(frozen=True)
class Factor:
    name: str
    description: str
    relation: Relation
    assumptions: tuple[Assumption, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "formula": self.relation.to_dict(),
            "assumptions": [assumption.to_dict() for assumption in self.assumptions],
        }


@dataclass(frozen=True)
class Scenario:
    id: str
    title: str
    enabled_assumptions: tuple[str, ...]
    note: str | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "enabled_assumptions": list(self.enabled_assumptions),
            "note": self.note,
        }


def _ensure_columns(data: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    frame = data.copy()
    for column in columns:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame


def _compute_ratio(numerator: pd.Series, denominator: pd.Series) -> tuple[pd.Series, pd.Series]:
    missing = numerator.isna() | denominator.isna()
    zero_denominator = denominator == 0
    abnormal = zero_denominator & ~missing
    unassessable = missing
    values = numerator / denominator
    values = values.where(~(abnormal | unassessable))
    status = pd.Series(STATUS_NORMAL, index=values.index)
    status = status.mask(unassessable, STATUS_UNASSESSABLE)
    status = status.mask(abnormal, STATUS_ABNORMAL)
    return values, status


def _compute_diff(left: pd.Series, right: pd.Series) -> tuple[pd.Series, pd.Series]:
    missing = left.isna() | right.isna()
    values = left - right
    values = values.where(~missing)
    status = pd.Series(STATUS_NORMAL, index=values.index)
    status = status.mask(missing, STATUS_UNASSESSABLE)
    return values, status


def _compute_yoy(
    data: pd.DataFrame, column: str, group_key: str, time_key: str
) -> tuple[pd.Series, pd.Series]:
    series = data[column]
    previous = series.groupby(data[group_key]).shift(1)
    missing = series.isna() | previous.isna()
    zero_previous = previous == 0
    abnormal = zero_previous & ~missing
    values = series / previous - 1
    values = values.where(~(missing | abnormal))
    status = pd.Series(STATUS_NORMAL, index=values.index)
    status = status.mask(missing, STATUS_UNASSESSABLE)
    status = status.mask(abnormal, STATUS_ABNORMAL)
    return values, status


def _evaluate_relation(
    data: pd.DataFrame, relation: Relation, group_key: str, time_key: str
) -> tuple[pd.Series, pd.Series]:
    operator = relation.operator
    if operator == "Ratio":
        numerator, denominator = relation.operands
        return _compute_ratio(data[numerator], data[denominator])
    if operator == "Diff":
        left, right = relation.operands
        return _compute_diff(data[left], data[right])
    if operator == "YoY":
        (column,) = relation.operands
        return _compute_yoy(data, column, group_key, time_key)
    raise ValueError(f"Unsupported operator: {operator}")


def _combine_status(base: pd.Series, updates: pd.Series) -> pd.Series:
    combined = base.copy()
    combined = combined.mask(updates == STATUS_UNASSESSABLE, STATUS_UNASSESSABLE)
    combined = combined.mask(updates == STATUS_ABNORMAL, STATUS_ABNORMAL)
    return combined


def compute_factor(
    data: pd.DataFrame, factor: Factor, group_key: str, time_key: str
) -> pd.DataFrame:
    data = data.copy()
    data = data.sort_values([group_key, time_key])
    required_columns = set(factor.relation.operands)
    data = _ensure_columns(data, required_columns)
    values, status = _evaluate_relation(data, factor.relation, group_key, time_key)

    for assumption in factor.assumptions:
        assumption_mask = assumption.evaluator(data, group_key, time_key).astype("boolean")
        missing = assumption_mask.isna()
        failed = assumption_mask.eq(False) & ~missing
        assumption_status = pd.Series(STATUS_NORMAL, index=data.index)
        assumption_status = assumption_status.mask(missing, STATUS_UNASSESSABLE)
        assumption_status = assumption_status.mask(failed & ~missing, STATUS_ABNORMAL)
        status = _combine_status(status, assumption_status)

    result = pd.DataFrame(
        {
            group_key: data[group_key],
            time_key: data[time_key],
            "factor_value": values,
            "status": status,
        }
    )
    return result


def _infer_time_key(factor_values: pd.DataFrame, group_key: str) -> str:
    candidates = [
        column
        for column in factor_values.columns
        if column not in {group_key, "factor_value", "status"}
    ]
    if len(candidates) != 1:
        raise ValueError("Unable to infer time key from factor values.")
    return candidates[0]


def cross_section_summary(
    factor_values: pd.DataFrame, year: int, group_key: str
) -> dict:
    time_key = _infer_time_key(factor_values, group_key)
    year_slice = factor_values[factor_values[time_key] == year]
    status_counts = year_slice["status"].value_counts().to_dict()
    total = int(len(year_slice))
    normal_values = year_slice[year_slice["status"] == STATUS_NORMAL]
    rankings = (
        normal_values.sort_values("factor_value", ascending=False)
        .loc[:, [group_key, "factor_value"]]
        .reset_index(drop=True)
    )
    quantiles = (
        normal_values["factor_value"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()
        if not normal_values.empty
        else {}
    )
    top_n = rankings.head(3)[group_key].tolist()
    bottom_n = rankings.tail(3)[group_key].tolist()

    coverage = {
        "total": total,
        "normal": int(status_counts.get(STATUS_NORMAL, 0)),
        "abnormal": int(status_counts.get(STATUS_ABNORMAL, 0)),
        "unassessable": int(status_counts.get(STATUS_UNASSESSABLE, 0)),
    }

    return {
        "year": year,
        "ranking": rankings.to_dict(orient="records"),
        "quantiles": quantiles,
        "groups": {"top": top_n, "bottom": bottom_n},
        "coverage": coverage,
    }


def _rank_frame(values: pd.DataFrame, group_key: str) -> pd.Series:
    ordered = values.sort_values("factor_value", ascending=False)
    ranks = ordered["factor_value"].rank(ascending=False, method="average")
    ranks.index = ordered[group_key]
    return ranks


def _rank_overlap(
    base: pd.Series, other: pd.Series, top_n: int = 3, bottom_n: int = 3
) -> dict:
    base = base.dropna()
    other = other.dropna()
    common = base.index.intersection(other.index)
    if common.empty:
        return {"spearman": np.nan, "top_overlap": [], "bottom_overlap": []}

    base_common = base.loc[common]
    other_common = other.loc[common]
    base_ranks = base_common.rank(ascending=True, method="average")
    other_ranks = other_common.rank(ascending=True, method="average")
    spearman = base_ranks.corr(other_ranks, method="pearson")
    base_top = set(base_common.nsmallest(top_n).index)
    other_top = set(other_common.nsmallest(top_n).index)
    base_bottom = set(base_common.nlargest(bottom_n).index)
    other_bottom = set(other_common.nlargest(bottom_n).index)

    return {
        "spearman": float(spearman) if spearman is not None else np.nan,
        "top_overlap": sorted(base_top & other_top),
        "bottom_overlap": sorted(base_bottom & other_bottom),
    }


def sensitivity_analysis(
    data: pd.DataFrame,
    base_factor: Factor,
    scenarios: Sequence[Scenario],
    group_key: str,
    time_key: str,
    years: Sequence[int],
) -> dict:
    scenario_results: dict[str, dict] = {}
    scenario_rankings: dict[str, dict[int, pd.Series]] = {}

    for scenario in scenarios:
        enabled = set(scenario.enabled_assumptions)
        assumptions = tuple(
            assumption for assumption in base_factor.assumptions if assumption.id in enabled
        )
        scenario_factor = Factor(
            name=base_factor.name,
            description=base_factor.description,
            relation=base_factor.relation,
            assumptions=assumptions,
        )
        factor_values = compute_factor(data, scenario_factor, group_key, time_key)
        summaries = {}
        rankings = {}
        for year in years:
            summary = cross_section_summary(factor_values, year, group_key)
            summaries[year] = summary
            ranking_frame = pd.DataFrame(summary["ranking"]) if summary["ranking"] else pd.DataFrame()
            if not ranking_frame.empty:
                rankings[year] = _rank_frame(ranking_frame, group_key)
            else:
                rankings[year] = pd.Series(dtype=float)
        scenario_results[scenario.id] = {
            "scenario": scenario.to_dict(),
            "factor": scenario_factor.to_dict(),
            "summaries": summaries,
        }
        scenario_rankings[scenario.id] = rankings

    comparisons = {}
    if scenarios:
        base_id = scenarios[0].id
        base_rankings = scenario_rankings.get(base_id, {})
        for scenario in scenarios[1:]:
            other_rankings = scenario_rankings.get(scenario.id, {})
            per_year = {}
            for year in years:
                per_year[year] = _rank_overlap(base_rankings.get(year, pd.Series(dtype=float)), other_rankings.get(year, pd.Series(dtype=float)))
            comparisons[scenario.id] = {
                "base": base_id,
                "scenario": scenario.id,
                "ranking_change": per_year,
            }

    return {"scenarios": scenario_results, "comparisons": comparisons}


def assumption_positive(column: str, *, assumption_id: str, description: str) -> Assumption:
    formula = f"{column} > 0"

    def evaluator(data: pd.DataFrame, group_key: str, time_key: str) -> pd.Series:
        return data[column] > 0

    return Assumption(id=assumption_id, description=description, formula=formula, evaluator=evaluator)
