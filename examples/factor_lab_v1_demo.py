from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from logic.factor_lab import (  # noqa: E402
    Factor,
    Relation,
    Scenario,
    assumption_positive,
    assumption_yoy_positive,
    sensitivity_analysis,
)


def build_mock_data() -> pd.DataFrame:
    data = pd.DataFrame(
        {
            "company_id": [
                "A",
                "A",
                "A",
                "B",
                "B",
                "B",
                "C",
                "C",
                "C",
                "D",
                "D",
                "D",
            ],
            "year": [2021, 2022, 2023] * 4,
            "CFO": [
                120,
                160,
                140,
                80,
                60,
                90,
                30,
                20,
                -10,
                200,
                220,
                210,
            ],
            "NetProfit": [
                100,
                110,
                120,
                -20,
                10,
                30,
                15,
                12,
                8,
                180,
                190,
                195,
            ],
            "Revenue": [
                500,
                540,
                600,
                400,
                380,
                410,
                200,
                220,
                210,
                800,
                820,
                900,
            ],
        }
    )
    return data


def build_factor() -> tuple[Factor, list[Scenario]]:
    relation = Relation(operator="Ratio", operands=("CFO", "NetProfit"))
    assumptions = (
        assumption_positive(
            "NetProfit",
            assumption_id="A1",
            description="净利润为正",
        ),
        assumption_yoy_positive(
            "Revenue",
            assumption_id="A2",
            description="收入同比为正",
        ),
    )
    factor = Factor(
        name="CashProfitRatio",
        description="盈利是否被现金流支持",
        relation=relation,
        assumptions=assumptions,
    )
    scenarios = [
        Scenario(id="S0", title="无约束", enabled_assumptions=()),
        Scenario(id="S1", title="净利润>0", enabled_assumptions=("A1",)),
        Scenario(
            id="S2",
            title="净利润>0 + 收入同比>0",
            enabled_assumptions=("A1", "A2"),
        ),
    ]
    return factor, scenarios


def main() -> None:
    data = build_mock_data()
    factor, scenarios = build_factor()
    results = sensitivity_analysis(
        data,
        base_factor=factor,
        scenarios=scenarios,
        group_key="company_id",
        time_key="year",
        years=[2022, 2023],
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
