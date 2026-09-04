import os
import numpy as np
import pandas as pd

# ============================================================
# PHASE 7A — MULTI-MINE AI GOVERNANCE RANKING
# ============================================================

INPUT_FILE = r"D:\CoalMineAI\outputs\production_risk_analysis.csv"

OUTPUT_DIR = r"D:\CoalMineAI\outputs"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "multi_mine_governance_ranking.csv"
)

SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "mine_governance_summary.csv"
)


# ============================================================
# WEIGHTS
# ============================================================

WEIGHTS = {
    "production": 0.30,
    "operational": 0.30,
    "target": 0.20,
    "trend": 0.10,
    "consistency": 0.10
}


# ============================================================
# UTILITIES
# ============================================================

def safe_float(value, default=0.0):

    try:

        value = float(value)

        if not np.isfinite(value):
            return default

        return value

    except:

        return default


def clamp(value, minimum=0, maximum=100):

    value = safe_float(value)

    return max(
        minimum,
        min(value, maximum)
    )


# ============================================================
# LOAD
# ============================================================

def load_data():

    print("\n[1] Loading production risk analysis...")

    print(
        f"File: {INPUT_FILE}"
    )

    if not os.path.exists(INPUT_FILE):

        raise FileNotFoundError(
            f"\nFile not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Rows loaded: {len(df)}"
    )

    print(
        f"Columns loaded: {len(df.columns)}"
    )

    # --------------------------------------------------------
    # Verify required columns
    # --------------------------------------------------------

    required = [

        "date",
        "subsidiary",

        "production_risk",

        "predicted_target_achievement_pct",

        "overall_operational_risk",

        "equipment_risk",

        "logistics_risk",

        "weather_risk",

        "workforce_risk"
    ]

    missing = [

        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "\nMissing required columns:\n"
            + "\n".join(
                f" - {x}"
                for x in missing
            )
        )

    return df


# ============================================================
# CLEAN
# ============================================================

def clean_data(df):

    print(
        "\n[2] Cleaning governance data..."
    )

    df = df.copy()

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df["subsidiary"] = (
        df["subsidiary"]
        .astype(str)
        .str.strip()
    )

    df = df.dropna(
        subset=[
            "date",
            "subsidiary"
        ]
    )

    invalid_subsidiaries = [

        "",
        "nan",
        "none",
        "null"
    ]

    df = df[
        ~df[
            "subsidiary"
        ]
        .str.lower()
        .isin(
            invalid_subsidiaries
        )
    ].copy()

    numeric_columns = [

        "predicted_production_mt",

        "production_target_mt",

        "predicted_target_achievement_pct",

        "expected_shortfall_mt",

        "expected_shortfall_pct",

        "equipment_risk",

        "logistics_risk",

        "weather_risk",

        "workforce_risk",

        "overall_operational_risk"
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    df = df.sort_values(
        [
            "subsidiary",
            "date"
        ]
    ).reset_index(
        drop=True
    )

    print(
        f"Clean rows: {len(df)}"
    )

    print(
        f"Subsidiaries detected: "
        f"{df['subsidiary'].nunique()}"
    )

    return df


# ============================================================
# PRODUCTION RISK SCORE
# ============================================================

def production_risk_score(value):

    value = str(
        value
    ).upper().strip()

    mapping = {

        "LOW": 20,

        "MEDIUM": 55,

        "HIGH": 85,

        "UNRELIABLE": 100
    }

    return mapping.get(
        value,
        50
    )


# ============================================================
# TARGET RISK
# ============================================================

def target_risk_score(achievement):

    achievement = safe_float(
        achievement,
        100
    )

    if achievement >= 100:

        return 0

    if achievement >= 95:

        return clamp(
            (100 - achievement) * 3
        )

    if achievement >= 90:

        return clamp(
            15 +
            (95 - achievement) * 5
        )

    return clamp(
        40 +
        (90 - achievement) * 3
    )


# ============================================================
# TREND + CONSISTENCY
# ============================================================

def calculate_mine_metrics(group):

    group = group.sort_values(
        "date"
    ).copy()

    risks = pd.to_numeric(
        group[
            "overall_operational_risk"
        ],
        errors="coerce"
    ).fillna(0)

    # --------------------------------------------------------
    # Recent trend
    # --------------------------------------------------------

    if len(risks) >= 3:

        recent = risks.tail(3)

        first = safe_float(
            recent.iloc[0]
        )

        last = safe_float(
            recent.iloc[-1]
        )

        trend_change = last - first

    else:

        trend_change = 0

    # --------------------------------------------------------
    # Trend score
    # --------------------------------------------------------

    trend_score = clamp(
        50 +
        trend_change * 5
    )

    if trend_change >= 5:

        trend_direction = "RAPIDLY_WORSENING"

    elif trend_change >= 2:

        trend_direction = "WORSENING"

    elif trend_change <= -5:

        trend_direction = "RAPIDLY_IMPROVING"

    elif trend_change <= -2:

        trend_direction = "IMPROVING"

    else:

        trend_direction = "STABLE"

    # --------------------------------------------------------
    # Consistency
    # --------------------------------------------------------

    if len(risks) >= 3:

        recent_std = risks.tail(
            min(6, len(risks))
        ).std()

        consistency_score = clamp(
            30 +
            safe_float(
                recent_std
            ) * 2
        )

    else:

        consistency_score = 50

    return (
        trend_score,
        trend_direction,
        consistency_score
    )


# ============================================================
# PRIORITY
# ============================================================

def priority(score):

    if score >= 75:

        return "CRITICAL"

    elif score >= 60:

        return "HIGH"

    elif score >= 40:

        return "MEDIUM"

    return "LOW"


# ============================================================
# MANAGEMENT ACTION
# ============================================================

def management_action(row):

    score = safe_float(
        row["governance_score"]
    )

    production = safe_float(
        row["production_risk_score"]
    )

    operational = safe_float(
        row["operational_risk_score"]
    )

    target = safe_float(
        row["target_risk_score"]
    )

    trend = safe_float(
        row["trend_risk_score"]
    )

    # --------------------------------------------------------
    # Critical
    # --------------------------------------------------------

    if score >= 75:

        return (
            "Immediate management intervention; "
            "review production, equipment and logistics."
        )

    # --------------------------------------------------------
    # Production
    # --------------------------------------------------------

    if production >= 75:

        return (
            "Investigate production shortfall "
            "and initiate recovery planning."
        )

    # --------------------------------------------------------
    # Operations
    # --------------------------------------------------------

    if operational >= 70:

        return (
            "Prioritize equipment and logistics "
            "risk mitigation."
        )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    if target >= 70:

        return (
            "Review target achievement and "
            "implement production recovery actions."
        )

    # --------------------------------------------------------
    # Trend
    # --------------------------------------------------------

    if trend >= 65:

        return (
            "Increase monitoring because "
            "operational risk is deteriorating."
        )

    return (
        "Continue routine monitoring."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n" + "=" * 70
    )

    print(
        "PHASE 7A — MULTI-MINE AI GOVERNANCE RANKING"
    )

    print(
        "=" * 70
    )

    # ========================================================
    # 1
    # ========================================================

    df = load_data()

    # ========================================================
    # 2
    # ========================================================

    df = clean_data(
        df
    )

    # ========================================================
    # 3
    # ========================================================

    print(
        "\n[3] Calculating governance risk components..."
    )

    df[
        "production_risk_score"
    ] = df[
        "production_risk"
    ].apply(
        production_risk_score
    )

    df[
        "operational_risk_score"
    ] = df[
        "overall_operational_risk"
    ].apply(
        lambda x: clamp(x)
    )

    df[
        "target_risk_score"
    ] = df[
        "predicted_target_achievement_pct"
    ].apply(
        target_risk_score
    )

    # ========================================================
    # 4
    # ========================================================

    print(
        "\n[4] Analysing risk trends..."
    )

    trend_scores = {}
    trend_directions = {}
    consistency_scores = {}

    for subsidiary in df[
        "subsidiary"
    ].unique():

        mine_data = df[
            df["subsidiary"] == subsidiary
        ]

        (
            trend,
            direction,
            consistency
        ) = calculate_mine_metrics(
            mine_data
        )

        trend_scores[
            subsidiary
        ] = trend

        trend_directions[
            subsidiary
        ] = direction

        consistency_scores[
            subsidiary
        ] = consistency

    df[
        "trend_risk_score"
    ] = df[
        "subsidiary"
    ].map(
        trend_scores
    )

    df[
        "risk_trend_direction"
    ] = df[
        "subsidiary"
    ].map(
        trend_directions
    )

    # ========================================================
    # 5
    # ========================================================

    print(
        "\n[5] Calculating risk consistency..."
    )

    df[
        "consistency_risk_score"
    ] = df[
        "subsidiary"
    ].map(
        consistency_scores
    )

    print(
        "\nConsistency scores calculated for:"
    )

    for mine in sorted(
        consistency_scores
    ):

        print(
            f" - {mine}: "
            f"{consistency_scores[mine]:.2f}"
        )

    # ========================================================
    # 6
    # ========================================================

    print(
        "\n[6] Calculating governance priority..."
    )

    df[
        "governance_score"
    ] = (

        df[
            "production_risk_score"
        ] * WEIGHTS["production"]

        +

        df[
            "operational_risk_score"
        ] * WEIGHTS["operational"]

        +

        df[
            "target_risk_score"
        ] * WEIGHTS["target"]

        +

        df[
            "trend_risk_score"
        ] * WEIGHTS["trend"]

        +

        df[
            "consistency_risk_score"
        ] * WEIGHTS["consistency"]
    )

    df[
        "governance_score"
    ] = df[
        "governance_score"
    ].apply(
        clamp
    )

    df[
        "governance_priority"
    ] = df[
        "governance_score"
    ].apply(
        priority
    )

    # ========================================================
    # 7
    # ========================================================

    print(
        "\n[7] Generating management actions..."
    )

    df[
        "recommended_management_action"
    ] = df.apply(
        management_action,
        axis=1
    )

    # ========================================================
    # 8 — LATEST MINE STATUS
    # ========================================================

    print(
        "\n[8] Selecting latest record for each mine..."
    )

    latest = (

        df.sort_values(
            "date"
        )

        .groupby(
            "subsidiary",
            as_index=False
        )

        .tail(1)

        .copy()
    )

    latest = latest.sort_values(
        "governance_score",
        ascending=False
    ).reset_index(
        drop=True
    )

    latest[
        "governance_rank"
    ] = np.arange(
        1,
        len(latest) + 1
    )

    # ========================================================
    # 9 — DISPLAY
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "MULTI-MINE GOVERNANCE RANKING"
    )

    print(
        "=" * 70
    )

    display_columns = [

        "governance_rank",

        "subsidiary",

        "date",

        "governance_score",

        "governance_priority",

        "production_risk",

        "overall_risk_level",

        "risk_trend_direction",

        "recommended_management_action"
    ]

    print(
        latest[
            display_columns
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # 10 — TOP PRIORITIES
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "TOP MANAGEMENT PRIORITIES"
    )

    print(
        "=" * 70
    )

    for _, row in latest.head(5).iterrows():

        print(
            f"\n#{int(row['governance_rank'])} "
            f"{row['subsidiary']}"
        )

        print(
            f"Governance Score : "
            f"{row['governance_score']:.2f}"
        )

        print(
            f"Priority         : "
            f"{row['governance_priority']}"
        )

        print(
            f"Production Risk  : "
            f"{row['production_risk']}"
        )

        print(
            f"Operational Risk : "
            f"{row['overall_risk_level']}"
        )

        print(
            f"Trend            : "
            f"{row['risk_trend_direction']}"
        )

        print(
            f"Action           : "
            f"{row['recommended_management_action']}"
        )

    # ========================================================
    # 11 — SUMMARY
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "GOVERNANCE SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        "\nPriority distribution:"
    )

    print(
        latest[
            "governance_priority"
        ].value_counts()
    )

    print(
        "\nRisk trend distribution:"
    )

    print(
        latest[
            "risk_trend_direction"
        ].value_counts()
    )

    # ========================================================
    # 12 — SAVE
    # ========================================================

    print(
        "\n[9] Saving outputs..."
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    latest.to_csv(
        SUMMARY_FILE,
        index=False
    )

    print(
        "\nFull governance analysis:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        "\nMine-level governance summary:"
    )

    print(
        SUMMARY_FILE
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "PHASE 7A COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":

    main()