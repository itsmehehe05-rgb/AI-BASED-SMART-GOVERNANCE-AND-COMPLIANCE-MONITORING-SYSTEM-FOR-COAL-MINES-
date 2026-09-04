from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PHASE 10.1
# AI DECISION EXPLAINABILITY & CONFIDENCE ENGINE
# ============================================================

BASE_DIR = Path(r"D:\CoalMineAI")
OUTPUT_DIR = BASE_DIR / "outputs"

INPUT_FILE = (
    OUTPUT_DIR /
    "integrated_governance_decision.csv"
)

OUTPUT_FILE = (
    OUTPUT_DIR /
    "explainable_governance_decisions.csv"
)

ACTION_FILE = (
    OUTPUT_DIR /
    "explainable_management_actions.csv"
)


# ============================================================
# HELPERS
# ============================================================

def num(df, column):

    if column not in df.columns:

        return pd.Series(
            0.0,
            index=df.index
        )

    return pd.to_numeric(
        df[column],
        errors="coerce"
    ).fillna(0)


def text(df, column, default="UNKNOWN"):

    if column not in df.columns:

        return pd.Series(
            default,
            index=df.index
        )

    return (
        df[column]
        .fillna(default)
        .astype(str)
    )


def confidence_level(row):

    score = float(
        row["integrated_governance_score"]
    )

    warning = str(
        row["warning_level"]
    ).upper()

    direction = str(
        row["escalation_direction"]
    ).upper()

    escalation_probability = float(
        row["escalation_probability"]
    )

    # Strong agreement between independent signals
    positive_signals = 0

    if warning in [
        "WATCH",
        "EARLY_WARNING",
        "ALERT"
    ]:
        positive_signals += 1

    if direction in [
        "ESCALATING",
        "WORSENING",
        "RAPIDLY_WORSENING"
    ]:
        positive_signals += 1

    if escalation_probability >= 40:
        positive_signals += 1

    if score >= 40:
        positive_signals += 1

    if positive_signals >= 3:
        return "HIGH"

    if positive_signals >= 2:
        return "MEDIUM"

    return "MODERATE"


def confidence_score(row):

    score = float(
        row["integrated_governance_score"]
    )

    probability = float(
        row["escalation_probability"]
    )

    warning = str(
        row["warning_level"]
    ).upper()

    direction = str(
        row["escalation_direction"]
    ).upper()

    value = 50.0

    # Strength of integrated score
    value += min(
        score * 0.20,
        20
    )

    # Escalation probability
    value += min(
        probability * 0.15,
        15
    )

    # Early warning agreement
    if warning in [
        "EARLY_WARNING",
        "ALERT"
    ]:
        value += 10

    elif warning == "WATCH":
        value += 5

    # Trend agreement
    if direction in [
        "ESCALATING",
        "RAPIDLY_WORSENING"
    ]:
        value += 10

    elif direction == "WORSENING":
        value += 5

    return round(
        min(
            value,
            100
        ),
        2
    )


# ============================================================
# DRIVER ANALYSIS
# ============================================================

def identify_drivers(row):

    components = {

        "OPERATIONAL_RISK":
            float(row["operational_risk"]),

        "GOVERNANCE_RISK":
            float(row["governance_risk"]),

        "EARLY_WARNING":
            float(row["early_warning_risk"]),

        "PREDICTIVE_ESCALATION":
            float(row["predicted_escalation_risk"]),

        "COMPLIANCE":
            float(row["compliance_risk_score"]),

        "REGULATORY":
            float(row["regulatory_exposure_score"]),

        "EVIDENCE_GAP":
            float(row["evidence_gap_risk"])

    }

    ordered = sorted(
        components.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return ordered


def driver_explanation(row):

    ordered = identify_drivers(
        row
    )

    primary = ordered[0]

    secondary = ordered[1]

    primary_name = primary[0]

    primary_score = primary[1]

    secondary_name = secondary[0]

    secondary_score = secondary[1]

    explanations = {

        "OPERATIONAL_RISK":
            "Operational conditions are the strongest contributor to management attention.",

        "GOVERNANCE_RISK":
            "Governance indicators are the strongest contributor to the decision.",

        "EARLY_WARNING":
            "Early-warning indicators are currently the strongest signal and require monitoring.",

        "PREDICTIVE_ESCALATION":
            "Predictive risk escalation is the strongest signal and indicates potential deterioration.",

        "COMPLIANCE":
            "Mine-specific compliance exposure is the strongest contributor to the decision.",

        "REGULATORY":
            "Regulatory exposure is the strongest contributor to the decision.",

        "EVIDENCE_GAP":
            "Compliance evidence verification exposure is the strongest contributor."

    }

    primary_reason = explanations.get(
        primary_name,
        "The primary risk driver requires management review."
    )

    return (
        primary_reason
        +
        f" Primary score: {primary_score:.2f}. "
        +
        f"Secondary driver: {secondary_name} "
        +
        f"({secondary_score:.2f})."
    )


# ============================================================
# DECISION LOGIC
# ============================================================

def final_decision(row):

    priority = str(
        row["integrated_priority"]
    ).upper()

    direction = str(
        row["escalation_direction"]
    ).upper()

    warning = str(
        row["warning_level"]
    ).upper()

    probability = float(
        row["escalation_probability"]
    )

    if priority == "CRITICAL":

        return (
            "IMMEDIATE_MANAGEMENT_INTERVENTION"
        )

    if priority == "HIGH":

        return (
            "HIGH_PRIORITY_MANAGEMENT_REVIEW"
        )

    if (
        warning == "EARLY_WARNING"
        or
        direction in [
            "ESCALATING",
            "RAPIDLY_WORSENING"
        ]
        and probability >= 40
    ):

        return (
            "ENHANCED_MONITORING_AND_PREVENTIVE_ACTION"
        )

    if priority == "MEDIUM":

        return (
            "ENHANCED_MONITORING"
        )

    if priority == "WATCH":

        return (
            "MANAGEMENT_WATCH"
        )

    return (
        "ROUTINE_MONITORING"
    )


def decision_action(row):

    decision = row[
        "final_ai_decision"
    ]

    if decision == (
        "IMMEDIATE_MANAGEMENT_INTERVENTION"
    ):

        return (
            "Immediately review operational, governance, "
            "regulatory and compliance indicators. "
            "Initiate mitigation and assign responsible ownership."
        )

    if decision == (
        "HIGH_PRIORITY_MANAGEMENT_REVIEW"
    ):

        return (
            "Conduct high-priority management review, "
            "identify the dominant risk driver and prepare "
            "preventive intervention."
        )

    if decision == (
        "ENHANCED_MONITORING_AND_PREVENTIVE_ACTION"
    ):

        return (
            "Increase monitoring frequency, investigate "
            "the deteriorating trajectory and prepare "
            "preventive measures."
        )

    if decision == (
        "ENHANCED_MONITORING"
    ):

        return (
            "Maintain enhanced monitoring and review "
            "the highest contributing risk indicators."
        )

    if decision == (
        "MANAGEMENT_WATCH"
    ):

        return (
            "Maintain management attention and verify "
            "the indicators driving the WATCH classification."
        )

    return (
        "Continue routine monitoring."
    )


# ============================================================
# DATA LIMITATIONS
# ============================================================

def limitation_text(row):

    limitations = []

    evidence_gap = float(
        row["evidence_gap_risk"]
    )

    if evidence_gap == 0:

        limitations.append(
            "Actual mine-level compliance evidence is not connected."
        )

    if (
        str(
            row["production_risk"]
        ).upper()
        ==
        "UNRELIABLE"
    ):

        limitations.append(
            "Production reliability classification requires continued observation."
        )

    if not limitations:

        return (
            "Decision based on currently available integrated AI indicators."
        )

    return " ".join(
        limitations
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "PHASE 10.1 — AI DECISION EXPLAINABILITY "
        "& CONFIDENCE ENGINE"
    )

    print("=" * 70)

    # ========================================================
    # [1] LOAD
    # ========================================================

    print(
        "\n[1] Loading Phase 10 integrated decisions..."
    )

    if not INPUT_FILE.exists():

        print(
            "ERROR: Input file not found:"
        )

        print(
            INPUT_FILE
        )

        return

    df = pd.read_csv(
        INPUT_FILE,
        encoding="utf-8-sig"
    )

    print(
        f"Records loaded: {len(df)}"
    )

    # ========================================================
    # [2] VALIDATION
    # ========================================================

    print(
        "\n[2] Validating integrated decision data..."
    )

    required = [

        "subsidiary",
        "integrated_governance_score",
        "integrated_priority",
        "operational_risk",
        "governance_risk",
        "early_warning_risk",
        "predicted_escalation_risk",
        "compliance_risk_score",
        "regulatory_exposure_score",
        "evidence_gap_risk",
        "warning_level",
        "escalation_probability",
        "escalation_direction",
        "production_risk"

    ]

    missing = [

        column
        for column in required
        if column not in df.columns

    ]

    if missing:

        print(
            "ERROR: Missing columns:"
        )

        for column in missing:

            print(
                " -",
                column
            )

        return

    print(
        "Validation successful."
    )

    # ========================================================
    # [3] NORMALIZE
    # ========================================================

    print(
        "\n[3] Normalizing decision inputs..."
    )

    numeric_columns = [

        "integrated_governance_score",
        "operational_risk",
        "governance_risk",
        "early_warning_risk",
        "predicted_escalation_risk",
        "compliance_risk_score",
        "regulatory_exposure_score",
        "evidence_gap_risk",
        "escalation_probability"

    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        ).fillna(0)

    # ========================================================
    # [4] CONFIDENCE
    # ========================================================

    print(
        "\n[4] Calculating AI decision confidence..."
    )

    df[
        "decision_confidence_score"
    ] = df.apply(
        confidence_score,
        axis=1
    )

    df[
        "decision_confidence"
    ] = df.apply(
        confidence_level,
        axis=1
    )

    # ========================================================
    # [5] DRIVER
    # ========================================================

    print(
        "\n[5] Identifying primary and secondary drivers..."
    )

    ordered_drivers = (

        df.apply(
            identify_drivers,
            axis=1
        )

    )

    df[
        "primary_risk_driver"
    ] = [

        drivers[0][0]
        for drivers in ordered_drivers

    ]

    df[
        "primary_driver_score"
    ] = [

        round(
            drivers[0][1],
            2
        )

        for drivers in ordered_drivers

    ]

    df[
        "secondary_risk_driver"
    ] = [

        drivers[1][0]
        for drivers in ordered_drivers

    ]

    df[
        "secondary_driver_score"
    ] = [

        round(
            drivers[1][1],
            2
        )

        for drivers in ordered_drivers

    ]

    # ========================================================
    # [6] EXPLANATION
    # ========================================================

    print(
        "\n[6] Generating decision explanations..."
    )

    df[
        "ai_decision_explanation"
    ] = df.apply(
        driver_explanation,
        axis=1
    )

    # ========================================================
    # [7] FINAL DECISION
    # ========================================================

    print(
        "\n[7] Generating final AI decisions..."
    )

    df[
        "final_ai_decision"
    ] = df.apply(
        final_decision,
        axis=1
    )

    # ========================================================
    # [8] MANAGEMENT ACTION
    # ========================================================

    print(
        "\n[8] Generating management actions..."
    )

    df[
        "final_management_action"
    ] = df.apply(
        decision_action,
        axis=1
    )

    # ========================================================
    # [9] LIMITATIONS
    # ========================================================

    print(
        "\n[9] Recording decision limitations..."
    )

    df[
        "decision_data_limitation"
    ] = df.apply(
        limitation_text,
        axis=1
    )

    # ========================================================
    # [10] PRIORITY RANK
    # ========================================================

    print(
        "\n[10] Creating final management ranking..."
    )

    df = df.sort_values(
        [
            "integrated_governance_score",
            "decision_confidence_score"
        ],
        ascending=[
            False,
            False
        ]
    ).reset_index(
        drop=True
    )

    if "integrated_rank" in df.columns:

        df = df.drop(
            columns=[
                "integrated_rank"
            ]
        )

    df.insert(
        0,
        "final_management_rank",
        range(
            1,
            len(df) + 1
        )
    )

    # ========================================================
    # [11] ACTION DATASET
    # ========================================================

    print(
        "\n[11] Creating management action register..."
    )

    actions = df[
        [

            "final_management_rank",
            "subsidiary",
            "integrated_governance_score",
            "integrated_priority",
            "primary_risk_driver",
            "secondary_risk_driver",
            "decision_confidence_score",
            "decision_confidence",
            "final_ai_decision",
            "final_management_action",
            "ai_decision_explanation",
            "decision_data_limitation"

        ]
    ].copy()

    # ========================================================
    # [12] SAVE
    # ========================================================

    print(
        "\n[12] Saving explainable decisions..."
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    actions.to_csv(
        ACTION_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # DISPLAY
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "EXPLAINABLE AI GOVERNANCE DECISIONS"
    )

    print(
        "=" * 70
    )

    display_columns = [

        "final_management_rank",
        "subsidiary",
        "integrated_governance_score",
        "integrated_priority",
        "primary_risk_driver",
        "decision_confidence",
        "final_ai_decision"

    ]

    print(
        df[
            display_columns
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # TOP DECISIONS
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "TOP EXPLAINABLE AI DECISIONS"
    )

    print(
        "=" * 70
    )

    for _, row in df.head(5).iterrows():

        print(
            f"\n#{int(row['final_management_rank'])} "
            f"{row['subsidiary']}"
        )

        print(
            "Integrated score : "
            f"{row['integrated_governance_score']:.2f}"
        )

        print(
            "Priority         : "
            f"{row['integrated_priority']}"
        )

        print(
            "Primary driver   : "
            f"{row['primary_risk_driver']}"
        )

        print(
            "Driver score     : "
            f"{row['primary_driver_score']:.2f}"
        )

        print(
            "Confidence       : "
            f"{row['decision_confidence']} "
            f"({row['decision_confidence_score']:.2f})"
        )

        print(
            "Decision         : "
            f"{row['final_ai_decision']}"
        )

        print(
            "Why              : "
            f"{row['ai_decision_explanation']}"
        )

        print(
            "Action           : "
            f"{row['final_management_action']}"
        )

        print(
            "Limitation       : "
            f"{row['decision_data_limitation']}"
        )

    # ========================================================
    # DISTRIBUTIONS
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "DECISION CONFIDENCE DISTRIBUTION"
    )

    print(
        "=" * 70
    )

    print(
        df[
            "decision_confidence"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nFinal AI decision distribution:"
    )

    print(
        df[
            "final_ai_decision"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nPrimary risk drivers:"
    )

    print(
        df[
            "primary_risk_driver"
        ]
        .value_counts()
        .to_string()
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "PHASE 10.1 COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nExplainable governance decisions:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        "\nManagement action register:"
    )

    print(
        ACTION_FILE
    )


if __name__ == "__main__":
    main()