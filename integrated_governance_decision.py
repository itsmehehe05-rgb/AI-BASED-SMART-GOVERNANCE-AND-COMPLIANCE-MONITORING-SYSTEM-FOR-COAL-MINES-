from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PHASE 10
# AI INTEGRATED GOVERNANCE DECISION ENGINE
# ============================================================

BASE_DIR = Path(r"D:\CoalMineAI")
OUTPUT_DIR = BASE_DIR / "outputs"


GOVERNANCE_FILE = (
    OUTPUT_DIR /
    "multi_mine_governance_ranking.csv"
)

EARLY_WARNING_FILE = (
    OUTPUT_DIR /
    "early_warning_mine_summary.csv"
)

ESCALATION_FILE = (
    OUTPUT_DIR /
    "mine_escalation_summary.csv"
)

COMPLIANCE_FILE = (
    OUTPUT_DIR /
    "mine_specific_compliance_summary.csv"
)

REGULATORY_FILE = (
    OUTPUT_DIR /
    "mine_regulatory_summary.csv"
)

EVIDENCE_FILE = (
    OUTPUT_DIR /
    "compliance_gap_mine_summary.csv"
)

INTERVENTION_FILE = (
    OUTPUT_DIR /
    "optimal_intervention_results.csv"
)

OUTPUT_FILE = (
    OUTPUT_DIR /
    "integrated_governance_decision.csv"
)

ACTION_FILE = (
    OUTPUT_DIR /
    "integrated_management_actions.csv"
)


# ============================================================
# HELPERS
# ============================================================

def load_csv(path):

    if not path.exists():

        print(
            f"WARNING: File not found: {path}"
        )

        return None

    try:

        df = pd.read_csv(
            path,
            encoding="utf-8-sig"
        )

        print(
            f"Loaded {path.name}: {len(df)} rows"
        )

        return df

    except Exception as error:

        print(
            f"WARNING: Could not load "
            f"{path.name}: {error}"
        )

        return None


def numeric(df, column):

    if column not in df.columns:

        return pd.Series(
            0.0,
            index=df.index
        )

    return pd.to_numeric(
        df[column],
        errors="coerce"
    ).fillna(0)


def first_existing(df, columns, default=0):

    for column in columns:

        if column in df.columns:

            return numeric(
                df,
                column
            )

    return pd.Series(
        default,
        index=df.index
    )


def text_column(df, columns, default="UNKNOWN"):

    for column in columns:

        if column in df.columns:

            return (

                df[column]
                .fillna(default)
                .astype(str)

            )

    return pd.Series(
        default,
        index=df.index
    )


# ============================================================
# DECISION CLASSIFICATION
# ============================================================

def classify_priority(score):

    if score >= 75:

        return "CRITICAL"

    if score >= 60:

        return "HIGH"

    if score >= 40:

        return "MEDIUM"

    if score >= 25:

        return "WATCH"

    return "LOW"


def generate_decision(row):

    score = float(
        row["integrated_governance_score"]
    )

    warning = str(
        row["warning_level"]
    ).upper()

    escalation = str(
        row["escalation_level"]
    ).upper()

    direction = str(
        row["escalation_direction"]
    ).upper()

    compliance = float(
        row["compliance_risk_score"]
    )

    regulatory = float(
        row["regulatory_exposure_score"]
    )

    operational = float(
        row["operational_risk"]
    )

    reasons = []

    if warning in [
        "EARLY_WARNING",
        "ALERT"
    ]:

        reasons.append(
            "early-warning signal detected"
        )

    if direction in [
        "ESCALATING",
        "RAPIDLY_WORSENING",
        "WORSENING"
    ]:

        reasons.append(
            "risk trajectory is deteriorating"
        )

    if escalation in [
        "MEDIUM",
        "HIGH",
        "CRITICAL"
    ]:

        reasons.append(
            "predictive risk escalation detected"
        )

    if operational >= 50:

        reasons.append(
            "elevated operational risk"
        )

    if compliance >= 50:

        reasons.append(
            "elevated compliance risk"
        )

    if regulatory >= 60:

        reasons.append(
            "high regulatory exposure"
        )

    if not reasons:

        reasons.append(
            "no major integrated risk driver detected"
        )

    reason = "; ".join(
        reasons
    )

    # --------------------------------------------------------
    # MANAGEMENT DECISION
    # --------------------------------------------------------

    if score >= 75:

        action = (
            "Immediate management intervention required. "
            "Review operational, regulatory and compliance "
            "risks and initiate mitigation."
        )

    elif score >= 60:

        action = (
            "High-priority management review required. "
            "Investigate major risk drivers and prepare "
            "preventive intervention."
        )

    elif score >= 40:

        action = (
            "Increase monitoring frequency and review "
            "emerging operational and compliance risks."
        )

    elif score >= 25:

        action = (
            "Maintain enhanced monitoring and verify "
            "identified risk indicators."
        )

    else:

        action = (
            "Continue routine monitoring."
        )

    # --------------------------------------------------------
    # Escalation override
    # --------------------------------------------------------

    if direction in [
        "RAPIDLY_WORSENING",
        "ESCALATING"
    ]:

        action += (
            " Risk trajectory should be reviewed "
            "before the next reporting cycle."
        )

    return reason, action


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "PHASE 10 — AI INTEGRATED GOVERNANCE "
        "DECISION ENGINE"
    )

    print("=" * 70)

    # ========================================================
    # [1] LOAD GOVERNANCE
    # ========================================================

    print(
        "\n[1] Loading governance intelligence..."
    )

    governance = load_csv(
        GOVERNANCE_FILE
    )

    if governance is None:

        print(
            "ERROR: Governance dataset is required."
        )

        return

    # ========================================================
    # [2] LOAD EARLY WARNING
    # ========================================================

    print(
        "\n[2] Loading early-warning intelligence..."
    )

    early = load_csv(
        EARLY_WARNING_FILE
    )

    # ========================================================
    # [3] LOAD ESCALATION
    # ========================================================

    print(
        "\n[3] Loading predictive escalation..."
    )

    escalation = load_csv(
        ESCALATION_FILE
    )

    # ========================================================
    # [4] LOAD COMPLIANCE
    # ========================================================

    print(
        "\n[4] Loading compliance intelligence..."
    )

    compliance = load_csv(
        COMPLIANCE_FILE
    )

    # ========================================================
    # [5] LOAD REGULATORY
    # ========================================================

    print(
        "\n[5] Loading regulatory intelligence..."
    )

    regulatory = load_csv(
        REGULATORY_FILE
    )

    # ========================================================
    # [6] LOAD EVIDENCE GAP
    # ========================================================

    print(
        "\n[6] Loading evidence-gap intelligence..."
    )

    evidence = load_csv(
        EVIDENCE_FILE
    )

    # ========================================================
    # [7] PREPARE BASE
    # ========================================================

    print(
        "\n[7] Preparing latest mine profiles..."
    )

    governance[
        "subsidiary"
    ] = (

        governance[
            "subsidiary"
        ]
        .fillna("UNKNOWN")
        .astype(str)
        .str.upper()
        .str.strip()

    )

    base = (

        governance
        .sort_values(
            "date"
        )
        .groupby(
            "subsidiary",
            as_index=False
        )
        .tail(1)
        .copy()

    )

    base = base[
        [
            column
            for column in [
                "subsidiary",
                "date",
                "governance_score",
                "governance_priority",
                "production_risk",
                "overall_risk_level",
                "overall_operational_risk"
            ]
            if column in base.columns
        ]
    ]

    print(
        f"Mines detected: {len(base)}"
    )

    # ========================================================
    # [8] EARLY WARNING MERGE
    # ========================================================

    print(
        "\n[8] Integrating early-warning signals..."
    )

    if early is not None:

        early[
            "subsidiary"
        ] = (

            early[
                "subsidiary"
            ]
            .astype(str)
            .str.upper()
            .str.strip()

        )

        early_latest = (

            early
            .sort_values(
                "date"
            )
            .groupby(
                "subsidiary",
                as_index=False
            )
            .tail(1)

        )

        early_columns = [

            "subsidiary",
            "early_warning_score",
            "warning_level",
            "trajectory",
            "operational_risk_change_3m",
            "governance_change_3m",
            "target_change_3m"

        ]

        early_columns = [

            c
            for c in early_columns
            if c in early_latest.columns

        ]

        base = base.merge(
            early_latest[
                early_columns
            ],
            on="subsidiary",
            how="left"
        )

    # ========================================================
    # [9] ESCALATION MERGE
    # ========================================================

    print(
        "\n[9] Integrating predictive escalation..."
    )

    if escalation is not None:

        escalation[
            "subsidiary"
        ] = (

            escalation[
                "subsidiary"
            ]
            .astype(str)
            .str.upper()
            .str.strip()

        )

        escalation_latest = (

            escalation
            .sort_values(
                "date"
                if "date" in escalation.columns
                else escalation.columns[0]
            )
            .groupby(
                "subsidiary",
                as_index=False
            )
            .tail(1)

        )

        escalation_columns = [

            "subsidiary",
            "predicted_risk_score",
            "escalation_probability",
            "escalation_level",
            "predicted_next_risk_level",
            "escalation_direction"

        ]

        escalation_columns = [

            c
            for c in escalation_columns
            if c in escalation_latest.columns

        ]

        base = base.merge(
            escalation_latest[
                escalation_columns
            ],
            on="subsidiary",
            how="left"
        )

    # ========================================================
    # [10] COMPLIANCE MERGE
    # ========================================================

    print(
        "\n[10] Integrating mine-specific compliance..."
    )

    if compliance is not None:

        compliance[
            "subsidiary"
        ] = (

            compliance[
                "subsidiary"
            ]
            .astype(str)
            .str.upper()
            .str.strip()

        )

        compliance_latest = (

            compliance
            .groupby(
                "subsidiary",
                as_index=False
            )
            .first()

        )

        compliance_columns = [

            "subsidiary",
            "mine_compliance_risk_score",
            "compliance_risk_level",
            "critical_requirements",
            "high_risk_requirements",
            "dominant_regulatory_domain",
            "management_priority"

        ]

        compliance_columns = [

            c
            for c in compliance_columns
            if c in compliance_latest.columns

        ]

        base = base.merge(
            compliance_latest[
                compliance_columns
            ],
            on="subsidiary",
            how="left"
        )

    # ========================================================
    # [11] REGULATORY MERGE
    # ========================================================

    print(
        "\n[11] Integrating regulatory exposure..."
    )

    if regulatory is not None:

        regulatory[
            "subsidiary"
        ] = (

            regulatory[
                "subsidiary"
            ]
            .astype(str)
            .str.upper()
            .str.strip()

        )

        regulatory_latest = (

            regulatory
            .groupby(
                "subsidiary",
                as_index=False
            )
            .first()

        )

        regulatory_columns = [

            "subsidiary",
            "regulatory_exposure_score",
            "regulatory_risk_level",
            "high_match_requirements",
            "medium_match_requirements",
            "top_regulatory_domain"

        ]

        regulatory_columns = [

            c
            for c in regulatory_columns
            if c in regulatory_latest.columns

        ]

        base = base.merge(
            regulatory_latest[
                regulatory_columns
            ],
            on="subsidiary",
            how="left"
        )

    # ========================================================
    # [12] EVIDENCE GAP MERGE
    # ========================================================

    print(
        "\n[12] Integrating evidence-gap priority..."
    )

    if evidence is not None:

        evidence[
            "subsidiary"
        ] = (

            evidence[
                "subsidiary"
            ]
            .astype(str)
            .str.upper()
            .str.strip()

        )

        evidence_latest = (

            evidence
            .groupby(
                "subsidiary",
                as_index=False
            )
            .first()

        )

        evidence_columns = [

            "subsidiary",
            "evidence_gap_exposure_score",
            "management_priority",
            "high_priority_gaps",
            "unknown"

        ]

        evidence_columns = [

            c
            for c in evidence_columns
            if c in evidence_latest.columns

        ]

        base = base.merge(
            evidence_latest[
                evidence_columns
            ],
            on="subsidiary",
            how="left",
            suffixes=(
                "",
                "_evidence"
            )
        )

    # ========================================================
    # [13] NORMALIZE RISK COMPONENTS
    # ========================================================

    print(
        "\n[13] Normalizing integrated risk components..."
    )

    base[
        "governance_risk"
    ] = first_existing(
        base,
        [
            "governance_score",
            "governance_priority_score"
        ]
    )

    base[
        "operational_risk"
    ] = first_existing(
        base,
        [
            "overall_operational_risk",
            "operational_risk"
        ]
    )

    base[
        "early_warning_risk"
    ] = first_existing(
        base,
        [
            "early_warning_score"
        ]
    )

    base[
        "predicted_escalation_risk"
    ] = first_existing(
        base,
        [
            "predicted_risk_score"
        ]
    )

    base[
        "compliance_risk_score"
    ] = first_existing(
        base,
        [
            "mine_compliance_risk_score",
            "compliance_risk_score"
        ]
    )

    base[
        "regulatory_exposure_score"
    ] = first_existing(
        base,
        [
            "regulatory_exposure_score"
        ]
    )

    base[
        "evidence_gap_risk"
    ] = first_existing(
        base,
        [
            "evidence_gap_exposure_score"
        ]
    )

    # ========================================================
    # [14] CAP SCORES
    # ========================================================

    print(
        "\n[14] Normalizing scores to 0–100..."
    )

    score_columns = [

        "governance_risk",
        "operational_risk",
        "early_warning_risk",
        "predicted_escalation_risk",
        "compliance_risk_score",
        "regulatory_exposure_score",
        "evidence_gap_risk"

    ]

    for column in score_columns:

        base[column] = (

            pd.to_numeric(
                base[column],
                errors="coerce"
            )
            .fillna(0)
            .clip(
                0,
                100
            )

        )

    # ========================================================
    # [15] INTEGRATED SCORE
    # ========================================================

    print(
        "\n[15] Calculating integrated governance score..."
    )

    base[
        "integrated_governance_score"
    ] = (

        base[
            "operational_risk"
        ] * 0.20

        +

        base[
            "governance_risk"
        ] * 0.15

        +

        base[
            "early_warning_risk"
        ] * 0.15

        +

        base[
            "predicted_escalation_risk"
        ] * 0.15

        +

        base[
            "compliance_risk_score"
        ] * 0.15

        +

        base[
            "regulatory_exposure_score"
        ] * 0.10

        +

        base[
            "evidence_gap_risk"
        ] * 0.10

    )

    base[
        "integrated_governance_score"
    ] = (

        base[
            "integrated_governance_score"
        ]
        .clip(
            0,
            100
        )
        .round(2)

    )

    # ========================================================
    # [16] PRIORITY
    # ========================================================

    print(
        "\n[16] Classifying management priority..."
    )

    base[
        "integrated_priority"
    ] = (

        base[
            "integrated_governance_score"
        ]
        .apply(
            classify_priority
        )

    )

    # ========================================================
    # [17] RISK SIGNALS
    # ========================================================

    print(
        "\n[17] Integrating risk trajectories..."
    )

    base[
        "warning_level"
    ] = text_column(
        base,
        [
            "warning_level"
        ],
        "STABLE"
    )

    base[
        "escalation_level"
    ] = text_column(
        base,
        [
            "escalation_level"
        ],
        "LOW"
    )

    base[
        "escalation_direction"
    ] = text_column(
        base,
        [
            "escalation_direction"
        ],
        "STABLE"
    )

    # ========================================================
    # [18] AI DECISION
    # ========================================================

    print(
        "\n[18] Generating integrated AI decisions..."
    )

    decisions = (

        base.apply(
            generate_decision,
            axis=1
        )

    )

    base[
        "decision_reason"
    ] = [

        item[0]
        for item in decisions

    ]

    base[
        "recommended_management_action"
    ] = [

        item[1]
        for item in decisions

    ]

    # ========================================================
    # [19] MANAGEMENT DOMAIN
    # ========================================================

    print(
        "\n[19] Identifying dominant management area..."
    )

    risk_map = {

        "OPERATIONAL":
            "operational_risk",

        "GOVERNANCE":
            "governance_risk",

        "EARLY_WARNING":
            "early_warning_risk",

        "PREDICTIVE_ESCALATION":
            "predicted_escalation_risk",

        "COMPLIANCE":
            "compliance_risk_score",

        "REGULATORY":
            "regulatory_exposure_score",

        "EVIDENCE_GAP":
            "evidence_gap_risk"

    }

    def dominant_area(row):

        values = {

            label:
            float(
                row[column]
            )

            for label, column
            in risk_map.items()

        }

        return max(
            values,
            key=values.get
        )

    base[
        "dominant_management_area"
    ] = base.apply(
        dominant_area,
        axis=1
    )

    # ========================================================
    # [20] RANK MINES
    # ========================================================

    print(
        "\n[20] Ranking mines for management attention..."
    )

    base = base.sort_values(
        [
            "integrated_governance_score"
        ],
        ascending=False
    ).reset_index(
        drop=True
    )

    base.insert(
        0,
        "integrated_rank",
        range(
            1,
            len(base) + 1
        )
    )

    # ========================================================
    # [21] MANAGEMENT ACTION DATASET
    # ========================================================

    print(
        "\n[21] Creating management action dataset..."
    )

    actions = base[
        [
            "integrated_rank",
            "subsidiary",
            "integrated_governance_score",
            "integrated_priority",
            "dominant_management_area",
            "decision_reason",
            "recommended_management_action"
        ]
    ].copy()

    # ========================================================
    # [22] SAVE
    # ========================================================

    print(
        "\n[22] Saving integrated outputs..."
    )

    base.to_csv(
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
    # FINAL DISPLAY
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "AI INTEGRATED GOVERNANCE DECISION"
    )

    print(
        "=" * 70
    )

    display_columns = [

        "integrated_rank",
        "subsidiary",
        "integrated_governance_score",
        "integrated_priority",
        "dominant_management_area",
        "warning_level",
        "escalation_direction"

    ]

    print(
        base[
            [
                c
                for c in display_columns
                if c in base.columns
            ]
        ]
        .to_string(
            index=False
        )
    )

    # ========================================================
    # TOP PRIORITIES
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "TOP MANAGEMENT DECISIONS"
    )

    print(
        "=" * 70
    )

    for _, row in base.head(5).iterrows():

        print(
            f"\n#{int(row['integrated_rank'])} "
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
            "Dominant area    : "
            f"{row['dominant_management_area']}"
        )

        print(
            "Reason           : "
            f"{row['decision_reason']}"
        )

        print(
            "Action           : "
            f"{row['recommended_management_action']}"
        )

    # ========================================================
    # PRIORITY DISTRIBUTION
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "INTEGRATED PRIORITY DISTRIBUTION"
    )

    print(
        "=" * 70
    )

    print(
        base[
            "integrated_priority"
        ]
        .value_counts()
        .to_string()
    )

    # ========================================================
    # OUTPUTS
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "PHASE 10 COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nIntegrated governance decisions:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        "\nManagement actions:"
    )

    print(
        ACTION_FILE
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()