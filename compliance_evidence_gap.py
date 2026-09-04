from pathlib import Path
import pandas as pd
import numpy as np
import re


# ============================================================
# PHASE 9E — AI COMPLIANCE EVIDENCE & GAP INTELLIGENCE
# ============================================================

BASE_DIR = Path(r"D:\CoalMineAI")
OUTPUT_DIR = BASE_DIR / "outputs"

COMPLIANCE_FILE = (
    OUTPUT_DIR /
    "mine_specific_compliance_risk.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR /
    "mine_specific_compliance_summary.csv"
)

OUTPUT_FILE = (
    OUTPUT_DIR /
    "compliance_evidence_gap_analysis.csv"
)

MINE_SUMMARY_FILE = (
    OUTPUT_DIR /
    "compliance_gap_mine_summary.csv"
)

PRIORITY_FILE = (
    OUTPUT_DIR /
    "compliance_gap_priority_actions.csv"
)


# ============================================================
# HELPERS
# ============================================================

def clean_text(value):

    if pd.isna(value):
        return ""

    text = str(value).lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def numeric(series):

    return pd.to_numeric(
        series,
        errors="coerce"
    ).fillna(0)


def classify_gap(
    compliance_score,
    requirement_priority,
    evidence_type,
    required_action
):

    score = float(compliance_score)
    priority = float(requirement_priority)

    action = str(
        required_action
    ).upper()

    # --------------------------------------------------------
    # IMPORTANT:
    # No evidence data is available yet.
    #
    # Therefore we cannot claim actual non-compliance.
    # The engine estimates verification priority.
    # --------------------------------------------------------

    if score >= 75 or priority >= 75:

        return "HIGH_VERIFICATION_PRIORITY"

    if score >= 60 or priority >= 60:

        return "MEDIUM_VERIFICATION_PRIORITY"

    if score >= 45:

        return "ROUTINE_VERIFICATION"

    return "LOW_PRIORITY"


def evidence_status():

    # No actual evidence repository has been connected yet.
    #
    # Therefore every requirement is UNKNOWN rather than
    # incorrectly classified as NON_COMPLIANT.

    return "UNKNOWN"


def evidence_confidence():

    return "NOT_AVAILABLE"


def generate_gap_reason(row):

    reasons = []

    score = float(
        row["mine_specific_compliance_score"]
    )

    priority = float(
        row["regulatory_priority_score"]
    )

    domain = str(
        row["regulatory_domain"]
    ).upper()

    operational = float(
        row["overall_operational_risk"]
    )

    governance = float(
        row["governance_score"]
    )

    if score >= 70:

        reasons.append(
            "high mine-specific regulatory exposure"
        )

    elif score >= 50:

        reasons.append(
            "moderate mine-specific regulatory exposure"
        )

    if priority >= 70:

        reasons.append(
            "high regulatory priority"
        )

    elif priority >= 45:

        reasons.append(
            "medium regulatory priority"
        )

    if operational >= 50:

        reasons.append(
            "elevated operational risk"
        )

    if governance >= 50:

        reasons.append(
            "elevated governance priority"
        )

    reasons.append(
        f"{domain.lower()} requirement"
    )

    if not reasons:

        return (
            "Requirement retained for compliance verification."
        )

    return (
        "Verification priority based on "
        + ", ".join(reasons)
        + "."
    )


def management_action(
    gap_level,
    evidence_status_value,
    required_action
):

    action = str(
        required_action
    ).upper()

    # --------------------------------------------------------
    # Unknown evidence
    # --------------------------------------------------------

    if evidence_status_value == "UNKNOWN":

        if gap_level == "HIGH_VERIFICATION_PRIORITY":

            return (
                "Priority verification required. "
                "Identify responsible party and verify "
                "whether the required compliance evidence exists."
            )

        if gap_level == "MEDIUM_VERIFICATION_PRIORITY":

            return (
                "Review requirement applicability and "
                "verify available compliance evidence."
            )

        return (
            "Include requirement in routine compliance review."
        )

    # --------------------------------------------------------
    # Future-proof actions if evidence data is connected
    # --------------------------------------------------------

    if evidence_status_value == "MISSING":

        if action == "SUBMIT":

            return (
                "Verify submission requirement and initiate "
                "required submission process."
            )

        if action == "REPORT":

            return (
                "Prepare and verify required regulatory report."
            )

        if action == "NOTIFY":

            return (
                "Verify notification requirement and "
                "supporting notification record."
            )

        return (
            "Obtain and retain the required compliance evidence."
        )

    if evidence_status_value == "OVERDUE":

        return (
            "Escalate overdue compliance evidence for "
            "management review."
        )

    if evidence_status_value == "VERIFIED":

        return (
            "Evidence verified; continue scheduled monitoring."
        )

    if evidence_status_value == "NOT_APPLICABLE":

        return (
            "Document the basis for non-applicability."
        )

    return (
        "Review compliance requirement and supporting evidence."
    )


def priority_score(row):

    exposure = float(
        row["mine_specific_compliance_score"]
    )

    regulatory = float(
        row["regulatory_priority_score"]
    )

    operational = float(
        row["overall_operational_risk"]
    )

    governance = float(
        row["governance_score"]
    )

    return round(
        (
            exposure * 0.40
            +
            regulatory * 0.25
            +
            operational * 0.20
            +
            governance * 0.15
        ),
        2
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "PHASE 9E — AI COMPLIANCE EVIDENCE & GAP INTELLIGENCE"
    )
    print("=" * 70)

    # ========================================================
    # [1] LOAD 9D
    # ========================================================

    print(
        "\n[1] Loading mine-specific compliance analysis..."
    )

    if not COMPLIANCE_FILE.exists():

        print(
            "ERROR: Required file not found:"
        )

        print(
            COMPLIANCE_FILE
        )

        return

    df = pd.read_csv(
        COMPLIANCE_FILE,
        encoding="utf-8-sig"
    )

    print(
        f"Compliance records: {len(df)}"
    )

    # ========================================================
    # [2] VALIDATE
    # ========================================================

    print(
        "\n[2] Validating compliance dataset..."
    )

    required_columns = [

        "subsidiary",
        "normalized_requirement_id",
        "regulatory_domain",
        "requirement",
        "required_action",
        "regulatory_priority_score",
        "mine_specific_compliance_score",
        "overall_operational_risk",
        "governance_score"

    ]

    missing = [

        column
        for column in required_columns
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
        "Input validation successful."
    )

    # ========================================================
    # [3] CLEAN
    # ========================================================

    print(
        "\n[3] Cleaning compliance data..."
    )

    df["subsidiary"] = (
        df["subsidiary"]
        .fillna("UNKNOWN")
        .astype(str)
        .str.upper()
        .str.strip()
    )

    df["regulatory_domain"] = (
        df["regulatory_domain"]
        .fillna("GENERAL")
        .astype(str)
        .str.upper()
        .str.strip()
    )

    df["requirement"] = (
        df["requirement"]
        .fillna("")
        .astype(str)
    )

    df["required_action"] = (
        df["required_action"]
        .fillna("OTHER")
        .astype(str)
        .str.upper()
    )

    for column in [

        "regulatory_priority_score",
        "mine_specific_compliance_score",
        "overall_operational_risk",
        "governance_score"

    ]:

        df[column] = numeric(
            df[column]
        )

    print(
        f"Clean records: {len(df)}"
    )

    # ========================================================
    # [4] EVIDENCE STATUS
    # ========================================================

    print(
        "\n[4] Assessing evidence availability..."
    )

    df[
        "evidence_status"
    ] = evidence_status()

    df[
        "evidence_confidence"
    ] = evidence_confidence()

    print(
        "Evidence source: not yet connected"
    )

    print(
        "Unknown evidence status assigned "
        "instead of assuming non-compliance."
    )

    # ========================================================
    # [5] VERIFICATION PRIORITY
    # ========================================================

    print(
        "\n[5] Calculating verification priority..."
    )

    df[
        "verification_priority_score"
    ] = df.apply(
        priority_score,
        axis=1
    )

    # ========================================================
    # [6] GAP CLASSIFICATION
    # ========================================================

    print(
        "\n[6] Classifying compliance verification gaps..."
    )

    df[
        "verification_priority"
    ] = df.apply(
        lambda row:
        classify_gap(
            row[
                "mine_specific_compliance_score"
            ],
            row[
                "regulatory_priority_score"
            ],
            row.get(
                "evidence_expected",
                "UNKNOWN"
            ),
            row[
                "required_action"
            ]
        ),
        axis=1
    )

    # ========================================================
    # [7] EXPLANATIONS
    # ========================================================

    print(
        "\n[7] Generating AI gap explanations..."
    )

    df[
        "gap_explanation"
    ] = df.apply(
        generate_gap_reason,
        axis=1
    )

    # ========================================================
    # [8] MANAGEMENT ACTION
    # ========================================================

    print(
        "\n[8] Generating management actions..."
    )

    df[
        "recommended_management_action"
    ] = df.apply(
        lambda row:
        management_action(
            row[
                "verification_priority"
            ],
            row[
                "evidence_status"
            ],
            row[
                "required_action"
            ]
        ),
        axis=1
    )

    # ========================================================
    # [9] REQUIREMENT RANKING
    # ========================================================

    print(
        "\n[9] Ranking verification requirements..."
    )

    df = df.sort_values(
        [
            "subsidiary",
            "verification_priority_score"
        ],
        ascending=[
            True,
            False
        ]
    ).reset_index(
        drop=True
    )

    df[
        "verification_rank"
    ] = (

        df
        .groupby(
            "subsidiary"
        )[
            "verification_priority_score"
        ]
        .rank(
            method="first",
            ascending=False
        )
        .astype(int)

    )

    # ========================================================
    # [10] MINE SUMMARY
    # ========================================================

    print(
        "\n[10] Building mine-level evidence-gap profiles..."
    )

    summary = (

        df
        .groupby(
            "subsidiary"
        )
        .agg(

            total_requirements=(
                "normalized_requirement_id",
                "count"
            ),

            high_verification_priority=(
                "verification_priority",
                lambda x:
                (
                    x ==
                    "HIGH_VERIFICATION_PRIORITY"
                ).sum()
            ),

            medium_verification_priority=(
                "verification_priority",
                lambda x:
                (
                    x ==
                    "MEDIUM_VERIFICATION_PRIORITY"
                ).sum()
            ),

            routine_verification=(
                "verification_priority",
                lambda x:
                (
                    x ==
                    "ROUTINE_VERIFICATION"
                ).sum()
            ),

            average_verification_score=(
                "verification_priority_score",
                "mean"
            ),

            maximum_verification_score=(
                "verification_priority_score",
                "max"
            ),

            unknown_evidence_count=(
                "evidence_status",
                lambda x:
                (
                    x ==
                    "UNKNOWN"
                ).sum()
            )

        )
        .reset_index()

    )

    # ========================================================
    # [11] MINE RISK INFORMATION
    # ========================================================

    print(
        "\n[11] Integrating mine operational risk..."
    )

    mine_risk_columns = [

        "subsidiary",
        "mine_date",
        "production_risk",
        "overall_risk_level",
        "overall_operational_risk",
        "governance_score"

    ]

    mine_risk_columns = [

        column
        for column in mine_risk_columns
        if column in df.columns

    ]

    mine_latest = (

        df
        .sort_values("mine_date")
        .groupby(
            "subsidiary",
            as_index=False
        )
        .tail(1)

    )

    mine_latest = mine_latest[
        mine_risk_columns
    ]

    summary = summary.merge(
        mine_latest,
        on="subsidiary",
        how="left"
    )

    # ========================================================
    # [12] FINAL MINE GAP SCORE
    # ========================================================

    print(
        "\n[12] Calculating final mine verification exposure..."
    )

    summary[
        "mine_verification_exposure_score"
    ] = (

        summary[
            "average_verification_score"
        ] * 0.50

        +

        summary[
            "maximum_verification_score"
        ] * 0.20

        +

        np.minimum(
            summary[
                "high_verification_priority"
            ] * 5,
            100
        ) * 0.20

        +

        summary[
            "governance_score"
        ] * 0.10

    )

    summary[
        "mine_verification_exposure_score"
    ] = (

        summary[
            "mine_verification_exposure_score"
        ]
        .clip(0, 100)
        .round(2)

    )

    # ========================================================
    # [13] MINE PRIORITY
    # ========================================================

    summary[
        "management_priority"
    ] = (

        summary[
            "mine_verification_exposure_score"
        ]
        .apply(
            lambda x:
            "IMMEDIATE_VERIFICATION"
            if x >= 70
            else
            "ENHANCED_VERIFICATION"
            if x >= 45
            else
            "ROUTINE_VERIFICATION"
        )

    )

    # ========================================================
    # [14] TOP REQUIREMENT
    # ========================================================

    print(
        "\n[14] Identifying highest-priority requirements..."
    )

    top = (

        df
        .sort_values(
            [
                "subsidiary",
                "verification_priority_score"
            ],
            ascending=[
                True,
                False
            ]
        )
        .groupby(
            "subsidiary",
            as_index=False
        )
        .first()

    )

    top = top[

        [
            "subsidiary",
            "normalized_requirement_id",
            "regulatory_domain",
            "verification_priority_score",
            "evidence_status",
            "evidence_confidence",
            "recommended_management_action"

        ]

    ]

    top = top.rename(

        columns={

            "normalized_requirement_id":
                "top_requirement_id",

            "regulatory_domain":
                "top_requirement_domain",

            "verification_priority_score":
                "top_requirement_priority",

            "evidence_status":
                "top_requirement_evidence_status",

            "evidence_confidence":
                "top_requirement_evidence_confidence",

            "recommended_management_action":
                "top_requirement_action"

        }

    )

    summary = summary.merge(
        top,
        on="subsidiary",
        how="left"
    )

    # ========================================================
    # [15] RANK MINES
    # ========================================================

    print(
        "\n[15] Ranking mine verification priorities..."
    )

    summary = summary.sort_values(
        [
            "mine_verification_exposure_score",
            "high_verification_priority",
            "maximum_verification_score"
        ],
        ascending=[
            False,
            False,
            False
        ]
    ).reset_index(
        drop=True
    )

    summary.insert(
        0,
        "verification_rank",
        range(
            1,
            len(summary) + 1
        )
    )

    # ========================================================
    # [16] PRIORITY ACTIONS
    # ========================================================

    print(
        "\n[16] Creating priority action list..."
    )

    priority_actions = (

        df[
            df[
                "verification_priority"
            ].isin(
                [
                    "HIGH_VERIFICATION_PRIORITY",
                    "MEDIUM_VERIFICATION_PRIORITY"
                ]
            )
        ]

        .sort_values(
            "verification_priority_score",
            ascending=False
        )

        .groupby(
            "subsidiary",
            as_index=False
        )

        .head(10)

        .copy()

    )

    # ========================================================
    # [17] SAVE
    # ========================================================

    print(
        "\n[17] Saving outputs..."
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    summary.to_csv(
        MINE_SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    priority_actions.to_csv(
        PRIORITY_FILE,
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
        "AI COMPLIANCE EVIDENCE & GAP INTELLIGENCE"
    )

    print(
        "=" * 70
    )

    display_columns = [

        "verification_rank",
        "subsidiary",
        "mine_verification_exposure_score",
        "management_priority",
        "high_verification_priority",
        "medium_verification_priority",
        "unknown_evidence_count",
        "average_verification_score",
        "maximum_verification_score"

    ]

    print(
        summary[
            display_columns
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
        "TOP COMPLIANCE VERIFICATION PRIORITIES"
    )

    print(
        "=" * 70
    )

    for _, row in summary.head(5).iterrows():

        print(
            f"\n#{int(row['verification_rank'])} "
            f"{row['subsidiary']}"
        )

        print(
            "Verification exposure : "
            f"{row['mine_verification_exposure_score']:.2f}"
        )

        print(
            "Management priority   : "
            f"{row['management_priority']}"
        )

        print(
            "High-priority checks  : "
            f"{int(row['high_verification_priority'])}"
        )

        print(
            "Unknown evidence      : "
            f"{int(row['unknown_evidence_count'])}"
        )

        print(
            "Top requirement       : "
            f"{row['top_requirement_id']}"
        )

        print(
            "Domain                : "
            f"{row['top_requirement_domain']}"
        )

        print(
            "Evidence status       : "
            f"{row['top_requirement_evidence_status']}"
        )

        print(
            "Action                : "
            f"{row['top_requirement_action']}"
        )

    # ========================================================
    # EVIDENCE SUMMARY
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "EVIDENCE STATUS"
    )

    print(
        "=" * 70
    )

    print(
        df[
            "evidence_status"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nVerification priority:"
    )

    print(
        df[
            "verification_priority"
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
        "PHASE 9E COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nFull evidence-gap analysis:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        "\nMine-level gap summary:"
    )

    print(
        MINE_SUMMARY_FILE
    )

    print(
        "\nPriority compliance actions:"
    )

    print(
        PRIORITY_FILE
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()