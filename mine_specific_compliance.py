from pathlib import Path
import pandas as pd
import numpy as np
import re


# ============================================================
# PHASE 9D — AI MINE-SPECIFIC COMPLIANCE RISK ENGINE
# ============================================================

BASE_DIR = Path(r"D:\CoalMineAI")
OUTPUT_DIR = BASE_DIR / "outputs"

RISK_FILE = (
    OUTPUT_DIR /
    "production_risk_analysis.csv"
)

MATCHING_FILE = (
    OUTPUT_DIR /
    "regulation_mine_matching.csv"
)

REGULATORY_FILE = (
    OUTPUT_DIR /
    "normalized_regulatory_requirements.csv"
)

OUTPUT_FILE = (
    OUTPUT_DIR /
    "mine_specific_compliance_risk.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR /
    "mine_specific_compliance_summary.csv"
)

TOP_ACTIONS_FILE = (
    OUTPUT_DIR /
    "mine_compliance_actions.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

RISK_WEIGHTS = {
    "production": 0.20,
    "equipment": 0.15,
    "logistics": 0.10,
    "weather": 0.10,
    "workforce": 0.15,
    "governance": 0.15,
    "regulatory": 0.15
}


DOMAIN_OPERATIONAL_MAPPING = {

    "SAFETY": [
        "equipment_risk",
        "workforce_risk"
    ],

    "ENVIRONMENT": [
        "weather_risk"
    ],

    "WASTE": [
        "weather_risk",
        "logistics_risk"
    ],

    "MINING": [
        "production_risk_score",
        "equipment_risk",
        "logistics_risk"
    ],

    "COMPLIANCE": [
        "governance_score"
    ],

    "GENERAL": [
        "governance_score"
    ]
}


# ============================================================
# HELPERS
# ============================================================

def numeric(series):
    return pd.to_numeric(
        series,
        errors="coerce"
    ).fillna(0)


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


def score_production_risk(value):

    mapping = {
        "LOW": 20,
        "MEDIUM": 55,
        "HIGH": 85,
        "UNRELIABLE": 95
    }

    return mapping.get(
        str(value).upper().strip(),
        20
    )


def classify(score):

    if score >= 75:
        return "CRITICAL"

    if score >= 60:
        return "HIGH"

    if score >= 40:
        return "MEDIUM"

    return "LOW"


def priority_label(score):

    if score >= 75:
        return "IMMEDIATE_ACTION"

    if score >= 60:
        return "HIGH_PRIORITY"

    if score >= 40:
        return "ENHANCED_MONITORING"

    return "ROUTINE_MONITORING"


# ============================================================
# DOMAIN RISK
# ============================================================

def calculate_domain_risk(
    domain,
    profile
):

    domain = str(domain).upper().strip()

    columns = DOMAIN_OPERATIONAL_MAPPING.get(
        domain,
        DOMAIN_OPERATIONAL_MAPPING["GENERAL"]
    )

    values = []

    for column in columns:

        if column in profile:

            values.append(
                float(
                    profile[column]
                )
            )

    if not values:
        return 25.0

    return float(
        np.mean(values)
    )


# ============================================================
# REQUIREMENT TEXT RELEVANCE
# ============================================================

def requirement_keyword_score(
    requirement,
    domain
):

    text = clean_text(
        requirement
    )

    domain_keywords = {

        "SAFETY": [
            "safety",
            "worker",
            "workman",
            "accident",
            "hazard",
            "protective",
            "inspection",
            "ventilation",
            "occupational",
            "helmet",
            "danger"
        ],

        "ENVIRONMENT": [
            "environment",
            "pollution",
            "air",
            "water",
            "emission",
            "dust",
            "noise",
            "effluent",
            "monitoring",
            "environmental"
        ],

        "WASTE": [
            "waste",
            "ash",
            "overburden",
            "dump",
            "disposal",
            "hazardous",
            "storage",
            "recycling",
            "residue"
        ],

        "MINING": [
            "mine",
            "mining",
            "coal",
            "production",
            "excavation",
            "quarry",
            "pit",
            "drilling",
            "blasting",
            "haul"
        ],

        "COMPLIANCE": [
            "compliance",
            "report",
            "return",
            "record",
            "register",
            "authority",
            "approval",
            "permit",
            "licence",
            "license"
        ]
    }

    keywords = domain_keywords.get(
        domain,
        []
    )

    if not keywords:
        return 20.0

    matches = sum(
        1
        for word in keywords
        if word in text
    )

    return min(
        100.0,
        matches / len(keywords) * 100
    )


# ============================================================
# EVIDENCE TYPE
# ============================================================

def infer_evidence_type(
    action,
    requirement
):

    action = str(
        action
    ).upper()

    text = clean_text(
        requirement
    )

    if action == "REPORT":

        return "REGULATORY_REPORT"

    if action == "SUBMIT":

        return "SUBMISSION_RECORD"

    if action == "MONITOR":

        return "MONITORING_RECORD"

    if action == "NOTIFY":

        return "NOTIFICATION_RECORD"

    if action == "MAINTAIN":

        return "MAINTENANCE_OR_REGISTER_RECORD"

    if action == "RECORD":

        return "REGISTER_OR_LOG"

    if action == "OBTAIN":

        return "PERMIT_OR_APPROVAL"

    if action == "PROHIBIT":

        return "INSPECTION_OR_CONTROL_RECORD"

    if action == "ENSURE":

        return "IMPLEMENTATION_EVIDENCE"

    if "inspection" in text:

        return "INSPECTION_RECORD"

    if "permit" in text or "licence" in text:

        return "PERMIT_OR_APPROVAL"

    return "DOCUMENTARY_COMPLIANCE_EVIDENCE"


# ============================================================
# MANAGEMENT ACTION
# ============================================================

def management_action(
    level,
    required_action,
    evidence
):

    action = str(
        required_action
    ).upper()

    if level == "CRITICAL":

        return (
            "Immediate management review; verify applicability, "
            "responsibility and compliance evidence."
        )

    if level == "HIGH":

        if action == "REPORT":

            return (
                "Verify reporting status, deadline and "
                "supporting regulatory records."
            )

        if action == "SUBMIT":

            return (
                "Verify submission status, responsible "
                "authority and deadline."
            )

        if action == "MONITOR":

            return (
                "Verify monitoring frequency, results and "
                "supporting evidence."
            )

        if action == "PROHIBIT":

            return (
                "Immediately verify that the prohibited "
                "activity is controlled."
            )

        return (
            "Conduct priority compliance review and "
            "verify supporting evidence."
        )

    if level == "MEDIUM":

        return (
            "Review requirement applicability and "
            "verify available compliance evidence."
        )

    return (
        "Retain for routine compliance monitoring."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "PHASE 9D — AI MINE-SPECIFIC COMPLIANCE RISK ENGINE"
    )
    print("=" * 70)

    # ========================================================
    # [1] LOAD RISK DATA
    # ========================================================

    print(
        "\n[1] Loading mine operational risk data..."
    )

    if not RISK_FILE.exists():

        print(
            "ERROR: Missing file:"
        )

        print(
            RISK_FILE
        )

        return

    risk = pd.read_csv(
        RISK_FILE,
        encoding="utf-8-sig"
    )

    print(
        f"Risk records: {len(risk)}"
    )

    # ========================================================
    # [2] LOAD MATCHING DATA
    # ========================================================

    print(
        "\n[2] Loading regulation-to-mine matches..."
    )

    if not MATCHING_FILE.exists():

        print(
            "ERROR: Missing file:"
        )

        print(
            MATCHING_FILE
        )

        return

    matches = pd.read_csv(
        MATCHING_FILE,
        encoding="utf-8-sig"
    )

    print(
        f"Matching records: {len(matches)}"
    )

    # ========================================================
    # [3] LOAD REGULATORY DATA
    # ========================================================

    print(
        "\n[3] Loading regulatory knowledge base..."
    )

    if not REGULATORY_FILE.exists():

        print(
            "ERROR: Missing file:"
        )

        print(
            REGULATORY_FILE
        )

        return

    regulations = pd.read_csv(
        REGULATORY_FILE,
        encoding="utf-8-sig"
    )

    print(
        f"Regulatory requirements: "
        f"{len(regulations)}"
    )

    # ========================================================
    # [4] VALIDATION
    # ========================================================

    print(
        "\n[4] Validating inputs..."
    )

    required_risk = [

        "date",
        "subsidiary",
        "production_risk",
        "equipment_risk",
        "logistics_risk",
        "weather_risk",
        "workforce_risk",
        "overall_operational_risk",
        "governance_priority_score"

    ]

    missing = [
        column
        for column in required_risk
        if column not in risk.columns
    ]

    if missing:

        print(
            "Missing required columns:"
        )

        for column in missing:

            print(
                " -",
                column
            )

        return

    required_match = [

        "subsidiary",
        "normalized_requirement_id",
        "regulatory_domain",
        "requirement",
        "required_action",
        "regulatory_priority_score",
        "match_score",
        "match_level"

    ]

    missing = [
        column
        for column in required_match
        if column not in matches.columns
    ]

    if missing:

        print(
            "Missing matching columns:"
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
    # [5] CLEAN DATA
    # ========================================================

    print(
        "\n[5] Cleaning data..."
    )

    risk["date"] = pd.to_datetime(
        risk["date"],
        errors="coerce"
    )

    risk["subsidiary"] = (
        risk["subsidiary"]
        .fillna("UNKNOWN")
        .astype(str)
        .str.upper()
        .str.strip()
    )

    for column in [

        "equipment_risk",
        "logistics_risk",
        "weather_risk",
        "workforce_risk",
        "overall_operational_risk",
        "governance_priority_score"

    ]:

        risk[column] = numeric(
            risk[column]
        )

    matches["subsidiary"] = (
        matches["subsidiary"]
        .fillna("UNKNOWN")
        .astype(str)
        .str.upper()
        .str.strip()
    )

    matches["match_score"] = numeric(
        matches["match_score"]
    )

    matches["regulatory_priority_score"] = numeric(
        matches["regulatory_priority_score"]
    )

    # ========================================================
    # [6] LATEST MINE PROFILE
    # ========================================================

    print(
        "\n[6] Building latest mine profiles..."
    )

    latest = (

        risk
        .sort_values("date")
        .groupby(
            "subsidiary",
            as_index=False
        )
        .tail(1)
        .copy()

    )

    latest = latest.set_index(
        "subsidiary"
    )

    print(
        f"Mines detected: {len(latest)}"
    )

    # ========================================================
    # [7] CALCULATE MINE-SPECIFIC REQUIREMENT RISK
    # ========================================================

    print(
        "\n[7] Calculating mine-specific regulatory risk..."
    )

    output_rows = []

    subsidiaries = sorted(
        matches["subsidiary"]
        .dropna()
        .unique()
    )

    for index, subsidiary in enumerate(
        subsidiaries,
        start=1
    ):

        print(
            f"  Processing {subsidiary} "
            f"({index}/{len(subsidiaries)})..."
        )

        if subsidiary not in latest.index:

            continue

        mine = latest.loc[
            subsidiary
        ]

        production_score = score_production_risk(
            mine["production_risk"]
        )

        equipment = float(
            mine["equipment_risk"]
        )

        logistics = float(
            mine["logistics_risk"]
        )

        weather = float(
            mine["weather_risk"]
        )

        workforce = float(
            mine["workforce_risk"]
        )

        operational = float(
            mine["overall_operational_risk"]
        )

        governance = float(
            mine["governance_priority_score"]
        )

        profile = {

            "production_risk_score":
                production_score,

            "equipment_risk":
                equipment,

            "logistics_risk":
                logistics,

            "weather_risk":
                weather,

            "workforce_risk":
                workforce,

            "overall_operational_risk":
                operational,

            "governance_score":
                governance

        }

        mine_matches = matches[
            matches["subsidiary"]
            == subsidiary
        ]

        # ----------------------------------------------------
        # Requirement-level analysis
        # ----------------------------------------------------

        for _, requirement in mine_matches.iterrows():

            domain = str(
                requirement[
                    "regulatory_domain"
                ]
            ).upper()

            domain_risk = calculate_domain_risk(
                domain,
                profile
            )

            keyword_score = requirement_keyword_score(
                requirement[
                    "requirement"
                ],
                domain
            )

            regulatory_priority = float(
                requirement[
                    "regulatory_priority_score"
                ]
            )

            base_match = float(
                requirement[
                    "match_score"
                ]
            )

            # ------------------------------------------------
            # Mine-specific score
            # ------------------------------------------------

            mine_specific_score = (

                base_match * 0.30

                +

                domain_risk * 0.30

                +

                regulatory_priority * 0.20

                +

                keyword_score * 0.20

            )

            mine_specific_score = float(
                np.clip(
                    mine_specific_score,
                    0,
                    100
                )
            )

            level = classify(
                mine_specific_score
            )

            action = management_action(
                level,
                requirement[
                    "required_action"
                ],
                infer_evidence_type(
                    requirement[
                        "required_action"
                    ],
                    requirement[
                        "requirement"
                    ]
                )
            )

            evidence = infer_evidence_type(
                requirement[
                    "required_action"
                ],
                requirement[
                    "requirement"
                ]
            )

            output_rows.append({

                "subsidiary":
                    subsidiary,

                "mine_date":
                    mine["date"],

                "normalized_requirement_id":
                    requirement[
                        "normalized_requirement_id"
                    ],

                "source_document":
                    requirement[
                        "source_document"
                    ],

                "page_number":
                    requirement[
                        "page_number"
                    ],

                "regulatory_domain":
                    domain,

                "requirement":
                    requirement[
                        "requirement"
                    ],

                "required_action":
                    requirement[
                        "required_action"
                    ],

                "responsible_party":
                    requirement.get(
                        "responsible_party",
                        "NOT_SPECIFIED"
                    ),

                "regulatory_priority_score":
                    regulatory_priority,

                "base_match_score":
                    base_match,

                "domain_risk_score":
                    round(
                        domain_risk,
                        2
                    ),

                "keyword_relevance_score":
                    round(
                        keyword_score,
                        2
                    ),

                "mine_specific_compliance_score":
                    round(
                        mine_specific_score,
                        2
                    ),

                "compliance_risk_level":
                    level,

                "evidence_expected":
                    evidence,

                "management_action":
                    action,

                "production_risk_score":
                    production_score,

                "equipment_risk":
                    equipment,

                "logistics_risk":
                    logistics,

                "weather_risk":
                    weather,

                "workforce_risk":
                    workforce,

                "overall_operational_risk":
                    operational,

                "governance_score":
                    governance

            })

    # ========================================================
    # [8] CREATE REQUIREMENT DATASET
    # ========================================================

    print(
        "\n[8] Creating mine-specific compliance dataset..."
    )

    result = pd.DataFrame(
        output_rows
    )

    if result.empty:

        print(
            "ERROR: No matching records were generated."
        )

        return

    result = result.sort_values(
        [
            "subsidiary",
            "mine_specific_compliance_score"
        ],
        ascending=[
            True,
            False
        ]
    ).reset_index(
        drop=True
    )

    # ========================================================
    # [9] RANK REQUIREMENTS
    # ========================================================

    print(
        "\n[9] Ranking requirements within each mine..."
    )

    result[
        "requirement_rank"
    ] = (

        result
        .groupby(
            "subsidiary"
        )[
            "mine_specific_compliance_score"
        ]
        .rank(
            method="first",
            ascending=False
        )
        .astype(int)

    )

    # ========================================================
    # [10] MINE-LEVEL AGGREGATION
    # ========================================================

    print(
        "\n[10] Building mine-level compliance profiles..."
    )

    summary = (

        result
        .groupby("subsidiary")
        .agg(

            total_applicable_requirements=(
                "normalized_requirement_id",
                "count"
            ),

            critical_requirements=(
                "compliance_risk_level",
                lambda x:
                    (x == "CRITICAL").sum()
            ),

            high_risk_requirements=(
                "compliance_risk_level",
                lambda x:
                    (x == "HIGH").sum()
            ),

            medium_risk_requirements=(
                "compliance_risk_level",
                lambda x:
                    (x == "MEDIUM").sum()
            ),

            average_compliance_score=(
                "mine_specific_compliance_score",
                "mean"
            ),

            maximum_compliance_score=(
                "mine_specific_compliance_score",
                "max"
            ),

            average_domain_risk=(
                "domain_risk_score",
                "mean"
            )

        )

        .reset_index()

    )

    # ========================================================
    # [11] MERGE MINE OPERATIONAL DATA
    # ========================================================

    print(
        "\n[11] Integrating operational risk..."
    )

    operational_columns = [

        "date",
        "production_risk",
        "overall_risk_level",
        "equipment_risk",
        "logistics_risk",
        "weather_risk",
        "workforce_risk",
        "overall_operational_risk",
        "governance_priority_score"

    ]

    operational_columns = [

        column
        for column in operational_columns
        if column in latest.columns

    ]

    operational_profile = (
        latest[
            operational_columns
        ]
        .reset_index()
        .rename(
            columns={
                "index": "subsidiary"
            }
        )
    )

    summary = summary.merge(
        operational_profile,
        on="subsidiary",
        how="left"
    )

    # ========================================================
    # [12] FINAL MINE COMPLIANCE RISK
    # ========================================================

    print(
        "\n[12] Calculating final mine compliance risk..."
    )

    summary[
        "production_risk_score"
    ] = summary[
        "production_risk"
    ].apply(
        score_production_risk
    )

    summary[
        "mine_compliance_risk_score"
    ] = (

        summary[
            "average_compliance_score"
        ] * 0.25

        +

        summary[
            "maximum_compliance_score"
        ] * 0.20

        +

        np.minimum(
            summary[
                "critical_requirements"
            ] * 12,
            100
        ) * 0.15

        +

        np.minimum(
            summary[
                "high_risk_requirements"
            ] * 5,
            100
        ) * 0.10

        +

        summary[
            "overall_operational_risk"
        ] * 0.10

        +

        summary[
            "governance_priority_score"
        ] * 0.10

        +

        summary[
            "production_risk_score"
        ] * 0.10

    )

    summary[
        "mine_compliance_risk_score"
    ] = (
        summary[
            "mine_compliance_risk_score"
        ]
        .clip(0, 100)
        .round(2)
    )

    # ========================================================
    # [13] CLASSIFICATION
    # ========================================================

    print(
        "\n[13] Classifying compliance risk..."
    )

    summary[
        "compliance_risk_level"
    ] = summary[
        "mine_compliance_risk_score"
    ].apply(
        classify
    )

    summary[
        "management_priority"
    ] = summary[
        "mine_compliance_risk_score"
    ].apply(
        priority_label
    )

    # ========================================================
    # [14] TOP DOMAIN
    # ========================================================

    print(
        "\n[14] Identifying dominant regulatory domains..."
    )

    domain_table = (

        result
        .groupby(
            [
                "subsidiary",
                "regulatory_domain"
            ]
        )[
            "mine_specific_compliance_score"
        ]
        .mean()
        .reset_index()

    )

    top_domains = (

        domain_table
        .sort_values(
            "mine_specific_compliance_score",
            ascending=False
        )
        .groupby(
            "subsidiary",
            as_index=False
        )
        .first()

    )

    top_domains = top_domains[
        [
            "subsidiary",
            "regulatory_domain",
            "mine_specific_compliance_score"
        ]
    ]

    top_domains = top_domains.rename(
        columns={
            "regulatory_domain":
                "dominant_regulatory_domain",

            "mine_specific_compliance_score":
                "dominant_domain_score"
        }
    )

    summary = summary.merge(
        top_domains,
        on="subsidiary",
        how="left"
    )

    # ========================================================
    # [15] TOP ACTION
    # ========================================================

    print(
        "\n[15] Generating mine-level actions..."
    )

    top_requirement = (

        result
        .sort_values(
            "mine_specific_compliance_score",
            ascending=False
        )
        .groupby(
            "subsidiary",
            as_index=False
        )
        .first()

    )

    top_requirement = top_requirement[
        [
            "subsidiary",
            "normalized_requirement_id",
            "regulatory_domain",
            "required_action",
            "evidence_expected",
            "management_action"
        ]
    ]

    top_requirement = top_requirement.rename(
        columns={
            "normalized_requirement_id":
                "top_priority_requirement_id",

            "regulatory_domain":
                "top_priority_domain",

            "required_action":
                "top_required_action",

            "evidence_expected":
                "top_expected_evidence",

            "management_action":
                "top_management_action"
        }
    )

    summary = summary.merge(
        top_requirement,
        on="subsidiary",
        how="left"
    )

    # ========================================================
    # [16] FINAL RANKING
    # ========================================================

    print(
        "\n[16] Ranking mines..."
    )

    summary = summary.sort_values(
        [
            "mine_compliance_risk_score",
            "critical_requirements",
            "high_risk_requirements"
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
        "compliance_rank",
        range(
            1,
            len(summary) + 1
        )
    )

    # ========================================================
    # [17] TOP ACTIONS
    # ========================================================

    print(
        "\n[17] Creating management action dataset..."
    )

    top_actions = (

        result[
            result[
                "compliance_risk_level"
            ].isin(
                [
                    "CRITICAL",
                    "HIGH",
                    "MEDIUM"
                ]
            )
        ]

        .sort_values(
            "mine_specific_compliance_score",
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
    # [18] SAVE
    # ========================================================

    print(
        "\n[18] Saving outputs..."
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    top_actions.to_csv(
        TOP_ACTIONS_FILE,
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
        "AI MINE-SPECIFIC COMPLIANCE RISK"
    )

    print(
        "=" * 70
    )

    display_columns = [

        "compliance_rank",
        "subsidiary",
        "mine_compliance_risk_score",
        "compliance_risk_level",
        "critical_requirements",
        "high_risk_requirements",
        "average_compliance_score",
        "overall_operational_risk",
        "governance_priority_score",
        "dominant_regulatory_domain",
        "management_priority"

    ]

    display_columns = [

        column
        for column in display_columns
        if column in summary.columns

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
        "TOP MINE COMPLIANCE PRIORITIES"
    )

    print(
        "=" * 70
    )

    for _, row in summary.head(5).iterrows():

        print(
            f"\n#{int(row['compliance_rank'])} "
            f"{row['subsidiary']}"
        )

        print(
            f"Compliance risk : "
            f"{row['mine_compliance_risk_score']:.2f}"
        )

        print(
            f"Risk level      : "
            f"{row['compliance_risk_level']}"
        )

        print(
            f"Critical reqs   : "
            f"{int(row['critical_requirements'])}"
        )

        print(
            f"High-risk reqs  : "
            f"{int(row['high_risk_requirements'])}"
        )

        print(
            f"Dominant domain : "
            f"{row['dominant_regulatory_domain']}"
        )

        print(
            f"Priority        : "
            f"{row['management_priority']}"
        )

        print(
            "Top requirement: "
            + str(
                row[
                    "top_priority_requirement_id"
                ]
            )
        )

        print(
            "Action          : "
            + str(
                row[
                    "top_management_action"
                ]
            )
        )

        print(
            "Evidence        : "
            + str(
                row[
                    "top_expected_evidence"
                ]
            )
        )

    # ========================================================
    # DISTRIBUTION
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "COMPLIANCE RISK DISTRIBUTION"
    )

    print(
        "=" * 70
    )

    print(
        summary[
            "compliance_risk_level"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nManagement priority distribution:"
    )

    print(
        summary[
            "management_priority"
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
        "PHASE 9D COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nMine-specific compliance analysis:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        "\nMine-level compliance summary:"
    )

    print(
        SUMMARY_FILE
    )

    print(
        "\nMine compliance actions:"
    )

    print(
        TOP_ACTIONS_FILE
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()