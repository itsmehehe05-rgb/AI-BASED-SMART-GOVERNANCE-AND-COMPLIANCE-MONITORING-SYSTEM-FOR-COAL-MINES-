from pathlib import Path
import pandas as pd
import numpy as np
import re


# ============================================================
# PHASE 9C — AI REGULATION-TO-MINE INTELLIGENT MATCHING
# ============================================================

BASE_DIR = Path(r"D:\CoalMineAI")

REGULATORY_FILE = (
    BASE_DIR
    / "outputs"
    / "normalized_regulatory_requirements.csv"
)

RISK_FILE = (
    BASE_DIR
    / "outputs"
    / "production_risk_analysis.csv"
)

OUTPUT_DIR = BASE_DIR / "outputs"

MATCHING_FILE = (
    OUTPUT_DIR
    / "regulation_mine_matching.csv"
)

MINE_SUMMARY_FILE = (
    OUTPUT_DIR
    / "mine_regulatory_summary.csv"
)

TOP_REQUIREMENTS_FILE = (
    OUTPUT_DIR
    / "mine_top_regulatory_requirements.csv"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# TEXT UTILITIES
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


def tokenize(text):

    return set(
        clean_text(text).split()
    )


# ============================================================
# KEYWORD GROUPS
# ============================================================

DOMAIN_KEYWORDS = {

    "SAFETY": {
        "mine",
        "worker",
        "workman",
        "safety",
        "accident",
        "danger",
        "hazard",
        "protective",
        "helmet",
        "inspection",
        "ventilation",
        "shaft",
        "underground",
        "occupational"
    },

    "ENVIRONMENT": {
        "environment",
        "pollution",
        "air",
        "water",
        "emission",
        "dust",
        "noise",
        "rainfall",
        "effluent",
        "monitoring",
        "environmental",
        "clearance"
    },

    "WASTE": {
        "waste",
        "ash",
        "overburden",
        "dump",
        "disposal",
        "solid",
        "hazardous",
        "storage",
        "recycling",
        "residue"
    },

    "MINING": {
        "mining",
        "coal",
        "production",
        "excavation",
        "quarry",
        "pit",
        "drilling",
        "blasting",
        "haul",
        "mine",
        "mineral"
    },

    "COMPLIANCE": {
        "compliance",
        "report",
        "return",
        "record",
        "register",
        "inspection",
        "authority",
        "approval",
        "permit",
        "licence",
        "license"
    }
}


# ============================================================
# OPERATIONAL RISK → REGULATORY DOMAIN
# ============================================================

DOMAIN_RISK_MAPPING = {

    "SAFETY": [
        "equipment_risk",
        "workforce_risk"
    ],

    "ENVIRONMENT": [
        "weather_risk"
    ],

    "WASTE": [
        "weather_risk"
    ],

    "MINING": [
        "production_risk_score",
        "equipment_risk"
    ],

    "COMPLIANCE": [
        "governance_score"
    ]
}


# ============================================================
# NORMALIZATION HELPERS
# ============================================================

def normalize_domain(value):

    value = str(value).upper().strip()

    if value in DOMAIN_KEYWORDS:
        return value

    if value == "GENERAL":
        return "COMPLIANCE"

    return value


def severity_score(value):

    value = str(value).upper()

    return {
        "LOW": 25,
        "MEDIUM": 55,
        "HIGH": 90
    }.get(value, 25)


def relevance_score(value):

    value = str(value).upper()

    return {
        "LOW": 20,
        "MEDIUM": 60,
        "HIGH": 90
    }.get(value, 20)


def actionability_score(value):

    value = str(value).upper()

    return {
        "LOW": 20,
        "MEDIUM": 60,
        "HIGH": 90
    }.get(value, 20)


# ============================================================
# KEYWORD MATCH SCORE
# ============================================================

def keyword_similarity(
    requirement_text,
    keyword_set
):

    requirement_tokens = tokenize(
        requirement_text
    )

    if not requirement_tokens:
        return 0

    overlap = (
        requirement_tokens
        & keyword_set
    )

    score = (
        len(overlap)
        /
        max(
            len(keyword_set),
            1
        )
    )

    return min(
        score * 100,
        100
    )


# ============================================================
# DOMAIN MATCH SCORE
# ============================================================

def domain_match_score(
    regulatory_domain,
    operational_profile
):

    domain = normalize_domain(
        regulatory_domain
    )

    if domain not in DOMAIN_RISK_MAPPING:

        return 25

    relevant_risks = (
        DOMAIN_RISK_MAPPING[domain]
    )

    values = []

    for risk_name in relevant_risks:

        if risk_name in operational_profile:

            values.append(
                operational_profile[
                    risk_name
                ]
            )

    if not values:

        return 25

    return float(
        np.mean(values)
    )


# ============================================================
# REQUIREMENT MATCH SCORE
# ============================================================

def calculate_match_score(
    requirement,
    operational_profile
):

    domain = normalize_domain(
        requirement[
            "regulatory_domain"
        ]
    )

    regulatory_priority = severity_score(
        requirement[
            "regulatory_priority"
        ]
    )

    regulatory_score = float(
        requirement[
            "regulatory_priority_score"
        ]
    )

    mine_relevance = relevance_score(
        requirement[
            "mine_relevance"
        ]
    )

    actionability = actionability_score(
        requirement[
            "actionability"
        ]
    )

    text = requirement[
        "requirement"
    ]

    # --------------------------------------------------------
    # Domain relevance
    # --------------------------------------------------------

    domain_score = domain_match_score(
        domain,
        operational_profile
    )

    # --------------------------------------------------------
    # Keyword relevance
    # --------------------------------------------------------

    keywords = DOMAIN_KEYWORDS.get(
        domain,
        set()
    )

    keyword_score = keyword_similarity(
        text,
        keywords
    )

    # --------------------------------------------------------
    # Regulatory importance
    # --------------------------------------------------------

    importance_score = (

        regulatory_score * 0.40

        +

        regulatory_priority * 0.20

        +

        mine_relevance * 0.20

        +

        actionability * 0.20

    )

    # --------------------------------------------------------
    # Final intelligent match
    # --------------------------------------------------------

    match_score = (

        domain_score * 0.30

        +

        keyword_score * 0.20

        +

        importance_score * 0.50

    )

    return round(
        float(
            np.clip(
                match_score,
                0,
                100
            )
        ),
        2
    )


# ============================================================
# MATCH LEVEL
# ============================================================

def classify_match(score):

    if score >= 70:
        return "HIGH"

    if score >= 45:
        return "MEDIUM"

    return "LOW"


# ============================================================
# WHY MATCHED
# ============================================================

def generate_match_reason(
    requirement,
    profile,
    score
):

    domain = normalize_domain(
        requirement[
            "regulatory_domain"
        ]
    )

    reasons = []

    if score >= 70:

        reasons.append(
            "high regulatory relevance"
        )

    if domain in [
        "SAFETY",
        "ENVIRONMENT",
        "WASTE",
        "MINING"
    ]:

        reasons.append(
            f"{domain.lower()} relevance"
        )

    if profile.get(
        "overall_operational_risk",
        0
    ) >= 50:

        reasons.append(
            "elevated operational risk"
        )

    if profile.get(
        "production_risk_score",
        0
    ) >= 60:

        reasons.append(
            "elevated production risk"
        )

    if profile.get(
        "governance_score",
        0
    ) >= 50:

        reasons.append(
            "elevated governance priority"
        )

    if not reasons:

        return (
            "Regulation is relevant based on "
            "its regulatory domain and requirement characteristics."
        )

    return (
        "Matched because of "
        + ", ".join(reasons)
        + "."
    )


# ============================================================
# MANAGEMENT ACTION
# ============================================================

def management_action(
    requirement,
    match_score,
    match_level
):

    action = str(
        requirement.get(
            "required_action",
            "OTHER"
        )
    ).upper()

    if match_level == "HIGH":

        if action == "MONITOR":

            return (
                "Verify monitoring activity, "
                "frequency and supporting records."
            )

        if action == "REPORT":

            return (
                "Verify required regulatory reporting "
                "and supporting evidence."
            )

        if action == "SUBMIT":

            return (
                "Verify submission status, deadline "
                "and supporting documentation."
            )

        if action == "MAINTAIN":

            return (
                "Verify required records are maintained "
                "and available for inspection."
            )

        if action == "PROHIBIT":

            return (
                "Immediately verify that the prohibited "
                "activity is controlled."
            )

        if action == "ENSURE":

            return (
                "Verify implementation of the required "
                "control or obligation."
            )

        return (
            "Conduct detailed compliance review "
            "against this requirement."
        )

    if match_level == "MEDIUM":

        return (
            "Review applicability and verify "
            "available compliance evidence."
        )

    return (
        "Retain for reference and periodic review."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "PHASE 9C — AI REGULATION-TO-MINE "
        "INTELLIGENT MATCHING"
    )
    print("=" * 70)

    # ========================================================
    # [1] LOAD REGULATIONS
    # ========================================================

    print(
        "\n[1] Loading normalized regulations..."
    )

    if not REGULATORY_FILE.exists():

        print(
            "ERROR: Regulatory file not found:"
        )

        print(
            REGULATORY_FILE
        )

        return

    regulatory = pd.read_csv(
        REGULATORY_FILE,
        encoding="utf-8-sig"
    )

    print(
        f"Regulatory requirements: "
        f"{len(regulatory)}"
    )

    # ========================================================
    # [2] LOAD RISK DATA
    # ========================================================

    print(
        "\n[2] Loading mine risk data..."
    )

    if not RISK_FILE.exists():

        print(
            "ERROR: Risk file not found:"
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
    # [3] CLEAN REGULATIONS
    # ========================================================

    print(
        "\n[3] Cleaning regulatory data..."
    )

    regulatory[
        "regulatory_domain"
    ] = (
        regulatory[
            "regulatory_domain"
        ]
        .fillna("GENERAL")
        .astype(str)
        .str.upper()
        .str.strip()
    )

    regulatory[
        "requirement"
    ] = (
        regulatory[
            "requirement"
        ]
        .fillna("")
        .astype(str)
    )

    regulatory[
        "regulatory_priority_score"
    ] = pd.to_numeric(
        regulatory[
            "regulatory_priority_score"
        ],
        errors="coerce"
    ).fillna(0)

    # ========================================================
    # [4] CLEAN RISK DATA
    # ========================================================

    print(
        "\n[4] Cleaning mine risk data..."
    )

    risk[
        "date"
    ] = pd.to_datetime(
        risk["date"],
        errors="coerce"
    )

    risk[
        "subsidiary"
    ] = (
        risk[
            "subsidiary"
        ]
        .fillna("UNKNOWN")
        .astype(str)
        .str.upper()
        .str.strip()
    )

    risk_numeric_columns = [

        "equipment_risk",
        "logistics_risk",
        "weather_risk",
        "workforce_risk",
        "overall_operational_risk",
        "governance_priority_score"

    ]

    for column in risk_numeric_columns:

        if column in risk.columns:

            risk[column] = pd.to_numeric(
                risk[column],
                errors="coerce"
            ).fillna(0)

    # ========================================================
    # [5] CREATE MINE PROFILES
    # ========================================================

    print(
        "\n[5] Building latest mine profiles..."
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

    latest = latest.sort_values(
        "subsidiary"
    )

    print(
        f"Mines detected: "
        f"{len(latest)}"
    )

    # ========================================================
    # [6] PREPARE PROFILES
    # ========================================================

    print(
        "\n[6] Preparing operational profiles..."
    )

    profiles = {}

    production_risk_map = {

        "LOW": 25,
        "MEDIUM": 55,
        "HIGH": 85,
        "UNRELIABLE": 90

    }

    for _, row in latest.iterrows():

        production_risk = str(
            row.get(
                "production_risk",
                "LOW"
            )
        ).upper()

        profile = {

            "equipment_risk":
                float(
                    row.get(
                        "equipment_risk",
                        0
                    )
                ),

            "logistics_risk":
                float(
                    row.get(
                        "logistics_risk",
                        0
                    )
                ),

            "weather_risk":
                float(
                    row.get(
                        "weather_risk",
                        0
                    )
                ),

            "workforce_risk":
                float(
                    row.get(
                        "workforce_risk",
                        0
                    )
                ),

            "overall_operational_risk":
                float(
                    row.get(
                        "overall_operational_risk",
                        0
                    )
                ),

            "governance_score":
                float(
                    row.get(
                        "governance_priority_score",
                        0
                    )
                ),

            "production_risk_score":
                production_risk_map.get(
                    production_risk,
                    25
                )

        }

        profiles[
            row["subsidiary"]
        ] = profile

    # ========================================================
    # [7] MATCH REGULATIONS
    # ========================================================

    print(
        "\n[7] Matching regulations to mines..."
    )

    results = []

    total_mines = len(profiles)

    for mine_index, (
        subsidiary,
        profile
    ) in enumerate(
        profiles.items(),
        start=1
    ):

        print(
            f"  Processing {subsidiary} "
            f"({mine_index}/{total_mines})..."
        )

        for _, requirement in regulatory.iterrows():

            score = calculate_match_score(
                requirement,
                profile
            )

            level = classify_match(
                score
            )

            reason = generate_match_reason(
                requirement,
                profile,
                score
            )

            action = management_action(
                requirement,
                score,
                level
            )

            results.append({

                "subsidiary":
                    subsidiary,

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
                    requirement[
                        "regulatory_domain"
                    ],

                "requirement":
                    requirement[
                        "requirement"
                    ],

                "required_action":
                    requirement.get(
                        "required_action",
                        "OTHER"
                    ),

                "responsible_party":
                    requirement.get(
                        "responsible_party",
                        "NOT_SPECIFIED"
                    ),

                "normalized_severity":
                    requirement.get(
                        "normalized_severity",
                        "MEDIUM"
                    ),

                "mine_relevance":
                    requirement.get(
                        "mine_relevance",
                        "MEDIUM"
                    ),

                "actionability":
                    requirement.get(
                        "actionability",
                        "LOW"
                    ),

                "regulatory_priority_score":
                    requirement[
                        "regulatory_priority_score"
                    ],

                "match_score":
                    score,

                "match_level":
                    level,

                "match_reason":
                    reason,

                "recommended_action":
                    action

            })

    # ========================================================
    # [8] DATAFRAME
    # ========================================================

    print(
        "\n[8] Creating matching dataset..."
    )

    matches = pd.DataFrame(
        results
    )

    matches = matches.sort_values(
        [
            "subsidiary",
            "match_score"
        ],
        ascending=[
            True,
            False
        ]
    ).reset_index(
        drop=True
    )

    # ========================================================
    # [9] RANK REQUIREMENTS WITHIN EACH MINE
    # ========================================================

    print(
        "\n[9] Ranking requirements for each mine..."
    )

    matches[
        "mine_requirement_rank"
    ] = (
        matches
        .groupby("subsidiary")
        ["match_score"]
        .rank(
            method="first",
            ascending=False
        )
        .astype(int)
    )

    # ========================================================
    # [10] MINE REGULATORY SCORE
    # ========================================================

    print(
        "\n[10] Calculating mine-level regulatory exposure..."
    )

    mine_summary = (

        matches
        .groupby("subsidiary")
        .agg(

            total_regulations=(
                "normalized_requirement_id",
                "count"
            ),

            high_match_requirements=(
                "match_level",
                lambda x:
                    (x == "HIGH").sum()
            ),

            medium_match_requirements=(
                "match_level",
                lambda x:
                    (x == "MEDIUM").sum()
            ),

            average_match_score=(
                "match_score",
                "mean"
            ),

            maximum_match_score=(
                "match_score",
                "max"
            ),

            average_regulatory_priority=(
                "regulatory_priority_score",
                "mean"
            )

        )
        .reset_index()
    )

    # ========================================================
    # [11] DOMAIN SUMMARY
    # ========================================================

    print(
        "\n[11] Calculating domain-specific exposure..."
    )

    domain_summary = (

        matches[
            matches["match_level"]
            .isin(
                [
                    "HIGH",
                    "MEDIUM"
                ]
            )
        ]

        .groupby(
            [
                "subsidiary",
                "regulatory_domain"
            ]
        )

        .agg(
            domain_average_score=(
                "match_score",
                "mean"
            ),

            domain_high_count=(
                "match_level",
                lambda x:
                    (x == "HIGH").sum()
            )

        )

        .reset_index()
    )

    domain_pivot = domain_summary.pivot(
        index="subsidiary",
        columns="regulatory_domain",
        values="domain_average_score"
    ).reset_index()

    domain_pivot.columns = [

        (
            f"{col}_regulatory_score"
            if col != "subsidiary"
            else col
        )

        for col in domain_pivot.columns

    ]

    # ========================================================
    # [12] ADD OPERATIONAL PROFILE
    # ========================================================

    print(
        "\n[12] Combining operational and regulatory risk..."
    )

    operational_columns = [

        "subsidiary",
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

        col
        for col in operational_columns
        if col in latest.columns

    ]

    operational_profile = latest[
        operational_columns
    ].copy()

    mine_summary = mine_summary.merge(
        operational_profile,
        on="subsidiary",
        how="left"
    )

    mine_summary = mine_summary.merge(
        domain_pivot,
        on="subsidiary",
        how="left"
    )

    # ========================================================
    # [13] FINAL COMPLIANCE EXPOSURE
    # ========================================================

    print(
        "\n[13] Calculating final regulatory exposure score..."
    )

    mine_summary[
        "regulatory_exposure_score"
    ] = (

        mine_summary[
            "average_match_score"
        ] * 0.45

        +

        mine_summary[
            "maximum_match_score"
        ] * 0.20

        +

        np.minimum(
            mine_summary[
                "high_match_requirements"
            ] * 5,
            100
        ) * 0.20

        +

        mine_summary[
            "governance_priority_score"
        ] * 0.15

    )

    mine_summary[
        "regulatory_exposure_score"
    ] = (
        mine_summary[
            "regulatory_exposure_score"
        ]
        .clip(0, 100)
        .round(2)
    )

    # ========================================================
    # [14] RISK LEVEL
    # ========================================================

    mine_summary[
        "regulatory_risk_level"
    ] = (
        mine_summary[
            "regulatory_exposure_score"
        ]
        .apply(
            lambda x:
                "HIGH"
                if x >= 70
                else
                "MEDIUM"
                if x >= 45
                else
                "LOW"
        )
    )

    # ========================================================
    # [15] TOP DOMAIN
    # ========================================================

    domain_score_columns = [

        col
        for col in mine_summary.columns
        if col.endswith(
            "_regulatory_score"
        )
        and col != "regulatory_exposure_score"

    ]

    def get_top_domain(row):

        values = {}

        for column in domain_score_columns:

            value = row[column]

            if pd.notna(value):

                domain = column.replace(
                    "_regulatory_score",
                    ""
                )

                values[domain] = value

        if not values:

            return "GENERAL"

        return max(
            values,
            key=values.get
        )

    mine_summary[
        "top_regulatory_domain"
    ] = mine_summary.apply(
        get_top_domain,
        axis=1
    )

    # ========================================================
    # [16] MANAGEMENT PRIORITY
    # ========================================================

    def management_priority(row):

        score = row[
            "regulatory_exposure_score"
        ]

        if score >= 70:
            return "IMMEDIATE_REVIEW"

        if score >= 45:
            return "ENHANCED_MONITORING"

        return "ROUTINE_MONITORING"

    mine_summary[
        "management_priority"
    ] = mine_summary.apply(
        management_priority,
        axis=1
    )

    # ========================================================
    # [17] SORT
    # ========================================================

    mine_summary = mine_summary.sort_values(
        "regulatory_exposure_score",
        ascending=False
    ).reset_index(
        drop=True
    )

    mine_summary.insert(
        0,
        "regulatory_rank",
        range(
            1,
            len(mine_summary) + 1
        )
    )

    # ========================================================
    # [18] TOP REQUIREMENTS
    # ========================================================

    print(
        "\n[14] Extracting top regulatory requirements..."
    )

    top_requirements = (

        matches
        .sort_values(
            [
                "subsidiary",
                "match_score"
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

        .head(10)

        .copy()
    )

    # ========================================================
    # [19] SAVE
    # ========================================================

    print(
        "\n[15] Saving outputs..."
    )

    matches.to_csv(
        MATCHING_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    mine_summary.to_csv(
        MINE_SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    top_requirements.to_csv(
        TOP_REQUIREMENTS_FILE,
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
        "AI REGULATION-TO-MINE MATCHING RESULTS"
    )

    print(
        "=" * 70
    )

    display_columns = [

        "regulatory_rank",
        "subsidiary",
        "regulatory_exposure_score",
        "regulatory_risk_level",
        "high_match_requirements",
        "medium_match_requirements",
        "average_match_score",
        "maximum_match_score",
        "top_regulatory_domain",
        "management_priority"

    ]

    print(
        mine_summary[
            display_columns
        ]
        .to_string(
            index=False
        )
    )

    # ========================================================
    # TOP REQUIREMENTS
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "TOP REGULATORY REQUIREMENTS BY MINE"
    )

    print(
        "=" * 70
    )

    for subsidiary in profiles:

        print(
            f"\n{subsidiary}"
        )

        mine_top = (
            top_requirements[
                top_requirements[
                    "subsidiary"
                ]
                == subsidiary
            ]
            .head(5)
        )

        for _, row in mine_top.iterrows():

            print(
                f"  {row['normalized_requirement_id']} "
                f"| {row['regulatory_domain']} "
                f"| Score: {row['match_score']:.2f} "
                f"| {row['match_level']}"
            )

    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "REGULATORY RISK DISTRIBUTION"
    )

    print(
        "=" * 70
    )

    print(
        mine_summary[
            "regulatory_risk_level"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nManagement priorities:"
    )

    print(
        mine_summary[
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
        "PHASE 9C COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nFull regulation-to-mine matching:"
    )

    print(
        MATCHING_FILE
    )

    print(
        "\nMine regulatory summary:"
    )

    print(
        MINE_SUMMARY_FILE
    )

    print(
        "\nTop regulatory requirements:"
    )

    print(
        TOP_REQUIREMENTS_FILE
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()