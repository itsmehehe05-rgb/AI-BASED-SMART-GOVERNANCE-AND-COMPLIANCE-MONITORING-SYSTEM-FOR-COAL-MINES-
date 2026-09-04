"""
CoalMineAI - Step 4.17
FINAL GOVERNANCE INTELLIGENCE

Combines:
    Operational Risk
    Early Warning
    Final ML Prediction
    SHAP Drivers
    Compliance Risk
    Evidence Gap
    Regulatory Retrieval

No model retraining.
No threshold tuning.
No conversational AI.
Missing evidence is NOT treated as non-compliance.
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(r"D:\CoalMineAI")
OUTPUTS = BASE_DIR / "outputs"
FINAL_DIR = OUTPUTS / "v5" / "FINAL"

RISK_FILE = OUTPUTS / "production_risk_analysis.csv"
EARLY_WARNING_FILE = OUTPUTS / "early_warning_analysis.csv"
ML_FILE = FINAL_DIR / "final_risk_predictions.csv"
SHAP_FILE = FINAL_DIR / "shap_latest_mine_explanations.csv"
REGULATORY_FILE = FINAL_DIR / "regulatory_retrieval_qc.csv"

COMPLIANCE_RISK_FILE = (
    OUTPUTS / "mine_specific_compliance_risk.csv"
)

COMPLIANCE_INTELLIGENCE_FILE = (
    OUTPUTS / "mine_compliance_intelligence.csv"
)

EVIDENCE_FILE = (
    OUTPUTS / "compliance_evidence_gap_analysis.csv"
)

EVIDENCE_SUMMARY_FILE = (
    OUTPUTS / "compliance_gap_mine_summary.csv"
)

COMPLIANCE_ACTION_FILE = (
    OUTPUTS / "mine_compliance_actions.csv"
)


# ============================================================
# OUTPUTS
# ============================================================

GOVERNANCE_FILE = (
    FINAL_DIR / "final_governance_intelligence.csv"
)

GOVERNANCE_SUMMARY_FILE = (
    FINAL_DIR / "final_governance_summary.csv"
)

GOVERNANCE_REGULATORY_FILE = (
    FINAL_DIR / "final_governance_regulatory_priorities.csv"
)

GOVERNANCE_SHAP_FILE = (
    FINAL_DIR / "final_governance_risk_drivers.csv"
)


# ============================================================
# HELPERS
# ============================================================

def clean(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def number(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def percent_to_100(value):
    value = number(value)

    if 0 <= value <= 1:
        return value * 100

    return value


def normalize_0_100(value):
    return max(
        0.0,
        min(
            100.0,
            number(value)
        )
    )


def find_column(df, candidates):

    lower_map = {
        str(c).lower().strip(): c
        for c in df.columns
    }

    for candidate in candidates:

        key = candidate.lower().strip()

        if key in lower_map:
            return lower_map[key]

    return None


def standardize_keys(df, name):

    """
    Make mine/date columns consistent across all sources.
    """

    df = df.copy()

    # --------------------------------------------------------
    # Mine/subsidiary
    # --------------------------------------------------------

    mine_col = find_column(
        df,
        [
            "subsidiary",
            "mine",
            "mine_name",
            "coal_mine"
        ]
    )

    if mine_col is None:

        raise ValueError(
            f"[{name}] Could not identify mine/subsidiary column."
        )

    if mine_col != "subsidiary":

        df = df.rename(
            columns={
                mine_col: "subsidiary"
            }
        )

    # --------------------------------------------------------
    # Date
    # --------------------------------------------------------

    date_col = find_column(
        df,
        [
            "record_date",
            "date",
            "month",
            "reporting_date",
            "observation_date"
        ]
    )

    if date_col is None:

        print(
            f"WARNING: [{name}] "
            f"No date column found."
        )

        df["record_date"] = pd.NaT

    else:

        if date_col != "record_date":

            df = df.rename(
                columns={
                    date_col: "record_date"
                }
            )

        df["record_date"] = pd.to_datetime(
            df["record_date"],
            errors="coerce"
        )

    return df


def latest_by_mine(df):

    if "record_date" not in df.columns:

        return (
            df
            .drop_duplicates(
                "subsidiary"
            )
        )

    # Prefer latest valid date.
    return (
        df
        .sort_values(
            "record_date",
            na_position="first"
        )
        .groupby(
            "subsidiary",
            as_index=False
        )
        .tail(1)
        .copy()
    )


def load_optional(path):

    if not path.exists():

        print(
            f"Not found: {path}"
        )

        return pd.DataFrame()

    try:

        df = pd.read_csv(path)

        print(
            f"Loaded {path.name}: {df.shape}"
        )

        return df

    except Exception as exc:

        print(
            f"WARNING: Could not load {path.name}: {exc}"
        )

        return pd.DataFrame()


def ensure_column(
    df,
    column,
    default
):

    if column not in df.columns:

        df[column] = default


# ============================================================
# START
# ============================================================

print("=" * 80)

print(
    "CoalMineAI - STEP 4.17 "
    "FINAL GOVERNANCE INTELLIGENCE"
)

print("=" * 80)


# ============================================================
# LOAD CORE DATA
# ============================================================

print("\n[1] Loading operational risk...")

risk = pd.read_csv(
    RISK_FILE
)

print(
    "Operational risk:",
    risk.shape
)


print("\n[2] Loading early warning...")

warning = pd.read_csv(
    EARLY_WARNING_FILE
)

print(
    "Early warning:",
    warning.shape
)


print("\n[3] Loading final ML predictions...")

ml = pd.read_csv(
    ML_FILE
)

print(
    "ML predictions:",
    ml.shape
)


print("\n[4] Loading SHAP explanations...")

shap = pd.read_csv(
    SHAP_FILE
)

print(
    "SHAP:",
    shap.shape
)


print("\n[5] Loading regulatory retrieval...")

regulatory = pd.read_csv(
    REGULATORY_FILE
)

print(
    "Regulatory retrieval:",
    regulatory.shape
)


# ============================================================
# LOAD OPTIONAL COMPLIANCE DATA
# ============================================================

print(
    "\n[6] Loading compliance intelligence..."
)

compliance_risk = load_optional(
    COMPLIANCE_RISK_FILE
)

compliance_intelligence = load_optional(
    COMPLIANCE_INTELLIGENCE_FILE
)

evidence = load_optional(
    EVIDENCE_FILE
)

evidence_summary = load_optional(
    EVIDENCE_SUMMARY_FILE
)

compliance_actions = load_optional(
    COMPLIANCE_ACTION_FILE
)


# ============================================================
# STANDARDIZE KEYS
# ============================================================

print(
    "\n[7] Standardizing mine/date keys..."
)

risk = standardize_keys(
    risk,
    "operational risk"
)

warning = standardize_keys(
    warning,
    "early warning"
)

ml = standardize_keys(
    ml,
    "ML predictions"
)


print(
    "Operational date range:",
    risk["record_date"].min(),
    "→",
    risk["record_date"].max()
)

print(
    "Warning date range:",
    warning["record_date"].min(),
    "→",
    warning["record_date"].max()
)

print(
    "ML date range:",
    ml["record_date"].min(),
    "→",
    ml["record_date"].max()
)


# ============================================================
# LATEST STATES
# ============================================================

print(
    "\n[8] Selecting latest state for every mine..."
)

risk_latest = latest_by_mine(
    risk
)

warning_latest = latest_by_mine(
    warning
)

ml_latest = latest_by_mine(
    ml
)


print(
    "Risk latest rows:",
    len(risk_latest)
)

print(
    "Warning latest rows:",
    len(warning_latest)
)

print(
    "ML latest rows:",
    len(ml_latest)
)


# ============================================================
# ML COLUMNS
# ============================================================

ml_probability_col = find_column(
    ml_latest,
    [
        "calibrated_escalation_probability",
        "predicted_escalation_probability",
        "escalation_probability",
        "probability"
    ]
)

ml_prediction_col = find_column(
    ml_latest,
    [
        "predicted_material_escalation",
        "predicted_escalation",
        "material_escalation_prediction"
    ]
)


print(
    "\nML probability column:",
    ml_probability_col
)

print(
    "ML prediction column:",
    ml_prediction_col
)


if ml_probability_col is None:

    raise ValueError(
        "Could not identify ML probability column."
    )


# ============================================================
# BUILD CORE
# ============================================================

print(
    "\n[9] Building core governance table..."
)

core = risk_latest.copy()


# ------------------------------------------------------------
# Add early warning using mine only
# ------------------------------------------------------------

warning_useful = [
    c
    for c in [
        "subsidiary",
        "record_date",
        "warning_level",
        "trajectory",
        "primary_driver",
        "early_warning_score"
    ]
    if c in warning_latest.columns
]


if "subsidiary" in warning_latest.columns:

    # Use latest mine-level state.
    warning_use = (
        warning_latest[
            warning_useful
        ]
        .drop_duplicates(
            "subsidiary"
        )
    )

    core = core.merge(
        warning_use,
        on="subsidiary",
        how="left",
        suffixes=(
            "",
            "_warning"
        )
    )


# ------------------------------------------------------------
# Add ML using mine only
# ------------------------------------------------------------

ml_useful = [
    c
    for c in [
        "subsidiary",
        "record_date",
        ml_probability_col,
        ml_prediction_col
    ]
    if c is not None
    and c in ml_latest.columns
]


ml_use = (
    ml_latest[
        ml_useful
    ]
    .drop_duplicates(
        "subsidiary"
    )
)


core = core.merge(
    ml_use,
    on="subsidiary",
    how="left",
    suffixes=(
        "",
        "_ml"
    )
)


# ============================================================
# NORMALIZE CORE COLUMNS
# ============================================================

ensure_column(
    core,
    "overall_operational_risk",
    0
)

ensure_column(
    core,
    "equipment_risk",
    0
)

ensure_column(
    core,
    "logistics_risk",
    0
)

ensure_column(
    core,
    "weather_risk",
    0
)

ensure_column(
    core,
    "workforce_risk",
    0
)

ensure_column(
    core,
    "warning_level",
    "UNKNOWN"
)

ensure_column(
    core,
    "trajectory",
    "UNKNOWN"
)

ensure_column(
    core,
    "primary_driver",
    ""
)

ensure_column(
    core,
    "early_warning_score",
    0
)


# ============================================================
# SHAP
# ============================================================

print(
    "\n[10] Integrating SHAP drivers..."
)

if (
    not shap.empty
    and
    "subsidiary" in shap.columns
):

    shap_useful = [
        c
        for c in [
            "subsidiary",
            "record_date",
            "driver_1",
            "driver_1_shap",
            "driver_1_direction",
            "driver_2",
            "driver_2_shap",
            "driver_2_direction",
            "driver_3",
            "driver_3_shap",
            "driver_3_direction",
            "driver_4",
            "driver_4_shap",
            "driver_4_direction",
            "driver_5",
            "driver_5_shap",
            "driver_5_direction"
        ]
        if c in shap.columns
    ]

    shap_use = (
        shap[
            shap_useful
        ]
        .drop_duplicates(
            "subsidiary"
        )
    )

    core = core.merge(
        shap_use,
        on="subsidiary",
        how="left",
        suffixes=(
            "",
            "_shap"
        )
    )


# ============================================================
# COMPLIANCE RISK
# ============================================================

print(
    "\n[11] Integrating compliance risk..."
)

ensure_column(
    core,
    "compliance_risk_score",
    0
)

ensure_column(
    core,
    "compliance_risk_level",
    "UNKNOWN"
)


if not compliance_risk.empty:

    comp_mine_col = find_column(
        compliance_risk,
        [
            "subsidiary",
            "mine",
            "mine_name"
        ]
    )

    if comp_mine_col:

        comp = compliance_risk.copy()

        comp = comp.rename(
            columns={
                comp_mine_col:
                    "subsidiary"
            }
        )

        comp_cols = [
            c
            for c in [
                "subsidiary",
                "compliance_risk_score",
                "compliance_risk_level"
            ]
            if c in comp.columns
        ]

        comp = comp[
            comp_cols
        ]

        # If file contains many requirement-level records,
        # aggregate to mine level.
        if len(comp) > comp[
            "subsidiary"
        ].nunique():

            numeric_cols = [
                c
                for c in [
                    "compliance_risk_score"
                ]
                if c in comp.columns
            ]

            aggregations = {}

            for c in numeric_cols:

                aggregations[c] = "max"

            if "compliance_risk_level" in comp.columns:

                # Highest available textual level
                def highest_level(series):

                    values = (
                        series
                        .astype(str)
                        .str.upper()
                    )

                    if values.isin(
                        [
                            "CRITICAL"
                        ]
                    ).any():

                        return "CRITICAL"

                    if values.isin(
                        [
                            "HIGH"
                        ]
                    ).any():

                        return "HIGH"

                    if values.isin(
                        [
                            "MEDIUM"
                        ]
                    ).any():

                        return "MEDIUM"

                    if values.isin(
                        [
                            "LOW"
                        ]
                    ).any():

                        return "LOW"

                    return "UNKNOWN"

                aggregations[
                    "compliance_risk_level"
                ] = highest_level

            comp = (
                comp
                .groupby(
                    "subsidiary",
                    as_index=False
                )
                .agg(
                    aggregations
                )
            )

        else:

            comp = comp.drop_duplicates(
                "subsidiary"
            )

        core = core.merge(
            comp,
            on="subsidiary",
            how="left",
            suffixes=(
                "",
                "_compliance"
            )
        )


# ============================================================
# EVIDENCE
# ============================================================

print(
    "\n[12] Integrating evidence status..."
)

ensure_column(
    core,
    "evidence_status",
    "UNKNOWN"
)

ensure_column(
    core,
    "verification_priority",
    "UNKNOWN"
)

ensure_column(
    core,
    "unknown_evidence_count",
    0
)


if not evidence_summary.empty:

    ev_mine_col = find_column(
        evidence_summary,
        [
            "subsidiary",
            "mine",
            "mine_name"
        ]
    )

    if ev_mine_col:

        ev = evidence_summary.copy()

        ev = ev.rename(
            columns={
                ev_mine_col:
                    "subsidiary"
            }
        )

        ev_cols = [
            c
            for c in [
                "subsidiary",
                "evidence_status",
                "verification_priority",
                "unknown_evidence_count"
            ]
            if c in ev.columns
        ]

        ev = (
            ev[
                ev_cols
            ]
            .drop_duplicates(
                "subsidiary"
            )
        )

        core = core.merge(
            ev,
            on="subsidiary",
            how="left",
            suffixes=(
                "",
                "_evidence"
            )
        )


# ============================================================
# REGULATORY INTELLIGENCE
# ============================================================

print(
    "\n[13] Integrating regulatory retrieval..."
)

if not regulatory.empty:

    reg = regulatory.copy()

    reg["mine"] = (
        reg["mine"]
        .astype(str)
        .str.strip()
    )

    reg_summary = (
        reg
        .groupby(
            "mine"
        )
        .agg(

            retrieved_regulations=(
                "requirement",
                "count"
            ),

            high_priority_regulations=(
                "regulatory_priority",
                lambda x:
                (
                    x.astype(str)
                    .str.upper()
                    == "HIGH"
                ).sum()
            ),

            medium_priority_regulations=(
                "regulatory_priority",
                lambda x:
                (
                    x.astype(str)
                    .str.upper()
                    == "MEDIUM"
                ).sum()
            ),

            average_retrieval_score=(
                "retrieval_score",
                "mean"
            ),

            maximum_retrieval_score=(
                "retrieval_score",
                "max"
            ),

            top_regulatory_domain=(
                "regulatory_domain",
                lambda x:
                (
                    x.mode().iloc[0]
                    if not x.mode().empty
                    else "UNKNOWN"
                )
            ),

            regulatory_domains=(
                "regulatory_domain",
                lambda x:
                ", ".join(
                    pd.unique(
                        x.astype(str)
                    )[:5]
                )
            )
        )
        .reset_index()
        .rename(
            columns={
                "mine":
                    "subsidiary"
            }
        )
    )

    core = core.merge(
        reg_summary,
        on="subsidiary",
        how="left"
    )


ensure_column(
    core,
    "retrieved_regulations",
    0
)

ensure_column(
    core,
    "high_priority_regulations",
    0
)

ensure_column(
    core,
    "medium_priority_regulations",
    0
)

ensure_column(
    core,
    "average_retrieval_score",
    0
)

ensure_column(
    core,
    "maximum_retrieval_score",
    0
)

ensure_column(
    core,
    "top_regulatory_domain",
    "UNKNOWN"
)

ensure_column(
    core,
    "regulatory_domains",
    "UNKNOWN"
)


# ============================================================
# GOVERNANCE SCORE
# ============================================================

print(
    "\n[14] Calculating governance score..."
)


def calculate_governance_score(row):

    # --------------------------------------------------------
    # Operational risk
    # --------------------------------------------------------

    operational = normalize_0_100(
        row.get(
            "overall_operational_risk",
            0
        )
    )

    # --------------------------------------------------------
    # ML escalation
    # --------------------------------------------------------

    ml_probability = percent_to_100(
        row.get(
            ml_probability_col,
            0
        )
    )

    ml_probability = normalize_0_100(
        ml_probability
    )

    # --------------------------------------------------------
    # Early warning
    # --------------------------------------------------------

    warning_level = clean(
        row.get(
            "warning_level",
            ""
        )
    ).upper()

    if warning_level == "CRITICAL":

        warning_component = 100

    elif warning_level == "EARLY_WARNING":

        warning_component = 75

    elif warning_level == "WATCH":

        warning_component = 50

    elif warning_level == "STABLE":

        warning_component = 15

    else:

        warning_component = 25

    # --------------------------------------------------------
    # Trajectory
    # --------------------------------------------------------

    trajectory = clean(
        row.get(
            "trajectory",
            ""
        )
    ).upper()

    if trajectory == "RAPIDLY_WORSENING":

        trajectory_component = 100

    elif trajectory == "WORSENING":

        trajectory_component = 75

    elif trajectory == "STABLE":

        trajectory_component = 20

    elif trajectory == "IMPROVING":

        trajectory_component = 10

    else:

        trajectory_component = 25

    # --------------------------------------------------------
    # Compliance
    # --------------------------------------------------------

    compliance = normalize_0_100(
        row.get(
            "compliance_risk_score",
            0
        )
    )

    # --------------------------------------------------------
    # Regulatory exposure
    # --------------------------------------------------------

    high_regs = number(
        row.get(
            "high_priority_regulations",
            0
        )
    )

    medium_regs = number(
        row.get(
            "medium_priority_regulations",
            0
        )
    )

    regulatory_component = min(
        100,
        high_regs * 20
        + medium_regs * 5
    )

    # --------------------------------------------------------
    # Evidence
    # --------------------------------------------------------

    evidence_status = clean(
        row.get(
            "evidence_status",
            "UNKNOWN"
        )
    ).upper()

    verification_priority = clean(
        row.get(
            "verification_priority",
            "UNKNOWN"
        )
    ).upper()

    # Unknown does not mean non-compliant.
    if evidence_status == "NON_COMPLIANT":

        evidence_component = 100

    elif evidence_status == "MISSING":

        evidence_component = 85

    elif verification_priority == "HIGH":

        evidence_component = 75

    elif verification_priority == "MEDIUM":

        evidence_component = 50

    else:

        evidence_component = 25

    # --------------------------------------------------------
    # FINAL SCORE
    # --------------------------------------------------------

    score = (

        0.25 * operational

        + 0.30 * ml_probability

        + 0.15 * warning_component

        + 0.10 * trajectory_component

        + 0.10 * compliance

        + 0.05 * regulatory_component

        + 0.05 * evidence_component
    )

    return round(
        max(
            0,
            min(
                100,
                score
            )
        ),
        2
    )


core[
    "governance_priority_score"
] = core.apply(
    calculate_governance_score,
    axis=1
)


# ============================================================
# GOVERNANCE LEVEL
# ============================================================

def governance_level(score):

    if score >= 70:

        return "HIGH"

    if score >= 45:

        return "MEDIUM"

    return "LOW"


core[
    "governance_priority_level"
] = (
    core[
        "governance_priority_score"
    ]
    .apply(
        governance_level
    )
)


# ============================================================
# STATUS
# ============================================================

def governance_status(level):

    if level == "HIGH":

        return "MANAGEMENT_REVIEW_REQUIRED"

    if level == "MEDIUM":

        return "PLANNED_REVIEW_REQUIRED"

    return "ROUTINE_MONITORING"


core[
    "governance_status"
] = (
    core[
        "governance_priority_level"
    ]
    .apply(
        governance_status
    )
)


# ============================================================
# REASON
# ============================================================

def governance_reason(row):

    reasons = []

    operational = normalize_0_100(
        row.get(
            "overall_operational_risk",
            0
        )
    )

    probability = percent_to_100(
        row.get(
            ml_probability_col,
            0
        )
    )

    warning_level = clean(
        row.get(
            "warning_level",
            ""
        )
    ).upper()

    trajectory = clean(
        row.get(
            "trajectory",
            ""
        )
    ).upper()

    driver = clean(
        row.get(
            "primary_driver",
            ""
        )
    )

    compliance_level = clean(
        row.get(
            "compliance_risk_level",
            ""
        )
    ).upper()

    if operational >= 60:

        reasons.append(
            "high operational risk"
        )

    elif operational >= 40:

        reasons.append(
            "elevated operational risk"
        )

    if probability >= 60:

        reasons.append(
            "high predicted escalation probability"
        )

    elif probability >= 40:

        reasons.append(
            "moderate predicted escalation probability"
        )

    if warning_level in {
        "CRITICAL",
        "EARLY_WARNING"
    }:

        reasons.append(
            f"{warning_level.lower().replace('_', ' ')} status"
        )

    if trajectory in {
        "WORSENING",
        "RAPIDLY_WORSENING"
    }:

        reasons.append(
            f"{trajectory.lower().replace('_', ' ')} risk trajectory"
        )

    if compliance_level in {
        "HIGH",
        "CRITICAL"
    }:

        reasons.append(
            "elevated compliance risk"
        )

    if driver:

        reasons.append(
            f"{driver.lower()} is a primary operational driver"
        )

    if not reasons:

        reasons.append(
            "no dominant high-risk indicator identified"
        )

    return "; ".join(
        reasons
    )


core[
    "governance_reason"
] = core.apply(
    governance_reason,
    axis=1
)


# ============================================================
# MANAGEMENT ACTION
# ============================================================

def management_action(row):

    level = row[
        "governance_priority_level"
    ]

    driver = clean(
        row.get(
            "primary_driver",
            ""
        )
    )

    trajectory = clean(
        row.get(
            "trajectory",
            ""
        )
    ).upper()

    warning = clean(
        row.get(
            "warning_level",
            ""
        )
    ).upper()

    if level == "HIGH":

        action = (
            "Immediate management review; "
            "investigate the dominant operational driver; "
            "review applicable statutory requirements; "
            "and verify relevant compliance evidence."
        )

    elif level == "MEDIUM":

        action = (
            "Schedule planned management review; "
            "monitor the identified risk trend; "
            "review applicable statutory requirements; "
            "and complete compliance verification."
        )

    else:

        action = (
            "Continue routine monitoring and "
            "periodic statutory compliance verification."
        )

    if driver:

        action += (
            f" Primary operational focus: {driver}."
        )

    if trajectory == "RAPIDLY_WORSENING":

        action += (
            " Risk trajectory is rapidly worsening; "
            "prioritize reassessment."
        )

    if warning == "EARLY_WARNING":

        action += (
            " Early-warning status requires closer monitoring."
        )

    return action


core[
    "recommended_management_action"
] = core.apply(
    management_action,
    axis=1
)


# ============================================================
# VERIFICATION ACTION
# ============================================================

def verification_action(row):

    status = clean(
        row.get(
            "evidence_status",
            "UNKNOWN"
        )
    ).upper()

    priority = clean(
        row.get(
            "verification_priority",
            "UNKNOWN"
        )
    ).upper()

    if status == "NON_COMPLIANT":

        return (
            "Escalate the identified non-compliance "
            "for corrective action."
        )

    if status == "MISSING":

        return (
            "Obtain the required compliance evidence "
            "and verify it against the applicable requirement."
        )

    if status in {
        "",
        "UNKNOWN",
        "NOT_AVAILABLE"
    }:

        return (
            "Evidence is unavailable in the current dataset; "
            "verification is required. Absence of evidence "
            "does not establish non-compliance."
        )

    if priority == "HIGH":

        return (
            "Perform priority verification of available "
            "compliance evidence."
        )

    return (
        "Verify evidence validity, applicability and recency."
    )


core[
    "verification_action"
] = core.apply(
    verification_action,
    axis=1
)


# ============================================================
# REGULATORY ATTENTION
# ============================================================

def regulatory_attention(row):

    high = number(
        row.get(
            "high_priority_regulations",
            0
        )
    )

    medium = number(
        row.get(
            "medium_priority_regulations",
            0
        )
    )

    if high > 0:

        return "HIGH"

    if medium > 0:

        return "MEDIUM"

    return "LOW"


core[
    "regulatory_attention_level"
] = core.apply(
    regulatory_attention,
    axis=1
)


# ============================================================
# SELECT FINAL COLUMNS
# ============================================================

final_columns = [

    "subsidiary",
    "record_date",

    "overall_operational_risk",
    "operational_risk_level",

    ml_probability_col,
    ml_prediction_col,

    "warning_level",
    "trajectory",
    "early_warning_score",
    "primary_driver",

    "equipment_risk",
    "logistics_risk",
    "weather_risk",
    "workforce_risk",

    "compliance_risk_score",
    "compliance_risk_level",

    "retrieved_regulations",
    "high_priority_regulations",
    "medium_priority_regulations",
    "top_regulatory_domain",
    "regulatory_domains",
    "average_retrieval_score",

    "evidence_status",
    "verification_priority",
    "unknown_evidence_count",

    "regulatory_attention_level",

    "governance_priority_score",
    "governance_priority_level",
    "governance_status",

    "governance_reason",

    "recommended_management_action",

    "verification_action"
]


final_columns = [
    c
    for c in final_columns
    if c in core.columns
]


final_df = (
    core[
        final_columns
    ]
    .sort_values(
        "governance_priority_score",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# SAVE GOVERNANCE OUTPUT
# ============================================================

print(
    "\n[15] Saving final governance output..."
)

final_df.to_csv(
    GOVERNANCE_FILE,
    index=False
)


# ============================================================
# GOVERNANCE SUMMARY
# ============================================================

summary = (
    final_df
    .groupby(
        "governance_priority_level"
    )
    .size()
    .reset_index(
        name="mine_count"
    )
)


summary.to_csv(
    GOVERNANCE_SUMMARY_FILE,
    index=False
)


# ============================================================
# SAVE REGULATORY DETAIL
# ============================================================

regulatory.to_csv(
    GOVERNANCE_REGULATORY_FILE,
    index=False
)


# ============================================================
# SAVE SHAP DETAIL
# ============================================================

if not shap.empty:

    shap.to_csv(
        GOVERNANCE_SHAP_FILE,
        index=False
    )


# ============================================================
# REPORT
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "STEP 4.17 COMPLETED"
)

print(
    "=" * 80
)

print(
    "\nMine governance priorities:"
)

print(
    final_df[
        [
            "subsidiary",
            "governance_priority_score",
            "governance_priority_level",
            "governance_status"
        ]
    ].to_string(
        index=False
    )
)


print(
    "\nGovernance distribution:"
)

print(
    summary.to_string(
        index=False
    )
)


print(
    "\nSaved:"
)

print(
    GOVERNANCE_FILE
)

print(
    GOVERNANCE_SUMMARY_FILE
)

print(
    GOVERNANCE_REGULATORY_FILE
)

print(
    GOVERNANCE_SHAP_FILE
)


print(
    "\nImportant:"
)

print(
    "Unknown/missing evidence is NOT treated as non-compliance."
)

print(
    "Predictive model was NOT retrained or modified."
)

print(
    "Conversational AI was NOT implemented."
)

print(
    "\nSTATUS: PASS"
)