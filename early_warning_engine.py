import os
import numpy as np
import pandas as pd


# ============================================================
# PHASE 7B — AI EARLY WARNING ENGINE — VERSION 3
# ============================================================
#
# Purpose:
# Detect deterioration BEFORE a mine reaches a critical state.
#
# Uses:
#   - Current operational risk
#   - Risk change
#   - Risk acceleration
#   - 3-month deterioration
#   - Production risk
#   - Target achievement
#   - Equipment risk
#   - Logistics risk
#   - Weather risk
#   - Workforce risk
#   - Consecutive increases
#
# Outputs:
#   - Early-warning score
#   - Warning level
#   - Risk trajectory
#   - Primary risk driver
#   - Warning reason
#   - Management recommendation
#
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

RISK_INPUT_FILE = (
    r"D:\CoalMineAI\outputs\production_risk_analysis.csv"
)

GOVERNANCE_INPUT_FILE = (
    r"D:\CoalMineAI\outputs\multi_mine_governance_ranking.csv"
)

OUTPUT_DIR = (
    r"D:\CoalMineAI\outputs"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "early_warning_analysis.csv"
)

MINE_SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "early_warning_mine_summary.csv"
)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def safe_float(value, default=0.0):

    try:

        value = float(value)

        if np.isfinite(value):

            return value

    except (TypeError, ValueError):

        pass

    return default


def clamp(
    value,
    minimum=0,
    maximum=100
):

    return float(
        np.clip(
            safe_float(value),
            minimum,
            maximum
        )
    )


def numeric_series(
    df,
    column,
    default=0
):

    if column not in df.columns:

        return pd.Series(
            default,
            index=df.index,
            dtype=float
        )

    return pd.to_numeric(
        df[column],
        errors="coerce"
    ).fillna(
        default
    )


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print(
        "\n[1] Loading risk-engine output..."
    )

    # --------------------------------------------------------
    # PRIMARY SOURCE:
    # Version 3 risk engine
    # --------------------------------------------------------

    if os.path.exists(
        RISK_INPUT_FILE
    ):

        print(
            "Using Version 3 risk-engine output:"
        )

        print(
            RISK_INPUT_FILE
        )

        df = pd.read_csv(
            RISK_INPUT_FILE
        )

        print(
            f"Rows loaded: {len(df)}"
        )

        print(
            f"Columns loaded: {len(df.columns)}"
        )

        return df

    # --------------------------------------------------------
    # FALLBACK:
    # Governance ranking
    # --------------------------------------------------------

    if os.path.exists(
        GOVERNANCE_INPUT_FILE
    ):

        print(
            "Version 3 risk-engine output not found."
        )

        print(
            "Using governance ranking as fallback:"
        )

        print(
            GOVERNANCE_INPUT_FILE
        )

        df = pd.read_csv(
            GOVERNANCE_INPUT_FILE
        )

        print(
            f"Rows loaded: {len(df)}"
        )

        return df

    raise FileNotFoundError(
        "\nNo suitable input file found.\n"
        f"Expected:\n"
        f" - {RISK_INPUT_FILE}\n"
        f" - {GOVERNANCE_INPUT_FILE}"
    )


# ============================================================
# NORMALIZE COLUMNS
# ============================================================

def normalize_columns(df):

    print(
        "\n[2] Normalizing input columns..."
    )

    df = df.copy()

    # --------------------------------------------------------
    # Governance score
    # --------------------------------------------------------

    if "governance_priority_score" in df.columns:

        df["governance_score"] = pd.to_numeric(
            df[
                "governance_priority_score"
            ],
            errors="coerce"
        )

    elif "governance_score" in df.columns:

        df["governance_score"] = pd.to_numeric(
            df[
                "governance_score"
            ],
            errors="coerce"
        )

    else:

        df["governance_score"] = 0.0

    # --------------------------------------------------------
    # Version 3 direct risk-change fields
    # --------------------------------------------------------

    if "risk_change" in df.columns:

        df[
            "risk_change_signal"
        ] = pd.to_numeric(
            df[
                "risk_change"
            ],
            errors="coerce"
        )

    if "risk_acceleration" in df.columns:

        df[
            "risk_acceleration_signal"
        ] = pd.to_numeric(
            df[
                "risk_acceleration"
            ],
            errors="coerce"
        )

    return df


# ============================================================
# VALIDATE
# ============================================================

def validate_columns(df):

    print(
        "\n[3] Validating early-warning inputs..."
    )

    required_columns = [

        "date",

        "subsidiary",

        "production_risk",

        "overall_operational_risk",

        "predicted_target_achievement_pct"
    ]

    missing = [

        column

        for column in required_columns

        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "\nMissing required columns:\n"
            +
            "\n".join(
                f" - {column}"
                for column in missing
            )
        )

    print(
        "\nWarning signals available:"
    )

    signals = [

        "risk_change",

        "risk_acceleration",

        "equipment_risk",

        "logistics_risk",

        "weather_risk",

        "workforce_risk",

        "governance_priority_score"
    ]

    for signal in signals:

        if signal in df.columns:

            print(
                f"  ✓ {signal}"
            )

        else:

            print(
                f"  - {signal}"
            )

    return df


# ============================================================
# CLEAN DATA
# ============================================================

def clean_data(df):

    print(
        "\n[4] Cleaning early-warning data..."
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

    numeric_columns = [

        "governance_score",

        "equipment_risk",

        "logistics_risk",

        "weather_risk",

        "workforce_risk",

        "overall_operational_risk",

        "predicted_target_achievement_pct",

        "risk_change",

        "risk_acceleration",

        "risk_change_signal",

        "risk_acceleration_signal"
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    df = df.dropna(
        subset=[
            "date",
            "subsidiary"
        ]
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
        f"Subsidiaries: "
        f"{df['subsidiary'].nunique()}"
    )

    return df


# ============================================================
# PRODUCTION RISK NUMERIC
# ============================================================

def production_risk_numeric(value):

    mapping = {

        "LOW": 20,

        "MEDIUM": 55,

        "HIGH": 85,

        "UNRELIABLE": 100
    }

    return mapping.get(
        str(value)
        .upper()
        .strip(),
        50
    )


# ============================================================
# TRAJECTORY ANALYSIS
# ============================================================

def calculate_trajectory(group):

    group = group.sort_values(
        "date"
    ).copy()

    # --------------------------------------------------------
    # Production risk
    # --------------------------------------------------------

    group[
        "production_risk_numeric"
    ] = group[
        "production_risk"
    ].apply(
        production_risk_numeric
    )

    # --------------------------------------------------------
    # Operational risk
    # --------------------------------------------------------

    operational = pd.to_numeric(
        group[
            "overall_operational_risk"
        ],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Governance
    # --------------------------------------------------------

    governance = pd.to_numeric(
        group[
            "governance_score"
        ],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Target achievement
    # --------------------------------------------------------

    target = pd.to_numeric(
        group[
            "predicted_target_achievement_pct"
        ],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Calculate local changes
    # --------------------------------------------------------

    calculated_change = (
        operational.diff()
    )

    calculated_acceleration = (
        calculated_change.diff()
    )

    # --------------------------------------------------------
    # Prefer Version 3 direct values
    # --------------------------------------------------------

    if "risk_change_signal" in group.columns:

        direct_change = pd.to_numeric(
            group[
                "risk_change_signal"
            ],
            errors="coerce"
        )

    else:

        direct_change = calculated_change

    if "risk_acceleration_signal" in group.columns:

        direct_acceleration = pd.to_numeric(
            group[
                "risk_acceleration_signal"
            ],
            errors="coerce"
        )

    else:

        direct_acceleration = (
            calculated_acceleration
        )

    # Fill only missing values
    direct_change = (
        direct_change
        .fillna(
            calculated_change
        )
        .fillna(0)
    )

    direct_acceleration = (
        direct_acceleration
        .fillna(
            calculated_acceleration
        )
        .fillna(0)
    )

    # --------------------------------------------------------
    # Store direct signals
    # --------------------------------------------------------

    group[
        "risk_change_signal"
    ] = direct_change

    group[
        "risk_acceleration_signal"
    ] = direct_acceleration

    # --------------------------------------------------------
    # Governance change
    # --------------------------------------------------------

    group[
        "governance_score_change"
    ] = governance.diff()

    # --------------------------------------------------------
    # Target change
    # --------------------------------------------------------

    group[
        "target_achievement_change"
    ] = target.diff()

    # --------------------------------------------------------
    # Production risk change
    # --------------------------------------------------------

    group[
        "production_risk_change"
    ] = (
        group[
            "production_risk_numeric"
        ].diff()
    )

    # --------------------------------------------------------
    # 3-period deterioration
    # --------------------------------------------------------

    group[
        "operational_risk_change_3m"
    ] = (
        operational
        -
        operational.shift(2)
    )

    group[
        "governance_change_3m"
    ] = (
        governance
        -
        governance.shift(2)
    )

    group[
        "target_change_3m"
    ] = (
        target
        -
        target.shift(2)
    )

    # --------------------------------------------------------
    # Consecutive increases
    # --------------------------------------------------------

    changes = operational.diff()

    consecutive = 0

    values = []

    for change in changes:

        if pd.isna(change):

            consecutive = 0

        elif change > 0:

            consecutive += 1

        else:

            consecutive = 0

        values.append(
            consecutive
        )

    group[
        "consecutive_risk_increases"
    ] = values

    return group


# ============================================================
# COMPONENT STRESS
# ============================================================

def component_stress(row):

    components = {}

    for column in [

        "equipment_risk",

        "logistics_risk",

        "weather_risk",

        "workforce_risk"
    ]:

        if column in row.index:

            value = safe_float(
                row[column],
                np.nan
            )

            if not pd.isna(value):

                components[
                    column
                ] = clamp(
                    value
                )

    if not components:

        return 0.0

    return float(
        np.mean(
            list(
                components.values()
            )
        )
    )


# ============================================================
# DRIVER DETECTION
# ============================================================

def identify_primary_driver(row):

    drivers = {}

    # --------------------------------------------------------
    # Operational components
    # --------------------------------------------------------

    for column, name in [

        (
            "equipment_risk",
            "Equipment stress"
        ),

        (
            "logistics_risk",
            "Logistics stress"
        ),

        (
            "weather_risk",
            "Weather stress"
        ),

        (
            "workforce_risk",
            "Workforce stress"
        )
    ]:

        if column in row.index:

            value = safe_float(
                row[column],
                np.nan
            )

            if not pd.isna(value):

                drivers[name] = value

    if not drivers:

        return "Risk deterioration"

    return max(
        drivers,
        key=drivers.get
    )


# ============================================================
# WARNING SCORE
# ============================================================

def calculate_warning_score(row):

    score = 0.0

    # ========================================================
    # 1. CURRENT RISK
    # ========================================================

    operational = clamp(
        row[
            "overall_operational_risk"
        ]
    )

    if operational >= 60:

        score += 20

    elif operational >= 50:

        score += 14

    elif operational >= 40:

        score += 8

    # ========================================================
    # 2. CURRENT RISK CHANGE
    # ========================================================

    change = safe_float(
        row.get(
            "risk_change_signal",
            0
        )
    )

    if change >= 15:

        score += 25

    elif change >= 10:

        score += 20

    elif change >= 5:

        score += 12

    elif change >= 2:

        score += 5

    # ========================================================
    # 3. ACCELERATION
    # ========================================================

    acceleration = safe_float(
        row.get(
            "risk_acceleration_signal",
            0
        )
    )

    if acceleration >= 10:

        score += 20

    elif acceleration >= 5:

        score += 14

    elif acceleration >= 2:

        score += 7

    # ========================================================
    # 4. THREE-MONTH DETERIORATION
    # ========================================================

    change_3m = safe_float(
        row.get(
            "operational_risk_change_3m",
            0
        )
    )

    if change_3m >= 15:

        score += 20

    elif change_3m >= 10:

        score += 15

    elif change_3m >= 5:

        score += 8

    elif change_3m >= 2:

        score += 4

    # ========================================================
    # 5. CONSECUTIVE INCREASES
    # ========================================================

    consecutive = safe_float(
        row.get(
            "consecutive_risk_increases",
            0
        )
    )

    if consecutive >= 4:

        score += 15

    elif consecutive >= 3:

        score += 12

    elif consecutive >= 2:

        score += 6

    # ========================================================
    # 6. PRODUCTION RISK
    # ========================================================

    production = str(
        row[
            "production_risk"
        ]
    ).upper().strip()

    if production == "UNRELIABLE":

        score += 15

    elif production == "HIGH":

        score += 12

    elif production == "MEDIUM":

        score += 6

    # ========================================================
    # 7. TARGET ACHIEVEMENT
    # ========================================================

    target = safe_float(
        row[
            "predicted_target_achievement_pct"
        ],
        100
    )

    if target < 85:

        score += 15

    elif target < 90:

        score += 10

    elif target < 95:

        score += 5

    # ========================================================
    # 8. TARGET DETERIORATION
    # ========================================================

    target_change = safe_float(
        row.get(
            "target_change_3m",
            0
        )
    )

    if target_change <= -15:

        score += 15

    elif target_change <= -10:

        score += 12

    elif target_change <= -5:

        score += 7

    # ========================================================
    # 9. COMPONENT STRESS
    # ========================================================

    stress = component_stress(
        row
    )

    if stress >= 75:

        score += 15

    elif stress >= 60:

        score += 10

    elif stress >= 45:

        score += 5

    return clamp(
        score
    )


# ============================================================
# WARNING LEVEL
# ============================================================

def classify_warning_level(row):

    score = safe_float(
        row[
            "early_warning_score"
        ]
    )

    operational = clamp(
        row[
            "overall_operational_risk"
        ]
    )

    change = safe_float(
        row.get(
            "risk_change_signal",
            0
        )
    )

    acceleration = safe_float(
        row.get(
            "risk_acceleration_signal",
            0
        )
    )

    production = str(
        row[
            "production_risk"
        ]
    ).upper().strip()

    # ========================================================
    # CRITICAL
    # ========================================================

    if score >= 75:

        return "CRITICAL"

    if (
        operational >= 60
        and
        change >= 10
    ):

        return "CRITICAL"

    if (
        production in [
            "HIGH",
            "UNRELIABLE"
        ]
        and
        change >= 15
        and
        acceleration >= 5
    ):

        return "CRITICAL"

    # ========================================================
    # EARLY WARNING
    # ========================================================

    if score >= 50:

        return "EARLY_WARNING"

    if (
        operational >= 50
        and
        change >= 10
    ):

        return "EARLY_WARNING"

    if (
        change >= 10
        and
        acceleration >= 5
    ):

        return "EARLY_WARNING"

    # ========================================================
    # WATCH
    # ========================================================

    if score >= 25:

        return "WATCH"

    if (
        change >= 5
        or
        acceleration >= 5
    ):

        return "WATCH"

    return "STABLE"


# ============================================================
# TRAJECTORY
# ============================================================

def classify_trajectory(row):

    change = safe_float(
        row.get(
            "risk_change_signal",
            0
        )
    )

    acceleration = safe_float(
        row.get(
            "risk_acceleration_signal",
            0
        )
    )

    # ========================================================
    # WORSENING
    # ========================================================

    # Risk is increasing AND the increase is accelerating
    if (
        change >= 10
        and
        acceleration >= 5
    ):
        return "RAPIDLY_WORSENING"

    # Risk is increasing
    if change >= 5:
        return "WORSENING"

    # ========================================================
    # IMPROVING
    # ========================================================

    # Risk is decreasing AND the decrease is accelerating
    if (
        change <= -10
        and
        acceleration <= -5
    ):
        return "RAPIDLY_IMPROVING"

    # Risk is decreasing
    if change <= -5:
        return "IMPROVING"

    # ========================================================
    # STABLE
    # ========================================================

    return "STABLE"


# ============================================================
# DRIVER-LEVEL REASON
# ============================================================

def generate_warning_reason(row):

    reasons = []

    operational = safe_float(
        row[
            "overall_operational_risk"
        ]
    )

    change = safe_float(
        row.get(
            "risk_change_signal",
            0
        )
    )

    acceleration = safe_float(
        row.get(
            "risk_acceleration_signal",
            0
        )
    )

    change_3m = safe_float(
        row.get(
            "operational_risk_change_3m",
            0
        )
    )

    consecutive = safe_float(
        row.get(
            "consecutive_risk_increases",
            0
        )
    )

    production = str(
        row[
            "production_risk"
        ]
    ).upper().strip()

    target = safe_float(
        row[
            "predicted_target_achievement_pct"
        ],
        100
    )

    # --------------------------------------------------------
    # Current risk
    # --------------------------------------------------------

    if operational >= 60:

        reasons.append(
            "Current operational risk is high"
        )

    elif operational >= 50:

        reasons.append(
            "Current operational risk is elevated"
        )

    # --------------------------------------------------------
    # Risk change
    # --------------------------------------------------------

    if change >= 15:

        reasons.append(
            "Operational risk has increased sharply"
        )

    elif change >= 5:

        reasons.append(
            "Operational risk is increasing"
        )

    # --------------------------------------------------------
    # Acceleration
    # --------------------------------------------------------

    if (
        change>0
        and
        acceleration >=10
    ):
        reasons.append(
            "Risk deterioration is accelerating"
        )
    elif(
        change > 0
        and acceleration >= 5
    ):
        reasons.append(
            "Risk deterioration is gaining momentum"
        )
    if(
        change<0
        and 
        acceleration <= -5
    ):
        reasons.append("Operational risk is improving")
    if change_3m >=15:
        reasons.append("Significant three-month deterioration detected")
    elif change_3m>=10:
        reasons.append("Three-month operational risk deterioration")
    if consecutive >= 3:

        reasons.append(
            "Risk has increased across multiple consecutive periods"
        )

    elif consecutive >= 2:

        reasons.append(
            "Risk has increased across consecutive periods"
        )

    # ========================================================
    # PRODUCTION RISK
    # ========================================================

    if production == "HIGH":

        reasons.append(
            "Production risk is high"
        )

    elif production == "UNRELIABLE":

        reasons.append(
            "Production forecast is unreliable"
        )

    elif production == "MEDIUM":

        reasons.append(
            "Production risk is elevated"
        )

    # ========================================================
    # TARGET ACHIEVEMENT
    # ========================================================

    if target < 85:

        reasons.append(
            "Predicted target achievement is significantly below target"
        )

    elif target < 95:

        reasons.append(
            "Predicted target achievement is below target"
        )

    # ========================================================
    # PRIMARY COMPONENT DRIVER
    # ========================================================

    driver = identify_primary_driver(
        row
    )

    column = driver_to_column(
        driver
    )

    if column:

        component_value = safe_float(
            row.get(
                column,
                0
            )
        )

        if component_value >= 70:

            reasons.append(
                f"{driver} is a major risk driver"
            )

        elif component_value >= 55:

            reasons.append(
                f"{driver} is elevated"
            )

    # ========================================================
    # DEFAULT
    # ========================================================

    if not reasons:

        return (
            "No significant deterioration detected"
        )

    return "; ".join(
        list(
            dict.fromkeys(
                reasons
            )
        )
    )

    # --------------------------------------------------------
    # 3-month trend
    # --------------------------------------------------------

    if change_3m >= 10:

        reasons.append(
            "Significant three-month deterioration detected"
        )

    # --------------------------------------------------------
    # Consecutive increases
    # --------------------------------------------------------

    if consecutive >= 3:

        reasons.append(
            "Risk has increased across multiple consecutive periods"
        )

    # --------------------------------------------------------
    # Production
    # --------------------------------------------------------

    if production == "HIGH":

        reasons.append(
            "Production risk is high"
        )

    elif production == "UNRELIABLE":

        reasons.append(
            "Production forecast is unreliable"
        )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    if target < 85:

        reasons.append(
            "Predicted target achievement is significantly below target"
        )

    elif target < 95:

        reasons.append(
            "Predicted target achievement is below target"
        )

    # --------------------------------------------------------
    # Primary component
    # --------------------------------------------------------

    driver = identify_primary_driver(
        row
    )

    component_value = safe_float(
        row.get(
            driver_to_column(driver),
            0
        )
    )

    if component_value >= 70:

        reasons.append(
            f"{driver} is a major risk driver"
        )

    elif component_value >= 55:

        reasons.append(
            f"{driver} is elevated"
        )

    # --------------------------------------------------------
    # Default
    # --------------------------------------------------------

    if not reasons:

        return (
            "No significant deterioration detected"
        )

    return "; ".join(
        list(
            dict.fromkeys(
                reasons
            )
        )
    )


# ============================================================
# DRIVER → COLUMN
# ============================================================

def driver_to_column(driver):

    mapping = {

        "Equipment stress":
            "equipment_risk",

        "Logistics stress":
            "logistics_risk",

        "Weather stress":
            "weather_risk",

        "Workforce stress":
            "workforce_risk"
    }

    return mapping.get(
        driver,
        ""
    )


# ============================================================
# MANAGEMENT ACTION
# ============================================================

def management_action(row):

    level = str(
        row[
            "warning_level"
        ]
    ).upper()

    driver = row[
        "primary_risk_driver"
    ]

    trajectory_value = row[
        "trajectory"
    ]

    if level == "CRITICAL":

        return (
            f"Immediate management review; investigate "
            f"{driver.lower()}, identify root causes and "
            f"initiate mitigation actions."
        )

    if level == "EARLY_WARNING":

        return (
            f"Increase monitoring frequency and review "
            f"{driver.lower()} before risk escalates further."
        )

    if level == "WATCH":

        return (
            f"Monitor {driver.lower()} closely and verify "
            f"whether the worsening trend continues."
        )

    if trajectory_value in [
        "IMPROVING",
        "RAPIDLY_IMPROVING"
    ]:

        return (
            "Risk is improving; continue monitoring to "
            "verify sustained recovery."
        )

    return (
        "Continue routine monitoring."
    )


# ============================================================
# LATEST STATUS
# ============================================================

def get_latest_status(df):

    print(
        "\n[10] Selecting latest status for each mine..."
    )

    latest = (

        df

        .sort_values(
            [
                "subsidiary",
                "date"
            ]
        )

        .groupby(
            "subsidiary",
            as_index=False
        )

        .tail(1)

        .copy()
    )

    # --------------------------------------------------------
    # Warning ranking
    # --------------------------------------------------------

    level_order = {

        "CRITICAL": 4,

        "EARLY_WARNING": 3,

        "WATCH": 2,

        "STABLE": 1
    }

    latest[
        "warning_level_numeric"
    ] = latest[
        "warning_level"
    ].map(
        level_order
    ).fillna(0)

    latest = latest.sort_values(
        [
            "warning_level_numeric",
            "early_warning_score",
            "governance_score"
        ],
        ascending=False
    ).reset_index(
        drop=True
    )

    latest[
        "warning_rank"
    ] = np.arange(
        1,
        len(latest) + 1
    )

    latest.drop(
        columns=[
            "warning_level_numeric"
        ],
        inplace=True,
        errors="ignore"
    )

    return latest


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n" + "=" * 70
    )

    print(
        "PHASE 7B — AI EARLY WARNING ENGINE — VERSION 3"
    )

    print(
        "=" * 70
    )

    # ========================================================
    # 1. LOAD
    # ========================================================

    df = load_data()

    # ========================================================
    # 2. NORMALIZE
    # ========================================================

    df = normalize_columns(
        df
    )

    # ========================================================
    # 3. VALIDATE
    # ========================================================

    df = validate_columns(
        df
    )

    # ========================================================
    # 4. CLEAN
    # ========================================================

    df = clean_data(
        df
    )

    # ========================================================
    # 5. TRAJECTORY
    # ========================================================

    print(
        "\n[5] Analysing mine risk trajectories..."
    )

    groups = []

    for subsidiary, group in df.groupby(
        "subsidiary",
        sort=False
    ):

        groups.append(
            calculate_trajectory(
                group
            )
        )

    df = pd.concat(
        groups,
        ignore_index=True
    )

    # ========================================================
    # 6. WARNING SCORE
    # ========================================================

    print(
        "\n[6] Calculating early-warning scores..."
    )

    df[
        "early_warning_score"
    ] = df.apply(
        calculate_warning_score,
        axis=1
    )

    # ========================================================
    # 7. WARNING LEVEL
    # ========================================================

    print(
        "\n[7] Classifying warning levels..."
    )

    df[
        "warning_level"
    ] = df.apply(
        classify_warning_level,
        axis=1
    )

    # ========================================================
    # 8. TRAJECTORY
    # ========================================================

    print(
        "\n[8] Classifying risk trajectories..."
    )

    df[
        "trajectory"
    ] = df.apply(
        classify_trajectory,
        axis=1
    )

    # ========================================================
    # 9. PRIMARY DRIVER
    # ========================================================

    print(
        "\n[9] Identifying primary risk drivers..."
    )

    df[
        "primary_risk_driver"
    ] = df.apply(
        identify_primary_driver,
        axis=1
    )

    # ========================================================
    # 10. REASON
    # ========================================================

    print(
        "\n[10] Generating warning explanations..."
    )

    df[
        "warning_reason"
    ] = df.apply(
        generate_warning_reason,
        axis=1
    )

    # ========================================================
    # 11. MANAGEMENT ACTION
    # ========================================================

    print(
        "\n[11] Generating management recommendations..."
    )

    df[
        "early_warning_recommendation"
    ] = df.apply(
        management_action,
        axis=1
    )

    # ========================================================
    # 12. LATEST MINE STATUS
    # ========================================================

    latest = get_latest_status(
        df
    )

    # ========================================================
    # DISPLAY
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "AI EARLY WARNING STATUS"
    )

    print(
        "=" * 70
    )

    display_columns = [

        "warning_rank",

        "subsidiary",

        "date",

        "overall_operational_risk",

        "production_risk",

        "predicted_target_achievement_pct",

        "early_warning_score",

        "warning_level",

        "trajectory",

        "primary_risk_driver",

        "risk_change_signal",

        "risk_acceleration_signal",

        "operational_risk_change_3m",

        "consecutive_risk_increases"
    ]

    display_columns = [

        column

        for column in display_columns

        if column in latest.columns
    ]

    print(
        latest[
            display_columns
        ].round(2).to_string(
            index=False
        )
    )

    # ========================================================
    # TOP WARNINGS
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "TOP EARLY WARNINGS"
    )

    print(
        "=" * 70
    )

    warning_records = latest[
        latest[
            "warning_level"
        ].isin(
            [
                "CRITICAL",
                "EARLY_WARNING",
                "WATCH"
            ]
        )
    ]

    if len(warning_records) == 0:

        print(
            "\nNo active early warnings detected."
        )

    else:

        for _, row in warning_records.iterrows():

            print(
                f"\n🚨 {row['subsidiary']}"
            )

            print(
                f"Warning Score : "
                f"{row['early_warning_score']:.2f}"
            )

            print(
                f"Warning Level : "
                f"{row['warning_level']}"
            )

            print(
                f"Trajectory    : "
                f"{row['trajectory']}"
            )

            print(
                f"Primary Driver: "
                f"{row['primary_risk_driver']}"
            )

            print(
                f"Current Risk  : "
                f"{row['overall_operational_risk']:.2f}"
            )

            print(
                f"Risk Change   : "
                f"{safe_float(row.get('risk_change_signal', 0)):.2f}"
            )

            print(
                f"Acceleration  : "
                f"{safe_float(row.get('risk_acceleration_signal', 0)):.2f}"
            )

            print(
                f"Reason        : "
                f"{row['warning_reason']}"
            )

            print(
                f"Action        : "
                f"{row['early_warning_recommendation']}"
            )

    # ========================================================
    # COMPONENT STATUS
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "LATEST OPERATIONAL RISK COMPONENTS"
    )

    print(
        "=" * 70
    )

    component_columns = [

        "subsidiary",

        "equipment_risk",

        "logistics_risk",

        "weather_risk",

        "workforce_risk",

        "overall_operational_risk"
    ]

    component_columns = [

        column

        for column in component_columns

        if column in latest.columns
    ]

    print(
        latest[
            component_columns
        ].round(2).to_string(
            index=False
        )
    )

    # ========================================================
    # WARNING DISTRIBUTION
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "EARLY WARNING SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        "\nWarning distribution:"
    )

    print(
        latest[
            "warning_level"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nTrajectory distribution:"
    )

    print(
        latest[
            "trajectory"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nPrimary risk drivers:"
    )

    print(
        latest[
            "primary_risk_driver"
        ]
        .value_counts()
        .to_string()
    )

    # ========================================================
    # SCORE STATISTICS
    # ========================================================

    print(
        "\nEarly-warning score statistics:"
    )

    print(
        latest[
            "early_warning_score"
        ]
        .describe()
        .round(2)
        .to_string()
    )

    # ========================================================
    # SAVE
    # ========================================================

    print(
        "\n[12] Saving early-warning outputs..."
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
        MINE_SUMMARY_FILE,
        index=False
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    print(
        "\nFull early-warning analysis:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        "\nLatest mine warning summary:"
    )

    print(
        MINE_SUMMARY_FILE
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "PHASE 7B VERSION 3 COMPLETE"
    )

    print(
        "=" * 70
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()