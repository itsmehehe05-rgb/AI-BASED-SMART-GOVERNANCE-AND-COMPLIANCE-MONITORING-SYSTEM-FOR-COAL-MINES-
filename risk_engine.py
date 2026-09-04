import os
import joblib
import numpy as np
import pandas as pd

from data_loader import load_training_data
from preprocessing import preprocess_data
from feature_engineering import create_features


# ============================================================
# CONFIGURATION
# ============================================================

TARGET = "target_next_month_production_mt"

MODEL_PATH = (
    r"D:\CoalMineAI\models\production_forecaster.pkl"
)

OUTPUT_PATH = (
    r"D:\CoalMineAI\outputs\production_risk_analysis.csv"
)


# ============================================================
# GENERAL UTILITIES
# ============================================================

def safe_float(value, default=np.nan):
    """Safely convert a value to float."""

    try:
        value = float(value)

        if np.isfinite(value):
            return value

    except (TypeError, ValueError):

        pass

    return default


def clip_score(value):

    if pd.isna(value):
        return 0.0

    return float(
        np.clip(value, 0, 100)
    )


def normalize(value, low, high):
    """
    Normalize a value into 0-100.
    """

    value = safe_float(value)

    if pd.isna(value):
        return np.nan

    if high <= low:
        return 0.0

    score = (
        (value - low)
        /
        (high - low)
    ) * 100

    return clip_score(score)


def inverse_score(value, good, bad):
    """
    Used when higher values mean lower risk.
    """

    value = safe_float(value)

    if pd.isna(value):
        return np.nan

    if good <= bad:
        return 0.0

    score = (
        (good - value)
        /
        (good - bad)
    ) * 100

    return clip_score(score)


# ============================================================
# DATA-DRIVEN CALIBRATION
# ============================================================

def build_calibration_ranges(df):
    """
    Build robust percentile-based ranges from the dataset.

    Percentiles are used instead of minimum/maximum so that
    extreme synthetic observations do not dominate the risk
    scale.
    """

    calibration = {}

    # --------------------------------------------------------
    # Weather
    # --------------------------------------------------------

    if "rainfall_mm" in df.columns:

        rainfall = pd.to_numeric(
            df["rainfall_mm"],
            errors="coerce"
        ).dropna()

        rainfall = rainfall[
            rainfall >= 0
        ]

        if len(rainfall) > 0:

            calibration["rainfall"] = (
                float(rainfall.quantile(0.10)),
                float(rainfall.quantile(0.90))
            )

    # --------------------------------------------------------
    # Temperature
    # --------------------------------------------------------

    if "temperature_c" in df.columns:

        temperature = pd.to_numeric(
            df["temperature_c"],
            errors="coerce"
        ).dropna()

        if len(temperature) > 0:

            calibration["temperature"] = (
                float(temperature.quantile(0.10)),
                float(temperature.quantile(0.90))
            )

    # --------------------------------------------------------
    # Training coverage
    # --------------------------------------------------------

    if "training_coverage_pct" in df.columns:

        training = pd.to_numeric(
            df["training_coverage_pct"],
            errors="coerce"
        ).dropna()

        training = training[
            (training >= 0)
            &
            (training <= 100)
        ]

        if len(training) > 0:

            calibration["training"] = (
                float(training.quantile(0.10)),
                float(training.quantile(0.90))
            )

    # --------------------------------------------------------
    # Absenteeism
    # --------------------------------------------------------

    if "absenteeism_rate_pct" in df.columns:

        absenteeism = pd.to_numeric(
            df["absenteeism_rate_pct"],
            errors="coerce"
        ).dropna()

        absenteeism = absenteeism[
            absenteeism >= 0
        ]

        if len(absenteeism) > 0:

            calibration["absenteeism"] = (
                float(absenteeism.quantile(0.10)),
                float(absenteeism.quantile(0.90))
            )

    return calibration


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(df):

    data = df.copy()

    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce"
    )

    X = data.drop(
        columns=[
            TARGET,
            "date",
            "synthetic_flag",
            "source_basis",
            "split",
            "financial_year"
        ],
        errors="ignore"
    )

    categorical_columns = X.select_dtypes(
        include=[
            "object",
            "string",
            "category"
        ]
    ).columns.tolist()

    if categorical_columns:

        X = pd.get_dummies(
            X,
            columns=categorical_columns,
            drop_first=False
        )

    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(
        X.median(numeric_only=True)
    )

    X = X.fillna(0)

    X.columns = X.columns.astype(str)

    return X


# ============================================================
# TARGET RISK
# ============================================================

def calculate_target_risk(
    predicted_production,
    production_target
):

    predicted_production = safe_float(
        predicted_production
    )

    production_target = safe_float(
        production_target
    )

    if (
        pd.isna(production_target)
        or production_target <= 0.05
        or pd.isna(predicted_production)
    ):

        return (
            np.nan,
            np.nan,
            np.nan,
            "UNRELIABLE"
        )

    achievement = (
        predicted_production
        /
        production_target
        *
        100
    )

    shortfall = max(
        production_target
        -
        predicted_production,
        0
    )

    shortfall_pct = (
        shortfall
        /
        production_target
        *
        100
    )

    if achievement >= 100:

        risk = "LOW"

    elif achievement >= 95:

        risk = "MEDIUM"

    else:

        risk = "HIGH"

    return (
        achievement,
        shortfall,
        shortfall_pct,
        risk
    )


# ============================================================
# EQUIPMENT RISK
# ============================================================

def calculate_equipment_risk(row):

    scores = []
    weights = []

    # --------------------------------------------------------
    # Equipment availability
    # --------------------------------------------------------

    if "equipment_availability_pct" in row.index:

        availability = safe_float(
            row["equipment_availability_pct"]
        )

        if not pd.isna(availability):

            availability = np.clip(
                availability,
                0,
                100
            )

            availability_risk = inverse_score(
                availability,
                100,
                70
            )

            scores.append(
                availability_risk
            )

            weights.append(0.50)

    # --------------------------------------------------------
    # Breakdown
    # --------------------------------------------------------

    if "equipment_breakdown_hours" in row.index:

        breakdown = safe_float(
            row["equipment_breakdown_hours"]
        )

        if not pd.isna(breakdown):

            breakdown = max(
                breakdown,
                0
            )

            breakdown_risk = normalize(
                breakdown,
                0,
                50
            )

            scores.append(
                breakdown_risk
            )

            weights.append(0.30)

    # --------------------------------------------------------
    # Utilization
    # --------------------------------------------------------

    if "equipment_utilization_pct" in row.index:

        utilization = safe_float(
            row["equipment_utilization_pct"]
        )

        if not pd.isna(utilization):

            utilization = np.clip(
                utilization,
                0,
                100
            )

            utilization_risk = inverse_score(
                utilization,
                85,
                45
            )

            scores.append(
                utilization_risk
            )

            weights.append(0.20)

    if not scores:

        return 0.0

    scores = np.array(scores)
    weights = np.array(weights)

    weights = weights / weights.sum()

    return clip_score(
        np.sum(
            scores * weights
        )
    )


# ============================================================
# LOGISTICS RISK
# ============================================================

def calculate_logistics_risk(row):

    scores = []
    weights = []

    # --------------------------------------------------------
    # Rail availability
    # --------------------------------------------------------

    if "rail_rake_availability_pct" in row.index:

        rail = safe_float(
            row["rail_rake_availability_pct"]
        )

        if not pd.isna(rail):

            rail = np.clip(
                rail,
                0,
                100
            )

            rail_risk = inverse_score(
                rail,
                100,
                70
            )

            scores.append(
                rail_risk
            )

            weights.append(0.60)

    # --------------------------------------------------------
    # Evacuation delay
    # --------------------------------------------------------

    if "evacuation_delay_hours" in row.index:

        delay = safe_float(
            row["evacuation_delay_hours"]
        )

        if not pd.isna(delay):

            delay = max(
                delay,
                0
            )

            delay_risk = normalize(
                delay,
                0,
                48
            )

            scores.append(
                delay_risk
            )

            weights.append(0.40)

    if not scores:

        return 0.0

    scores = np.array(scores)
    weights = np.array(weights)

    weights = weights / weights.sum()

    return clip_score(
        np.sum(
            scores * weights
        )
    )


# ============================================================
# WEATHER RISK — VERSION 3
# ============================================================

def calculate_weather_risk(
    row,
    calibration
):

    rainfall_risk = 0.0
    temperature_risk = 0.0

    # --------------------------------------------------------
    # Rainfall
    # --------------------------------------------------------

    if "rainfall_mm" in row.index:

        rainfall = safe_float(
            row["rainfall_mm"]
        )

        if not pd.isna(rainfall):

            rainfall = max(
                rainfall,
                0
            )

            if "rainfall" in calibration:

                low, high = calibration[
                    "rainfall"
                ]

                # Expand the percentile range slightly
                # so the score is not immediately saturated.
                low = max(
                    0,
                    low
                )

                high = max(
                    low + 1,
                    high
                )

                rainfall_risk = normalize(
                    rainfall,
                    low,
                    high
                )

            else:

                rainfall_risk = normalize(
                    rainfall,
                    0,
                    100
                )

    # --------------------------------------------------------
    # Temperature
    # --------------------------------------------------------

    if "temperature_c" in row.index:

        temperature = safe_float(
            row["temperature_c"]
        )

        if not pd.isna(temperature):

            if "temperature" in calibration:

                low, high = calibration[
                    "temperature"
                ]

                # Risk is highest outside the normal
                # temperature range.

                if temperature < low:

                    temperature_risk = normalize(
                        low - temperature,
                        0,
                        max(5, abs(low))
                    )

                elif temperature > high:

                    temperature_risk = normalize(
                        temperature - high,
                        0,
                        max(5, abs(high))
                    )

                else:

                    temperature_risk = 0.0

            else:

                temperature_risk = 0.0

    # --------------------------------------------------------
    # Combined weather risk
    # --------------------------------------------------------

    if "temperature_c" in row.index:

        weather_risk = (
            rainfall_risk * 0.75
            +
            temperature_risk * 0.25
        )

    else:

        weather_risk = rainfall_risk

    return clip_score(
        weather_risk
    )


# ============================================================
# WORKFORCE RISK — VERSION 3
# ============================================================

def calculate_workforce_risk(
    row,
    calibration
):

    scores = []
    weights = []

    # --------------------------------------------------------
    # Training coverage
    # --------------------------------------------------------

    if "training_coverage_pct" in row.index:

        training = safe_float(
            row["training_coverage_pct"]
        )

        if not pd.isna(training):

            training = np.clip(
                training,
                0,
                100
            )

            if "training" in calibration:

                low, high = calibration[
                    "training"
                ]

                # Low training coverage means higher risk.
                training_risk = inverse_score(
                    training,
                    high,
                    low
                )

            else:

                training_risk = inverse_score(
                    training,
                    100,
                    70
                )

            scores.append(
                training_risk
            )

            weights.append(0.60)

    # --------------------------------------------------------
    # Absenteeism
    # --------------------------------------------------------

    if "absenteeism_rate_pct" in row.index:

        absenteeism = safe_float(
            row["absenteeism_rate_pct"]
        )

        if not pd.isna(absenteeism):

            absenteeism = max(
                absenteeism,
                0
            )

            if "absenteeism" in calibration:

                low, high = calibration[
                    "absenteeism"
                ]

                absenteeism_risk = normalize(
                    absenteeism,
                    low,
                    max(
                        low + 0.1,
                        high
                    )
                )

            else:

                absenteeism_risk = normalize(
                    absenteeism,
                    0,
                    15
                )

            scores.append(
                absenteeism_risk
            )

            weights.append(0.40)

    if not scores:

        return 0.0

    scores = np.array(scores)
    weights = np.array(weights)

    weights = weights / weights.sum()

    return clip_score(
        np.sum(
            scores * weights
        )
    )


# ============================================================
# OPERATIONAL RISK
# ============================================================

def calculate_operational_risk(
    row,
    calibration
):

    equipment_risk = (
        calculate_equipment_risk(
            row
        )
    )

    logistics_risk = (
        calculate_logistics_risk(
            row
        )
    )

    weather_risk = (
        calculate_weather_risk(
            row,
            calibration
        )
    )

    workforce_risk = (
        calculate_workforce_risk(
            row,
            calibration
        )
    )

    # --------------------------------------------------------
    # Overall operational risk
    # --------------------------------------------------------

    overall = (

        equipment_risk * 0.40

        +

        logistics_risk * 0.30

        +

        weather_risk * 0.15

        +

        workforce_risk * 0.15
    )

    overall = clip_score(
        overall
    )

    # --------------------------------------------------------
    # Risk level
    # --------------------------------------------------------

    if overall >= 70:

        overall_level = "HIGH"

    elif overall >= 40:

        overall_level = "MEDIUM"

    else:

        overall_level = "LOW"

    return {

        "equipment_risk":
            equipment_risk,

        "logistics_risk":
            logistics_risk,

        "weather_risk":
            weather_risk,

        "workforce_risk":
            workforce_risk,

        "overall_operational_risk":
            overall,

        "overall_risk_level":
            overall_level
    }


# ============================================================
# RISK TREND
# ============================================================

def calculate_risk_trend(df):

    df = df.copy()

    df["risk_change"] = 0.0

    df["risk_acceleration"] = 0.0

    if (
        "subsidiary" not in df.columns
        or "date" not in df.columns
    ):

        return df

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df = df.sort_values(
        [
            "subsidiary",
            "date"
        ]
    )

    df["previous_risk"] = (
        df.groupby(
            "subsidiary"
        )[
            "overall_operational_risk"
        ]
        .shift(1)
    )

    df["risk_change"] = (
        df[
            "overall_operational_risk"
        ]
        -
        df[
            "previous_risk"
        ]
    )

    df["previous_change"] = (
        df.groupby(
            "subsidiary"
        )[
            "risk_change"
        ]
        .shift(1)
    )

    df["risk_acceleration"] = (
        df[
            "risk_change"
        ]
        -
        df[
            "previous_change"
        ]
    )

    df["risk_change"] = (
        df["risk_change"]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
    )

    df["risk_acceleration"] = (
        df["risk_acceleration"]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
    )

    df.drop(
        columns=[
            "previous_risk",
            "previous_change"
        ],
        inplace=True,
        errors="ignore"
    )

    return df


# ============================================================
# GOVERNANCE PRIORITY
# ============================================================

def calculate_governance_priority(df):

    production_risk_score = np.select(

        [
            df["production_risk"]
            ==
            "HIGH",

            df["production_risk"]
            ==
            "MEDIUM",

            df["production_risk"]
            ==
            "LOW"
        ],

        [
            100,
            50,
            0
        ],

        default=np.nan
    )

    production_risk_score = np.where(

        df["production_risk"]
        ==
        "UNRELIABLE",

        np.nan,

        production_risk_score
    )

    # --------------------------------------------------------
    # Base governance score
    # --------------------------------------------------------

    df[
        "governance_priority_score"
    ] = (

        df[
            "overall_operational_risk"
        ] * 0.60

        +

        np.nan_to_num(
            production_risk_score,
            nan=0
        ) * 0.40
    )

    # --------------------------------------------------------
    # Deterioration adjustment
    # --------------------------------------------------------

    deterioration_bonus = (
        np.clip(
            df["risk_change"],
            0,
            20
        )
        *
        0.25
    )

    acceleration_bonus = (
        np.clip(
            df["risk_acceleration"],
            0,
            20
        )
        *
        0.15
    )

    df[
        "governance_priority_score"
    ] += (
        deterioration_bonus
        +
        acceleration_bonus
    )

    # --------------------------------------------------------
    # Protect unreliable production targets
    # --------------------------------------------------------

    unreliable_mask = (
        df["production_risk"]
        ==
        "UNRELIABLE"
    )

    df.loc[
        unreliable_mask,
        "governance_priority_score"
    ] = (

        df.loc[
            unreliable_mask,
            "overall_operational_risk"
        ]
        *
        0.60

        +

        deterioration_bonus.loc[
            unreliable_mask
        ]

        +

        acceleration_bonus.loc[
            unreliable_mask
        ]
    )

    df[
        "governance_priority_score"
    ] = (
        df[
            "governance_priority_score"
        ]
        .clip(
            0,
            100
        )
    )

    # --------------------------------------------------------
    # Priority level
    # --------------------------------------------------------

    df[
        "governance_priority"
    ] = np.select(

        [

            df[
                "governance_priority_score"
            ]
            >=
            70,

            df[
                "governance_priority_score"
            ]
            >=
            40
        ],

        [
            "HIGH",
            "MEDIUM"
        ],

        default="LOW"
    )

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)

    print(
        "PHASE 5B — CALIBRATED AI PRODUCTION & OPERATIONAL RISK ENGINE"
    )

    print("=" * 70)

    # ========================================================
    # 1. LOAD MODEL
    # ========================================================

    print(
        "\n[1] Loading trained model..."
    )

    package = joblib.load(
        MODEL_PATH
    )

    model = package["model"]

    model_name = package[
        "model_name"
    ]

    model_features = package[
        "features"
    ]

    print(
        f"Model: {model_name}"
    )

    # ========================================================
    # 2. LOAD DATA
    # ========================================================

    print(
        "\n[2] Loading dataset..."
    )

    df = load_training_data()

    df = preprocess_data(
        df
    )

    df = create_features(
        df
    )

    print(
        f"Rows: {len(df)}"
    )

    # ========================================================
    # 3. BUILD CALIBRATION
    # ========================================================

    print(
        "\n[3] Building data-driven risk calibration..."
    )

    calibration = build_calibration_ranges(
        df
    )

    print(
        "\nCalibration ranges:"
    )

    for key, value in calibration.items():

        print(
            f"{key}: "
            f"{value[0]:.3f} → "
            f"{value[1]:.3f}"
        )

    # ========================================================
    # 4. PREPARE FEATURES
    # ========================================================

    print(
        "\n[4] Preparing features..."
    )

    X = prepare_features(
        df
    )

    X = X.reindex(
        columns=model_features,
        fill_value=0
    )

    # ========================================================
    # 5. GENERATE FORECAST
    # ========================================================

    print(
        "\n[5] Generating AI forecasts..."
    )

    raw_predictions = model.predict(
        X
    )

    predictions = np.clip(
        raw_predictions,
        0,
        None
    )

    # ========================================================
    # 6. SANITY BOUNDS
    # ========================================================

    current_production = pd.to_numeric(
        df[
            "monthly_production_mt"
        ],
        errors="coerce"
    ).fillna(0)

    lower_bound = (
        current_production * 0.25
    )

    upper_bound = (
        current_production * 1.75
    )

    predictions = np.clip(
        predictions,
        lower_bound,
        upper_bound
    )

    df[
        "predicted_production_mt"
    ] = predictions

    # ========================================================
    # 7. PRODUCTION TARGET RISK
    # ========================================================

    print(
        "\n[6] Calculating production target risk..."
    )

    target_results = df.apply(

        lambda row:

        calculate_target_risk(

            row[
                "predicted_production_mt"
            ],

            row[
                "production_target_mt"
            ]
        ),

        axis=1
    )

    target_results_df = pd.DataFrame(

        target_results.tolist(),

        columns=[

            "predicted_target_achievement_pct",

            "expected_shortfall_mt",

            "expected_shortfall_pct",

            "production_risk"
        ],

        index=df.index
    )

    df = pd.concat(
        [
            df,
            target_results_df
        ],
        axis=1
    )

    # ========================================================
    # 8. OPERATIONAL RISK
    # ========================================================

    print(
        "\n[7] Calculating calibrated operational risk..."
    )

    operational_results = df.apply(

        lambda row:

        calculate_operational_risk(
            row,
            calibration
        ),

        axis=1
    )

    operational_df = pd.DataFrame(
        operational_results.tolist(),
        index=df.index
    )

    df = pd.concat(
        [
            df,
            operational_df
        ],
        axis=1
    )

    # ========================================================
    # 9. RISK TREND
    # ========================================================

    print(
        "\n[8] Calculating risk deterioration..."
    )

    df = calculate_risk_trend(
        df
    )

    # ========================================================
    # 10. GOVERNANCE PRIORITY
    # ========================================================

    print(
        "\n[9] Calculating governance priority..."
    )

    df = calculate_governance_priority(
        df
    )

    # ========================================================
    # 11. OUTPUT
    # ========================================================

    output_columns = [

        "date",

        "subsidiary",

        "monthly_production_mt",

        "production_target_mt",

        TARGET,

        "predicted_production_mt",

        "predicted_target_achievement_pct",

        "expected_shortfall_mt",

        "expected_shortfall_pct",

        "production_risk",

        "equipment_risk",

        "logistics_risk",

        "weather_risk",

        "workforce_risk",

        "overall_operational_risk",

        "overall_risk_level",

        "risk_change",

        "risk_acceleration",

        "governance_priority_score",

        "governance_priority"
    ]

    output_columns = [

        column

        for column in output_columns

        if column in df.columns
    ]

    result = df[
        output_columns
    ].copy()

    # ========================================================
    # 12. SAVE
    # ========================================================

    print(
        "\n[10] Saving results..."
    )

    os.makedirs(
        os.path.dirname(
            OUTPUT_PATH
        ),
        exist_ok=True
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # ========================================================
    # 13. SUMMARY
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "CALIBRATED RISK SUMMARY"
    )

    print(
        "=" * 70
    )

    # Production
    print(
        "\nProduction Risk:"
    )

    print(
        result[
            "production_risk"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    # Operational
    print(
        "\nOperational Risk:"
    )

    print(
        result[
            "overall_risk_level"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    # Governance
    print(
        "\nGovernance Priority:"
    )

    print(
        result[
            "governance_priority"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    # ========================================================
    # 14. RISK STATISTICS
    # ========================================================

    print(
        "\nOperational Risk Statistics:"
    )

    print(
        result[
            "overall_operational_risk"
        ]
        .describe()
        .round(2)
        .to_string()
    )

    # ========================================================
    # 15. COMPONENT STATISTICS
    # ========================================================

    print(
        "\nRisk Component Statistics:"
    )

    component_columns = [

        "equipment_risk",

        "logistics_risk",

        "weather_risk",

        "workforce_risk",

        "overall_operational_risk"
    ]

    component_stats = (
        result[
            component_columns
        ]
        .describe()
        .round(2)
    )

    print(
        component_stats.to_string()
    )

    # ========================================================
    # 16. MINE SUMMARY
    # ========================================================

    if "subsidiary" in result.columns:

        print(
            "\nMine-level risk summary:"
        )

        mine_summary = (

            result

            .groupby(
                "subsidiary"
            )

            .agg(

                average_operational_risk=(
                    "overall_operational_risk",
                    "mean"
                ),

                maximum_operational_risk=(
                    "overall_operational_risk",
                    "max"
                ),

                average_governance_score=(
                    "governance_priority_score",
                    "mean"
                ),

                maximum_governance_score=(
                    "governance_priority_score",
                    "max"
                )
            )

            .sort_values(
                "average_governance_score",
                ascending=False
            )
        )

        print(
            mine_summary
            .round(2)
            .to_string()
        )

    # ========================================================
    # 17. TOP RECORDS
    # ========================================================

    print(
        "\nTop 10 highest-priority records:"
    )

    top_records = (

        result

        .sort_values(
            "governance_priority_score",
            ascending=False
        )

        .head(10)
    )

    print(
        top_records.to_string(
            index=False
        )
    )

    # ========================================================
    # 18. SANITY CHECK
    # ========================================================

    negative_predictions = (

        result[
            "predicted_production_mt"
        ]
        < 0

    ).sum()

    print(
        "\nPrediction sanity check:"
    )

    print(
        f"Negative predictions: "
        f"{negative_predictions}"
    )

    # ========================================================
    # 19. SATURATION CHECK
    # ========================================================

    print(
        "\nRisk saturation check:"
    )

    for column in [

        "equipment_risk",

        "logistics_risk",

        "weather_risk",

        "workforce_risk"
    ]:

        if column in result.columns:

            saturated = (
                result[column] >= 99.9
            ).sum()

            print(
                f"{column}: "
                f"{saturated} records at ~100"
            )

    # ========================================================
    # COMPLETE
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "PHASE 5B COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nCalibrated risk analysis saved to:"
    )

    print(
        OUTPUT_PATH
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()