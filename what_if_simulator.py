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

MODEL_PATH = r"D:\CoalMineAI\models\production_forecaster.pkl"

RISK_FILE = r"D:\CoalMineAI\outputs\production_risk_analysis.csv"

OUTPUT_FILE = r"D:\CoalMineAI\outputs\scenario_simulation_results.csv"

TARGET = "target_next_month_production_mt"


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))


def safe_float(value, default=0.0):

    try:
        value = float(value)

        if np.isnan(value) or np.isinf(value):
            return default

        return value

    except Exception:
        return default


# ============================================================
# FEATURE PREPARATION
# ============================================================

def prepare_features(df):

    data = df.copy()

    if "date" in data.columns:

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
# PREDICTION
# ============================================================

def predict(model, row, model_features):

    X = row.reindex(
        columns=model_features,
        fill_value=0
    )

    prediction = model.predict(X)[0]

    return max(
        safe_float(prediction),
        0
    )


# ============================================================
# TARGET ACHIEVEMENT
# ============================================================

def target_achievement(
    prediction,
    target
):

    target = safe_float(target)

    if target <= 0.05:
        return np.nan

    return (
        prediction
        /
        target
        *
        100
    )


# ============================================================
# PRODUCTION RISK
# ============================================================

def production_risk(achievement):

    if pd.isna(achievement):

        return "UNRELIABLE"

    if achievement >= 100:

        return "LOW"

    if achievement >= 95:

        return "MEDIUM"

    return "HIGH"


# ============================================================
# OPERATIONAL RISK ENGINE
# ============================================================

def operational_risk(row):

    equipment_availability = safe_float(
        row.get(
            "equipment_availability_pct",
            100
        ),
        100
    )

    equipment_utilization = safe_float(
        row.get(
            "equipment_utilization_pct",
            100
        ),
        100
    )

    breakdown_hours = safe_float(
        row.get(
            "equipment_breakdown_hours",
            0
        )
    )

    rail_availability = safe_float(
        row.get(
            "rail_rake_availability_pct",
            100
        ),
        100
    )

    evacuation_delay = safe_float(
        row.get(
            "evacuation_delay_hours",
            0
        )
    )

    fmc_availability = safe_float(
        row.get(
            "fmc_availability_pct",
            100
        ),
        100
    )

    rainfall = safe_float(
        row.get(
            "rainfall_mm",
            0
        )
    )

    workforce = safe_float(
        row.get(
            "workforce_strength",
            0
        )
    )

    # --------------------------------------------------------
    # EQUIPMENT RISK
    # --------------------------------------------------------

    equipment_risk = (

        (100 - equipment_availability)
        * 0.45

        +

        min(
            breakdown_hours,
            100
        )
        * 0.35

        +

        (100 - equipment_utilization)
        * 0.20
    )

    equipment_risk = clamp(
        equipment_risk,
        0,
        100
    )

    # --------------------------------------------------------
    # LOGISTICS RISK
    # --------------------------------------------------------

    logistics_risk = (

        (100 - rail_availability)
        * 0.45

        +

        min(
            evacuation_delay * 2,
            100
        )
        * 0.35

        +

        (100 - fmc_availability)
        * 0.20
    )

    logistics_risk = clamp(
        logistics_risk,
        0,
        100
    )

    # --------------------------------------------------------
    # WEATHER RISK
    # --------------------------------------------------------

    weather_risk = clamp(
        rainfall / 5,
        0,
        100
    )

    # --------------------------------------------------------
    # WORKFORCE RISK
    # --------------------------------------------------------

    workforce_risk = 0

    if workforce <= 0:

        workforce_risk = 50

    else:

        # Relative workforce pressure.
        # Uses the dataset's median as a reference.

        workforce_risk = 0

    # --------------------------------------------------------
    # OVERALL RISK
    # --------------------------------------------------------

    overall = (

        equipment_risk * 0.40

        +

        logistics_risk * 0.30

        +

        weather_risk * 0.20

        +

        workforce_risk * 0.10
    )

    overall = clamp(
        overall,
        0,
        100
    )

    if overall >= 70:

        level = "HIGH"

    elif overall >= 40:

        level = "MEDIUM"

    else:

        level = "LOW"

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
            level
    }


# ============================================================
# APPLY SCENARIO
# ============================================================

def apply_scenario(
    base_row,
    changes
):

    scenario = base_row.copy()

    for column, change in changes.items():

        if column not in scenario.index:
            continue

        current = safe_float(
            scenario[column]
        )

        new_value = current + change

        # Percentage constraints
        if column.endswith("_pct"):

            new_value = clamp(
                new_value,
                0,
                100
            )

        # Hours cannot be negative
        elif column.endswith("_hours"):

            new_value = max(
                new_value,
                0
            )

        # Workforce cannot be negative
        elif column == "workforce_strength":

            new_value = max(
                new_value,
                0
            )

        scenario[column] = new_value

    return scenario


# ============================================================
# RUN SCENARIO
# ============================================================

def run_scenario(
    name,
    base_row,
    changes,
    model,
    model_features
):

    scenario = apply_scenario(
        base_row,
        changes
    )

    scenario_df = pd.DataFrame(
        [scenario]
    )

    scenario_features = prepare_features(
        scenario_df
    )

    scenario_features = scenario_features.reindex(
        columns=model_features,
        fill_value=0
    )

    prediction = predict(
        model,
        scenario_features,
        model_features
    )

    target = safe_float(
        scenario[
            "production_target_mt"
        ]
    )

    achievement = target_achievement(
        prediction,
        target
    )

    prod_risk = production_risk(
        achievement
    )

    operational = operational_risk(
        scenario
    )

    return {

        "scenario":
            name,

        "predicted_production_mt":
            prediction,

        "target_achievement_pct":
            achievement,

        "production_risk":
            prod_risk,

        "equipment_risk":
            operational[
                "equipment_risk"
            ],

        "logistics_risk":
            operational[
                "logistics_risk"
            ],

        "weather_risk":
            operational[
                "weather_risk"
            ],

        "workforce_risk":
            operational[
                "workforce_risk"
            ],

        "overall_operational_risk":
            operational[
                "overall_operational_risk"
            ],

        "overall_risk_level":
            operational[
                "overall_risk_level"
            ],

        "scenario_row":
            scenario
    }


# ============================================================
# RANK SCENARIOS
# ============================================================

def calculate_scenario_score(
    result,
    baseline
):

    forecast_improvement = (

        result[
            "predicted_production_mt"
        ]

        -

        baseline[
            "predicted_production_mt"
        ]
    )

    risk_reduction = (

        baseline[
            "overall_operational_risk"
        ]

        -

        result[
            "overall_operational_risk"
        ]
    )

    achievement_improvement = (

        safe_float(
            result[
                "target_achievement_pct"
            ]
        )

        -

        safe_float(
            baseline[
                "target_achievement_pct"
            ]
        )
    )

    # Composite management score
    score = (

        forecast_improvement * 50

        +

        achievement_improvement * 2

        +

        risk_reduction * 1.5
    )

    return score


# ============================================================
# SENSITIVITY ANALYSIS
# ============================================================

def sensitivity_analysis(
    base_row,
    model,
    model_features
):

    print(
        "\n" + "=" * 70
    )

    print(
        "SENSITIVITY ANALYSIS"
    )

    print(
        "=" * 70
    )

    variables = {

        "equipment_availability_pct":
            [2, 5, 8, 10],

        "equipment_breakdown_hours":
            [-10, -20, -30, -40],

        "rail_rake_availability_pct":
            [2, 5, 8, 10],

        "evacuation_delay_hours":
            [-2, -4, -6, -8],

        "fmc_availability_pct":
            [2, 5, 8, 10]
    }

    rows = []

    for variable, changes in variables.items():

        print(
            f"\nTesting: {variable}"
        )

        for change in changes:

            scenario = apply_scenario(
                base_row,
                {
                    variable: change
                }
            )

            scenario_df = pd.DataFrame(
                [scenario]
            )

            scenario_features = prepare_features(
                scenario_df
            )

            scenario_features = (
                scenario_features.reindex(
                    columns=model_features,
                    fill_value=0
                )
            )

            prediction = predict(
                model,
                scenario_features,
                model_features
            )

            rows.append({

                "variable":
                    variable,

                "change":
                    change,

                "predicted_production_mt":
                    prediction
            })

    sensitivity_df = pd.DataFrame(
        rows
    )

    return sensitivity_df


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n" + "=" * 70
    )

    print(
        "PHASE 8B — ADVANCED AI SCENARIO SIMULATOR"
    )

    print(
        "=" * 70
    )

    # ========================================================
    # 1. MODEL
    # ========================================================

    print(
        "\n[1] Loading trained model..."
    )

    package = joblib.load(
        MODEL_PATH
    )

    model = package["model"]

    model_features = package[
        "features"
    ]

    print(
        f"Model: {package['model_name']}"
    )

    print(
        f"Model features: "
        f"{len(model_features)}"
    )

    # ========================================================
    # 2. DATA
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
        f"Rows available: {len(df)}"
    )

    # ========================================================
    # 3. FEATURES
    # ========================================================

    print(
        "\n[3] Preparing ML features..."
    )

    X = prepare_features(
        df
    )

    X = X.reindex(
        columns=model_features,
        fill_value=0
    )

    # ========================================================
    # 4. SELECT HIGH-RISK DEMO RECORD
    # ========================================================

    print(
        "\n[4] Selecting demonstration record..."
    )

    if os.path.exists(
        RISK_FILE
    ):

        risk_df = pd.read_csv(
            RISK_FILE
        )

        high_risk = risk_df[
            risk_df[
                "production_risk"
            ]
            ==
            "HIGH"
        ]

        if len(high_risk) > 0:

            selected = high_risk.iloc[0]

            selected_date = pd.to_datetime(
                selected["date"]
            )

            selected_subsidiary = (
                selected["subsidiary"]
            )

            matches = df[
                (
                    pd.to_datetime(
                        df["date"]
                    )
                    ==
                    selected_date
                )
                &
                (
                    df["subsidiary"]
                    ==
                    selected_subsidiary
                )
            ]

            if len(matches) > 0:

                row_index = matches.index[0]

            else:

                row_index = 0

        else:

            row_index = 0

    else:

        row_index = 0

    base_row = df.loc[
        row_index
    ].copy()

    base_features = X.loc[
        row_index
    ].to_frame().T

    # ========================================================
    # 5. BASELINE
    # ========================================================

    print(
        "\n[5] Calculating baseline..."
    )

    baseline_prediction = predict(
        model,
        base_features,
        model_features
    )

    target = safe_float(
        base_row[
            "production_target_mt"
        ]
    )

    baseline_achievement = (
        target_achievement(
            baseline_prediction,
            target
        )
    )

    baseline_production_risk = (
        production_risk(
            baseline_achievement
        )
    )

    baseline_operational = (
        operational_risk(
            base_row
        )
    )

    baseline = {

        "scenario":
            "BASELINE",

        "predicted_production_mt":
            baseline_prediction,

        "target_achievement_pct":
            baseline_achievement,

        "production_risk":
            baseline_production_risk,

        **baseline_operational
    }

    print(
        "\n" + "-" * 70
    )

    print(
        "BASELINE"
    )

    print(
        "-" * 70
    )

    print(
        f"Mine: "
        f"{base_row['subsidiary']}"
    )

    print(
        f"Date: "
        f"{base_row['date']}"
    )

    print(
        f"Current production: "
        f"{safe_float(base_row['monthly_production_mt']):.3f} MT"
    )

    print(
        f"Target: "
        f"{target:.3f} MT"
    )

    print(
        f"AI forecast: "
        f"{baseline_prediction:.3f} MT"
    )

    print(
        f"Target achievement: "
        f"{baseline_achievement:.2f}%"
    )

    print(
        f"Production risk: "
        f"{baseline_production_risk}"
    )

    print(
        f"Operational risk: "
        f"{baseline_operational['overall_operational_risk']:.2f}"
    )

    # ========================================================
    # 6. DEFINE INTERVENTIONS
    # ========================================================

    print(
        "\n[6] Creating intervention scenarios..."
    )

    scenarios = {

        "Maintenance Recovery": {

            "equipment_availability_pct":
                +8,

            "equipment_utilization_pct":
                +5,

            "equipment_breakdown_hours":
                -25
        },

        "Logistics Recovery": {

            "rail_rake_availability_pct":
                +8,

            "evacuation_delay_hours":
                -5,

            "fmc_availability_pct":
                +5
        },

        "Combined Recovery": {

            "equipment_availability_pct":
                +8,

            "equipment_utilization_pct":
                +5,

            "equipment_breakdown_hours":
                -25,

            "rail_rake_availability_pct":
                +8,

            "evacuation_delay_hours":
                -5,

            "fmc_availability_pct":
                +5
        }
    }

    # ========================================================
    # 7. RUN ALL SCENARIOS
    # ========================================================

    print(
        "\n[7] Running AI simulations..."
    )

    results = [
        baseline
    ]

    for name, changes in scenarios.items():

        print(
            f"  → Simulating: {name}"
        )

        result = run_scenario(
            name,
            base_row,
            changes,
            model,
            model_features
        )

        result["forecast_change_mt"] = (

            result[
                "predicted_production_mt"
            ]

            -

            baseline[
                "predicted_production_mt"
            ]
        )

        result[
            "forecast_change_pct"
        ] = (

            result[
                "forecast_change_mt"
            ]

            /

            max(
                baseline[
                    "predicted_production_mt"
                ],
                0.001
            )

            *

            100
        )

        result[
            "achievement_change_pp"
        ] = (

            safe_float(
                result[
                    "target_achievement_pct"
                ]
            )

            -

            safe_float(
                baseline[
                    "target_achievement_pct"
                ]
            )
        )

        result[
            "operational_risk_change"
        ] = (

            result[
                "overall_operational_risk"
            ]

            -

            baseline[
                "overall_operational_risk"
            ]
        )

        result[
            "risk_reduction"
        ] = (

            baseline[
                "overall_operational_risk"
            ]

            -

            result[
                "overall_operational_risk"
            ]
        )

        result[
            "scenario_score"
        ] = calculate_scenario_score(
            result,
            baseline
        )

        results.append(
            result
        )

    # ========================================================
    # 8. COMPARISON TABLE
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "SCENARIO COMPARISON"
    )

    print(
        "=" * 70
    )

    comparison_rows = []

    for result in results:

        comparison_rows.append({

            "scenario":
                result["scenario"],

            "forecast_mt":
                result[
                    "predicted_production_mt"
                ],

            "target_achievement_pct":
                result[
                    "target_achievement_pct"
                ],

            "production_risk":
                result[
                    "production_risk"
                ],

            "operational_risk":
                result[
                    "overall_operational_risk"
                ],

            "operational_level":
                result[
                    "overall_risk_level"
                ],

            "forecast_change_mt":
                result.get(
                    "forecast_change_mt",
                    0
                ),

            "forecast_change_pct":
                result.get(
                    "forecast_change_pct",
                    0
                ),

            "risk_reduction":
                result.get(
                    "risk_reduction",
                    0
                ),

            "scenario_score":
                result.get(
                    "scenario_score",
                    0
                )
        })

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    print(
        comparison_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.3f}"
        )
    )

    # ========================================================
    # 9. BEST SCENARIO
    # ========================================================

    scenario_only = comparison_df[
        comparison_df[
            "scenario"
        ]
        !=
        "BASELINE"
    ]

    best = scenario_only.sort_values(
        "scenario_score",
        ascending=False
    ).iloc[0]

    print(
        "\n" + "=" * 70
    )

    print(
        "AI-RECOMMENDED INTERVENTION"
    )

    print(
        "=" * 70
    )

    print(
        f"\nRecommended scenario:"
        f" {best['scenario']}"
    )

    print(
        f"Forecast:"
        f" {best['forecast_mt']:.3f} MT"
    )

    print(
        f"Forecast improvement:"
        f" {best['forecast_change_mt']:+.3f} MT"
    )

    print(
        f"Target achievement:"
        f" {best['target_achievement_pct']:.2f}%"
    )

    print(
        f"Operational risk reduction:"
        f" {best['risk_reduction']:.2f}"
    )

    print(
        f"Production risk:"
        f" {best['production_risk']}"
    )

    print(
        f"Operational level:"
        f" {best['operational_level']}"
    )

    # ========================================================
    # 10. SENSITIVITY
    # ========================================================

    sensitivity_df = sensitivity_analysis(
        base_row,
        model,
        model_features
    )

    # ========================================================
    # 11. SAVE RESULTS
    # ========================================================

    print(
        "\n[8] Saving scenario results..."
    )

    os.makedirs(
        os.path.dirname(
            OUTPUT_FILE
        ),
        exist_ok=True
    )

    comparison_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    sensitivity_file = (
        r"D:\CoalMineAI\outputs"
        r"\scenario_sensitivity.csv"
    )

    sensitivity_df.to_csv(
        sensitivity_file,
        index=False
    )

    # ========================================================
    # 12. FINAL
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "PHASE 8B COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nScenario comparison saved to:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        "\nSensitivity analysis saved to:"
    )

    print(
        sensitivity_file
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()