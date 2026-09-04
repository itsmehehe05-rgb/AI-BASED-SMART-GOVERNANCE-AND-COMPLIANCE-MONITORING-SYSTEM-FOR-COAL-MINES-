import os
import joblib
import numpy as np
import pandas as pd

from data_loader import load_training_data
from preprocessing import preprocess_data
from feature_engineering import create_features


# ============================================================
# PHASE 8C
# AI OPTIMAL INTERVENTION ENGINE
# ============================================================

MODEL_PATH = r"D:\CoalMineAI\models\production_forecaster.pkl"

OUTPUT_DIR = r"D:\CoalMineAI\outputs"

RESULT_FILE = os.path.join(
    OUTPUT_DIR,
    "optimal_intervention_results.csv"
)

RANKING_FILE = os.path.join(
    OUTPUT_DIR,
    "intervention_ranking.csv"
)

SENSITIVITY_FILE = os.path.join(
    OUTPUT_DIR,
    "intervention_sensitivity.csv"
)

TARGET = "target_next_month_production_mt"


# ============================================================
# UTILITY
# ============================================================

def safe_float(value, default=0.0):

    try:
        value = float(value)

        if not np.isfinite(value):
            return default

        return value

    except Exception:
        return default


def clamp(value, low, high):

    return max(
        low,
        min(value, high)
    )


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(df):

    data = df.copy()

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

    categorical = X.select_dtypes(
        include=[
            "object",
            "string",
            "category"
        ]
    ).columns.tolist()

    if categorical:

        X = pd.get_dummies(
            X,
            columns=categorical,
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
# MODEL PREDICTION
# ============================================================

def predict_model(
    model,
    row,
    model_features
):

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
# OPERATIONAL RISK
# ============================================================

def calculate_operational_risk(row):

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

    # --------------------------------------------------------
    # EQUIPMENT
    # --------------------------------------------------------

    equipment_risk = (

        (100 - equipment_availability)
        * 0.45

        +

        clamp(
            breakdown_hours,
            0,
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
    # LOGISTICS
    # --------------------------------------------------------

    logistics_risk = (

        (100 - rail_availability)
        * 0.45

        +

        clamp(
            evacuation_delay * 2,
            0,
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
    # WEATHER
    # --------------------------------------------------------

    weather_risk = clamp(
        rainfall / 5,
        0,
        100
    )

    # --------------------------------------------------------
    # OVERALL
    # --------------------------------------------------------

    overall = (

        equipment_risk * 0.45

        +

        logistics_risk * 0.35

        +

        weather_risk * 0.20
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

        "overall_operational_risk":
            overall,

        "overall_risk_level":
            level
    }


# ============================================================
# TARGET ACHIEVEMENT
# ============================================================

def target_achievement(
    prediction,
    target
):

    target = safe_float(target)

    if target <= 0:

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

def production_risk(
    achievement
):

    if pd.isna(achievement):

        return "UNRELIABLE"

    if achievement >= 100:

        return "LOW"

    if achievement >= 95:

        return "MEDIUM"

    return "HIGH"


# ============================================================
# APPLY INTERVENTION
# ============================================================

def apply_intervention(
    base_row,
    intervention
):

    scenario = base_row.copy()

    for column, change in intervention.items():

        if column not in scenario.index:

            continue

        current = safe_float(
            scenario[column]
        )

        new_value = (
            current + change
        )

        if column.endswith("_pct"):

            new_value = clamp(
                new_value,
                0,
                100
            )

        elif column.endswith("_hours"):

            new_value = max(
                new_value,
                0
            )

        elif column == "workforce_strength":

            new_value = max(
                new_value,
                0
            )

        scenario[column] = new_value

    return scenario


# ============================================================
# EVALUATE INTERVENTION
# ============================================================

def evaluate_intervention(
    name,
    base_row,
    intervention,
    model,
    model_features,
    baseline
):

    scenario = apply_intervention(
        base_row,
        intervention
    )

    scenario_df = pd.DataFrame(
        [scenario]
    )

    scenario_features = (
        prepare_features(
            scenario_df
        )
    )

    scenario_features = (
        scenario_features.reindex(
            columns=model_features,
            fill_value=0
        )
    )

    prediction = predict_model(
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

    risk = calculate_operational_risk(
        scenario
    )

    forecast_change = (
        prediction
        -
        baseline[
            "forecast"
        ]
    )

    achievement_change = (
        achievement
        -
        baseline[
            "achievement"
        ]
    )

    risk_reduction = (
        baseline[
            "operational_risk"
        ]
        -
        risk[
            "overall_operational_risk"
        ]
    )

    # --------------------------------------------------------
    # DECISION SCORE
    # --------------------------------------------------------

    score = (

        forecast_change * 50

        +

        achievement_change * 2

        +

        risk_reduction * 1.5
    )

    return {

        "intervention":
            name,

        "predicted_production_mt":
            prediction,

        "target_achievement_pct":
            achievement,

        "production_risk":
            prod_risk,

        "equipment_risk":
            risk[
                "equipment_risk"
            ],

        "logistics_risk":
            risk[
                "logistics_risk"
            ],

        "weather_risk":
            risk[
                "weather_risk"
            ],

        "overall_operational_risk":
            risk[
                "overall_operational_risk"
            ],

        "operational_level":
            risk[
                "overall_risk_level"
            ],

        "forecast_change_mt":
            forecast_change,

        "achievement_change_pp":
            achievement_change,

        "risk_reduction":
            risk_reduction,

        "decision_score":
            score
    }


# ============================================================
# CREATE INTERVENTION SEARCH SPACE
# ============================================================

def create_intervention_space():

    interventions = []

    # --------------------------------------------------------
    # MAINTENANCE INTERVENTIONS
    # --------------------------------------------------------

    for equipment_gain in [
        2,
        5,
        8,
        10
    ]:

        for breakdown_reduction in [
            10,
            20,
            30,
            40
        ]:

            interventions.append({

                "name":
                    f"Maintenance "
                    f"A+{equipment_gain}% "
                    f"B-{breakdown_reduction}h",

                "changes": {

                    "equipment_availability_pct":
                        equipment_gain,

                    "equipment_utilization_pct":
                        min(
                            equipment_gain * 0.5,
                            8
                        ),

                    "equipment_breakdown_hours":
                        -breakdown_reduction
                }
            })

    # --------------------------------------------------------
    # LOGISTICS INTERVENTIONS
    # --------------------------------------------------------

    for rail_gain in [
        2,
        5,
        8,
        10
    ]:

        for delay_reduction in [
            2,
            5,
            8
        ]:

            interventions.append({

                "name":
                    f"Logistics "
                    f"R+{rail_gain}% "
                    f"D-{delay_reduction}h",

                "changes": {

                    "rail_rake_availability_pct":
                        rail_gain,

                    "evacuation_delay_hours":
                        -delay_reduction,

                    "fmc_availability_pct":
                        min(
                            rail_gain * 0.5,
                            8
                        )
                }
            })

    # --------------------------------------------------------
    # COMBINED INTERVENTIONS
    # --------------------------------------------------------

    for equipment_gain in [
        5,
        8,
        10
    ]:

        for rail_gain in [
            5,
            8,
            10
        ]:

            interventions.append({

                "name":
                    f"Combined "
                    f"E+{equipment_gain}% "
                    f"R+{rail_gain}%",

                "changes": {

                    "equipment_availability_pct":
                        equipment_gain,

                    "equipment_utilization_pct":
                        min(
                            equipment_gain * 0.5,
                            8
                        ),

                    "equipment_breakdown_hours":
                        -30,

                    "rail_rake_availability_pct":
                        rail_gain,

                    "evacuation_delay_hours":
                        -5,

                    "fmc_availability_pct":
                        min(
                            rail_gain * 0.5,
                            8
                        )
                }
            })

    return interventions


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
        "INTERVENTION SENSITIVITY ANALYSIS"
    )

    print(
        "=" * 70
    )

    tests = {

        "equipment_availability_pct":
            [2, 5, 8, 10],

        "equipment_breakdown_hours":
            [-10, -20, -30, -40],

        "equipment_utilization_pct":
            [2, 5, 8],

        "rail_rake_availability_pct":
            [2, 5, 8, 10],

        "evacuation_delay_hours":
            [-2, -5, -8],

        "fmc_availability_pct":
            [2, 5, 8]
    }

    rows = []

    for variable, values in tests.items():

        for change in values:

            scenario = apply_intervention(
                base_row,
                {
                    variable: change
                }
            )

            scenario_df = pd.DataFrame(
                [scenario]
            )

            scenario_features = (
                prepare_features(
                    scenario_df
                )
            )

            scenario_features = (
                scenario_features.reindex(
                    columns=model_features,
                    fill_value=0
                )
            )

            prediction = predict_model(
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

    return pd.DataFrame(
        rows
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n" + "=" * 70
    )

    print(
        "PHASE 8C — AI OPTIMAL INTERVENTION ENGINE"
    )

    print(
        "=" * 70
    )

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

    model_features = package[
        "features"
    ]

    print(
        f"Model: "
        f"{package['model_name']}"
    )

    print(
        f"Features: "
        f"{len(model_features)}"
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
    # 3. PREPARE FEATURES
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
    # 4. SELECT DEMO RECORD
    # ========================================================

    print(
        "\n[4] Selecting demonstration mine..."
    )

    # Select a realistic production record
    # with a meaningful target.

    candidates = df[
        df[
            "production_target_mt"
        ]
        >
        1
    ]

    if len(candidates) == 0:

        row_index = 0

    else:

        row_index = candidates.index[0]

    base_row = df.loc[
        row_index
    ].copy()

    base_features = X.loc[
        row_index
    ].to_frame().T

    print(
        f"Mine: "
        f"{base_row['subsidiary']}"
    )

    print(
        f"Date: "
        f"{base_row['date']}"
    )

    # ========================================================
    # 5. BASELINE
    # ========================================================

    print(
        "\n[5] Calculating baseline..."
    )

    baseline_forecast = predict_model(
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
            baseline_forecast,
            target
        )
    )

    baseline_risk = (
        calculate_operational_risk(
            base_row
        )
    )

    baseline = {

        "forecast":
            baseline_forecast,

        "achievement":
            baseline_achievement,

        "operational_risk":
            baseline_risk[
                "overall_operational_risk"
            ]
    }

    print(
        "\nBASELINE"
    )

    print(
        "-" * 70
    )

    print(
        f"Production: "
        f"{safe_float(base_row['monthly_production_mt']):.3f} MT"
    )

    print(
        f"Target: "
        f"{target:.3f} MT"
    )

    print(
        f"AI Forecast: "
        f"{baseline_forecast:.3f} MT"
    )

    print(
        f"Target Achievement: "
        f"{baseline_achievement:.2f}%"
    )

    print(
        f"Operational Risk: "
        f"{baseline_risk['overall_operational_risk']:.2f}"
    )

    # ========================================================
    # 6. SEARCH SPACE
    # ========================================================

    print(
        "\n[6] Creating intervention search space..."
    )

    interventions = (
        create_intervention_space()
    )

    print(
        f"Intervention combinations: "
        f"{len(interventions)}"
    )

    # ========================================================
    # 7. OPTIMIZATION
    # ========================================================

    print(
        "\n[7] Evaluating interventions..."
    )

    results = []

    for i, intervention in enumerate(
        interventions,
        start=1
    ):

        result = evaluate_intervention(

            intervention["name"],

            base_row,

            intervention["changes"],

            model,

            model_features,

            baseline
        )

        results.append(
            result
        )

        if i % 25 == 0:

            print(
                f"  Evaluated "
                f"{i}/"
                f"{len(interventions)}"
            )

    results_df = pd.DataFrame(
        results
    )

    # ========================================================
    # 8. RANK
    # ========================================================

    print(
        "\n[8] Ranking interventions..."
    )

    results_df = results_df.sort_values(
        "decision_score",
        ascending=False
    ).reset_index(
        drop=True
    )

    results_df[
        "rank"
    ] = np.arange(
        1,
        len(results_df) + 1
    )

    # ========================================================
    # 9. BEST INTERVENTION
    # ========================================================

    best = results_df.iloc[0]

    print(
        "\n" + "=" * 70
    )

    print(
        "AI OPTIMAL INTERVENTION"
    )

    print(
        "=" * 70
    )

    print(
        f"\nRecommended action:"
    )

    print(
        f"  {best['intervention']}"
    )

    print(
        "\nExpected impact:"
    )

    print(
        f"  Forecast:"
        f" {best['predicted_production_mt']:.3f} MT"
    )

    print(
        f"  Forecast change:"
        f" {best['forecast_change_mt']:+.3f} MT"
    )

    print(
        f"  Target achievement:"
        f" {best['target_achievement_pct']:.2f}%"
    )

    print(
        f"  Achievement change:"
        f" {best['achievement_change_pp']:+.2f} pp"
    )

    print(
        f"  Operational risk:"
        f" {best['overall_operational_risk']:.2f}"
    )

    print(
        f"  Risk reduction:"
        f" {best['risk_reduction']:.2f}"
    )

    print(
        f"  Production risk:"
        f" {best['production_risk']}"
    )

    print(
        f"  Operational level:"
        f" {best['operational_level']}"
    )

    # ========================================================
    # 10. TOP 10
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "TOP 10 AI-RECOMMENDED INTERVENTIONS"
    )

    print(
        "=" * 70
    )

    display_columns = [

        "rank",

        "intervention",

        "predicted_production_mt",

        "target_achievement_pct",

        "forecast_change_mt",

        "achievement_change_pp",

        "overall_operational_risk",

        "risk_reduction",

        "production_risk",

        "operational_level",

        "decision_score"
    ]

    print(
        results_df[
            display_columns
        ]
        .head(10)
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:.3f}"
        )
    )

    # ========================================================
    # 11. SENSITIVITY
    # ========================================================

    sensitivity_df = sensitivity_analysis(
        base_row,
        model,
        model_features
    )

    # ========================================================
    # 12. SAVE
    # ========================================================

    print(
        "\n[9] Saving outputs..."
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    results_df.to_csv(
        RESULT_FILE,
        index=False
    )

    results_df.head(50).to_csv(
        RANKING_FILE,
        index=False
    )

    sensitivity_df.to_csv(
        SENSITIVITY_FILE,
        index=False
    )

    # ========================================================
    # 13. MANAGEMENT INTERPRETATION
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "AI MANAGEMENT RECOMMENDATION"
    )

    print(
        "=" * 70
    )

    risk_reduction = safe_float(
        best[
            "risk_reduction"
        ]
    )

    forecast_change = safe_float(
        best[
            "forecast_change_mt"
        ]
    )

    achievement_change = safe_float(
        best[
            "achievement_change_pp"
        ]
    )

    if (
        forecast_change > 0
        and
        risk_reduction > 0
    ):

        print(
            "\n✓ Recommended intervention "
            "improves forecast AND reduces risk."
        )

    elif risk_reduction > 0:

        print(
            "\n✓ Recommended intervention "
            "primarily reduces operational risk."
        )

    elif forecast_change > 0:

        print(
            "\n✓ Recommended intervention "
            "primarily improves production forecast."
        )

    else:

        print(
            "\n⚠ No intervention produced "
            "a positive combined impact."
        )

    if achievement_change > 0:

        print(
            "✓ Target achievement is improved."
        )

    else:

        print(
            "⚠ Target achievement does not improve "
            "under the selected intervention."
        )

    print(
        "\nThis is an AI decision-support "
        "recommendation, not a guaranteed outcome."
    )

    # ========================================================
    # FINAL
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "PHASE 8C COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nOptimal intervention results:"
    )

    print(
        RESULT_FILE
    )

    print(
        "\nIntervention ranking:"
    )

    print(
        RANKING_FILE
    )

    print(
        "\nSensitivity analysis:"
    )

    print(
        SENSITIVITY_FILE
    )


if __name__ == "__main__":

    main()