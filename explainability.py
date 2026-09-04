import os
import joblib
import numpy as np
import pandas as pd
import shap

from data_loader import load_training_data
from preprocessing import preprocess_data
from feature_engineering import create_features


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = (
    r"D:\CoalMineAI\models\production_forecaster.pkl"
)

OUTPUT_PATH = (
    r"D:\CoalMineAI\outputs\shap_explanations.csv"
)

TARGET = "target_next_month_production_mt"


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(df):

    data = df.copy()

    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce"
    )

    # Remove target and metadata columns
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

    # Detect categorical columns
    categorical_columns = X.select_dtypes(
        include=[
            "object",
            "string",
            "category"
        ]
    ).columns.tolist()

    # One-hot encode categorical data
    if categorical_columns:

        X = pd.get_dummies(
            X,
            columns=categorical_columns,
            drop_first=False
        )

    # Convert everything to numeric
    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Replace infinity
    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Fill missing values
    X = X.fillna(
        X.median(numeric_only=True)
    )

    X = X.fillna(0)

    X.columns = X.columns.astype(str)

    return X


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print(
        "PHASE 6 — EXPLAINABLE AI"
    )
    print("=" * 70)

    # ========================================================
    # 1. LOAD TRAINED MODEL
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

    print(
        f"Model features: "
        f"{len(model_features)}"
    )

    # ========================================================
    # 2. LOAD DATA
    # ========================================================

    print(
        "\n[2] Loading dataset..."
    )

    df = load_training_data()

    print(
        f"Rows loaded: {len(df)}"
    )

    # ========================================================
    # 3. PREPROCESSING
    # ========================================================

    print(
        "\n[3] Running preprocessing..."
    )

    df = preprocess_data(
        df
    )

    # ========================================================
    # 4. FEATURE ENGINEERING
    # ========================================================

    print(
        "\n[4] Creating advanced features..."
    )

    df = create_features(
        df
    )

    print(
        f"Rows after feature engineering: "
        f"{len(df)}"
    )

    # ========================================================
    # 5. PREPARE ML FEATURES
    # ========================================================

    print(
        "\n[5] Preparing ML features..."
    )

    X = prepare_features(
        df
    )

    # Make sure feature columns exactly match training
    X = X.reindex(
        columns=model_features,
        fill_value=0
    )

    print(
        f"Final feature count: {X.shape[1]}"
    )

    print(
        f"Feature rows: {X.shape[0]}"
    )

    # ========================================================
    # 6. CREATE PREDICTIONS
    # ========================================================

    print(
        "\n[6] Generating predictions..."
    )

    predictions = model.predict(
        X
    )

    # Production cannot be negative
    predictions = np.clip(
        predictions,
        0,
        None
    )

    df[
        "predicted_production_mt"
    ] = predictions

    print(
        "Predictions generated."
    )

    # ========================================================
    # 7. CREATE SHAP EXPLAINER
    # ========================================================

    print(
        "\n[7] Creating SHAP explainer..."
    )

    explainer = shap.TreeExplainer(
        model
    )

    print(
        "SHAP TreeExplainer created."
    )

    # ========================================================
    # 8. CALCULATE SHAP VALUES
    # ========================================================

    print(
        "\n[8] Calculating SHAP values..."
    )

    shap_values = explainer.shap_values(
        X
    )

    print(
        "SHAP values calculated."
    )

    # ========================================================
    # 9. GLOBAL SHAP IMPORTANCE
    # ========================================================

    print(
        "\n[9] Calculating global SHAP importance..."
    )

    mean_abs_shap = np.abs(
        shap_values
    ).mean(
        axis=0
    )

    global_importance = pd.DataFrame({

        "feature":
            X.columns,

        "mean_abs_shap":
            mean_abs_shap

    })

    global_importance[
        "importance_pct"
    ] = (
        global_importance[
            "mean_abs_shap"
        ]
        /
        global_importance[
            "mean_abs_shap"
        ].sum()
        *
        100
    )

    global_importance = (
        global_importance
        .sort_values(
            "mean_abs_shap",
            ascending=False
        )
    )

    # ========================================================
    # 10. PRINT GLOBAL DRIVERS
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "TOP 20 SHAP MODEL DRIVERS"
    )

    print(
        "=" * 70
    )

    print(
        global_importance
        .head(20)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # 11. CREATE LOCAL EXPLANATIONS
    # ========================================================

    print(
        "\n[10] Creating individual explanations..."
    )

    explanation_rows = []

    # We will store the top 10 drivers
    # for every row.

    for row_index in range(
        len(df)
    ):

        row = df.iloc[
            row_index
        ]

        row_shap = shap_values[
            row_index
        ]

        # Get feature positions by absolute impact
        ranked_indices = np.argsort(
            np.abs(row_shap)
        )[::-1]

        # Top 10
        top_indices = ranked_indices[
            :10
        ]

        for rank, feature_index in enumerate(
            top_indices,
            start=1
        ):

            feature_name = X.columns[
                feature_index
            ]

            shap_value = float(
                row_shap[
                    feature_index
                ]
            )

            feature_value = X.iloc[
                row_index,
                feature_index
            ]

            if shap_value > 0:

                direction = "INCREASES_FORECAST"

            elif shap_value < 0:

                direction = "DECREASES_FORECAST"

            else:

                direction = "NEUTRAL"

            explanation_rows.append({

                "date":
                    row["date"],

                "subsidiary":
                    row["subsidiary"],

                "predicted_production_mt":
                    row[
                        "predicted_production_mt"
                    ],

                "production_target_mt":
                    row[
                        "production_target_mt"
                    ],

                "feature":
                    feature_name,

                "feature_value":
                    feature_value,

                "shap_value":
                    shap_value,

                "absolute_shap_value":
                    abs(shap_value),

                "direction":
                    direction,

                "rank":
                    rank
            })

    # ========================================================
    # 12. SAVE LOCAL EXPLANATIONS
    # ========================================================

    explanation_df = pd.DataFrame(
        explanation_rows
    )

    os.makedirs(
        os.path.dirname(
            OUTPUT_PATH
        ),
        exist_ok=True
    )

    explanation_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # ========================================================
    # 13. SHOW EXAMPLE EXPLANATION
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "EXAMPLE AI EXPLANATION"
    )

    print(
        "=" * 70
    )

    example_index = 0

    example_row = df.iloc[
        example_index
    ]

    print(
        f"\nMine: "
        f"{example_row['subsidiary']}"
    )

    print(
        f"Date: "
        f"{example_row['date']}"
    )

    print(
        f"Predicted production: "
        f"{example_row['predicted_production_mt']:.3f} MT"
    )

    print(
        f"Production target: "
        f"{example_row['production_target_mt']:.3f} MT"
    )

    print(
        "\nTop factors:"
    )

    example_explanations = (
        explanation_df[
            explanation_df[
                "date"
            ] == example_row[
                "date"
            ]
        ]
        .head(10)
    )

    print(
        example_explanations[
            [
                "feature",
                "feature_value",
                "shap_value",
                "direction"
            ]
        ]
        .to_string(
            index=False
        )
    )

    # ========================================================
    # 14. FINAL SUMMARY
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "PHASE 6 COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nSHAP explanations saved to:"
    )

    print(
        OUTPUT_PATH
    )

    print(
        "\nTotal explanation records:"
    )

    print(
        len(explanation_df)
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()