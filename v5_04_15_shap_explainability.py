"""
CoalMineAI - Step 4.15
SHAP Explainability for Final Predictive Risk Model

IMPORTANT:
- Uses the frozen final model.
- Does NOT retrain the model.
- Does NOT modify the model.
- Extracts the tree estimator from the sklearn Pipeline.
"""

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(r"D:\CoalMineAI")

DATA_PATH = (
    BASE_DIR
    / "outputs"
    / "v5"
    / "v5_04_10_selected_features_dataset.csv"
)

FINAL_DIR = BASE_DIR / "outputs" / "v5" / "FINAL"

MODEL_PATH = FINAL_DIR / "final_risk_model.joblib"
PREPROCESSOR_PATH = FINAL_DIR / "final_preprocessor.joblib"
CALIBRATOR_PATH = FINAL_DIR / "final_calibrator.joblib"
CONFIG_PATH = FINAL_DIR / "final_model_config.json"


GLOBAL_OUTPUT = FINAL_DIR / "shap_global_feature_importance.csv"
RECORD_OUTPUT = FINAL_DIR / "shap_record_explanations.csv"
LATEST_OUTPUT = FINAL_DIR / "shap_latest_mine_explanations.csv"
SUMMARY_OUTPUT = FINAL_DIR / "shap_summary.csv"
VALUES_OUTPUT = FINAL_DIR / "shap_feature_values.csv"


RANDOM_STATE = 42
MAX_SHAP_ROWS = 500
TOP_K = 5


# ============================================================
# HELPERS
# ============================================================

def find_column(df, candidates):

    lower_map = {
        str(c).lower(): c
        for c in df.columns
    }

    for candidate in candidates:

        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    return None


def clean_feature_name(name):

    name = str(name)

    # Remove sklearn transformer prefixes where present
    for prefix in [
        "num__",
        "cat__",
        "remainder__"
    ]:
        if name.startswith(prefix):
            name = name[len(prefix):]

    return name.replace("_", " ").strip()


def risk_direction(value):

    if value > 0:
        return "INCREASES_ESCALATION_RISK"

    if value < 0:
        return "DECREASES_ESCALATION_RISK"

    return "NEUTRAL"


def make_reason(feature, shap_value, feature_value):

    readable = clean_feature_name(feature)

    if shap_value > 0:
        direction = "increases"

    elif shap_value < 0:
        direction = "reduces"

    else:
        direction = "has little effect on"

    if pd.notna(feature_value):

        try:
            value_text = f"{float(feature_value):.3f}"

            return (
                f"{readable} ({value_text}) "
                f"{direction} predicted escalation risk."
            )

        except Exception:
            pass

    return (
        f"{readable} {direction} "
        f"predicted escalation risk."
    )


# ============================================================
# FIND TREE MODEL INSIDE PIPELINE
# ============================================================

def extract_tree_model(model):

    """
    Extract the actual tree-based estimator from the
    saved sklearn Pipeline.

    Supports:
        RandomForestClassifier
        ExtraTreesClassifier
        GradientBoostingClassifier
        HistGradientBoostingClassifier
        XGBClassifier
        etc.
    """

    # --------------------------------------------------------
    # Direct tree model
    # --------------------------------------------------------

    if hasattr(model, "estimators_"):

        return model

    # --------------------------------------------------------
    # sklearn Pipeline
    # --------------------------------------------------------

    if hasattr(model, "named_steps"):

        print("\nPipeline steps:")

        for name, step in model.named_steps.items():

            print(
                f"  - {name}: "
                f"{type(step).__name__}"
            )

        # Search from final step backwards
        steps = list(
            model.named_steps.items()
        )

        for name, step in reversed(steps):

            if hasattr(step, "estimators_"):

                print(
                    f"\nTree estimator found: "
                    f"{name}"
                )

                print(
                    f"Estimator type: "
                    f"{type(step).__name__}"
                )

                return step

    raise TypeError(
        "Could not locate a tree-based estimator "
        "inside the saved model."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("CoalMineAI - STEP 4.15 SHAP EXPLAINABILITY")
    print("=" * 80)

    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    for path in [
        DATA_PATH,
        MODEL_PATH,
        PREPROCESSOR_PATH,
        CALIBRATOR_PATH,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

    FINAL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # IMPORT SHAP
    # --------------------------------------------------------

    try:

        import shap

    except ImportError:

        print("\nSHAP is not installed.")

        print(
            "\nRun:"
            "\n  pip install shap"
        )

        return

    print("\nSHAP version:", shap.__version__)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print("\n[1] Loading data...")

    df = pd.read_csv(DATA_PATH)

    print(
        f"Loaded dataset: {df.shape}"
    )

    # --------------------------------------------------------
    # IDENTIFY COLUMNS
    # --------------------------------------------------------

    target_col = find_column(
        df,
        ["material_escalation"]
    )

    subsidiary_col = find_column(
        df,
        ["subsidiary"]
    )

    record_date_col = find_column(
        df,
        ["record_date"]
    )

    date_col = find_column(
        df,
        ["date"]
    )

    if target_col is None:

        raise ValueError(
            "material_escalation column not found."
        )

    # --------------------------------------------------------
    # LOAD FROZEN ARTIFACTS
    # --------------------------------------------------------

    print(
        "\n[2] Loading frozen final model..."
    )

    model = joblib.load(
        MODEL_PATH
    )

    preprocessor = joblib.load(
        PREPROCESSOR_PATH
    )

    calibrator = joblib.load(
        CALIBRATOR_PATH
    )

    print(
        "Full model:",
        type(model).__name__
    )

    print(
        "Preprocessor:",
        type(preprocessor).__name__
    )

    print(
        "Calibrator:",
        type(calibrator).__name__
    )

    # --------------------------------------------------------
    # EXTRACT TREE MODEL
    # --------------------------------------------------------

    print(
        "\n[3] Extracting tree estimator..."
    )

    tree_model = extract_tree_model(
        model
    )

    # --------------------------------------------------------
    # BUILD FEATURE MATRIX
    # --------------------------------------------------------

    metadata = {
        target_col
    }

    if record_date_col:
        metadata.add(
            record_date_col
        )

    if date_col:
        metadata.add(
            date_col
        )

    # IMPORTANT:
    # subsidiary is NOT removed because it was part
    # of the final model.

    feature_columns = [
        c
        for c in df.columns
        if c not in metadata
    ]

    feature_columns = [
        c
        for c in feature_columns
        if c not in [
            "material_escalation_reason",
            "material_escalation_source"
        ]
    ]

    X_raw = df[
        feature_columns
    ].copy()

    print(
        "\nCandidate feature columns:",
        len(feature_columns)
    )

    # --------------------------------------------------------
    # APPLY SAME PREPROCESSOR
    # --------------------------------------------------------

    print(
        "\n[4] Transforming features..."
    )

    X_transformed = (
        preprocessor.transform(
            X_raw
        )
    )

    if hasattr(
        X_transformed,
        "toarray"
    ):

        X_transformed = (
            X_transformed.toarray()
        )

    X_transformed = np.asarray(
        X_transformed
    )

    print(
        "Transformed feature matrix:",
        X_transformed.shape
    )

    # --------------------------------------------------------
    # FEATURE NAMES
    # --------------------------------------------------------

    try:

        feature_names = (
            preprocessor
            .get_feature_names_out()
        )

    except Exception:

        feature_names = np.array(
            [
                f"feature_{i}"
                for i in range(
                    X_transformed.shape[1]
                )
            ]
        )

    feature_names = np.asarray(
        feature_names
    )

    print(
        "Transformed feature names:",
        len(feature_names)
    )

    # --------------------------------------------------------
    # SELECT SHAP ROWS
    # --------------------------------------------------------

    n_rows = len(df)

    if n_rows > MAX_SHAP_ROWS:

        rng = np.random.default_rng(
            RANDOM_STATE
        )

        shap_indices = np.sort(
            rng.choice(
                n_rows,
                size=MAX_SHAP_ROWS,
                replace=False
            )
        )

    else:

        shap_indices = np.arange(
            n_rows
        )

    X_shap = X_transformed[
        shap_indices
    ]

    print(
        "\n[5] Calculating SHAP values..."
    )

    print(
        f"Rows explained: "
        f"{len(shap_indices)}"
    )

    # --------------------------------------------------------
    # TREE EXPLAINER
    # --------------------------------------------------------

    explainer = shap.TreeExplainer(
        tree_model
    )

    shap_result = explainer.shap_values(
        X_shap
    )

    # --------------------------------------------------------
    # HANDLE SHAP OUTPUT VERSIONS
    # --------------------------------------------------------

    if isinstance(
        shap_result,
        list
    ):

        if len(shap_result) == 2:

            shap_values = np.asarray(
                shap_result[1]
            )

        else:

            shap_values = np.asarray(
                shap_result[-1]
            )

    else:

        shap_values = np.asarray(
            shap_result
        )

        if shap_values.ndim == 3:

            # samples x features x classes
            if (
                shap_values.shape[-1]
                == 2
            ):

                shap_values = (
                    shap_values[:, :, 1]
                )

            # classes x samples x features
            elif (
                shap_values.shape[0]
                == 2
            ):

                shap_values = (
                    shap_values[1]
                )

    shap_values = np.asarray(
        shap_values
    )

    print(
        "SHAP matrix:",
        shap_values.shape
    )

    if (
        shap_values.ndim != 2
    ):

        raise ValueError(
            "Unexpected SHAP dimensions: "
            f"{shap_values.shape}"
        )

    if (
        shap_values.shape[1]
        != len(feature_names)
    ):

        raise ValueError(
            "SHAP feature count mismatch.\n"
            f"SHAP features: "
            f"{shap_values.shape[1]}\n"
            f"Feature names: "
            f"{len(feature_names)}"
        )

    # --------------------------------------------------------
    # GLOBAL IMPORTANCE
    # --------------------------------------------------------

    print(
        "\n[6] Creating global SHAP importance..."
    )

    mean_abs = np.mean(
        np.abs(shap_values),
        axis=0
    )

    mean_signed = np.mean(
        shap_values,
        axis=0
    )

    global_df = pd.DataFrame({

        "feature":
            feature_names,

        "mean_abs_shap":
            mean_abs,

        "mean_shap":
            mean_signed,
    })

    total = (
        global_df[
            "mean_abs_shap"
        ].sum()
    )

    if total > 0:

        global_df[
            "importance_pct"
        ] = (
            global_df[
                "mean_abs_shap"
            ]
            / total
            * 100
        )

    else:

        global_df[
            "importance_pct"
        ] = 0.0

    global_df[
        "direction"
    ] = np.where(

        global_df[
            "mean_shap"
        ] > 0,

        "INCREASES_RISK",

        np.where(

            global_df[
                "mean_shap"
            ] < 0,

            "DECREASES_RISK",

            "NEUTRAL"
        )
    )

    global_df[
        "readable_feature"
    ] = (
        global_df[
            "feature"
        ]
        .apply(
            clean_feature_name
        )
    )

    global_df = (
        global_df
        .sort_values(
            "mean_abs_shap",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    global_df.insert(
        0,
        "rank",
        np.arange(
            1,
            len(global_df) + 1
        )
    )

    global_df.to_csv(
        GLOBAL_OUTPUT,
        index=False
    )

    print(
        "\nTop 20 SHAP drivers:"
    )

    print(
        global_df[
            [
                "rank",
                "readable_feature",
                "importance_pct",
                "direction"
            ]
        ]
        .head(20)
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # RECORD-LEVEL EXPLANATIONS
    # --------------------------------------------------------

    print(
        "\n[7] Creating record-level explanations..."
    )

    explanation_rows = []

    for local_idx, original_idx in enumerate(
        shap_indices
    ):

        values = shap_values[
            local_idx
        ]

        order = np.argsort(
            np.abs(values)
        )[::-1]

        top_indices = order[
            :TOP_K
        ]

        for rank, feature_idx in enumerate(
            top_indices,
            start=1
        ):

            feature = feature_names[
                feature_idx
            ]

            shap_value = float(
                values[
                    feature_idx
                ]
            )

            feature_value = (
                X_shap[
                    local_idx,
                    feature_idx
                ]
            )

            try:

                feature_value = float(
                    feature_value
                )

            except Exception:

                feature_value = np.nan

            row = {

                "record_index":
                    int(original_idx),

                "driver_rank":
                    rank,

                "feature":
                    feature,

                "readable_feature":
                    clean_feature_name(
                        feature
                    ),

                "feature_value":
                    feature_value,

                "shap_value":
                    shap_value,

                "absolute_shap_value":
                    abs(shap_value),

                "direction":
                    risk_direction(
                        shap_value
                    ),

                "reason":
                    make_reason(
                        feature,
                        shap_value,
                        feature_value
                    ),
            }

            if subsidiary_col:

                row[
                    "subsidiary"
                ] = df.iloc[
                    original_idx
                ][subsidiary_col]

            if record_date_col:

                row[
                    "record_date"
                ] = df.iloc[
                    original_idx
                ][record_date_col]

            elif date_col:

                row[
                    "record_date"
                ] = df.iloc[
                    original_idx
                ][date_col]

            explanation_rows.append(
                row
            )

    record_df = pd.DataFrame(
        explanation_rows
    )

    record_df.to_csv(
        RECORD_OUTPUT,
        index=False
    )

    print(
        f"Saved: {RECORD_OUTPUT}"
    )

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    print(
        "\n[8] Generating final-model probabilities..."
    )

    # IMPORTANT:
    # Use the SAME full pipeline for prediction.
    raw_probability = (
        model.predict_proba(
            X_raw
        )[:, 1]
    )

    calibrated_probability = (
        calibrator.predict_proba(
            raw_probability.reshape(-1,1)
        )[:,1]
    )

    # Default threshold from final pipeline
    threshold = 0.36

    if CONFIG_PATH.exists():

        try:

            with open(
                CONFIG_PATH,
                "r",
                encoding="utf-8"
            ) as f:

                config = json.load(f)

            for key in [
                "threshold",
                "optimal_threshold",
                "calibrated_threshold",
                "decision_threshold"
            ]:

                if key in config:

                    threshold = float(
                        config[key]
                    )

                    break

        except Exception:
            pass

    prediction = (
        calibrated_probability
        >= threshold
    ).astype(int)

    print(
        f"Decision threshold: "
        f"{threshold:.3f}"
    )

    # --------------------------------------------------------
    # LATEST RECORD PER MINE
    # --------------------------------------------------------

    print(
        "\n[9] Creating latest mine explanations..."
    )

    pred_rows = []

    for i in range(
        len(df)
    ):

        row = {

            "record_index":
                i,

            "predicted_escalation_probability":
                float(
                    calibrated_probability[i]
                ),

            "predicted_escalation":
                int(
                    prediction[i]
                ),
        }

        if subsidiary_col:

            row[
                "subsidiary"
            ] = df.iloc[
                i
            ][subsidiary_col]

        if record_date_col:

            row[
                "record_date"
            ] = df.iloc[
                i
            ][record_date_col]

        elif date_col:

            row[
                "record_date"
            ] = df.iloc[
                i
            ][date_col]

        pred_rows.append(
            row
        )

    prediction_df = pd.DataFrame(
        pred_rows
    )

    prediction_df[
        "_parsed_date"
    ] = pd.to_datetime(
        prediction_df[
            "record_date"
        ],
        errors="coerce"
    )

    if subsidiary_col:

        latest_df = (
            prediction_df
            .sort_values(
                [
                    "subsidiary",
                    "_parsed_date"
                ]
            )
            .groupby(
                "subsidiary",
                as_index=False
            )
            .tail(1)
            .copy()
        )

    else:

        latest_df = (
            prediction_df
            .sort_values(
                "_parsed_date"
            )
            .tail(1)
            .copy()
        )

    latest_df = latest_df.drop(
        columns=[
            "_parsed_date"
        ]
    )

    # --------------------------------------------------------
    # ADD TOP SHAP DRIVERS
    # --------------------------------------------------------

    latest_output = []

    for _, row in latest_df.iterrows():

        record_index = int(
            row[
                "record_index"
            ]
        )

        output = row.to_dict()

        matching = (
            record_df[
                record_df[
                    "record_index"
                ]
                == record_index
            ]
            .sort_values(
                "driver_rank"
            )
        )

        for _, driver in matching.iterrows():

            rank = int(
                driver[
                    "driver_rank"
                ]
            )

            output[
                f"driver_{rank}"
            ] = driver[
                "readable_feature"
            ]

            output[
                f"driver_{rank}_shap"
            ] = driver[
                "shap_value"
            ]

            output[
                f"driver_{rank}_direction"
            ] = driver[
                "direction"
            ]

            output[
                f"driver_{rank}_reason"
            ] = driver[
                "reason"
            ]

        latest_output.append(
            output
        )

    latest_output_df = pd.DataFrame(
        latest_output
    )

    latest_output_df.to_csv(
        LATEST_OUTPUT,
        index=False
    )

    # --------------------------------------------------------
    # FEATURE VALUES
    # --------------------------------------------------------

    print(
        "\n[10] Saving SHAP feature values..."
    )

    values_df = pd.DataFrame(
        X_shap,
        columns=feature_names
    )

    values_df.insert(
        0,
        "record_index",
        shap_indices
    )

    values_df.to_csv(
        VALUES_OUTPUT,
        index=False
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary = pd.DataFrame({

        "metric": [

            "input_records",

            "shap_records",

            "transformed_features",

            "tree_estimator",

            "top_global_feature",

            "top_global_importance_pct",

            "decision_threshold",

            "predicted_escalations",

            "predicted_non_escalations",
        ],

        "value": [

            len(df),

            len(shap_indices),

            len(feature_names),

            type(
                tree_model
            ).__name__,

            (
                global_df.iloc[
                    0
                ][
                    "readable_feature"
                ]
                if len(
                    global_df
                )
                else ""
            ),

            (
                global_df.iloc[
                    0
                ][
                    "importance_pct"
                ]
                if len(
                    global_df
                )
                else 0
            ),

            threshold,

            int(
                prediction.sum()
            ),

            int(
                (prediction == 0)
                .sum()
            ),
        ]
    })

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("STEP 4.15 COMPLETED SUCCESSFULLY")
    print("=" * 80)

    print(
        "\nGenerated:"
    )

    print(
        f"1. {GLOBAL_OUTPUT}"
    )

    print(
        f"2. {RECORD_OUTPUT}"
    )

    print(
        f"3. {LATEST_OUTPUT}"
    )

    print(
        f"4. {SUMMARY_OUTPUT}"
    )

    print(
        f"5. {VALUES_OUTPUT}"
    )

    print(
        "\nFrozen predictive model was NOT changed."
    )

    print(
        "No retraining was performed."
    )

    print(
        "\nArchitecture:"
    )

    print(
        "Predict → Calibrate → Explain → "
        "Regulatory RAG → Evidence → Recommendation"
    )


if __name__ == "__main__":
    main()