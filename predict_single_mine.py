"""
CoalMineAI - COMPLETE SINGLE-MINE INTELLIGENCE

Purpose
-------
For one mine and one HISTORICAL date, produce a consolidated
intelligence report containing:

1. Operational risk
2. Early warning
3. Frozen ML prediction
4. Probability calibration
5. SHAP drivers
6. Relevant regulatory requirements
7. Existing compliance-risk intelligence
8. Evidence availability status
9. Governance priority
10. Governance reason
11. Recommended management action
12. Verification action

IMPORTANT
---------
The project currently has NO real mine-side evidence documents.

Therefore this script:
    - does NOT claim compliance
    - does NOT claim non-compliance
    - does NOT convert the 2,468 regulations into missing evidence
    - does NOT use unavailable evidence as a governance penalty
    - reports evidence as NOT_AVAILABLE / NOT_ASSESSED

The predictive model is frozen.
No retraining is performed.
No threshold optimization is performed.
No conversational AI is implemented.

Example
-------
python src/predict_single_mine.py --mine NCL --year 2025 --month 4
"""


# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path
import argparse
import json
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(r"D:\CoalMineAI")

OUTPUTS_DIR = BASE_DIR / "outputs"
V5_DIR = OUTPUTS_DIR / "v5"
FINAL_DIR = V5_DIR / "FINAL"


# ============================================================
# INPUT DATASETS
# ============================================================

SELECTED_DATASET = (
    V5_DIR /
    "v5_04_10_selected_features_dataset.csv"
)

RISK_FILE = (
    OUTPUTS_DIR /
    "production_risk_analysis.csv"
)

EARLY_WARNING_FILE = (
    OUTPUTS_DIR /
    "early_warning_analysis.csv"
)

ML_PREDICTIONS_FILE = (
    FINAL_DIR /
    "final_risk_predictions.csv"
)

SHAP_FILE = (
    FINAL_DIR /
    "shap_latest_mine_explanations.csv"
)

REGULATORY_FILE = (
    FINAL_DIR /
    "regulatory_retrieval_qc.csv"
)

COMPLIANCE_RISK_FILE = (
    OUTPUTS_DIR /
    "mine_specific_compliance_risk.csv"
)

COMPLIANCE_SUMMARY_FILE = (
    OUTPUTS_DIR /
    "mine_specific_compliance_summary.csv"
)

COMPLIANCE_INTELLIGENCE_FILE = (
    OUTPUTS_DIR /
    "mine_compliance_intelligence.csv"
)

COMPLIANCE_ACTION_FILE = (
    OUTPUTS_DIR /
    "mine_compliance_actions.csv"
)


# ============================================================
# FROZEN MODEL ARTIFACTS
# ============================================================

MODEL_PATH = (
    FINAL_DIR /
    "final_risk_model.joblib"
)

PREPROCESSOR_PATH = (
    FINAL_DIR /
    "final_preprocessor.joblib"
)

CALIBRATOR_PATH = (
    FINAL_DIR /
    "final_calibrator.joblib"
)

CONFIG_PATH = (
    FINAL_DIR /
    "final_model_config.json"
)


# ============================================================
# OUTPUT
# ============================================================

SINGLE_MINE_OUTPUT = (
    FINAL_DIR /
    "single_mine_intelligence.csv"
)


# ============================================================
# FROZEN THRESHOLD FALLBACK
# ============================================================

FALLBACK_THRESHOLD = 0.36


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def clean(value):
    """
    Convert value to safe readable text.
    """
    if pd.isna(value):
        return ""

    return str(value).strip()


def number(value, default=0.0):
    """
    Safely convert a value to float.
    """
    try:
        if pd.isna(value):
            return default

        return float(value)

    except Exception:
        return default


def find_column(df, candidates):
    """
    Find a column using case-insensitive matching.
    """

    lookup = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in candidates:

        key = (
            str(candidate)
            .strip()
            .lower()
        )

        if key in lookup:

            return lookup[key]

    return None


def load_csv(
    path,
    label,
    required=True
):
    """
    Load CSV with useful error handling.
    """

    if not path.exists():

        if required:

            raise FileNotFoundError(
                f"{label} not found:\n{path}"
            )

        print(
            f"WARNING: {label} not found:\n{path}"
        )

        return pd.DataFrame()

    try:

        df = pd.read_csv(
            path
        )

        print(
            f"{label}: {df.shape}"
        )

        return df

    except Exception as exc:

        if required:

            raise RuntimeError(
                f"Could not load {label}:\n"
                f"{path}\n"
                f"Error: {exc}"
            )

        print(
            f"WARNING: Could not load {label}: {exc}"
        )

        return pd.DataFrame()


def load_joblib(
    path,
    label
):
    """
    Load a joblib artifact.
    """

    if not path.exists():

        raise FileNotFoundError(
            f"{label} not found:\n{path}"
        )

    try:

        return joblib.load(
            path
        )

    except Exception as exc:

        raise RuntimeError(
            f"Could not load {label}:\n"
            f"{path}\n"
            f"Error: {exc}"
        )


def standardize_keys(
    df,
    label
):
    """
    Standardize mine/date columns.

    Mine:
        subsidiary / mine / mine_name
        -> subsidiary

    Date:
        record_date / date / reporting_date
        -> record_date
    """

    if df.empty:

        return df

    df = df.copy()

    # --------------------------------------------------------
    # Mine
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

    if mine_col is not None:

        if mine_col != "subsidiary":

            df = df.rename(
                columns={
                    mine_col:
                    "subsidiary"
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
            "reporting_date",
            "observation_date"
        ]
    )

    if date_col is not None:

        if date_col != "record_date":

            df = df.rename(
                columns={
                    date_col:
                    "record_date"
                }
            )

        df["record_date"] = pd.to_datetime(
            df["record_date"],
            errors="coerce"
        )

    return df


def exact_record(
    df,
    mine,
    date
):
    """
    Find one mine/date record.
    """

    if df.empty:

        return None

    if (
        "subsidiary" not in df.columns
        or
        "record_date" not in df.columns
    ):

        return None

    matches = df[
        (
            df["subsidiary"]
            .astype(str)
            .str.strip()
            .str.upper()
            ==
            mine.upper()
        )
        &
        (
            df["record_date"] == date
        )
    ].copy()

    if matches.empty:

        return None

    return matches.iloc[0]


def latest_record(
    df,
    mine
):
    """
    Find latest available record for a mine.
    """

    if df.empty:

        return None

    if "subsidiary" not in df.columns:

        return None

    matches = df[
        df["subsidiary"]
        .astype(str)
        .str.strip()
        .str.upper()
        ==
        mine.upper()
    ].copy()

    if matches.empty:

        return None

    if "record_date" in matches.columns:

        matches = matches.sort_values(
            "record_date"
        )

    return matches.iloc[-1]


# ============================================================
# THRESHOLD
# ============================================================

def load_threshold():
    """
    Load the threshold from the frozen model configuration.

    Fallback:
        0.36
    """

    if CONFIG_PATH.exists():

        try:

            with open(
                CONFIG_PATH,
                "r",
                encoding="utf-8"
            ) as file:

                config = json.load(
                    file
                )

            possible_keys = [
                "selected_threshold",
                "decision_threshold",
                "threshold",
                "optimal_threshold",
                "operating_threshold"
            ]

            for key in possible_keys:

                if key in config:

                    value = float(
                        config[key]
                    )

                    if 0 < value < 1:

                        print(
                            f"Threshold loaded from config: "
                            f"{value:.3f}"
                        )

                        return value

        except Exception:

            pass

    print(
        f"Using frozen fallback threshold: "
        f"{FALLBACK_THRESHOLD:.3f}"
    )

    return FALLBACK_THRESHOLD


# ============================================================
# CALIBRATION
# ============================================================

def calibrate_probability(
    calibrator,
    raw_probability
):
    """
    Apply the frozen LogisticRegression calibrator.

    First tries direct raw probability input.
    Then tries logit(raw probability).
    """

    p = float(
        np.clip(
            raw_probability,
            1e-6,
            1 - 1e-6
        )
    )

    # --------------------------------------------------------
    # Direct probability
    # --------------------------------------------------------

    if hasattr(
        calibrator,
        "predict_proba"
    ):

        try:

            result = calibrator.predict_proba(
                np.array(
                    [[p]],
                    dtype=float
                )
            )

            result = np.asarray(
                result,
                dtype=float
            )

            if (
                result.ndim == 2
                and
                result.shape[1] >= 2
            ):

                value = float(
                    result[0, 1]
                )

                if 0 <= value <= 1:

                    return value

        except Exception:

            pass

        # ----------------------------------------------------
        # Logit input
        # ----------------------------------------------------

        logit = np.log(
            p / (1 - p)
        )

        try:

            result = calibrator.predict_proba(
                np.array(
                    [[logit]],
                    dtype=float
                )
            )

            result = np.asarray(
                result,
                dtype=float
            )

            if (
                result.ndim == 2
                and
                result.shape[1] >= 2
            ):

                value = float(
                    result[0, 1]
                )

                if 0 <= value <= 1:

                    return value

        except Exception:

            pass

    # --------------------------------------------------------
    # Logistic coefficient fallback
    # --------------------------------------------------------

    if (
        hasattr(calibrator, "coef_")
        and
        hasattr(calibrator, "intercept_")
    ):

        coefficient = float(
            np.asarray(
                calibrator.coef_
            )
            .ravel()[0]
        )

        intercept = float(
            np.asarray(
                calibrator.intercept_
            )
            .ravel()[0]
        )

        score = (
            intercept
            + coefficient * p
        )

        calibrated = (
            1.0
            /
            (
                1.0
                +
                np.exp(
                    -np.clip(
                        score,
                        -50,
                        50
                    )
                )
            )
        )

        return float(
            calibrated
        )

    print(
        "WARNING: Could not apply calibrator. "
        "Using raw probability."
    )

    return p


# ============================================================
# MODEL FEATURE EXTRACTION
# ============================================================

def recover_model_features(
    model,
    dataset
):
    """
    Recover the original columns expected by the frozen
    ColumnTransformer inside the frozen Pipeline.
    """

    features = []

    # --------------------------------------------------------
    # Pipeline -> ColumnTransformer
    # --------------------------------------------------------

    if hasattr(
        model,
        "named_steps"
    ):

        for (
            step_name,
            step
        ) in model.named_steps.items():

            if not hasattr(
                step,
                "transformers_"
            ):

                continue

            try:

                for (
                    transformer_name,
                    transformer,
                    columns
                ) in step.transformers_:

                    if transformer == "drop":

                        continue

                    if isinstance(
                        columns,
                        (list, tuple)
                    ):

                        features.extend(
                            list(columns)
                        )

                    elif hasattr(
                        columns,
                        "tolist"
                    ):

                        features.extend(
                            columns.tolist()
                        )

            except Exception:

                continue

    # --------------------------------------------------------
    # Deduplicate while preserving order
    # --------------------------------------------------------

    features = list(
        dict.fromkeys(
            features
        )
    )

    # --------------------------------------------------------
    # Keep features present in dataset
    # --------------------------------------------------------

    features = [
        column
        for column in features
        if column in dataset.columns
    ]

    return features


# ============================================================
# SHAP DRIVER EXTRACTION
# ============================================================

def extract_shap_drivers(
    shap_df,
    mine
):
    """
    Extract top 5 SHAP drivers for one mine.

    This uses the existing SHAP output.
    It does NOT recalculate SHAP.
    """

    drivers = []

    if shap_df.empty:

        return drivers

    if "subsidiary" not in shap_df.columns:

        return drivers

    subset = shap_df[
        shap_df["subsidiary"]
        .astype(str)
        .str.strip()
        .str.upper()
        ==
        mine.upper()
    ].copy()

    if subset.empty:

        return drivers

    row = subset.iloc[-1]

    for rank in range(
        1,
        6
    ):

        feature_col = (
            f"driver_{rank}"
        )

        value_col = (
            f"driver_{rank}_shap"
        )

        direction_col = (
            f"driver_{rank}_direction"
        )

        if feature_col not in row.index:

            continue

        feature = clean(
            row[
                feature_col
            ]
        )

        if not feature:

            continue

        shap_value = number(
            row.get(
                value_col,
                0
            )
        )

        direction = clean(
            row.get(
                direction_col,
                ""
            )
        )

        drivers.append(
            {
                "rank": rank,
                "feature": feature,
                "shap_value": shap_value,
                "direction": direction
            }
        )

    return drivers


# ============================================================
# REGULATORY EXTRACTION
# ============================================================

def get_regulatory_results(
    regulatory_df,
    mine
):

    if regulatory_df.empty:

        return pd.DataFrame()

    if "mine" not in regulatory_df.columns:

        return pd.DataFrame()

    result = regulatory_df[
        regulatory_df["mine"]
        .astype(str)
        .str.strip()
        .str.upper()
        ==
        mine.upper()
    ].copy()

    if (
        "rank" in result.columns
    ):

        result = result.sort_values(
            "rank"
        )

    elif (
        "retrieval_score" in result.columns
    ):

        result = result.sort_values(
            "retrieval_score",
            ascending=False
        )

    return result


# ============================================================
# COMPLIANCE RISK EXTRACTION
# ============================================================

def get_compliance_risk(
    compliance_df,
    mine
):
    """
    Extract mine-specific compliance intelligence.

    NOTE:
    This is the existing computed compliance-risk layer.
    It is NOT equivalent to evidence verification.
    """

    result = {
        "score": 0.0,
        "level": "UNKNOWN",
        "found": False
    }

    if compliance_df.empty:

        return result

    mine_col = find_column(
        compliance_df,
        [
            "subsidiary",
            "mine",
            "mine_name"
        ]
    )

    if mine_col is None:

        return result

    rows = compliance_df[
        compliance_df[mine_col]
        .astype(str)
        .str.strip()
        .str.upper()
        ==
        mine.upper()
    ].copy()

    if rows.empty:

        return result

    result["found"] = True

    # --------------------------------------------------------
    # Score
    # --------------------------------------------------------

    score_col = find_column(
        rows,
        [
            "compliance_risk_score"
        ]
    )

    if score_col:

        values = pd.to_numeric(
            rows[
                score_col
            ],
            errors="coerce"
        ).dropna()

        if not values.empty:

            result["score"] = float(
                values.max()
            )

    # --------------------------------------------------------
    # Level
    # --------------------------------------------------------

    level_col = find_column(
        rows,
        [
            "compliance_risk_level"
        ]
    )

    if level_col:

        levels = (
            rows[
                level_col
            ]
            .astype(str)
            .str.upper()
        )

        if "CRITICAL" in levels.values:

            result["level"] = "CRITICAL"

        elif "HIGH" in levels.values:

            result["level"] = "HIGH"

        elif "MEDIUM" in levels.values:

            result["level"] = "MEDIUM"

        elif "LOW" in levels.values:

            result["level"] = "LOW"

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # COMMAND ARGUMENTS
    # ========================================================

    parser = argparse.ArgumentParser(
        description=(
            "Complete historical single-mine "
            "CoalMineAI intelligence."
        )
    )

    parser.add_argument(
        "--mine",
        required=True,
        help="Mine/subsidiary, e.g. NCL"
    )

    parser.add_argument(
        "--year",
        required=True,
        type=int,
        help="Year, e.g. 2025"
    )

    parser.add_argument(
        "--month",
        required=True,
        type=int,
        choices=range(1, 13),
        help="Month number 1-12"
    )

    args = parser.parse_args()

    mine = args.mine.strip()

    requested_date = pd.Timestamp(
        year=args.year,
        month=args.month,
        day=1
    )

    # ========================================================
    # HEADER
    # ========================================================

    print("=" * 80)

    print(
        "CoalMineAI - COMPLETE SINGLE-MINE INTELLIGENCE"
    )

    print("=" * 80)

    print(
        f"\nMine : {mine}"
    )

    print(
        f"Date : "
        f"{requested_date.strftime('%Y-%m-%d')}"
    )

    print(
        "\nMode : FROZEN HISTORICAL EVALUATION"
    )

    # ========================================================
    # 1. LOAD SELECTED FEATURE DATA
    # ========================================================

    print(
        "\n[1] Loading selected-feature dataset..."
    )

    selected_df = load_csv(
        SELECTED_DATASET,
        "Selected-feature dataset"
    )

    selected_df = standardize_keys(
        selected_df,
        "selected-feature dataset"
    )

    # ========================================================
    # 2. FIND EXACT RECORD
    # ========================================================

    print(
        "\n[2] Finding requested mine/date..."
    )

    selected_record = exact_record(
        selected_df,
        mine,
        requested_date
    )

    if selected_record is None:

        print(
            f"\nNo record found for "
            f"{mine} on "
            f"{requested_date.strftime('%Y-%m-%d')}."
        )

        mine_rows = selected_df[
            selected_df["subsidiary"]
            .astype(str)
            .str.strip()
            .str.upper()
            ==
            mine.upper()
        ].copy()

        if not mine_rows.empty:

            available_dates = sorted(
                mine_rows[
                    "record_date"
                ]
                .dropna()
                .dt.strftime(
                    "%Y-%m-%d"
                )
                .unique()
            )

            print(
                "\nAvailable dates:"
            )

            for available in available_dates:

                print(
                    f"  {available}"
                )

        raise SystemExit(1)

    print(
        "Requested record: FOUND"
    )

    # ========================================================
    # 3. LOAD SUPPORTING DATA
    # ========================================================

    print(
        "\n[3] Loading supporting intelligence..."
    )

    risk_df = load_csv(
        RISK_FILE,
        "Operational risk"
    )

    warning_df = load_csv(
        EARLY_WARNING_FILE,
        "Early warning"
    )

    ml_df = load_csv(
        ML_PREDICTIONS_FILE,
        "Final ML predictions",
        required=False
    )

    shap_df = load_csv(
        SHAP_FILE,
        "SHAP explanations",
        required=False
    )

    regulatory_df = load_csv(
        REGULATORY_FILE,
        "Regulatory retrieval",
        required=False
    )

    compliance_risk_df = load_csv(
        COMPLIANCE_RISK_FILE,
        "Mine-specific compliance risk",
        required=False
    )

    compliance_summary_df = load_csv(
        COMPLIANCE_SUMMARY_FILE,
        "Mine-specific compliance summary",
        required=False
    )

    compliance_intelligence_df = load_csv(
        COMPLIANCE_INTELLIGENCE_FILE,
        "Mine compliance intelligence",
        required=False
    )

    compliance_action_df = load_csv(
        COMPLIANCE_ACTION_FILE,
        "Mine compliance actions",
        required=False
    )

    # ========================================================
    # 4. STANDARDIZE KEYS
    # ========================================================

    risk_df = standardize_keys(
        risk_df,
        "operational risk"
    )

    warning_df = standardize_keys(
        warning_df,
        "early warning"
    )

    ml_df = standardize_keys(
        ml_df,
        "ML predictions"
    )

    # ========================================================
    # 5. SUPPORTING RECORDS
    # ========================================================

    risk_record = exact_record(
        risk_df,
        mine,
        requested_date
    )

    if risk_record is None:

        risk_record = latest_record(
            risk_df,
            mine
        )

        print(
            "WARNING: Exact operational-risk record "
            "not found. Using latest available record."
        )


    warning_record = exact_record(
        warning_df,
        mine,
        requested_date
    )

    if warning_record is None:

        warning_record = latest_record(
            warning_df,
            mine
        )

        print(
            "WARNING: Exact early-warning record "
            "not found. Using latest available record."
        )


    # ========================================================
    # 6. LOAD FROZEN MODEL
    # ========================================================

    print(
        "\n[4] Loading frozen predictive model..."
    )

    model = load_joblib(
        MODEL_PATH,
        "Final risk model"
    )

    calibrator = load_joblib(
        CALIBRATOR_PATH,
        "Final calibrator"
    )

    preprocessor = load_joblib(
        PREPROCESSOR_PATH,
        "Final preprocessor"
    )

    print(
        "Model:",
        type(model).__name__
    )

    print(
        "Calibrator:",
        type(calibrator).__name__
    )

    print(
        "Preprocessor:",
        type(preprocessor).__name__
    )

    # ========================================================
    # 7. SHOW PIPELINE
    # ========================================================

    print(
        "\n[5] Inspecting frozen model Pipeline..."
    )

    if hasattr(
        model,
        "named_steps"
    ):

        print(
            "Pipeline steps:"
        )

        for name, step in (
            model.named_steps.items()
        ):

            print(
                f"  - {name}: "
                f"{type(step).__name__}"
            )

    # ========================================================
    # 8. RECOVER ORIGINAL MODEL FEATURES
    # ========================================================

    print(
        "\n[6] Preparing original model features..."
    )

    model_features = (
        recover_model_features(
            model,
            selected_df
        )
    )

    # Fallback
    if not model_features:

        metadata_columns = {
            "subsidiary",
            "date",
            "record_date",
            "material_escalation",
            "_prediction_date"
        }

        model_features = [
            column
            for column in selected_df.columns
            if column not in metadata_columns
            and not column.startswith("_")
        ]

    print(
        "Model input features:",
        len(model_features)
    )

    if len(model_features) != 68:

        print(
            "WARNING: Expected 68 model input features, "
            f"found {len(model_features)}."
        )

    # Check missing columns
    missing_features = [
        feature
        for feature in model_features
        if feature not in selected_record.index
    ]

    if missing_features:

        raise RuntimeError(
            "Required model features are missing:\n"
            +
            "\n".join(
                missing_features
            )
        )

    X_one = pd.DataFrame(
        [
            {
                feature:
                selected_record[
                    feature
                ]
                for feature in model_features
            }
        ]
    )

    print(
        "Original input shape:",
        X_one.shape
    )

    # ========================================================
    # 9. MODEL PREDICTION
    # ========================================================

    print(
        "\n[7] Running frozen predictive model..."
    )

    if not hasattr(
        model,
        "predict_proba"
    ):

        raise RuntimeError(
            "Frozen model does not provide predict_proba()."
        )

    try:

        raw_probability = float(
            model.predict_proba(
                X_one
            )[0, 1]
        )

    except Exception as exc:

        raise RuntimeError(
            "Frozen model prediction failed:\n"
            f"{exc}"
        )

    print(
        f"Raw probability: "
        f"{raw_probability:.6f}"
    )

    # ========================================================
    # 10. CALIBRATION
    # ========================================================

    print(
        "\n[8] Applying frozen probability calibration..."
    )

    calibrated_probability = (
        calibrate_probability(
            calibrator,
            raw_probability
        )
    )

    print(
        f"Calibrated probability: "
        f"{calibrated_probability:.6f}"
    )

    # ========================================================
    # 11. THRESHOLD
    # ========================================================

    threshold = load_threshold()

    print(
        f"Decision threshold: "
        f"{threshold:.6f}"
    )

    predicted_class = int(
        calibrated_probability
        >= threshold
    )

    if predicted_class == 1:

        prediction_label = (
            "MATERIAL ESCALATION PREDICTED"
        )

    else:

        prediction_label = (
            "NO MATERIAL ESCALATION PREDICTED"
        )

    # ========================================================
    # 12. HISTORICAL ACTUAL
    # ========================================================

    target_col = find_column(
        selected_df,
        [
            "material_escalation"
        ]
    )

    actual_target = None

    if target_col is not None:

        value = selected_record[
            target_col
        ]

        if not pd.isna(value):

            actual_target = int(
                float(value)
            )

    actual_label = "UNAVAILABLE"

    prediction_correct = None

    if actual_target is not None:

        if actual_target == 1:

            actual_label = (
                "MATERIAL ESCALATION"
            )

        else:

            actual_label = (
                "NO MATERIAL ESCALATION"
            )

        prediction_correct = (
            predicted_class
            ==
            actual_target
        )

    # ========================================================
    # 13. OPERATIONAL STATE
    # ========================================================

    print(
        "\n[9] Reading operational state..."
    )

    if risk_record is not None:

        overall_operational_risk = number(
            risk_record.get(
                "overall_operational_risk",
                selected_record.get(
                    "overall_operational_risk",
                    0
                )
            )
        )

        equipment_risk = number(
            risk_record.get(
                "equipment_risk",
                selected_record.get(
                    "equipment_risk",
                    0
                )
            )
        )

        logistics_risk = number(
            risk_record.get(
                "logistics_risk",
                selected_record.get(
                    "logistics_risk",
                    0
                )
            )
        )

        weather_risk = number(
            risk_record.get(
                "weather_risk",
                selected_record.get(
                    "weather_risk",
                    0
                )
            )
        )

        workforce_risk = number(
            risk_record.get(
                "workforce_risk",
                selected_record.get(
                    "workforce_risk",
                    0
                )
            )
        )

    else:

        overall_operational_risk = number(
            selected_record.get(
                "overall_operational_risk",
                0
            )
        )

        equipment_risk = number(
            selected_record.get(
                "equipment_risk",
                0
            )
        )

        logistics_risk = number(
            selected_record.get(
                "logistics_risk",
                0
            )
        )

        weather_risk = number(
            selected_record.get(
                "weather_risk",
                0
            )
        )

        workforce_risk = number(
            selected_record.get(
                "workforce_risk",
                0
            )
        )

    # ========================================================
    # 14. EARLY WARNING
    # ========================================================

    if warning_record is not None:

        warning_level = clean(
            warning_record.get(
                "warning_level",
                "UNKNOWN"
            )
        )

        trajectory = clean(
            warning_record.get(
                "trajectory",
                "UNKNOWN"
            )
        )

        primary_driver = clean(
            warning_record.get(
                "primary_driver",
                ""
            )
        )

        early_warning_score = number(
            warning_record.get(
                "early_warning_score",
                0
            )
        )

    else:

        warning_level = "UNKNOWN"

        trajectory = "UNKNOWN"

        primary_driver = ""

        early_warning_score = 0.0

    # ========================================================
    # 15. SHAP
    # ========================================================

    print(
        "\n[10] Reading SHAP drivers..."
    )

    shap_drivers = extract_shap_drivers(
        shap_df,
        mine
    )

    # ========================================================
    # 16. REGULATIONS
    # ========================================================

    print(
        "\n[11] Reading relevant regulations..."
    )

    mine_regulations = (
        get_regulatory_results(
            regulatory_df,
            mine
        )
    )

    retrieved_regulations = len(
        mine_regulations
    )

    high_priority_count = 0
    medium_priority_count = 0

    top_regulatory_domain = "UNKNOWN"

    if not mine_regulations.empty:

        if "regulatory_priority" in (
            mine_regulations.columns
        ):

            priority_values = (
                mine_regulations[
                    "regulatory_priority"
                ]
                .astype(str)
                .str.upper()
            )

            high_priority_count = int(
                (
                    priority_values
                    == "HIGH"
                ).sum()
            )

            medium_priority_count = int(
                (
                    priority_values
                    == "MEDIUM"
                ).sum()
            )

        if "regulatory_domain" in (
            mine_regulations.columns
        ):

            modes = (
                mine_regulations[
                    "regulatory_domain"
                ]
                .dropna()
                .astype(str)
                .mode()
            )

            if not modes.empty:

                top_regulatory_domain = (
                    modes.iloc[0]
                )

    # ========================================================
    # 17. COMPLIANCE RISK
    # ========================================================

    print(
        "\n[12] Reading existing compliance intelligence..."
    )

    compliance_result = (
        get_compliance_risk(
            compliance_risk_df,
            mine
        )
    )

    compliance_score = (
        compliance_result["score"]
    )

    compliance_level = (
        compliance_result["level"]
    )

    # ========================================================
    # 18. EVIDENCE STATUS
    # ========================================================

    print(
        "\n[13] Evidence status..."
    )

    # --------------------------------------------------------
    # IMPORTANT
    # --------------------------------------------------------
    # No actual mine-side evidence data currently exists.
    #
    # Therefore we do NOT use:
    #     compliance_evidence_gap_analysis.csv
    #
    # to manufacture a missing-evidence count.
    #
    # 2,468 regulatory requirements != 2,468 missing documents.
    # --------------------------------------------------------

    evidence_status = "NOT_AVAILABLE"

    verification_priority = "NOT_ASSESSED"

    unknown_evidence_count = 0

    evidence_statement = (
        "Mine-side compliance evidence is not available "
        "in the current dataset. No compliance or "
        "non-compliance determination is made."
    )

    verification_action_text = (
        "Obtain applicable mine-side compliance evidence "
        "before making a compliance determination."
    )

    # ========================================================
    # 19. GOVERNANCE SCORE
    # ========================================================

    print(
        "\n[14] Calculating governance priority..."
    )

    operational_component = max(
        0.0,
        min(
            100.0,
            overall_operational_risk
        )
    )

    escalation_component = (
        calibrated_probability
        * 100.0
    )

    warning_component = {
        "CRITICAL": 100.0,
        "EARLY_WARNING": 75.0,
        "WATCH": 50.0,
        "STABLE": 15.0
    }.get(
        warning_level.upper(),
        25.0
    )

    trajectory_component = {
        "RAPIDLY_WORSENING": 100.0,
        "WORSENING": 75.0,
        "STABLE": 20.0,
        "IMPROVING": 10.0
    }.get(
        trajectory.upper(),
        25.0
    )

    compliance_component = max(
        0.0,
        min(
            100.0,
            compliance_score
        )
    )

    # --------------------------------------------------------
    # Regulatory relevance signal
    # --------------------------------------------------------
    #
    # We use retrieved regulatory information only as a
    # modest governance signal.
    #
    # Because all 15 current retrieved items are MEDIUM
    # priority, they should NOT overwhelm the operational
    # and predictive signals.
    # --------------------------------------------------------

    regulatory_component = min(
        100.0,
        high_priority_count * 20.0
        +
        medium_priority_count * 5.0
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Evidence is unavailable, therefore it contributes ZERO
    # to the governance score.
    #
    # This avoids penalizing a mine simply because evidence
    # has not been supplied to the prototype.
    # --------------------------------------------------------

    governance_score = (

        0.28
        * operational_component

        +

        0.34
        * escalation_component

        +

        0.17
        * warning_component

        +

        0.11
        * trajectory_component

        +

        0.07
        * compliance_component

        +

        0.03
        * regulatory_component
    )

    governance_score = round(
        max(
            0.0,
            min(
                100.0,
                governance_score
            )
        ),
        2
    )

    # ========================================================
    # 20. GOVERNANCE LEVEL
    # ========================================================

    if governance_score >= 70:

        governance_level = "HIGH"

    elif governance_score >= 45:

        governance_level = "MEDIUM"

    else:

        governance_level = "LOW"

    # ========================================================
    # 21. GOVERNANCE STATUS
    # ========================================================

    if governance_level == "HIGH":

        governance_status = (
            "MANAGEMENT_REVIEW_REQUIRED"
        )

    elif governance_level == "MEDIUM":

        governance_status = (
            "PLANNED_REVIEW_REQUIRED"
        )

    else:

        governance_status = (
            "ROUTINE_MONITORING"
        )

    # ========================================================
    # 22. GOVERNANCE REASON
    # ========================================================

    reasons = []

    if operational_component >= 60:

        reasons.append(
            "high operational risk"
        )

    elif operational_component >= 40:

        reasons.append(
            "elevated operational risk"
        )

    if escalation_component >= 60:

        reasons.append(
            "high predicted escalation probability"
        )

    elif escalation_component >= 40:

        reasons.append(
            "moderate predicted escalation probability"
        )

    if warning_level.upper() in {
        "CRITICAL",
        "EARLY_WARNING"
    }:

        reasons.append(
            f"{warning_level.lower().replace('_', ' ')} status"
        )

    if trajectory.upper() in {
        "WORSENING",
        "RAPIDLY_WORSENING"
    }:

        reasons.append(
            f"{trajectory.lower().replace('_', ' ')} risk trajectory"
        )

    if compliance_level.upper() in {
        "HIGH",
        "CRITICAL"
    }:

        reasons.append(
            "elevated compliance risk"
        )

    if primary_driver:

        reasons.append(
            f"{primary_driver.lower()} is a primary operational driver"
        )

    if not reasons:

        reasons.append(
            "no dominant high-risk indicator identified"
        )

    governance_reason = "; ".join(
        reasons
    )

    # ========================================================
    # 23. MANAGEMENT ACTION
    # ========================================================

    if governance_level == "HIGH":

        management_action = (
            "Immediate management review; "
            "investigate the dominant operational risk driver; "
            "review applicable statutory requirements; "
            "and prioritize reassessment."
        )

    elif governance_level == "MEDIUM":

        management_action = (
            "Schedule a planned management review; "
            "monitor the identified risk trend; "
            "review applicable statutory requirements; "
            "and complete appropriate compliance verification."
        )

    else:

        management_action = (
            "Continue routine monitoring and periodic "
            "statutory compliance verification."
        )

    if primary_driver:

        management_action += (
            f" Primary operational focus: "
            f"{primary_driver}."
        )

    if trajectory.upper() == (
        "RAPIDLY_WORSENING"
    ):

        management_action += (
            " Risk trajectory is rapidly worsening; "
            "prioritize reassessment."
        )

    if warning_level.upper() == (
        "EARLY_WARNING"
    ):

        management_action += (
            " Early-warning status requires closer monitoring."
        )

    # ========================================================
    # 24. DISPLAY RESULT
    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "COMPLETE SINGLE-MINE INTELLIGENCE"
    )

    print(
        "=" * 80
    )

    print(
        f"\nMine                : {mine}"
    )

    print(
        f"Date                : "
        f"{requested_date.strftime('%Y-%m-%d')}"
    )

    # --------------------------------------------------------
    # OPERATIONAL RISK
    # --------------------------------------------------------

    print(
        "\n--- OPERATIONAL RISK ---"
    )

    print(
        f"Overall risk        : "
        f"{overall_operational_risk:.2f}"
    )

    print(
        f"Equipment risk      : "
        f"{equipment_risk:.2f}"
    )

    print(
        f"Logistics risk      : "
        f"{logistics_risk:.2f}"
    )

    print(
        f"Weather risk        : "
        f"{weather_risk:.2f}"
    )

    print(
        f"Workforce risk      : "
        f"{workforce_risk:.2f}"
    )

    # --------------------------------------------------------
    # EARLY WARNING
    # --------------------------------------------------------

    print(
        "\n--- EARLY WARNING ---"
    )

    print(
        f"Warning level       : "
        f"{warning_level or 'UNKNOWN'}"
    )

    print(
        f"Trajectory          : "
        f"{trajectory or 'UNKNOWN'}"
    )

    print(
        f"Early warning score : "
        f"{early_warning_score:.2f}"
    )

    print(
        f"Primary driver      : "
        f"{primary_driver or 'UNKNOWN'}"
    )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    print(
        "\n--- PREDICTIVE MODEL ---"
    )

    print(
        f"Raw probability     : "
        f"{raw_probability * 100:.2f}%"
    )

    print(
        f"Calibrated probability: "
        f"{calibrated_probability * 100:.2f}%"
    )

    print(
        f"Threshold           : "
        f"{threshold * 100:.2f}%"
    )

    print(
        f"Prediction          : "
        f"{prediction_label}"
    )

    print(
        f"Actual historical   : "
        f"{actual_label}"
    )

    if prediction_correct is True:

        print(
            "Historical check    : CORRECT"
        )

    elif prediction_correct is False:

        print(
            "Historical check    : INCORRECT"
        )

    else:

        print(
            "Historical check    : UNAVAILABLE"
        )

    # --------------------------------------------------------
    # SHAP
    # --------------------------------------------------------

    print(
        "\n--- SHAP DRIVERS ---"
    )

    if shap_drivers:

        for driver in shap_drivers:

            print(
                f"{driver['rank']}. "
                f"{driver['feature']} "
                f"| SHAP={driver['shap_value']:.5f} "
                f"| {driver['direction']}"
            )

        print(
            "\nNote: SHAP values indicate model contribution, "
            "not causation."
        )

    else:

        print(
            "SHAP drivers unavailable."
        )

    # --------------------------------------------------------
    # REGULATORY
    # --------------------------------------------------------

    print(
        "\n--- REGULATORY INTELLIGENCE ---"
    )

    print(
        f"Relevant requirements: "
        f"{retrieved_regulations}"
    )

    print(
        f"HIGH priority        : "
        f"{high_priority_count}"
    )

    print(
        f"MEDIUM priority      : "
        f"{medium_priority_count}"
    )

    print(
        f"Top domain           : "
        f"{top_regulatory_domain}"
    )

    if not mine_regulations.empty:

        print(
            "\nTop 5 relevant requirements:"
        )

        display_columns = [
            c
            for c in [
                "rank",
                "regulatory_domain",
                "required_action",
                "regulatory_priority",
                "source_document",
                "page_number",
                "section_reference",
                "requirement"
            ]
            if c in mine_regulations.columns
        ]

        print(
            mine_regulations[
                display_columns
            ]
            .head(5)
            .to_string(
                index=False
            )
        )

    else:

        print(
            "No regulatory results available."
        )

    # --------------------------------------------------------
    # COMPLIANCE
    # --------------------------------------------------------

    print(
        "\n--- COMPLIANCE INTELLIGENCE ---"
    )

    print(
        f"Compliance risk     : "
        f"{compliance_score:.2f}"
    )

    print(
        f"Compliance level    : "
        f"{compliance_level}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Compliance-risk intelligence is not the same as "
        "verified mine-side evidence."
    )

    # --------------------------------------------------------
    # EVIDENCE
    # --------------------------------------------------------

    print(
        "\n--- EVIDENCE ---"
    )

    print(
        f"Evidence status     : "
        f"{evidence_status}"
    )

    print(
        f"Verification priority: "
        f"{verification_priority}"
    )

    print(
        f"Evidence count      : "
        f"{unknown_evidence_count}"
    )

    print(
        f"\n{evidence_statement}"
    )

    # --------------------------------------------------------
    # GOVERNANCE
    # --------------------------------------------------------

    print(
        "\n--- FINAL GOVERNANCE ---"
    )

    print(
        f"Governance score    : "
        f"{governance_score:.2f}"
    )

    print(
        f"Governance level    : "
        f"{governance_level}"
    )

    print(
        f"Governance status   : "
        f"{governance_status}"
    )

    print(
        "\nWhy:"
    )

    print(
        governance_reason
    )

    print(
        "\nRecommended management action:"
    )

    print(
        management_action
    )

    print(
        "\nVerification action:"
    )

    print(
        verification_action_text
    )

    # ========================================================
    # 25. SAVE RESULT
    # ========================================================

    output_row = {

        "subsidiary":
            mine,

        "record_date":
            requested_date.strftime(
                "%Y-%m-%d"
            ),

        # ----------------------------------------------------
        # Operational
        # ----------------------------------------------------

        "overall_operational_risk":
            overall_operational_risk,

        "equipment_risk":
            equipment_risk,

        "logistics_risk":
            logistics_risk,

        "weather_risk":
            weather_risk,

        "workforce_risk":
            workforce_risk,

        # ----------------------------------------------------
        # Early warning
        # ----------------------------------------------------

        "warning_level":
            warning_level,

        "trajectory":
            trajectory,

        "early_warning_score":
            early_warning_score,

        "primary_driver":
            primary_driver,

        # ----------------------------------------------------
        # ML
        # ----------------------------------------------------

        "raw_model_probability":
            raw_probability,

        "calibrated_escalation_probability":
            calibrated_probability,

        "decision_threshold":
            threshold,

        "predicted_material_escalation":
            predicted_class,

        "prediction_label":
            prediction_label,

        "actual_material_escalation":
            actual_target,

        "prediction_correct":
            prediction_correct,

        # ----------------------------------------------------
        # Regulatory
        # ----------------------------------------------------

        "retrieved_regulations":
            retrieved_regulations,

        "high_priority_regulations":
            high_priority_count,

        "medium_priority_regulations":
            medium_priority_count,

        "top_regulatory_domain":
            top_regulatory_domain,

        # ----------------------------------------------------
        # Compliance
        # ----------------------------------------------------

        "compliance_risk_score":
            compliance_score,

        "compliance_risk_level":
            compliance_level,

        # ----------------------------------------------------
        # Evidence
        # ----------------------------------------------------

        "evidence_status":
            evidence_status,

        "verification_priority":
            verification_priority,

        "unknown_evidence_count":
            unknown_evidence_count,

        # ----------------------------------------------------
        # Governance
        # ----------------------------------------------------

        "governance_priority_score":
            governance_score,

        "governance_priority_level":
            governance_level,

        "governance_status":
            governance_status,

        "governance_reason":
            governance_reason,

        "recommended_management_action":
            management_action,

        "verification_action":
            verification_action_text
    }

    output_df = pd.DataFrame(
        [output_row]
    )

    # --------------------------------------------------------
    # Append/update log
    # --------------------------------------------------------

    if SINGLE_MINE_OUTPUT.exists():

        try:

            previous = pd.read_csv(
                SINGLE_MINE_OUTPUT
            )

            combined = pd.concat(
                [
                    previous,
                    output_df
                ],
                ignore_index=True
            )

            combined = (
                combined
                .drop_duplicates(
                    subset=[
                        "subsidiary",
                        "record_date"
                    ],
                    keep="last"
                )
            )

            combined.to_csv(
                SINGLE_MINE_OUTPUT,
                index=False
            )

        except Exception:

            output_df.to_csv(
                SINGLE_MINE_OUTPUT,
                index=False
            )

    else:

        output_df.to_csv(
            SINGLE_MINE_OUTPUT,
            index=False
        )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "SINGLE-MINE INTELLIGENCE COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        "\nSaved:"
    )

    print(
        SINGLE_MINE_OUTPUT
    )

    print(
        "\nModel retrained     : NO"
    )

    print(
        "Threshold changed   : NO"
    )

    print(
        "Evidence invented   : NO"
    )

    print(
        "Compliance inferred : NO"
    )

    print(
        "Status               : PASS"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print(
            "\nProcess interrupted by user."
        )

        raise SystemExit(130)

    except Exception as exc:

        print(
            "\n" + "=" * 80
        )

        print(
            "SINGLE-MINE INTELLIGENCE FAILED"
        )

        print(
            "=" * 80
        )

        print(
            f"\nError:\n{exc}"
        )

        raise SystemExit(1)