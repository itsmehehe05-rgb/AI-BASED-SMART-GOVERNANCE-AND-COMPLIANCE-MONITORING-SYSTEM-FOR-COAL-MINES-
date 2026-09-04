"""
CoalMineAI - V5 Step 4.14
Random Forest + Temporal Probability Calibration
+ Robust Threshold Optimization

FINAL PREDICTIVE-RISK PIPELINE

Architecture:
    Leakage-safe dataset
        ↓
    Selected features
        ↓
    Chronological development/test split
        ↓
    Temporal OOF Random Forest predictions
        ↓
    Platt probability calibration
        ↓
    Calibrated threshold optimization
        ↓
    Final RF trained on complete development data
        ↓
    Calibrated untouched test predictions
        ↓
    Final metrics + model persistence

Important:
- Final test period remains untouched during model selection.
- Calibration is trained only on temporal OOF development predictions.
- Threshold is selected only from calibrated development OOF predictions.
- Final test is used only once for final evaluation.
"""

import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    brier_score_loss
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"D:\CoalMineAI"

INPUT_FILE = os.path.join(
    BASE_DIR,
    "outputs",
    "v5",
    "v5_04_10_selected_features_dataset.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "v5"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET = "material_escalation"
DATE_COLUMN = "record_date"

RANDOM_STATE = 42

# Final holdout = latest 15% unique dates
TEST_DATE_FRACTION = 0.15

# Number of temporal OOF folds
N_FOLDS = 4

# Random Forest configuration
RF_PARAMS = {
    "n_estimators": 700,
    "max_depth": 8,
    "min_samples_split": 6,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "class_weight": {0: 1.0, 1: 2.5},
    "bootstrap": True,
    "random_state": RANDOM_STATE,
    "n_jobs": -1
}


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def print_header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def find_date_column(df):
    """
    Find the strongest available date column.
    """
    preferred = [
        "record_date",
        "date",
        "Date",
        "month",
        "Month"
    ]

    for col in preferred:
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce")

            if parsed.notna().sum() > 0:
                return col

    raise ValueError(
        "No valid date column found. "
        "Expected record_date or date."
    )


def prepare_date(df):
    """
    Convert the date column to datetime.
    """
    df = df.copy()

    date_col = find_date_column(df)

    df[date_col] = pd.to_datetime(
        df[date_col],
        errors="coerce"
    )

    if df[date_col].isna().any():
        bad = df[date_col].isna().sum()

        raise ValueError(
            f"{bad} rows have invalid dates."
        )

    return df, date_col


def build_preprocessor(X):
    """
    Build leakage-safe preprocessing.

    Numeric:
        median imputation

    Categorical:
        most-frequent imputation
        one-hot encoding
    """

    numeric_cols = X.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    categorical_cols = [
        c for c in X.columns
        if c not in numeric_cols
    ]

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    transformers = []

    if numeric_cols:
        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_cols
            )
        )

    if categorical_cols:
        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_cols
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    return preprocessor, numeric_cols, categorical_cols


def build_rf_pipeline(X):
    """
    Build complete RF preprocessing + model pipeline.
    """

    preprocessor, numeric_cols, categorical_cols = (
        build_preprocessor(X)
    )

    model = RandomForestClassifier(
        **RF_PARAMS
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    return (
        pipeline,
        numeric_cols,
        categorical_cols
    )


def calculate_metrics(y_true, probabilities, threshold):
    """
    Calculate classification metrics.
    """

    predictions = (
        probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    balanced_accuracy = balanced_accuracy_score(
        y_true,
        predictions
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    try:
        roc_auc = roc_auc_score(
            y_true,
            probabilities
        )
    except Exception:
        roc_auc = np.nan

    try:
        pr_auc = average_precision_score(
            y_true,
            probabilities
        )
    except Exception:
        pr_auc = np.nan

    brier = brier_score_loss(
        y_true,
        probabilities
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    fpr = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    missed = (
        fn / (fn + tp)
        if (fn + tp) > 0
        else 0.0
    )

    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "fpr": fpr,
        "missed_escalation": missed,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }


def optimize_threshold(
    y_true,
    probabilities
):
    """
    Precision-aware threshold optimization.

    Primary practical constraints:
        precision >= 55%
        recall >= 30%

    Objective:
        F1 + balanced accuracy + precision + recall

    If no threshold satisfies the practical constraints,
    fallback to the best overall composite score.
    """

    rows = []

    thresholds = np.arange(
        0.20,
        0.901,
        0.01
    )

    for threshold in thresholds:

        metrics = calculate_metrics(
            y_true,
            probabilities,
            threshold
        )

        metrics["meets_constraints"] = (
            metrics["precision"] >= 0.55
            and
            metrics["recall"] >= 0.30
        )

        # Precision-aware composite objective
        metrics["composite_score"] = (
            0.35 * metrics["f1"]
            +
            0.30 * metrics["balanced_accuracy"]
            +
            0.25 * metrics["precision"]
            +
            0.10 * metrics["recall"]
        )

        rows.append(metrics)

    threshold_df = pd.DataFrame(rows)

    valid = threshold_df[
        threshold_df["meets_constraints"]
    ].copy()

    if len(valid) > 0:

        # Prefer strong F1 and balanced accuracy,
        # while maintaining precision.
        valid = valid.sort_values(
            by=[
                "f1",
                "balanced_accuracy",
                "precision",
                "recall"
            ],
            ascending=False
        )

        selected = valid.iloc[0]

        selection_reason = (
            "Selected from thresholds satisfying "
            "precision >= 55% and recall >= 30%."
        )

    else:

        threshold_df = threshold_df.sort_values(
            by=[
                "composite_score",
                "balanced_accuracy",
                "precision"
            ],
            ascending=False
        )

        selected = threshold_df.iloc[0]

        selection_reason = (
            "No threshold satisfied both practical "
            "precision and recall constraints; "
            "selected best precision-aware composite."
        )

    return (
        float(selected["threshold"]),
        threshold_df,
        selection_reason
    )


def create_temporal_folds(
    development_df,
    n_folds=4
):
    """
    Create expanding-window temporal folds.

    Each fold:
        training = all earlier records
        validation = next chronological block

    The final test period is NOT included.
    """

    dates = np.array(
        sorted(
            development_df[
                DATE_COLUMN
            ].dt.to_period("M").unique()
        )
    )

    if len(dates) < n_folds + 2:
        raise ValueError(
            "Not enough unique dates for temporal CV."
        )

    # Keep a reasonable initial training period.
    min_train_dates = max(
        24,
        int(len(dates) * 0.45)
    )

    remaining_dates = len(dates) - min_train_dates

    validation_dates = max(
        1,
        remaining_dates // n_folds
    )

    folds = []

    for i in range(n_folds):

        train_end_idx = (
            min_train_dates
            +
            i * validation_dates
        )

        val_start_idx = train_end_idx

        if i == n_folds - 1:
            val_end_idx = len(dates)
        else:
            val_end_idx = min(
                val_start_idx + validation_dates,
                len(dates)
            )

        train_dates = dates[
            :train_end_idx
        ]

        val_dates = dates[
            val_start_idx:val_end_idx
        ]

        if len(val_dates) == 0:
            continue

        train_periods = set(train_dates)
        val_periods = set(val_dates)

        train_mask = (
            development_df[DATE_COLUMN]
            .dt.to_period("M")
            .isin(train_periods)
        )

        val_mask = (
            development_df[DATE_COLUMN]
            .dt.to_period("M")
            .isin(val_periods)
        )

        train_idx = development_df.index[
            train_mask
        ].to_numpy()

        val_idx = development_df.index[
            val_mask
        ].to_numpy()

        if len(train_idx) == 0 or len(val_idx) == 0:
            continue

        folds.append(
            {
                "fold": len(folds) + 1,
                "train_idx": train_idx,
                "val_idx": val_idx,
                "train_start": str(train_dates[0]),
                "train_end": str(train_dates[-1]),
                "val_start": str(val_dates[0]),
                "val_end": str(val_dates[-1])
            }
        )

    return folds


# ============================================================
# LOAD DATA
# ============================================================

print_header(
    "STEP 4.14 - RANDOM FOREST + PROBABILITY CALIBRATION"
)

print(
    f"Loading dataset:\n{INPUT_FILE}"
)

df = pd.read_csv(
    INPUT_FILE
)

print(
    f"Loaded dataset shape: {df.shape}"
)

if TARGET not in df.columns:
    raise ValueError(
        f"Target '{TARGET}' not found."
    )

# ------------------------------------------------------------
# Date
# ------------------------------------------------------------

df, detected_date = prepare_date(df)

if detected_date != DATE_COLUMN:

    print(
        f"Detected date column: {detected_date}"
    )

    if DATE_COLUMN not in df.columns:
        df[DATE_COLUMN] = df[
            detected_date
        ]

else:

    print(
        f"Using date column: {DATE_COLUMN}"
    )

# ------------------------------------------------------------
# Target cleanup
# ------------------------------------------------------------

df[TARGET] = pd.to_numeric(
    df[TARGET],
    errors="coerce"
)

df = df[
    df[TARGET].isin([0, 1])
].copy()

df[TARGET] = df[TARGET].astype(int)

# ------------------------------------------------------------
# Sort chronologically
# ------------------------------------------------------------

df = df.sort_values(
    by=[DATE_COLUMN],
    kind="stable"
).reset_index(drop=True)

print(
    f"Date range: "
    f"{df[DATE_COLUMN].min().date()} "
    f"→ "
    f"{df[DATE_COLUMN].max().date()}"
)

print(
    f"Total records: {len(df)}"
)

print(
    "\nTarget distribution:"
)

print(
    df[TARGET]
    .value_counts()
    .sort_index()
)

positive_rate = df[TARGET].mean()

print(
    f"\nOverall positive rate: "
    f"{positive_rate:.2%}"
)


# ============================================================
# FINAL DEVELOPMENT / TEST SPLIT
# ============================================================

print_header(
    "CHRONOLOGICAL DEVELOPMENT / FINAL TEST SPLIT"
)

unique_dates = np.array(
    sorted(
        df[DATE_COLUMN]
        .dt.to_period("M")
        .unique()
    )
)

n_dates = len(unique_dates)

test_date_count = max(
    1,
    int(np.ceil(
        n_dates * TEST_DATE_FRACTION
    ))
)

test_dates = unique_dates[
    -test_date_count:
]

development_dates = unique_dates[
    :-test_date_count
]

development_mask = (
    df[DATE_COLUMN]
    .dt.to_period("M")
    .isin(development_dates)
)

test_mask = (
    df[DATE_COLUMN]
    .dt.to_period("M")
    .isin(test_dates)
)

development_df = df[
    development_mask
].copy()

test_df = df[
    test_mask
].copy()

print(
    f"Development records: "
    f"{len(development_df)}"
)

print(
    f"Final test records: "
    f"{len(test_df)}"
)

print(
    f"Development period: "
    f"{development_df[DATE_COLUMN].min().date()} "
    f"→ "
    f"{development_df[DATE_COLUMN].max().date()}"
)

print(
    f"Final test period: "
    f"{test_df[DATE_COLUMN].min().date()} "
    f"→ "
    f"{test_df[DATE_COLUMN].max().date()}"
)

print(
    "\nDevelopment target:"
)

print(
    development_df[TARGET]
    .value_counts()
    .sort_index()
)

print(
    "\nFinal test target:"
)

print(
    test_df[TARGET]
    .value_counts()
    .sort_index()
)


# ============================================================
# BUILD FEATURE MATRIX
# ============================================================

METADATA_COLUMNS = {
    TARGET,
    DATE_COLUMN,
    "date",
    "Date"
}

feature_columns = [
    c
    for c in df.columns
    if c not in METADATA_COLUMNS
]

# Remove any accidental index columns
feature_columns = [
    c
    for c in feature_columns
    if not c.lower().startswith("unnamed:")
]

print_header(
    "FEATURE PREPARATION"
)

print(
    f"Candidate ML features: "
    f"{len(feature_columns)}"
)

X_dev = development_df[
    feature_columns
].copy()

y_dev = development_df[
    TARGET
].copy()

X_test = test_df[
    feature_columns
].copy()

y_test = test_df[
    TARGET
].copy()

print(
    f"Development feature shape: "
    f"{X_dev.shape}"
)

print(
    f"Test feature shape: "
    f"{X_test.shape}"
)


# ============================================================
# TEMPORAL CROSS-VALIDATION
# ============================================================

print_header(
    "TEMPORAL OUT-OF-FOLD RANDOM FOREST PREDICTIONS"
)

folds = create_temporal_folds(
    development_df,
    N_FOLDS
)

print(
    f"Temporal folds created: {len(folds)}"
)

oof_probabilities = np.full(
    len(development_df),
    np.nan
)

fold_results = []

for fold_info in folds:

    fold_number = fold_info["fold"]

    train_idx = fold_info["train_idx"]
    val_idx = fold_info["val_idx"]

    print(
        f"\nFold {fold_number}"
    )

    print(
        f"Train: {len(train_idx)} | "
        f"Validation: {len(val_idx)}"
    )

    print(
        f"Train period: "
        f"{fold_info['train_start']} "
        f"→ "
        f"{fold_info['train_end']}"
    )

    print(
        f"Validation period: "
        f"{fold_info['val_start']} "
        f"→ "
        f"{fold_info['val_end']}"
    )

    X_train_fold = X_dev.loc[
        train_idx
    ]

    y_train_fold = y_dev.loc[
        train_idx
    ]

    X_val_fold = X_dev.loc[
        val_idx
    ]

    y_val_fold = y_dev.loc[
        val_idx
    ]

    print(
        f"Train positives: "
        f"{int(y_train_fold.sum())}"
    )

    print(
        f"Validation positives: "
        f"{int(y_val_fold.sum())}"
    )

    if y_train_fold.nunique() < 2:
        print(
            "Skipping fold because training "
            "contains only one class."
        )
        continue

    if y_val_fold.nunique() < 2:
        print(
            "Skipping fold because validation "
            "contains only one class."
        )
        continue

    pipeline, _, _ = build_rf_pipeline(
        X_train_fold
    )

    pipeline.fit(
        X_train_fold,
        y_train_fold
    )

    fold_probability = pipeline.predict_proba(
        X_val_fold
    )[:, 1]

    oof_probabilities[
        development_df.index.get_indexer(
            val_idx
        )
    ] = fold_probability

    fold_metrics = calculate_metrics(
        y_val_fold,
        fold_probability,
        0.50
    )

    fold_metrics["fold"] = fold_number
    fold_metrics["train_size"] = len(train_idx)
    fold_metrics["validation_size"] = len(val_idx)

    fold_results.append(
        fold_metrics
    )

    print(
        f"Raw RF @ 0.50 | "
        f"Precision={fold_metrics['precision']:.3f} | "
        f"Recall={fold_metrics['recall']:.3f} | "
        f"F1={fold_metrics['f1']:.3f} | "
        f"Balanced={fold_metrics['balanced_accuracy']:.3f}"
    )


# ============================================================
# OOF DATASET
# ============================================================

oof_mask = (
    ~np.isnan(oof_probabilities)
)

oof_y = y_dev.iloc[
    np.where(oof_mask)[0]
].to_numpy()

oof_raw = oof_probabilities[
    oof_mask
]

print_header(
    "OOF PREDICTION SUMMARY"
)

print(
    f"OOF records: {len(oof_y)}"
)

print(
    f"OOF positives: {int(oof_y.sum())}"
)

print(
    f"OOF positive rate: "
    f"{oof_y.mean():.2%}"
)


# ============================================================
# PLATT PROBABILITY CALIBRATION
# ============================================================

print_header(
    "PLATT PROBABILITY CALIBRATION"
)

print(
    "Training calibration model only on "
    "temporal OOF predictions..."
)

# Platt scaling:
# logistic regression on raw model probabilities

calibrator = LogisticRegression(
    C=1.0,
    solver="lbfgs",
    random_state=RANDOM_STATE
)

calibrator.fit(
    oof_raw.reshape(-1, 1),
    oof_y
)

oof_calibrated = calibrator.predict_proba(
    oof_raw.reshape(-1, 1)
)[:, 1]

raw_brier = brier_score_loss(
    oof_y,
    oof_raw
)

calibrated_brier = brier_score_loss(
    oof_y,
    oof_calibrated
)

print(
    f"Raw OOF Brier score: "
    f"{raw_brier:.6f}"
)

print(
    f"Calibrated OOF Brier score: "
    f"{calibrated_brier:.6f}"
)

print(
    f"Calibration intercept: "
    f"{calibrator.intercept_[0]:.6f}"
)

print(
    f"Calibration coefficient: "
    f"{calibrator.coef_[0][0]:.6f}"
)


# ============================================================
# CALIBRATED THRESHOLD OPTIMIZATION
# ============================================================

print_header(
    "CALIBRATED THRESHOLD OPTIMIZATION"
)

selected_threshold, threshold_df, selection_reason = (
    optimize_threshold(
        oof_y,
        oof_calibrated
    )
)

print(
    f"Selected calibrated threshold: "
    f"{selected_threshold:.2f}"
)

print(
    selection_reason
)

selected_oof_metrics = calculate_metrics(
    oof_y,
    oof_calibrated,
    selected_threshold
)

print(
    "\nSelected OOF operating point:"
)

for key in [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "brier_score",
    "fpr",
    "missed_escalation"
]:

    print(
        f"{key}: "
        f"{selected_oof_metrics[key]:.4f}"
    )


# ============================================================
# FINAL RANDOM FOREST
# ============================================================

print_header(
    "FINAL RANDOM FOREST TRAINING"
)

print(
    "Training final RF on the complete "
    "development period..."
)

final_pipeline, numeric_cols, categorical_cols = (
    build_rf_pipeline(X_dev)
)

final_pipeline.fit(
    X_dev,
    y_dev
)

print(
    "Final Random Forest training complete."
)

print(
    f"Numeric features: "
    f"{len(numeric_cols)}"
)

print(
    f"Categorical features: "
    f"{len(categorical_cols)}"
)

if categorical_cols:
    print(
        f"Categorical columns: "
        f"{categorical_cols}"
    )


# ============================================================
# FINAL TEST PREDICTION
# ============================================================

print_header(
    "FINAL UNTOUCHED TEST EVALUATION"
)

test_raw_probability = (
    final_pipeline.predict_proba(
        X_test
    )[:, 1]
)

test_calibrated_probability = (
    calibrator.predict_proba(
        test_raw_probability.reshape(-1, 1)
    )[:, 1]
)

test_metrics = calculate_metrics(
    y_test.to_numpy(),
    test_calibrated_probability,
    selected_threshold
)

print(
    "\nFINAL TEST RESULTS"
)

print("-" * 50)

print(
    f"Accuracy:            "
    f"{test_metrics['accuracy']:.2%}"
)

print(
    f"Balanced Accuracy:   "
    f"{test_metrics['balanced_accuracy']:.2%}"
)

print(
    f"Precision:           "
    f"{test_metrics['precision']:.2%}"
)

print(
    f"Recall:              "
    f"{test_metrics['recall']:.2%}"
)

print(
    f"F1 Score:            "
    f"{test_metrics['f1']:.2%}"
)

print(
    f"ROC-AUC:             "
    f"{test_metrics['roc_auc']:.4f}"
)

print(
    f"PR-AUC:              "
    f"{test_metrics['pr_auc']:.4f}"
)

print(
    f"Brier Score:         "
    f"{test_metrics['brier_score']:.4f}"
)

print(
    f"False Positive Rate: "
    f"{test_metrics['fpr']:.2%}"
)

print(
    f"Missed Escalation:   "
    f"{test_metrics['missed_escalation']:.2%}"
)

print("-" * 50)

print(
    f"TN = {test_metrics['tn']}"
)

print(
    f"FP = {test_metrics['fp']}"
)

print(
    f"FN = {test_metrics['fn']}"
)

print(
    f"TP = {test_metrics['tp']}"
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print_header(
    "RANDOM FOREST FEATURE IMPORTANCE"
)

rf_model = final_pipeline.named_steps[
    "model"
]

preprocessor = final_pipeline.named_steps[
    "preprocessor"
]

try:

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    importances = rf_model.feature_importances_

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    importance_df[
        "importance_pct"
    ] = (
        importance_df["importance"] * 100
    )

    print(
        "\nTop 20 features:"
    )

    print(
        importance_df.head(20).to_string(
            index=False
        )
    )

except Exception as e:

    print(
        f"Feature importance extraction "
        f"warning: {e}"
    )

    importance_df = pd.DataFrame()


# ============================================================
# SAVE OOF PREDICTIONS
# ============================================================

print_header(
    "SAVING OUTPUTS"
)

oof_output = development_df.iloc[
    np.where(oof_mask)[0]
].copy()

oof_output[
    "raw_rf_probability"
] = oof_raw

oof_output[
    "calibrated_probability"
] = oof_calibrated

oof_output[
    "predicted_escalation"
] = (
    oof_calibrated >= selected_threshold
).astype(int)

oof_output_file = os.path.join(
    OUTPUT_DIR,
    "v5_04_14_calibrated_oof_predictions.csv"
)

oof_output.to_csv(
    oof_output_file,
    index=False
)

print(
    f"Saved:\n{oof_output_file}"
)


# ============================================================
# SAVE THRESHOLD ANALYSIS
# ============================================================

threshold_output_file = os.path.join(
    OUTPUT_DIR,
    "v5_04_14_calibrated_threshold_analysis.csv"
)

threshold_df.to_csv(
    threshold_output_file,
    index=False
)

print(
    f"Saved:\n{threshold_output_file}"
)


# ============================================================
# SAVE FOLD RESULTS
# ============================================================

fold_output_file = os.path.join(
    OUTPUT_DIR,
    "v5_04_14_temporal_fold_results.csv"
)

pd.DataFrame(
    fold_results
).to_csv(
    fold_output_file,
    index=False
)

print(
    f"Saved:\n{fold_output_file}"
)


# ============================================================
# SAVE FINAL TEST PREDICTIONS
# ============================================================

test_output = test_df.copy()

test_output[
    "raw_rf_probability"
] = test_raw_probability

test_output[
    "calibrated_escalation_probability"
] = test_calibrated_probability

test_output[
    "selected_threshold"
] = selected_threshold

test_output[
    "predicted_material_escalation"
] = (
    test_calibrated_probability >=
    selected_threshold
).astype(int)

test_output[
    "prediction_confidence"
] = pd.cut(
    test_calibrated_probability,
    bins=[
        -np.inf,
        0.25,
        0.50,
        0.75,
        np.inf
    ],
    labels=[
        "LOW",
        "MODERATE",
        "HIGH",
        "VERY_HIGH"
    ]
)

test_prediction_file = os.path.join(
    OUTPUT_DIR,
    "v5_04_14_final_test_predictions.csv"
)

test_output.to_csv(
    test_prediction_file,
    index=False
)

print(
    f"Saved:\n{test_prediction_file}"
)


# ============================================================
# SAVE FEATURE IMPORTANCE
# ============================================================

importance_output_file = os.path.join(
    OUTPUT_DIR,
    "v5_04_14_rf_feature_importance.csv"
)

importance_df.to_csv(
    importance_output_file,
    index=False
)

print(
    f"Saved:\n{importance_output_file}"
)


# ============================================================
# SAVE FINAL METRICS
# ============================================================

metrics_record = {
    "model": "Precision-Focused Random Forest + Platt Calibration",
    "dataset": os.path.basename(INPUT_FILE),
    "total_records": int(len(df)),
    "development_records": int(len(development_df)),
    "final_test_records": int(len(test_df)),
    "positive_rate": float(positive_rate),
    "n_ml_features": int(len(feature_columns)),
    "n_numeric_features": int(len(numeric_cols)),
    "n_categorical_features": int(len(categorical_cols)),
    "n_temporal_folds": int(len(folds)),
    "selected_threshold": float(selected_threshold),
    "threshold_selection_reason": selection_reason,

    "oof_raw_brier": float(raw_brier),
    "oof_calibrated_brier": float(calibrated_brier),

    "test_accuracy": float(
        test_metrics["accuracy"]
    ),

    "test_balanced_accuracy": float(
        test_metrics["balanced_accuracy"]
    ),

    "test_precision": float(
        test_metrics["precision"]
    ),

    "test_recall": float(
        test_metrics["recall"]
    ),

    "test_f1": float(
        test_metrics["f1"]
    ),

    "test_roc_auc": float(
        test_metrics["roc_auc"]
    ),

    "test_pr_auc": float(
        test_metrics["pr_auc"]
    ),

    "test_brier_score": float(
        test_metrics["brier_score"]
    ),

    "test_fpr": float(
        test_metrics["fpr"]
    ),

    "test_missed_escalation": float(
        test_metrics["missed_escalation"]
    ),

    "test_tn": int(
        test_metrics["tn"]
    ),

    "test_fp": int(
        test_metrics["fp"]
    ),

    "test_fn": int(
        test_metrics["fn"]
    ),

    "test_tp": int(
        test_metrics["tp"]
    )
}

metrics_output_file = os.path.join(
    OUTPUT_DIR,
    "v5_04_14_final_metrics.csv"
)

pd.DataFrame(
    [metrics_record]
).to_csv(
    metrics_output_file,
    index=False
)

print(
    f"Saved:\n{metrics_output_file}"
)


# ============================================================
# SAVE MODELS
# ============================================================

model_output_file = os.path.join(
    OUTPUT_DIR,
    "v5_04_14_final_rf_model.joblib"
)

calibrator_output_file = os.path.join(
    OUTPUT_DIR,
    "v5_04_14_platt_calibrator.joblib"
)

preprocessor_output_file = os.path.join(
    OUTPUT_DIR,
    "v5_04_14_preprocessor.joblib"
)

config_output_file = os.path.join(
    OUTPUT_DIR,
    "v5_04_14_model_config.json"
)

# Complete RF pipeline
joblib.dump(
    final_pipeline,
    model_output_file
)

# Calibration model
joblib.dump(
    calibrator,
    calibrator_output_file
)

# Explicit preprocessor
joblib.dump(
    preprocessor,
    preprocessor_output_file
)

# Configuration
config = {
    "model": "RandomForestClassifier",
    "calibration": "Platt Scaling",
    "target": TARGET,
    "date_column": DATE_COLUMN,
    "random_state": RANDOM_STATE,
    "rf_parameters": RF_PARAMS,
    "selected_threshold": selected_threshold,
    "feature_count": len(feature_columns),
    "numeric_feature_count": len(numeric_cols),
    "categorical_feature_count": len(categorical_cols),
    "development_records": len(development_df),
    "test_records": len(test_df),
    "development_start": str(
        development_df[DATE_COLUMN].min()
    ),
    "development_end": str(
        development_df[DATE_COLUMN].max()
    ),
    "test_start": str(
        test_df[DATE_COLUMN].min()
    ),
    "test_end": str(
        test_df[DATE_COLUMN].max()
    )
}

with open(
    config_output_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        config,
        f,
        indent=4,
        default=str
    )

print(
    f"\nSaved RF model:\n{model_output_file}"
)

print(
    f"Saved calibrator:\n{calibrator_output_file}"
)

print(
    f"Saved preprocessor:\n{preprocessor_output_file}"
)

print(
    f"Saved configuration:\n{config_output_file}"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print_header(
    "STEP 4.14 COMPLETE"
)

print(
    "FINAL ARCHITECTURE:"
)

print(
    "Leakage-Safe Features"
)

print(
    "        ↓"
)

print(
    "Feature Selection"
)

print(
    "        ↓"
)

print(
    "Temporal Random Forest"
)

print(
    "        ↓"
)

print(
    "OOF Platt Probability Calibration"
)

print(
    "        ↓"
)

print(
    "Calibrated Threshold Optimization"
)

print(
    "        ↓"
)

print(
    "Final Risk Escalation Prediction"
)

print(
    "        ↓"
)

print(
    "SHAP Explanation (Step 4.15)"
)

print(
    "\nFINAL TEST:"
)

print(
    f"Accuracy          : {test_metrics['accuracy']:.2%}"
)

print(
    f"Balanced Accuracy : "
    f"{test_metrics['balanced_accuracy']:.2%}"
)

print(
    f"Precision         : "
    f"{test_metrics['precision']:.2%}"
)

print(
    f"Recall            : "
    f"{test_metrics['recall']:.2%}"
)

print(
    f"F1                : "
    f"{test_metrics['f1']:.2%}"
)

print(
    f"ROC-AUC           : "
    f"{test_metrics['roc_auc']:.4f}"
)

print(
    f"PR-AUC            : "
    f"{test_metrics['pr_auc']:.4f}"
)

print(
    f"Brier Score       : "
    f"{test_metrics['brier_score']:.4f}"
)

print(
    f"FPR               : "
    f"{test_metrics['fpr']:.2%}"
)

print(
    f"Selected Threshold: "
    f"{selected_threshold:.2f}"
)

print(
    "\nAll Step 4.14 artifacts saved successfully."
)