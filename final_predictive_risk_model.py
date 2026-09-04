"""
CoalMineAI
FINAL PREDICTIVE RISK MODEL

Architecture
-------------
Leakage-safe features
        ↓
Feature selection
        ↓
Temporal Random Forest
        ↓
Temporal OOF predictions
        ↓
Platt probability calibration
        ↓
Calibrated threshold optimization
        ↓
Final untouched test evaluation
        ↓
Production model artifacts

Model:
    Precision-focused Random Forest

Calibration:
    Platt Scaling

Important:
    The final test period is NEVER used for:
        - feature selection
        - model selection
        - calibration
        - threshold selection
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
    brier_score_loss,
    confusion_matrix
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
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
    "v5",
    "FINAL"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

TARGET = "material_escalation"
DATE_COLUMN = "record_date"

RANDOM_STATE = 42

# Last 15% of chronological months = untouched test
TEST_FRACTION = 0.15

# Number of expanding temporal validation folds
N_FOLDS = 4

RF_PARAMS = {
    "n_estimators": 700,
    "max_depth": 8,
    "min_samples_split": 6,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "class_weight": {
        0: 1.0,
        1: 2.5
    },
    "bootstrap": True,
    "random_state": RANDOM_STATE,
    "n_jobs": -1
}


# ============================================================
# PRINT HELPERS
# ============================================================

def header(text):

    print("\n")
    print("=" * 80)
    print(text)
    print("=" * 80)


# ============================================================
# METRICS
# ============================================================

def evaluate(
    y_true,
    probabilities,
    threshold
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    balanced = balanced_accuracy_score(
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
        else 0
    )

    missed = (
        fn / (fn + tp)
        if (fn + tp) > 0
        else 0
    )

    return {

        "threshold": float(threshold),

        "accuracy": float(accuracy),

        "balanced_accuracy": float(
            balanced
        ),

        "precision": float(
            precision
        ),

        "recall": float(
            recall
        ),

        "f1": float(
            f1
        ),

        "roc_auc": float(
            roc_auc
        ),

        "pr_auc": float(
            pr_auc
        ),

        "brier_score": float(
            brier
        ),

        "fpr": float(
            fpr
        ),

        "missed_escalation": float(
            missed
        ),

        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }


# ============================================================
# PREPROCESSOR
# ============================================================

def create_pipeline(X):

    numeric_columns = (
        X
        .select_dtypes(
            include=[np.number]
        )
        .columns
        .tolist()
    )

    categorical_columns = [
        column
        for column in X.columns
        if column not in numeric_columns
    ]

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    transformers = []

    if numeric_columns:

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_columns
            )
        )

    if categorical_columns:

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
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
        numeric_columns,
        categorical_columns
    )


# ============================================================
# TEMPORAL FOLD GENERATOR
# ============================================================

def temporal_folds(
    development_df,
    n_folds
):

    monthly_dates = np.array(
        sorted(
            development_df[
                DATE_COLUMN
            ]
            .dt
            .to_period("M")
            .unique()
        )
    )

    total_dates = len(
        monthly_dates
    )

    # Use approximately first 50% as initial
    # training history.
    initial_train_dates = max(
        24,
        int(
            total_dates * 0.50
        )
    )

    remaining = (
        total_dates
        -
        initial_train_dates
    )

    validation_size = max(
        1,
        remaining // n_folds
    )

    folds = []

    for i in range(n_folds):

        train_end = (
            initial_train_dates
            +
            i * validation_size
        )

        val_start = train_end

        if i == n_folds - 1:

            val_end = total_dates

        else:

            val_end = min(
                val_start + validation_size,
                total_dates
            )

        if val_start >= val_end:
            continue

        train_periods = set(
            monthly_dates[
                :train_end
            ]
        )

        validation_periods = set(
            monthly_dates[
                val_start:val_end
            ]
        )

        train_mask = (
            development_df[
                DATE_COLUMN
            ]
            .dt
            .to_period("M")
            .isin(train_periods)
        )

        validation_mask = (
            development_df[
                DATE_COLUMN
            ]
            .dt
            .to_period("M")
            .isin(validation_periods)
        )

        train_indices = (
            development_df
            .index[
                train_mask
            ]
            .to_numpy()
        )

        validation_indices = (
            development_df
            .index[
                validation_mask
            ]
            .to_numpy()
        )

        if (
            len(train_indices) == 0
            or
            len(validation_indices) == 0
        ):
            continue

        folds.append(
            {
                "fold": len(folds) + 1,
                "train_idx": train_indices,
                "val_idx": validation_indices,
                "train_start": str(
                    monthly_dates[0]
                ),
                "train_end": str(
                    monthly_dates[
                        train_end - 1
                    ]
                ),
                "val_start": str(
                    monthly_dates[
                        val_start
                    ]
                ),
                "val_end": str(
                    monthly_dates[
                        val_end - 1
                    ]
                )
            }
        )

    return folds


# ============================================================
# THRESHOLD OPTIMIZATION
# ============================================================

def optimize_threshold(
    y_true,
    calibrated_probability
):

    results = []

    thresholds = np.arange(
        0.20,
        0.901,
        0.01
    )

    for threshold in thresholds:

        metrics = evaluate(
            y_true,
            calibrated_probability,
            threshold
        )

        precision = metrics[
            "precision"
        ]

        recall = metrics[
            "recall"
        ]

        f1 = metrics[
            "f1"
        ]

        balanced = metrics[
            "balanced_accuracy"
        ]

        # Practical precision floor.
        precision_ok = (
            precision >= 0.55
        )

        # Avoid extremely conservative
        # thresholds that detect almost nothing.
        recall_ok = (
            recall >= 0.30
        )

        metrics[
            "constraint_satisfied"
        ] = (
            precision_ok
            and
            recall_ok
        )

        # Precision-aware objective.
        metrics[
            "selection_score"
        ] = (
            0.35 * f1
            +
            0.30 * balanced
            +
            0.25 * precision
            +
            0.10 * recall
        )

        results.append(
            metrics
        )

    results_df = pd.DataFrame(
        results
    )

    valid = results_df[
        results_df[
            "constraint_satisfied"
        ]
    ].copy()

    if len(valid) > 0:

        # F1 first, then balanced accuracy,
        # then precision.
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

        reason = (
            "Selected from thresholds with "
            "precision >= 55% and "
            "recall >= 30%."
        )

    else:

        results_df = results_df.sort_values(
            by=[
                "selection_score",
                "balanced_accuracy",
                "precision"
            ],
            ascending=False
        )

        selected = results_df.iloc[0]

        reason = (
            "No threshold satisfied the "
            "precision/recall constraints. "
            "Selected best composite score."
        )

    return (
        float(
            selected["threshold"]
        ),
        results_df,
        reason
    )


# ============================================================
# LOAD DATA
# ============================================================

header(
    "FINAL PREDICTIVE RISK MODEL"
)

print(
    f"Input:\n{INPUT_FILE}"
)

df = pd.read_csv(
    INPUT_FILE
)

print(
    f"\nLoaded: {df.shape}"
)

if TARGET not in df.columns:

    raise ValueError(
        f"Missing target: {TARGET}"
    )

if DATE_COLUMN not in df.columns:

    raise ValueError(
        f"Missing date column: {DATE_COLUMN}"
    )

df[DATE_COLUMN] = pd.to_datetime(
    df[DATE_COLUMN],
    errors="coerce"
)

if df[DATE_COLUMN].isna().any():

    raise ValueError(
        "Invalid dates detected."
    )

df[TARGET] = pd.to_numeric(
    df[TARGET],
    errors="coerce"
)

df = df[
    df[TARGET].isin([0, 1])
].copy()

df[TARGET] = df[
    TARGET
].astype(int)

df = df.sort_values(
    DATE_COLUMN
).reset_index(
    drop=True
)

print(
    f"Date range: "
    f"{df[DATE_COLUMN].min().date()} "
    f"→ "
    f"{df[DATE_COLUMN].max().date()}"
)

print(
    "\nTarget distribution:"
)

print(
    df[TARGET]
    .value_counts()
    .sort_index()
)


# ============================================================
# FINAL HOLDOUT
# ============================================================

header(
    "FINAL CHRONOLOGICAL HOLDOUT"
)

unique_months = np.array(
    sorted(
        df[DATE_COLUMN]
        .dt
        .to_period("M")
        .unique()
    )
)

number_of_months = len(
    unique_months
)

test_months_count = max(
    1,
    int(
        np.ceil(
            number_of_months
            *
            TEST_FRACTION
        )
    )
)

test_months = unique_months[
    -test_months_count:
]

development_months = unique_months[
    :-test_months_count
]

development_mask = (
    df[DATE_COLUMN]
    .dt
    .to_period("M")
    .isin(
        development_months
    )
)

test_mask = (
    df[DATE_COLUMN]
    .dt
    .to_period("M")
    .isin(
        test_months
    )
)

development_df = df[
    development_mask
].copy()

test_df = df[
    test_mask
].copy()

print(
    f"Development: {len(development_df)}"
)

print(
    f"Final test:  {len(test_df)}"
)

print(
    f"Development: "
    f"{development_df[DATE_COLUMN].min().date()} "
    f"→ "
    f"{development_df[DATE_COLUMN].max().date()}"
)

print(
    f"Final test: "
    f"{test_df[DATE_COLUMN].min().date()} "
    f"→ "
    f"{test_df[DATE_COLUMN].max().date()}"
)


# ============================================================
# FEATURES
# ============================================================

header(
    "FEATURE PREPARATION"
)

metadata = {
    TARGET,
    DATE_COLUMN,
    "date",
    "Date"
}

feature_columns = [
    column
    for column in df.columns
    if column not in metadata
]

# Remove accidental CSV index columns.
feature_columns = [
    column
    for column in feature_columns
    if not column.lower().startswith(
        "unnamed:"
    )
]

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
    f"ML features: {len(feature_columns)}"
)

print(
    f"Development X: {X_dev.shape}"
)

print(
    f"Test X: {X_test.shape}"
)


# ============================================================
# TEMPORAL OOF
# ============================================================

header(
    "TEMPORAL OUT-OF-FOLD TRAINING"
)

folds = temporal_folds(
    development_df,
    N_FOLDS
)

print(
    f"Temporal folds: {len(folds)}"
)

oof_probability = np.full(
    len(development_df),
    np.nan
)

fold_records = []

for fold in folds:

    fold_number = fold[
        "fold"
    ]

    train_idx = fold[
        "train_idx"
    ]

    val_idx = fold[
        "val_idx"
    ]

    print(
        f"\nFold {fold_number}"
    )

    print(
        f"Train: {len(train_idx)}"
    )

    print(
        f"Validation: {len(val_idx)}"
    )

    print(
        f"Train period: "
        f"{fold['train_start']} "
        f"→ "
        f"{fold['train_end']}"
    )

    print(
        f"Validation period: "
        f"{fold['val_start']} "
        f"→ "
        f"{fold['val_end']}"
    )

    X_train = X_dev.loc[
        train_idx
    ]

    y_train = y_dev.loc[
        train_idx
    ]

    X_val = X_dev.loc[
        val_idx
    ]

    y_val = y_dev.loc[
        val_idx
    ]

    if y_train.nunique() < 2:

        print(
            "Skipping fold: "
            "training has one class."
        )

        continue

    if y_val.nunique() < 2:

        print(
            "Skipping fold: "
            "validation has one class."
        )

        continue

    pipeline, _, _ = create_pipeline(
        X_train
    )

    pipeline.fit(
        X_train,
        y_train
    )

    probabilities = (
        pipeline
        .predict_proba(
            X_val
        )[:, 1]
    )

    # Convert global dataframe indices
    # into positional indices.
    positions = (
        development_df
        .index
        .get_indexer(
            val_idx
        )
    )

    oof_probability[
        positions
    ] = probabilities

    metrics = evaluate(
        y_val,
        probabilities,
        0.50
    )

    metrics[
        "fold"
    ] = fold_number

    metrics[
        "train_size"
    ] = len(train_idx)

    metrics[
        "validation_size"
    ] = len(val_idx)

    fold_records.append(
        metrics
    )

    print(
        f"Precision: "
        f"{metrics['precision']:.3f}"
    )

    print(
        f"Recall: "
        f"{metrics['recall']:.3f}"
    )

    print(
        f"F1: "
        f"{metrics['f1']:.3f}"
    )

    print(
        f"Balanced Accuracy: "
        f"{metrics['balanced_accuracy']:.3f}"
    )


# ============================================================
# OOF DATA
# ============================================================

valid_oof = ~np.isnan(
    oof_probability
)

oof_y = y_dev.iloc[
    np.where(valid_oof)[0]
].to_numpy()

oof_raw = oof_probability[
    valid_oof
]

print(
    f"\nOOF records: {len(oof_y)}"
)

print(
    f"OOF positive records: "
    f"{int(oof_y.sum())}"
)


# ============================================================
# PLATT CALIBRATION
# ============================================================

header(
    "PLATT PROBABILITY CALIBRATION"
)

calibrator = LogisticRegression(
    C=1.0,
    solver="lbfgs",
    random_state=RANDOM_STATE
)

calibrator.fit(
    oof_raw.reshape(-1, 1),
    oof_y
)

oof_calibrated = (
    calibrator
    .predict_proba(
        oof_raw.reshape(-1, 1)
    )[:, 1]
)

raw_brier = (
    brier_score_loss(
        oof_y,
        oof_raw
    )
)

calibrated_brier = (
    brier_score_loss(
        oof_y,
        oof_calibrated
    )
)

print(
    f"Raw Brier: "
    f"{raw_brier:.6f}"
)

print(
    f"Calibrated Brier: "
    f"{calibrated_brier:.6f}"
)

print(
    f"Calibration improvement: "
    f"{raw_brier - calibrated_brier:.6f}"
)


# ============================================================
# THRESHOLD
# ============================================================

header(
    "CALIBRATED THRESHOLD OPTIMIZATION"
)

selected_threshold, threshold_df, threshold_reason = (
    optimize_threshold(
        oof_y,
        oof_calibrated
    )
)

print(
    f"Selected threshold: "
    f"{selected_threshold:.2f}"
)

print(
    threshold_reason
)

oof_metrics = evaluate(
    oof_y,
    oof_calibrated,
    selected_threshold
)

print(
    "\nCalibrated OOF metrics:"
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
        f"{oof_metrics[key]:.4f}"
    )


# ============================================================
# FINAL RF
# ============================================================

header(
    "FINAL RANDOM FOREST"
)

final_pipeline, numeric_cols, categorical_cols = (
    create_pipeline(
        X_dev
    )
)

final_pipeline.fit(
    X_dev,
    y_dev
)

print(
    "Final RF trained on complete development data."
)

print(
    f"Numeric features: "
    f"{len(numeric_cols)}"
)

print(
    f"Categorical features: "
    f"{len(categorical_cols)}"
)


# ============================================================
# FINAL TEST
# ============================================================

header(
    "FINAL UNTOUCHED TEST"
)

test_raw_probability = (
    final_pipeline
    .predict_proba(
        X_test
    )[:, 1]
)

test_calibrated_probability = (
    calibrator
    .predict_proba(
        test_raw_probability.reshape(
            -1,
            1
        )
    )[:, 1]
)

test_metrics = evaluate(
    y_test.to_numpy(),
    test_calibrated_probability,
    selected_threshold
)

print(
    "\nFINAL TEST RESULTS"
)

print(
    "-" * 55
)

print(
    f"Accuracy: "
    f"{test_metrics['accuracy']:.2%}"
)

print(
    f"Balanced Accuracy: "
    f"{test_metrics['balanced_accuracy']:.2%}"
)

print(
    f"Precision: "
    f"{test_metrics['precision']:.2%}"
)

print(
    f"Recall: "
    f"{test_metrics['recall']:.2%}"
)

print(
    f"F1: "
    f"{test_metrics['f1']:.2%}"
)

print(
    f"ROC-AUC: "
    f"{test_metrics['roc_auc']:.4f}"
)

print(
    f"PR-AUC: "
    f"{test_metrics['pr_auc']:.4f}"
)

print(
    f"Brier Score: "
    f"{test_metrics['brier_score']:.4f}"
)

print(
    f"False Positive Rate: "
    f"{test_metrics['fpr']:.2%}"
)

print(
    f"Missed Escalation: "
    f"{test_metrics['missed_escalation']:.2%}"
)

print(
    "-" * 55
)

print(
    f"TN: {test_metrics['tn']}"
)

print(
    f"FP: {test_metrics['fp']}"
)

print(
    f"FN: {test_metrics['fn']}"
)

print(
    f"TP: {test_metrics['tp']}"
)


# ============================================================
# FINAL PREDICTIONS
# ============================================================

predictions_df = test_df.copy()

predictions_df[
    "raw_rf_probability"
] = test_raw_probability

predictions_df[
    "calibrated_escalation_probability"
] = test_calibrated_probability

predictions_df[
    "selected_threshold"
] = selected_threshold

predictions_df[
    "predicted_material_escalation"
] = (
    test_calibrated_probability
    >= selected_threshold
).astype(int)

predictions_df[
    "risk_status"
] = np.where(
    test_calibrated_probability
    >= selected_threshold,
    "ESCALATION",
    "NO_ESCALATION"
)

predictions_df[
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

prediction_file = os.path.join(
    OUTPUT_DIR,
    "final_risk_predictions.csv"
)

predictions_df.to_csv(
    prediction_file,
    index=False
)


# ============================================================
# OOF OUTPUT
# ============================================================

oof_df = development_df.iloc[
    np.where(valid_oof)[0]
].copy()

oof_df[
    "raw_probability"
] = oof_raw

oof_df[
    "calibrated_probability"
] = oof_calibrated

oof_df[
    "selected_threshold"
] = selected_threshold

oof_df[
    "predicted_escalation"
] = (
    oof_calibrated
    >= selected_threshold
).astype(int)

oof_file = os.path.join(
    OUTPUT_DIR,
    "final_oof_predictions.csv"
)

oof_df.to_csv(
    oof_file,
    index=False
)


# ============================================================
# THRESHOLD OUTPUT
# ============================================================

threshold_file = os.path.join(
    OUTPUT_DIR,
    "final_threshold_analysis.csv"
)

threshold_df.to_csv(
    threshold_file,
    index=False
)


# ============================================================
# FOLD OUTPUT
# ============================================================

fold_file = os.path.join(
    OUTPUT_DIR,
    "final_temporal_fold_results.csv"
)

pd.DataFrame(
    fold_records
).to_csv(
    fold_file,
    index=False
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

header(
    "FEATURE IMPORTANCE"
)

rf_model = final_pipeline[
    "model"
]

preprocessor = final_pipeline[
    "preprocessor"
]

try:

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    importances = (
        rf_model
        .feature_importances_
    )

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances
        }
    )

    importance_df[
        "importance_pct"
    ] = (
        importance_df[
            "importance"
        ]
        * 100
    )

    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    print(
        importance_df
        .head(20)
        .to_string(
            index=False
        )
    )

except Exception as error:

    print(
        f"Feature importance warning: "
        f"{error}"
    )

    importance_df = pd.DataFrame()


importance_file = os.path.join(
    OUTPUT_DIR,
    "final_feature_importance.csv"
)

importance_df.to_csv(
    importance_file,
    index=False
)


# ============================================================
# FINAL METRICS
# ============================================================

metrics = {

    "architecture":
        "Leakage-safe Temporal Random Forest + Platt Calibration",

    "model":
        "RandomForestClassifier",

    "calibration":
        "Platt Scaling",

    "target":
        TARGET,

    "records":
        int(len(df)),

    "development_records":
        int(len(development_df)),

    "final_test_records":
        int(len(test_df)),

    "ml_features":
        int(len(feature_columns)),

    "numeric_features":
        int(len(numeric_cols)),

    "categorical_features":
        int(len(categorical_cols)),

    "temporal_folds":
        int(len(folds)),

    "selected_threshold":
        float(selected_threshold),

    "raw_oof_brier":
        float(raw_brier),

    "calibrated_oof_brier":
        float(calibrated_brier),

    "test_accuracy":
        test_metrics[
            "accuracy"
        ],

    "test_balanced_accuracy":
        test_metrics[
            "balanced_accuracy"
        ],

    "test_precision":
        test_metrics[
            "precision"
        ],

    "test_recall":
        test_metrics[
            "recall"
        ],

    "test_f1":
        test_metrics[
            "f1"
        ],

    "test_roc_auc":
        test_metrics[
            "roc_auc"
        ],

    "test_pr_auc":
        test_metrics[
            "pr_auc"
        ],

    "test_brier_score":
        test_metrics[
            "brier_score"
        ],

    "test_fpr":
        test_metrics[
            "fpr"
        ],

    "test_missed_escalation":
        test_metrics[
            "missed_escalation"
        ],

    "tn":
        test_metrics["tn"],

    "fp":
        test_metrics["fp"],

    "fn":
        test_metrics["fn"],

    "tp":
        test_metrics["tp"]
}

metrics_file = os.path.join(
    OUTPUT_DIR,
    "final_metrics.csv"
)

pd.DataFrame(
    [metrics]
).to_csv(
    metrics_file,
    index=False
)


# ============================================================
# SAVE MODELS
# ============================================================

header(
    "SAVING PRODUCTION ARTIFACTS"
)

model_file = os.path.join(
    OUTPUT_DIR,
    "final_risk_model.joblib"
)

calibrator_file = os.path.join(
    OUTPUT_DIR,
    "final_calibrator.joblib"
)

preprocessor_file = os.path.join(
    OUTPUT_DIR,
    "final_preprocessor.joblib"
)

config_file = os.path.join(
    OUTPUT_DIR,
    "final_model_config.json"
)

joblib.dump(
    final_pipeline,
    model_file
)

joblib.dump(
    calibrator,
    calibrator_file
)

joblib.dump(
    preprocessor,
    preprocessor_file
)

config = {

    "model":
        "RandomForestClassifier",

    "calibration":
        "Platt Scaling",

    "target":
        TARGET,

    "date_column":
        DATE_COLUMN,

    "selected_threshold":
        selected_threshold,

    "threshold_reason":
        threshold_reason,

    "feature_columns":
        feature_columns,

    "numeric_columns":
        numeric_cols,

    "categorical_columns":
        categorical_cols,

    "rf_parameters":
        RF_PARAMS,

    "development_start":
        str(
            development_df[
                DATE_COLUMN
            ].min()
        ),

    "development_end":
        str(
            development_df[
                DATE_COLUMN
            ].max()
        ),

    "test_start":
        str(
            test_df[
                DATE_COLUMN
            ].min()
        ),

    "test_end":
        str(
            test_df[
                DATE_COLUMN
            ].max()
        )
}

with open(
    config_file,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        config,
        file,
        indent=4,
        default=str
    )

print(
    f"Model:\n{model_file}"
)

print(
    f"Calibrator:\n{calibrator_file}"
)

print(
    f"Preprocessor:\n{preprocessor_file}"
)

print(
    f"Config:\n{config_file}"
)


# ============================================================
# COMPLETE
# ============================================================

header(
    "FINAL PREDICTIVE MODEL COMPLETE"
)

print(
    """
FINAL ARCHITECTURE

Historical Mine Data
        ↓
Leakage-Safe Features
        ↓
Feature Selection
        ↓
Temporal Random Forest
        ↓
Temporal OOF Predictions
        ↓
Platt Probability Calibration
        ↓
Calibrated Threshold Optimization
        ↓
Final Risk Prediction
        ↓
SHAP Explainability
        ↓
Regulatory RAG
        ↓
Evidence-Gap Intelligence
        ↓
Governance Recommendation
"""
)

print(
    "\nFinal test metrics:"
)

print(
    f"Accuracy           : "
    f"{test_metrics['accuracy']:.2%}"
)

print(
    f"Balanced Accuracy  : "
    f"{test_metrics['balanced_accuracy']:.2%}"
)

print(
    f"Precision          : "
    f"{test_metrics['precision']:.2%}"
)

print(
    f"Recall             : "
    f"{test_metrics['recall']:.2%}"
)

print(
    f"F1                 : "
    f"{test_metrics['f1']:.2%}"
)

print(
    f"ROC-AUC            : "
    f"{test_metrics['roc_auc']:.4f}"
)

print(
    f"PR-AUC             : "
    f"{test_metrics['pr_auc']:.4f}"
)

print(
    f"Calibrated threshold: "
    f"{selected_threshold:.2f}"
)

print(
    "\nFINAL MODEL SAVED SUCCESSFULLY."
)