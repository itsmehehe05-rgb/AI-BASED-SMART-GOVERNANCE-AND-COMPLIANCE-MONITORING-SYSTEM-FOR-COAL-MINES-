import os
import warnings
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

warnings.filterwarnings("ignore")

# ============================================================
# OPTIONAL XGBOOST
# ============================================================

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# ============================================================
# PHASE 7C VERSION 4
# ADVANCED ML PREDICTIVE RISK ESCALATION ENGINE
# ============================================================

INPUT_FILE = r"D:\CoalMineAI\outputs\early_warning_analysis.csv"

OUTPUT_DIR = r"D:\CoalMineAI\outputs"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "predictive_risk_escalation.csv"
)

SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "mine_escalation_summary.csv"
)

MODEL_REPORT_FILE = os.path.join(
    OUTPUT_DIR,
    "predictive_escalation_model_performance.csv"
)

FEATURE_IMPORTANCE_FILE = os.path.join(
    OUTPUT_DIR,
    "predictive_escalation_feature_importance.csv"
)

RANDOM_STATE = 42


# ============================================================
# UTILITIES
# ============================================================

def num(value, default=0.0):

    try:
        value = float(value)

        if not np.isfinite(value):
            return default

        return value

    except Exception:
        return default


def clamp(value, low=0.0, high=100.0):

    return max(
        low,
        min(num(value), high)
    )


def risk_numeric(value):

    mapping = {
        "LOW": 20,
        "MEDIUM": 55,
        "HIGH": 85,
        "UNRELIABLE": 100
    }

    return mapping.get(
        str(value).upper().strip(),
        50
    )


def risk_level(score):

    score = num(score)

    if score >= 75:
        return "HIGH"

    if score >= 40:
        return "MEDIUM"

    return "LOW"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("\n[1] Loading early-warning analysis...", flush=True)

    print(
        f"File: {INPUT_FILE}",
        flush=True
    )

    if not os.path.exists(INPUT_FILE):

        raise FileNotFoundError(
            f"\nFile not found:\n{INPUT_FILE}\n"
            "\nRun Phase 7B first."
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Rows loaded   : {len(df)}",
        flush=True
    )

    print(
        f"Columns loaded: {len(df.columns)}",
        flush=True
    )

    required = [
        "date",
        "subsidiary",
        "production_risk",
        "governance_score",
        "early_warning_score",
        "overall_operational_risk"
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:

        raise ValueError(
            "\nMissing required columns:\n"
            + "\n".join(
                f" - {c}"
                for c in missing
            )
        )

    return df


# ============================================================
# CLEAN DATA
# ============================================================

def clean_data(df):

    print(
        "\n[2] Cleaning data...",
        flush=True
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

    for col in df.columns:

        if col not in [
            "date",
            "subsidiary",
            "production_risk"
        ]:

            df[col] = pd.to_numeric(
                df[col],
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
        f"Clean rows   : {len(df)}",
        flush=True
    )

    print(
        f"Subsidiaries : "
        f"{df['subsidiary'].nunique()}",
        flush=True
    )

    return df


# ============================================================
# TEMPORAL FEATURE ENGINEERING
# ============================================================

def build_temporal_features(group):

    group = group.sort_values(
        "date"
    ).copy()

    # --------------------------------------------------------
    # Convert production risk to numerical representation
    # --------------------------------------------------------

    group["risk_numeric"] = (
        group["production_risk"]
        .apply(risk_numeric)
    )

    base_features = [
        "risk_numeric",
        "governance_score",
        "early_warning_score",
        "overall_operational_risk"
    ]

    optional_features = [
        "equipment_risk",
        "logistics_risk",
        "weather_risk",
        "workforce_risk",
        "predicted_target_achievement_pct"
    ]

    for col in optional_features:

        if col in group.columns:

            base_features.append(
                col
            )

    # --------------------------------------------------------
    # Lag features
    # --------------------------------------------------------

    for col in base_features:

        for lag in [1, 2, 3]:

            group[
                f"{col}_lag_{lag}"
            ] = (
                group[col]
                .shift(lag)
            )

    # --------------------------------------------------------
    # Rolling features
    # Shift first to prevent current observation
    # from contaminating historical rolling values.
    # --------------------------------------------------------

    for col in base_features:

        historical = (
            group[col]
            .shift(1)
        )

        group[
            f"{col}_rolling_mean_3"
        ] = (
            historical
            .rolling(
                3,
                min_periods=1
            )
            .mean()
        )

        group[
            f"{col}_rolling_std_3"
        ] = (
            historical
            .rolling(
                3,
                min_periods=1
            )
            .std()
        )

        group[
            f"{col}_rolling_mean_6"
        ] = (
            historical
            .rolling(
                6,
                min_periods=1
            )
            .mean()
        )

    # --------------------------------------------------------
    # Change features
    # --------------------------------------------------------

    for col in base_features:

        group[
            f"{col}_change_1"
        ] = (
            group[col]
            - group[col].shift(1)
        )

        group[
            f"{col}_change_3"
        ] = (
            group[col]
            - group[col].shift(3)
        )

    # --------------------------------------------------------
    # Risk dynamics
    # --------------------------------------------------------

    group["risk_change_1"] = (
        group["risk_numeric"]
        - group["risk_numeric"].shift(1)
    )

    group["risk_change_3"] = (
        group["risk_numeric"]
        - group["risk_numeric"].shift(3)
    )

    group["risk_acceleration"] = (
        group["risk_change_1"]
        - group["risk_change_1"].shift(1)
    )

    # --------------------------------------------------------
    # Warning dynamics
    # --------------------------------------------------------

    warning_change = (
        group["early_warning_score"]
        .diff()
    )

    group[
        "warning_acceleration"
    ] = (
        warning_change
        - warning_change.shift(1)
    )

    # --------------------------------------------------------
    # Operational dynamics
    # --------------------------------------------------------

    operational_change = (
        group["overall_operational_risk"]
        .diff()
    )

    group[
        "operational_acceleration"
    ] = (
        operational_change
        - operational_change.shift(1)
    )

    # --------------------------------------------------------
    # Governance dynamics
    # --------------------------------------------------------

    governance_change = (
        group["governance_score"]
        .diff()
    )

    group[
        "governance_acceleration"
    ] = (
        governance_change
        - governance_change.shift(1)
    )

    return group


def build_features(df):

    print(
        "\n[3] Building temporal ML features...",
        flush=True
    )

    groups = []

    for subsidiary, group in df.groupby(
        "subsidiary",
        sort=False
    ):

        groups.append(
            build_temporal_features(
                group
            )
        )

    result = pd.concat(
        groups,
        ignore_index=True
    )

    result = result.sort_values(
        [
            "date",
            "subsidiary"
        ]
    ).reset_index(
        drop=True
    )

    print(
        f"Temporal feature rows: "
        f"{len(result)}",
        flush=True
    )

    return result


# ============================================================
# FUTURE ESCALATION TARGET
# ============================================================

def create_target(group):

    group = group.sort_values(
        "date"
    ).copy()

    current = (
        group["risk_numeric"]
    )

    future = (
        group["risk_numeric"]
        .shift(-1)
    )

    group[
        "future_risk_numeric"
    ] = future

    group[
        "future_risk_change"
    ] = (
        future - current
    )

    group[
        "future_risk_level"
    ] = future.apply(
        lambda x:
        risk_level(x)
        if pd.notna(x)
        else np.nan
    )

    # --------------------------------------------------------
    # TARGET
    #
    # 1 = next month's risk is higher
    # 0 = same or lower
    # --------------------------------------------------------

    group[
        "future_escalation"
    ] = (
        group[
            "future_risk_change"
        ] > 0
    ).astype(int)

    return group


def build_target(df):

    print(
        "\n[4] Creating future escalation target...",
        flush=True
    )

    groups = []

    for subsidiary, group in df.groupby(
        "subsidiary",
        sort=False
    ):

        groups.append(
            create_target(
                group
            )
        )

    df = pd.concat(
        groups,
        ignore_index=True
    )

    before = len(df)

    # Last month of every mine has no future target.
    df = df[
        df["future_risk_numeric"].notna()
    ].copy()

    removed = (
        before - len(df)
    )

    print(
        f"Rows without future target removed: "
        f"{removed}",
        flush=True
    )

    print(
        "\nFuture escalation distribution:",
        flush=True
    )

    print(
        df[
            "future_escalation"
        ].value_counts()
        .sort_index(),
        flush=True
    )

    positive_rate = (
        df[
            "future_escalation"
        ].mean()
        * 100
    )

    print(
        f"\nPositive escalation rate: "
        f"{positive_rate:.2f}%",
        flush=True
    )

    return df


# ============================================================
# FEATURE SELECTION
# ============================================================

def select_features(df):

    print(
        "\n[5] Selecting leakage-safe features...",
        flush=True
    )

    excluded = {
        "date",
        "production_risk",

        "future_risk_numeric",
        "future_risk_change",
        "future_risk_level",
        "future_escalation"
    }

    leakage_terms = [
        "future",
        "next_month",
        "actual_next",
        "target_label"
    ]

    feature_columns = []

    for col in df.columns:

        if col in excluded:
            continue

        lower = col.lower()

        if any(
            term in lower
            for term in leakage_terms
        ):
            continue

        if (
            pd.api.types.is_numeric_dtype(
                df[col]
            )
            or col == "subsidiary"
        ):

            feature_columns.append(
                col
            )

    print(
        f"Selected features: "
        f"{len(feature_columns)}",
        flush=True
    )

    for feature in feature_columns:

        print(
            f"  - {feature}",
            flush=True
        )

    return feature_columns


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

def chronological_split(df):

    print(
        "\n[6] Creating chronological split...",
        flush=True
    )

    unique_dates = sorted(
        df["date"]
        .dropna()
        .unique()
    )

    n_dates = len(
        unique_dates
    )

    train_end = int(
        n_dates * 0.70
    )

    validation_end = int(
        n_dates * 0.85
    )

    train_dates = unique_dates[
        :train_end
    ]

    validation_dates = unique_dates[
        train_end:validation_end
    ]

    test_dates = unique_dates[
        validation_end:
    ]

    train = df[
        df["date"].isin(
            train_dates
        )
    ].copy()

    validation = df[
        df["date"].isin(
            validation_dates
        )
    ].copy()

    test = df[
        df["date"].isin(
            test_dates
        )
    ].copy()

    print(
        f"Train      : {len(train)} rows",
        flush=True
    )

    print(
        f"Validation : {len(validation)} rows",
        flush=True
    )

    print(
        f"Test       : {len(test)} rows",
        flush=True
    )

    print(
        f"\nTrain dates:"
        f" {train['date'].min().date()} → "
        f"{train['date'].max().date()}",
        flush=True
    )

    print(
        f"Validation dates:"
        f" {validation['date'].min().date()} → "
        f"{validation['date'].max().date()}",
        flush=True
    )

    print(
        f"Test dates:"
        f" {test['date'].min().date()} → "
        f"{test['date'].max().date()}",
        flush=True
    )

    return (
        train,
        validation,
        test
    )


# ============================================================
# PREPROCESSOR
# ============================================================

def create_preprocessor(
    feature_columns,
    df
):

    categorical_features = []

    numeric_features = []

    for col in feature_columns:

        if col == "subsidiary":

            categorical_features.append(
                col
            )

        elif pd.api.types.is_numeric_dtype(
            df[col]
        ):

            numeric_features.append(
                col
            )

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    # Compatibility with both old and new
    # scikit-learn versions.

    try:

        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )

    except TypeError:

        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse=False
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
                "onehot",
                encoder
            )
        ]
    )

    transformers = []

    if numeric_features:

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_features
            )
        )

    if categorical_features:

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )


# ============================================================
# MODEL DEFINITIONS
# ============================================================

def create_models(
    y_train
):

    models = {}

    # --------------------------------------------------------
    # Class imbalance calculation
    # --------------------------------------------------------

    positive = max(
        int((y_train == 1).sum()),
        1
    )

    negative = max(
        int((y_train == 0).sum()),
        1
    )

    scale_pos_weight = (
        negative / positive
    )

    print(
        "\nClass imbalance:",
        flush=True
    )

    print(
        f"Negative samples: {negative}",
        flush=True
    )

    print(
        f"Positive samples: {positive}",
        flush=True
    )

    print(
        f"Scale positive weight: "
        f"{scale_pos_weight:.2f}",
        flush=True
    )

    # --------------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------------

    models[
        "Logistic Regression"
    ] = LogisticRegression(
        class_weight="balanced",
        max_iter=3000,
        C=0.5,
        random_state=RANDOM_STATE
    )

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    models[
        "Random Forest"
    ] = RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=4,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    # --------------------------------------------------------
    # HistGradientBoosting
    # --------------------------------------------------------

    models[
        "HistGradientBoosting"
    ] = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.04,
        max_leaf_nodes=15,
        min_samples_leaf=12,
        l2_regularization=1.0,
        random_state=RANDOM_STATE
    )

    # --------------------------------------------------------
    # XGBoost
    # --------------------------------------------------------

    if XGBOOST_AVAILABLE:

        models[
            "XGBoost"
        ] = XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.03,
            min_child_weight=5,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=2.0,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    else:

        print(
            "\nXGBoost is not installed.",
            flush=True
        )

        print(
            "XGBoost will be skipped.",
            flush=True
        )

    return models


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    probabilities,
    threshold
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0
    )

    missed_escalation_rate = (
        fn / (fn + tp)
        if (fn + tp) > 0
        else 0
    )

    return {

        "accuracy":
            accuracy_score(
                y_true,
                predictions
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                y_true,
                predictions
            ),

        "precision":
            precision_score(
                y_true,
                predictions,
                zero_division=0
            ),

        "recall":
            recall_score(
                y_true,
                predictions,
                zero_division=0
            ),

        "f1":
            f1_score(
                y_true,
                predictions,
                zero_division=0
            ),

        "roc_auc":
            roc_auc_score(
                y_true,
                probabilities
            )
            if len(
                np.unique(y_true)
            ) > 1
            else np.nan,

        "pr_auc":
            average_precision_score(
                y_true,
                probabilities
            )
            if len(
                np.unique(y_true)
            ) > 1
            else np.nan,

        "specificity":
            specificity,

        "false_positive_rate":
            false_positive_rate,

        "missed_escalation_rate":
            missed_escalation_rate,

        "true_positive":
            tp,

        "false_positive":
            fp,

        "true_negative":
            tn,

        "false_negative":
            fn,

        "threshold":
            threshold
    }


# ============================================================
# THRESHOLD OPTIMIZATION
# ============================================================

def optimize_threshold(
    y_true,
    probabilities
):

    print(
        "\n[7] Optimizing threshold...",
        flush=True
    )

    best_threshold = 0.50

    best_score = -np.inf

    best_metrics = None

    thresholds = np.arange(
        0.10,
        0.91,
        0.01
    )

    for threshold in thresholds:

        metrics = calculate_metrics(
            y_true,
            probabilities,
            threshold
        )

        # Balanced objective.
        score = (
            0.60 * metrics["f1"]
            +
            0.40 * metrics[
                "balanced_accuracy"
            ]
        )

        if score > best_score:

            best_score = score

            best_threshold = (
                threshold
            )

            best_metrics = metrics

    print(
        f"Optimal threshold: "
        f"{best_threshold:.2f}",
        flush=True
    )

    print(
        f"Validation F1: "
        f"{best_metrics['f1']:.4f}",
        flush=True
    )

    print(
        f"Validation precision: "
        f"{best_metrics['precision']:.4f}",
        flush=True
    )

    print(
        f"Validation recall: "
        f"{best_metrics['recall']:.4f}",
        flush=True
    )

    print(
        f"Validation balanced accuracy: "
        f"{best_metrics['balanced_accuracy']:.4f}",
        flush=True
    )

    return (
        best_threshold,
        best_metrics
    )


# ============================================================
# TRAIN + COMPARE
# ============================================================

def train_and_compare(
    train,
    validation,
    test,
    feature_columns
):

    print(
        "\n" + "=" * 70,
        flush=True
    )

    print(
        "MODEL TRAINING & COMPARISON",
        flush=True
    )

    print(
        "=" * 70,
        flush=True
    )

    X_train = train[
        feature_columns
    ]

    y_train = train[
        "future_escalation"
    ]

    X_validation = validation[
        feature_columns
    ]

    y_validation = validation[
        "future_escalation"
    ]

    X_test = test[
        feature_columns
    ]

    y_test = test[
        "future_escalation"
    ]

    models = create_models(
        y_train
    )

    results = []

    fitted_models = {}

    validation_probabilities = {}

    test_probabilities = {}

    for name, model in models.items():

        print(
            "\n" + "-" * 60,
            flush=True
        )

        print(
            f"Training: {name}",
            flush=True
        )

        print(
            "-" * 60,
            flush=True
        )

        preprocessor = create_preprocessor(
            feature_columns,
            train
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

        try:

            pipeline.fit(
                X_train,
                y_train
            )

            val_probability = (
                pipeline.predict_proba(
                    X_validation
                )[:, 1]
            )

            test_probability = (
                pipeline.predict_proba(
                    X_test
                )[:, 1]
            )

            metrics = calculate_metrics(
                y_validation,
                val_probability,
                0.50
            )

            print(
                f"Validation F1: "
                f"{metrics['f1']:.4f}",
                flush=True
            )

            print(
                f"Validation Precision: "
                f"{metrics['precision']:.4f}",
                flush=True
            )

            print(
                f"Validation Recall: "
                f"{metrics['recall']:.4f}",
                flush=True
            )

            print(
                f"Validation PR-AUC: "
                f"{metrics['pr_auc']:.4f}",
                flush=True
            )

            results.append(
                {
                    "model":
                        name,

                    "validation_f1":
                        metrics["f1"],

                    "validation_precision":
                        metrics[
                            "precision"
                        ],

                    "validation_recall":
                        metrics[
                            "recall"
                        ],

                    "validation_balanced_accuracy":
                        metrics[
                            "balanced_accuracy"
                        ],

                    "validation_roc_auc":
                        metrics[
                            "roc_auc"
                        ],

                    "validation_pr_auc":
                        metrics[
                            "pr_auc"
                        ]
                }
            )

            fitted_models[
                name
            ] = pipeline

            validation_probabilities[
                name
            ] = val_probability

            test_probabilities[
                name
            ] = test_probability

        except Exception as exc:

            print(
                f"Model failed: {name}",
                flush=True
            )

            print(
                f"Reason: {exc}",
                flush=True
            )

    if not results:

        raise RuntimeError(
            "All ML models failed."
        )

    comparison = pd.DataFrame(
        results
    )

    comparison = comparison.sort_values(
        [
            "validation_f1",
            "validation_pr_auc",
            "validation_balanced_accuracy"
        ],
        ascending=False
    ).reset_index(
        drop=True
    )

    print(
        "\n" + "=" * 70,
        flush=True
    )

    print(
        "MODEL COMPARISON",
        flush=True
    )

    print(
        "=" * 70,
        flush=True
    )

    print(
        comparison.to_string(
            index=False
        ),
        flush=True
    )

    best_model_name = (
        comparison.iloc[0]["model"]
    )

    print(
        "\nBEST MODEL:",
        flush=True
    )

    print(
        best_model_name,
        flush=True
    )

    best_model = fitted_models[
        best_model_name
    ]

    validation_probability = (
        validation_probabilities[
            best_model_name
        ]
    )

    test_probability = (
        test_probabilities[
            best_model_name
        ]
    )

    optimal_threshold, validation_metrics = (
        optimize_threshold(
            y_validation,
            validation_probability
        )
    )

    # --------------------------------------------------------
    # FINAL TEST
    # --------------------------------------------------------

    print(
        "\n[8] Evaluating untouched test data...",
        flush=True
    )

    test_metrics = calculate_metrics(
        y_test,
        test_probability,
        optimal_threshold
    )

    print(
        "\n" + "=" * 70,
        flush=True
    )

    print(
        "FINAL TEST PERFORMANCE",
        flush=True
    )

    print(
        "=" * 70,
        flush=True
    )

    print(
        f"Model                : "
        f"{best_model_name}",
        flush=True
    )

    print(
        f"Threshold            : "
        f"{optimal_threshold:.2f}",
        flush=True
    )

    print(
        f"Accuracy             : "
        f"{test_metrics['accuracy']:.4f}",
        flush=True
    )

    print(
        f"Balanced Accuracy    : "
        f"{test_metrics['balanced_accuracy']:.4f}",
        flush=True
    )

    print(
        f"Precision            : "
        f"{test_metrics['precision']:.4f}",
        flush=True
    )

    print(
        f"Recall               : "
        f"{test_metrics['recall']:.4f}",
        flush=True
    )

    print(
        f"F1                   : "
        f"{test_metrics['f1']:.4f}",
        flush=True
    )

    print(
        f"ROC-AUC              : "
        f"{test_metrics['roc_auc']:.4f}",
        flush=True
    )

    print(
        f"PR-AUC               : "
        f"{test_metrics['pr_auc']:.4f}",
        flush=True
    )

    print(
        f"False Positive Rate  : "
        f"{test_metrics['false_positive_rate']:.4f}",
        flush=True
    )

    print(
        f"Missed Escalation    : "
        f"{test_metrics['missed_escalation_rate']:.4f}",
        flush=True
    )

    print(
        "\nConfusion Matrix:",
        flush=True
    )

    print(
        f"TN={test_metrics['true_negative']} "
        f"FP={test_metrics['false_positive']}",
        flush=True
    )

    print(
        f"FN={test_metrics['false_negative']} "
        f"TP={test_metrics['true_positive']}",
        flush=True
    )

    # --------------------------------------------------------
    # MODEL REPORT
    # --------------------------------------------------------

    report_rows = []

    for _, row in comparison.iterrows():

        is_best = (
            row["model"]
            == best_model_name
        )

        report_rows.append(
            {
                "model":
                    row["model"],

                "validation_f1":
                    row[
                        "validation_f1"
                    ],

                "validation_precision":
                    row[
                        "validation_precision"
                    ],

                "validation_recall":
                    row[
                        "validation_recall"
                    ],

                "validation_balanced_accuracy":
                    row[
                        "validation_balanced_accuracy"
                    ],

                "validation_roc_auc":
                    row[
                        "validation_roc_auc"
                    ],

                "validation_pr_auc":
                    row[
                        "validation_pr_auc"
                    ],

                "selected_model":
                    is_best,

                "test_accuracy":
                    test_metrics[
                        "accuracy"
                    ]
                    if is_best
                    else np.nan,

                "test_balanced_accuracy":
                    test_metrics[
                        "balanced_accuracy"
                    ]
                    if is_best
                    else np.nan,

                "test_precision":
                    test_metrics[
                        "precision"
                    ]
                    if is_best
                    else np.nan,

                "test_recall":
                    test_metrics[
                        "recall"
                    ]
                    if is_best
                    else np.nan,

                "test_f1":
                    test_metrics[
                        "f1"
                    ]
                    if is_best
                    else np.nan,

                "test_roc_auc":
                    test_metrics[
                        "roc_auc"
                    ]
                    if is_best
                    else np.nan,

                "test_pr_auc":
                    test_metrics[
                        "pr_auc"
                    ]
                    if is_best
                    else np.nan,

                "optimal_threshold":
                    optimal_threshold
                    if is_best
                    else np.nan
            }
        )

    model_report = pd.DataFrame(
        report_rows
    )

    return (
        best_model,
        best_model_name,
        optimal_threshold,
        test_metrics,
        model_report
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def extract_feature_importance(
    fitted_pipeline
):

    print(
        "\n[9] Extracting feature importance...",
        flush=True
    )

    try:

        preprocessor = (
            fitted_pipeline
            .named_steps[
                "preprocessor"
            ]
        )

        model = (
            fitted_pipeline
            .named_steps[
                "model"
            ]
        )

        feature_names = (
            preprocessor
            .get_feature_names_out()
        )

        if hasattr(
            model,
            "feature_importances_"
        ):

            importance = (
                model.feature_importances_
            )

        elif hasattr(
            model,
            "coef_"
        ):

            importance = np.abs(
                model.coef_[0]
            )

        else:

            print(
                "Feature importance unavailable.",
                flush=True
            )

            return None

        importance_df = pd.DataFrame(
            {
                "feature":
                    feature_names,

                "importance":
                    importance
            }
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

        total = (
            importance_df[
                "importance"
            ].sum()
        )

        if total > 0:

            importance_df[
                "importance_pct"
            ] = (
                importance_df[
                    "importance"
                ]
                / total
                * 100
            )

        else:

            importance_df[
                "importance_pct"
            ] = 0

        importance_df.to_csv(
            FEATURE_IMPORTANCE_FILE,
            index=False
        )

        print(
            "\nTop 20 features:",
            flush=True
        )

        print(
            importance_df.head(20)
            .to_string(
                index=False
            ),
            flush=True
        )

        return importance_df

    except Exception as exc:

        print(
            f"Feature importance failed: "
            f"{exc}",
            flush=True
        )

        return None


# ============================================================
# PREDICTED RISK SCORE
# ============================================================

def predicted_risk_score(row):

    current = num(
        row["risk_numeric"]
    )

    probability = num(
        row["escalation_probability"]
    )

    # Presentation-oriented forward risk score.
    # Actual ML prediction remains the probability.

    projected = (
        current * 0.60
        +
        probability * 0.40
    )

    return clamp(
        projected
    )


# ============================================================
# ESCALATION LEVEL
# ============================================================

def escalation_level(
    probability,
    threshold
):

    probability = num(
        probability
    )

    if probability >= threshold:

        return "HIGH"

    if probability >= (
        threshold * 0.60
    ):

        return "MEDIUM"

    return "LOW"


# ============================================================
# ESCALATION DIRECTION
# ============================================================

def escalation_direction(row):

    current = num(
        row["risk_numeric"]
    )

    projected = num(
        row["predicted_risk_score"]
    )

    difference = (
        projected - current
    )

    if difference >= 15:

        return "RAPID_ESCALATION"

    if difference >= 5:

        return "ESCALATING"

    if difference <= -15:

        return "RAPIDLY_IMPROVING"

    if difference <= -5:

        return "IMPROVING"

    return "STABLE"


# ============================================================
# EXPLANATION
# ============================================================

def escalation_reason(row):

    reasons = []

    operational_change = num(
        row.get(
            "overall_operational_risk_change_3",
            0
        )
    )

    governance_change = num(
        row.get(
            "governance_score_change_3",
            0
        )
    )

    warning_change = num(
        row.get(
            "early_warning_score_change_3",
            0
        )
    )

    risk_acceleration = num(
        row.get(
            "risk_acceleration",
            0
        )
    )

    probability = num(
        row[
            "escalation_probability"
        ]
    )

    threshold = num(
        row[
            "model_threshold"
        ]
    )

    if operational_change > 5:

        reasons.append(
            "Operational risk rising"
        )

    if governance_change > 5:

        reasons.append(
            "Governance risk increasing"
        )

    if warning_change > 10:

        reasons.append(
            "Early-warning signal increasing"
        )

    if risk_acceleration > 5:

        reasons.append(
            "Risk deterioration accelerating"
        )

    if probability >= threshold:

        reasons.append(
            "ML model indicates elevated "
            "future escalation probability"
        )

    if not reasons:

        return (
            "No dominant escalation driver detected"
        )

    return "; ".join(
        reasons
    )


# ============================================================
# MANAGEMENT ACTION
# ============================================================

def management_action(row):

    level = row[
        "escalation_level"
    ]

    direction = row[
        "escalation_direction"
    ]

    if level == "HIGH":

        return (
            "Prioritize management review, "
            "investigate contributing factors, "
            "and initiate preventive intervention."
        )

    if level == "MEDIUM":

        return (
            "Increase monitoring frequency, "
            "review major risk drivers, "
            "and prepare mitigation measures."
        )

    if direction in [
        "RAPIDLY_IMPROVING",
        "IMPROVING"
    ]:

        return (
            "Continue monitoring recovery "
            "and verify that improvement is sustained."
        )

    return (
        "Continue routine monitoring."
    )


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

def generate_predictions(
    df,
    feature_columns,
    fitted_model,
    threshold,
    best_model_name
):

    print(
        "\n[10] Generating ML predictions...",
        flush=True
    )

    df = df.copy()

    X = df[
        feature_columns
    ]

    probabilities = (
        fitted_model
        .predict_proba(X)[:, 1]
    )

    df[
        "escalation_probability"
    ] = (
        probabilities * 100
    )

    df[
        "model_threshold"
    ] = (
        threshold * 100
    )

    df[
        "predicted_risk_score"
    ] = df.apply(
        predicted_risk_score,
        axis=1
    )

    df[
        "escalation_level"
    ] = df.apply(
        lambda row:
        escalation_level(
            row[
                "escalation_probability"
            ] / 100,
            threshold
        ),
        axis=1
    )

    df[
        "predicted_next_risk_level"
    ] = df[
        "predicted_risk_score"
    ].apply(
        risk_level
    )

    df[
        "escalation_direction"
    ] = df.apply(
        escalation_direction,
        axis=1
    )

    df[
        "escalation_reason"
    ] = df.apply(
        escalation_reason,
        axis=1
    )

    df[
        "escalation_management_action"
    ] = df.apply(
        management_action,
        axis=1
    )

    df[
        "ml_model"
    ] = best_model_name

    df[
        "ml_model_version"
    ] = "Phase 7C Version 4"

    return df


# ============================================================
# MINE SUMMARY
# ============================================================

def create_mine_summary(df):

    print(
        "\n[11] Creating mine-level summary...",
        flush=True
    )

    latest = (
        df.sort_values(
            "date"
        )
        .groupby(
            "subsidiary",
            as_index=False
        )
        .tail(1)
        .copy()
    )

    latest = latest.sort_values(
        [
            "escalation_probability",
            "predicted_risk_score"
        ],
        ascending=False
    ).reset_index(
        drop=True
    )

    latest[
        "escalation_rank"
    ] = np.arange(
        1,
        len(latest) + 1
    )

    return latest


# ============================================================
# DISPLAY
# ============================================================

def display_results(
    latest,
    best_model_name,
    threshold
):

    print(
        "\n" + "=" * 70,
        flush=True
    )

    print(
        "PREDICTIVE RISK ESCALATION STATUS",
        flush=True
    )

    print(
        "=" * 70,
        flush=True
    )

    print(
        f"ML Model  : {best_model_name}",
        flush=True
    )

    print(
        f"Threshold : {threshold:.2f}",
        flush=True
    )

    columns = [
        "escalation_rank",
        "subsidiary",
        "date",
        "production_risk",
        "governance_score",
        "early_warning_score",
        "predicted_risk_score",
        "escalation_probability",
        "escalation_level",
        "predicted_next_risk_level",
        "escalation_direction"
    ]

    available = [
        c for c in columns
        if c in latest.columns
    ]

    print(
        latest[
            available
        ].to_string(
            index=False
        ),
        flush=True
    )

    print(
        "\n" + "=" * 70,
        flush=True
    )

    print(
        "ESCALATION SUMMARY",
        flush=True
    )

    print(
        "=" * 70,
        flush=True
    )

    print(
        latest[
            "escalation_level"
        ].value_counts(),
        flush=True
    )

    print(
        "\nPredicted next risk levels:",
        flush=True
    )

    print(
        latest[
            "predicted_next_risk_level"
        ].value_counts(),
        flush=True
    )

    print(
        "\nEscalation directions:",
        flush=True
    )

    print(
        latest[
            "escalation_direction"
        ].value_counts(),
        flush=True
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n" + "=" * 70,
        flush=True
    )

    print(
        "PHASE 7C VERSION 4",
        flush=True
    )

    print(
        "ADVANCED ML PREDICTIVE RISK ESCALATION ENGINE",
        flush=True
    )

    print(
        "=" * 70,
        flush=True
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 1. Load
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # 2. Clean
    # --------------------------------------------------------

    df = clean_data(
        df
    )

    # --------------------------------------------------------
    # 3. Temporal features
    # --------------------------------------------------------

    df = build_features(
        df
    )

    # --------------------------------------------------------
    # 4. Future target
    # --------------------------------------------------------

    df = build_target(
        df
    )

    # --------------------------------------------------------
    # 5. Feature selection
    # --------------------------------------------------------

    feature_columns = select_features(
        df
    )

    # --------------------------------------------------------
    # 6. Chronological split
    # --------------------------------------------------------

    (
        train,
        validation,
        test
    ) = chronological_split(
        df
    )

    # --------------------------------------------------------
    # 7. Train models
    # --------------------------------------------------------

    (
        best_model,
        best_model_name,
        threshold,
        test_metrics,
        model_report
    ) = train_and_compare(
        train,
        validation,
        test,
        feature_columns
    )

    # --------------------------------------------------------
    # 8. Feature importance
    # --------------------------------------------------------

    extract_feature_importance(
        best_model
    )

    # --------------------------------------------------------
    # 9. Generate predictions
    # --------------------------------------------------------

    prediction_df = generate_predictions(
        df,
        feature_columns,
        best_model,
        threshold,
        best_model_name
    )

    # --------------------------------------------------------
    # 10. Latest mine status
    # --------------------------------------------------------

    latest = create_mine_summary(
        prediction_df
    )

    # --------------------------------------------------------
    # 11. Display
    # --------------------------------------------------------

    display_results(
        latest,
        best_model_name,
        threshold
    )

    # --------------------------------------------------------
    # 12. Save full output
    # --------------------------------------------------------

    print(
        "\n[12] Saving outputs...",
        flush=True
    )

    prediction_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    latest.to_csv(
        SUMMARY_FILE,
        index=False
    )

    model_report.to_csv(
        MODEL_REPORT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # 13. Final files
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70,
        flush=True
    )

    print(
        "FILES CREATED",
        flush=True
    )

    print(
        "=" * 70,
        flush=True
    )

    print(
        f"\nPredictive escalation:"
        f"\n{OUTPUT_FILE}",
        flush=True
    )

    print(
        f"\nMine summary:"
        f"\n{SUMMARY_FILE}",
        flush=True
    )

    print(
        f"\nModel performance:"
        f"\n{MODEL_REPORT_FILE}",
        flush=True
    )

    print(
        f"\nFeature importance:"
        f"\n{FEATURE_IMPORTANCE_FILE}",
        flush=True
    )

    print(
        "\n" + "=" * 70,
        flush=True
    )

    print(
        "PHASE 7C VERSION 4 COMPLETE",
        flush=True
    )

    print(
        "=" * 70,
        flush=True
    )


if __name__ == "__main__":

    main()