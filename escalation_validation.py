from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)


# ============================================================
# PHASE 11C
# PREDICTIVE ESCALATION MODEL VALIDATION
# & THRESHOLD OPTIMIZATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"

INPUT_FILE = OUTPUT_DIR / "predictive_risk_escalation.csv"

FULL_OUTPUT = OUTPUT_DIR / "escalation_validation.csv"
MINE_OUTPUT = OUTPUT_DIR / "escalation_validation_by_mine.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "escalation_validation_summary.csv"
THRESHOLD_OUTPUT = OUTPUT_DIR / "escalation_threshold_analysis.csv"
CONFUSION_OUTPUT = OUTPUT_DIR / "escalation_confusion_matrix.csv"


# ============================================================
# RISK MAPPING
# ============================================================

RISK_MAP = {
    "LOW": 0,
    "MEDIUM": 1,
    "HIGH": 2
}


def clean_risk(value):

    value = str(value).strip().upper()

    if value in RISK_MAP:
        return value

    return np.nan


def safe_divide(a, b):

    if b == 0:
        return 0.0

    return a / b


# ============================================================
# METRICS
# ============================================================

def binary_metrics(y_true, y_pred):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    balanced_accuracy = balanced_accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    false_alarm_rate = safe_divide(
        fp,
        fp + tn
    )

    missed_escalation_rate = safe_divide(
        fn,
        fn + tp
    )

    return {
        "accuracy_pct": accuracy * 100,
        "balanced_accuracy_pct": balanced_accuracy * 100,
        "precision_pct": precision * 100,
        "recall_pct": recall * 100,
        "f1_pct": f1 * 100,
        "false_alarm_rate_pct": false_alarm_rate * 100,
        "missed_escalation_rate_pct": missed_escalation_rate * 100,
        "true_positive": int(tp),
        "false_positive": int(fp),
        "true_negative": int(tn),
        "false_negative": int(fn)
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "PHASE 11C — PREDICTIVE ESCALATION MODEL VALIDATION "
        "& THRESHOLD OPTIMIZATION"
    )
    print("=" * 70)

    # ========================================================
    # 1. LOAD DATA
    # ========================================================

    print("\n[1] Loading predictive escalation data...")

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Required input file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Rows loaded: {len(df)}"
    )

    # ========================================================
    # 2. VALIDATE
    # ========================================================

    print("\n[2] Validating required fields...")

    required = [
        "date",
        "subsidiary",
        "overall_risk_level",
        "predicted_next_risk_level",
        "escalation_probability"
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(missing)
        )

    print(
        "Input validation successful."
    )

    # ========================================================
    # 3. CLEAN
    # ========================================================

    print("\n[3] Cleaning data...")

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df["subsidiary"] = (
        df["subsidiary"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["overall_risk_level"] = (
        df["overall_risk_level"]
        .apply(clean_risk)
    )

    df["predicted_next_risk_level"] = (
        df["predicted_next_risk_level"]
        .apply(clean_risk)
    )

    df["escalation_probability"] = pd.to_numeric(
        df["escalation_probability"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "date",
            "subsidiary",
            "overall_risk_level",
            "escalation_probability"
        ]
    )

    df = df.sort_values(
        ["subsidiary", "date"]
    ).reset_index(drop=True)

    print(
        f"Clean records: {len(df)}"
    )

    # ========================================================
    # 4. CREATE ACTUAL NEXT-MONTH OUTCOME
    # ========================================================

    print(
        "\n[4] Creating genuine next-month escalation target..."
    )

    df["next_date"] = (
        df.groupby("subsidiary")["date"]
        .shift(-1)
    )

    df["actual_next_risk_level"] = (
        df.groupby("subsidiary")["overall_risk_level"]
        .shift(-1)
    )

    expected_next_date = (
        df["date"]
        + pd.offsets.MonthBegin(1)
    )

    df["continuous_month"] = (
        df["next_date"] == expected_next_date
    )

    df["current_risk_numeric"] = (
        df["overall_risk_level"]
        .map(RISK_MAP)
    )

    df["actual_next_risk_numeric"] = (
        df["actual_next_risk_level"]
        .map(RISK_MAP)
    )

    df["actual_risk_change"] = (
        df["actual_next_risk_numeric"]
        -
        df["current_risk_numeric"]
    )

    df["actual_escalation"] = (
        df["actual_risk_change"] > 0
    ).astype(int)

    df = df[
        df["continuous_month"]
    ].copy()

    print(
        f"Continuous backtest records: {len(df)}"
    )

    print(
        f"Actual escalation events: "
        f"{df['actual_escalation'].sum()}"
    )

    # ========================================================
    # 5. CREATE CHRONOLOGICAL SPLIT
    # ========================================================

    print(
        "\n[5] Applying chronological validation split..."
    )

    train_mask = (
        df["date"]
        < pd.Timestamp("2023-07-01")
    )

    validation_mask = (
        (df["date"] >= pd.Timestamp("2023-07-01"))
        &
        (df["date"] < pd.Timestamp("2025-01-01"))
    )

    test_mask = (
        df["date"]
        >= pd.Timestamp("2025-01-01")
    )

    train = df[
        train_mask
    ].copy()

    validation = df[
        validation_mask
    ].copy()

    test = df[
        test_mask
    ].copy()

    print(
        f"Training records   : {len(train)}"
    )

    print(
        f"Validation records : {len(validation)}"
    )

    print(
        f"Test records       : {len(test)}"
    )

    # ========================================================
    # 6. PREPARE PROBABILITY
    # ========================================================

    print(
        "\n[6] Preparing escalation probabilities..."
    )

    for part in [train, validation, test]:

        part["escalation_probability"] = (
            part["escalation_probability"]
            .clip(0, 100)
        )

        part["probability"] = (
            part["escalation_probability"]
            / 100
        )

    # ========================================================
    # 7. THRESHOLD SEARCH
    # ========================================================

    print(
        "\n[7] Optimizing escalation threshold "
        "using validation data only..."
    )

    threshold_rows = []

    thresholds = np.arange(
        0.10,
        0.91,
        0.01
    )

    for threshold in thresholds:

        y_true = validation[
            "actual_escalation"
        ]

        y_probability = validation[
            "probability"
        ]

        y_pred = (
            y_probability >= threshold
        ).astype(int)

        metrics = binary_metrics(
            y_true,
            y_pred
        )

        threshold_rows.append({
            "threshold": threshold,
            **metrics
        })

    threshold_df = pd.DataFrame(
        threshold_rows
    )

    # ========================================================
    # 8. SELECT BEST THRESHOLD
    # ========================================================

    print(
        "\n[8] Selecting optimal threshold..."
    )

    # Primary objective:
    # maximize F1
    #
    # Secondary objective:
    # prefer lower false-alarm rate

    threshold_df = threshold_df.sort_values(
        [
            "f1_pct",
            "balanced_accuracy_pct",
            "precision_pct"
        ],
        ascending=False
    ).reset_index(drop=True)

    best_threshold = float(
        threshold_df.iloc[0]["threshold"]
    )

    print(
        f"Optimal validation threshold: "
        f"{best_threshold:.2f}"
    )

    # ========================================================
    # 9. EVALUATE VALIDATION
    # ========================================================

    print(
        "\n[9] Evaluating validation performance..."
    )

    validation["optimized_prediction"] = (
        validation["probability"]
        >= best_threshold
    ).astype(int)

    validation_metrics = binary_metrics(
        validation["actual_escalation"],
        validation["optimized_prediction"]
    )

    # ========================================================
    # 10. EVALUATE TEST
    # ========================================================

    print(
        "\n[10] Evaluating untouched test period..."
    )

    test["optimized_prediction"] = (
        test["probability"]
        >= best_threshold
    ).astype(int)

    test_metrics = binary_metrics(
        test["actual_escalation"],
        test["optimized_prediction"]
    )

    # ========================================================
    # 11. CURRENT MODEL PERFORMANCE
    # ========================================================

    print(
        "\n[11] Comparing current Phase 7C threshold..."
    )

    # Phase 7C effectively treats any predicted
    # next risk higher than current risk as escalation.

    test["current_model_prediction"] = (
        test["predicted_next_risk_level"]
        .map(RISK_MAP)
        >
        test["current_risk_numeric"]
    ).astype(int)

    current_metrics = binary_metrics(
        test["actual_escalation"],
        test["current_model_prediction"]
    )

    # ========================================================
    # 12. PROBABILITY QUALITY
    # ========================================================

    print(
        "\n[12] Evaluating probability quality..."
    )

    try:

        if test["actual_escalation"].nunique() == 2:

            roc_auc = roc_auc_score(
                test["actual_escalation"],
                test["probability"]
            )

            pr_auc = average_precision_score(
                test["actual_escalation"],
                test["probability"]
            )

        else:

            roc_auc = np.nan
            pr_auc = np.nan

    except Exception:

        roc_auc = np.nan
        pr_auc = np.nan

    # ========================================================
    # 13. MINE-LEVEL PERFORMANCE
    # ========================================================

    print(
        "\n[13] Calculating performance by subsidiary..."
    )

    mine_rows = []

    for mine, group in test.groupby(
        "subsidiary"
    ):

        metrics = binary_metrics(
            group["actual_escalation"],
            group["optimized_prediction"]
        )

        current = binary_metrics(
            group["actual_escalation"],
            group["current_model_prediction"]
        )

        mine_rows.append({

            "subsidiary": mine,

            "records": len(group),

            "actual_escalations":
                int(group["actual_escalation"].sum()),

            "optimized_predicted_escalations":
                int(group["optimized_prediction"].sum()),

            "current_predicted_escalations":
                int(group["current_model_prediction"].sum()),

            "optimized_precision_pct":
                metrics["precision_pct"],

            "optimized_recall_pct":
                metrics["recall_pct"],

            "optimized_f1_pct":
                metrics["f1_pct"],

            "optimized_balanced_accuracy_pct":
                metrics["balanced_accuracy_pct"],

            "optimized_false_alarm_rate_pct":
                metrics["false_alarm_rate_pct"],

            "optimized_missed_escalation_rate_pct":
                metrics["missed_escalation_rate_pct"],

            "current_precision_pct":
                current["precision_pct"],

            "current_recall_pct":
                current["recall_pct"],

            "current_f1_pct":
                current["f1_pct"]

        })

    mine_df = pd.DataFrame(
        mine_rows
    )

    # ========================================================
    # 14. CONFUSION MATRIX
    # ========================================================

    print(
        "\n[14] Creating test confusion matrix..."
    )

    cm = confusion_matrix(
        test["actual_escalation"],
        test["optimized_prediction"],
        labels=[0, 1]
    )

    confusion_df = pd.DataFrame(
        cm,
        index=[
            "Actual_No_Escalation",
            "Actual_Escalation"
        ],
        columns=[
            "Predicted_No_Escalation",
            "Predicted_Escalation"
        ]
    )

    # ========================================================
    # 15. BUILD FULL VALIDATION DATASET
    # ========================================================

    print(
        "\n[15] Creating full escalation validation dataset..."
    )

    train["evaluation_split"] = "TRAIN"
    validation["evaluation_split"] = "VALIDATION"
    test["evaluation_split"] = "TEST"

    full = pd.concat(
        [
            train,
            validation,
            test
        ],
        ignore_index=True
    )

    full["optimized_threshold"] = (
        best_threshold
    )

    # ========================================================
    # 16. SUMMARY DATASET
    # ========================================================

    summary = pd.DataFrame([

        {
            "dataset": "VALIDATION",
            "records": len(validation),
            "actual_escalations":
                int(validation["actual_escalation"].sum()),
            "predicted_escalations":
                int(validation["optimized_prediction"].sum()),
            "accuracy_pct":
                validation_metrics["accuracy_pct"],
            "balanced_accuracy_pct":
                validation_metrics["balanced_accuracy_pct"],
            "precision_pct":
                validation_metrics["precision_pct"],
            "recall_pct":
                validation_metrics["recall_pct"],
            "f1_pct":
                validation_metrics["f1_pct"],
            "false_alarm_rate_pct":
                validation_metrics["false_alarm_rate_pct"],
            "missed_escalation_rate_pct":
                validation_metrics["missed_escalation_rate_pct"],
            "threshold":
                best_threshold
        },

        {
            "dataset": "TEST_OPTIMIZED",
            "records": len(test),
            "actual_escalations":
                int(test["actual_escalation"].sum()),
            "predicted_escalations":
                int(test["optimized_prediction"].sum()),
            "accuracy_pct":
                test_metrics["accuracy_pct"],
            "balanced_accuracy_pct":
                test_metrics["balanced_accuracy_pct"],
            "precision_pct":
                test_metrics["precision_pct"],
            "recall_pct":
                test_metrics["recall_pct"],
            "f1_pct":
                test_metrics["f1_pct"],
            "false_alarm_rate_pct":
                test_metrics["false_alarm_rate_pct"],
            "missed_escalation_rate_pct":
                test_metrics["missed_escalation_rate_pct"],
            "threshold":
                best_threshold
        },

        {
            "dataset": "TEST_CURRENT_PHASE_7C",
            "records": len(test),
            "actual_escalations":
                int(test["actual_escalation"].sum()),
            "predicted_escalations":
                int(test["current_model_prediction"].sum()),
            "accuracy_pct":
                current_metrics["accuracy_pct"],
            "balanced_accuracy_pct":
                current_metrics["balanced_accuracy_pct"],
            "precision_pct":
                current_metrics["precision_pct"],
            "recall_pct":
                current_metrics["recall_pct"],
            "f1_pct":
                current_metrics["f1_pct"],
            "false_alarm_rate_pct":
                current_metrics["false_alarm_rate_pct"],
            "missed_escalation_rate_pct":
                current_metrics["missed_escalation_rate_pct"],
            "threshold":
                np.nan
        }

    ])

    # ========================================================
    # 17. SAVE
    # ========================================================

    print(
        "\n[16] Saving validation outputs..."
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    full.to_csv(
        FULL_OUTPUT,
        index=False
    )

    mine_df.to_csv(
        MINE_OUTPUT,
        index=False
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False
    )

    threshold_df.to_csv(
        THRESHOLD_OUTPUT,
        index=False
    )

    confusion_df.to_csv(
        CONFUSION_OUTPUT
    )

    # ========================================================
    # 18. PRINT RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print(
        "PHASE 11C — PREDICTIVE ESCALATION VALIDATION RESULTS"
    )
    print("=" * 70)

    print(
        f"\nOptimal threshold      : "
        f"{best_threshold:.2f}"
    )

    print("\nVALIDATION PERFORMANCE")

    print(
        f"Accuracy               : "
        f"{validation_metrics['accuracy_pct']:.2f}%"
    )

    print(
        f"Balanced accuracy      : "
        f"{validation_metrics['balanced_accuracy_pct']:.2f}%"
    )

    print(
        f"Precision              : "
        f"{validation_metrics['precision_pct']:.2f}%"
    )

    print(
        f"Recall                 : "
        f"{validation_metrics['recall_pct']:.2f}%"
    )

    print(
        f"F1                     : "
        f"{validation_metrics['f1_pct']:.2f}%"
    )

    print("\nUNTOUCHED TEST PERFORMANCE")

    print(
        f"Accuracy               : "
        f"{test_metrics['accuracy_pct']:.2f}%"
    )

    print(
        f"Balanced accuracy      : "
        f"{test_metrics['balanced_accuracy_pct']:.2f}%"
    )

    print(
        f"Precision              : "
        f"{test_metrics['precision_pct']:.2f}%"
    )

    print(
        f"Recall                 : "
        f"{test_metrics['recall_pct']:.2f}%"
    )

    print(
        f"F1                     : "
        f"{test_metrics['f1_pct']:.2f}%"
    )

    print(
        f"False alarm rate       : "
        f"{test_metrics['false_alarm_rate_pct']:.2f}%"
    )

    print(
        f"Missed escalation rate : "
        f"{test_metrics['missed_escalation_rate_pct']:.2f}%"
    )

    print("\nPROBABILITY QUALITY")

    if pd.notna(roc_auc):

        print(
            f"ROC-AUC                : "
            f"{roc_auc:.4f}"
        )

        print(
            f"PR-AUC                 : "
            f"{pr_auc:.4f}"
        )

    else:

        print(
            "ROC-AUC / PR-AUC       : "
            "Not calculable"
        )

    print("\nCURRENT PHASE 7C vs OPTIMIZED")

    print(
        f"Current F1             : "
        f"{current_metrics['f1_pct']:.2f}%"
    )

    print(
        f"Optimized F1           : "
        f"{test_metrics['f1_pct']:.2f}%"
    )

    print(
        f"Current precision      : "
        f"{current_metrics['precision_pct']:.2f}%"
    )

    print(
        f"Optimized precision    : "
        f"{test_metrics['precision_pct']:.2f}%"
    )

    print(
        f"Current recall         : "
        f"{current_metrics['recall_pct']:.2f}%"
    )

    print(
        f"Optimized recall       : "
        f"{test_metrics['recall_pct']:.2f}%"
    )

    print("\nTEST CONFUSION MATRIX")

    print(
        confusion_df.to_string()
    )

    print("\n" + "=" * 70)
    print("PHASE 11C COMPLETE")
    print("=" * 70)

    print("\nFull validation:")
    print(FULL_OUTPUT)

    print("\nMine-level validation:")
    print(MINE_OUTPUT)

    print("\nValidation summary:")
    print(SUMMARY_OUTPUT)

    print("\nThreshold analysis:")
    print(THRESHOLD_OUTPUT)

    print("\nConfusion matrix:")
    print(CONFUSION_OUTPUT)


if __name__ == "__main__":
    main()