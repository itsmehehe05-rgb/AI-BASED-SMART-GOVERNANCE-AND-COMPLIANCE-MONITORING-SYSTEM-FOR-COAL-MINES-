from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# PHASE 11B — FUTURE RISK & EARLY-WARNING BACKTESTING
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"

INPUT_FILE = OUTPUT_DIR / "predictive_risk_escalation.csv"

OUTPUT_FILE = OUTPUT_DIR / "risk_backtesting.csv"
MINE_OUTPUT_FILE = OUTPUT_DIR / "risk_backtesting_by_mine.csv"
TRANSITION_OUTPUT_FILE = OUTPUT_DIR / "risk_transition_matrix.csv"


# ============================================================
# UTILITIES
# ============================================================

RISK_ORDER = {
    "LOW": 0,
    "MEDIUM": 1,
    "HIGH": 2,
    "UNRELIABLE": np.nan
}


def clean_risk(value):
    value = str(value).strip().upper()

    if value in ["LOW", "MEDIUM", "HIGH"]:
        return value

    return np.nan


def safe_divide(a, b):
    if b == 0 or pd.isna(b):
        return 0.0

    return float(a / b)


def calculate_classification_metrics(actual, predicted):

    actual = pd.Series(actual).reset_index(drop=True)
    predicted = pd.Series(predicted).reset_index(drop=True)

    valid = actual.notna() & predicted.notna()

    actual = actual[valid]
    predicted = predicted[valid]

    if len(actual) == 0:
        return {
            "accuracy_pct": 0.0,
            "balanced_accuracy_pct": 0.0,
            "precision_pct": 0.0,
            "recall_pct": 0.0,
            "f1_pct": 0.0
        }

    labels = ["LOW", "MEDIUM", "HIGH"]

    recalls = []
    precisions = []

    for label in labels:

        tp = ((actual == label) & (predicted == label)).sum()
        fp = ((actual != label) & (predicted == label)).sum()
        fn = ((actual == label) & (predicted != label)).sum()

        recall = safe_divide(tp, tp + fn)
        precision = safe_divide(tp, tp + fp)

        recalls.append(recall)
        precisions.append(precision)

    accuracy = (actual == predicted).mean()

    balanced_accuracy = np.mean(recalls)

    precision_macro = np.mean(precisions)

    f1_values = []

    for precision, recall in zip(precisions, recalls):

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        f1_values.append(f1)

    f1_macro = np.mean(f1_values)

    return {
        "accuracy_pct": accuracy * 100,
        "balanced_accuracy_pct": balanced_accuracy * 100,
        "precision_pct": precision_macro * 100,
        "recall_pct": balanced_accuracy * 100,
        "f1_pct": f1_macro * 100
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PHASE 11B — FUTURE RISK & EARLY-WARNING BACKTESTING")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------------

    print("\n[1] Loading predictive risk escalation data...")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Required input not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"Rows loaded: {len(df)}")

    # --------------------------------------------------------
    # 2. VALIDATE REQUIRED COLUMNS
    # --------------------------------------------------------

    print("\n[2] Validating required fields...")

    required_columns = [
        "date",
        "subsidiary",
        "overall_risk_level",
        "predicted_next_risk_level",
        "escalation_probability",
        "warning_level",
        "trajectory",
        "early_warning_score",
        "predicted_risk_score",
        "escalation_direction"
    ]

    missing = [
        c for c in required_columns
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(missing)
        )

    print("Input validation successful.")

    # --------------------------------------------------------
    # 3. CLEAN DATA
    # --------------------------------------------------------

    print("\n[3] Cleaning risk data...")

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

    df["early_warning_score"] = pd.to_numeric(
        df["early_warning_score"],
        errors="coerce"
    )

    df["predicted_risk_score"] = pd.to_numeric(
        df["predicted_risk_score"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "date",
            "subsidiary",
            "overall_risk_level"
        ]
    )

    df = df.sort_values(
        ["subsidiary", "date"]
    ).reset_index(drop=True)

    print(f"Clean records: {len(df)}")

    # --------------------------------------------------------
    # 4. CREATE FUTURE ACTUAL RISK
    # --------------------------------------------------------

    print("\n[4] Creating next-month actual risk outcomes...")

    df["next_date"] = (
        df.groupby("subsidiary")["date"]
        .shift(-1)
    )

    df["actual_next_risk_level"] = (
        df.groupby("subsidiary")["overall_risk_level"]
        .shift(-1)
    )

    # --------------------------------------------------------
    # 5. VERIFY MONTHLY CONTINUITY
    # --------------------------------------------------------

    print("\n[5] Checking monthly continuity...")

    expected_next_date = (
        df["date"]
        + pd.offsets.MonthBegin(1)
    )

    df["future_observation_available"] = (
        df["next_date"] == expected_next_date
    )

    continuous = df[
        df["future_observation_available"]
    ].copy()

    print(
        f"Continuous month-to-month records: "
        f"{len(continuous)}"
    )

    # --------------------------------------------------------
    # 6. CREATE FUTURE ESCALATION TARGET
    # --------------------------------------------------------

    print("\n[6] Creating future escalation target...")

    continuous["current_risk_numeric"] = (
        continuous["overall_risk_level"]
        .map(RISK_ORDER)
    )

    continuous["actual_next_risk_numeric"] = (
        continuous["actual_next_risk_level"]
        .map(RISK_ORDER)
    )

    continuous["actual_risk_change"] = (
        continuous["actual_next_risk_numeric"]
        -
        continuous["current_risk_numeric"]
    )

    continuous["actual_escalation"] = (
        continuous["actual_risk_change"] > 0
    ).astype(int)

    continuous["actual_deterioration"] = (
        continuous["actual_risk_change"] > 0
    ).astype(int)

    continuous["actual_improvement"] = (
        continuous["actual_risk_change"] < 0
    ).astype(int)

    continuous["actual_stable"] = (
        continuous["actual_risk_change"] == 0
    ).astype(int)

    # --------------------------------------------------------
    # 7. PREDICTED ESCALATION
    # --------------------------------------------------------

    print("\n[7] Evaluating predicted escalation...")

    continuous["predicted_escalation"] = (
        continuous["predicted_next_risk_level"].map(
            RISK_ORDER
        )
        >
        continuous["current_risk_numeric"]
    ).astype(int)

    continuous["correct_risk_prediction"] = (
        continuous["predicted_next_risk_level"]
        ==
        continuous["actual_next_risk_level"]
    ).astype(int)

    # --------------------------------------------------------
    # 8. EARLY WARNING SIGNAL
    # --------------------------------------------------------

    print("\n[8] Evaluating early-warning signals...")

    warning_positive_values = [
        "WATCH",
        "WARNING",
        "HIGH",
        "CRITICAL",
        "ESCALATING"
    ]

    continuous["early_warning_prediction"] = (
        continuous["warning_level"]
        .astype(str)
        .str.upper()
        .str.strip()
        .isin(warning_positive_values)
        |
        continuous["trajectory"]
        .astype(str)
        .str.upper()
        .str.strip()
        .eq("ESCALATING")
    ).astype(int)

    # --------------------------------------------------------
    # 9. EARLY-WARNING CONFUSION MATRIX
    # --------------------------------------------------------

    print("\n[9] Calculating early-warning performance...")

    ew_actual = continuous["actual_escalation"]
    ew_pred = continuous["early_warning_prediction"]

    tp = ((ew_actual == 1) & (ew_pred == 1)).sum()
    fp = ((ew_actual == 0) & (ew_pred == 1)).sum()
    tn = ((ew_actual == 0) & (ew_pred == 0)).sum()
    fn = ((ew_actual == 1) & (ew_pred == 0)).sum()

    ew_precision = safe_divide(tp, tp + fp)
    ew_recall = safe_divide(tp, tp + fn)

    if ew_precision + ew_recall == 0:
        ew_f1 = 0.0
    else:
        ew_f1 = (
            2 * ew_precision * ew_recall
            /
            (ew_precision + ew_recall)
        )

    ew_accuracy = safe_divide(
        tp + tn,
        tp + tn + fp + fn
    )

    false_alarm_rate = safe_divide(
        fp,
        fp + tn
    )

    missed_escalation_rate = safe_divide(
        fn,
        fn + tp
    )

    # --------------------------------------------------------
    # 10. RISK CLASSIFICATION PERFORMANCE
    # --------------------------------------------------------

    print("\n[10] Calculating next-risk classification performance...")

    risk_metrics = calculate_classification_metrics(
        continuous["actual_next_risk_level"],
        continuous["predicted_next_risk_level"]
    )

    # --------------------------------------------------------
    # 11. ESCALATION PROBABILITY PERFORMANCE
    # --------------------------------------------------------

    print("\n[11] Evaluating escalation probabilities...")

    continuous["probability_prediction"] = (
        continuous["escalation_probability"]
        .clip(0, 100)
        / 100
    )

    continuous["probability_bucket"] = pd.cut(
        continuous["probability_prediction"],
        bins=[
            -0.01,
            0.20,
            0.40,
            0.60,
            0.80,
            1.01
        ],
        labels=[
            "0-20%",
            "20-40%",
            "40-60%",
            "60-80%",
            "80-100%"
        ]
    )

    # --------------------------------------------------------
    # 12. CREATE BACKTEST RESULT
    # --------------------------------------------------------

    print("\n[12] Creating backtesting dataset...")

    result_columns = [
        "date",
        "next_date",
        "subsidiary",
        "overall_risk_level",
        "actual_next_risk_level",
        "predicted_next_risk_level",
        "actual_risk_change",
        "actual_escalation",
        "predicted_escalation",
        "correct_risk_prediction",
        "early_warning_prediction",
        "early_warning_score",
        "warning_level",
        "trajectory",
        "escalation_probability",
        "predicted_risk_score",
        "escalation_direction",
        "probability_bucket"
    ]

    result = continuous[result_columns].copy()

    # --------------------------------------------------------
    # 13. MINE-LEVEL PERFORMANCE
    # --------------------------------------------------------

    print("\n[13] Calculating performance by subsidiary...")

    mine_results = []

    for mine, group in result.groupby("subsidiary"):

        actual = group["actual_next_risk_level"]
        predicted = group["predicted_next_risk_level"]

        metrics = calculate_classification_metrics(
            actual,
            predicted
        )

        ew_actual_m = group["actual_escalation"]
        ew_pred_m = group["early_warning_prediction"]

        tp_m = (
            (ew_actual_m == 1)
            &
            (ew_pred_m == 1)
        ).sum()

        fp_m = (
            (ew_actual_m == 0)
            &
            (ew_pred_m == 1)
        ).sum()

        tn_m = (
            (ew_actual_m == 0)
            &
            (ew_pred_m == 0)
        ).sum()

        fn_m = (
            (ew_actual_m == 1)
            &
            (ew_pred_m == 0)
        ).sum()

        precision_m = safe_divide(
            tp_m,
            tp_m + fp_m
        )

        recall_m = safe_divide(
            tp_m,
            tp_m + fn_m
        )

        if precision_m + recall_m == 0:
            f1_m = 0.0
        else:
            f1_m = (
                2 * precision_m * recall_m
                /
                (precision_m + recall_m)
            )

        mine_results.append({
            "subsidiary": mine,
            "records": len(group),
            "actual_escalations": int(
                ew_actual_m.sum()
            ),
            "predicted_escalations": int(
                ew_pred_m.sum()
            ),
            "risk_prediction_accuracy_pct":
                metrics["accuracy_pct"],
            "risk_balanced_accuracy_pct":
                metrics["balanced_accuracy_pct"],
            "risk_precision_pct":
                metrics["precision_pct"],
            "risk_recall_pct":
                metrics["recall_pct"],
            "risk_f1_pct":
                metrics["f1_pct"],
            "early_warning_precision_pct":
                precision_m * 100,
            "early_warning_recall_pct":
                recall_m * 100,
            "early_warning_f1_pct":
                f1_m * 100,
            "false_alarm_rate_pct":
                safe_divide(
                    fp_m,
                    fp_m + tn_m
                ) * 100,
            "missed_escalation_rate_pct":
                safe_divide(
                    fn_m,
                    fn_m + tp_m
                ) * 100
        })

    mine_df = pd.DataFrame(mine_results)

    # --------------------------------------------------------
    # 14. RISK TRANSITION MATRIX
    # --------------------------------------------------------

    print("\n[14] Creating risk transition matrix...")

    transition_matrix = pd.crosstab(
        continuous["overall_risk_level"],
        continuous["actual_next_risk_level"]
    )

    transition_matrix = transition_matrix.reindex(
        index=["LOW", "MEDIUM", "HIGH"],
        columns=["LOW", "MEDIUM", "HIGH"],
        fill_value=0
    )

    # --------------------------------------------------------
    # 15. SAVE OUTPUTS
    # --------------------------------------------------------

    print("\n[15] Saving backtesting outputs...")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    mine_df.to_csv(
        MINE_OUTPUT_FILE,
        index=False
    )

    transition_matrix.to_csv(
        TRANSITION_OUTPUT_FILE
    )

    # --------------------------------------------------------
    # 16. SUMMARY
    # --------------------------------------------------------

    total = len(result)

    print("\n" + "=" * 70)
    print("PHASE 11B — FUTURE RISK BACKTESTING RESULTS")
    print("=" * 70)

    print(f"\nBacktest records        : {total}")

    print(
        f"Actual escalations     : "
        f"{int(result['actual_escalation'].sum())}"
    )

    print(
        f"Predicted escalations  : "
        f"{int(result['predicted_escalation'].sum())}"
    )

    print("\nNext-risk prediction:")
    print(
        f"Accuracy               : "
        f"{risk_metrics['accuracy_pct']:.2f}%"
    )

    print(
        f"Balanced accuracy      : "
        f"{risk_metrics['balanced_accuracy_pct']:.2f}%"
    )

    print(
        f"Precision              : "
        f"{risk_metrics['precision_pct']:.2f}%"
    )

    print(
        f"Recall                 : "
        f"{risk_metrics['recall_pct']:.2f}%"
    )

    print(
        f"F1 score               : "
        f"{risk_metrics['f1_pct']:.2f}%"
    )

    print("\nEarly-warning performance:")

    print(
        f"Accuracy               : "
        f"{ew_accuracy * 100:.2f}%"
    )

    print(
        f"Precision              : "
        f"{ew_precision * 100:.2f}%"
    )

    print(
        f"Recall                 : "
        f"{ew_recall * 100:.2f}%"
    )

    print(
        f"F1 score               : "
        f"{ew_f1 * 100:.2f}%"
    )

    print(
        f"False alarm rate       : "
        f"{false_alarm_rate * 100:.2f}%"
    )

    print(
        f"Missed escalation rate : "
        f"{missed_escalation_rate * 100:.2f}%"
    )

    print("\nActual next-month risk distribution:")

    print(
        result["actual_next_risk_level"]
        .value_counts()
        .to_string()
    )

    print("\nRisk transition matrix:")

    print(
        transition_matrix.to_string()
    )

    print("\n" + "=" * 70)
    print("PHASE 11B COMPLETE")
    print("=" * 70)

    print("\nBacktesting dataset:")
    print(OUTPUT_FILE)

    print("\nMine-level validation:")
    print(MINE_OUTPUT_FILE)

    print("\nRisk transition matrix:")
    print(TRANSITION_OUTPUT_FILE)


if __name__ == "__main__":
    main()