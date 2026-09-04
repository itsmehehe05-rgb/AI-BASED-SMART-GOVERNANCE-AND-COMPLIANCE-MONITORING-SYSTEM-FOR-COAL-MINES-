from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PHASE 11A
# PRODUCTION FORECAST VALIDATION
# ============================================================

BASE_DIR = Path(r"D:\CoalMineAI")
OUTPUT_DIR = BASE_DIR / "outputs"

INPUT_FILE = (
    OUTPUT_DIR /
    "production_risk_analysis.csv"
)

OVERALL_OUTPUT = (
    OUTPUT_DIR /
    "production_forecast_validation.csv"
)

MINE_OUTPUT = (
    OUTPUT_DIR /
    "production_forecast_validation_by_mine.csv"
)

TIME_OUTPUT = (
    OUTPUT_DIR /
    "production_forecast_validation_by_period.csv"
)


# ============================================================
# HELPERS
# ============================================================

def safe_mape(actual, predicted):

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mask = actual != 0

    if mask.sum() == 0:
        return np.nan

    return np.mean(
        np.abs(
            (
                actual[mask]
                -
                predicted[mask]
            )
            /
            actual[mask]
        )
    ) * 100


def safe_smape(actual, predicted):

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    denominator = (
        np.abs(actual)
        +
        np.abs(predicted)
    )

    mask = denominator != 0

    if mask.sum() == 0:
        return np.nan

    return np.mean(
        2
        *
        np.abs(
            predicted[mask]
            -
            actual[mask]
        )
        /
        denominator[mask]
    ) * 100


def calculate_metrics(group):

    actual = group[
        "monthly_production_mt"
    ].astype(float)

    predicted = group[
        "predicted_production_mt"
    ].astype(float)

    error = (
        predicted
        -
        actual
    )

    absolute_error = np.abs(
        error
    )

    squared_error = (
        error ** 2
    )

    mae = absolute_error.mean()

    rmse = np.sqrt(
        squared_error.mean()
    )

    mape = safe_mape(
        actual,
        predicted
    )

    smape = safe_smape(
        actual,
        predicted
    )

    bias = error.mean()

    # --------------------------------------------------------
    # R²
    # --------------------------------------------------------

    denominator = (

        (
            actual
            -
            actual.mean()
        )
        ** 2

    ).sum()

    numerator = squared_error.sum()

    if denominator == 0:

        r2 = np.nan

    else:

        r2 = 1 - (
            numerator
            /
            denominator
        )

    # --------------------------------------------------------
    # Forecast direction accuracy
    # --------------------------------------------------------

    actual_direction = (
        actual.diff()
        .fillna(0)
        > 0
    )

    predicted_direction = (
        predicted.diff()
        .fillna(0)
        > 0
    )

    direction_accuracy = (
        actual_direction
        ==
        predicted_direction
    ).mean() * 100

    # --------------------------------------------------------
    # Within-error accuracy
    # --------------------------------------------------------

    if actual.abs().mean() != 0:

        relative_error = (
            absolute_error
            /
            actual.abs()
        )

        within_10 = (
            relative_error <= 0.10
        ).mean() * 100

        within_20 = (
            relative_error <= 0.20
        ).mean() * 100

    else:

        within_10 = np.nan
        within_20 = np.nan

    return {

        "records":
            len(group),

        "actual_mean_mt":
            actual.mean(),

        "predicted_mean_mt":
            predicted.mean(),

        "mae_mt":
            mae,

        "rmse_mt":
            rmse,

        "mape_pct":
            mape,

        "smape_pct":
            smape,

        "r2":
            r2,

        "bias_mt":
            bias,

        "direction_accuracy_pct":
            direction_accuracy,

        "within_10pct_pct":
            within_10,

        "within_20pct_pct":
            within_20

    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "PHASE 11A — PRODUCTION FORECAST VALIDATION"
    )

    print("=" * 70)

    # ========================================================
    # [1] LOAD DATA
    # ========================================================

    print(
        "\n[1] Loading production risk analysis..."
    )

    if not INPUT_FILE.exists():

        print(
            "ERROR: Input file not found:"
        )

        print(
            INPUT_FILE
        )

        return

    df = pd.read_csv(
        INPUT_FILE,
        encoding="utf-8-sig"
    )

    print(
        f"Rows loaded: {len(df)}"
    )

    # ========================================================
    # [2] VALIDATE
    # ========================================================

    print(
        "\n[2] Validating forecast fields..."
    )

    required = [

        "date",
        "subsidiary",
        "monthly_production_mt",
        "predicted_production_mt"

    ]

    missing = [

        c
        for c in required
        if c not in df.columns

    ]

    if missing:

        print(
            "ERROR: Missing columns:"
        )

        for c in missing:

            print(
                " -",
                c
            )

        return

    print(
        "Validation successful."
    )

    # ========================================================
    # [3] CLEAN
    # ========================================================

    print(
        "\n[3] Cleaning forecast data..."
    )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df["monthly_production_mt"] = pd.to_numeric(
        df["monthly_production_mt"],
        errors="coerce"
    )

    df["predicted_production_mt"] = pd.to_numeric(
        df["predicted_production_mt"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "date",
            "subsidiary",
            "monthly_production_mt",
            "predicted_production_mt"
        ]
    )

    df = df.sort_values(
        [
            "subsidiary",
            "date"
        ]
    )

    print(
        f"Clean forecast records: {len(df)}"
    )

    # ========================================================
    # [4] CALCULATE ERROR
    # ========================================================

    print(
        "\n[4] Calculating forecast errors..."
    )

    df[
        "forecast_error_mt"
    ] = (

        df[
            "predicted_production_mt"
        ]
        -
        df[
            "monthly_production_mt"
        ]

    )

    df[
        "absolute_error_mt"
    ] = np.abs(
        df[
            "forecast_error_mt"
        ]
    )

    df[
        "absolute_percentage_error"
    ] = np.where(

        df[
            "monthly_production_mt"
        ] != 0,

        df[
            "absolute_error_mt"
        ]
        /
        np.abs(
            df[
                "monthly_production_mt"
            ]
        )
        * 100,

        np.nan

    )

    # ========================================================
    # [5] OVERALL METRICS
    # ========================================================

    print(
        "\n[5] Calculating overall forecast performance..."
    )

    overall = calculate_metrics(
        df
    )

    overall_df = pd.DataFrame(
        [
            overall
        ]
    )

    overall_df.insert(
        0,
        "scope",
        "OVERALL"
    )

    overall_df.to_csv(
        OVERALL_OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # [6] MINE METRICS
    # ========================================================

    print(
        "\n[6] Calculating performance by subsidiary..."
    )

    mine_rows = []

    for subsidiary, group in df.groupby(
        "subsidiary"
    ):

        metrics = calculate_metrics(
            group
        )

        metrics[
            "subsidiary"
        ] = subsidiary

        mine_rows.append(
            metrics
        )

    mine_df = pd.DataFrame(
        mine_rows
    )

    mine_df = mine_df[
        [
            "subsidiary",
            "records",
            "actual_mean_mt",
            "predicted_mean_mt",
            "mae_mt",
            "rmse_mt",
            "mape_pct",
            "smape_pct",
            "r2",
            "bias_mt",
            "direction_accuracy_pct",
            "within_10pct_pct",
            "within_20pct_pct"
        ]
    ]

    mine_df = mine_df.sort_values(
        "mae_mt"
    )

    mine_df.to_csv(
        MINE_OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # [7] TIME PERIOD VALIDATION
    # ========================================================

    print(
        "\n[7] Calculating performance by time period..."
    )

    df[
        "year"
    ] = df[
        "date"
    ].dt.year

    period_rows = []

    for year, group in df.groupby(
        "year"
    ):

        metrics = calculate_metrics(
            group
        )

        metrics[
            "year"
        ] = year

        period_rows.append(
            metrics
        )

    period_df = pd.DataFrame(
        period_rows
    )

    period_df = period_df[
        [
            "year",
            "records",
            "actual_mean_mt",
            "predicted_mean_mt",
            "mae_mt",
            "rmse_mt",
            "mape_pct",
            "smape_pct",
            "r2",
            "bias_mt",
            "direction_accuracy_pct",
            "within_10pct_pct",
            "within_20pct_pct"
        ]
    ]

    period_df.to_csv(
        TIME_OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # [8] ERROR DISTRIBUTION
    # ========================================================

    print(
        "\n[8] Analysing error distribution..."
    )

    error_mean = df[
        "forecast_error_mt"
    ].mean()

    error_std = df[
        "forecast_error_mt"
    ].std()

    median_absolute_error = df[
        "absolute_error_mt"
    ].median()

    p90_absolute_error = df[
        "absolute_error_mt"
    ].quantile(
        0.90
    )

    # ========================================================
    # [9] PRINT RESULTS
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "OVERALL FORECAST PERFORMANCE"
    )

    print(
        "=" * 70
    )

    print(
        f"Records                  : "
        f"{overall['records']}"
    )

    print(
        f"Actual mean production   : "
        f"{overall['actual_mean_mt']:.2f} MT"
    )

    print(
        f"Predicted mean production: "
        f"{overall['predicted_mean_mt']:.2f} MT"
    )

    print(
        f"MAE                      : "
        f"{overall['mae_mt']:.2f} MT"
    )

    print(
        f"RMSE                     : "
        f"{overall['rmse_mt']:.2f} MT"
    )

    print(
        f"MAPE                     : "
        f"{overall['mape_pct']:.2f}%"
    )

    print(
        f"SMAPE                    : "
        f"{overall['smape_pct']:.2f}%"
    )

    print(
        f"R²                       : "
        f"{overall['r2']:.4f}"
    )

    print(
        f"Bias                     : "
        f"{overall['bias_mt']:.2f} MT"
    )

    print(
        f"Direction accuracy       : "
        f"{overall['direction_accuracy_pct']:.2f}%"
    )

    print(
        f"Within ±10%              : "
        f"{overall['within_10pct_pct']:.2f}%"
    )

    print(
        f"Within ±20%              : "
        f"{overall['within_20pct_pct']:.2f}%"
    )

    print(
        "\nError statistics:"
    )

    print(
        f"Mean error               : "
        f"{error_mean:.2f} MT"
    )

    print(
        f"Error standard deviation : "
        f"{error_std:.2f} MT"
    )

    print(
        f"Median absolute error    : "
        f"{median_absolute_error:.2f} MT"
    )

    print(
        f"90th percentile error    : "
        f"{p90_absolute_error:.2f} MT"
    )

    # ========================================================
    # [10] MINE RESULTS
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "FORECAST PERFORMANCE BY MINE"
    )

    print(
        "=" * 70
    )

    print(
        mine_df.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.2f}"
        )
    )

    # ========================================================
    # [11] BEST / WORST
    # ========================================================

    best = mine_df.iloc[
        mine_df[
            "mae_mt"
        ].argmin()
    ]

    worst = mine_df.iloc[
        mine_df[
            "mae_mt"
        ].argmax()
    ]

    print(
        "\nBest forecast MAE:"
    )

    print(
        f"{best['subsidiary']} — "
        f"{best['mae_mt']:.2f} MT"
    )

    print(
        "\nHighest forecast MAE:"
    )

    print(
        f"{worst['subsidiary']} — "
        f"{worst['mae_mt']:.2f} MT"
    )

    # ========================================================
    # [12] SAVE SUMMARY
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "PHASE 11A COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nOverall validation:"
    )

    print(
        OVERALL_OUTPUT
    )

    print(
        "\nMine-level validation:"
    )

    print(
        MINE_OUTPUT
    )

    print(
        "\nTime-period validation:"
    )

    print(
        TIME_OUTPUT
    )


if __name__ == "__main__":
    main()