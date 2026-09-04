import pandas as pd
import numpy as np
import os

from data_loader import load_training_data


TARGET = "target_next_month_production_mt"


def preprocess_data(df):

    print("\n" + "=" * 60)
    print("PHASE 2 — DATA PREPROCESSING")
    print("=" * 60)

    df = df.copy()

    # =====================================================
    # 1. CLEAN COLUMN NAMES
    # =====================================================

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    print("\n[1] Column names cleaned.")

    # =====================================================
    # 2. CHECK REQUIRED COLUMNS
    # =====================================================

    required_columns = [
        "date",
        "subsidiary",
        "monthly_production_mt",
        "production_target_mt",
        TARGET,
        "split"
    ]

    missing_required = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_required:

        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(missing_required)
        )

    print("[2] Required columns verified.")

    # =====================================================
    # 3. DATE CONVERSION
    # =====================================================

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    invalid_dates = df["date"].isna().sum()

    print(
        f"[3] Invalid dates: {invalid_dates}"
    )

    if invalid_dates > 0:

        raise ValueError(
            "Invalid dates found in dataset."
        )

    # =====================================================
    # 4. SORT CHRONOLOGICALLY
    # =====================================================

    df = df.sort_values(
        by=["date", "subsidiary"]
    ).reset_index(
        drop=True
    )

    print("[4] Dataset sorted chronologically.")

    # =====================================================
    # 5. CONVERT NUMERIC COLUMNS
    # =====================================================

    numeric_columns = [
        "year",
        "month",
        "quarter",
        "monthly_production_mt",
        "monthly_offtake_mt",
        "production_target_mt",
        "workforce_strength",
        "rainfall_mm",
        "rainy_days",
        "avg_temperature_c",
        "equipment_availability_pct",
        "equipment_utilization_pct",
        "equipment_breakdown_hours",
        "rail_rake_availability_pct",
        "evacuation_delay_hours",
        "fmc_availability_pct",
        "closing_stock_mt",
        "coking_share_pct",
        "non_coking_share_pct",
        "avg_gcv_kcal_kg",
        "avg_ash_pct",
        "employees_trained",
        "training_man_days",
        "production_lag_1_mt",
        "production_lag_12_mt",
        "production_rolling_mean_3_mt",
        "production_growth_1m_pct",
        "offtake_growth_1m_pct",
        "production_per_employee_kt",
        "target_achievement_pct",
        TARGET
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    print("[5] Numeric columns converted.")

    # =====================================================
    # 6. MISSING VALUE CHECK
    # =====================================================

    missing = df.isnull().sum()

    missing = missing[
        missing > 0
    ]

    print("\n[6] Missing values:")

    if len(missing) == 0:

        print("    No missing values.")

    else:

        print(missing)

    # =====================================================
    # 7. DUPLICATE CHECK
    # =====================================================

    duplicate_count = df.duplicated().sum()

    print(
        f"\n[7] Duplicate rows: {duplicate_count}"
    )

    if duplicate_count > 0:

        df = df.drop_duplicates()

        print(
            "    Duplicate rows removed."
        )

    # =====================================================
    # 8. TARGET VALIDATION
    # =====================================================

    print("\n[8] TARGET VALIDATION")

    print(
        f"Target: {TARGET}"
    )

    print(
        f"Minimum: {df[TARGET].min():.4f}"
    )

    print(
        f"Maximum: {df[TARGET].max():.4f}"
    )

    print(
        f"Mean: {df[TARGET].mean():.4f}"
    )

    if (
        df[TARGET] < 0
    ).any():

        raise ValueError(
            "Negative target production detected."
        )

    # =====================================================
    # 9. CHECK PRODUCTION VALUES
    # =====================================================

    print("\n[9] PRODUCTION VALIDATION")

    production_columns = [
        "monthly_production_mt",
        "monthly_offtake_mt",
        "production_target_mt"
    ]

    for column in production_columns:

        if column in df.columns:

            negative = (
                df[column] < 0
            ).sum()

            print(
                f"{column}: "
                f"{negative} negative values"
            )

    # =====================================================
    # 10. PERCENTAGE VALIDATION
    # =====================================================

    print("\n[10] PERCENTAGE VALIDATION")

    percentage_columns = [
        "equipment_availability_pct",
        "equipment_utilization_pct",
        "rail_rake_availability_pct",
        "fmc_availability_pct",
        "coking_share_pct",
        "non_coking_share_pct"
    ]

    for column in percentage_columns:

        if column in df.columns:

            invalid = (
                (df[column] < 0)
                |
                (df[column] > 100)
            ).sum()

            print(
                f"{column}: "
                f"{invalid} invalid values"
            )

    # =====================================================
    # 11. SPLIT VALIDATION
    # =====================================================

    print("\n[11] DATA SPLIT")

    print(
        df["split"]
        .value_counts()
        .to_string()
    )

    valid_splits = {
        "Train",
        "Validation",
        "Test"
    }

    actual_splits = set(
        df["split"]
        .dropna()
        .unique()
    )

    unexpected = (
        actual_splits
        -
        valid_splits
    )

    if unexpected:

        print(
            f"WARNING: Unexpected split values: "
            f"{unexpected}"
        )

    # =====================================================
    # 12. SUBSIDIARY VALIDATION
    # =====================================================

    print("\n[12] SUBSIDIARIES")

    subsidiaries = (
        df["subsidiary"]
        .dropna()
        .unique()
    )

    print(
        f"Number of subsidiaries: "
        f"{len(subsidiaries)}"
    )

    for subsidiary in subsidiaries:

        print(
            f" - {subsidiary}"
        )

    # =====================================================
    # 13. TARGET LEAKAGE CHECK
    # =====================================================

    print("\n[13] BASIC TARGET LEAKAGE CHECK")

    suspicious_columns = []

    for column in df.columns:

        if column == TARGET:
            continue

        column_lower = column.lower()

        if (
            "next_month" in column_lower
            or "future" in column_lower
            or "target_next" in column_lower
        ):

            suspicious_columns.append(
                column
            )

    if suspicious_columns:

        print(
            "Potential future-information columns:"
        )

        for column in suspicious_columns:

            print(
                f" - {column}"
            )

    else:

        print(
            "No obvious future-information "
            "columns detected."
        )

    # =====================================================
    # 14. METADATA CHECK
    # =====================================================

    print("\n[14] METADATA COLUMNS")

    metadata_columns = [
        "synthetic_flag",
        "source_basis",
        "split"
    ]

    for column in metadata_columns:

        if column in df.columns:

            print(
                f" - {column}"
            )

    # =====================================================
    # 15. FINAL SUMMARY
    # =====================================================

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)

    print(
        f"Rows: {df.shape[0]}"
    )

    print(
        f"Columns: {df.shape[1]}"
    )

    print(
        f"Target: {TARGET}"
    )

    return df


if __name__ == "__main__":

    df = load_training_data()

    clean_df = preprocess_data(df)

    output_path = (
        "D:/CoalMineAI/outputs/"
        "cleaned_coal_production_data.csv"
    )

    clean_df.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nCleaned dataset saved to:"
    )

    print(output_path)