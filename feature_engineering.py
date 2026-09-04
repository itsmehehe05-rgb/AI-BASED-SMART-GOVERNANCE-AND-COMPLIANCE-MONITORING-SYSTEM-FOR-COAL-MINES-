import pandas as pd
import numpy as np

from data_loader import load_training_data
from preprocessing import preprocess_data


TARGET = "target_next_month_production_mt"


def create_features(df):

    print("\n" + "=" * 60)
    print("PHASE 3 — ADVANCED FEATURE ENGINEERING")
    print("=" * 60)

    df = df.copy()

    # =====================================================
    # 1. DATE FEATURES
    # =====================================================

    print("\n[1] Creating temporal features...")

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df["year"] = df["date"].dt.year

    df["month"] = df["date"].dt.month

    df["quarter"] = df["date"].dt.quarter

    # Cyclic representation of month
    df["month_sin"] = np.sin(
        2 * np.pi * df["month"] / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * df["month"] / 12
    )

    print("Temporal features created.")

    # =====================================================
    # 2. PRODUCTION MOMENTUM
    # =====================================================

    print("\n[2] Creating production momentum features...")

    if "monthly_production_mt" in df.columns:

        # Difference from previous month
        df["production_change_1m_mt"] = (
            df.groupby("subsidiary")
            ["monthly_production_mt"]
            .diff(1)
        )

        # 3-month production momentum
        df["production_momentum_3m"] = (
            df.groupby("subsidiary")
            ["monthly_production_mt"]
            .transform(
                lambda x:
                x.rolling(
                    3,
                    min_periods=1
                ).mean()
            )
        )

        # 6-month production momentum
        df["production_momentum_6m"] = (
            df.groupby("subsidiary")
            ["monthly_production_mt"]
            .transform(
                lambda x:
                x.rolling(
                    6,
                    min_periods=1
                ).mean()
            )
        )

        # Difference between current production
        # and 3-month average
        df["production_vs_3m_average_pct"] = (
            (
                df["monthly_production_mt"]
                -
                df["production_momentum_3m"]
            )
            /
            df["production_momentum_3m"].replace(
                0,
                np.nan
            )
            * 100
        )

    print("Production momentum features created.")

    # =====================================================
    # 3. EQUIPMENT STRESS
    # =====================================================

    print("\n[3] Creating equipment risk features...")

    if {
        "equipment_availability_pct",
        "equipment_utilization_pct",
        "equipment_breakdown_hours"
    }.issubset(df.columns):

        # Availability risk
        df["equipment_availability_risk"] = (
            100
            -
            df["equipment_availability_pct"]
        )

        # Utilization stress
        df["equipment_utilization_stress"] = (
            df["equipment_utilization_pct"]
            /
            df["equipment_availability_pct"].replace(
                0,
                np.nan
            )
        )

        # Combined equipment stress
        df["equipment_stress_index"] = (
            (
                100
                -
                df["equipment_availability_pct"]
            )
            *
            0.4
            +
            df["equipment_breakdown_hours"]
            *
            0.4
            +
            (
                100
                -
                df["equipment_utilization_pct"]
            )
            *
            0.2
        )

    print("Equipment features created.")

    # =====================================================
    # 4. LOGISTICS STRESS
    # =====================================================

    print("\n[4] Creating logistics risk features...")

    if {
        "rail_rake_availability_pct",
        "evacuation_delay_hours"
    }.issubset(df.columns):

        df["rail_availability_risk"] = (
            100
            -
            df["rail_rake_availability_pct"]
        )

        df["logistics_stress_index"] = (
            (
                100
                -
                df["rail_rake_availability_pct"]
            )
            *
            0.6
            +
            df["evacuation_delay_hours"]
            *
            0.4
        )

    print("Logistics features created.")

    # =====================================================
    # 5. WORKFORCE EFFICIENCY
    # =====================================================

    print("\n[5] Creating workforce features...")

    if {
        "monthly_production_mt",
        "workforce_strength"
    }.issubset(df.columns):

        df["production_per_worker"] = (
            df["monthly_production_mt"]
            /
            df["workforce_strength"].replace(
                0,
                np.nan
            )
        )

    if {
        "employees_trained",
        "workforce_strength"
    }.issubset(df.columns):

        df["training_coverage_pct"] = (
            df["employees_trained"]
            /
            df["workforce_strength"].replace(
                0,
                np.nan
            )
            * 100
        )

    print("Workforce features created.")

    # =====================================================
    # 6. TARGET PRESSURE
    # =====================================================

    print("\n[6] Creating target pressure features...")

    if {
        "production_target_mt",
        "monthly_production_mt"
    }.issubset(df.columns):

        df["target_gap_mt"] = (
            df["production_target_mt"]
            -
            df["monthly_production_mt"]
        )

        df["target_gap_pct"] = (
            (
                df["production_target_mt"]
                -
                df["monthly_production_mt"]
            )
            /
            df["production_target_mt"].replace(
                0,
                np.nan
            )
            * 100
        )

        df["above_target_flag"] = (
            df["monthly_production_mt"]
            >=
            df["production_target_mt"]
        ).astype(int)

    print("Target pressure features created.")

    # =====================================================
    # 7. OFFTAKE / STOCK PRESSURE
    # =====================================================

    print("\n[7] Creating stock and offtake features...")

    if {
        "closing_stock_mt",
        "monthly_offtake_mt"
    }.issubset(df.columns):

        df["stock_to_offtake_ratio"] = (
            df["closing_stock_mt"]
            /
            df["monthly_offtake_mt"].replace(
                0,
                np.nan
            )
        )

    if {
        "monthly_production_mt",
        "monthly_offtake_mt"
    }.issubset(df.columns):

        df["production_offtake_balance_mt"] = (
            df["monthly_production_mt"]
            -
            df["monthly_offtake_mt"]
        )

    print("Stock/offtake features created.")

    # =====================================================
    # 8. WEATHER STRESS
    # =====================================================

    print("\n[8] Creating weather stress features...")

    if {
        "rainfall_mm",
        "rainy_days"
    }.issubset(df.columns):

        df["rainfall_intensity"] = (
            df["rainfall_mm"]
            /
            df["rainy_days"].replace(
                0,
                np.nan
            )
        )

    if {
        "rainfall_mm",
        "rainy_days"
    }.issubset(df.columns):

        df["weather_stress_index"] = (
            df["rainfall_mm"] * 0.7
            +
            df["rainy_days"] * 0.3
        )

    print("Weather features created.")

    # =====================================================
    # 9. DATA QUALITY CLEANUP
    # =====================================================

    print("\n[9] Cleaning generated features...")

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Don't fill target
    feature_columns = [
        column
        for column in df.columns
        if column != TARGET
    ]

    numeric_columns = df[
        feature_columns
    ].select_dtypes(
        include=np.number
    ).columns

    for column in numeric_columns:

        if df[column].isnull().any():

            df[column] = df[column].fillna(
                df[column].median()
            )

    # =====================================================
    # 10. FEATURE SUMMARY
    # =====================================================

    original_columns = 37

    new_columns = (
        len(df.columns)
        -
        original_columns
    )

    print(
        f"\nOriginal columns : {original_columns}"
    )

    print(
        f"Final columns    : {len(df.columns)}"
    )

    print(
        f"New features     : {new_columns}"
    )

    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 60)

    return df


if __name__ == "__main__":

    df = load_training_data()

    df = preprocess_data(df)

    df = create_features(df)

    output_path = (
        "D:/CoalMineAI/outputs/"
        "engineered_coal_features.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nEngineered dataset saved to:"
    )

    print(output_path)