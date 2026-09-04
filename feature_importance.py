import os
import joblib
import pandas as pd
import numpy as np

from data_loader import load_training_data
from preprocessing import preprocess_data
from feature_engineering import create_features


TARGET = "target_next_month_production_mt"

MODEL_PATH = (
    r"D:\CoalMineAI\models\production_forecaster.pkl"
)

OUTPUT_PATH = (
    r"D:\CoalMineAI\outputs\feature_importance.csv"
)


def prepare_features(df):

    data = df.copy()

    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce"
    )

    X = data.drop(
        columns=[
            TARGET,
            "date",
            "synthetic_flag",
            "source_basis",
            "split",
            "financial_year"
        ],
        errors="ignore"
    )

    categorical_columns = X.select_dtypes(
        include=[
            "object",
            "string",
            "category"
        ]
    ).columns.tolist()

    if categorical_columns:

        X = pd.get_dummies(
            X,
            columns=categorical_columns,
            drop_first=False
        )

    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(
        X.median(numeric_only=True)
    )

    X = X.fillna(0)

    X.columns = X.columns.astype(str)

    return X


def main():

    print("\n" + "=" * 60)
    print("PHASE 5A — FEATURE IMPORTANCE")
    print("=" * 60)

    # Load model
    package = joblib.load(
        MODEL_PATH
    )

    model = package["model"]

    model_features = package[
        "features"
    ]

    print(
        f"\nLoaded model: "
        f"{package['model_name']}"
    )

    # Load dataset
    df = load_training_data()

    df = preprocess_data(df)

    df = create_features(df)

    X = prepare_features(df)

    # Make sure feature order matches training
    X = X.reindex(
        columns=model_features,
        fill_value=0
    )

    # Feature importance
    importance = model.feature_importances_

    importance_df = pd.DataFrame({

        "feature": X.columns,

        "importance": importance
    })

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False
    )

    importance_df[
        "importance_pct"
    ] = (
        importance_df["importance"]
        /
        importance_df["importance"].sum()
        * 100
    )

    print("\n" + "=" * 60)
    print("TOP 20 MODEL DRIVERS")
    print("=" * 60)

    print(
        importance_df.head(20).to_string(
            index=False
        )
    )

    # Save
    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True
    )

    importance_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        f"\nFeature importance saved to:"
    )

    print(OUTPUT_PATH)

    print("\n" + "=" * 60)
    print("PHASE 5A COMPLETE")
    print("=" * 60)


if __name__ == "__main__":

    main()