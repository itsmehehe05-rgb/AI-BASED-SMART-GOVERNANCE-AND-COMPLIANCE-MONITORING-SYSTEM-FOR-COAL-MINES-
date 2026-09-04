import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor

from data_loader import load_training_data
from preprocessing import preprocess_data
from feature_engineering import create_features


# ============================================================
# CONFIGURATION
# ============================================================

TARGET = "target_next_month_production_mt"

MODEL_DIR = r"D:\CoalMineAI\models"
OUTPUT_DIR = r"D:\CoalMineAI\outputs"


# ============================================================
# FEATURE PREPARATION
# ============================================================

def prepare_features(df):

    print("\nPreparing ML features...")

    data = df.copy()

    # --------------------------------------------------------
    # Make sure date is datetime
    # --------------------------------------------------------

    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    y = pd.to_numeric(
        data[TARGET],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Remove target and metadata
    # --------------------------------------------------------

    X = data.drop(
        columns=[
            TARGET,
            "date",
            "synthetic_flag",
            "source_basis",
            "split"
        ],
        errors="ignore"
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # financial_year is not required because year/month
    # already provide temporal information.
    # --------------------------------------------------------

    X = X.drop(
        columns=[
            "financial_year"
        ],
        errors="ignore"
    )

    # --------------------------------------------------------
    # Convert categorical columns
    # --------------------------------------------------------

    categorical_columns = X.select_dtypes(
        include=[
            "object",
            "string",
            "category"
        ]
    ).columns.tolist()

    if categorical_columns:

        print(
            "\nCategorical columns:"
        )

        for column in categorical_columns:

            print(
                f" - {column}"
            )

        X = pd.get_dummies(
            X,
            columns=categorical_columns,
            drop_first=False
        )

    # --------------------------------------------------------
    # Convert everything to numeric
    # --------------------------------------------------------

    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # --------------------------------------------------------
    # Replace infinite values
    # --------------------------------------------------------

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # --------------------------------------------------------
    # Missing feature values
    # --------------------------------------------------------

    missing_features = X.isnull().sum().sum()

    if missing_features > 0:

        print(
            f"\nMissing feature values: "
            f"{missing_features}"
        )

        X = X.fillna(
            X.median(numeric_only=True)
        )

        X = X.fillna(0)

    else:

        print(
            "\nMissing feature values: 0"
        )

    # --------------------------------------------------------
    # Make sure all column names are strings
    # --------------------------------------------------------

    X.columns = X.columns.astype(str)

    print(
        f"\nFinal feature count: {X.shape[1]}"
    )

    return X, y


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    # --------------------------------------------------------
    # MAPE
    # --------------------------------------------------------

    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)

    non_zero = (
        np.abs(y_true_array) > 1e-8
    )

    if np.any(non_zero):

        mape = np.mean(
            np.abs(
                (
                    y_true_array[non_zero]
                    -
                    y_pred_array[non_zero]
                )
                /
                y_true_array[non_zero]
            )
        ) * 100

    else:

        mape = np.nan

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape
    }


# ============================================================
# MODELS
# ============================================================

def create_models():

    models = {

        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=4,
            min_samples_leaf=3,
            random_state=42
        ),

        "XGBoost": XGBRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=5,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            eval_metric="mae",
            random_state=42,
            n_jobs=-1
        )
    }

    return models


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("PHASE 4 — AI COAL PRODUCTION FORECASTING")
    print("=" * 70)

    # ========================================================
    # 1. LOAD DATA
    # ========================================================

    print("\n[1] Loading dataset...")

    df = load_training_data()

    print(
        f"Loaded rows: {len(df)}"
    )

    # ========================================================
    # 2. PREPROCESS
    # ========================================================

    print("\n[2] Running preprocessing...")

    df = preprocess_data(df)

    # ========================================================
    # 3. FEATURE ENGINEERING
    # ========================================================

    print("\n[3] Creating advanced features...")

    df = create_features(df)

    # ========================================================
    # 4. SORT DATA
    # ========================================================

    df = df.sort_values(
        by="date"
    ).reset_index(
        drop=True
    )

    # ========================================================
    # 5. PREPARE FEATURES
    # ========================================================

    print("\n[4] Preparing feature matrix...")

    X, y = prepare_features(df)

    print(
        f"Rows: {X.shape[0]}"
    )

    print(
        f"Features: {X.shape[1]}"
    )

    # ========================================================
    # 6. DATA SPLIT
    # ========================================================

    print("\n[5] Applying predefined dataset split...")

    train_mask = (
        df["split"]
        .astype(str)
        .str.strip()
        .str.lower()
        == "train"
    )

    validation_mask = (
        df["split"]
        .astype(str)
        .str.strip()
        .str.lower()
        == "validation"
    )

    test_mask = (
        df["split"]
        .astype(str)
        .str.strip()
        .str.lower()
        == "test"
    )

    X_train = X.loc[
        train_mask
    ]

    y_train = y.loc[
        train_mask
    ]

    X_validation = X.loc[
        validation_mask
    ]

    y_validation = y.loc[
        validation_mask
    ]

    X_test = X.loc[
        test_mask
    ]

    y_test = y.loc[
        test_mask
    ]

    print("\nDataset split:")
    print(
        f"Training   : {len(X_train)} rows"
    )
    print(
        f"Validation : {len(X_validation)} rows"
    )
    print(
        f"Test       : {len(X_test)} rows"
    )

    # ========================================================
    # 7. CHECK SPLIT
    # ========================================================

    if len(X_train) == 0:
        raise ValueError(
            "Training dataset is empty."
        )

    if len(X_validation) == 0:
        raise ValueError(
            "Validation dataset is empty."
        )

    if len(X_test) == 0:
        raise ValueError(
            "Test dataset is empty."
        )

    # ========================================================
    # 8. CREATE MODELS
    # ========================================================

    print("\n[6] Creating ML models...")

    models = create_models()

    results = []

    trained_models = {}

    test_predictions = {}

    # ========================================================
    # 9. TRAIN MODELS
    # ========================================================

    for model_name, model in models.items():

        print("\n")
        print("=" * 60)
        print(
            f"TRAINING MODEL: {model_name}"
        )
        print("=" * 60)

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        model.fit(
            X_train,
            y_train
        )

        print(
            "Training completed."
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        validation_pred = model.predict(
            X_validation
        )

        validation_metrics = calculate_metrics(
            y_validation,
            validation_pred
        )

        print("\nValidation performance:")

        print(
            f"MAE  : "
            f"{validation_metrics['MAE']:.4f}"
        )

        print(
            f"RMSE : "
            f"{validation_metrics['RMSE']:.4f}"
        )

        print(
            f"R²   : "
            f"{validation_metrics['R2']:.4f}"
        )

        print(
            f"MAPE : "
            f"{validation_metrics['MAPE']:.2f}%"
        )

        # ----------------------------------------------------
        # Test
        # ----------------------------------------------------

        test_pred = model.predict(
            X_test
        )

        test_metrics = calculate_metrics(
            y_test,
            test_pred
        )

        print("\nTest performance:")

        print(
            f"MAE  : "
            f"{test_metrics['MAE']:.4f}"
        )

        print(
            f"RMSE : "
            f"{test_metrics['RMSE']:.4f}"
        )

        print(
            f"R²   : "
            f"{test_metrics['R2']:.4f}"
        )

        print(
            f"MAPE : "
            f"{test_metrics['MAPE']:.2f}%"
        )

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        results.append({

            "Model":
                model_name,

            "Validation_MAE":
                validation_metrics["MAE"],

            "Validation_RMSE":
                validation_metrics["RMSE"],

            "Validation_R2":
                validation_metrics["R2"],

            "Validation_MAPE":
                validation_metrics["MAPE"],

            "Test_MAE":
                test_metrics["MAE"],

            "Test_RMSE":
                test_metrics["RMSE"],

            "Test_R2":
                test_metrics["R2"],

            "Test_MAPE":
                test_metrics["MAPE"]
        })

        trained_models[
            model_name
        ] = model

        test_predictions[
            model_name
        ] = test_pred

    # ========================================================
    # 10. MODEL COMPARISON
    # ========================================================

    print("\n")
    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    results_df = pd.DataFrame(
        results
    )

    # Lower validation MAE is better
    results_df = results_df.sort_values(
        by="Validation_MAE",
        ascending=True
    ).reset_index(
        drop=True
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    # ========================================================
    # 11. SELECT BEST MODEL
    # ========================================================

    best_model_name = (
        results_df.iloc[0]["Model"]
    )

    best_model = trained_models[
        best_model_name
    ]

    best_test_prediction = test_predictions[
        best_model_name
    ]

    print("\n")
    print("=" * 70)
    print(
        f"BEST MODEL: {best_model_name}"
    )
    print("=" * 70)

    # ========================================================
    # 12. SAVE DIRECTORIES
    # ========================================================

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # ========================================================
    # 13. SAVE MODEL
    # ========================================================

    model_path = os.path.join(
        MODEL_DIR,
        "production_forecaster.pkl"
    )

    model_package = {

        "model":
            best_model,

        "model_name":
            best_model_name,

        "features":
            list(X.columns),

        "target":
            TARGET
    }

    joblib.dump(
        model_package,
        model_path
    )

    print(
        f"\nBest model saved to:"
    )

    print(
        model_path
    )

    # ========================================================
    # 14. SAVE MODEL COMPARISON
    # ========================================================

    comparison_path = os.path.join(
        OUTPUT_DIR,
        "model_comparison.csv"
    )

    results_df.to_csv(
        comparison_path,
        index=False
    )

    print(
        f"\nModel comparison saved to:"
    )

    print(
        comparison_path
    )

    # ========================================================
    # 15. CREATE TEST PREDICTION FILE
    # ========================================================

    prediction_df = df.loc[
        test_mask,
        [
            "date",
            "subsidiary",
            "monthly_production_mt",
            "production_target_mt",
            TARGET
        ]
    ].copy()

    prediction_df[
        "predicted_production_mt"
    ] = best_test_prediction

    # --------------------------------------------------------
    # Prediction error
    # --------------------------------------------------------

    prediction_df[
        "prediction_error_mt"
    ] = (
        prediction_df[TARGET]
        -
        prediction_df[
            "predicted_production_mt"
        ]
    )

    # --------------------------------------------------------
    # Absolute error
    # --------------------------------------------------------

    prediction_df[
        "absolute_error_mt"
    ] = np.abs(
        prediction_df[
            "prediction_error_mt"
        ]
    )

    # --------------------------------------------------------
    # Predicted target achievement
    # --------------------------------------------------------

    prediction_df[
        "predicted_target_achievement_pct"
    ] = (
        prediction_df[
            "predicted_production_mt"
        ]
        /
        prediction_df[
            "production_target_mt"
        ].replace(
            0,
            np.nan
        )
        * 100
    )

    # --------------------------------------------------------
    # Predicted shortfall
    # --------------------------------------------------------

    prediction_df[
        "predicted_shortfall_mt"
    ] = (
        prediction_df[
            "production_target_mt"
        ]
        -
        prediction_df[
            "predicted_production_mt"
        ]
    )

    # --------------------------------------------------------
    # Risk classification
    # --------------------------------------------------------

    prediction_df[
        "production_risk"
    ] = np.select(

        [
            prediction_df[
                "predicted_target_achievement_pct"
            ] >= 100,

            prediction_df[
                "predicted_target_achievement_pct"
            ] >= 95
        ],

        [
            "LOW",

            "MEDIUM"
        ],

        default="HIGH"
    )

    prediction_path = os.path.join(
        OUTPUT_DIR,
        "test_predictions.csv"
    )

    prediction_df.to_csv(
        prediction_path,
        index=False
    )

    print(
        f"\nTest predictions saved to:"
    )

    print(
        prediction_path
    )

    # ========================================================
    # 16. FINAL SUMMARY
    # ========================================================

    print("\n")
    print("=" * 70)
    print("PHASE 4 COMPLETE")
    print("=" * 70)

    print(
        f"Best model       : {best_model_name}"
    )

    print(
        f"Training rows    : {len(X_train)}"
    )

    print(
        f"Validation rows  : {len(X_validation)}"
    )

    print(
        f"Test rows        : {len(X_test)}"
    )

    print(
        f"Features used    : {X.shape[1]}"
    )

    print("\nOutput files:")

    print(
        f"1. {model_path}"
    )

    print(
        f"2. {comparison_path}"
    )

    print(
        f"3. {prediction_path}"
    )

    print("\n")
    print("=" * 70)
    print("READY FOR PHASE 5 — AI RISK ENGINE")
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()