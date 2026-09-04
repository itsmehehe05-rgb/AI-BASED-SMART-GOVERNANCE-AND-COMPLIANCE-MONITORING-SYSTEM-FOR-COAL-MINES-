import pandas as pd
import os


DATA_PATH = r"D:\CoalMineAI\data\CIL_Coal_Production_Forecasting_Dataset.xlsx"

SHEET_NAME = "Forecasting_Training_Data"


def load_training_data():

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    print("\nLoading dataset...")
    print(f"File: {DATA_PATH}")

    # Read sheet without assuming where the header is
    raw = pd.read_excel(
        DATA_PATH,
        sheet_name=SHEET_NAME,
        engine="openpyxl",
        header=None
    )

    # ---------------------------------------------
    # Find the actual header row
    # ---------------------------------------------

    header_row = None

    for i in range(len(raw)):

        row_values = (
            raw.iloc[i]
            .astype(str)
            .str.strip()
            .str.lower()
            .tolist()
        )

        if "date" in row_values and "subsidiary" in row_values:
            header_row = i
            break

    if header_row is None:

        raise ValueError(
            "Could not automatically find the header row."
        )

    print(f"\nActual header found at Excel row: {header_row + 1}")

    # ---------------------------------------------
    # Set the actual header
    # ---------------------------------------------

    df = raw.iloc[header_row + 1:].copy()

    df.columns = raw.iloc[header_row].tolist()

    # ---------------------------------------------
    # Remove completely empty rows/columns
    # ---------------------------------------------

    df = df.dropna(
        axis=0,
        how="all"
    )

    df = df.dropna(
        axis=1,
        how="all"
    )

    # ---------------------------------------------
    # Reset index
    # ---------------------------------------------

    df = df.reset_index(
        drop=True
    )

    return df


if __name__ == "__main__":

    df = load_training_data()

    print("\n" + "=" * 60)
    print("CIL COAL PRODUCTION DATASET - INSPECTION")
    print("=" * 60)

    # ---------------------------------------------
    # Dataset size
    # ---------------------------------------------

    print("\n[1] DATASET SIZE")
    print("-" * 40)

    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    # ---------------------------------------------
    # Columns
    # ---------------------------------------------

    print("\n[2] COLUMN NAMES")
    print("-" * 40)

    for i, column in enumerate(df.columns, start=1):

        print(f"{i:2}. {column}")

    # ---------------------------------------------
    # Data types
    # ---------------------------------------------

    print("\n[3] DATA TYPES")
    print("-" * 40)

    for column in df.columns:

        print(
            f"{column}: {df[column].dtype}"
        )

    # ---------------------------------------------
    # First 5 rows
    # ---------------------------------------------

    print("\n[4] FIRST 5 DATA ROWS")
    print("-" * 40)

    print(
        df.head(5).to_string(
            index=False
        )
    )

    # ---------------------------------------------
    # Missing values
    # ---------------------------------------------

    print("\n[5] MISSING VALUES")
    print("-" * 40)

    missing = df.isnull().sum()

    missing = missing[
        missing > 0
    ]

    if len(missing) == 0:

        print("No missing values.")

    else:

        print(missing)

    # ---------------------------------------------
    # Duplicate rows
    # ---------------------------------------------

    print("\n[6] DUPLICATE ROWS")
    print("-" * 40)

    print(
        f"Duplicate rows: {df.duplicated().sum()}"
    )

    # ---------------------------------------------
    # Date information
    # ---------------------------------------------

    if "date" in df.columns:

        print("\n[7] DATE INFORMATION")
        print("-" * 40)

        dates = pd.to_datetime(
            df["date"],
            errors="coerce"
        )

        print(
            f"Minimum date: {dates.min()}"
        )

        print(
            f"Maximum date: {dates.max()}"
        )

    # ---------------------------------------------
    # Subsidiaries
    # ---------------------------------------------

    if "subsidiary" in df.columns:

        print("\n[8] SUBSIDIARIES")
        print("-" * 40)

        subsidiaries = (
            df["subsidiary"]
            .dropna()
            .unique()
        )

        for subsidiary in subsidiaries:

            print(
                f"- {subsidiary}"
            )

    # ---------------------------------------------
    # Train / Validation / Test
    # ---------------------------------------------

    if "split" in df.columns:

        print("\n[9] DATA SPLIT")
        print("-" * 40)

        print(
            df["split"]
            .value_counts(dropna=False)
            .to_string()
        )

    # ---------------------------------------------
    # Target
    # ---------------------------------------------

    target = (
        "target_next_month_production_mt"
    )

    if target in df.columns:

        print("\n[10] FORECASTING TARGET")
        print("-" * 40)

        print(
            f"Target column: {target}"
        )

        target_values = pd.to_numeric(
            df[target],
            errors="coerce"
        )

        print(
            f"Minimum : {target_values.min()}"
        )

        print(
            f"Maximum : {target_values.max()}"
        )

        print(
            f"Mean    : {target_values.mean():.4f}"
        )

        print(
            f"Missing : {target_values.isnull().sum()}"
        )

    print("\n" + "=" * 60)
    print("DATASET INSPECTION COMPLETE")
    print("=" * 60)