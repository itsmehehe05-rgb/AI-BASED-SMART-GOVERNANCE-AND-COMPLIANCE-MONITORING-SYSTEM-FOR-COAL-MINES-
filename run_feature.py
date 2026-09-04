"""
CoalMineAI
ONE-COMMAND RUNNER
AI-Powered Statutory Compliance & Governance Intelligence

Run:
    python run_feature.py

This orchestrator executes the completed pipeline stages
in sequence.

IMPORTANT:
    - It does not contain the ML algorithms itself.
    - It calls the existing tested scripts.
    - It stops immediately if a stage fails.
    - It verifies important output artifacts.
    - Conversational AI is intentionally excluded.
"""

from pathlib import Path
import subprocess
import sys
import time


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

BASE_DIR = Path(r"D:\CoalMineAI")
SRC_DIR = BASE_DIR / "src"
FINAL_DIR = BASE_DIR / "outputs" / "v5" / "FINAL"


# ============================================================
# PIPELINE STAGES
# ============================================================

# IMPORTANT:
# Keep only scripts that are actually FINAL/APPROVED.
#
# The first stages below are placeholders for your existing
# completed pipeline scripts. Once we identify their exact
# filenames, put them here.

STAGES = [
    # --------------------------------------------------------
    # Existing core ML pipeline
    # --------------------------------------------------------
    # ("STEP 4.9 - Leakage-safe dataset",
    #  "v5_04_09_leakage_safe_feature_engineering.py"),

    # ("STEP 4.10 - Feature selection",
    #  "v5_04_10_feature_selection.py"),

    # ("STEP 4.14 - Final RF model",
    #  "v5_04_14_random_forest_calibrated.py"),

    # ("STEP 4.15 - SHAP",
    #  "v5_04_15_shap_explainability.py"),

    # --------------------------------------------------------
    # Regulatory layer
    # --------------------------------------------------------

    (
        "STEP 4.16.1 - Regulatory embeddings",
        "v5_04_16_1_build_regulatory_embeddings.py",
    ),

    (
        "STEP 4.16.2 - Regulatory retrieval",
        "v5_04_16_2_regulatory_retrieval.py",
    ),

    (
        "STEP 4.16.3 - Regulatory retrieval QC",
        "v5_04_16_3_regulatory_retrieval_qc.py",
    ),

    # --------------------------------------------------------
    # Governance layer
    # --------------------------------------------------------

    (
        "STEP 4.17 - Final governance intelligence",
        "v5_04_17_final_governance_intelligence.py",
    ),
]


# ============================================================
# REQUIRED FINAL OUTPUTS
# ============================================================

FINAL_OUTPUTS = [
    FINAL_DIR / "final_governance_intelligence.csv",
    FINAL_DIR / "final_governance_summary.csv",
    FINAL_DIR / "final_governance_regulatory_priorities.csv",
    FINAL_DIR / "final_governance_risk_drivers.csv",
]


# ============================================================
# UTILITIES
# ============================================================

def print_header(text: str) -> None:
    print("\n" + "=" * 80)
    print(text)
    print("=" * 80)


def run_stage(
    title: str,
    script_name: str,
) -> None:

    script_path = SRC_DIR / script_name

    print_header(title)

    print(f"Script: {script_path}")

    if not script_path.exists():

        raise FileNotFoundError(
            f"\nRequired script does not exist:\n"
            f"{script_path}\n\n"
            f"Create/correct the script name before running."
        )

    start = time.time()

    print("\nStarting...\n")

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
        ],
        cwd=str(BASE_DIR),
        text=True,
    )

    elapsed = time.time() - start

    print(
        f"\nStage finished in {elapsed:.2f} seconds."
    )

    if result.returncode != 0:

        raise RuntimeError(
            f"\nFAILED: {title}\n"
            f"Script: {script_path}\n"
            f"Exit code: {result.returncode}"
        )

    print(
        f"\nPASS: {title}"
    )


def verify_outputs() -> None:

    print_header(
        "FINAL OUTPUT VALIDATION"
    )

    missing = []

    for path in FINAL_OUTPUTS:

        exists = path.exists()

        status = "OK" if exists else "MISSING"

        print(
            f"{status:<8} {path}"
        )

        if not exists:

            missing.append(path)

    if missing:

        raise RuntimeError(
            "\nPipeline completed, but required "
            "final outputs are missing:\n"
            + "\n".join(
                str(p)
                for p in missing
            )
        )


def validate_governance_file() -> None:

    print_header(
        "GOVERNANCE OUTPUT CHECK"
    )

    import pandas as pd

    path = (
        FINAL_DIR /
        "final_governance_intelligence.csv"
    )

    df = pd.read_csv(
        path
    )

    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    required_columns = [
        "subsidiary",
        "governance_priority_score",
        "governance_priority_level",
        "governance_status",
    ]

    missing_columns = [
        c
        for c in required_columns
        if c not in df.columns
    ]

    if missing_columns:

        raise RuntimeError(
            "Governance output is missing columns:\n"
            + "\n".join(
                missing_columns
            )
        )

    duplicate_mines = (
        df["subsidiary"]
        .duplicated()
        .sum()
    )

    missing_scores = (
        df[
            "governance_priority_score"
        ]
        .isna()
        .sum()
    )

    invalid_scores = (
        (
            df[
                "governance_priority_score"
            ] < 0
        )
        |
        (
            df[
                "governance_priority_score"
            ] > 100
        )
    ).sum()

    missing_levels = (
        df[
            "governance_priority_level"
        ]
        .isna()
        .sum()
    )

    print(
        f"Duplicate mines: {duplicate_mines}"
    )

    print(
        f"Missing governance scores: {missing_scores}"
    )

    print(
        f"Scores outside 0-100: {invalid_scores}"
    )

    print(
        f"Missing priority levels: {missing_levels}"
    )

    if duplicate_mines != 0:
        raise RuntimeError(
            "Duplicate mine records detected."
        )

    if missing_scores != 0:
        raise RuntimeError(
            "Missing governance scores detected."
        )

    if invalid_scores != 0:
        raise RuntimeError(
            "Governance scores outside 0-100 detected."
        )

    if missing_levels != 0:
        raise RuntimeError(
            "Missing governance priority labels detected."
        )

    print(
        "\nGovernance output integrity: PASS"
    )

    print(
        "\nFinal governance priorities:"
    )

    print(
        df[
            [
                "subsidiary",
                "governance_priority_score",
                "governance_priority_level",
                "governance_status",
            ]
        ]
        .sort_values(
            "governance_priority_score",
            ascending=False
        )
        .to_string(
            index=False
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    overall_start = time.time()

    print_header(
        "CoalMineAI - ONE COMMAND PIPELINE"
    )

    print(
        "Feature:"
    )

    print(
        "AI-Powered Statutory Compliance & "
        "Governance Intelligence"
    )

    print(
        f"\nProject: {BASE_DIR}"
    )

    print(
        f"Python: {sys.executable}"
    )

    print(
        f"Number of stages: {len(STAGES)}"
    )

    print(
        "\nConversational AI: NOT INCLUDED"
    )

    print(
        "Model retraining is controlled by the individual stages."
    )


    # --------------------------------------------------------
    # CHECK DIRECTORIES
    # --------------------------------------------------------

    if not BASE_DIR.exists():

        raise FileNotFoundError(
            f"Project directory not found:\n{BASE_DIR}"
        )

    if not SRC_DIR.exists():

        raise FileNotFoundError(
            f"Source directory not found:\n{SRC_DIR}"
        )

    FINAL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # RUN STAGES
    # --------------------------------------------------------

    completed = 0

    for title, script_name in STAGES:

        run_stage(
            title,
            script_name
        )

        completed += 1


    # --------------------------------------------------------
    # VALIDATE
    # --------------------------------------------------------

    verify_outputs()

    validate_governance_file()


    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    elapsed = (
        time.time()
        - overall_start
    )

    print_header(
        "FEATURE COMPLETE"
    )

    print(
        f"Completed stages: "
        f"{completed}/{len(STAGES)}"
    )

    print(
        f"Total runtime: "
        f"{elapsed:.2f} seconds"
    )

    print(
        "\nFINAL OUTPUT:"
    )

    print(
        FINAL_DIR /
        "final_governance_intelligence.csv"
    )

    print(
        "\nSTATUS: PASS"
    )

    print(
        "\nThe current feature is ready to be frozen."
    )


if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print(
            "\n\nPipeline interrupted by user."
        )

        sys.exit(130)

    except Exception as exc:

        print(
            "\n" + "=" * 80
        )

        print(
            "PIPELINE FAILED"
        )

        print(
            "=" * 80
        )

        print(
            f"\nError:\n{exc}"
        )

        sys.exit(1)