import os
import sys
import subprocess
import time

# ============================================================
# COALMINE AI - MASTER PIPELINE RUNNER
# ============================================================

BASE_DIR = r"D:\CoalMineAI"
SRC_DIR = os.path.join(BASE_DIR, "src")

PYTHON = sys.executable


# ============================================================
# PIPELINE
# ============================================================

PIPELINE = [
    ("Data Loader", "data_loader.py"),
    ("Preprocessing", "preprocessing.py"),
    ("Feature Engineering", "feature_engineering.py"),
    ("Production Forecasting", "train_forecaster.py"),
    ("Risk Engine", "risk_engine.py"),
    ("Multi-Mine Governance", "multi_mine_governance.py"),
    ("Early Warning Engine", "early_warning_engine.py"),
    ("Predictive Risk Escalation", "predictive_risk_escalation.py"),
    ("Management Recommendation", "recommendation_engine.py"),
]


# ============================================================
# HEADER
# ============================================================

print("\n")
print("=" * 80)
print("             COALMINE AI - MASTER AI PIPELINE")
print("=" * 80)

print(f"\nProject directory:")
print(BASE_DIR)

print(f"\nPython interpreter:")
print(PYTHON)

print("\nPipeline:")
for number, (name, filename) in enumerate(PIPELINE, start=1):
    print(f"  {number}. {name}")


# ============================================================
# RUN EACH PHASE
# ============================================================

total_start = time.time()

for index, (phase_name, filename) in enumerate(PIPELINE, start=1):

    script_path = os.path.join(SRC_DIR, filename)

    print("\n")
    print("=" * 80)
    print(f"PHASE {index}/{len(PIPELINE)} - {phase_name}")
    print("=" * 80)

    if not os.path.exists(script_path):
        print(f"\nERROR: File not found:")
        print(script_path)

        print("\nPipeline stopped.")
        sys.exit(1)

    start_time = time.time()

    print(f"\nRunning:")
    print(script_path)

    try:

        result = subprocess.run(
            [PYTHON, script_path],
            cwd=BASE_DIR,
            check=False
        )

    except Exception as error:

        print("\nERROR while starting phase:")
        print(error)

        print("\nPipeline stopped.")
        sys.exit(1)

    elapsed = time.time() - start_time

    if result.returncode != 0:

        print("\n")
        print("!" * 80)
        print(f"FAILED: {phase_name}")
        print(f"Exit code: {result.returncode}")
        print("!" * 80)

        print("\nPipeline stopped.")
        print(
            "Fix the failed phase before continuing."
        )

        sys.exit(result.returncode)

    print("\n")
    print("-" * 80)
    print(f"COMPLETED: {phase_name}")
    print(f"Time: {elapsed:.2f} seconds")
    print("-" * 80)


# ============================================================
# FINAL OUTPUT CHECK
# ============================================================

print("\n")
print("=" * 80)
print("                 FINAL OUTPUT CHECK")
print("=" * 80)

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

expected_outputs = [
    "cleaned_coal_production_data.csv",
    "engineered_coal_features.csv",
    "production_risk_analysis.csv",
    "mine_governance_summary.csv",
    "early_warning_analysis.csv",
    "mine_escalation_summary.csv",
    "predictive_risk_escalation.csv",
    "final_management_recommendations.csv",
]

missing = []

for filename in expected_outputs:

    path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    if os.path.exists(path):

        size = os.path.getsize(path)

        print(
            f"OK       {filename:<45} "
            f"{size:,} bytes"
        )

    else:

        print(
            f"MISSING  {filename}"
        )

        missing.append(filename)


# ============================================================
# FINAL SUMMARY
# ============================================================

total_time = time.time() - total_start

print("\n")
print("=" * 80)

if missing:

    print("PIPELINE COMPLETED WITH MISSING OUTPUTS")

    print("\nMissing files:")

    for filename in missing:
        print(f"  - {filename}")

else:

    print("       COALMINE AI PIPELINE COMPLETED SUCCESSFULLY")

print("=" * 80)

print(
    f"\nTotal execution time: "
    f"{total_time:.2f} seconds"
)

print(
    f"\nOutput directory:\n"
    f"{OUTPUT_DIR}"
)

print("\n")
