import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

PIPELINE = [
    ("Data Loader", "data_loader.py"),
    ("Preprocessing", "preprocessing.py"),
    ("Feature Engineering", "feature_engineering.py"),
    ("Production Forecasting", "train_forecaster.py"),
    ("Operational Risk", "risk_engine.py"),
    ("Early Warning", "early_warning_engine.py"),
    ("Predictive Risk Escalation", "predictive_risk_escalation.py"),
    ("Regulatory Intelligence", "regulatory_intelligence.py"),
    ("Regulatory Normalization", "regulatory_normalization.py"),
    ("Regulation-to-Mine Matching", "regulation_mine_matching.py"),
    ("Mine-Specific Compliance Risk", "mine_specific_compliance.py"),
    ("Compliance Evidence & Gap Intelligence", "compliance_evidence_gap.py"),
    ("Compliance Evidence Ingestion", "compliance_evidence_ingestion.py"),
]

def main():
    print("=" * 70)
    print("COALMINEAI - RUN ALL")
    print("AI STATUTORY & GOVERNANCE PIPELINE - PHASE 1 TO 9")
    print("=" * 70)

    for i, (name, filename) in enumerate(PIPELINE, 1):
        script = SRC / filename
        print(f"\n[{i}/{len(PIPELINE)}] {name}")
        print(f"Running: {script}")

        if not script.exists():
            print(f"ERROR: File not found: {script}")
            sys.exit(1)

        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(ROOT)
        )

        if result.returncode != 0:
            print(f"\nFAILED: {name}")
            print(f"Exit code: {result.returncode}")
            print("Pipeline stopped.")
            sys.exit(result.returncode)

        print(f"COMPLETED: {name}")

    print("\n" + "=" * 70)
    print("ALL PHASE 1-9 PHASES COMPLETED")
    print("=" * 70)
    print(f"Outputs: {ROOT / 'outputs'}")
    print("\nIf Phase 9F says '0 evidence PDFs', place evidence files in:")
    print(ROOT / "compliance_evidence")

if __name__ == "__main__":
    main()
