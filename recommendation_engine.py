import pandas as pd
import os

# ============================================================
# COALMINE AI - FINAL MANAGEMENT RECOMMENDATION ENGINE
# ============================================================

BASE_DIR = r"D:\CoalMineAI"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

INPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "mine_escalation_summary.csv"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "final_management_recommendations.csv"
)

print("=" * 70)
print("COALMINE AI - FINAL MANAGEMENT RECOMMENDATION ENGINE")
print("=" * 70)


# ============================================================
# 1. LOAD PHASE 7C OUTPUT
# ============================================================

print("\n[1] Loading Phase 7C output...")

if not os.path.exists(INPUT_FILE):
    print("\nERROR: Phase 7C output not found:")
    print(INPUT_FILE)
    print("\nRun Phase 7C first.")
    exit()

df = pd.read_csv(INPUT_FILE)

print(f"Rows loaded: {len(df)}")
print(f"Columns loaded: {len(df.columns)}")


# ============================================================
# 2. CLEAN COLUMN NAMES
# ============================================================

print("\n[2] Cleaning data...")

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

print("Columns detected:")
print(list(df.columns))


# ============================================================
# 3. CONVERT NUMERIC VALUES
# ============================================================

numeric_columns = [
    "governance_score",
    "early_warning_score",
    "predicted_risk_score",
    "escalation_probability"
]

for column in numeric_columns:
    if column in df.columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


# ============================================================
# 4. CALCULATE MANAGEMENT PRIORITY
# ============================================================

print("\n[3] Calculating management priority...")


def calculate_priority(row):

    predicted_score = row.get(
        "predicted_risk_score", 0
    )

    escalation_probability = row.get(
        "escalation_probability", 0
    )

    current_risk = str(
        row.get(
            "production_risk",
            ""
        )
    ).upper()

    direction = str(
        row.get(
            "escalation_direction",
            ""
        )
    ).upper()

    if (
        predicted_score >= 75
        or current_risk == "UNRELIABLE"
    ):
        return "HIGH"

    if (
        escalation_probability >= 40
        or direction == "ESCALATING"
        or predicted_score >= 45
    ):
        return "MEDIUM"

    return "LOW"


df["management_priority"] = df.apply(
    calculate_priority,
    axis=1
)


# ============================================================
# 5. GENERATE RECOMMENDATIONS
# ============================================================

print("\n[4] Generating management recommendations...")


def generate_action(row):

    priority = row["management_priority"]

    direction = str(
        row.get(
            "escalation_direction",
            ""
        )
    ).upper()

    current_risk = str(
        row.get(
            "production_risk",
            ""
        )
    ).upper()

    predicted_risk = str(
        row.get(
            "predicted_next_risk_level",
            ""
        )
    ).upper()

    if priority == "HIGH":

        if current_risk == "UNRELIABLE":
            return (
                "Prioritize management review, "
                "investigate root causes, and initiate "
                "preventive intervention."
            )

        return (
            "Immediate risk review and mitigation. "
            "Increase monitoring frequency."
        )

    if priority == "MEDIUM":

        if direction == "ESCALATING":
            return (
                "Increase monitoring frequency and "
                "prepare preventive mitigation measures."
            )

        return (
            "Conduct management review and closely "
            "monitor key risk indicators."
        )

    if direction == "STABLE":
        return (
            "Maintain current controls and routine monitoring."
        )

    return (
        "Continue monitoring and maintain existing controls."
    )


df["recommended_action"] = df.apply(
    generate_action,
    axis=1
)


# ============================================================
# 6. GENERATE EXPLANATION
# ============================================================

print("\n[5] Generating decision explanations...")


def generate_explanation(row):

    direction = str(
        row.get(
            "escalation_direction",
            "UNKNOWN"
        )
    )

    probability = row.get(
        "escalation_probability",
        0
    )

    score = row.get(
        "predicted_risk_score",
        0
    )

    return (
        f"Predicted risk score is {score:.2f}, "
        f"with {probability:.1f}% escalation probability. "
        f"Current trajectory is {direction.lower()}."
    )


df["decision_explanation"] = df.apply(
    generate_explanation,
    axis=1
)


# ============================================================
# 7. RANK MINES
# ============================================================

print("\n[6] Ranking mines...")


priority_order = {
    "HIGH": 1,
    "MEDIUM": 2,
    "LOW": 3
}

df["priority_rank"] = (
    df["management_priority"]
    .map(priority_order)
)

df = df.sort_values(
    by=[
        "priority_rank",
        "escalation_probability",
        "predicted_risk_score"
    ],
    ascending=[
        True,
        False,
        False
    ]
)

df["management_rank"] = range(
    1,
    len(df) + 1
)


# ============================================================
# 8. SAVE FINAL FILE
# ============================================================

print("\n[7] Saving final recommendations...")

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nOutput created successfully:")
print(OUTPUT_FILE)


# ============================================================
# 9. DISPLAY FINAL RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("FINAL MANAGEMENT PRIORITY")
print("=" * 70)

for _, row in df.iterrows():

    print("\n" + "-" * 60)

    print(
        f"Rank       : {row['management_rank']}"
    )

    print(
        f"Mine       : {row.get('subsidiary', 'N/A')}"
    )

    print(
        f"Current    : {row.get('production_risk', 'N/A')}"
    )

    print(
        f"Next Risk  : {row.get('predicted_next_risk_level', 'N/A')}"
    )

    print(
        f"Score      : {row.get('predicted_risk_score', 0):.2f}"
    )

    print(
        f"Escalation : {row.get('escalation_probability', 0):.1f}%"
    )

    print(
        f"Direction  : {row.get('escalation_direction', 'N/A')}"
    )

    print(
        f"Priority   : {row['management_priority']}"
    )

    print(
        f"Action     : {row['recommended_action']}"
    )


# ============================================================
# 10. SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("MANAGEMENT SUMMARY")
print("=" * 70)

counts = df["management_priority"].value_counts()

print(
    f"\nHIGH     : {counts.get('HIGH', 0)}"
)

print(
    f"MEDIUM   : {counts.get('MEDIUM', 0)}"
)

print(
    f"LOW      : {counts.get('LOW', 0)}"
)

print(
    f"\nTotal mines analyzed: {len(df)}"
)

print("\n" + "=" * 70)
print("FINAL RECOMMENDATION ENGINE COMPLETE")
print("=" * 70)