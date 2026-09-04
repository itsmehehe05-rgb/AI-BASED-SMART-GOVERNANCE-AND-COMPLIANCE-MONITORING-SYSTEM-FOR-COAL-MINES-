# =============================================================================
# V5 STEP 4.10
# LEAKAGE-SAFE FEATURE SELECTION & REDUNDANCY REMOVAL
# =============================================================================
#
# Input:
#   D:\CoalMineAI\outputs\v5\v5_04_9_leakage_safe_dataset.csv
#
# Output:
#   D:\CoalMineAI\outputs\v5\v5_04_10_selected_features_dataset.csv
#   D:\CoalMineAI\outputs\v5\v5_04_10_feature_selection_audit.csv
#
# Purpose:
#   - Keep final test untouched
#   - Select features using DEVELOPMENT data only
#   - Remove metadata
#   - Remove exact duplicates
#   - Remove highly correlated redundant features
#   - Remove weak/uninformative features
#   - Preserve important temporal and operational features
#
# IMPORTANT:
#   This script DOES NOT train the final model.
#
# =============================================================================

import os
import warnings
import numpy as np
import pandas as pd

from sklearn.feature_selection import mutual_info_classif

warnings.filterwarnings("ignore")


# =============================================================================
# 1. PATHS
# =============================================================================

BASE_DIR = r"D:\CoalMineAI"

INPUT_FILE = os.path.join(
    BASE_DIR,
    "outputs",
    "v5",
    "v5_04_9_leakage_safe_dataset.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "v5"
)

OUTPUT_DATASET = os.path.join(
    OUTPUT_DIR,
    "v5_04_10_selected_features_dataset.csv"
)

OUTPUT_AUDIT = os.path.join(
    OUTPUT_DIR,
    "v5_04_10_feature_selection_audit.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# 2. CONFIGURATION
# =============================================================================

TARGET = "material_escalation"

DATE_COL = "date"
RECORD_DATE_COL = "record_date"
MINE_COL = "subsidiary"

# Approximately last 15% of unique dates is held out.
FINAL_TEST_FRACTION = 0.15

# Correlation threshold for redundancy removal.
CORRELATION_THRESHOLD = 0.95

# Minimum mutual-information percentile.
# We don't aggressively remove features solely from MI because nonlinear
# relationships may be important for tree models.
MI_PERCENTILE = 20


# =============================================================================
# 3. HELPER
# =============================================================================

def print_section(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def add_audit(rows, feature, status, reason, value=None):

    rows.append({
        "feature": feature,
        "status": status,
        "reason": reason,
        "score": value
    })


# =============================================================================
# 4. LOAD DATA
# =============================================================================

print_section("1. LOAD LEAKAGE-SAFE DATASET")

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print(
    f"Loaded dataset: {df.shape}"
)

if TARGET not in df.columns:
    raise ValueError(
        f"Target '{TARGET}' not found."
    )

if MINE_COL not in df.columns:
    raise ValueError(
        f"Mine column '{MINE_COL}' not found."
    )


# =============================================================================
# 5. NORMALIZE / RECOVER DATE
# =============================================================================

print_section("2. DATE AND TARGET VALIDATION")

if RECORD_DATE_COL in df.columns:

    df[RECORD_DATE_COL] = pd.to_datetime(
        df[RECORD_DATE_COL],
        errors="coerce"
    )

    actual_date_col = RECORD_DATE_COL

elif DATE_COL in df.columns:

    df[DATE_COL] = pd.to_datetime(
        df[DATE_COL],
        errors="coerce"
    )

    actual_date_col = DATE_COL

else:

    raise ValueError(
        "No date column found."
    )

if df[actual_date_col].isna().any():

    raise ValueError(
        "Missing/invalid dates detected."
    )

print(
    f"Date column: {actual_date_col}"
)

print(
    f"Date range: "
    f"{df[actual_date_col].min()} → "
    f"{df[actual_date_col].max()}"
)

print()
print("Target distribution:")
print(
    df[TARGET].value_counts(dropna=False)
)


# =============================================================================
# 6. SORT DATA
# =============================================================================

df = df.sort_values(
    [actual_date_col, MINE_COL]
).reset_index(drop=True)


# =============================================================================
# 7. DEVELOPMENT / FINAL TEST SPLIT
# =============================================================================

print_section("3. DEVELOPMENT / FINAL TEST SPLIT")

unique_dates = sorted(
    df[actual_date_col].unique()
)

n_dates = len(unique_dates)

test_start_index = int(
    n_dates * (1 - FINAL_TEST_FRACTION)
)

test_start_index = min(
    max(test_start_index, 1),
    n_dates - 1
)

test_start_date = unique_dates[
    test_start_index
]

development = df[
    df[actual_date_col] < test_start_date
].copy()

final_test = df[
    df[actual_date_col] >= test_start_date
].copy()

print(
    f"Unique dates: {n_dates}"
)

print(
    f"Development records: {len(development)}"
)

print(
    f"Final test records: {len(final_test)}"
)

print()
print(
    f"Development: "
    f"{development[actual_date_col].min()} → "
    f"{development[actual_date_col].max()}"
)

print(
    f"Final test: "
    f"{final_test[actual_date_col].min()} → "
    f"{final_test[actual_date_col].max()}"
)

print()
print("Development target:")
print(
    development[TARGET].value_counts()
)

print()
print("Final test target:")
print(
    final_test[TARGET].value_counts()
)


# =============================================================================
# 8. IDENTIFY METADATA
# =============================================================================

print_section("4. IDENTIFY METADATA")

metadata_columns = {
    TARGET,
    DATE_COL,
    RECORD_DATE_COL,
    MINE_COL,
}

metadata_present = [
    c for c in metadata_columns
    if c in df.columns
]

print("Metadata columns:")
for col in metadata_present:
    print(f"  - {col}")


# =============================================================================
# 9. IDENTIFY CANDIDATE FEATURES
# =============================================================================

candidate_features = [
    col
    for col in df.columns
    if col not in metadata_columns
]

print()
print(
    f"Candidate ML features: "
    f"{len(candidate_features)}"
)


# =============================================================================
# 10. REMOVE OBVIOUS NON-PREDICTIVE / IDENTIFIER FEATURES
# =============================================================================

print_section("5. REMOVE IDENTIFIER / NON-PREDICTIVE FEATURES")

audit_rows = []

remove_names = {
    "financial_year",
    "split",
    "source_basis",
    "synthetic_flag",
}

# Explicitly avoid raw date-derived identifiers.
remove_names_lower = {
    x.lower()
    for x in remove_names
}

initial_candidates = []

for feature in candidate_features:

    if feature.lower() in remove_names_lower:

        add_audit(
            audit_rows,
            feature,
            "REMOVE",
            "Identifier / metadata / source field"
        )

    else:

        initial_candidates.append(feature)


# =============================================================================
# 11. REMOVE CONSTANT FEATURES
# =============================================================================

print_section("6. REMOVE CONSTANT FEATURES")

constant_features = []

for feature in initial_candidates:

    if development[feature].nunique(
        dropna=False
    ) <= 1:

        constant_features.append(feature)

        add_audit(
            audit_rows,
            feature,
            "REMOVE",
            "Constant feature"
        )

print(
    f"Constant features removed: "
    f"{len(constant_features)}"
)

initial_candidates = [
    f
    for f in initial_candidates
    if f not in constant_features
]


# =============================================================================
# 12. REMOVE FEATURES WITH EXTREME MISSINGNESS
# =============================================================================

print_section("7. MISSING VALUE AUDIT")

missing_features = []

for feature in initial_candidates:

    missing_rate = (
        development[feature]
        .isna()
        .mean()
    )

    if missing_rate > 0.40:

        missing_features.append(feature)

        add_audit(
            audit_rows,
            feature,
            "REMOVE",
            f"Development missing rate > 40% ({missing_rate:.1%})",
            missing_rate
        )

print(
    f"High-missingness features removed: "
    f"{len(missing_features)}"
)

initial_candidates = [
    f
    for f in initial_candidates
    if f not in missing_features
]


# =============================================================================
# 13. EXACT DUPLICATE FEATURES
# =============================================================================

print_section("8. EXACT DUPLICATE FEATURE REMOVAL")

numeric_candidates = [
    f
    for f in initial_candidates
    if pd.api.types.is_numeric_dtype(
        development[f]
    )
]

duplicate_features = []
seen_features = []

for feature in numeric_candidates:

    is_duplicate = False
    duplicate_of = None

    for previous in seen_features:

        if development[feature].equals(
            development[previous]
        ):

            is_duplicate = True
            duplicate_of = previous
            break

    if is_duplicate:

        duplicate_features.append(feature)

        add_audit(
            audit_rows,
            feature,
            "REMOVE",
            f"Exact duplicate of {duplicate_of}"
        )

    else:

        seen_features.append(feature)


print(
    f"Exact duplicates removed: "
    f"{len(duplicate_features)}"
)

initial_candidates = [
    f
    for f in initial_candidates
    if f not in duplicate_features
]


# =============================================================================
# 14. HIGH CORRELATION REDUNDANCY
# =============================================================================

print_section("9. HIGH-CORRELATION REDUNDANCY REMOVAL")

numeric_candidates = [
    f
    for f in initial_candidates
    if pd.api.types.is_numeric_dtype(
        development[f]
    )
]

if len(numeric_candidates) > 1:

    corr_df = development[
        numeric_candidates
    ].corr(
        method="spearman"
    ).abs()

    to_remove = set()

    # Feature priority:
    # A feature with stronger relationship to target gets preference.
    target_corr = {}

    y_dev = development[TARGET]

    for feature in numeric_candidates:

        try:

            target_corr[feature] = abs(
                development[
                    [feature, TARGET]
                ]
                .corr(
                    method="spearman"
                )
                .iloc[0, 1]
            )

        except Exception:

            target_corr[feature] = 0.0

    for i, feature_a in enumerate(
        numeric_candidates
    ):

        if feature_a in to_remove:
            continue

        for feature_b in numeric_candidates[
            i + 1:
        ]:

            if feature_b in to_remove:
                continue

            corr_value = corr_df.loc[
                feature_a,
                feature_b
            ]

            if (
                pd.notna(corr_value)
                and corr_value >= CORRELATION_THRESHOLD
            ):

                # Keep the feature with stronger
                # relationship to target.
                score_a = target_corr.get(
                    feature_a,
                    0
                )

                score_b = target_corr.get(
                    feature_b,
                    0
                )

                if score_a >= score_b:

                    remove = feature_b
                    keep = feature_a

                else:

                    remove = feature_a
                    keep = feature_b

                to_remove.add(remove)

                add_audit(
                    audit_rows,
                    remove,
                    "REMOVE",
                    (
                        f"Highly correlated with {keep}; "
                        f"Spearman correlation={corr_value:.4f}"
                    ),
                    corr_value
                )

    initial_candidates = [
        f
        for f in initial_candidates
        if f not in to_remove
    ]

    print(
        f"Highly correlated features removed: "
        f"{len(to_remove)}"
    )

else:

    print(
        "Not enough numeric features for correlation analysis."
    )


# =============================================================================
# 15. MUTUAL INFORMATION SCREENING
# =============================================================================

print_section("10. MUTUAL INFORMATION SCREENING")

numeric_candidates = [
    f
    for f in initial_candidates
    if pd.api.types.is_numeric_dtype(
        development[f]
    )
]

# We use MI as a screening/audit tool, NOT as the sole
# reason to remove features.
mi_scores = {}

if len(numeric_candidates) > 0:

    X_mi = development[
        numeric_candidates
    ].copy()

    y_mi = development[
        TARGET
    ].astype(int)

    # Median imputation only for this analysis.
    # This does not create the final model preprocessing.
    X_mi = X_mi.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X_mi = X_mi.fillna(
        X_mi.median(numeric_only=True)
    )

    # Any columns still completely empty.
    X_mi = X_mi.fillna(0)

    try:

        mi_values = mutual_info_classif(
            X_mi,
            y_mi,
            random_state=42
        )

        mi_scores = dict(
            zip(
                numeric_candidates,
                mi_values
            )
        )

    except Exception as exc:

        print(
            "WARNING: Mutual information failed:"
        )

        print(exc)

else:

    print(
        "No numeric candidates available."
    )


# =============================================================================
# 16. DISPLAY TOP FEATURES
# =============================================================================

print_section("11. TOP FEATURES BY MUTUAL INFORMATION")

if mi_scores:

    mi_sorted = sorted(
        mi_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    for feature, score in mi_sorted[:30]:

        print(
            f"{feature:55s} "
            f"{score:.6f}"
        )

else:

    mi_sorted = []


# =============================================================================
# 17. WEAK FEATURE REVIEW
# =============================================================================

print_section("12. WEAK FEATURE REVIEW")

# IMPORTANT:
# We do NOT aggressively delete low-MI features.
#
# Tree models can use nonlinear interactions.
# Instead, weak features are marked REVIEW.
#
# This avoids destroying useful temporal interactions.

review_features = []

if mi_scores:

    mi_values = np.array(
        list(mi_scores.values())
    )

    if len(mi_values) > 3:

        cutoff = np.percentile(
            mi_values,
            MI_PERCENTILE
        )

        for feature, score in mi_scores.items():

            if score < cutoff:

                review_features.append(
                    feature
                )

                add_audit(
                    audit_rows,
                    feature,
                    "REVIEW",
                    (
                        f"Low mutual information "
                        f"({score:.6f}); retained for "
                        f"nonlinear/interactions"
                    ),
                    score
                )

print(
    f"Features marked REVIEW: "
    f"{len(review_features)}"
)


# =============================================================================
# 18. MANUAL PROTECTION OF CORE FEATURES
# =============================================================================

print_section("13. PROTECT CORE RISK FEATURES")

# These are important operational predictors.
# Even if MI is low, keep them for the next model comparison.

CORE_FEATURE_PATTERNS = [
    "overall_operational_risk",
    "equipment_risk",
    "logistics_risk",
    "weather_risk",
    "workforce_risk",
    "risk_change",
    "risk_acceleration",
    "operational_risk_change_3m",
    "consecutive_risk_increases",
    "component_risk_mean",
    "component_risk_max",
    "component_risk_std",
    "component_risk_range",
    "monthly_production_mt",
    "production_lag_1",
    "production_lag_2",
    "production_lag_3",
    "production_change_1",
    "production_change_3",
    "production_previous_mean_3",
    "target_gap_mt",
    "target_achievement_ratio",
    "month",
    "quarter",
    "month_sin",
    "month_cos",
]

protected_features = []

for feature in initial_candidates:

    lower = feature.lower()

    if any(
        pattern.lower() in lower
        for pattern in CORE_FEATURE_PATTERNS
    ):

        protected_features.append(feature)

print(
    f"Protected core features: "
    f"{len(protected_features)}"
)


# =============================================================================
# 19. FINAL FEATURE SET
# =============================================================================

print_section("14. BUILD FINAL FEATURE SET")

final_features = []

for feature in initial_candidates:

    # Keep everything remaining after objective
    # redundancy checks.
    final_features.append(feature)

# Make sure protected features are present.
for feature in protected_features:

    if (
        feature not in final_features
        and feature in initial_candidates
    ):

        final_features.append(feature)


# Remove accidental duplicates.
final_features = list(
    dict.fromkeys(final_features)
)

print(
    f"Final selected ML features: "
    f"{len(final_features)}"
)


# =============================================================================
# 20. FINAL FEATURE LIST
# =============================================================================

print()
print("Selected features:")

for i, feature in enumerate(
    final_features,
    start=1
):

    print(
        f"{i:3d}. {feature}"
    )


# =============================================================================
# 21. BUILD OUTPUT DATASET
# =============================================================================

print_section("15. BUILD OUTPUT DATASET")

output_columns = []

# Metadata first.
for col in [
    RECORD_DATE_COL,
    DATE_COL,
    MINE_COL,
    TARGET,
]:

    if col in df.columns and col not in output_columns:

        output_columns.append(col)

# Then selected ML features.
for feature in final_features:

    if feature in df.columns:

        output_columns.append(feature)

selected_df = df[
    output_columns
].copy()

print(
    f"Output dataset shape: "
    f"{selected_df.shape}"
)


# =============================================================================
# 22. FINAL CHECK FOR FUTURE-LOOKING FEATURES
# =============================================================================

print_section("16. FINAL FUTURE-LOOKING FEATURE CHECK")

future_patterns = [
    "next_month",
    "future_",
    "future",
    "predicted_next",
    "predicted_risk",
    "escalation_probability",
    "warning_level",
    "early_warning",
    "predicted_next_risk",
]

suspicious = []

for feature in final_features:

    lower = feature.lower()

    for pattern in future_patterns:

        if pattern in lower:

            suspicious.append(feature)

            break

if suspicious:

    print(
        "WARNING: suspicious features found:"
    )

    for feature in suspicious:

        print(
            f"  - {feature}"
        )

else:

    print(
        "PASS — no obvious future-looking "
        "features detected."
    )


# =============================================================================
# 23. SAVE DATASET
# =============================================================================

print_section("17. SAVE SELECTED FEATURE DATASET")

selected_df.to_csv(
    OUTPUT_DATASET,
    index=False
)

print(
    f"Saved:\n{OUTPUT_DATASET}"
)


# =============================================================================
# 24. COMPLETE AUDIT
# =============================================================================

print_section("18. SAVE FEATURE AUDIT")

audited_features = {
    row["feature"]
    for row in audit_rows
}

for feature in final_features:

    if feature not in audited_features:

        score = mi_scores.get(
            feature,
            np.nan
        )

        add_audit(
            audit_rows,
            feature,
            "KEEP",
            "Retained after redundancy and leakage checks",
            score
        )


audit_df = pd.DataFrame(
    audit_rows
)

audit_df = audit_df.drop_duplicates(
    subset=["feature"],
    keep="first"
)

audit_df = audit_df.sort_values(
    ["status", "feature"]
).reset_index(drop=True)

audit_df.to_csv(
    OUTPUT_AUDIT,
    index=False
)

print(
    f"Saved:\n{OUTPUT_AUDIT}"
)


# =============================================================================
# 25. FINAL SUMMARY
# =============================================================================

print_section("19. FINAL SUMMARY")

print(
    f"Input rows: "
    f"{len(df)}"
)

print(
    f"Development rows: "
    f"{len(development)}"
)

print(
    f"Final test rows: "
    f"{len(final_test)}"
)

print(
    f"Initial candidate features: "
    f"{len(candidate_features)}"
)

print(
    f"Final ML features: "
    f"{len(final_features)}"
)

print(
    f"Exact duplicates removed: "
    f"{len(duplicate_features)}"
)

print(
    f"High-correlation features removed: "
    f"{len(duplicate_features) + 0}"
    if False
    else ""
)

print(
    f"Final dataset shape: "
    f"{selected_df.shape}"
)

print()

print(
    "Development positive rate: "
    f"{development[TARGET].mean():.2%}"
)

print(
    "Final test positive rate: "
    f"{final_test[TARGET].mean():.2%}"
)

print()

if suspicious:

    print(
        "STATUS: REVIEW REQUIRED"
    )

else:

    print(
        "STATUS: FEATURE SELECTION COMPLETED"
    )

print()
print(
    "V5 STEP 4.10 COMPLETED"
)

print(
    "Next: temporal model training using the selected features."
)

print("=" * 80)