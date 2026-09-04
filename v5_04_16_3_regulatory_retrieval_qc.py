import os
import re
import numpy as np
import pandas as pd
import faiss

from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"D:\CoalMineAI"

OUTPUT_DIR = os.path.join(
    BASE_DIR, "outputs", "v5", "FINAL"
)

REGULATORY_FILE = os.path.join(
    BASE_DIR,
    "outputs",
    "normalized_regulatory_requirements.csv"
)

RISK_FILE = os.path.join(
    BASE_DIR,
    "outputs",
    "production_risk_analysis.csv"
)

EMBEDDING_FILE = os.path.join(
    OUTPUT_DIR,
    "regulatory_embeddings.npy"
)

FAISS_FILE = os.path.join(
    OUTPUT_DIR,
    "regulatory_faiss.index"
)


# ============================================================
# SETTINGS
# ============================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

INITIAL_TOP_K = 50
FINAL_TOP_K = 15

# Maximum results from one source PDF
MAX_PER_DOCUMENT = 4

# Maximum results from one regulatory domain
MAX_PER_DOMAIN = 8

# Ignore extremely weak semantic matches
MIN_SIMILARITY = 0.40


# Ranking weights
W_SEMANTIC = 0.70
W_PRIORITY = 0.15
W_DOMAIN = 0.10
W_RISK = 0.05


# ============================================================
# HELPERS
# ============================================================

def clean(value):

    if pd.isna(value):
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(value)
    ).strip()


def priority_value(value):

    value = clean(value).upper()

    if value == "HIGH":
        return 1.0

    if value == "MEDIUM":
        return 0.60

    if value == "LOW":
        return 0.25

    return 0.50


def tokenize(text):

    text = clean(text).lower()

    return set(
        re.findall(
            r"[a-z0-9]+",
            text
        )
    )


def jaccard_similarity(a, b):

    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


def near_duplicate(
    text,
    previous_texts,
    threshold=0.85
):

    current = tokenize(text)

    for previous in previous_texts:

        if jaccard_similarity(
            current,
            previous
        ) >= threshold:

            return True

    return False


# ============================================================
# LOAD REGULATIONS
# ============================================================

print("=" * 70)
print("STEP 4.16.3 - REGULATORY RETRIEVAL QUALITY CONTROL")
print("=" * 70)

print("\nLoading regulatory data...")

reg_df = pd.read_csv(
    REGULATORY_FILE
)

print(
    "Regulatory data:",
    reg_df.shape
)


# ============================================================
# LOAD EMBEDDINGS + FAISS
# ============================================================

print("\nLoading embeddings...")

embeddings = np.load(
    EMBEDDING_FILE
)

print(
    "Embeddings:",
    embeddings.shape
)


print("\nLoading FAISS index...")

index = faiss.read_index(
    FAISS_FILE
)

print(
    "FAISS vectors:",
    index.ntotal
)


# Safety checks

if len(reg_df) != len(embeddings):

    raise ValueError(
        "Regulatory rows and embeddings do not match."
    )


if len(reg_df) != index.ntotal:

    raise ValueError(
        "Regulatory rows and FAISS index do not match."
    )


print("Alignment check: PASS")


# ============================================================
# PREPARE REGULATORY FIELDS
# ============================================================

reg_df["domain_clean"] = (
    reg_df["regulatory_domain"]
    .fillna("GENERAL")
    .astype(str)
    .str.upper()
)

reg_df["priority_numeric"] = (
    reg_df["regulatory_priority"]
    .apply(priority_value)
)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

model = SentenceTransformer(
    MODEL_NAME
)

print("Embedding model loaded.")


# ============================================================
# LOAD RISK DATA
# ============================================================

print("\nLoading mine risk data...")

risk_df = pd.read_csv(
    RISK_FILE
)

print(
    "Risk data:",
    risk_df.shape
)


# ============================================================
# GET LATEST RECORD FOR EACH MINE
# ============================================================

if "record_date" in risk_df.columns:

    risk_df["record_date"] = pd.to_datetime(
        risk_df["record_date"],
        errors="coerce"
    )

elif "date" in risk_df.columns:

    risk_df["date"] = pd.to_datetime(
        risk_df["date"],
        errors="coerce"
    )

    risk_df["record_date"] = risk_df["date"]

else:

    raise ValueError(
        "No date column found in risk data."
    )


latest = (
    risk_df
    .sort_values("record_date")
    .groupby("subsidiary")
    .tail(1)
    .copy()
)

latest = latest.sort_values(
    "subsidiary"
)

print(
    "\nLatest mine records:"
)

print(
    latest[
        [
            "subsidiary",
            "record_date"
        ]
    ].to_string(index=False)
)


# ============================================================
# DOMAIN RELEVANCE
# ============================================================

def calculate_domain_scores(row):

    operational = float(
        row.get(
            "overall_operational_risk",
            0
        )
    )

    equipment = float(
        row.get(
            "equipment_risk",
            0
        )
    )

    logistics = float(
        row.get(
            "logistics_risk",
            0
        )
    )

    weather = float(
        row.get(
            "weather_risk",
            0
        )
    )

    workforce = float(
        row.get(
            "workforce_risk",
            0
        )
    )

    scores = {

        "SAFETY":
            0.40 * operational
            + 0.30 * equipment
            + 0.20 * workforce
            + 0.10 * logistics,

        "ENVIRONMENT":
            0.70 * weather
            + 0.30 * operational,

        "WASTE":
            0.50 * weather
            + 0.50 * operational,

        "MINING":
            0.60 * operational
            + 0.40 * logistics,

        "COMPLIANCE":
            operational,

        "GENERAL":
            0.50 * operational
    }

    max_score = max(
        scores.values()
    )

    if max_score > 0:

        for key in scores:

            scores[key] /= max_score

    return scores


# ============================================================
# QUERY CREATION
# ============================================================

def build_query(row):

    mine = clean(
        row["subsidiary"]
    )

    operational = clean(
        row.get(
            "operational_risk_level",
            ""
        )
    )

    production = clean(
        row.get(
            "production_risk_level",
            ""
        )
    )

    driver = clean(
        row.get(
            "primary_driver",
            ""
        )
    )

    return f"""
    Statutory and regulatory compliance requirements
    for Indian coal mine {mine}.
    Current operational risk: {operational}.
    Current production risk: {production}.
    Main operational concern: {driver}.
    Identify relevant mining safety, environmental,
    waste management and statutory compliance
    requirements applicable to the mine.
    """


# ============================================================
# CREATE QUERIES
# ============================================================

queries = []

mine_rows = []

for _, row in latest.iterrows():

    queries.append(
        build_query(row)
    )

    mine_rows.append(row)


print(
    "\nBuilding query embeddings..."
)

query_embeddings = model.encode(
    queries,
    normalize_embeddings=True,
    show_progress_bar=True
)

query_embeddings = np.asarray(
    query_embeddings,
    dtype="float32"
)

print(
    "Query embeddings:",
    query_embeddings.shape
)


# ============================================================
# FAISS RETRIEVAL
# ============================================================

print(
    "\nRunning initial FAISS retrieval..."
)

similarities, indices = index.search(
    query_embeddings,
    INITIAL_TOP_K
)


# ============================================================
# QUALITY CONTROL
# ============================================================

all_results = []
all_summary = []


for mine_idx, row in enumerate(
    mine_rows
):

    mine = row["subsidiary"]

    domain_scores = (
        calculate_domain_scores(row)
    )

    operational_risk = float(
        row.get(
            "overall_operational_risk",
            0
        )
    )

    risk_signal = min(
        max(
            operational_risk / 100,
            0
        ),
        1
    )

    candidates = []

    # --------------------------------------------------------
    # Rank all FAISS candidates
    # --------------------------------------------------------

    for position in range(
        INITIAL_TOP_K
    ):

        reg_idx = int(
            indices[
                mine_idx,
                position
            ]
        )

        if reg_idx < 0:
            continue

        semantic = float(
            similarities[
                mine_idx,
                position
            ]
        )

        if semantic < MIN_SIMILARITY:
            continue

        reg = reg_df.iloc[
            reg_idx
        ]

        domain = reg[
            "domain_clean"
        ]

        priority = float(
            reg[
                "priority_numeric"
            ]
        )

        domain_score = float(
            domain_scores.get(
                domain,
                0
            )
        )

        # ----------------------------------------------------
        # Combined ranking
        # ----------------------------------------------------

        score = (
            W_SEMANTIC * semantic
            + W_PRIORITY * priority
            + W_DOMAIN * domain_score
            + W_RISK * risk_signal
        )

        candidates.append(
            {
                "mine": mine,
                "reg_idx": reg_idx,
                "semantic_similarity": semantic,
                "priority_score": priority,
                "domain_relevance": domain_score,
                "risk_signal": risk_signal,
                "retrieval_score": score
            }
        )

    # --------------------------------------------------------
    # Sort by final score
    # --------------------------------------------------------

    candidates.sort(
        key=lambda x:
            x["retrieval_score"],
        reverse=True
    )

    # --------------------------------------------------------
    # Diversity-aware selection
    # --------------------------------------------------------

    selected = []

    document_count = {}
    domain_count = {}

    previous_texts = []

    for candidate in candidates:

        if len(selected) >= FINAL_TOP_K:
            break

        reg = reg_df.iloc[
            candidate["reg_idx"]
        ]

        source = clean(
            reg["source_document"]
        )

        domain = clean(
            reg["domain_clean"]
        )

        requirement = clean(
            reg["requirement"]
        )

        # ----------------------------------------------------
        # Document diversity
        # ----------------------------------------------------

        if document_count.get(
            source,
            0
        ) >= MAX_PER_DOCUMENT:

            continue

        # ----------------------------------------------------
        # Domain diversity
        # ----------------------------------------------------

        if domain_count.get(
            domain,
            0
        ) >= MAX_PER_DOMAIN:

            continue

        # ----------------------------------------------------
        # Duplicate removal
        # ----------------------------------------------------

        if near_duplicate(
            requirement,
            previous_texts
        ):

            continue

        # ----------------------------------------------------
        # Add result
        # ----------------------------------------------------

        result = {
            "mine": mine,
            "rank": len(selected) + 1,

            "normalized_requirement_id":
                reg[
                    "normalized_requirement_id"
                ],

            "requirement_id":
                reg[
                    "requirement_id"
                ],

            "source_document":
                source,

            "page_number":
                reg[
                    "page_number"
                ],

            "section_reference":
                clean(
                    reg[
                        "section_reference"
                    ]
                ),

            "regulatory_domain":
                domain,

            "requirement":
                requirement,

            "required_action":
                clean(
                    reg[
                        "required_action"
                    ]
                ),

            "responsible_party":
                clean(
                    reg[
                        "responsible_party"
                    ]
                ),

            "frequency":
                clean(
                    reg[
                        "normalized_frequency"
                    ]
                ),

            "regulatory_priority":
                clean(
                    reg[
                        "regulatory_priority"
                    ]
                ),

            "normalized_severity":
                clean(
                    reg[
                        "normalized_severity"
                    ]
                ),

            "semantic_similarity":
                candidate[
                    "semantic_similarity"
                ],

            "priority_score":
                candidate[
                    "priority_score"
                ],

            "domain_relevance":
                candidate[
                    "domain_relevance"
                ],

            "retrieval_score":
                candidate[
                    "retrieval_score"
                ]
        }

        selected.append(
            result
        )

        document_count[
            source
        ] = document_count.get(
            source,
            0
        ) + 1

        domain_count[
            domain
        ] = domain_count.get(
            domain,
            0
        ) + 1

        previous_texts.append(
            tokenize(requirement)
        )

    # --------------------------------------------------------
    # Mine summary
    # --------------------------------------------------------

    high_count = sum(
        r["regulatory_priority"].upper()
        == "HIGH"
        for r in selected
    )

    medium_count = sum(
        r["regulatory_priority"].upper()
        == "MEDIUM"
        for r in selected
    )

    unique_documents = len(
        set(
            r["source_document"]
            for r in selected
        )
    )

    unique_domains = len(
        set(
            r["regulatory_domain"]
            for r in selected
        )
    )

    average_similarity = (
        np.mean(
            [
                r["semantic_similarity"]
                for r in selected
            ]
        )
        if selected
        else 0
    )

    all_results.extend(
        selected
    )

    all_summary.append(
        {
            "mine": mine,
            "retrieved_count": len(
                selected
            ),
            "high_priority_count":
                int(high_count),

            "medium_priority_count":
                int(medium_count),

            "unique_source_documents":
                unique_documents,

            "unique_domains":
                unique_domains,

            "average_semantic_similarity":
                average_similarity
        }
    )


# ============================================================
# SAVE
# ============================================================

results_df = pd.DataFrame(
    all_results
)

summary_df = pd.DataFrame(
    all_summary
)


results_path = os.path.join(
    OUTPUT_DIR,
    "regulatory_retrieval_qc.csv"
)

summary_path = os.path.join(
    OUTPUT_DIR,
    "regulatory_retrieval_qc_summary.csv"
)


results_df.to_csv(
    results_path,
    index=False
)

summary_df.to_csv(
    summary_path,
    index=False
)


# ============================================================
# METRICS
# ============================================================

metrics = {

    "total_results":
        len(results_df),

    "mines":
        results_df["mine"].nunique()
        if not results_df.empty
        else 0,

    "unique_source_documents":
        results_df[
            "source_document"
        ].nunique()
        if not results_df.empty
        else 0,

    "unique_domains":
        results_df[
            "regulatory_domain"
        ].nunique()
        if not results_df.empty
        else 0,

    "high_priority_results":
        (
            results_df[
                "regulatory_priority"
            ]
            .str.upper()
            .eq("HIGH")
            .sum()
        )
        if not results_df.empty
        else 0,

    "medium_priority_results":
        (
            results_df[
                "regulatory_priority"
            ]
            .str.upper()
            .eq("MEDIUM")
            .sum()
        )
        if not results_df.empty
        else 0,

    "average_similarity":
        (
            results_df[
                "semantic_similarity"
            ].mean()
        )
        if not results_df.empty
        else 0
}


metrics_path = os.path.join(
    OUTPUT_DIR,
    "regulatory_retrieval_qc_metrics.csv"
)


pd.DataFrame(
    [metrics]
).to_csv(
    metrics_path,
    index=False
)


# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 70)
print("STEP 4.16.3 COMPLETE")
print("=" * 70)

print(
    "\nTotal results:",
    len(results_df)
)

print(
    "Unique documents:",
    metrics["unique_source_documents"]
)

print(
    "Unique domains:",
    metrics["unique_domains"]
)

print(
    "HIGH priority:",
    metrics["high_priority_results"]
)

print(
    "MEDIUM priority:",
    metrics["medium_priority_results"]
)

print(
    "Average similarity:",
    round(
        metrics["average_similarity"],
        4
    )
)

print("\nMine summary:")
print(
    summary_df.to_string(
        index=False
    )
)

print("\nSaved:")
print(results_path)
print(summary_path)
print(metrics_path)

print("\nSTATUS: PASS")