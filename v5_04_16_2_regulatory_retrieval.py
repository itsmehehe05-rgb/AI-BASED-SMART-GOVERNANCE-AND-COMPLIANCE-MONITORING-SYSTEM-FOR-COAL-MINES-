"""
CoalMineAI - Step 4.16.2
Regulatory Retrieval & Ranking

Purpose:
    Retrieve the most relevant statutory/regulatory requirements
    for each mine using semantic embeddings + FAISS, followed
    by regulatory-priority re-ranking.

This is NOT Conversational AI.

Input:
    normalized_regulatory_requirements.csv
    regulatory_embeddings.npy
    regulatory_faiss.index
    regulatory_embedding_metadata.csv
    production_risk_analysis.csv
    early_warning_analysis.csv
    predictive_risk_escalation.csv

Output:
    mine_regulatory_rag_results.csv
    mine_regulatory_rag_summary.csv
"""

from pathlib import Path
import re
import warnings

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(r"D:\CoalMineAI")

OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
)

FINAL_DIR = (
    OUTPUT_DIR
    / "v5"
    / "FINAL"
)

REGULATORY_PATH = (
    OUTPUT_DIR
    / "normalized_regulatory_requirements.csv"
)

EMBEDDINGS_PATH = (
    FINAL_DIR
    / "regulatory_embeddings.npy"
)

FAISS_PATH = (
    FINAL_DIR
    / "regulatory_faiss.index"
)

METADATA_PATH = (
    FINAL_DIR
    / "regulatory_embedding_metadata.csv"
)

RISK_PATH = (
    OUTPUT_DIR
    / "production_risk_analysis.csv"
)

EARLY_WARNING_PATH = (
    OUTPUT_DIR
    / "early_warning_analysis.csv"
)

PREDICTIVE_PATH = (
    OUTPUT_DIR
    / "predictive_risk_escalation.csv"
)

RESULT_PATH = (
    FINAL_DIR
    / "mine_regulatory_rag_results.csv"
)

SUMMARY_PATH = (
    FINAL_DIR
    / "mine_regulatory_rag_summary.csv"
)


# ============================================================
# SETTINGS
# ============================================================

MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Retrieve more candidates initially,
# then rerank them.
INITIAL_TOP_K = 50

# Final regulations retained per mine.
FINAL_TOP_K = 15

BATCH_SIZE = 32


# ============================================================
# HELPERS
# ============================================================

def clean_text(value):

    if pd.isna(value):
        return ""

    text = str(value)

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def find_column(df, candidates):

    lower_map = {
        str(c).lower(): c
        for c in df.columns
    }

    for candidate in candidates:

        if candidate.lower() in lower_map:

            return lower_map[
                candidate.lower()
            ]

    return None


def numeric_value(value, default=0.0):

    try:

        if pd.isna(value):
            return default

        return float(value)

    except Exception:

        return default


def normalize_score(series):

    values = pd.to_numeric(
        series,
        errors="coerce"
    ).fillna(0.0)

    minimum = values.min()
    maximum = values.max()

    if maximum - minimum < 1e-12:

        return pd.Series(
            np.ones(
                len(values)
            ) * 0.5,
            index=series.index
        )

    return (
        (values - minimum)
        / (maximum - minimum)
    )


# ============================================================
# BUILD MINE QUERY TEXT
# ============================================================

def build_mine_query(row, columns):

    subsidiary = clean_text(
        row.get(
            columns["subsidiary"],
            ""
        )
    )

    operational_risk = clean_text(
        row.get(
            columns["operational_risk"],
            ""
        )
    )

    production_risk = clean_text(
        row.get(
            columns["production_risk"],
            ""
        )
    )

    warning_level = clean_text(
        row.get(
            columns["warning_level"],
            ""
        )
    )

    trajectory = clean_text(
        row.get(
            columns["trajectory"],
            ""
        )
    )

    equipment_risk = clean_text(
        row.get(
            columns["equipment_risk"],
            ""
        )
    )

    logistics_risk = clean_text(
        row.get(
            columns["logistics_risk"],
            ""
        )
    )

    weather_risk = clean_text(
        row.get(
            columns["weather_risk"],
            ""
        )
    )

    workforce_risk = clean_text(
        row.get(
            columns["workforce_risk"],
            ""
        )
    )

    predicted_level = clean_text(
        row.get(
            columns["predicted_level"],
            ""
        )
    )

    query = (
        f"Coal mine statutory compliance requirements. "
        f"Mine subsidiary: {subsidiary}. "
        f"Operational risk level: {operational_risk}. "
        f"Production risk level: {production_risk}. "
        f"Early warning level: {warning_level}. "
        f"Risk trajectory: {trajectory}. "
        f"Equipment risk: {equipment_risk}. "
        f"Logistics risk: {logistics_risk}. "
        f"Weather risk: {weather_risk}. "
        f"Workforce risk: {workforce_risk}. "
        f"Predicted escalation level: {predicted_level}. "
        f"Relevant mining safety, environmental, "
        f"waste-management, statutory reporting, "
        f"operational control and compliance requirements."
    )

    return query


# ============================================================
# DOMAIN RELEVANCE
# ============================================================

def calculate_domain_relevance(
    regulatory_domain,
    row
):

    """
    Give a modest domain relevance adjustment.

    This is intentionally not a hard filter because a mine's
    risk can be relevant to multiple regulatory domains.
    """

    domain = (
        clean_text(
            regulatory_domain
        )
        .upper()
    )

    scores = {
        "SAFETY": 0.0,
        "ENVIRONMENT": 0.0,
        "WASTE": 0.0,
        "COMPLIANCE": 0.0,
        "MINING": 0.0,
        "GENERAL": 0.0,
    }

    operational = clean_text(
        row.get(
            "operational_risk",
            ""
        )
    ).upper()

    warning = clean_text(
        row.get(
            "warning_level",
            ""
        )
    ).upper()

    trajectory = clean_text(
        row.get(
            "trajectory",
            ""
        )
    ).upper()

    # Safety is the principal domain for
    # operational-risk governance.
    if domain == "SAFETY":

        scores[domain] = 1.0

    elif domain == "MINING":

        scores[domain] = 0.8

    elif domain == "COMPLIANCE":

        scores[domain] = 0.7

    elif domain == "ENVIRONMENT":

        scores[domain] = 0.6

    elif domain == "WASTE":

        scores[domain] = 0.5

    else:

        scores[domain] = 0.3

    # Additional emphasis during worsening states.
    if (
        warning in {
            "EARLY_WARNING",
            "CRITICAL"
        }
        or trajectory in {
            "WORSENING",
            "RAPIDLY_WORSENING"
        }
    ):

        if domain == "SAFETY":
            scores[domain] += 0.15

        if domain == "COMPLIANCE":
            scores[domain] += 0.10

    return min(
        scores.get(
            domain,
            0.3
        ),
        1.0
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print(
        "CoalMineAI - STEP 4.16.2 "
        "REGULATORY RETRIEVAL & RANKING"
    )
    print("=" * 80)

    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    required_files = [
        REGULATORY_PATH,
        EMBEDDINGS_PATH,
        FAISS_PATH,
        METADATA_PATH,
        RISK_PATH,
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"\nRequired file not found:\n{path}"
            )

    FINAL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # LOAD REGULATORY METADATA
    # --------------------------------------------------------

    print(
        "\n[1] Loading regulatory metadata..."
    )

    metadata = pd.read_csv(
        METADATA_PATH
    )

    print(
        "Regulatory metadata:",
        metadata.shape
    )

    # --------------------------------------------------------
    # LOAD EMBEDDINGS
    # --------------------------------------------------------

    print(
        "\n[2] Loading regulatory embeddings..."
    )

    embeddings = np.load(
        EMBEDDINGS_PATH
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    print(
        "Embedding matrix:",
        embeddings.shape
    )

    # --------------------------------------------------------
    # LOAD FAISS
    # --------------------------------------------------------

    print(
        "\n[3] Loading FAISS index..."
    )

    index = faiss.read_index(
        str(FAISS_PATH)
    )

    print(
        "FAISS vectors:",
        index.ntotal
    )

    if index.ntotal != len(metadata):

        raise ValueError(
            "FAISS vector count does not "
            "match metadata count."
        )

    # --------------------------------------------------------
    # LOAD REGULATORY DATA
    # --------------------------------------------------------

    regulatory_df = pd.read_csv(
        REGULATORY_PATH
    )

    if len(regulatory_df) != len(metadata):

        print(
            "Warning: normalized regulatory dataset "
            "and embedding metadata have different "
            "row counts."
        )

    # --------------------------------------------------------
    # LOAD EMBEDDING MODEL
    # --------------------------------------------------------

    print(
        "\n[4] Loading embedding model..."
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    print(
        "Embedding model loaded."
    )

    # --------------------------------------------------------
    # LOAD RISK DATA
    # --------------------------------------------------------

    print(
        "\n[5] Loading mine risk data..."
    )

    risk_df = pd.read_csv(
        RISK_PATH
    )

    print(
        "Risk data:",
        risk_df.shape
    )

    # --------------------------------------------------------
    # OPTIONAL EARLY WARNING
    # --------------------------------------------------------

    if EARLY_WARNING_PATH.exists():

        early_df = pd.read_csv(
            EARLY_WARNING_PATH
        )

        print(
            "Early warning data:",
            early_df.shape
        )

    else:

        early_df = None

        print(
            "Early warning file not found."
            " Continuing with available risk data."
        )

    # --------------------------------------------------------
    # OPTIONAL PREDICTIVE RISK
    # --------------------------------------------------------

    if PREDICTIVE_PATH.exists():

        predictive_df = pd.read_csv(
            PREDICTIVE_PATH
        )

        print(
            "Predictive escalation data:",
            predictive_df.shape
        )

    else:

        predictive_df = None

        print(
            "Predictive escalation file not found."
            " Continuing without it."
        )

    # --------------------------------------------------------
    # IDENTIFY RISK COLUMNS
    # --------------------------------------------------------

    subsidiary_col = find_column(
        risk_df,
        [
            "subsidiary",
            "mine",
            "mine_name"
        ]
    )

    date_col = find_column(
        risk_df,
        [
            "record_date",
            "date"
        ]
    )

    operational_risk_col = find_column(
        risk_df,
        [
            "overall_operational_risk",
            "operational_risk"
        ]
    )

    production_risk_col = find_column(
        risk_df,
        [
            "production_risk_level",
            "production_risk",
            "production_risk_category"
        ]
    )

    equipment_risk_col = find_column(
        risk_df,
        [
            "equipment_risk"
        ]
    )

    logistics_risk_col = find_column(
        risk_df,
        [
            "logistics_risk"
        ]
    )

    weather_risk_col = find_column(
        risk_df,
        [
            "weather_risk"
        ]
    )

    workforce_risk_col = find_column(
        risk_df,
        [
            "workforce_risk"
        ]
    )

    warning_level_col = None
    trajectory_col = None

    if early_df is not None:

        warning_level_col = find_column(
            early_df,
            [
                "warning_level"
            ]
        )

        trajectory_col = find_column(
            early_df,
            [
                "trajectory"
            ]
        )

    predicted_level_col = None

    if predictive_df is not None:

        predicted_level_col = find_column(
            predictive_df,
            [
                "predicted_next_risk_level",
                "escalation_level"
            ]
        )

    if subsidiary_col is None:

        raise ValueError(
            "Could not identify subsidiary/mine column."
        )

    if operational_risk_col is None:

        raise ValueError(
            "Could not identify operational risk column."
        )

    # --------------------------------------------------------
    # BUILD MERGED MINE DATA
    # --------------------------------------------------------

    print(
        "\n[6] Preparing mine risk context..."
    )

    mine_df = risk_df.copy()

    # Rename core fields to standardized names.
    rename_map = {

        subsidiary_col:
            "subsidiary",

        operational_risk_col:
            "operational_risk",
    }

    if date_col:

        rename_map[
            date_col
        ] = "record_date"

    if production_risk_col:

        rename_map[
            production_risk_col
        ] = "production_risk"

    if equipment_risk_col:

        rename_map[
            equipment_risk_col
        ] = "equipment_risk"

    if logistics_risk_col:

        rename_map[
            logistics_risk_col
        ] = "logistics_risk"

    if weather_risk_col:

        rename_map[
            weather_risk_col
        ] = "weather_risk"

    if workforce_risk_col:

        rename_map[
            workforce_risk_col
        ] = "workforce_risk"

    mine_df = mine_df.rename(
        columns=rename_map
    )

    # --------------------------------------------------------
    # ADD EARLY WARNING DATA
    # --------------------------------------------------------

    if early_df is not None:

        ew_subsidiary = find_column(
            early_df,
            [
                "subsidiary",
                "mine",
                "mine_name"
            ]
        )

        ew_date = find_column(
            early_df,
            [
                "record_date",
                "date"
            ]
        )

        if (
            ew_subsidiary
            and
            ew_date
        ):

            ew_columns = [
                ew_subsidiary,
                ew_date
            ]

            if warning_level_col:

                ew_columns.append(
                    warning_level_col
                )

            if trajectory_col:

                ew_columns.append(
                    trajectory_col
                )

            ew = early_df[
                ew_columns
            ].copy()

            ew = ew.rename(
                columns={
                    ew_subsidiary:
                        "subsidiary",

                    ew_date:
                        "record_date",

                    warning_level_col:
                        "warning_level"
                    if warning_level_col
                    else "warning_level",

                    trajectory_col:
                        "trajectory"
                    if trajectory_col
                    else "trajectory",
                }
            )

            mine_df = mine_df.merge(
                ew,
                on=[
                    "subsidiary",
                    "record_date"
                ],
                how="left"
            )

    # --------------------------------------------------------
    # ADD PREDICTIVE ESCALATION
    # --------------------------------------------------------

    if predictive_df is not None:

        pr_subsidiary = find_column(
            predictive_df,
            [
                "subsidiary",
                "mine",
                "mine_name"
            ]
        )

        pr_date = find_column(
            predictive_df,
            [
                "record_date",
                "date"
            ]
        )

        if (
            pr_subsidiary
            and
            pr_date
        ):

            pr_columns = [
                pr_subsidiary,
                pr_date
            ]

            if predicted_level_col:

                pr_columns.append(
                    predicted_level_col
                )

            pr = predictive_df[
                pr_columns
            ].copy()

            pr = pr.rename(
                columns={
                    pr_subsidiary:
                        "subsidiary",

                    pr_date:
                        "record_date",

                    predicted_level_col:
                        "predicted_next_risk_level"
                    if predicted_level_col
                    else "predicted_next_risk_level",
                }
            )

            mine_df = mine_df.merge(
                pr,
                on=[
                    "subsidiary",
                    "record_date"
                ],
                how="left"
            )

    # --------------------------------------------------------
    # STANDARDIZE OPTIONAL COLUMNS
    # --------------------------------------------------------

    optional_columns = [
        "production_risk",
        "equipment_risk",
        "logistics_risk",
        "weather_risk",
        "workforce_risk",
        "warning_level",
        "trajectory",
        "predicted_next_risk_level",
    ]

    for column in optional_columns:

        if column not in mine_df.columns:

            mine_df[
                column
            ] = ""

    # --------------------------------------------------------
    # SELECT LATEST RECORD PER MINE
    # --------------------------------------------------------

    print(
        "\n[7] Selecting latest mine state..."
    )

    if "record_date" in mine_df.columns:

        mine_df[
            "record_date"
        ] = pd.to_datetime(
            mine_df[
                "record_date"
            ],
            errors="coerce"
        )

        latest_mines = (
            mine_df
            .sort_values(
                "record_date"
            )
            .groupby(
                "subsidiary",
                as_index=False
            )
            .tail(1)
            .copy()
        )

    else:

        latest_mines = (
            mine_df
            .groupby(
                "subsidiary",
                as_index=False
            )
            .tail(1)
            .copy()
        )

    latest_mines = (
        latest_mines
        .sort_values(
            "subsidiary"
        )
        .reset_index(
            drop=True
        )
    )

    print(
        "Mines:",
        len(latest_mines)
    )

    print(
        latest_mines[
            [
                "subsidiary",
                "operational_risk",
                "production_risk",
                "warning_level",
                "trajectory"
            ]
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # BUILD QUERY TEXT
    # --------------------------------------------------------

    print(
        "\n[8] Building mine regulatory queries..."
    )

    standardized_columns = {

        "subsidiary":
            "subsidiary",

        "operational_risk":
            "operational_risk",

        "production_risk":
            "production_risk",

        "warning_level":
            "warning_level",

        "trajectory":
            "trajectory",

        "equipment_risk":
            "equipment_risk",

        "logistics_risk":
            "logistics_risk",

        "weather_risk":
            "weather_risk",

        "workforce_risk":
            "workforce_risk",

        "predicted_level":
            "predicted_next_risk_level",
    }

    latest_mines[
        "regulatory_query"
    ] = latest_mines.apply(
        lambda row:
            build_mine_query(
                row,
                standardized_columns
            ),
        axis=1
    )

    # --------------------------------------------------------
    # QUERY EMBEDDINGS
    # --------------------------------------------------------

    print(
        "\n[9] Generating mine query embeddings..."
    )

    query_embeddings = model.encode(
        latest_mines[
            "regulatory_query"
        ].tolist(),
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    query_embeddings = np.asarray(
        query_embeddings,
        dtype=np.float32
    )

    print(
        "Query embedding matrix:",
        query_embeddings.shape
    )

    # --------------------------------------------------------
    # FAISS SEARCH
    # --------------------------------------------------------

    print(
        "\n[10] Performing semantic regulatory retrieval..."
    )

    similarity_scores, indices = (
        index.search(
            query_embeddings,
            INITIAL_TOP_K
        )
    )

    print(
        "Initial candidates per mine:",
        INITIAL_TOP_K
    )

    # --------------------------------------------------------
    # RANK RESULTS
    # --------------------------------------------------------

    print(
        "\n[11] Re-ranking regulatory requirements..."
    )

    results = []

    for mine_idx in range(
        len(latest_mines)
    ):

        mine_row = latest_mines.iloc[
            mine_idx
        ]

        mine_name = clean_text(
            mine_row[
                "subsidiary"
            ]
        )

        for candidate_rank in range(
            INITIAL_TOP_K
        ):

            regulatory_index = int(
                indices[
                    mine_idx,
                    candidate_rank
                ]
            )

            if regulatory_index < 0:
                continue

            similarity = float(
                similarity_scores[
                    mine_idx,
                    candidate_rank
                ]
            )

            if regulatory_index >= len(
                metadata
            ):

                continue

            reg = metadata.iloc[
                regulatory_index
            ]

            # ----------------------------------------------
            # Regulatory priority
            # ----------------------------------------------

            priority_score = numeric_value(
                reg.get(
                    "regulatory_priority_score",
                    0
                )
            )

            priority_normalized = (
                priority_score
                / 100.0
            )

            # ----------------------------------------------
            # Severity
            # ----------------------------------------------

            severity_score = numeric_value(
                reg.get(
                    "severity_score",
                    0
                )
            )

            severity_normalized = (
                severity_score
                / 100.0
            )

            # ----------------------------------------------
            # Mine relevance
            # ----------------------------------------------

            mine_relevance_score = numeric_value(
                reg.get(
                    "mine_relevance_score",
                    0
                )
            )

            mine_relevance_normalized = (
                mine_relevance_score
                / 100.0
            )

            # ----------------------------------------------
            # Actionability
            # ----------------------------------------------

            actionability_score = numeric_value(
                reg.get(
                    "actionability_score",
                    0
                )
            )

            actionability_normalized = (
                actionability_score
                / 100.0
            )

            # ----------------------------------------------
            # Domain relevance
            # ----------------------------------------------

            domain_relevance = (
                calculate_domain_relevance(
                    reg.get(
                        "regulatory_domain",
                        ""
                    ),
                    mine_row
                )
            )

            # ----------------------------------------------
            # Combined score
            # ----------------------------------------------

            final_score = (

                0.50
                * similarity

                +

                0.15
                * priority_normalized

                +

                0.15
                * mine_relevance_normalized

                +

                0.10
                * severity_normalized

                +

                0.05
                * actionability_normalized

                +

                0.05
                * domain_relevance
            )

            results.append({

                "mine":
                    mine_name,

                "record_date":
                    mine_row.get(
                        "record_date",
                        ""
                    ),

                "operational_risk":
                    mine_row.get(
                        "operational_risk",
                        ""
                    ),

                "production_risk":
                    mine_row.get(
                        "production_risk",
                        ""
                    ),

                "warning_level":
                    mine_row.get(
                        "warning_level",
                        ""
                    ),

                "trajectory":
                    mine_row.get(
                        "trajectory",
                        ""
                    ),

                "predicted_next_risk_level":
                    mine_row.get(
                        "predicted_next_risk_level",
                        ""
                    ),

                "regulatory_requirement_id":
                    reg.get(
                        "normalized_requirement_id",
                        ""
                    ),

                "requirement_id":
                    reg.get(
                        "requirement_id",
                        ""
                    ),

                "requirement":
                    reg.get(
                        "requirement",
                        ""
                    ),

                "regulatory_domain":
                    reg.get(
                        "regulatory_domain",
                        ""
                    ),

                "requirement_type":
                    reg.get(
                        "requirement_type",
                        ""
                    ),

                "required_action":
                    reg.get(
                        "required_action",
                        ""
                    ),

                "responsible_party":
                    reg.get(
                        "responsible_party",
                        ""
                    ),

                "frequency":
                    reg.get(
                        "frequency",
                        ""
                    ),

                "evidence_required":
                    reg.get(
                        "evidence_required",
                        ""
                    ),

                "normalized_severity":
                    reg.get(
                        "normalized_severity",
                        ""
                    ),

                "mine_relevance":
                    reg.get(
                        "mine_relevance",
                        ""
                    ),

                "actionability":
                    reg.get(
                        "actionability",
                        ""
                    ),

                "regulatory_priority":
                    reg.get(
                        "regulatory_priority",
                        ""
                    ),

                "regulatory_priority_score":
                    priority_score,

                "source_document":
                    reg.get(
                        "source_document",
                        ""
                    ),

                "page_number":
                    reg.get(
                        "page_number",
                        ""
                    ),

                "total_pages":
                    reg.get(
                        "total_pages",
                        ""
                    ),

                "section_reference":
                    reg.get(
                        "section_reference",
                        ""
                    ),

                "semantic_similarity":
                    similarity,

                "severity_score":
                    severity_score,

                "mine_relevance_score":
                    mine_relevance_score,

                "actionability_score":
                    actionability_score,

                "domain_relevance_score":
                    domain_relevance,

                "final_relevance_score":
                    final_score,

                "initial_semantic_rank":
                    candidate_rank + 1,
            })

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # FINAL RANKING
    # --------------------------------------------------------

    print(
        "\n[12] Creating final regulatory rankings..."
    )

    results_df = (
        results_df
        .sort_values(
            [
                "mine",
                "final_relevance_score",
                "regulatory_priority_score",
            ],
            ascending=[
                True,
                False,
                False,
            ]
        )
        .reset_index(
            drop=True
        )
    )

    results_df[
        "regulatory_rank"
    ] = (
        results_df
        .groupby(
            "mine"
        )
        .cumcount()
        + 1
    )

    final_results = results_df[
        results_df[
            "regulatory_rank"
        ]
        <= FINAL_TOP_K
    ].copy()

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary_rows = []

    for mine in sorted(
        final_results[
            "mine"
        ].unique()
    ):

        mine_results = (
            final_results[
                final_results[
                    "mine"
                ] == mine
            ]
        )

        domain_counts = (
            mine_results[
                "regulatory_domain"
            ]
            .value_counts()
        )

        dominant_domain = (
            domain_counts.index[0]
            if len(domain_counts)
            else ""
        )

        summary_rows.append({

            "mine":
                mine,

            "record_date":
                mine_results[
                    "record_date"
                ].iloc[0],

            "operational_risk":
                mine_results[
                    "operational_risk"
                ].iloc[0],

            "production_risk":
                mine_results[
                    "production_risk"
                ].iloc[0],

            "warning_level":
                mine_results[
                    "warning_level"
                ].iloc[0],

            "trajectory":
                mine_results[
                    "trajectory"
                ].iloc[0],

            "predicted_next_risk_level":
                mine_results[
                    "predicted_next_risk_level"
                ].iloc[0],

            "retrieved_requirements":
                len(mine_results),

            "high_priority_requirements":
                int(
                    (
                        mine_results[
                            "regulatory_priority"
                        ]
                        .astype(str)
                        .str.upper()
                        == "HIGH"
                    ).sum()
                ),

            "medium_priority_requirements":
                int(
                    (
                        mine_results[
                            "regulatory_priority"
                        ]
                        .astype(str)
                        .str.upper()
                        == "MEDIUM"
                    ).sum()
                ),

            "dominant_regulatory_domain":
                dominant_domain,

            "average_semantic_similarity":
                mine_results[
                    "semantic_similarity"
                ].mean(),

            "average_final_relevance":
                mine_results[
                    "final_relevance_score"
                ].mean(),

            "maximum_final_relevance":
                mine_results[
                    "final_relevance_score"
                ].max(),

            "maximum_regulatory_priority":
                mine_results[
                    "regulatory_priority_score"
                ].max(),
        })

    summary_df = pd.DataFrame(
        summary_rows
    )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    print(
        "\n[13] Saving results..."
    )

    final_results.to_csv(
        RESULT_PATH,
        index=False
    )

    summary_df.to_csv(
        SUMMARY_PATH,
        index=False
    )

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    print(
        "\nTop retrieved regulations:"
    )

    display_columns = [
        "mine",
        "regulatory_rank",
        "regulatory_domain",
        "required_action",
        "regulatory_priority",
        "semantic_similarity",
        "final_relevance_score",
        "source_document",
        "page_number",
    ]

    print(
        final_results[
            display_columns
        ]
        .head(40)
        .to_string(
            index=False
        )
    )

    print(
        "\nMine-level summary:"
    )

    print(
        summary_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # FINAL STATUS
    # --------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "STEP 4.16.2 COMPLETED"
    )

    print(
        "=" * 80
    )

    print(
        f"\nMines processed: "
        f"{len(latest_mines)}"
    )

    print(
        f"Initial candidates per mine: "
        f"{INITIAL_TOP_K}"
    )

    print(
        f"Final regulations per mine: "
        f"{FINAL_TOP_K}"
    )

    print(
        f"Final retrieved records: "
        f"{len(final_results)}"
    )

    print(
        "\nGenerated:"
    )

    print(
        f"1. {RESULT_PATH}"
    )

    print(
        f"2. {SUMMARY_PATH}"
    )

    print(
        "\nArchitecture:"
    )

    print(
        "Mine Risk Context"
        " → Query Embedding"
        " → FAISS Semantic Retrieval"
        " → Regulatory Priority Re-ranking"
        " → Top Statutory Requirements"
    )

    print(
        "\nConversational AI: NOT IMPLEMENTED"
    )

    print(
        "Predictive model: NOT MODIFIED"
    )

    print(
        "Regulatory dataset: NOT MODIFIED"
    )


if __name__ == "__main__":
    main()