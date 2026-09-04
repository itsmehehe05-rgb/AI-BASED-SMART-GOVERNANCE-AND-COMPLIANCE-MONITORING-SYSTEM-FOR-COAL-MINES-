"""
CoalMineAI - Step 4.16.1
Regulatory Embedding Generation

Purpose:
    Convert the normalized regulatory requirements into semantic
    embeddings and build a FAISS vector index.

Input:
    outputs/normalized_regulatory_requirements.csv

Outputs:
    outputs/v5/FINAL/regulatory_embeddings.npy
    outputs/v5/FINAL/regulatory_faiss.index
    outputs/v5/FINAL/regulatory_embedding_metadata.csv

This is a retrieval component only.
It does NOT create a chatbot or conversational AI.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import faiss

from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(r"D:\CoalMineAI")

INPUT_PATH = (
    BASE_DIR
    / "outputs"
    / "normalized_regulatory_requirements.csv"
)

FINAL_DIR = (
    BASE_DIR
    / "outputs"
    / "v5"
    / "FINAL"
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


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

BATCH_SIZE = 32

NORMALIZE_EMBEDDINGS = True


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(value):

    if pd.isna(value):
        return ""

    text = str(value)

    # Remove excessive whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


# ============================================================
# BUILD EMBEDDING TEXT
# ============================================================

def build_embedding_text(row):

    """
    Construct the semantic text used for embedding.

    Requirement text receives the main semantic information.
    Supporting regulatory metadata is included to improve retrieval.
    """

    requirement = clean_text(
        row.get(
            "requirement",
            ""
        )
    )

    regulatory_domain = clean_text(
        row.get(
            "regulatory_domain",
            ""
        )
    )

    requirement_type = clean_text(
        row.get(
            "requirement_type",
            ""
        )
    )

    required_action = clean_text(
        row.get(
            "required_action",
            ""
        )
    )

    responsible_party = clean_text(
        row.get(
            "responsible_party",
            ""
        )
    )

    frequency = clean_text(
        row.get(
            "normalized_frequency",
            ""
        )
    )

    evidence_required = clean_text(
        row.get(
            "evidence_required",
            ""
        )
    )

    severity = clean_text(
        row.get(
            "normalized_severity",
            ""
        )
    )

    mine_relevance = clean_text(
        row.get(
            "mine_relevance",
            ""
        )
    )

    actionability = clean_text(
        row.get(
            "actionability",
            ""
        )
    )

    text = (
        f"Regulatory Domain: {regulatory_domain}. "
        f"Requirement Type: {requirement_type}. "
        f"Statutory Requirement: {requirement}. "
        f"Required Action: {required_action}. "
        f"Responsible Party: {responsible_party}. "
        f"Frequency: {frequency}. "
        f"Evidence Required: {evidence_required}. "
        f"Severity: {severity}. "
        f"Mine Relevance: {mine_relevance}. "
        f"Actionability: {actionability}."
    )

    return text


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print(
        "CoalMineAI - STEP 4.16.1 "
        "REGULATORY EMBEDDING GENERATION"
    )
    print("=" * 80)

    # --------------------------------------------------------
    # CHECK INPUT
    # --------------------------------------------------------

    if not INPUT_PATH.exists():

        raise FileNotFoundError(
            f"\nRegulatory dataset not found:\n"
            f"{INPUT_PATH}"
        )

    FINAL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print(
        "\n[1] Loading normalized regulatory requirements..."
    )

    df = pd.read_csv(
        INPUT_PATH
    )

    print(
        "Dataset shape:",
        df.shape
    )

    # --------------------------------------------------------
    # VALIDATE REQUIRED COLUMNS
    # --------------------------------------------------------

    required_columns = [
        "normalized_requirement_id",
        "requirement_id",
        "source_document",
        "page_number",
        "regulatory_domain",
        "requirement_type",
        "requirement",
        "required_action",
        "responsible_party",
        "frequency",
        "normalized_frequency",
        "evidence_required",
        "normalized_severity",
        "mine_relevance",
        "actionability",
        "regulatory_priority_score",
        "regulatory_priority",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(
                missing_columns
            )
        )

    print(
        "Required columns: PASS"
    )

    # --------------------------------------------------------
    # REMOVE EMPTY REQUIREMENTS
    # --------------------------------------------------------

    before = len(df)

    df["requirement"] = (
        df["requirement"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df = df[
        df["requirement"] != ""
    ].copy()

    removed = before - len(df)

    print(
        "Empty requirements removed:",
        removed
    )

    print(
        "Requirements to embed:",
        len(df)
    )

    # --------------------------------------------------------
    # BUILD TEXT
    # --------------------------------------------------------

    print(
        "\n[2] Building embedding text..."
    )

    df["embedding_text"] = (
        df.apply(
            build_embedding_text,
            axis=1
        )
    )

    print(
        "Embedding text generated:"
        f" {len(df)}"
    )

    print(
        "\nExample embedding text:"
    )

    print(
        df.iloc[0][
            "embedding_text"
        ]
    )

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print(
        "\n[3] Loading embedding model..."
    )

    print(
        "Model:",
        MODEL_NAME
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    print(
        "Embedding model loaded."
    )

    # --------------------------------------------------------
    # GENERATE EMBEDDINGS
    # --------------------------------------------------------

    print(
        "\n[4] Generating embeddings..."
    )

    embedding_texts = (
        df[
            "embedding_text"
        ]
        .tolist()
    )

    embeddings = model.encode(
        embedding_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=(
            NORMALIZE_EMBEDDINGS
        ),
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    print(
        "Embedding matrix:",
        embeddings.shape
    )

    print(
        "Embedding dimension:",
        embeddings.shape[1]
    )

    # --------------------------------------------------------
    # SANITY CHECK
    # --------------------------------------------------------

    if not np.isfinite(
        embeddings
    ).all():

        raise ValueError(
            "Embedding matrix contains "
            "NaN or infinite values."
        )

    print(
        "Embedding numerical check: PASS"
    )

    # --------------------------------------------------------
    # BUILD FAISS INDEX
    # --------------------------------------------------------

    print(
        "\n[5] Building FAISS index..."
    )

    dimension = embeddings.shape[1]

    if NORMALIZE_EMBEDDINGS:

        # For normalized vectors:
        # inner product = cosine similarity
        index = faiss.IndexFlatIP(
            dimension
        )

    else:

        index = faiss.IndexFlatL2(
            dimension
        )

    index.add(
        embeddings
    )

    print(
        "FAISS index vectors:",
        index.ntotal
    )

    print(
        "FAISS dimension:",
        index.d
    )

    # --------------------------------------------------------
    # SAVE EMBEDDINGS
    # --------------------------------------------------------

    print(
        "\n[6] Saving embeddings..."
    )

    np.save(
        EMBEDDINGS_PATH,
        embeddings
    )

    # --------------------------------------------------------
    # SAVE FAISS
    # --------------------------------------------------------

    print(
        "Saving FAISS index..."
    )

    faiss.write_index(
        index,
        str(FAISS_PATH)
    )

    # --------------------------------------------------------
    # SAVE METADATA
    # --------------------------------------------------------

    print(
        "Saving regulatory metadata..."
    )

    metadata_columns = [
        "normalized_requirement_id",
        "requirement_id",
        "source_document",
        "page_number",
        "total_pages",
        "section_reference",
        "regulatory_domain",
        "requirement_type",
        "requirement",
        "required_action",
        "responsible_party",
        "frequency",
        "normalized_frequency",
        "evidence_required",
        "initial_severity",
        "normalized_severity",
        "mine_relevance",
        "actionability",
        "severity_score",
        "actionability_score",
        "mine_relevance_score",
        "regulatory_priority_score",
        "regulatory_priority",
        "generic_provision",
    ]

    metadata_columns = [
        column
        for column in metadata_columns
        if column in df.columns
    ]

    metadata = df[
        metadata_columns
    ].copy()

    metadata.insert(
        0,
        "embedding_index",
        np.arange(
            len(metadata)
        )
    )

    metadata.to_csv(
        METADATA_PATH,
        index=False
    )

    # --------------------------------------------------------
    # VERIFY SAVED INDEX
    # --------------------------------------------------------

    print(
        "\n[7] Verifying saved FAISS index..."
    )

    loaded_index = faiss.read_index(
        str(FAISS_PATH)
    )

    if (
        loaded_index.ntotal
        != len(df)
    ):

        raise ValueError(
            "FAISS verification failed: "
            "vector count mismatch."
        )

    print(
        "FAISS verification: PASS"
    )

    # --------------------------------------------------------
    # VERIFY EMBEDDINGS
    # --------------------------------------------------------

    loaded_embeddings = np.load(
        EMBEDDINGS_PATH
    )

    if (
        loaded_embeddings.shape
        != embeddings.shape
    ):

        raise ValueError(
            "Embedding verification failed."
        )

    print(
        "Embedding verification: PASS"
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "STEP 4.16.1 COMPLETED"
    )

    print(
        "=" * 80
    )

    print(
        f"\nRequirements embedded: {len(df)}"
    )

    print(
        f"Embedding dimension: {dimension}"
    )

    print(
        f"FAISS vectors: {loaded_index.ntotal}"
    )

    print(
        "\nGenerated files:"
    )

    print(
        f"1. {EMBEDDINGS_PATH}"
    )

    print(
        f"2. {FAISS_PATH}"
    )

    print(
        f"3. {METADATA_PATH}"
    )

    print(
        "\nRetrieval architecture:"
    )

    print(
        "Normalized Regulations"
        " → Embeddings"
        " → FAISS"
        " → Semantic Retrieval"
    )

    print(
        "\nConversational AI: NOT IMPLEMENTED"
    )

    print(
        "Predictive model: NOT MODIFIED"
    )

    print(
        "Regulatory source data: NOT MODIFIED"
    )


if __name__ == "__main__":
    main()