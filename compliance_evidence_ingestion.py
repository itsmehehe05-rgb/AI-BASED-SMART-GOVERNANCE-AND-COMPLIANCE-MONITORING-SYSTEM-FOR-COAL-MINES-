from pathlib import Path
import re
import hashlib
import pandas as pd
import numpy as np

from pypdf import PdfReader


# ============================================================
# PHASE 9F
# AI COMPLIANCE EVIDENCE INGESTION & REQUIREMENT MATCHING
# ============================================================

BASE_DIR = Path(r"D:\CoalMineAI")

OUTPUT_DIR = BASE_DIR / "outputs"

EVIDENCE_DIR = BASE_DIR / "compliance_evidence"

COMPLIANCE_FILE = (
    OUTPUT_DIR /
    "mine_specific_compliance_risk.csv"
)

EVIDENCE_OUTPUT = (
    OUTPUT_DIR /
    "compliance_evidence_inventory.csv"
)

MATCH_OUTPUT = (
    OUTPUT_DIR /
    "requirement_evidence_matching.csv"
)

MINE_SUMMARY_OUTPUT = (
    OUTPUT_DIR /
    "mine_evidence_status_summary.csv"
)

GAP_OUTPUT = (
    OUTPUT_DIR /
    "ai_compliance_gap_register.csv"
)


# ============================================================
# SETTINGS
# ============================================================

MIN_TEXT_LENGTH = 50

MATCH_THRESHOLD = 35

STRONG_MATCH_THRESHOLD = 55


# ============================================================
# TEXT UTILITIES
# ============================================================

def clean_text(value):

    if value is None:
        return ""

    text = str(value).lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def tokens(text):

    return set(
        clean_text(text).split()
    )


def sha256_file(path):

    sha = hashlib.sha256()

    with open(
        path,
        "rb"
    ) as file:

        while True:

            block = file.read(
                1024 * 1024
            )

            if not block:
                break

            sha.update(block)

    return sha.hexdigest()


# ============================================================
# MINE DETECTION
# ============================================================

SUBSIDIARIES = [
    "BCCL",
    "CCL",
    "ECL",
    "MCL",
    "NCL",
    "NEC",
    "SECL",
    "WCL"
]


def detect_mine(text, filename):

    combined = (
        str(filename)
        + " "
        + str(text)
    ).upper()

    matches = []

    for subsidiary in SUBSIDIARIES:

        if re.search(
            rf"\b{re.escape(subsidiary)}\b",
            combined
        ):

            matches.append(
                subsidiary
            )

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        return "MULTIPLE"

    return "UNKNOWN"


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf(path):

    pages = []

    try:

        reader = PdfReader(
            str(path)
        )

        for page_number, page in enumerate(
            reader.pages,
            start=1
        ):

            try:

                text = page.extract_text()

            except Exception:

                text = ""

            if text is None:
                text = ""

            pages.append({

                "page_number":
                    page_number,

                "text":
                    text

            })

    except Exception as error:

        print(
            f"  ERROR reading {path.name}: "
            f"{error}"
        )

    return pages


# ============================================================
# KEYWORD EXTRACTION
# ============================================================

STOPWORDS = {

    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "shall",
    "must",
    "where",
    "which",
    "into",
    "such",
    "their",
    "there",
    "under",
    "within",
    "upon",
    "being",
    "have",
    "has",
    "been",
    "were",
    "will",
    "would",
    "should",
    "may",
    "any",
    "all",
    "each",
    "other",
    "than",
    "also",
    "only",
    "required"

}


def meaningful_tokens(text):

    values = tokens(text)

    values = {

        word
        for word in values

        if len(word) >= 4
        and word not in STOPWORDS

    }

    return values


# ============================================================
# REQUIREMENT ↔ EVIDENCE SIMILARITY
# ============================================================

def similarity_score(
    requirement_text,
    evidence_text
):

    requirement_tokens = meaningful_tokens(
        requirement_text
    )

    evidence_tokens = meaningful_tokens(
        evidence_text
    )

    if not requirement_tokens:
        return 0.0

    overlap = (
        requirement_tokens
        &
        evidence_tokens
    )

    # --------------------------------------------------------
    # Recall-style relevance:
    # how much of the requirement vocabulary appears
    # in the evidence.
    # --------------------------------------------------------

    recall = (
        len(overlap)
        /
        max(
            len(requirement_tokens),
            1
        )
    )

    # --------------------------------------------------------
    # Jaccard component
    # --------------------------------------------------------

    union = (
        requirement_tokens
        |
        evidence_tokens
    )

    jaccard = (

        len(overlap)
        /
        max(
            len(union),
            1
        )

    )

    score = (

        recall * 70

        +

        jaccard * 30

    )

    return round(
        min(
            score,
            100
        ),
        2
    )


# ============================================================
# EVIDENCE TYPE
# ============================================================

def infer_evidence_type(
    required_action,
    requirement_text
):

    action = str(
        required_action
    ).upper()

    text = clean_text(
        requirement_text
    )

    if action == "REPORT":
        return "REGULATORY_REPORT"

    if action == "SUBMIT":
        return "SUBMISSION_RECORD"

    if action == "NOTIFY":
        return "NOTIFICATION_RECORD"

    if action == "MONITOR":
        return "MONITORING_RECORD"

    if action == "MAINTAIN":
        return "MAINTENANCE_OR_REGISTER"

    if action == "RECORD":
        return "REGISTER_OR_LOG"

    if action == "OBTAIN":
        return "PERMIT_OR_APPROVAL"

    if action == "PROHIBIT":
        return "INSPECTION_OR_CONTROL_RECORD"

    if action == "ENSURE":
        return "IMPLEMENTATION_EVIDENCE"

    if "inspection" in text:
        return "INSPECTION_RECORD"

    if "permit" in text:
        return "PERMIT_OR_APPROVAL"

    if "approval" in text:
        return "APPROVAL_DOCUMENT"

    if "monitoring" in text:
        return "MONITORING_RECORD"

    return "DOCUMENTARY_EVIDENCE"


# ============================================================
# EVIDENCE STATUS
# ============================================================

def classify_evidence(
    best_score,
    mine_match,
    evidence_found
):

    if not evidence_found:

        return "UNKNOWN"

    if mine_match == "UNKNOWN":

        return "REVIEW_REQUIRED"

    if mine_match == "MULTIPLE":

        return "REVIEW_REQUIRED"

    if best_score >= STRONG_MATCH_THRESHOLD:

        return "FOUND"

    if best_score >= MATCH_THRESHOLD:

        return "POSSIBLE_MATCH"

    return "NOT_FOUND"


# ============================================================
# GAP CLASSIFICATION
# ============================================================

def classify_gap(
    evidence_status,
    compliance_score,
    regulatory_priority
):

    compliance_score = float(
        compliance_score
    )

    regulatory_priority = float(
        regulatory_priority
    )

    if evidence_status == "FOUND":

        return "NO_EVIDENCE_GAP_DETECTED"

    if evidence_status == "POSSIBLE_MATCH":

        return "EVIDENCE_REVIEW_REQUIRED"

    if evidence_status == "REVIEW_REQUIRED":

        return "MANUAL_VERIFICATION_REQUIRED"

    if evidence_status == "NOT_FOUND":

        if (
            compliance_score >= 60
            or
            regulatory_priority >= 60
        ):

            return "HIGH_PRIORITY_EVIDENCE_GAP"

        return "EVIDENCE_GAP"

    return "EVIDENCE_STATUS_UNKNOWN"


# ============================================================
# MANAGEMENT ACTION
# ============================================================

def management_action(
    gap_type,
    evidence_type
):

    if gap_type == "HIGH_PRIORITY_EVIDENCE_GAP":

        return (
            "Immediately verify whether the required "
            "compliance evidence exists and escalate "
            "for management review if unavailable."
        )

    if gap_type == "EVIDENCE_GAP":

        return (
            "Locate or obtain the required compliance "
            "evidence and record its verification status."
        )

    if gap_type == "EVIDENCE_REVIEW_REQUIRED":

        return (
            "Review the identified document and confirm "
            "whether it satisfies the requirement."
        )

    if gap_type == "MANUAL_VERIFICATION_REQUIRED":

        return (
            "Manually verify mine applicability and "
            "evidence ownership before making a decision."
        )

    if gap_type == "NO_EVIDENCE_GAP_DETECTED":

        return (
            "Evidence candidate identified; verify "
            "document validity and continue monitoring."
        )

    return (
        "Maintain requirement in the compliance "
        "verification queue."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "PHASE 9F — AI COMPLIANCE EVIDENCE INGESTION "
        "& REQUIREMENT MATCHING"
    )

    print("=" * 70)

    # ========================================================
    # [1] CHECK EVIDENCE DIRECTORY
    # ========================================================

    print(
        "\n[1] Checking compliance evidence directory..."
    )

    EVIDENCE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    pdf_files = sorted(
        EVIDENCE_DIR.glob(
            "*.pdf"
        )
    )

    print(
        f"PDF evidence files found: "
        f"{len(pdf_files)}"
    )

    if not pdf_files:

        print(
            "\nNo evidence PDFs were found."
        )

        print(
            "Place compliance/inspection/reporting "
            "PDFs inside:"
        )

        print(
            EVIDENCE_DIR
        )

        print(
            "\nPhase 9F cannot perform evidence matching "
            "until documents are provided."
        )

        return

    # ========================================================
    # [2] LOAD 9D DATA
    # ========================================================

    print(
        "\n[2] Loading mine-specific compliance analysis..."
    )

    if not COMPLIANCE_FILE.exists():

        print(
            "ERROR: Missing:"
        )

        print(
            COMPLIANCE_FILE
        )

        return

    compliance = pd.read_csv(
        COMPLIANCE_FILE,
        encoding="utf-8-sig"
    )

    print(
        f"Compliance records: "
        f"{len(compliance)}"
    )

    required_columns = [

        "subsidiary",
        "normalized_requirement_id",
        "source_document",
        "page_number",
        "regulatory_domain",
        "requirement",
        "required_action",
        "regulatory_priority_score",
        "mine_specific_compliance_score",
        "overall_operational_risk",
        "governance_score"

    ]

    missing = [

        column
        for column in required_columns
        if column not in compliance.columns

    ]

    if missing:

        print(
            "ERROR: Missing columns:"
        )

        for column in missing:
            print(
                " -",
                column
            )

        return

    # ========================================================
    # [3] CLEAN 9D DATA
    # ========================================================

    print(
        "\n[3] Cleaning compliance records..."
    )

    compliance[
        "subsidiary"
    ] = (

        compliance[
            "subsidiary"
        ]
        .fillna("UNKNOWN")
        .astype(str)
        .str.upper()
        .str.strip()

    )

    compliance[
        "requirement"
    ] = (

        compliance[
            "requirement"
        ]
        .fillna("")
        .astype(str)

    )

    compliance[
        "regulatory_domain"
    ] = (

        compliance[
            "regulatory_domain"
        ]
        .fillna("GENERAL")
        .astype(str)
        .str.upper()
        .str.strip()

    )

    compliance[
        "required_action"
    ] = (

        compliance[
            "required_action"
        ]
        .fillna("OTHER")
        .astype(str)
        .str.upper()

    )

    for column in [

        "regulatory_priority_score",
        "mine_specific_compliance_score",
        "overall_operational_risk",
        "governance_score"

    ]:

        compliance[column] = pd.to_numeric(
            compliance[column],
            errors="coerce"
        ).fillna(0)

    # ========================================================
    # [4] INGEST PDF DOCUMENTS
    # ========================================================

    print(
        "\n[4] Ingesting evidence documents..."
    )

    evidence_records = []

    for index, pdf in enumerate(
        pdf_files,
        start=1
    ):

        print(
            f"  Reading {pdf.name} "
            f"({index}/{len(pdf_files)})..."
        )

        file_hash = sha256_file(
            pdf
        )

        pages = extract_pdf(
            pdf
        )

        full_text = "\n".join(
            page["text"]
            for page in pages
        )

        mine = detect_mine(
            full_text,
            pdf.name
        )

        if len(
            clean_text(full_text)
        ) < MIN_TEXT_LENGTH:

            extraction_status = "INSUFFICIENT_TEXT"

        else:

            extraction_status = "TEXT_EXTRACTED"

        evidence_records.append({

            "evidence_document":
                pdf.name,

            "evidence_path":
                str(pdf),

            "file_hash":
                file_hash,

            "detected_mine":
                mine,

            "page_count":
                len(pages),

            "text_length":
                len(full_text),

            "extraction_status":
                extraction_status,

            "full_text":
                full_text

        })

    evidence_inventory = pd.DataFrame(
        evidence_records
    )

    print(
        f"Documents ingested: "
        f"{len(evidence_inventory)}"
    )

    # ========================================================
    # [5] SAVE INVENTORY
    # ========================================================

    print(
        "\n[5] Saving evidence inventory..."
    )

    evidence_inventory[
        [
            "evidence_document",
            "evidence_path",
            "file_hash",
            "detected_mine",
            "page_count",
            "text_length",
            "extraction_status"
        ]
    ].to_csv(
        EVIDENCE_OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # [6] CREATE PAGE-LEVEL EVIDENCE
    # ========================================================

    print(
        "\n[6] Creating page-level evidence index..."
    )

    evidence_pages = []

    for record in evidence_records:

        pages = extract_pdf(
            Path(
                record["evidence_path"]
            )
        )

        for page in pages:

            text = page["text"]

            if not text:
                continue

            evidence_pages.append({

                "evidence_document":
                    record[
                        "evidence_document"
                    ],

                "detected_mine":
                    record[
                        "detected_mine"
                    ],

                "page_number":
                    page[
                        "page_number"
                    ],

                "page_text":
                    text

            })

    evidence_pages_df = pd.DataFrame(
        evidence_pages
    )

    print(
        f"Evidence pages indexed: "
        f"{len(evidence_pages_df)}"
    )

    # ========================================================
    # [7] MATCH REQUIREMENTS
    # ========================================================

    print(
        "\n[7] Matching requirements to evidence..."
    )

    result_rows = []

    total_requirements = len(
        compliance
    )

    for requirement_index, (
        _,
        requirement
    ) in enumerate(
        compliance.iterrows(),
        start=1
    ):

        if requirement_index % 100 == 0:

            print(
                f"  Requirements processed: "
                f"{requirement_index}/"
                f"{total_requirements}"
            )

        subsidiary = requirement[
            "subsidiary"
        ]

        requirement_text = requirement[
            "requirement"
        ]

        best_score = 0.0

        best_document = ""

        best_page = np.nan

        best_mine = "UNKNOWN"

        # ----------------------------------------------------
        # Select evidence relevant to this mine
        # ----------------------------------------------------

        candidate_pages = evidence_pages_df.copy()

        if not candidate_pages.empty:

            candidate_pages = candidate_pages[
                (
                    candidate_pages[
                        "detected_mine"
                    ]
                    .isin(
                        [
                            subsidiary,
                            "UNKNOWN",
                            "MULTIPLE"
                        ]
                    )
                )
            ]

        # ----------------------------------------------------
        # Compare requirement with each evidence page
        # ----------------------------------------------------

        for _, evidence in candidate_pages.iterrows():

            score = similarity_score(

                requirement_text,

                evidence[
                    "page_text"
                ]

            )

            if score > best_score:

                best_score = score

                best_document = (
                    evidence[
                        "evidence_document"
                    ]
                )

                best_page = (
                    evidence[
                        "page_number"
                    ]
                )

                best_mine = (
                    evidence[
                        "detected_mine"
                    ]
                )

        evidence_found = (
            len(candidate_pages) > 0
        )

        evidence_status = classify_evidence(

            best_score,

            best_mine,

            evidence_found

        )

        evidence_type = infer_evidence_type(

            requirement[
                "required_action"
            ],

            requirement_text

        )

        gap_type = classify_gap(

            evidence_status,

            requirement[
                "mine_specific_compliance_score"
            ],

            requirement[
                "regulatory_priority_score"
            ]

        )

        action = management_action(

            gap_type,

            evidence_type

        )

        result_rows.append({

            "subsidiary":
                subsidiary,

            "normalized_requirement_id":
                requirement[
                    "normalized_requirement_id"
                ],

            "regulatory_domain":
                requirement[
                    "regulatory_domain"
                ],

            "source_document":
                requirement[
                    "source_document"
                ],

            "regulation_page":
                requirement[
                    "page_number"
                ],

            "requirement":
                requirement_text,

            "required_action":
                requirement[
                    "required_action"
                ],

            "responsible_party":
                requirement.get(
                    "responsible_party",
                    "NOT_SPECIFIED"
                ),

            "regulatory_priority_score":
                requirement[
                    "regulatory_priority_score"
                ],

            "mine_specific_compliance_score":
                requirement[
                    "mine_specific_compliance_score"
                ],

            "evidence_expected":
                evidence_type,

            "matched_evidence_document":
                best_document,

            "matched_evidence_page":
                best_page,

            "evidence_detected_mine":
                best_mine,

            "evidence_match_score":
                round(
                    best_score,
                    2
                ),

            "evidence_status":
                evidence_status,

            "compliance_gap_type":
                gap_type,

            "recommended_management_action":
                action

        })

    # ========================================================
    # [8] CREATE MATCH DATASET
    # ========================================================

    print(
        "\n[8] Creating requirement-evidence dataset..."
    )

    result = pd.DataFrame(
        result_rows
    )

    result = result.sort_values(
        [
            "subsidiary",
            "evidence_match_score"
        ],
        ascending=[
            True,
            False
        ]
    ).reset_index(
        drop=True
    )

    # ========================================================
    # [9] RANK GAPS
    # ========================================================

    print(
        "\n[9] Ranking compliance evidence gaps..."
    )

    result[
        "gap_priority_score"
    ] = (

        result[
            "mine_specific_compliance_score"
        ] * 0.45

        +

        result[
            "regulatory_priority_score"
        ] * 0.35

        +

        result[
            "evidence_match_score"
        ] * 0.20

    )

    result[
        "gap_priority_score"
    ] = (

        result[
            "gap_priority_score"
        ]
        .clip(
            0,
            100
        )
        .round(2)

    )

    # ========================================================
    # [10] MINE SUMMARY
    # ========================================================

    print(
        "\n[10] Building mine evidence summaries..."
    )

    summary = (

        result
        .groupby(
            "subsidiary"
        )
        .agg(

            total_requirements=(
                "normalized_requirement_id",
                "count"
            ),

            evidence_found=(
                "evidence_status",
                lambda x:
                (
                    x ==
                    "FOUND"
                ).sum()
            ),

            possible_matches=(
                "evidence_status",
                lambda x:
                (
                    x ==
                    "POSSIBLE_MATCH"
                ).sum()
            ),

            not_found=(
                "evidence_status",
                lambda x:
                (
                    x ==
                    "NOT_FOUND"
                ).sum()
            ),

            unknown=(
                "evidence_status",
                lambda x:
                (
                    x ==
                    "UNKNOWN"
                ).sum()
            ),

            review_required=(
                "evidence_status",
                lambda x:
                (
                    x ==
                    "REVIEW_REQUIRED"
                ).sum()
            ),

            high_priority_gaps=(
                "compliance_gap_type",
                lambda x:
                (
                    x ==
                    "HIGH_PRIORITY_EVIDENCE_GAP"
                ).sum()
            ),

            average_gap_priority=(
                "gap_priority_score",
                "mean"
            ),

            maximum_gap_priority=(
                "gap_priority_score",
                "max"
            )

        )

        .reset_index()

    )

    # ========================================================
    # [11] GAP EXPOSURE
    # ========================================================

    print(
        "\n[11] Calculating mine evidence-gap exposure..."
    )

    summary[
        "evidence_gap_exposure_score"
    ] = (

        summary[
            "average_gap_priority"
        ] * 0.45

        +

        summary[
            "maximum_gap_priority"
        ] * 0.20

        +

        np.minimum(
            summary[
                "high_priority_gaps"
            ] * 8,
            100
        ) * 0.20

        +

        np.minimum(
            summary[
                "not_found"
            ] /
            np.maximum(
                summary[
                    "total_requirements"
                ],
                1
            )
            * 100,
            100
        ) * 0.15

    )

    summary[
        "evidence_gap_exposure_score"
    ] = (

        summary[
            "evidence_gap_exposure_score"
        ]
        .clip(
            0,
            100
        )
        .round(2)

    )

    # ========================================================
    # [12] PRIORITY CLASSIFICATION
    # ========================================================

    summary[
        "management_priority"
    ] = (

        summary[
            "evidence_gap_exposure_score"
        ]
        .apply(

            lambda x:

            "IMMEDIATE_REVIEW"
            if x >= 70

            else

            "ENHANCED_REVIEW"
            if x >= 45

            else

            "ROUTINE_REVIEW"

        )

    )

    # ========================================================
    # [13] TOP GAPS
    # ========================================================

    print(
        "\n[13] Creating top compliance gap register..."
    )

    gaps = (

        result[
            result[
                "compliance_gap_type"
            ].isin(

                [
                    "HIGH_PRIORITY_EVIDENCE_GAP",
                    "EVIDENCE_GAP",
                    "EVIDENCE_REVIEW_REQUIRED",
                    "MANUAL_VERIFICATION_REQUIRED"

                ]

            )
        ]

        .sort_values(
            "gap_priority_score",
            ascending=False
        )

        .head(100)

        .copy()

    )

    # ========================================================
    # [14] SAVE
    # ========================================================

    print(
        "\n[14] Saving Phase 9F outputs..."
    )

    result.to_csv(
        MATCH_OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )

    summary.to_csv(
        MINE_SUMMARY_OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )

    gaps.to_csv(
        GAP_OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # DISPLAY
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "AI COMPLIANCE EVIDENCE STATUS"
    )

    print(
        "=" * 70
    )

    display_columns = [

        "subsidiary",
        "total_requirements",
        "evidence_found",
        "possible_matches",
        "not_found",
        "unknown",
        "high_priority_gaps",
        "evidence_gap_exposure_score",
        "management_priority"

    ]

    print(
        summary[
            display_columns
        ]
        .to_string(
            index=False
        )
    )

    # ========================================================
    # EVIDENCE DISTRIBUTION
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "EVIDENCE STATUS DISTRIBUTION"
    )

    print(
        "=" * 70
    )

    print(
        result[
            "evidence_status"
        ]
        .value_counts()
        .to_string()
    )

    # ========================================================
    # GAP DISTRIBUTION
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "COMPLIANCE GAP DISTRIBUTION"
    )

    print(
        "=" * 70
    )

    print(
        result[
            "compliance_gap_type"
        ]
        .value_counts()
        .to_string()
    )

    # ========================================================
    # TOP GAPS
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "TOP AI-IDENTIFIED COMPLIANCE VERIFICATION GAPS"
    )

    print(
        "=" * 70
    )

    for _, row in gaps.head(10).iterrows():

        print(
            f"\n{row['subsidiary']} | "
            f"{row['normalized_requirement_id']}"
        )

        print(
            "Domain       : "
            + str(
                row[
                    "regulatory_domain"
                ]
            )
        )

        print(
            "Gap type     : "
            + str(
                row[
                    "compliance_gap_type"
                ]
            )
        )

        print(
            "Evidence     : "
            + str(
                row[
                    "evidence_status"
                ]
            )
        )

        print(
            "Match score  : "
            + f"{row['evidence_match_score']:.2f}"
        )

        print(
            "Priority     : "
            + f"{row['gap_priority_score']:.2f}"
        )

        print(
            "Document     : "
            + str(
                row[
                    "matched_evidence_document"
                ]
            )
        )

        print(
            "Action       : "
            + str(
                row[
                    "recommended_management_action"
                ]
            )
        )

    # ========================================================
    # OUTPUTS
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "PHASE 9F COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nEvidence inventory:"
    )

    print(
        EVIDENCE_OUTPUT
    )

    print(
        "\nRequirement-evidence matching:"
    )

    print(
        MATCH_OUTPUT
    )

    print(
        "\nMine evidence summary:"
    )

    print(
        MINE_SUMMARY_OUTPUT
    )

    print(
        "\nAI compliance gap register:"
    )

    print(
        GAP_OUTPUT
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()