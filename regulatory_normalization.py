from pathlib import Path
import re
import hashlib
import pandas as pd


# ============================================================
# PHASE 9A.1 — REGULATORY REQUIREMENT NORMALIZATION
# ============================================================

BASE_DIR = Path(r"D:\CoalMineAI")

INPUT_FILE = BASE_DIR / "outputs" / "regulatory_requirements.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "normalized_regulatory_requirements.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text)
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = text.replace("\t", " ")

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# REQUIREMENT FINGERPRINT
# ============================================================

def requirement_hash(text):
    text = clean_text(text).lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


# ============================================================
# ACTION DETECTION
# ============================================================

def detect_action(text):

    text_lower = text.lower()

    action_patterns = {

        "SUBMIT": [
            "shall submit",
            "must submit",
            "submit the",
            "submit a"
        ],

        "REPORT": [
            "shall report",
            "must report",
            "report to"
        ],

        "MONITOR": [
            "shall monitor",
            "must monitor",
            "monitoring of",
            "monitor regularly"
        ],

        "MAINTAIN": [
            "shall maintain",
            "must maintain",
            "maintain records",
            "maintain a record"
        ],

        "RECORD": [
            "shall record",
            "must record",
            "record the"
        ],

        "OBTAIN": [
            "shall obtain",
            "must obtain",
            "obtain the",
            "obtain a"
        ],

        "PROVIDE": [
            "shall provide",
            "must provide",
            "provide adequate"
        ],

        "ENSURE": [
            "shall ensure",
            "must ensure",
            "ensure that"
        ],

        "NOTIFY": [
            "shall notify",
            "must notify",
            "notify the"
        ],

        "COMPLY": [
            "shall comply",
            "must comply",
            "comply with"
        ],

        "PROHIBIT": [
            "shall not",
            "must not",
            "prohibited"
        ]
    }

    for action, patterns in action_patterns.items():

        for pattern in patterns:

            if pattern in text_lower:
                return action

    return "OTHER"


# ============================================================
# RESPONSIBLE PARTY
# ============================================================

def detect_responsible_party(text):

    text_lower = text.lower()

    if "project proponent" in text_lower:
        return "PROJECT_PROPONENT"

    if "project proposer" in text_lower:
        return "PROJECT_PROPONENT"

    if "mine manager" in text_lower:
        return "MINE_MANAGER"

    if "manager" in text_lower:
        return "MINE_MANAGER"

    if "mine owner" in text_lower:
        return "OWNER"

    if "owner" in text_lower:
        return "OWNER"

    if "occupier" in text_lower:
        return "OCCUPIER"

    if "employer" in text_lower:
        return "EMPLOYER"

    if "contractor" in text_lower:
        return "CONTRACTOR"

    if "competent authority" in text_lower:
        return "AUTHORITY"

    if "authority" in text_lower:
        return "AUTHORITY"

    if "central government" in text_lower:
        return "GOVERNMENT"

    if "state government" in text_lower:
        return "GOVERNMENT"

    if "government" in text_lower:
        return "GOVERNMENT"

    return "NOT_SPECIFIED"


# ============================================================
# FREQUENCY / DEADLINE
# ============================================================

def detect_frequency(text):

    text_lower = text.lower()

    frequency_patterns = {

        "DAILY": [
            "daily",
            "each day",
            "every day"
        ],

        "WEEKLY": [
            "weekly",
            "each week"
        ],

        "FORTNIGHTLY": [
            "fortnightly"
        ],

        "MONTHLY": [
            "monthly",
            "each month"
        ],

        "QUARTERLY": [
            "quarterly"
        ],

        "HALF_YEARLY": [
            "half-yearly",
            "half yearly",
            "six monthly"
        ],

        "ANNUAL": [
            "annual",
            "annually",
            "yearly",
            "every year"
        ]
    }

    for frequency, patterns in frequency_patterns.items():

        for pattern in patterns:

            if pattern in text_lower:
                return frequency

    # Example:
    # within 15 days

    match = re.search(
        r"within\s+(\d+)\s+days?",
        text_lower
    )

    if match:
        return f"WITHIN_{match.group(1)}_DAYS"

    # Example:
    # within 3 months

    match = re.search(
        r"within\s+(\d+)\s+months?",
        text_lower
    )

    if match:
        return f"WITHIN_{match.group(1)}_MONTHS"

    return "AS_SPECIFIED"


# ============================================================
# EVIDENCE DETECTION
# ============================================================

def detect_evidence(text):

    text_lower = text.lower()

    evidence = []

    evidence_patterns = {

        "REPORT": [
            "report",
            "reporting"
        ],

        "REGISTER": [
            "register",
            "record"
        ],

        "CERTIFICATE": [
            "certificate",
            "certification"
        ],

        "PERMIT": [
            "permit",
            "permission"
        ],

        "APPROVAL": [
            "approval",
            "approved"
        ],

        "NOTICE": [
            "notice"
        ],

        "INSPECTION_RECORD": [
            "inspection",
            "inspection report"
        ],

        "MONITORING_DATA": [
            "monitoring",
            "monitoring data",
            "monitoring report"
        ],

        "LICENSE": [
            "licence",
            "license"
        ],

        "RETURN": [
            "return",
            "returns"
        ]
    }

    for evidence_type, patterns in evidence_patterns.items():

        for pattern in patterns:

            if pattern in text_lower:

                evidence.append(evidence_type)

                break

    if not evidence:
        return "NOT_SPECIFIED"

    return ";".join(sorted(set(evidence)))


# ============================================================
# GENERIC PROVISION DETECTION
# ============================================================

def is_generic_provision(text):

    text_lower = text.lower()

    generic_patterns = [

        "the central government shall",
        "the central government may",
        "the state government may",
        "the officer shall",
        "the court may",
        "where any direction",
        "where the government",
        "for the purposes of this rule",
        "in this rule",
        "in these rules",
        "the provisions of this",
        "the authority may",
        "the authority shall"

    ]

    matches = sum(
        pattern in text_lower
        for pattern in generic_patterns
    )

    if len(text) < 100 and matches >= 1:
        return True

    if matches >= 2:
        return True

    return False


# ============================================================
# MINE RELEVANCE
# ============================================================

def calculate_mine_relevance(text, domain):

    text_lower = text.lower()

    mining_keywords = [

        "mine",
        "mining",
        "coal",
        "colliery",
        "excavation",
        "quarry",
        "miner",
        "workman",
        "worker",
        "explosive",
        "blasting",
        "overburden",
        "dump",
        "haul road",
        "production",
        "pit",
        "shaft",
        "drilling"

    ]

    count = 0

    for keyword in mining_keywords:

        if keyword in text_lower:
            count += 1

    if domain in [
        "SAFETY",
        "MINING",
        "ENVIRONMENT",
        "WASTE"
    ]:
        count += 2

    if count >= 4:
        return "HIGH"

    if count >= 2:
        return "MEDIUM"

    return "LOW"


# ============================================================
# SEVERITY
# ============================================================

def calculate_severity(text, requirement_type, initial_severity):

    text_lower = text.lower()

    high_indicators = [

        "fatal",
        "death",
        "serious accident",
        "major accident",
        "explosion",
        "prohibited",
        "closure",
        "revocation",
        "cancellation",
        "penalty",
        "dangerous",
        "hazardous substance",
        "environmental damage"

    ]

    if any(
        indicator in text_lower
        for indicator in high_indicators
    ):
        return "HIGH"

    if requirement_type in [
        "MANDATORY",
        "COMPLIANCE",
        "CONTROL",
        "REPORTING",
        "MONITORING"
    ]:
        return "MEDIUM"

    if (
        "shall" in text_lower
        or "must" in text_lower
        or "required" in text_lower
        or "non-compliance" in text_lower
        or "contravention" in text_lower
    ):
        return "MEDIUM"

    if initial_severity in [
        "HIGH",
        "MEDIUM",
        "LOW"
    ]:
        return initial_severity

    return "LOW"


# ============================================================
# ACTIONABILITY
# ============================================================

def calculate_actionability(
    action,
    responsible_party,
    evidence,
    frequency
):

    score = 0

    if action != "OTHER":
        score += 1

    if responsible_party != "NOT_SPECIFIED":
        score += 1

    if evidence != "NOT_SPECIFIED":
        score += 1

    if frequency != "AS_SPECIFIED":
        score += 1

    if score >= 3:
        return "HIGH"

    if score == 2:
        return "MEDIUM"

    return "LOW"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PHASE 9A.1 — REGULATORY REQUIREMENT NORMALIZATION")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    print("\n[1] Loading regulatory knowledge base...")

    if not INPUT_FILE.exists():

        print("\nERROR: Input file not found:")
        print(INPUT_FILE)

        return

    df = pd.read_csv(
        INPUT_FILE,
        encoding="utf-8-sig"
    )

    original_count = len(df)

    print(f"Rows loaded: {len(df)}")
    print(f"Columns loaded: {len(df.columns)}")

    # --------------------------------------------------------
    # REQUIRED COLUMNS
    # --------------------------------------------------------

    required_columns = [

        "requirement_id",
        "source_document",
        "page_number",
        "regulatory_domain",
        "requirement_type",
        "requirement",
        "frequency",
        "initial_severity"

    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:

        print("\nERROR — Missing columns:")

        for col in missing_columns:
            print(f" - {col}")

        return

    # --------------------------------------------------------
    # CLEAN TEXT
    # --------------------------------------------------------

    print("\n[2] Cleaning regulatory text...")

    df["requirement"] = (
        df["requirement"]
        .apply(clean_text)
    )

    df["regulatory_domain"] = (
        df["regulatory_domain"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    df["requirement_type"] = (
        df["requirement_type"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    df = df[
        df["requirement"].str.len() >= 40
    ].copy()

    print(
        f"Rows after text cleaning: {len(df)}"
    )

    # --------------------------------------------------------
    # GENERIC PROVISIONS
    # --------------------------------------------------------

    print("\n[3] Detecting generic provisions...")

    df["generic_provision"] = (
        df["requirement"]
        .apply(is_generic_provision)
    )

    print(
        "Generic provisions detected:",
        int(df["generic_provision"].sum())
    )

    # --------------------------------------------------------
    # ACTION
    # --------------------------------------------------------

    print("\n[4] Extracting regulatory actions...")

    df["required_action"] = (
        df["requirement"]
        .apply(detect_action)
    )

    # --------------------------------------------------------
    # RESPONSIBLE PARTY
    # --------------------------------------------------------

    print("\n[5] Detecting responsible entities...")

    df["responsible_party"] = (
        df["requirement"]
        .apply(detect_responsible_party)
    )

    # --------------------------------------------------------
    # FREQUENCY
    # --------------------------------------------------------

    print("\n[6] Detecting frequency / deadlines...")

    df["normalized_frequency"] = (
        df["requirement"]
        .apply(detect_frequency)
    )

    # --------------------------------------------------------
    # EVIDENCE
    # --------------------------------------------------------

    print("\n[7] Detecting compliance evidence...")

    df["evidence_required"] = (
        df["requirement"]
        .apply(detect_evidence)
    )

    # --------------------------------------------------------
    # SEVERITY
    # --------------------------------------------------------

    print("\n[8] Recalculating severity...")

    df["normalized_severity"] = df.apply(
        lambda row: calculate_severity(
            row["requirement"],
            row["requirement_type"],
            row["initial_severity"]
        ),
        axis=1
    )

    # --------------------------------------------------------
    # MINE RELEVANCE
    # --------------------------------------------------------

    print("\n[9] Calculating mine relevance...")

    df["mine_relevance"] = df.apply(
        lambda row: calculate_mine_relevance(
            row["requirement"],
            row["regulatory_domain"]
        ),
        axis=1
    )

    # --------------------------------------------------------
    # ACTIONABILITY
    # --------------------------------------------------------

    print("\n[10] Calculating requirement actionability...")

    df["actionability"] = df.apply(
        lambda row: calculate_actionability(
            row["required_action"],
            row["responsible_party"],
            row["evidence_required"],
            row["normalized_frequency"]
        ),
        axis=1
    )

    # --------------------------------------------------------
    # HASH
    # --------------------------------------------------------

    print("\n[11] Creating requirement fingerprints...")

    df["requirement_hash"] = (
        df["requirement"]
        .apply(requirement_hash)
    )

    # --------------------------------------------------------
    # DUPLICATES
    # --------------------------------------------------------

    before = len(df)

    df = df.drop_duplicates(
        subset=["requirement_hash"]
    ).copy()

    duplicates_removed = before - len(df)

    print(
        f"Exact duplicate requirements removed: "
        f"{duplicates_removed}"
    )

    # --------------------------------------------------------
    # FILTER GENERIC LOW-VALUE PROVISIONS
    # --------------------------------------------------------

    print("\n[12] Filtering low-value provisions...")

    before = len(df)

    df = df[
        ~(
            (df["generic_provision"] == True)
            &
            (df["actionability"] == "LOW")
            &
            (df["mine_relevance"] == "LOW")
        )
    ].copy()

    low_value_removed = before - len(df)

    print(
        f"Low-value generic provisions removed: "
        f"{low_value_removed}"
    )

    # --------------------------------------------------------
    # SCORE COMPONENTS
    # --------------------------------------------------------

    print("\n[13] Calculating regulatory priority score...")

    severity_score = {
        "LOW": 25,
        "MEDIUM": 55,
        "HIGH": 90
    }

    actionability_score = {
        "LOW": 20,
        "MEDIUM": 60,
        "HIGH": 90
    }

    relevance_score = {
        "LOW": 20,
        "MEDIUM": 60,
        "HIGH": 90
    }

    df["severity_score"] = (
        df["normalized_severity"]
        .map(severity_score)
        .fillna(25)
    )

    df["actionability_score"] = (
        df["actionability"]
        .map(actionability_score)
        .fillna(20)
    )

    df["mine_relevance_score"] = (
        df["mine_relevance"]
        .map(relevance_score)
        .fillna(20)
    )

    # Weighted score

    df["regulatory_priority_score"] = (

        0.40 * df["severity_score"]

        + 0.30 * df["actionability_score"]

        + 0.30 * df["mine_relevance_score"]

    ).round(2)

    # --------------------------------------------------------
    # PRIORITY LEVEL
    # --------------------------------------------------------

    def priority_level(score):

        if score >= 70:
            return "HIGH"

        if score >= 45:
            return "MEDIUM"

        return "LOW"

    df["regulatory_priority"] = (
        df["regulatory_priority_score"]
        .apply(priority_level)
    )

    # --------------------------------------------------------
    # NORMALIZED ID
    # --------------------------------------------------------

    df.insert(
        0,
        "normalized_requirement_id",
        [
            f"NREQ_{i:05d}"
            for i in range(1, len(df) + 1)
        ]
    )

    # --------------------------------------------------------
    # FINAL COLUMNS
    # --------------------------------------------------------

    final_columns = [

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

        "requirement_hash"

    ]

    final_columns = [
        col
        for col in final_columns
        if col in df.columns
    ]

    df = df[final_columns]

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    df = df.sort_values(
        by=[
            "regulatory_priority_score",
            "regulatory_domain"
        ],
        ascending=[
            False,
            True
        ]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    print("\n[14] Saving normalized regulatory knowledge base...")

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("NORMALIZED REGULATORY INTELLIGENCE SUMMARY")
    print("=" * 70)

    print(
        f"Original requirements : {original_count}"
    )

    print(
        f"Normalized requirements: {len(df)}"
    )

    print(
        f"Duplicates removed     : {duplicates_removed}"
    )

    print(
        f"Low-value removed      : {low_value_removed}"
    )

    print("\nDomains:")

    print(
        df["regulatory_domain"]
        .value_counts()
        .to_string()
    )

    print("\nNormalized severity:")

    print(
        df["normalized_severity"]
        .value_counts()
        .to_string()
    )

    print("\nMine relevance:")

    print(
        df["mine_relevance"]
        .value_counts()
        .to_string()
    )

    print("\nActionability:")

    print(
        df["actionability"]
        .value_counts()
        .to_string()
    )

    print("\nRegulatory priority:")

    print(
        df["regulatory_priority"]
        .value_counts()
        .to_string()
    )

    print("\nRequired actions:")

    print(
        df["required_action"]
        .value_counts()
        .head(15)
        .to_string()
    )

    print("\nResponsible parties:")

    print(
        df["responsible_party"]
        .value_counts()
        .head(15)
        .to_string()
    )

    # --------------------------------------------------------
    # TOP REQUIREMENTS
    # --------------------------------------------------------

    print("\nTop 10 prioritized requirements:")

    preview_columns = [

        "normalized_requirement_id",

        "source_document",

        "page_number",

        "regulatory_domain",

        "required_action",

        "responsible_party",

        "normalized_severity",

        "mine_relevance",

        "actionability",

        "regulatory_priority_score",

        "regulatory_priority"

    ]

    print(
        df[preview_columns]
        .head(10)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PHASE 9A.1 COMPLETE")
    print("=" * 70)

    print("\nNormalized regulatory knowledge base saved to:")

    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()