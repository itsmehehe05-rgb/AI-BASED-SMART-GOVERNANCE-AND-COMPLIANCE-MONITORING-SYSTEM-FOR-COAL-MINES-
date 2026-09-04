from pathlib import Path
import hashlib
import re
import pandas as pd
from pypdf import PdfReader


# ============================================================
# PHASE 9A — AI REGULATORY INTELLIGENCE ENGINE
# ============================================================

BASE_DIR = Path(r"D:\CoalMineAI")

REGULATIONS_DIR = BASE_DIR / "regulations"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "regulatory_requirements.csv"


# ============================================================
# CONFIGURATION
# ============================================================

MIN_REQUIREMENT_LENGTH = 30
MAX_REQUIREMENT_LENGTH = 1200


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def file_hash(path):
    """
    Calculate SHA256 hash so duplicate PDFs
    can be detected automatically.
    """

    sha256 = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            sha256.update(block)

    return sha256.hexdigest()


def clean_text(text):
    """
    Clean extracted PDF text.
    """

    if not text:
        return ""

    text = text.replace("\x00", " ")

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def classify_domain(text):
    """
    Classify regulatory requirement into
    broad governance domains.
    """

    text_lower = text.lower()

    environment_words = [
        "environment",
        "environmental",
        "emission",
        "effluent",
        "pollution",
        "air quality",
        "water quality",
        "environment clearance",
        "eia",
        "emp",
        "monitoring",
    ]

    waste_words = [
        "waste",
        "hazardous waste",
        "solid waste",
        "plastic waste",
        "e-waste",
        "battery waste",
        "bio-medical",
        "disposal",
        "recycling",
    ]

    safety_words = [
        "safety",
        "accident",
        "danger",
        "worker",
        "occupational",
        "protective equipment",
        "rescue",
        "emergency",
        "training",
        "explosive",
    ]

    mining_words = [
        "mine",
        "mining",
        "coal mine",
        "mineral",
        "mining lease",
        "production",
        "excavation",
        "quarry",
    ]

    compliance_words = [
        "compliance",
        "inspection",
        "report",
        "notice",
        "authority",
        "record",
        "register",
        "return",
        "certificate",
        "approval",
    ]

    if any(word in text_lower for word in environment_words):
        return "ENVIRONMENT"

    if any(word in text_lower for word in waste_words):
        return "WASTE"

    if any(word in text_lower for word in safety_words):
        return "SAFETY"

    if any(word in text_lower for word in mining_words):
        return "MINING"

    if any(word in text_lower for word in compliance_words):
        return "COMPLIANCE"

    return "GENERAL"


def detect_requirement_type(text):
    """
    Identify the type of requirement.
    """

    text_lower = text.lower()

    if any(x in text_lower for x in [
        "shall obtain",
        "shall possess",
        "shall have",
        "shall maintain",
        "must obtain",
        "must maintain"
    ]):
        return "MANDATORY"

    if any(x in text_lower for x in [
        "shall submit",
        "shall furnish",
        "shall report",
        "submit a report",
        "submit the report"
    ]):
        return "REPORTING"

    if any(x in text_lower for x in [
        "shall monitor",
        "monitoring shall",
        "monitoring of",
        "monitor regularly"
    ]):
        return "MONITORING"

    if any(x in text_lower for x in [
        "shall comply",
        "must comply",
        "compliance with",
        "in accordance with"
    ]):
        return "COMPLIANCE"

    if any(x in text_lower for x in [
        "shall ensure",
        "must ensure",
        "ensure that"
    ]):
        return "CONTROL"

    return "REGULATORY_PROVISION"


def detect_frequency(text):
    """
    Detect obvious reporting/monitoring frequency.
    """

    text_lower = text.lower()

    if "daily" in text_lower:
        return "DAILY"

    if "weekly" in text_lower:
        return "WEEKLY"

    if "fortnightly" in text_lower:
        return "FORTNIGHTLY"

    if "monthly" in text_lower:
        return "MONTHLY"

    if "quarterly" in text_lower:
        return "QUARTERLY"

    if "half-yearly" in text_lower or "half yearly" in text_lower:
        return "HALF-YEARLY"

    if "annual" in text_lower or "annually" in text_lower:
        return "ANNUAL"

    return "AS_SPECIFIED"


def detect_severity(text):
    """
    Initial rule-based severity estimation.

    This is a prototype classification and should
    not be treated as a legal determination.
    """

    text_lower = text.lower()

    high_words = [
        "fatal",
        "death",
        "serious accident",
        "danger",
        "hazard",
        "prohibited",
        "closure",
        "penalty",
        "cancel",
        "revocation",
        "environmental damage",
    ]

    medium_words = [
        "shall",
        "must",
        "non-compliance",
        "contravention",
        "notice",
        "inspection",
        "monitoring",
        "report",
    ]

    if any(word in text_lower for word in high_words):
        return "HIGH"

    if any(word in text_lower for word in medium_words):
        return "MEDIUM"

    return "LOW"


def extract_section(text):
    """
    Try to identify regulation/section/rule numbers.
    """

    patterns = [
        r"\bRegulation\s+[\w().-]+",
        r"\bRule\s+[\w().-]+",
        r"\bSection\s+[\w().-]+",
        r"\bParagraph\s+[\w().-]+",
        r"\bSchedule\s+[IVX0-9A-Za-z().-]+",
        r"\bAppendix\s+[IVX0-9A-Za-z().-]+",
    ]

    for pattern in patterns:

        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            return match.group(0)

    return ""


def looks_like_requirement(text):
    """
    Determine whether a sentence/paragraph looks like
    a regulatory requirement.

    This intentionally uses conservative rule-based
    extraction rather than pretending that every
    paragraph is a requirement.
    """

    text_lower = text.lower()

    requirement_phrases = [
        "shall",
        "must",
        "required to",
        "is required",
        "no person shall",
        "every owner",
        "every employer",
        "every occupier",
        "the project proponent shall",
        "the occupier shall",
        "the holder shall",
        "compliance with",
        "shall ensure",
        "shall maintain",
        "shall submit",
        "shall obtain",
        "shall provide",
    ]

    if not any(phrase in text_lower for phrase in requirement_phrases):
        return False

    if len(text) < MIN_REQUIREMENT_LENGTH:
        return False

    if len(text) > MAX_REQUIREMENT_LENGTH:
        return False

    return True


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf_requirements(pdf_path):
    """
    Extract candidate regulatory requirements
    page-by-page.
    """

    records = []

    try:
        reader = PdfReader(str(pdf_path))

    except Exception as e:

        print(f"ERROR reading {pdf_path.name}: {e}")

        return records

    total_pages = len(reader.pages)

    for page_number, page in enumerate(reader.pages, start=1):

        try:
            raw_text = page.extract_text() or ""

        except Exception:

            raw_text = ""

        text = clean_text(raw_text)

        if not text:
            continue

        # Split into sentence-like units.
        chunks = re.split(
            r"(?<=[.;:])\s+(?=[A-Z0-9(])",
            text
        )

        for chunk in chunks:

            chunk = clean_text(chunk)

            if not looks_like_requirement(chunk):
                continue

            section = extract_section(chunk)

            domain = classify_domain(chunk)

            requirement_type = detect_requirement_type(chunk)

            frequency = detect_frequency(chunk)

            severity = detect_severity(chunk)

            records.append({
                "source_document": pdf_path.name,
                "page_number": page_number,
                "total_pages": total_pages,
                "section_reference": section,
                "regulatory_domain": domain,
                "requirement_type": requirement_type,
                "requirement": chunk,
                "frequency": frequency,
                "initial_severity": severity,
            })

    return records


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PHASE 9A — AI REGULATORY INTELLIGENCE ENGINE")
    print("=" * 70)

    print("\n[1] Checking regulatory directory...")

    if not REGULATIONS_DIR.exists():

        print("ERROR:")
        print(f"Regulatory directory not found:")
        print(REGULATIONS_DIR)

        return

    pdf_files = sorted(
        REGULATIONS_DIR.glob("*.pdf")
    )

    print(f"PDF files found: {len(pdf_files)}")

    if not pdf_files:

        print("No PDF files found.")

        return

    # --------------------------------------------------------
    # Duplicate detection
    # --------------------------------------------------------

    print("\n[2] Detecting duplicate PDFs...")

    hashes = {}

    unique_files = []

    duplicate_files = []

    for pdf in pdf_files:

        try:
            digest = file_hash(pdf)

        except Exception as e:

            print(f"Could not hash {pdf.name}: {e}")

            continue

        if digest in hashes:

            duplicate_files.append(pdf)

        else:

            hashes[digest] = pdf.name
            unique_files.append(pdf)

    print(f"Unique PDFs     : {len(unique_files)}")
    print(f"Duplicates      : {len(duplicate_files)}")

    if duplicate_files:

        print("\nDuplicate files ignored:")

        for pdf in duplicate_files:
            print(f" - {pdf.name}")

    # --------------------------------------------------------
    # Extract requirements
    # --------------------------------------------------------

    print("\n[3] Extracting regulatory requirements...")

    all_records = []

    for index, pdf in enumerate(unique_files, start=1):

        print(
            f"\nProcessing {index}/{len(unique_files)}: "
            f"{pdf.name}"
        )

        records = extract_pdf_requirements(pdf)

        all_records.extend(records)

        print(
            f"Candidate requirements extracted: "
            f"{len(records)}"
        )

    print("\n[4] Combining extracted requirements...")

    if not all_records:

        print("No regulatory requirements were extracted.")

        print(
            "\nSome PDFs may be scanned/image-only documents. "
            "Those will require OCR in a later version."
        )

        return

    df = pd.DataFrame(all_records)

    # --------------------------------------------------------
    # Remove exact duplicates
    # --------------------------------------------------------

    print("\n[5] Removing duplicate requirements...")

    before = len(df)

    df = df.drop_duplicates(
        subset=[
            "source_document",
            "page_number",
            "requirement"
        ]
    )

    after = len(df)

    print(f"Removed duplicates: {before - after}")

    # --------------------------------------------------------
    # Create requirement ID
    # --------------------------------------------------------

    df.insert(
        0,
        "requirement_id",
        [
            f"REQ_{i:05d}"
            for i in range(1, len(df) + 1)
        ]
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    df = df.sort_values(
        by=[
            "regulatory_domain",
            "source_document",
            "page_number"
        ]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    print("\n[6] Saving regulatory knowledge base...")

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("REGULATORY INTELLIGENCE SUMMARY")
    print("=" * 70)

    print(f"PDFs discovered       : {len(pdf_files)}")
    print(f"Unique PDFs processed  : {len(unique_files)}")
    print(f"Duplicate PDFs ignored : {len(duplicate_files)}")

    print(
        f"Requirements extracted : {len(df)}"
    )

    print("\nRequirements by domain:")

    print(
        df["regulatory_domain"]
        .value_counts()
        .to_string()
    )

    print("\nRequirements by type:")

    print(
        df["requirement_type"]
        .value_counts()
        .to_string()
    )

    print("\nRequirements by severity:")

    print(
        df["initial_severity"]
        .value_counts()
        .to_string()
    )

    print("\nTop source documents:")

    print(
        df["source_document"]
        .value_counts()
        .head(10)
        .to_string()
    )

    print("\nExample extracted requirements:")

    preview_columns = [
        "requirement_id",
        "source_document",
        "page_number",
        "regulatory_domain",
        "requirement_type",
        "requirement"
    ]

    print(
        df[preview_columns]
        .head(10)
        .to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("PHASE 9A COMPLETE")
    print("=" * 70)

    print("\nRegulatory knowledge base saved to:")

    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()