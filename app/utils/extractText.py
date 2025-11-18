import re
import pdfplumber
import docx


# ---------------------------------------------------------
# TEXT CLEANING (IMPROVED)
# ---------------------------------------------------------
def clean(text: str) -> str:
    # Remove control characters
    text = re.sub(r"[\x00-\x1F]", " ", text)

    # Fix hyphenated line breaks
    text = re.sub(r"-\s+", "", text)

    # Insert missing spaces between lowercase → uppercase (e.g., "theGENIUS")
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)

    # Insert missing spaces between uppercase sequences and words (e.g., GENIUSAct)
    text = re.sub(r"([A-Z])([A-Z][a-z])", r"\1 \2", text)

    # Collapse multiple whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ---------------------------------------------------------
# TXT EXTRACTION
# ---------------------------------------------------------
def extract_text_from_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return clean(f.read())


# ---------------------------------------------------------
# PDF EXTRACTION (IMPROVED)
# ---------------------------------------------------------
def extract_text_from_pdf(path: str) -> str:
    text_chunks = []

    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text() or ""

            # Remove common PDF header/footer noise:
            # Page numbers, “— End of Page —”, repeated court headers.
            extracted = re.sub(r"Page\s+\d+\s+of\s+\d+", " ", extracted)
            extracted = re.sub(r"^\s*\d+\s*$", " ", extracted)  # standalone numbers

            text_chunks.append(extracted)

    return clean(" ".join(text_chunks))


# ---------------------------------------------------------
# DOCX EXTRACTION
# ---------------------------------------------------------
def extract_text_from_docx(path: str) -> str:
    document = docx.Document(path)
    paragraphs = [p.text for p in document.paragraphs]
    return clean(" ".join(paragraphs))


# ---------------------------------------------------------
# MAIN FILE DISPATCHER
# ---------------------------------------------------------
def extract_text(file_path: str) -> str:
    """Auto-detect file type and return cleaned extracted text."""

    fp = file_path.lower()

    if fp.endswith(".txt"):
        return extract_text_from_txt(file_path)

    if fp.endswith(".pdf"):
        return extract_text_from_pdf(file_path)

    if fp.endswith(".docx"):
        return extract_text_from_docx(file_path)

    raise ValueError("Unsupported file format. Use TXT, PDF, or DOCX.")
