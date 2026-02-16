"""Text normalization for Ancient Greek.

Preprocessing steps that operate on plain text (after XML cleaning),
used before tokenization for models that require specific normalization.
"""

import re
import unicodedata


def strip_accents(text: str) -> str:
    """Remove combining diacritical marks (accents, breathings, etc.)."""
    nfd = unicodedata.normalize("NFD", text)
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


def strip_accents_and_lowercase(text: str) -> str:
    """Strip accents and lowercase — needed for bert-base-greek-uncased."""
    return strip_accents(text).lower()


def strip_section_numbers(text: str) -> str:
    """Remove bracketed or bare section numbers from passage text.

    Matches patterns like: [1], [12 3], 1., 12, etc.
    From author_reader.ipynb post-processing step.
    """
    return re.sub(r"\[*[0-9]+\s*[0-9]*\]*\.*\]*", " ", text)
