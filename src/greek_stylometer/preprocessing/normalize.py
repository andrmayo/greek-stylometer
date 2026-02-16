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


def iota_subscript_to_adscript(text: str) -> str:
    """Convert iota subscripts to adscripts (ᾳ → αι, ῃ → ηι, ῳ → ωι).

    Uses NFD decomposition to replace the combining iota subscript
    (U+0345) with a regular iota (ι), then recomposes. This handles
    all precomposed variants uniformly.

    Must be called before strip_accents, which would silently remove
    the subscript iota as a combining mark.
    """
    nfd = unicodedata.normalize("NFD", text)
    replaced = nfd.replace("\u0345", "\u03b9")
    return unicodedata.normalize("NFC", replaced)


_EDITORIAL_PUNCTUATION = re.compile(
    r"[,\"'\"\"''«»\-–—\(\)\[\]\{\}<>]"
)


def strip_editorial_punctuation(text: str) -> str:
    """Remove punctuation that reflects editorial choices, not authorial style.

    Strips commas, quotation marks, brackets, parentheses, and dashes.
    Preserves sentence-ending punctuation (period, raised dot ·,
    Greek question mark ;) since sentence structure is a legitimate
    stylometric signal.
    """
    stripped = _EDITORIAL_PUNCTUATION.sub("", text)
    return re.sub(r"  +", " ", stripped).strip()


def strip_section_numbers(text: str) -> str:
    """Remove bracketed or bare section numbers from passage text.

    Matches patterns like: [1], [12 3], 1., 12, etc.
    From author_reader.ipynb post-processing step.
    """
    return re.sub(r"\[*[0-9]+\s*[0-9]*\]*\.*\]*", " ", text)
