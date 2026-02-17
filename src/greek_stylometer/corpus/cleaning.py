"""XML cleaning pipeline for TEI EpiDoc texts.

Transforms raw TEI XML strings by stripping markup that is irrelevant
to textual content (line breaks, page breaks, editorial notes, etc.)
while preserving the text itself. Ported from author_reader.ipynb.
"""

import re
from collections.abc import Callable


def strip_bom(xml: str) -> str:
    """Remove UTF-8 BOM (U+FEFF) if present."""
    return xml.lstrip("\ufeff")


def remove_line_breaks(xml: str) -> str:
    """Remove <lb/> line break elements."""
    return re.sub(r"<lb.*?/>", " ", xml)


def remove_page_breaks(xml: str) -> str:
    """Remove <pb/> page break elements."""
    return re.sub(r"<pb.*?/>", " ", xml)


def remove_milestones(xml: str) -> str:
    """Remove <milestone/> elements (pagination markers, etc.)."""
    return re.sub(r"<milestone.*?/>", " ", xml)


def unwrap_supplied(xml: str) -> str:
    """Remove <supplied> tags but keep their text content."""
    xml = re.sub(r"<supplied.*?>", "", xml)
    return re.sub(r"</supplied>", "", xml)


def unwrap_add(xml: str) -> str:
    """Remove <add> tags but keep their text content."""
    xml = re.sub(r"<add.*?>", "", xml)
    return re.sub(r"</add>", "", xml)


def remove_del(xml: str) -> str:
    """Remove <del> elements entirely (deleted text)."""
    return re.sub(r"<del.*?>.+?</del>", "", xml)


def replace_quotes(xml: str) -> str:
    """Replace <quote> elements with guillemets."""
    xml = re.sub(r"<quote.*?>", " \u00ab", xml)
    return re.sub(r"</quote>", "\u00bb ", xml)


def remove_notes(xml: str) -> str:
    """Remove <note> elements and their content."""
    return re.sub(r"<note.+?/note>", " ", xml)


def replace_gaps(xml: str) -> str:
    """Replace <gap/> elements with [...] lacuna marker."""
    return re.sub(r"<gap.*?/>", "[...]", xml)


def unwrap_verse(xml: str) -> str:
    """Remove <l>, <lg> verse line/group elements."""
    xml = re.sub(r"<l .+?>", " ", xml)
    xml = xml.replace("<l>", " ")
    xml = xml.replace("</l>", " ")
    xml = xml.replace("<lg>", " ")
    xml = xml.replace("</lg>", " ")
    return xml


def normalize_whitespace(xml: str) -> str:
    """Collapse runs of whitespace (including newlines) to single spaces."""
    xml = xml.replace("\n", " ")
    return re.sub(r"\s\s+", " ", xml)


# The full pipeline in application order, matching author_reader.ipynb.
CLEANING_STEPS: list[tuple[str, Callable[[str], str]]] = [
    ("strip_bom", strip_bom),
    ("remove_line_breaks", remove_line_breaks),
    ("unwrap_supplied", unwrap_supplied),
    ("unwrap_add", unwrap_add),
    ("remove_del", remove_del),
    ("remove_page_breaks", remove_page_breaks),
    ("remove_milestones", remove_milestones),
    ("replace_quotes", replace_quotes),
    ("remove_notes", remove_notes),
    ("replace_gaps", replace_gaps),
    ("normalize_whitespace", normalize_whitespace),
    ("unwrap_verse", unwrap_verse),
]


def clean_xml(xml: str) -> str:
    """Apply the full cleaning pipeline to a TEI XML string."""
    for _name, step in CLEANING_STEPS:
        xml = step(xml)
    return xml
