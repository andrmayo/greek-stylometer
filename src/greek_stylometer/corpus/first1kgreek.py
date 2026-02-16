"""First1KGreek corpus reader.

Reads TEI EpiDoc XML files from the First1KGreek project directory
structure and yields Passage records. The corpus is organized as:

    data/
      tlg{author_id}/
        __cts__.xml          (author group metadata)
        tlg{work_id}/
          __cts__.xml        (work metadata)
          tlg{author}.tlg{work}.{edition}.xml  (text files)
"""

import logging
from collections.abc import Iterator
from pathlib import Path

from lxml import etree

from greek_stylometer.corpus.cleaning import clean_xml
from greek_stylometer.schemas import Passage

logger = logging.getLogger(__name__)

TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}
CTS_NS = {"ti": "http://chs.harvard.edu/xmlns/cts"}

# XPath expressions for extracting paragraph text, tried in order
# of decreasing specificity (3-level div nesting, then 2-level).
PARAGRAPH_XPATHS = [
    "//tei:TEI/tei:text/tei:body/tei:div/tei:div/tei:div/tei:p/text()",
    "//tei:TEI/tei:text/tei:body/tei:div/tei:div/tei:p/text()",
]


def read_author_name(cts_path: Path) -> str | None:
    """Extract the English author name from a group-level __cts__.xml."""
    try:
        tree = etree.parse(cts_path)
    except etree.XMLSyntaxError:
        logger.warning("Failed to parse %s", cts_path)
        return None

    # Prefer English name, fall back to Latin
    for lang in ("eng", "lat"):
        result = tree.xpath(
            f"//ti:groupname[@xml:lang='{lang}']/text()", namespaces=CTS_NS
        )
        if isinstance(result, list) and result:
            return str(result[0]).strip()
    return None


def read_work_title(cts_path: Path) -> str | None:
    """Extract the work title from a work-level __cts__.xml."""
    try:
        tree = etree.parse(cts_path)
    except etree.XMLSyntaxError:
        logger.warning("Failed to parse %s", cts_path)
        return None

    result = tree.xpath("//ti:title/text()", namespaces=CTS_NS)
    if isinstance(result, list) and result:
        return str(result[0]).strip()
    return None


def extract_passages_from_file(xml_path: Path) -> list[str]:
    """Parse a single TEI XML file and return its paragraph texts.

    Applies the cleaning pipeline, then extracts paragraphs via XPath.
    """
    try:
        tree = etree.parse(xml_path)
    except etree.XMLSyntaxError:
        logger.warning("Failed to parse %s", xml_path)
        return []

    xml_string = etree.tostring(tree, encoding="unicode")
    cleaned = clean_xml(xml_string)

    try:
        root = etree.fromstring(cleaned.encode("utf-8"))
    except etree.XMLSyntaxError:
        logger.warning("Failed to re-parse cleaned XML from %s", xml_path)
        return []

    for xpath in PARAGRAPH_XPATHS:
        result = root.xpath(xpath, namespaces=TEI_NS)
        if isinstance(result, list) and result:
            return [str(p) for p in result if str(p).strip()]

    logger.debug("No paragraphs found in %s", xml_path)
    return []


def _edition_from_filename(filename: str) -> str:
    """Extract edition identifier from a filename like 'tlg0057.tlg001.1st1K-grc1.xml'."""
    parts = filename.removesuffix(".xml").split(".")
    if len(parts) >= 3:
        return parts[2]
    return ""


class First1KGreekReader:
    """Reads passages from a First1KGreek author directory.

    Usage::

        reader = First1KGreekReader()
        for passage in reader.read(Path("First1KGreek/data/tlg0057")):
            print(passage.author, passage.text[:80])

    The source path should point to an author-level directory
    (e.g. ``data/tlg0057/``) containing work subdirectories.
    """

    def read(self, source: Path) -> Iterator[Passage]:
        """Yield Passage records from all works under the given author directory."""
        author_id = source.name  # e.g. "tlg0057"

        # Read author name from __cts__.xml
        author_cts = source / "__cts__.xml"
        author_name = read_author_name(author_cts) if author_cts.exists() else None
        if author_name is None:
            author_name = author_id
            logger.warning(
                "No __cts__.xml found for %s, using directory name", author_id
            )

        passage_idx = 0

        # Iterate over work subdirectories
        for work_dir in sorted(source.iterdir()):
            if not work_dir.is_dir():
                continue

            work_id = work_dir.name  # e.g. "tlg001"

            # Read work title for metadata
            work_cts = work_dir / "__cts__.xml"
            work_title = read_work_title(work_cts) if work_cts.exists() else None

            # Process each XML text file in the work directory
            for xml_file in sorted(work_dir.glob("*.xml")):
                if xml_file.name == "__cts__.xml":
                    continue

                edition = _edition_from_filename(xml_file.name)
                paragraphs = extract_passages_from_file(xml_file)

                for text in paragraphs:
                    yield Passage(
                        text=text,
                        author=author_name,
                        author_id=author_id,
                        work_id=work_id,
                        passage_idx=passage_idx,
                        source="first1kgreek",
                        metadata={
                            "edition": edition,
                            **({"work_title": work_title} if work_title else {}),
                            "file": xml_file.name,
                        },
                    )
                    passage_idx += 1


def ingest(
    input_paths: list[Path],
    output: Path,
    strip_section_numbers: bool = False,
) -> int:
    """Ingest First1KGreek author directories into a JSONL file.

    Args:
        input_paths: Author-level directories (e.g. ``data/tlg0057/``).
        output: Path to write the output JSONL file.
        strip_section_numbers: Remove bracketed section numbers from text.

    Returns:
        Number of passages written.
    """
    from greek_stylometer.corpus.jsonl import write_corpus
    from greek_stylometer.preprocessing.normalize import (
        strip_section_numbers as do_strip,
    )

    reader = First1KGreekReader()

    def passages():
        for path in input_paths:
            for passage in reader.read(path):
                if strip_section_numbers:
                    passage.text = do_strip(passage.text)
                yield passage

    return write_corpus(passages(), output)
