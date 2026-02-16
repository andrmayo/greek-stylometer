"""Tests for the XML cleaning pipeline."""

from greek_stylometer.corpus.cleaning import (
    clean_xml,
    normalize_whitespace,
    remove_del,
    remove_line_breaks,
    remove_notes,
    remove_page_breaks,
    replace_gaps,
    replace_quotes,
    unwrap_add,
    unwrap_supplied,
    unwrap_verse,
)


def test_remove_line_breaks():
    assert remove_line_breaks("foo<lb/>bar") == "foo bar"
    assert remove_line_breaks('foo<lb n="1"/>bar') == "foo bar"


def test_remove_page_breaks():
    assert remove_page_breaks('text<pb n="5"/>more') == "text more"


def test_unwrap_supplied():
    assert unwrap_supplied("<supplied>text</supplied>") == "text"
    assert unwrap_supplied('<supplied reason="lost">τοῦ</supplied>') == "τοῦ"


def test_unwrap_add():
    assert unwrap_add("<add>extra</add>") == "extra"


def test_remove_del():
    assert remove_del("<del>removed</del>") == ""
    assert remove_del('<del rend="erasure">gone</del>') == ""


def test_replace_quotes():
    result = replace_quotes("<quote>ἔπη</quote>")
    assert "\u00ab" in result
    assert "\u00bb" in result
    assert "ἔπη" in result


def test_remove_notes():
    assert remove_notes("<note>editorial comment</note>").strip() == ""
    assert remove_notes('<note type="footnote">fn</note>').strip() == ""


def test_replace_gaps():
    assert replace_gaps("<gap/>") == "[...]"
    assert replace_gaps('<gap reason="lost"/>') == "[...]"


def test_unwrap_verse():
    assert unwrap_verse("<l>verse</l>").strip() == "verse"
    assert unwrap_verse('<l n="1">verse</l>').strip() == "verse"
    assert unwrap_verse("<lg><l>a</l><l>b</l></lg>").strip() == "a  b"


def test_normalize_whitespace():
    assert normalize_whitespace("a  b   c") == "a b c"
    assert normalize_whitespace("a\nb") == "a b"


def test_clean_xml_full_pipeline():
    xml = (
        '<p>Εἰ<lb/>μὲν<pb n="1"/>'
        "<supplied>τοῦ</supplied>"
        "<del>bad</del>"
        '<note type="ed">note</note>'
        "<gap/>"
        "<quote>ἔπη</quote>"
        "<l>verse</l></p>"
    )
    result = clean_xml(xml)
    assert "Εἰ" in result
    assert "μὲν" in result
    assert "τοῦ" in result
    assert "bad" not in result
    assert "note" not in result
    assert "[...]" in result
    assert "\u00ab" in result
    assert "verse" in result
    # No residual tags from cleaned elements
    assert "<lb" not in result
    assert "<pb" not in result
    assert "<del" not in result
