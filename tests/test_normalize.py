"""Tests for text normalization."""

from greek_stylometer.preprocessing.normalize import (
    iota_subscript_to_adscript,
    strip_accents,
    strip_accents_and_lowercase,
    strip_editorial_punctuation,
    strip_section_numbers,
)


def test_strip_accents():
    # ά (alpha with oxia) → α
    assert strip_accents("ά") == "α"
    # ῶ (omega with perispomeni) → ω
    assert strip_accents("ῶ") == "ω"
    # Combined: τοῦ → του
    assert strip_accents("τοῦ") == "του"


def test_strip_accents_and_lowercase():
    assert strip_accents_and_lowercase("Τοῦ") == "του"
    assert strip_accents_and_lowercase("ΓΑΛΗΝΟΣ") == "γαληνος"


def test_strip_section_numbers():
    assert strip_section_numbers("[1]").strip() == ""
    assert strip_section_numbers("[12 3]").strip() == ""
    assert strip_section_numbers("text 1. more").strip() == "text   more"
    # Preserve non-numeric text
    assert "λόγος" in strip_section_numbers("λόγος [1] τοῦ")


def test_strip_editorial_punctuation():
    # Strips commas, quotes, brackets, dashes
    assert strip_editorial_punctuation('τοῦ, "καὶ" (ἐν)') == "τοῦ καὶ ἐν"
    assert strip_editorial_punctuation("λόγος — τοῦ") == "λόγος τοῦ"
    assert strip_editorial_punctuation("«φησίν»") == "φησίν"


def test_iota_subscript_to_adscript():
    assert iota_subscript_to_adscript("ᾳ") == "αι"
    assert iota_subscript_to_adscript("ῃ") == "ηι"
    assert iota_subscript_to_adscript("ῳ") == "ωι"


def test_iota_adscript_with_accents():
    # Accented forms: ᾷ (alpha with perispomeni and iota subscript)
    result = iota_subscript_to_adscript("ᾷ")
    assert "ι" in result
    assert "\u0345" not in result


def test_iota_adscript_no_change():
    # Text without iota subscripts passes through unchanged
    assert iota_subscript_to_adscript("λόγος τοῦ") == "λόγος τοῦ"


def test_strip_editorial_preserves_sentence_punctuation():
    # Period, raised dot, Greek question mark preserved
    assert strip_editorial_punctuation("λόγος. τοῦ") == "λόγος. τοῦ"
    assert strip_editorial_punctuation("λόγος· τοῦ") == "λόγος· τοῦ"
    assert strip_editorial_punctuation("τί ἐστιν;") == "τί ἐστιν;"
