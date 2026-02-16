"""Tests for text normalization."""

from greek_stylometer.preprocessing.normalize import (
    strip_accents,
    strip_accents_and_lowercase,
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
