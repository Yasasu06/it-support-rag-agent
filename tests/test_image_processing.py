"""
Fast unit tests for image_processing.py.

Pure logic only — validation rules and query merging. No real image decoding
and no API/vision calls (extract_text_from_image is exercised live in the
end-to-end step, not here).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from image_processing import (
    validate_image,
    build_combined_query,
    MAX_IMAGE_SIZE_MB,
)


# --- validate_image ---------------------------------------------------------
def test_rejects_unsupported_content_type():
    ok, err = validate_image(b"x" * 500, "application/pdf")
    assert ok is False
    assert "Unsupported image type" in err


def test_rejects_oversized_image():
    # Fake an oversized payload without allocating a real 10MB+ buffer:
    # bytes([0]) * N gives exactly N bytes.
    too_big = b"\x00" * int((MAX_IMAGE_SIZE_MB + 1) * 1024 * 1024)
    ok, err = validate_image(too_big, "image/png")
    assert ok is False
    assert "too large" in err.lower()


def test_rejects_near_empty_content():
    ok, err = validate_image(b"tiny", "image/png")
    assert ok is False
    assert "empty or corrupted" in err.lower()


def test_accepts_reasonable_png():
    ok, err = validate_image(b"\x89PNG\r\n" + b"\x00" * 2000, "image/png")
    assert ok is True
    assert err is None


def test_accepts_jpeg_and_webp():
    assert validate_image(b"\x00" * 2000, "image/jpeg")[0] is True
    assert validate_image(b"\x00" * 2000, "image/webp")[0] is True


# --- build_combined_query ---------------------------------------------------
def test_combines_typed_and_image_text():
    result = build_combined_query("VPN won't connect", "Error 0x0000011b on screen")
    assert "VPN won't connect" in result
    assert "Error 0x0000011b on screen" in result
    assert "uploaded screenshot" in result.lower()


def test_only_typed_question_returns_just_that():
    assert build_combined_query("printer offline", "") == "printer offline"


def test_only_image_text_returns_just_that():
    assert build_combined_query("", "Blue screen: CRITICAL_PROCESS_DIED") == \
        "Blue screen: CRITICAL_PROCESS_DIED"


def test_neither_returns_empty_string():
    assert build_combined_query("", "") == ""
    assert build_combined_query(None, None) == ""
