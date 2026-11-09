import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deidentify import deidentify
from rehydrate import rehydrate


def test_email_is_masked():
    text = "Contact john@example.com for details."
    masked, mapping = deidentify(text)

    assert "john@example.com" not in masked
    assert "[EMAIL_ADDRESS_1]" in masked


def test_repeated_identifier_uses_same_placeholder():
    text = "Contact john@example.com. Email john@example.com again."
    masked, mapping = deidentify(text)

    assert masked.count("[EMAIL_ADDRESS_1]") == 2


def test_rehydration():
    original = "Contact john@example.com."
    masked, mapping = deidentify(original)

    restored = rehydrate(masked, mapping)

    assert restored == original


def test_empty_text():
    masked, mapping = deidentify("")

    assert masked == ""
    assert mapping == {}


def test_relative_time_is_preserved():
    text = "Patient requires review in 3 days post-op."
    masked, mapping = deidentify(text)

    assert "3 days post-op" in masked