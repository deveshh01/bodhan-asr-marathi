"""Text normalisation for Marathi transcripts.

Two levels are used on purpose:

* ``clean_target`` - light cleaning applied to *training targets*. It keeps
  punctuation and casing because Indic-Transcribe is run with ``pnc="yes"`` and
  we don't want to teach it to drop punctuation.
* ``normalize_for_wer`` - aggressive normalisation applied to *both* reference and
  hypothesis only at scoring time, so WER/CER measure word choice, not
  punctuation/spacing conventions (standard practice for Indic ASR benchmarks).
"""

from __future__ import annotations

import re
import unicodedata

# Zero-width chars that show up in scraped/annotated Devanagari text.
# ZWJ/ZWNJ (U+200D/U+200C) can change conjunct rendering but not the word, so
# they're dropped for scoring; they're kept in targets.
_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿"), None)

# Annotation tags such as <noise>, [laugh], (inaudible) used by some corpora.
_TAG_RE = re.compile(r"<[^>]*>|\[[^\]]*\]|\{[^}]*\}")
_WS_RE = re.compile(r"\s+")

# Everything that is punctuation/symbol in Unicode, plus the Devanagari danda(s).
_PUNCT_CHARS = "".join(
    chr(c)
    for c in range(0x110000)
    if unicodedata.category(chr(c)).startswith(("P", "S"))
) + "।॥"
_PUNCT_TABLE = dict.fromkeys(map(ord, _PUNCT_CHARS), " ")


def clean_target(text: str) -> str:
    """Light cleanup for training targets (keeps punctuation)."""
    text = unicodedata.normalize("NFC", text)
    text = _TAG_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def normalize_for_wer(text: str) -> str:
    """Aggressive normalisation used only for WER/CER scoring."""
    text = unicodedata.normalize("NFC", text).translate(_ZERO_WIDTH)
    text = _TAG_RE.sub(" ", text)
    text = text.translate(_PUNCT_TABLE).lower()  # lower() only affects code-mixed Latin
    return _WS_RE.sub(" ", text).strip()


_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def script_stats(text: str) -> dict[str, int]:
    """Counts of Devanagari vs Latin letters - used to profile code-mixing."""
    return {
        "devanagari": len(_DEVANAGARI_RE.findall(text)),
        "latin": len(_LATIN_RE.findall(text)),
    }
