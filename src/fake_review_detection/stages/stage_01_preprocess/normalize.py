# Cleans review and title strings before models and rules see them.

import html
import re
import unicodedata

_WS = re.compile(r"\s+")


def normalize_review_text(text: str) -> str:
    # Body: NFKC, unescape HTML, single spaces, lowercase — matches training prep for TF-IDF.
    if not text or not isinstance(text, str):
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = html.unescape(t)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = _WS.sub(" ", t).strip().lower()
    return t


def normalize_title(title: str | None) -> str:
    # Same cleanup as body but keep case so “shouting” title heuristics still work.
    if not title or not isinstance(title, str):
        return ""
    t = unicodedata.normalize("NFKC", title)
    t = html.unescape(t)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = _WS.sub(" ", t).strip()
    return t
def normalize_review_text_for_bert(text: str) -> str:
    # Light cleanup for transformer input: preserve case, punctuation, repetition.
    if not text or not isinstance(text, str):
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = html.unescape(t)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = _WS.sub(" ", t).strip()
    return t