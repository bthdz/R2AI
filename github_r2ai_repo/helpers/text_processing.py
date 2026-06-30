"""Small text normalization helpers used by the API wrapper."""

from __future__ import annotations

import re
import unicodedata


LEGAL_ID_RE = re.compile(r"\b\d{1,4}/\d{4}/[A-ZĐ0-9]+(?:-[A-ZĐ0-9]+)*\b", re.I)
LEGAL_ID_SHORT_RE = re.compile(r"\b\d{1,4}/[A-ZĐ0-9]+(?:-[A-ZĐ0-9]+)+\b", re.I)

STOPWORDS = {
    "cua",
    "cho",
    "voi",
    "trong",
    "khi",
    "neu",
    "thi",
    "la",
    "va",
    "hoac",
    "cac",
    "nhung",
    "duoc",
    "khong",
    "phai",
    "theo",
    "quy",
    "dinh",
    "phap",
    "luat",
}


def norm_space(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def fold_vi(text: object) -> str:
    value = norm_space(text).lower()
    value = unicodedata.normalize("NFD", value)
    value = "".join(ch for ch in value if unicodedata.category(ch) != "Mn")
    return value.replace("đ", "d").replace("ð", "d")


def extract_legal_id(text: object) -> str:
    value = norm_space(text)
    match = LEGAL_ID_RE.search(value) or LEGAL_ID_SHORT_RE.search(value)
    return match.group(0).upper().replace("Ð", "Đ") if match else ""


def tokenize_vi(text: object, limit: int = 80) -> list[str]:
    raw = re.sub(r"[^a-z0-9%/.,-]+", " ", fold_vi(text)).split()
    tokens: list[str] = []
    seen: set[str] = set()
    for item in raw:
        token = item.strip(".,-")
        if len(token) < 3 or token in STOPWORDS or token in seen:
            continue
        seen.add(token)
        tokens.append(token)
        if len(tokens) >= limit:
            break
    return tokens

