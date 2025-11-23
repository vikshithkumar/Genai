from __future__ import annotations

import re
from typing import Iterable, List


_whitespace_re = re.compile(r"\s+")
_non_alnum_re = re.compile(r"[^0-9a-zA-Z]+")
_trailing_digits_re = re.compile(r"\d+$")


def normalize_text(text: str) -> str:
    """Normalize transaction description text.

    Operations:
    - Lowercase
    - Strip leading/trailing whitespace
    - Remove trailing digits (e.g. POS 1234)
    - Collapse non-alphanumeric to single space
    - Collapse repeated whitespace
    """

    t = text.strip().lower()
    t = _trailing_digits_re.sub("", t)
    t = _non_alnum_re.sub(" ", t)
    t = _whitespace_re.sub(" ", t)
    return t.strip()


def normalize_batch(texts: Iterable[str]) -> List[str]:
    return [normalize_text(t) for t in texts]
