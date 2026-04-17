from __future__ import annotations

import math
import re


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def normalize_tokens(text: str) -> set[str]:
    return set(TOKEN_PATTERN.findall(text.lower()))


def lexical_overlap(left: str, right: str) -> float:
    left_tokens = normalize_tokens(left)
    right_tokens = normalize_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    union = left_tokens | right_tokens
    if not union:
        return 0.0
    return len(left_tokens & right_tokens) / len(union)


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def rank_score(rank: int) -> float:
    if rank <= 0:
        return 0.0
    return 1.0 / math.sqrt(rank)
