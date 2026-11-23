"""Retrieval and answer metrics. Pure functions, no network."""
from __future__ import annotations

import math
import re
from collections import Counter


def first_rank(retrieved: list[str], gold: set[str] | list[str]) -> int | None:
    """1-based rank of the first gold unit (unit-level match, not page-level)."""
    gold = set(gold)
    for i, uid in enumerate(retrieved, 1):
        if uid in gold:
            return i
    return None


def recall_at_k(ranks: list[int | None], k: int) -> float:
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks) if ranks else float("nan")


def mrr(ranks: list[int | None]) -> float:
    return sum(1.0 / r for r in ranks if r) / len(ranks) if ranks else float("nan")


def precision_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    top = retrieved[:k]
    return sum(1 for u in top if u in gold) / k if k else float("nan")


def ndcg_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, u in enumerate(retrieved[:k], 1) if u in gold)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(k, len(gold)) + 1))
    return dcg / ideal if ideal else float("nan")


def summarise_ranks(ranks: list[int | None], ks=(1, 3, 5)) -> dict:
    out = {"n": len(ranks)}
    for k in ks:
        out[f"recall@{k}"] = round(recall_at_k(ranks, k), 4)
    out["mrr"] = round(mrr(ranks), 4)
    return out


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    v = sorted(values)
    idx = (len(v) - 1) * p / 100
    lo, hi = math.floor(idx), math.ceil(idx)
    return v[lo] + (v[hi] - v[lo]) * (idx - lo)


_W = re.compile(r"[a-z0-9]+")


def token_f1(pred: str, gold: str) -> float:
    """Key-fact proxy: token-overlap F1 (cheap, no calls; undercounts paraphrase)."""
    p, g = Counter(_W.findall(pred.lower())), Counter(_W.findall(gold.lower()))
    common = sum((p & g).values())
    if not common:
        return 0.0
    prec, rec = common / sum(p.values()), common / sum(g.values())
    return 2 * prec * rec / (prec + rec)


def yes_no(text: str) -> str | None:
    t = text.strip().lower()
    if re.match(r"^(yes|yep|correct|it is|they are|included)\b", t):
        return "yes"
    if re.match(r"^(no|not|they do not|it does not|these do not|separate)\b", t):
        return "no"
    return None
