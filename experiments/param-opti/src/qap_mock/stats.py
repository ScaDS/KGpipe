from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def stdev(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def rankdata(xs: Sequence[float]) -> List[int]:
    # Simple dense ranking (ties get same rank).
    sorted_unique = sorted(set(xs))
    rank = {v: i + 1 for i, v in enumerate(sorted_unique)}
    return [rank[v] for v in xs]


def pearsonr(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    mx = mean(x)
    my = mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    denx = math.sqrt(sum((a - mx) ** 2 for a in x))
    deny = math.sqrt(sum((b - my) ** 2 for b in y))
    if denx == 0.0 or deny == 0.0:
        return float("nan")
    return num / (denx * deny)


def spearmanr(x: Sequence[float], y: Sequence[float]) -> float:
    rx = rankdata(x)
    ry = rankdata(y)
    return pearsonr(rx, ry)


def mae(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or not x:
        return float("nan")
    return sum(abs(a - b) for a, b in zip(x, y)) / len(x)


def topk_agreement(x: Sequence[float], y: Sequence[float], k: int) -> float:
    if len(x) != len(y) or not x:
        return float("nan")
    n = len(x)
    k = max(1, min(k, n))
    topx = set(sorted(range(n), key=lambda i: x[i], reverse=True)[:k])
    topy = set(sorted(range(n), key=lambda i: y[i], reverse=True)[:k])
    return len(topx & topy) / k

