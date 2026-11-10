"""Word error rate, character error rate and word alignment.

Written from scratch (no jiwer) so the alignment can also tell us *which*
recognised words were wrong. The benchmark uses that to check whether the
words Shrutlekh flags are really the ones that are wrong.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass


def normalize(text: str) -> str:
    """Lower-case, unify Unicode forms and drop punctuation and symbols.

    Punctuation is removed by Unicode category rather than with a regex like
    [^\\w\\s], because \\w does not match Devanagari vowel signs (matras) and
    would break Hindi words apart.
    """
    text = unicodedata.normalize("NFKC", text).lower()
    kept = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat.startswith("P") or cat.startswith("S"):
            continue
        kept.append(ch)
    return " ".join("".join(kept).split())


def tokenize(text: str) -> list[str]:
    return normalize(text).split()


@dataclass(frozen=True)
class AlignOp:
    op: str  # "equal" | "sub" | "del" | "ins"
    ref: str | None
    hyp: str | None
    hyp_index: int | None  # position in the hypothesis token list, None for deletions


def align(ref: list[str], hyp: list[str]) -> list[AlignOp]:
    """Levenshtein alignment with a backtrace. Ties prefer equal/sub, then del, then ins."""
    n, m = len(ref), len(hyp)
    dist = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dist[i][0] = i
    for j in range(m + 1):
        dist[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dist[i][j] = min(dist[i - 1][j - 1] + cost, dist[i - 1][j] + 1, dist[i][j - 1] + 1)

    ops: list[AlignOp] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            if dist[i][j] == dist[i - 1][j - 1] + cost:
                ops.append(AlignOp("equal" if cost == 0 else "sub", ref[i - 1], hyp[j - 1], j - 1))
                i, j = i - 1, j - 1
                continue
        if i > 0 and dist[i][j] == dist[i - 1][j] + 1:
            ops.append(AlignOp("del", ref[i - 1], None, None))
            i -= 1
        else:
            ops.append(AlignOp("ins", None, hyp[j - 1], j - 1))
            j -= 1
    ops.reverse()
    return ops


@dataclass(frozen=True)
class ErrorCounts:
    substitutions: int
    deletions: int
    insertions: int
    reference_length: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def rate(self) -> float:
        if self.reference_length == 0:
            return 0.0 if self.errors == 0 else 1.0
        return self.errors / self.reference_length


def _count(ref: list[str], hyp: list[str]) -> ErrorCounts:
    ops = align(ref, hyp)
    return ErrorCounts(
        substitutions=sum(o.op == "sub" for o in ops),
        deletions=sum(o.op == "del" for o in ops),
        insertions=sum(o.op == "ins" for o in ops),
        reference_length=len(ref),
    )


def word_errors(reference: str, hypothesis: str) -> ErrorCounts:
    return _count(tokenize(reference), tokenize(hypothesis))


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate = (substitutions + deletions + insertions) / words in the reference."""
    return word_errors(reference, hypothesis).rate


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate, ignoring spaces."""
    ref = list(normalize(reference).replace(" ", ""))
    hyp = list(normalize(hypothesis).replace(" ", ""))
    return _count(ref, hyp).rate


def wrong_hypothesis_tokens(reference: str, hypothesis_tokens: list[str]) -> set[int]:
    """Indices of hypothesis tokens that were substituted or inserted."""
    ops = align(tokenize(reference), hypothesis_tokens)
    return {o.hyp_index for o in ops if o.op in ("sub", "ins") and o.hyp_index is not None}
