"""
sequence_machine.fast_surf
===========================

"Fast Surfs" are cheap, deterministic heuristics that each try to explain
a sequence with a specific, well-understood shape: a polynomial, a
geometric progression, a constant-coefficient linear recurrence, a simple
transform of a simpler sequence, or a known named sequence (Fibonacci,
Catalan, primes, ...).

Every surf has the same shape::

    def my_surf(seq: Sequence) -> List[Candidate]:
        ...

and is registered with the ``@fast_surf(name)`` decorator. The engine
discovers and runs every registered surf without needing to know about it
in advance - new heuristics can be dropped in here (or in a third-party
module that imports ``register_fast_surf``) without touching
``engine.py``.

All surfs are pure functions of a :class:`~sequence_machine.core.Sequence`
and must not raise on "this doesn't apply" - they should simply return an
empty list. The engine defensively catches unexpected exceptions anyway,
but well-behaved surfs return ``[]`` for the common case.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Callable, Dict, List, Optional, Tuple

from .core import (
    Candidate,
    CandidateKind,
    Sequence,
    build_polynomial_expr,
    fit_minimal_polynomial,
    format_number,
    render_top_level,
)

FastSurfFn = Callable[[Sequence], List[Candidate]]

# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

_FAST_SURFS: List[Tuple[str, FastSurfFn]] = []


def fast_surf(name: str):
    """Decorator that registers a function as a Fast Surf heuristic."""

    def _decorator(fn: FastSurfFn) -> FastSurfFn:
        _FAST_SURFS.append((name, fn))
        return fn

    return _decorator


def register_fast_surf(name: str, fn: FastSurfFn) -> None:
    """Programmatic equivalent of the ``@fast_surf`` decorator, for
    third-party code that wants to extend Sequence Machine without
    editing this file."""
    _FAST_SURFS.append((name, fn))


def registered_fast_surfs() -> List[str]:
    return [name for name, _ in _FAST_SURFS]


def run_fast_surfs(seq: Sequence, exclude: Optional[set] = None) -> List[Candidate]:
    """Run every registered Fast Surf against ``seq`` and pool the results.

    A single misbehaving detector cannot crash the pipeline: exceptions
    are swallowed and treated as "no candidate from this surf".
    """
    exclude = exclude or set()
    out: List[Candidate] = []
    for name, fn in _FAST_SURFS:
        if name in exclude:
            continue
        try:
            found = fn(seq)
        except Exception:
            found = None
        if found:
            out.extend(found)
    return out


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def berlekamp_massey(s: List[Fraction]) -> List[Fraction]:
    """Find the shortest constant-coefficient linear recurrence satisfied
    by ``s``, using exact rational arithmetic.

    Returns coefficients ``[c1, c2, ..., cL]`` (possibly empty) such that
    ``s[i] = c1*s[i-1] + c2*s[i-2] + ... + cL*s[i-L]`` for every valid
    ``i >= L``. This is the classic Berlekamp-Massey / linear-feedback-
    shift-register algorithm, adapted from GF(2)/floating point folklore
    implementations to operate over the exact field of rationals.
    """
    n = len(s)
    if n == 0 or all(x == 0 for x in s):
        return []
    ls: List[Fraction] = []
    cur: List[Fraction] = []
    lf = 0
    ld = Fraction(0)
    for i in range(n):
        t = Fraction(0)
        for j in range(len(cur)):
            t += cur[j] * s[i - 1 - j]
        delta = s[i] - t
        if delta == 0:
            continue
        if not cur:
            cur = [Fraction(0)] * (i + 1)
            lf = i
            ld = delta
            continue
        k = delta / ld
        c = [Fraction(0)] * (i - lf - 1) + [k] + [-k * x for x in ls]
        if len(c) < len(cur):
            c += [Fraction(0)] * (len(cur) - len(c))
        for j in range(len(cur)):
            c[j] += cur[j]
        if i - len(cur) >= lf - len(ls):
            ls, lf, ld = cur, i, delta
        cur = c
    return cur


# --------------------------------------------------------------------------- #
# Surf 1: constant sequences
# --------------------------------------------------------------------------- #

@fast_surf("constant")
def detect_constant(seq: Sequence) -> List[Candidate]:
    vals = seq.values
    if all(v == vals[0] for v in vals):
        c = vals[0]

        def predict(n: int, c=c) -> Fraction:
            return c

        return [
            Candidate(
                description=f"a(n) = {format_number(c)}",
                kind=CandidateKind.CLOSED_FORM,
                source="fast_surf.constant",
                predict_fn=predict,
                complexity=1,
                exact=True,
            )
        ]
    return []


# --------------------------------------------------------------------------- #
# Surf 2: polynomial sequences (finite differences + exact interpolation)
# --------------------------------------------------------------------------- #

@fast_surf("polynomial")
def detect_polynomial(seq: Sequence) -> List[Candidate]:
    vals = list(seq.values)
    if len(vals) < 3:
        return []
    # Require at least one point of redundancy beyond the minimum needed to
    # fit a degree-d polynomial, otherwise *every* sequence of length n
    # trivially "fits" a degree (n-1) polynomial and the detection is
    # meaningless (pure overfitting, not a genuine discovery).
    coeffs = fit_minimal_polynomial(seq.indices(), vals, min_redundancy=1)
    if coeffs is None:
        return []
    expr = build_polynomial_expr(coeffs)

    def predict(n: int, coeffs=tuple(coeffs)) -> Fraction:
        total = Fraction(0)
        for power, c in enumerate(coeffs):
            if c != 0:
                total += c * Fraction(n) ** power
        return total

    degree = max((p for p, c in enumerate(coeffs) if c != 0), default=0)
    cand = Candidate(
        description=f"a(n) = {render_top_level(expr)}",
        kind=CandidateKind.CLOSED_FORM,
        source="fast_surf.polynomial",
        predict_fn=predict,
        complexity=expr.complexity(),
        exact=True,
        notes=f"degree-{degree} polynomial fit by finite differences",
    )
    return [cand]


# --------------------------------------------------------------------------- #
# Surf 3: geometric / exponential progressions (plain and shifted)
# --------------------------------------------------------------------------- #

def _int_pow(base: Fraction, exponent: int) -> Fraction:
    if exponent >= 0:
        return base ** exponent
    if base == 0:
        raise ZeroDivisionError
    return Fraction(1, 1) / (base ** (-exponent))


@fast_surf("geometric")
def detect_geometric(seq: Sequence) -> List[Candidate]:
    vals = list(seq.values)
    out: List[Candidate] = []
    start = seq.start_index

    # -- Plain geometric: a(n) = a0 * r^(n - start) --------------------------
    if all(v != 0 for v in vals[:-1]):
        ratios = [vals[i + 1] / vals[i] for i in range(len(vals) - 1)]
        if all(r == ratios[0] for r in ratios):
            r = ratios[0]
            a0 = vals[0]

            def predict(n: int, a0=a0, r=r, start=start) -> Fraction:
                return a0 * _int_pow(r, n - start)

            base_str = f"{format_number(a0)} * ({format_number(r)})^n" if start == 0 else (
                f"{format_number(a0)} * ({format_number(r)})^(n-{start})"
            )
            out.append(
                Candidate(
                    description=f"a(n) = {base_str}",
                    kind=CandidateKind.CLOSED_FORM,
                    source="fast_surf.geometric",
                    predict_fn=predict,
                    complexity=3,
                    exact=True,
                )
            )
            return out  # a pure geometric fit is strong enough; skip the shifted check

    # -- Shifted geometric: a(n) = A * r^n + C -------------------------------
    diffs = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
    if len(diffs) >= 3 and all(d != 0 for d in diffs[:-1]):
        dratios = [diffs[i + 1] / diffs[i] for i in range(len(diffs) - 1)]
        if all(dr == dratios[0] for dr in dratios) and dratios[0] not in (0, 1):
            r = dratios[0]
            # diffs[0] = a(start+1)-a(start) = A*r^start*(r-1)
            A = diffs[0] / ((r - 1) * _int_pow(r, start))
            C = vals[0] - A * _int_pow(r, start)

            def predict(n: int, A=A, r=r, C=C) -> Fraction:
                return A * _int_pow(r, n) + C

            desc = f"a(n) = {format_number(A)}*({format_number(r)})^n"
            if C > 0:
                desc += f" + {format_number(C)}"
            elif C < 0:
                desc += f" - {format_number(-C)}"
            out.append(
                Candidate(
                    description=desc,
                    kind=CandidateKind.CLOSED_FORM,
                    source="fast_surf.geometric_shifted",
                    predict_fn=predict,
                    complexity=6,
                    exact=True,
                )
            )
    return out


# --------------------------------------------------------------------------- #
# Surf 4: constant-coefficient linear recurrences (Berlekamp-Massey)
# --------------------------------------------------------------------------- #

@fast_surf("linear_recurrence")
def detect_linear_recurrence(seq: Sequence) -> List[Candidate]:
    vals = list(seq.values)
    n = len(vals)
    if n < 5:
        return []
    coeffs = berlekamp_massey(vals)
    order = len(coeffs)
    # Require the recurrence order to be small relative to the amount of
    # data available, and enough confirming points beyond the minimum
    # needed to fit it, so we don't report a trivial overfit.
    min_confirmations = 2
    if order == 0 or 2 * order + min_confirmations > n:
        return []

    from .core import RecurrenceRule  # local import to avoid a cycle at module load time

    initial = tuple(vals[:order])
    rule = RecurrenceRule(order=order, coeffs=tuple(coeffs), initial_terms=initial, start_index=seq.start_index)

    def predict(n_abs: int, rule=rule) -> Fraction:
        return rule.value_at(n_abs)

    return [
        Candidate(
            description=rule.to_str(),
            kind=CandidateKind.RECURRENCE,
            source="fast_surf.linear_recurrence",
            predict_fn=predict,
            complexity=rule.complexity(),
            exact=True,
            notes=f"order-{order} homogeneous linear recurrence (Berlekamp-Massey)",
        )
    ]


# --------------------------------------------------------------------------- #
# Surf 5: simple transforms - alternating sign, interleaving, cumulative sum
# --------------------------------------------------------------------------- #

_NON_RECURSIVE_EXCLUDE = {"alternating_sign", "interleaved", "cumulative_sum"}


@fast_surf("alternating_sign")
def detect_alternating_sign(seq: Sequence) -> List[Candidate]:
    """Detects a(n) = (-1)^n * b(n) (or -(-1)^n) where b(n) is itself
    something a Fast Surf can explain."""
    vals = list(seq.values)
    if any(v == 0 for v in vals):
        return []
    signs = [1 if v > 0 else -1 for v in vals]
    alt_pos = all(s == (1 if i % 2 == 0 else -1) for i, s in enumerate(signs))
    alt_neg = all(s == (-1 if i % 2 == 0 else 1) for i, s in enumerate(signs))
    if not (alt_pos or alt_neg):
        return []
    magnitudes = [abs(v) for v in vals]
    inner_seq = Sequence(values=tuple(magnitudes), start_index=seq.start_index)
    inner_candidates = run_fast_surfs(inner_seq, exclude=_NON_RECURSIVE_EXCLUDE)
    if not inner_candidates:
        return []
    inner_candidates.sort(key=lambda c: c.complexity)
    best_inner = inner_candidates[0]
    sign_at_start = 1 if alt_pos else -1

    def predict(n: int, best_inner=best_inner, start=seq.start_index, sign_at_start=sign_at_start) -> Fraction:
        sign = sign_at_start if (n - start) % 2 == 0 else -sign_at_start
        return sign * best_inner.predict(n)

    desc = f"a(n) = {'(-1)^n' if sign_at_start == 1 else '-(-1)^n'} * ({best_inner.description.split('=', 1)[1].strip()})"
    return [
        Candidate(
            description=desc,
            kind=CandidateKind.COMPOSITE,
            source="fast_surf.alternating_sign",
            predict_fn=predict,
            complexity=best_inner.complexity + 2,
            exact=best_inner.exact,
            notes="alternating sign wrapped around: " + best_inner.notes,
        )
    ]


@fast_surf("interleaved")
def detect_interleaved(seq: Sequence) -> List[Candidate]:
    """Detects sequences formed by interleaving two independently simple
    sequences at even/odd positions, e.g. 1, 10, 2, 20, 3, 30, ..."""
    vals = list(seq.values)
    if len(vals) < 6:
        return []
    evens = vals[0::2]
    odds = vals[1::2]
    if len(evens) < 3 or len(odds) < 3:
        return []
    even_seq = Sequence(values=tuple(evens), start_index=0)
    odd_seq = Sequence(values=tuple(odds), start_index=0)
    even_cands = run_fast_surfs(even_seq, exclude=_NON_RECURSIVE_EXCLUDE)
    odd_cands = run_fast_surfs(odd_seq, exclude=_NON_RECURSIVE_EXCLUDE)
    if not even_cands or not odd_cands:
        return []
    even_cands.sort(key=lambda c: c.complexity)
    odd_cands.sort(key=lambda c: c.complexity)
    best_even, best_odd = even_cands[0], odd_cands[0]
    start = seq.start_index

    def predict(n: int, best_even=best_even, best_odd=best_odd, start=start) -> Fraction:
        offset = n - start
        if offset % 2 == 0:
            return best_even.predict(offset // 2)
        return best_odd.predict(offset // 2)

    desc = (
        f"a(n) = [even n: {best_even.description}]  |  [odd n: {best_odd.description}]"
    )
    return [
        Candidate(
            description=desc,
            kind=CandidateKind.PIECEWISE,
            source="fast_surf.interleaved",
            predict_fn=predict,
            complexity=best_even.complexity + best_odd.complexity + 2,
            exact=best_even.exact and best_odd.exact,
            notes="even/odd index interleaving of two simpler sequences",
        )
    ]


@fast_surf("cumulative_sum")
def detect_cumulative_sum(seq: Sequence) -> List[Candidate]:
    """Detects a(n) that is the running total of an underlying sequence
    whose *first differences* are simple, but which a plain polynomial fit
    on a(n) itself would not directly reveal (e.g. sums of a geometric or
    dictionary sequence)."""
    vals = list(seq.values)
    if len(vals) < 5:
        return []
    diffs = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
    diff_seq = Sequence(values=tuple(diffs), start_index=seq.start_index + 1)
    diff_candidates = run_fast_surfs(diff_seq, exclude=_NON_RECURSIVE_EXCLUDE | {"polynomial", "affine_of_index"})
    if not diff_candidates:
        return []
    diff_candidates.sort(key=lambda c: c.complexity)
    best_diff = diff_candidates[0]
    a_start = vals[0]
    start = seq.start_index

    def predict(n: int, best_diff=best_diff, a_start=a_start, start=start) -> Fraction:
        if n == start:
            return a_start
        total = a_start
        for k in range(start + 1, n + 1):
            total += best_diff.predict(k)
        return total

    return [
        Candidate(
            description=f"a(n) = a({start}) + sum_{{k={start + 1}}}^n [{best_diff.description}]",
            kind=CandidateKind.COMPOSITE,
            source="fast_surf.cumulative_sum",
            predict_fn=predict,
            complexity=best_diff.complexity + 3,
            exact=best_diff.exact,
            notes="running sum of a simpler difference sequence: " + best_diff.notes,
        )
    ]


# --------------------------------------------------------------------------- #
# Surf 6: dictionary of named integer sequences
# --------------------------------------------------------------------------- #

def _fib(n: int) -> Fraction:
    if n < 0:
        f = _fib(-n)
        return f if n % 2 == 1 else -f
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return Fraction(a)


def _lucas(n: int) -> Fraction:
    if n < 0:
        l = _lucas(-n)
        return l if n % 2 == 0 else -l
    a, b = 2, 1
    for _ in range(n):
        a, b = b, a + b
    return Fraction(a)


def _catalan(n: int) -> Fraction:
    if n < 0:
        raise ValueError
    return Fraction(math.comb(2 * n, n), n + 1)


def _factorial(n: int) -> Fraction:
    if n < 0:
        raise ValueError
    return Fraction(math.factorial(n))


def _double_factorial(n: int) -> Fraction:
    if n < 0:
        raise ValueError
    result = 1
    k = n
    while k > 0:
        result *= k
        k -= 2
    return Fraction(result)


def _bell(n: int) -> Fraction:
    if n < 0:
        raise ValueError
    row = [1]
    for _ in range(n):
        new_row = [row[-1]]
        for x in row:
            new_row.append(new_row[-1] + x)
        row = new_row
    return Fraction(row[0])


def _motzkin(n: int) -> Fraction:
    if n < 0:
        raise ValueError
    m = [1, 1]
    for i in range(2, n + 1):
        m.append(((2 * i + 1) * m[i - 1] + (3 * i - 3) * m[i - 2]) // (i + 2))
    return Fraction(m[n])


_PRIME_CACHE = [2, 3]


def _nth_prime(n: int) -> Fraction:
    """1-indexed: _nth_prime(1) == 2."""
    if n < 1:
        raise ValueError
    while len(_PRIME_CACHE) < n:
        cand = _PRIME_CACHE[-1] + 2
        while True:
            if all(cand % p != 0 for p in _PRIME_CACHE if p * p <= cand):
                _PRIME_CACHE.append(cand)
                break
            cand += 2
    return Fraction(_PRIME_CACHE[n - 1])


_KNOWN_SEQUENCES: Dict[str, Callable[[int], Fraction]] = {
    "Fibonacci": _fib,
    "Lucas numbers": _lucas,
    "Catalan numbers": _catalan,
    "factorial": _factorial,
    "double factorial": _double_factorial,
    "triangular numbers": lambda n: Fraction(n * (n + 1), 2) if n >= 0 else Fraction(0),
    "square numbers": lambda n: Fraction(n * n),
    "cube numbers": lambda n: Fraction(n ** 3),
    "powers of two": lambda n: _int_pow(Fraction(2), n),
    "powers of three": lambda n: _int_pow(Fraction(3), n),
    "n-th prime": _nth_prime,
    "Bell numbers": _bell,
    "Motzkin numbers": _motzkin,
    "central binomial coefficient C(2n,n)": lambda n: Fraction(math.comb(2 * n, n)) if n >= 0 else Fraction(0),
}


@fast_surf("dictionary")
def detect_dictionary(seq: Sequence) -> List[Candidate]:
    vals = list(seq.values)
    n_pts = len(vals)
    out: List[Candidate] = []
    for name, fn in _KNOWN_SEQUENCES.items():
        for offset in range(-3, 4):
            try:
                generated = [fn(seq.start_index + i + offset) for i in range(n_pts)]
            except (ValueError, OverflowError):
                continue
            if generated == vals:
                def predict(n: int, fn=fn, offset=offset) -> Fraction:
                    return fn(n + offset)

                arg = "n" if offset == 0 else (f"n+{offset}" if offset > 0 else f"n{offset}")
                out.append(
                    Candidate(
                        description=f"a(n) = {name}({arg})",
                        kind=CandidateKind.DICTIONARY,
                        source=f"fast_surf.dictionary[{name}]",
                        predict_fn=predict,
                        complexity=3 + abs(offset),
                        exact=True,
                    )
                )
                break  # this named sequence matched at one offset; no need to try more
    return out


# --------------------------------------------------------------------------- #
# Surf 7: affine (offset/scale) transform of the index
# --------------------------------------------------------------------------- #

@fast_surf("affine_of_index")
def detect_affine(seq: Sequence) -> List[Candidate]:
    """Detects the simplest possible pattern, a(n) = m*n + b, as a fast
    special case ahead of the general polynomial surf (kept separate for
    clarity/readability of the resulting formula and so it fires even for
    very short 2-3 term sequences where finite differences are too thin
    to trust)."""
    vals = list(seq.values)
    idx = seq.indices()
    if len(vals) < 2:
        return []
    if idx[1] == idx[0]:
        return []
    m = (vals[1] - vals[0]) / (idx[1] - idx[0])
    if m == 0:
        return []  # the constant surf already covers this, more clearly
    b = vals[0] - m * idx[0]
    if all(v == m * n + b for n, v in zip(idx, vals)):
        def predict(n: int, m=m, b=b) -> Fraction:
            return m * n + b

        if m == 1:
            n_term = "n"
        elif m == -1:
            n_term = "-n"
        else:
            n_term = f"{format_number(m)}*n"
        desc = f"a(n) = {n_term}"
        if b > 0:
            desc += f" + {format_number(b)}"
        elif b < 0:
            desc += f" - {format_number(-b)}"
        return [
            Candidate(
                description=desc,
                kind=CandidateKind.CLOSED_FORM,
                source="fast_surf.affine_of_index",
                predict_fn=predict,
                complexity=2,
                exact=True,
            )
        ]
    return []
