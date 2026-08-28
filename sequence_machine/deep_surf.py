"""
sequence_machine.deep_surf
============================

"Deep Surfs" are the heavier search engines that only run when the cheap
Fast Surfs (see ``fast_surf.py``) fail to explain a sequence exactly.
Each Deep Surf receives a hard wall-clock ``deadline`` (an absolute
``time.monotonic()`` timestamp) and must stop searching and return
whatever it has - or nothing - once that deadline passes. The engine runs
every registered Deep Surf concurrently (see :func:`run_deep_surfs`) so
that one slow search does not starve the others out of the shared time
budget.

Implemented Deep Surfs:

    * ``symbolic_regression``       - genetic-programming search over small
                                       expression trees (tree-based AST search).
    * ``nonhomogeneous_recurrence`` - linear recurrences with a polynomial
                                       forcing term, and higher orders than
                                       the Fast Surf's Berlekamp-Massey pass
                                       is allowed to consider.
    * ``number_theoretic``          - Euler's totient, divisor functions,
                                       prime-factor counts, digit sums.
    * ``continued_fraction``        - recognises sequences of convergents of
                                       classic irrational constants.
    * ``periodic_cycle``            - genuinely repeating (mod-cycle) value
                                       sequences of small period.
    * ``compositional_log_power``   - a(n) = base ** (polynomial in n): fits
                                       a polynomial to the *exponent* of a
                                       geometric base (a fast-primitive
                                       composed with itself, as requested).
    * ``compositional_ratio_poly``  - a(n) = a(n-1) * R(n) where R(n) is a
                                       polynomial in n (captures factorial-
                                       like product formulas).
    * ``oeis_lookup``                - optional, network-dependent OEIS API
                                       query; disabled unless explicitly
                                       enabled, and fails silently offline.

A note on concurrency
----------------------
These workers are dispatched on a :class:`concurrent.futures.ThreadPoolExecutor`.
Pure-Python CPU-bound code does not get true parallelism across threads
because of the GIL, but it does get cooperative interleaving, which is
enough to honour a *shared* wall-clock deadline across several searches
running "at once". Every search loop below periodically checks
``time.monotonic() < deadline`` so no single worker can block the others
indefinitely. A production deployment with true multi-core parallelism
would move these to a ``ProcessPoolExecutor`` - the trade-off is that a
``Candidate``'s ``predict_fn`` closures are not picklable, so that would
also require refactoring predictions into picklable, module-level
callables (e.g. via ``functools.partial``) instead of closures.
"""

from __future__ import annotations

import math
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
from typing import Callable, Dict, List, Optional, Tuple

from .core import (
    BinOp,
    Candidate,
    CandidateKind,
    Const,
    Expr,
    Index,
    RecurrenceRule,
    Sequence,
    UnaryOp,
    build_polynomial_expr,
    fit_minimal_polynomial,
    format_number,
    render_top_level,
    simplify_expr,
    solve_linear_system,
)

DeepSurfFn = Callable[[Sequence, float], List[Candidate]]

# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

_DEEP_SURFS: List[Tuple[str, DeepSurfFn]] = []


def deep_surf(name: str):
    """Decorator that registers a function as a Deep Surf heuristic.

    The decorated function must accept ``(seq: Sequence, deadline: float)``
    and return a ``List[Candidate]``.
    """

    def _decorator(fn: DeepSurfFn) -> DeepSurfFn:
        _DEEP_SURFS.append((name, fn))
        return fn

    return _decorator


def register_deep_surf(name: str, fn: DeepSurfFn) -> None:
    """Programmatic equivalent of ``@deep_surf`` for third-party extensions."""
    _DEEP_SURFS.append((name, fn))


def registered_deep_surfs() -> List[str]:
    return [name for name, _ in _DEEP_SURFS]


def run_deep_surfs(
    seq: Sequence,
    timeout: float = 4.0,
    max_workers: Optional[int] = None,
    enable_oeis: bool = False,
) -> List[Candidate]:
    """Dispatch every registered Deep Surf against ``seq`` with a shared
    wall-clock time budget, gathering whatever candidates come back before
    the budget is exhausted.
    """
    deadline = time.monotonic() + timeout
    tasks = [(name, fn) for name, fn in _DEEP_SURFS if name != "oeis_lookup" or enable_oeis]
    if not tasks:
        return []
    out: List[Candidate] = []
    with ThreadPoolExecutor(max_workers=max_workers or len(tasks)) as pool:
        futures = {pool.submit(fn, seq, deadline): name for name, fn in tasks}
        # Give every worker a small grace period past its own deadline to
        # unwind and return, but never wait indefinitely.
        overall_wait = timeout + 2.0
        try:
            for future in as_completed(futures, timeout=overall_wait):
                name = futures[future]
                try:
                    result = future.result(timeout=0)
                except Exception:
                    continue
                if result:
                    out.extend(result)
        except TimeoutError:
            pass
    return out


# --------------------------------------------------------------------------- #
# Deep Surf 1: symbolic regression via genetic programming
# --------------------------------------------------------------------------- #

_GP_BINARY_OPS = ("+", "-", "*", "/")
_GP_UNARY_OPS = ("neg", "abs")


def _random_expr(rng: random.Random, depth: int, consts: List[int]) -> Expr:
    """Grow a random expression tree, biased toward terminals as depth runs out."""
    if depth <= 0 or rng.random() < 0.32:
        if rng.random() < 0.55:
            return Index()
        return Const(Fraction(rng.choice(consts)))
    roll = rng.random()
    if roll < 0.15:
        # bounded integer power keeps evaluation well-defined and fast
        return BinOp("**", _random_expr(rng, depth - 1, consts), Const(Fraction(rng.choice([0, 1, 2, 3]))))
    if roll < 0.30:
        return UnaryOp(rng.choice(_GP_UNARY_OPS), _random_expr(rng, depth - 1, consts))
    return BinOp(rng.choice(_GP_BINARY_OPS), _random_expr(rng, depth - 1, consts), _random_expr(rng, depth - 1, consts))


def _tree_nodes(expr: Expr) -> List[Expr]:
    nodes = [expr]
    if isinstance(expr, BinOp):
        nodes.extend(_tree_nodes(expr.left))
        nodes.extend(_tree_nodes(expr.right))
    elif isinstance(expr, UnaryOp):
        nodes.extend(_tree_nodes(expr.child))
    return nodes


def _tree_depth(expr: Expr) -> int:
    if isinstance(expr, BinOp):
        return 1 + max(_tree_depth(expr.left), _tree_depth(expr.right))
    if isinstance(expr, UnaryOp):
        return 1 + _tree_depth(expr.child)
    return 1


def _replace_node(expr: Expr, target: Expr, replacement: Expr) -> Expr:
    if expr is target:
        return replacement
    if isinstance(expr, BinOp):
        return BinOp(expr.op, _replace_node(expr.left, target, replacement), _replace_node(expr.right, target, replacement))
    if isinstance(expr, UnaryOp):
        return UnaryOp(expr.op, _replace_node(expr.child, target, replacement))
    return expr


def _crossover(a: Expr, b: Expr, rng: random.Random) -> Expr:
    donor_subtree = rng.choice(_tree_nodes(b))
    target = rng.choice(_tree_nodes(a))
    return _replace_node(a, target, donor_subtree)


def _mutate(expr: Expr, rng: random.Random, consts: List[int], max_depth: int) -> Expr:
    target = rng.choice(_tree_nodes(expr))
    replacement = _random_expr(rng, rng.randint(0, 2), consts)
    child = _replace_node(expr, target, replacement)
    if _tree_depth(child) > max_depth + 2:
        return _random_expr(rng, max_depth, consts)
    return child


def _gp_fitness(expr: Expr, indices: List[int], values: List[Fraction]) -> Tuple[float, bool]:
    total_err = 0.0
    exact = True
    for n, expected in zip(indices, values):
        try:
            predicted = expr.eval(Fraction(n))
        except (ZeroDivisionError, ValueError, OverflowError):
            return -1e15, False
        if predicted != expected:
            exact = False
        total_err += abs(float(predicted) - float(expected))
    score = -total_err - 0.02 * expr.complexity()
    if exact:
        score += 1e6
    return score, exact


@deep_surf("symbolic_regression")
def symbolic_regression_search(
    seq: Sequence,
    deadline: float,
    population_size: int = 150,
    max_depth: int = 4,
    seed: int = 20240521,
) -> List[Candidate]:
    """A small tree-based genetic programming engine that evolves closed-form
    expressions in ``n`` against the observed sequence, favouring exact
    matches and, among those, the simplest (lowest node-count) trees."""
    if len(seq) < 4:
        return []
    rng = random.Random(seed)
    indices = seq.indices()
    values = list(seq.values)
    int_terms = [int(v) for v in values if v.denominator == 1]
    consts = sorted(set([0, 1, 2, -1, -2, 3] + int_terms[:3]))
    consts = [c for c in consts if abs(c) <= 50] or [0, 1, 2]

    population = [_random_expr(rng, max_depth, consts) for _ in range(population_size)]
    best_expr: Optional[Expr] = None
    best_score = -math.inf
    generations = 0

    while time.monotonic() < deadline:
        generations += 1
        scored = [(_gp_fitness(ind, indices, values), ind) for ind in population]
        scored.sort(key=lambda t: -t[0][0])
        top_score, top_exact = scored[0][0]
        if top_score > best_score:
            best_score, best_expr = top_score, scored[0][1]
        if top_exact:
            best_expr = scored[0][1]
            break
        survivors = [ind for (_, ind) in scored[: max(4, population_size // 6)]]
        next_pop = list(survivors)
        while len(next_pop) < population_size:
            if rng.random() < 0.75:
                child = _crossover(rng.choice(survivors), rng.choice(survivors), rng)
                if rng.random() < 0.35:
                    child = _mutate(child, rng, consts, max_depth)
            else:
                child = _random_expr(rng, max_depth, consts)
            if _tree_depth(child) > max_depth + 3:
                child = _random_expr(rng, max_depth, consts)
            next_pop.append(child)
        population = next_pop
        if generations > 300:
            break

    if best_expr is None:
        return []
    best_expr = simplify_expr(best_expr)
    fake_seq = Sequence(values=tuple(values), start_index=seq.start_index)
    candidate = Candidate(
        description=f"a(n) = {render_top_level(best_expr)}",
        kind=CandidateKind.SYMBOLIC,
        source="deep_surf.symbolic_regression",
        predict_fn=lambda n, e=best_expr: e.eval(Fraction(n)),
        complexity=best_expr.complexity(),
        notes=f"genetic-programming search, {generations} generations",
    )
    candidate.exact, candidate.max_abs_error = candidate.verify(fake_seq)
    if not candidate.exact and candidate.max_abs_error > 1e-6:
        # Only surface inexact symbolic fits if they are at least close;
        # wildly wrong random trees are not useful to report.
        return []
    return [candidate]


# --------------------------------------------------------------------------- #
# Deep Surf 2: non-homogeneous / higher-order linear recurrences
# --------------------------------------------------------------------------- #

@deep_surf("nonhomogeneous_recurrence")
def nonhomogeneous_recurrence_search(
    seq: Sequence,
    deadline: float,
    max_order: int = 4,
    max_poly_degree: int = 3,
    extra_confirmations: int = 2,
) -> List[Candidate]:
    """Fits ``a(n) = c1*a(n-1) + ... + ck*a(n-k) + P(n)`` for a polynomial
    forcing term ``P``, trying increasingly complex ``(order, degree)``
    combinations and stopping at the first exact, well-confirmed fit.

    This generalises the Fast Surf's homogeneous Berlekamp-Massey pass to
    recurrences with an additive polynomial-in-n term, e.g.
    ``a(n) = a(n-1) + n**2``.
    """
    vals = list(seq.values)
    idx = seq.indices()
    n_data = len(vals)
    start = seq.start_index

    best: Optional[Candidate] = None
    best_total = math.inf
    for total in range(1, max_order + max_poly_degree + 2):
        if time.monotonic() > deadline:
            break
        for k in range(1, min(max_order, total) + 1):
            d = total - k
            if d < 0 or d > max_poly_degree:
                continue
            unknowns = k + d + 1
            if n_data - k < unknowns + extra_confirmations:
                continue
            rows: List[List[Fraction]] = []
            rhs: List[Fraction] = []
            for eq_i in range(unknowns):
                pos = k + eq_i
                n_abs = idx[pos]
                row = [vals[pos - 1 - i] for i in range(k)] + [Fraction(n_abs) ** p for p in range(d + 1)]
                rows.append(row)
                rhs.append(vals[pos])
            solution = solve_linear_system(rows, rhs)
            if solution is None:
                continue
            c = solution[:k]
            b = solution[k:]
            ok = True
            for pos in range(k + unknowns, n_data):
                n_abs = idx[pos]
                predicted = sum(c[i] * vals[pos - 1 - i] for i in range(k)) + sum(
                    b[p] * Fraction(n_abs) ** p for p in range(d + 1)
                )
                if predicted != vals[pos]:
                    ok = False
                    break
            if not ok:
                continue
            inhomogeneous = build_polynomial_expr(b)
            rule = RecurrenceRule(
                order=k,
                coeffs=tuple(c),
                initial_terms=tuple(vals[:k]),
                start_index=start,
                inhomogeneous=inhomogeneous,
            )
            candidate = Candidate(
                description=rule.to_str(),
                kind=CandidateKind.RECURRENCE,
                source="deep_surf.nonhomogeneous_recurrence",
                predict_fn=lambda n, rule=rule: rule.value_at(n),
                complexity=rule.complexity(),
                exact=True,
                notes=f"order-{k} recurrence with a degree-{d} polynomial forcing term",
            )
            if total < best_total:
                best_total, best = total, candidate
        if best is not None:
            break
    return [best] if best is not None else []


# --------------------------------------------------------------------------- #
# Deep Surf 3: number-theoretic functions
# --------------------------------------------------------------------------- #

def _omega_distinct(n: int) -> int:
    if n < 1:
        raise ValueError
    count, x, d = 0, n, 2
    while d * d <= x:
        if x % d == 0:
            count += 1
            while x % d == 0:
                x //= d
        d += 1
    if x > 1:
        count += 1
    return count


def _big_omega(n: int) -> int:
    if n < 1:
        raise ValueError
    count, x, d = 0, n, 2
    while d * d <= x:
        while x % d == 0:
            count += 1
            x //= d
        d += 1
    if x > 1:
        count += 1
    return count


def _euler_phi(n: int) -> int:
    if n < 1:
        raise ValueError
    result, x, p = n, n, 2
    while p * p <= x:
        if x % p == 0:
            while x % p == 0:
                x //= p
            result -= result // p
        p += 1
    if x > 1:
        result -= result // x
    return result


def _num_divisors(n: int) -> int:
    if n < 1:
        raise ValueError
    count, i = 0, 1
    while i * i <= n:
        if n % i == 0:
            count += 1 if i * i == n else 2
        i += 1
    return count


def _sum_divisors(n: int) -> int:
    if n < 1:
        raise ValueError
    total, i = 0, 1
    while i * i <= n:
        if n % i == 0:
            total += i
            if i * i != n:
                total += n // i
        i += 1
    return total


def _digit_sum(n: int) -> int:
    if n < 0:
        raise ValueError
    return sum(int(ch) for ch in str(n))


_NUMBER_THEORETIC_FUNCS: Dict[str, Callable[[int], int]] = {
    "the number of distinct prime factors, omega(n)": _omega_distinct,
    "the number of prime factors with multiplicity, Omega(n)": _big_omega,
    "Euler's totient, phi(n)": _euler_phi,
    "the number of divisors, d(n)": _num_divisors,
    "the sum of divisors, sigma(n)": _sum_divisors,
    "the digit sum of n": _digit_sum,
}


@deep_surf("number_theoretic")
def number_theoretic_search(seq: Sequence, deadline: float) -> List[Candidate]:
    if not seq.is_all_integer():
        return []
    vals = [int(v) for v in seq.values]
    idx = seq.indices()
    out: List[Candidate] = []
    for name, fn in _NUMBER_THEORETIC_FUNCS.items():
        if time.monotonic() > deadline:
            break
        for offset in range(-5, 6):
            try:
                generated = [fn(n + offset) for n in idx]
            except (ValueError, ZeroDivisionError):
                continue
            if generated == vals:
                arg = "n" if offset == 0 else (f"n+{offset}" if offset > 0 else f"n{offset}")
                out.append(
                    Candidate(
                        description=f"a(n) = {name}, evaluated at {arg}",
                        kind=CandidateKind.DICTIONARY,
                        source=f"deep_surf.number_theoretic[{name}]",
                        predict_fn=lambda n, fn=fn, offset=offset: Fraction(fn(n + offset)),
                        complexity=5 + abs(offset),
                        exact=True,
                    )
                )
                break
    return out


# --------------------------------------------------------------------------- #
# Deep Surf 4: continued-fraction convergent recognition
# --------------------------------------------------------------------------- #

def _cf_terms_sqrt(k: int, count: int) -> List[int]:
    a0 = math.isqrt(k)
    terms = [a0]
    if a0 * a0 == k:
        return terms + [0] * (count - 1)  # perfect square: rational, degenerate
    m, d, a = 0, 1, a0
    while len(terms) < count:
        m = d * a - m
        d = (k - m * m) // d
        a = (a0 + m) // d
        terms.append(a)
    return terms


def _cf_terms_golden(count: int) -> List[int]:
    return [1] * count


def _cf_terms_e(count: int) -> List[int]:
    terms = [2]
    k = 1
    while len(terms) < count:
        terms.append(1)
        if len(terms) >= count:
            break
        terms.append(2 * k)
        k += 1
        if len(terms) >= count:
            break
        terms.append(1)
    return terms[:count]


# Pre-computed continued-fraction terms of pi (irregular, no known simple
# pattern) - enough terms to recognise short input windows.
_PI_CF_TERMS = [
    3, 7, 15, 1, 292, 1, 1, 1, 2, 1, 3, 1, 14, 2, 1, 1, 2, 2, 2, 2,
    1, 84, 2, 1, 1, 15, 3, 13, 1, 4, 2, 6, 6, 99, 1, 2, 2, 6, 3, 5,
]


def _convergents_from_terms(terms: List[int]) -> List[Fraction]:
    h_prev2, h_prev1 = 0, 1
    k_prev2, k_prev1 = 1, 0
    convergents = []
    for a in terms:
        h = a * h_prev1 + h_prev2
        k = a * k_prev1 + k_prev2
        convergents.append(Fraction(h, k))
        h_prev2, h_prev1 = h_prev1, h
        k_prev2, k_prev1 = k_prev1, k
    return convergents


_CF_CONSTANTS: Dict[str, Callable[[int], List[Fraction]]] = {
    "sqrt(2)": lambda count: _convergents_from_terms(_cf_terms_sqrt(2, count)),
    "sqrt(3)": lambda count: _convergents_from_terms(_cf_terms_sqrt(3, count)),
    "sqrt(5)": lambda count: _convergents_from_terms(_cf_terms_sqrt(5, count)),
    "the golden ratio": lambda count: _convergents_from_terms(_cf_terms_golden(count)),
    "e": lambda count: _convergents_from_terms(_cf_terms_e(count)),
    "pi": lambda count: _convergents_from_terms(_PI_CF_TERMS[:count]),
}


@deep_surf("continued_fraction")
def continued_fraction_search(seq: Sequence, deadline: float) -> List[Candidate]:
    vals = list(seq.values)
    n_pts = len(vals)
    if n_pts < 3:
        return []
    start = seq.start_index
    out: List[Candidate] = []
    for name, generator in _CF_CONSTANTS.items():
        if time.monotonic() > deadline:
            break
        try:
            window = generator(n_pts)
        except Exception:
            continue
        if window == vals:
            def predict(n: int, generator=generator, start=start) -> Fraction:
                offset = n - start
                terms = generator(offset + 1)
                return terms[offset]

            out.append(
                Candidate(
                    description=f"a(n) = n-th continued-fraction convergent of {name}",
                    kind=CandidateKind.DICTIONARY,
                    source=f"deep_surf.continued_fraction[{name}]",
                    predict_fn=predict,
                    complexity=6,
                    exact=True,
                    notes="matched against precomputed continued-fraction expansion",
                )
            )
    return out


# --------------------------------------------------------------------------- #
# Deep Surf 5: periodic ("modulo cycle") value sequences
# --------------------------------------------------------------------------- #

@deep_surf("periodic_cycle")
def periodic_cycle_search(seq: Sequence, deadline: float) -> List[Candidate]:
    """Detects a(n) = a(n - p) for all valid n, for the smallest period p
    that is well-confirmed by the data (requires at least one full extra
    cycle beyond the first, to rule out coincidence)."""
    vals = list(seq.values)
    n = len(vals)
    start = seq.start_index
    max_period = n // 2
    for p in range(1, max_period + 1):
        if time.monotonic() > deadline:
            break
        if n < 2 * p + 1:
            break
        if all(vals[i] == vals[i - p] for i in range(p, n)):
            cycle = tuple(vals[:p])

            def predict(nn: int, cycle=cycle, start=start, p=p) -> Fraction:
                return cycle[(nn - start) % p]

            cycle_str = ", ".join(format_number(c) for c in cycle)
            return [
                Candidate(
                    description=f"a(n) repeats with period {p}: [{cycle_str}]",
                    kind=CandidateKind.DICTIONARY,
                    source="deep_surf.periodic_cycle",
                    predict_fn=predict,
                    complexity=2 + p,
                    exact=True,
                    notes=f"repeating cycle of length {p}, confirmed over {n // p} full cycles",
                )
            ]
    return []


# --------------------------------------------------------------------------- #
# Deep Surf 6: compositional search - polynomial exponent of a fixed base
# --------------------------------------------------------------------------- #

def _integer_log(value: int, base: int) -> Optional[int]:
    """Return the integer e with base**e == value exactly, or None."""
    if value <= 0 or base <= 1:
        return None
    if value == 1:
        return 0
    approx = math.log(value) / math.log(base)
    for e in {math.floor(approx) - 1, math.floor(approx), math.ceil(approx), math.ceil(approx) + 1}:
        if e < 0:
            continue
        try:
            if base ** e == value:
                return e
        except OverflowError:
            continue
    return None


@deep_surf("compositional_log_power")
def compositional_log_power_search(seq: Sequence, deadline: float, max_poly_degree: int = 3) -> List[Candidate]:
    """Detects a(n) = base ** P(n) for an integer base and polynomial P,
    by fitting a polynomial to the discrete "exponent sequence" once a
    candidate integer base is found - a fast primitive (polynomial fit)
    composed with another (a geometric base)."""
    vals = list(seq.values)
    if len(vals) < 4 or not seq.is_all_integer() or any(v <= 0 for v in vals):
        return []
    idx = seq.indices()
    int_vals = [v.numerator for v in vals]

    candidate_bases = {2, 3, 5, 6, 7, 10}
    for v in (int_vals[0], int_vals[-1]):
        x, d = v, 2
        while d * d <= x and d < 50:
            if x % d == 0:
                candidate_bases.add(d)
                while x % d == 0:
                    x //= d
            d += 1

    out: List[Candidate] = []
    for base in sorted(candidate_bases):
        if time.monotonic() > deadline:
            break
        if base < 2:
            continue
        exponents: List[Fraction] = []
        ok = True
        for v in int_vals:
            e = _integer_log(v, base)
            if e is None:
                ok = False
                break
            exponents.append(Fraction(e))
        if not ok:
            continue
        coeffs = fit_minimal_polynomial(idx, exponents, min_redundancy=1)
        if coeffs is None:
            continue
        if all(c == 0 for c in coeffs[1:]):
            continue  # degree-0 exponent is just a plain geometric progression
        exponent_expr = build_polynomial_expr(coeffs)

        def predict(n: int, base=base, coeffs=tuple(coeffs)) -> Fraction:
            exponent_val = sum(c * Fraction(n) ** p for p, c in enumerate(coeffs))
            if exponent_val.denominator != 1:
                raise ValueError("fitted exponent is not an integer at this index")
            e = exponent_val.numerator
            return Fraction(base) ** e if e >= 0 else Fraction(1, base ** (-e))

        out.append(
            Candidate(
                description=f"a(n) = {base}^({render_top_level(exponent_expr)})",
                kind=CandidateKind.COMPOSITE,
                source=f"deep_surf.compositional_log_power[base={base}]",
                predict_fn=predict,
                complexity=4 + exponent_expr.complexity(),
                exact=True,
                notes="polynomial fitted to the exponent of a geometric base",
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Deep Surf 7: compositional search - polynomial ratio (product formulas)
# --------------------------------------------------------------------------- #

@deep_surf("compositional_ratio_polynomial")
def compositional_ratio_polynomial_search(seq: Sequence, deadline: float) -> List[Candidate]:
    """Detects a(n) = a(n-1) * R(n) where R(n) is a polynomial in n -
    the shape behind factorial-like product formulas, e.g. a(n) = n!
    (R(n) = n) or a(n) = product of the first n odd numbers (R(n) = 2n-1)."""
    vals = list(seq.values)
    idx = seq.indices()
    if len(vals) < 5 or any(v == 0 for v in vals[:-1]):
        return []
    ratios = [vals[i + 1] / vals[i] for i in range(len(vals) - 1)]
    ratio_idx = idx[1:]
    coeffs = fit_minimal_polynomial(ratio_idx, ratios, min_redundancy=1)
    if coeffs is None or all(c == 0 for c in coeffs[1:]):
        return []  # degree-0 ratio is a plain geometric progression, not this pattern
    expr = build_polynomial_expr(coeffs)
    a_start = vals[0]
    start = seq.start_index

    def predict(n: int, coeffs=tuple(coeffs), a_start=a_start, start=start) -> Fraction:
        if n < start:
            raise ValueError("index before the start of the sequence")
        total = a_start
        for k in range(start + 1, n + 1):
            ratio = sum(c * Fraction(k) ** p for p, c in enumerate(coeffs))
            total *= ratio
        return total

    # The fitted polynomial is printed with "n" as its bound variable (that's
    # what build_polynomial_expr always uses); inside a product running over
    # k it reads more clearly renamed to k. Const's own rendering never
    # contains the letter "n", so this textual substitution is unambiguous.
    ratio_str = render_top_level(expr).replace("n", "k")
    return [
        Candidate(
            description=f"a(n) = a({start}) * prod_{{k={start + 1}}}^n [{ratio_str}]",
            kind=CandidateKind.COMPOSITE,
            source="deep_surf.compositional_ratio_polynomial",
            predict_fn=predict,
            complexity=expr.complexity() + 4,
            exact=True,
            notes="the ratio a(n)/a(n-1) is a polynomial in n (product formula)",
        )
    ]


# --------------------------------------------------------------------------- #
# Deep Surf 8: OEIS lookup (optional, requires network access)
# --------------------------------------------------------------------------- #

@deep_surf("oeis_lookup")
def oeis_lookup_search(seq: Sequence, deadline: float, request_timeout: float = 3.0) -> List[Candidate]:
    """Queries the OEIS (oeis.org) search API for the input sequence.

    Disabled by default (see ``enable_oeis`` on :func:`run_deep_surfs` /
    :func:`sequence_machine.engine.analyze`) because it depends on outbound
    network access. Fails silently - returning no candidates - on any
    network error, timeout, or unexpected response shape, so it is always
    safe to enable speculatively.
    """
    if not seq.is_all_integer():
        return []
    import json
    import urllib.error
    import urllib.request

    terms = ",".join(str(v.numerator) for v in seq.values)
    url = f"https://oeis.org/search?q={terms}&fmt=json"
    remaining = max(0.5, min(request_timeout, deadline - time.monotonic()))
    try:
        with urllib.request.urlopen(url, timeout=remaining) as response:
            payload = json.load(response)
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return []

    results = payload.get("results") if isinstance(payload, dict) else None
    if not results:
        return []

    out: List[Candidate] = []
    for entry in results[:3]:
        oeis_number = entry.get("number")
        name = entry.get("name", "an OEIS sequence")
        data_str = entry.get("data", "")
        offset_str = entry.get("offset", "0,0")
        try:
            oeis_offset = int(offset_str.split(",")[0])
            table = {oeis_offset + i: Fraction(int(x)) for i, x in enumerate(data_str.split(","))}
        except (ValueError, IndexError):
            continue

        def predict(n: int, table=table) -> Fraction:
            if n not in table:
                raise ValueError("index outside the data window returned by OEIS")
            return table[n]

        label = f"A{oeis_number:06d}" if isinstance(oeis_number, int) else "OEIS match"
        out.append(
            Candidate(
                description=f"Matches OEIS {label}: {name}",
                kind=CandidateKind.EXTERNAL,
                source="deep_surf.oeis_lookup",
                predict_fn=predict,
                complexity=8,
                exact=True,
                notes="prediction only covers indices within the terms OEIS returned",
            )
        )
    return out
