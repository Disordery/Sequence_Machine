"""
sequence_machine.core
======================

Core domain models shared by every other module in Sequence Machine:

    * Exact-arithmetic number parsing (``to_fraction``).
    * ``Sequence``          - a cleaned, validated input sequence.
    * ``Expr`` and friends  - a tiny mathematical AST used to represent and
                              evaluate closed-form formulas.
    * ``RecurrenceRule``    - a compact representation of a (possibly
                              non-homogeneous) linear recurrence.
    * ``Candidate``         - a single discovered rule, together with the
                              metadata the scoring engine needs.
    * ``AnalysisResult``    - the final, ranked output of a full analysis.

Nothing in this module knows anything about *how* patterns are found -
that is the job of ``fast_surf.py`` and ``deep_surf.py``. Keeping the
domain model separate from the search heuristics is what lets new
detectors be added later without touching this file.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from typing import Callable, Iterable, List, Optional, Sequence as TypingSequence, Tuple, Union

Number = Union[int, float, Fraction]


# --------------------------------------------------------------------------- #
# Number parsing / cleaning
# --------------------------------------------------------------------------- #

def to_fraction(value: Union[int, float, str, Fraction]) -> Fraction:
    """Convert a raw scalar into an exact :class:`fractions.Fraction`.

    Accepts ints, floats, ``Fraction`` instances, and strings such as
    ``"3"``, ``"-2.5"``, ``"1/3"``, or ``"-7/2"``. Floats are converted via
    their exact binary value's :class:`Fraction` representation (no
    precision is silently lost beyond what the float already carries).

    Raises
    ------
    ValueError
        If the value cannot be interpreted as a rational number.
    """
    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool):  # bool is a subclass of int - reject explicitly
        raise ValueError(f"Boolean is not a valid sequence term: {value!r}")
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ValueError(f"Cannot use non-finite float as a sequence term: {value!r}")
        return Fraction(value)
    if isinstance(value, str):
        token = value.strip().replace("_", "")
        if not token:
            raise ValueError("Empty string is not a valid sequence term")
        try:
            return Fraction(token)
        except (ValueError, ZeroDivisionError):
            pass
        try:
            f = float(token)
        except ValueError as exc:
            raise ValueError(f"Could not parse {value!r} as a number") from exc
        if math.isnan(f) or math.isinf(f):
            raise ValueError(f"Cannot use non-finite value as a sequence term: {value!r}")
        return Fraction(f)
    raise ValueError(f"Unsupported term type: {type(value)!r}")


def format_number(value: Fraction) -> str:
    """Render a Fraction in the most human-readable way available.

    Integers print without a denominator; other rationals print as
    ``p/q``; the function never silently rounds.
    """
    if isinstance(value, Fraction):
        if value.denominator == 1:
            return str(value.numerator)
        return f"{value.numerator}/{value.denominator}"
    return str(value)


# --------------------------------------------------------------------------- #
# Sequence container
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Sequence:
    """A cleaned, validated sequence of exact rational numbers.

    Attributes
    ----------
    values:
        The sequence terms, stored as exact ``Fraction`` objects.
    start_index:
        The index ``n`` associated with ``values[0]`` (0 or 1 are the
        common conventions; any integer is accepted).
    """

    values: Tuple[Fraction, ...]
    start_index: int = 0

    @classmethod
    def from_raw(
        cls,
        raw: Iterable[Union[int, float, str, Fraction]],
        start_index: int = 0,
    ) -> "Sequence":
        """Build a :class:`Sequence` from arbitrary raw scalars.

        Performs the "data cleaning" step: parsing strings/floats into
        exact fractions and validating there is enough data to search.
        """
        parsed = tuple(to_fraction(v) for v in raw)
        if len(parsed) < 2:
            raise ValueError(
                "Sequence Machine needs at least 2 terms to look for a pattern "
                f"(got {len(parsed)})."
            )
        return cls(values=parsed, start_index=start_index)

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, i: int) -> Fraction:
        return self.values[i]

    def __iter__(self):
        return iter(self.values)

    def indices(self) -> List[int]:
        """Absolute indices ``n`` corresponding to each stored value."""
        return list(range(self.start_index, self.start_index + len(self.values)))

    def as_float_tuple(self) -> Tuple[float, ...]:
        return tuple(float(v) for v in self.values)

    def is_all_integer(self) -> bool:
        return all(v.denominator == 1 for v in self.values)

    def differences(self, order: int = 1) -> Tuple[Fraction, ...]:
        """Return the ``order``-th forward finite difference of the sequence."""
        vals = list(self.values)
        for _ in range(order):
            if len(vals) < 2:
                return tuple()
            vals = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
        return tuple(vals)

    def sub(self, start: int, stop: Optional[int] = None) -> "Sequence":
        """A contiguous sub-sequence, preserving correct absolute indices."""
        stop = len(self.values) if stop is None else stop
        return Sequence(values=self.values[start:stop], start_index=self.start_index + start)


# --------------------------------------------------------------------------- #
# Minimal mathematical AST for closed-form expressions
# --------------------------------------------------------------------------- #

class Expr(ABC):
    """Base class for a node in a closed-form expression tree over ``n``."""

    @abstractmethod
    def eval(self, n: Fraction) -> Fraction:
        """Evaluate the expression at integer (or rational) index ``n``."""

    @abstractmethod
    def to_str(self) -> str:
        """Render the expression as a human-readable formula fragment."""

    @abstractmethod
    def complexity(self) -> int:
        """A simple Occam's-razor proxy: total AST node count."""

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.to_str()


@dataclass(frozen=True)
class Const(Expr):
    """A constant rational literal."""

    value: Fraction

    def eval(self, n: Fraction) -> Fraction:
        return self.value

    def to_str(self) -> str:
        return format_number(self.value)

    def complexity(self) -> int:
        return 1


@dataclass(frozen=True)
class Index(Expr):
    """The free variable ``n`` (the sequence index)."""

    def eval(self, n: Fraction) -> Fraction:
        return n

    def to_str(self) -> str:
        return "n"

    def complexity(self) -> int:
        return 1


def _safe_pow(base: Fraction, exponent: Fraction) -> Fraction:
    """Exponentiation that stays exact for integer exponents.

    Falls back to a floating-point evaluation (re-wrapped as a Fraction)
    only when the exponent is genuinely fractional, since ``Fraction**Fraction``
    is not defined in general (e.g. cube roots of non-perfect-cubes).
    """
    if exponent.denominator == 1:
        e = exponent.numerator
        if e >= 0:
            return base ** e
        if base == 0:
            raise ZeroDivisionError("0 cannot be raised to a negative power")
        return Fraction(1, 1) / (base ** (-e))
    # Fractional exponent: evaluate approximately in floating point.
    return Fraction(float(base) ** float(exponent))


@dataclass(frozen=True)
class BinOp(Expr):
    """A binary operation node: ``left OP right``."""

    op: str  # one of '+', '-', '*', '/', '**'
    left: Expr
    right: Expr

    _OPS: "Optional[dict]" = field(default=None, repr=False, compare=False)

    def eval(self, n: Fraction) -> Fraction:
        l, r = self.left.eval(n), self.right.eval(n)
        if self.op == "+":
            return l + r
        if self.op == "-":
            return l - r
        if self.op == "*":
            return l * r
        if self.op == "/":
            if r == 0:
                raise ZeroDivisionError("division by zero while evaluating expression")
            return l / r
        if self.op == "**":
            return _safe_pow(l, r)
        raise ValueError(f"Unknown operator: {self.op!r}")

    def to_str(self) -> str:
        symbol = {"+": "+", "-": "-", "*": "*", "/": "/", "**": "^"}[self.op]
        if self.op == "+" and isinstance(self.right, Const) and self.right.value < 0:
            return f"({self.left.to_str()} - {format_number(-self.right.value)})"
        if self.op == "-" and isinstance(self.right, Const) and self.right.value < 0:
            return f"({self.left.to_str()} + {format_number(-self.right.value)})"
        return f"({self.left.to_str()} {symbol} {self.right.to_str()})"

    def complexity(self) -> int:
        return 1 + self.left.complexity() + self.right.complexity()


@dataclass(frozen=True)
class UnaryOp(Expr):
    """A unary operation node: ``OP child``."""

    op: str  # one of 'neg', 'abs', 'sqrt', 'fact'
    child: Expr

    def eval(self, n: Fraction) -> Fraction:
        v = self.child.eval(n)
        if self.op == "neg":
            return -v
        if self.op == "abs":
            return abs(v)
        if self.op == "sqrt":
            if v < 0:
                raise ValueError("sqrt of a negative number")
            f = math.isqrt(v.numerator) if v.denominator == 1 else None
            if f is not None and f * f == v.numerator:
                return Fraction(f)
            return Fraction(math.sqrt(float(v)))
        if self.op == "fact":
            if v.denominator != 1 or v < 0:
                raise ValueError("factorial requires a non-negative integer")
            return Fraction(math.factorial(v.numerator))
        raise ValueError(f"Unknown unary operator: {self.op!r}")

    def to_str(self) -> str:
        if self.op == "neg":
            return f"-{self.child.to_str()}"
        if self.op == "abs":
            return f"|{self.child.to_str()}|"
        if self.op == "sqrt":
            return f"sqrt({self.child.to_str()})"
        if self.op == "fact":
            return f"{self.child.to_str()}!"
        return f"{self.op}({self.child.to_str()})"

    def complexity(self) -> int:
        return 1 + self.child.complexity()


@dataclass(frozen=True)
class Func(Expr):
    """A named function applied to argument sub-expressions, e.g. ``fib(n)``."""

    name: str
    args: Tuple[Expr, ...]
    impl: Callable[..., Fraction] = field(compare=False)

    def eval(self, n: Fraction) -> Fraction:
        return self.impl(*(a.eval(n) for a in self.args))

    def to_str(self) -> str:
        return f"{self.name}({', '.join(a.to_str() for a in self.args)})"

    def complexity(self) -> int:
        return 1 + sum(a.complexity() for a in self.args)


def render_top_level(expr: Expr) -> str:
    """Render an expression the way it should look as a top-level formula:
    like ``Expr.to_str()``, but without the single redundant outermost
    parenthesis pair that ``BinOp`` always adds for unambiguous nesting.
    """
    s = expr.to_str()
    if s.startswith("(") and s.endswith(")"):
        return s[1:-1]
    return s


def build_polynomial_expr(coeffs: TypingSequence[Fraction]) -> Expr:
    """Build an ``Expr`` tree for ``coeffs[0] + coeffs[1]*n + coeffs[2]*n**2 + ...``.

    Zero coefficients are omitted from the tree for readability. Returns
    ``Const(0)`` if every coefficient is zero.
    """
    terms: List[Expr] = []
    for power, c in enumerate(coeffs):
        if c == 0:
            continue
        if power == 0:
            terms.append(Const(c))
        elif power == 1:
            term: Expr = Index()
            if c != 1:
                term = BinOp("*", Const(c), term)
            terms.append(term)
        else:
            term = BinOp("**", Index(), Const(Fraction(power)))
            if c != 1:
                term = BinOp("*", Const(c), term)
            terms.append(term)
    if not terms:
        return Const(Fraction(0))
    expr = terms[0]
    for t in terms[1:]:
        expr = BinOp("+", expr, t)
    return expr


# --------------------------------------------------------------------------- #
# Shared exact numeric primitives (used by both fast_surf and deep_surf)
# --------------------------------------------------------------------------- #

def solve_linear_system(matrix: List[List[Fraction]], rhs: List[Fraction]) -> Optional[List[Fraction]]:
    """Exact Gauss-Jordan elimination over the rationals.

    ``matrix`` is a square (k x k) matrix of Fractions, ``rhs`` a length-k
    vector. Returns the solution vector, or ``None`` if the system is
    singular. Used anywhere a small exact linear system needs solving:
    polynomial interpolation, non-homogeneous recurrence fitting, etc.
    """
    n = len(matrix)
    aug = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot_row = None
        for r in range(col, n):
            if aug[r][col] != 0:
                pivot_row = r
                break
        if pivot_row is None:
            return None
        aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
        pivot_val = aug[col][col]
        aug[col] = [x / pivot_val for x in aug[col]]
        for r in range(n):
            if r != col and aug[r][col] != 0:
                factor = aug[r][col]
                aug[r] = [a - factor * b for a, b in zip(aug[r], aug[col])]
    return [aug[i][n] for i in range(n)]


def forward_differences(vals: TypingSequence[Fraction]) -> List[List[Fraction]]:
    """All forward-difference levels of ``vals``, starting with ``vals`` itself."""
    levels: List[List[Fraction]] = [list(vals)]
    while len(levels[-1]) > 1:
        prev = levels[-1]
        levels.append([prev[i + 1] - prev[i] for i in range(len(prev) - 1)])
    return levels


def fit_minimal_polynomial(
    indices: TypingSequence[int], values: TypingSequence[Fraction], min_redundancy: int = 1
) -> Optional[List[Fraction]]:
    """Find the lowest-degree polynomial that exactly fits ``values`` at
    ``indices``, requiring at least ``min_redundancy`` confirming points
    beyond the minimum needed - otherwise every length-k sequence trivially
    "fits" a degree (k-1) polynomial and the result would be meaningless
    overfitting rather than a genuine discovery.

    Returns coefficients ``[a0, a1, a2, ...]`` (lowest degree first) for
    ``a0 + a1*n + a2*n**2 + ...``, or ``None`` if no such polynomial exists
    within the available redundancy budget.
    """
    values = list(values)
    n_points = len(values)
    if n_points < 2 + min_redundancy:
        return None
    diffs = forward_differences(values)
    max_degree = n_points - 1 - min_redundancy
    for d in range(0, max_degree + 1):
        level = diffs[d]
        if len(level) < 1 + min_redundancy:
            break
        if all(x == level[0] for x in level):
            matrix = [[Fraction(n) ** p for p in range(d + 1)] for n in indices[: d + 1]]
            coeffs = solve_linear_system(matrix, list(values[: d + 1]))
            if coeffs is None:
                continue
            return coeffs
    return None


def simplify_expr(expr: Expr, _passes: int = 4) -> Expr:
    """A small, purely-algebraic cleanup pass for closed-form expressions.

    Folds constant sub-expressions, drops additive/multiplicative
    identities (``x+0``, ``x*1``, ...), and collapses trivially-zero
    differences (``x-x``). This does not change what an expression
    *computes* - it exists so that search engines like the genetic-
    programming symbolic regressor (which routinely produces "dead"
    sub-expressions such as ``+(n-n)``) can present a formula a human
    would actually want to read, and so its reported complexity reflects
    the formula's true simplicity rather than incidental search bloat.
    """
    current = expr
    for _ in range(_passes):
        simplified = _simplify_once(current)
        if simplified == current:
            return simplified
        current = simplified
    return current


def _simplify_once(expr: Expr) -> Expr:
    if isinstance(expr, BinOp):
        left = _simplify_once(expr.left)
        right = _simplify_once(expr.right)
        if isinstance(left, Const) and isinstance(right, Const):
            try:
                return Const(BinOp(expr.op, left, right).eval(Fraction(0)))
            except (ZeroDivisionError, ValueError, OverflowError):
                pass
        if expr.op == "+":
            if isinstance(right, Const) and right.value == 0:
                return left
            if isinstance(left, Const) and left.value == 0:
                return right
        elif expr.op == "-":
            if isinstance(right, Const) and right.value == 0:
                return left
            if left == right:
                return Const(Fraction(0))
        elif expr.op == "*":
            if (isinstance(left, Const) and left.value == 0) or (isinstance(right, Const) and right.value == 0):
                return Const(Fraction(0))
            if isinstance(right, Const) and right.value == 1:
                return left
            if isinstance(left, Const) and left.value == 1:
                return right
        elif expr.op == "/":
            if isinstance(right, Const) and right.value == 1:
                return left
            if left == right:
                return Const(Fraction(1))
        elif expr.op == "**":
            if isinstance(right, Const) and right.value == 0:
                return Const(Fraction(1))
            if isinstance(right, Const) and right.value == 1:
                return left
        return BinOp(expr.op, left, right)
    if isinstance(expr, UnaryOp):
        child = _simplify_once(expr.child)
        if expr.op == "neg" and isinstance(child, UnaryOp) and child.op == "neg":
            return child.child
        if isinstance(child, Const):
            try:
                return Const(UnaryOp(expr.op, child).eval(Fraction(0)))
            except (ZeroDivisionError, ValueError, OverflowError):
                pass
        return UnaryOp(expr.op, child)
    return expr


# --------------------------------------------------------------------------- #
# Recurrence relations
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class RecurrenceRule:
    """A (possibly non-homogeneous) constant-coefficient linear recurrence.

    Represents::

        a(n) = coeffs[0]*a(n-1) + coeffs[1]*a(n-2) + ... + coeffs[k-1]*a(n-k)
               + inhomogeneous(n)          # optional extra term

    seeded with ``initial_terms`` for the first ``order`` values.
    """

    order: int
    coeffs: Tuple[Fraction, ...]
    initial_terms: Tuple[Fraction, ...]
    start_index: int = 0
    inhomogeneous: Optional[Expr] = None

    def generate(self, count: int) -> List[Fraction]:
        """Generate the first ``count`` terms starting at ``start_index``."""
        if count <= len(self.initial_terms):
            return list(self.initial_terms[:count])
        vals = list(self.initial_terms)
        while len(vals) < count:
            n = self.start_index + len(vals)
            total = Fraction(0)
            for i, c in enumerate(self.coeffs):
                total += c * vals[-(i + 1)]
            if self.inhomogeneous is not None:
                total += self.inhomogeneous.eval(Fraction(n))
            vals.append(total)
        return vals

    def value_at(self, n: int) -> Fraction:
        """Evaluate a(n) for an absolute index ``n`` (recomputes from the start)."""
        if n < self.start_index:
            raise IndexError("index before the start of the recurrence")
        return self.generate(n - self.start_index + 1)[-1]

    def to_str(self) -> str:
        parts = []
        for i, c in enumerate(self.coeffs):
            if c == 0:
                continue
            parts.append(f"{format_number(c)}*a(n-{i + 1})")
        rhs = " + ".join(parts) if parts else "0"
        if self.inhomogeneous is not None:
            rhs += f" + {self.inhomogeneous.to_str()}"
        seeds = ", ".join(
            f"a({self.start_index + i})={format_number(v)}" for i, v in enumerate(self.initial_terms)
        )
        return f"a(n) = {rhs}   [with {seeds}]"

    def complexity(self) -> int:
        base = 2 * self.order + len(self.initial_terms)
        if self.inhomogeneous is not None:
            base += self.inhomogeneous.complexity()
        return base


# --------------------------------------------------------------------------- #
# Candidate rules and final results
# --------------------------------------------------------------------------- #

class CandidateKind(str, Enum):
    """The broad category a discovered rule falls into."""

    CLOSED_FORM = "closed_form"
    RECURRENCE = "recurrence"
    DICTIONARY = "dictionary"
    PIECEWISE = "piecewise"
    COMPOSITE = "composite"
    SYMBOLIC = "symbolic_regression"
    EXTERNAL = "external_lookup"


@dataclass
class Candidate:
    """A single discovered rule for a sequence, plus scoring metadata.

    ``predict_fn`` is the single source of truth for the candidate's
    behaviour: given an absolute index ``n`` it must return the predicted
    value at that index. Everything else on this object exists to let the
    scoring engine rank candidates and to let callers render them nicely.
    """

    description: str
    kind: CandidateKind
    source: str
    predict_fn: Callable[[int], Fraction]
    complexity: int
    exact: bool = False
    max_abs_error: float = 0.0
    notes: str = ""
    score: float = field(default=0.0, init=False)

    def predict(self, n: int) -> Fraction:
        return self.predict_fn(n)

    def predict_many(self, indices: Iterable[int]) -> List[Fraction]:
        return [self.predict_fn(n) for n in indices]

    def verify(self, sequence: Sequence) -> Tuple[bool, float]:
        """Check this candidate against the *known* data.

        Returns ``(is_exact, max_absolute_error)`` over the whole input
        sequence. A candidate whose ``predict_fn`` raises on any known
        index is treated as failing verification (infinite error).
        """
        max_err = 0.0
        exact = True
        for n, expected in zip(sequence.indices(), sequence.values):
            try:
                predicted = self.predict_fn(n)
            except (ZeroDivisionError, ValueError, OverflowError):
                return False, math.inf
            err = abs(float(predicted) - float(expected))
            if predicted != expected:
                exact = False
            max_err = max(max_err, err)
        return exact, max_err


@dataclass
class AnalysisResult:
    """The final, ranked output of :func:`sequence_machine.engine.analyze`."""

    sequence: Sequence
    candidates: List[Candidate]
    elapsed_seconds: float
    surfs_run: List[str]
    used_deep_surf: bool

    @property
    def best(self) -> Optional[Candidate]:
        return self.candidates[0] if self.candidates else None

    def predictions(self, count: int = 5) -> List[Fraction]:
        """Out-of-bounds prediction of the next ``count`` terms using the
        best candidate, or an empty list if nothing was found."""
        if self.best is None:
            return []
        next_start = self.sequence.start_index + len(self.sequence)
        return self.best.predict_many(range(next_start, next_start + count))

    def to_dict(self, predict_n: int = 5, top_k: int = 5) -> dict:
        """A JSON-serialisable summary, suitable for ``--json`` CLI output."""
        top = self.candidates[:top_k]
        return {
            "input": [format_number(v) for v in self.sequence.values],
            "start_index": self.sequence.start_index,
            "elapsed_seconds": round(self.elapsed_seconds, 4),
            "surfs_run": self.surfs_run,
            "used_deep_surf": self.used_deep_surf,
            "best": _candidate_to_dict(self.best, self.sequence, predict_n) if self.best else None,
            "candidates": [_candidate_to_dict(c, self.sequence, predict_n) for c in top],
        }


def _candidate_to_dict(candidate: Candidate, sequence: Sequence, predict_n: int) -> dict:
    next_start = sequence.start_index + len(sequence)
    try:
        preds = [format_number(candidate.predict(n)) for n in range(next_start, next_start + predict_n)]
    except (ZeroDivisionError, ValueError, OverflowError):
        preds = []
    return {
        "description": candidate.description,
        "kind": candidate.kind.value,
        "source": candidate.source,
        "exact": candidate.exact,
        "max_abs_error": candidate.max_abs_error,
        "complexity": candidate.complexity,
        "score": round(candidate.score, 4),
        "notes": candidate.notes,
        "next_terms": preds,
    }
