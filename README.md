# Sequence Machine

Give it a sequence of numbers - integers, floats, rational fractions, or
mixed - and it tries to find the formula, recurrence, or generative rule
behind them.

```
$ sequence-machine 1 1 2 3 5 8 13 21
Input sequence (n=0..): 1, 1, 2, 3, 5, 8, 13, 21

1. [✓] a(n) = Fibonacci(n+1)
     source: fast_surf.dictionary[Fibonacci]   complexity: 4   score: 980.0
2. [✓] a(n) = 1*a(n-1) + 1*a(n-2)   [with a(0)=1, a(1)=1]
     source: fast_surf.linear_recurrence   complexity: 6   score: 970.0
3. [✓] a(n) = a(0) + sum_{k=1}^n [a(n) = Fibonacci(n-1)]
     source: fast_surf.cumulative_sum   complexity: 9   score: 965.0

Predicted next 5 term(s) (using the top candidate): 34, 55, 89, 144, 233
```

## Install

```
pip install -e .
```

This installs the `sequence-machine` command and the `sequence_machine`
Python package. There are no required third-party dependencies.

## Command line usage

```
sequence-machine 1 4 9 16 25 36 49          # a few numbers as arguments
sequence-machine "1, 3, 6, 10, 15"          # comma-separated also works
echo "2 4 6 8 10" | sequence-machine        # or pipe them in
sequence-machine 1 1 2 3 5 --start-index 1  # a(1)=1, a(2)=1, a(3)=2, ...
sequence-machine 1 2 6 24 120 --json        # machine-readable output
sequence-machine 1 2 4 7 11 16 --timeout 8  # give the deep search more time
```

Run `sequence-machine --help` for the full flag list (`--top`, `--predict`,
`--verbose`, `--oeis`, `--workers`, ...).

## Library usage

```python
from sequence_machine import analyze

result = analyze([1, 1, 2, 3, 5, 8, 13])
print(result.best.description)         # "a(n) = Fibonacci(n+1)"
print(result.predictions(5))           # next 5 terms, as Fractions

for candidate in result.candidates[:3]:
    print(candidate.score, candidate.source, candidate.description)
```

`analyze()` accepts ints, floats, `Fraction`s, or numeric strings
(including `"1/3"`-style fractions), an optional `start_index`, and a
handful of tuning knobs - see its docstring for the full parameter list.

## How it works

Sequence Machine uses a two-tier search, matching the mental model of
"try something cheap first, only reach for heavy machinery if you have
to":

1. **Fast Surfs** (`fast_surf.py`) - cheap, deterministic heuristics that
   run in full, every time: constant/polynomial fitting via finite
   differences, geometric/exponential ratio checks, linear recurrences via
   an exact-arithmetic Berlekamp-Massey pass, simple transforms
   (alternating sign, even/odd interleaving, cumulative sums), and a
   dictionary of ~14 named integer sequences (Fibonacci, Catalan,
   factorials, primes, Bell numbers, ...).

2. **Deep Surfs** (`deep_surf.py`) - heavier, time-budgeted searches that
   only run if no Fast Surf produced a *confident* exact match: a small
   tree-based genetic-programming symbolic regressor, a non-homogeneous /
   higher-order linear recurrence solver (handles forcing terms like
   `a(n) = a(n-1) + n**2`), number-theoretic function matching (Euler's
   totient, divisor functions, digit sums, ...), continued-fraction
   convergent recognition (√2, √3, √5, φ, e, π), periodic-cycle detection,
   two "compositional" searches that fit a polynomial to the *exponent* or
   *ratio* of the sequence (catching things like `2**(n**2)` or product
   formulas like `n!`), and an optional OEIS API lookup.

Every Deep Surf receives a shared wall-clock deadline and is dispatched
concurrently via a thread pool, so one slow search can't starve the
others - see the "a note on concurrency" section at the top of
`deep_surf.py` for the GIL trade-off this implies and how you'd move to
true multi-core parallelism in production.

3. **Scoring** (`engine.py`) - every candidate that comes back (from
   either tier) is independently re-verified against the input, then
   ranked: exact matches beat inexact ones, and among exact matches,
   fewer AST/recurrence nodes wins (Occam's razor). The short-circuit
   decision - "was the Fast Surf's best answer good enough to skip Deep
   Surfs?" - is based on the actual best-ranked Fast Surf candidate, not
   just "did anything match": a bare `RECURRENCE` match specifically
   always gets double-checked by the non-homogeneous recurrence search,
   since Berlekamp-Massey can only find *homogeneous* recurrences and a
   much simpler non-homogeneous rule is often hiding underneath one.

## Project layout

```
sequence_machine/
    core.py       domain models: Sequence, the closed-form AST (Expr/Const/
                  BinOp/...), RecurrenceRule, Candidate, AnalysisResult,
                  and shared exact-arithmetic helpers (Gaussian elimination,
                  polynomial fitting, expression simplification).
    fast_surf.py  the Fast Surf heuristics + their @fast_surf registry.
    deep_surf.py  the Deep Surf search engines + their @deep_surf registry.
    engine.py     orchestration: run fast, decide whether to run deep,
                  score, rank, return.
    cli.py        the `sequence-machine` command-line tool.
tests/
    test_sequence_machine.py   unittest suite (no extra dependencies).
```

## Adding a new detector

New heuristics don't require touching `engine.py`. Register a function
with the matching decorator and it's picked up automatically:

```python
from sequence_machine.core import Candidate, CandidateKind, Sequence
from sequence_machine.fast_surf import fast_surf   # or: deep_surf.deep_surf

@fast_surf("my_new_heuristic")
def detect_my_pattern(seq: Sequence) -> list[Candidate]:
    ...  # return [] if it doesn't apply
```

A Fast Surf takes just a `Sequence`; a Deep Surf additionally takes a
`deadline: float` (an absolute `time.monotonic()` timestamp) and should
stop searching once `time.monotonic() >= deadline`. Third-party code that
can't edit `fast_surf.py`/`deep_surf.py` directly can use
`register_fast_surf(name, fn)` / `register_deep_surf(name, fn)` instead of
the decorators.

## Notes and limitations

- **OEIS lookup** (`deep_surf.oeis_lookup_search`) queries `oeis.org` over
  the network and is **off by default** (`--oeis` / `enable_oeis=True` to
  turn it on). It fails silently - contributing no candidates - if the
  network is unreachable, so it's always safe to enable speculatively.
- **Symbolic regression** is a best-effort tree search; it will not always
  find an existing closed form within its time budget, and (by design)
  only ever reports a result once it's been independently re-verified as
  either exact or extremely close (≤1e-6 max error, to allow for floating
  point noise in irrational-looking targets).
- Recurrence-generated predictions recompute the whole sequence from the
  start on each call rather than memoizing, which is simple and correct
  but O(n) (or O(n·order)) per prediction - fine for the CLI's typical
  "predict the next few terms" use case, not tuned for predicting very
  distant indices of high-order recurrences.
