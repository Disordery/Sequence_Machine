"""
sequence_machine.cli
======================

The ``sequence-machine`` command-line tool.

Examples
--------
    $ sequence-machine 1 1 2 3 5 8 13
    $ sequence-machine "1, 4, 9, 16, 25" --predict 3
    $ echo "2 4 6 8 10" | sequence-machine
    $ sequence-machine 1 3 8 19 42 89 184 375 758 --json
    $ sequence-machine 1 1 2 3 5 --start-index 1 --verbose
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import List

from .core import AnalysisResult, format_number
from .engine import analyze

_SPLIT_RE = re.compile(r"[\s,]+")


def _tokenize(raw: str) -> List[str]:
    return [tok for tok in _SPLIT_RE.split(raw.strip()) if tok]


def _read_sequence_tokens(args: argparse.Namespace) -> List[str]:
    if args.sequence:
        # Support both `prog 1 2 3` and `prog "1,2,3"` (or a mix).
        tokens: List[str] = []
        for chunk in args.sequence:
            tokens.extend(_tokenize(chunk))
        return tokens
    if sys.stdin.isatty():
        raise SystemExit(
            "No sequence provided. Pass terms as arguments (e.g. `sequence-machine 1 1 2 3 5`) "
            "or pipe them in (e.g. `echo '1 1 2 3 5' | sequence-machine`)."
        )
    return _tokenize(sys.stdin.read())


def _print_human(result: AnalysisResult, top_k: int, predict_n: int, verbose: bool) -> None:
    seq_str = ", ".join(format_number(v) for v in result.sequence.values)
    print(f"Input sequence (n={result.sequence.start_index}..): {seq_str}")

    if verbose:
        mode = "fast surfs only (confident exact match found)" if not result.used_deep_surf else "fast + deep surfs"
        print(f"Search: {mode}  |  elapsed: {result.elapsed_seconds * 1000:.1f} ms")

    if not result.candidates:
        print()
        print("No formula, recurrence, or known pattern was found that exactly explains this sequence.")
        print("Try providing more terms, or re-run with a longer --timeout to let the deep search work harder.")
        return

    print()
    shown = result.candidates[:top_k]
    for rank, candidate in enumerate(shown, start=1):
        marker = "✓" if candidate.exact else "≈"
        print(f"{rank}. [{marker}] {candidate.description}")
        detail = f"     source: {candidate.source}   complexity: {candidate.complexity}   score: {candidate.score:.1f}"
        if not candidate.exact:
            detail += f"   max_error: {candidate.max_abs_error:.4g}"
        print(detail)
        if candidate.notes:
            print(f"     note: {candidate.notes}")

    best = result.best
    if best is not None and predict_n > 0:
        try:
            preds = result.predictions(predict_n)
            pred_str = ", ".join(format_number(p) for p in preds)
            print()
            print(f"Predicted next {predict_n} term(s) (using the top candidate): {pred_str}")
        except (ZeroDivisionError, ValueError, OverflowError):
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sequence-machine",
        description="Discover the closed-form formula, recurrence, or generative rule behind a sequence of numbers.",
    )
    parser.add_argument(
        "sequence",
        nargs="*",
        help="The sequence terms (space and/or comma separated). Reads from stdin if omitted. "
        "Accepts integers, decimals, and fractions like '1/3'.",
    )
    parser.add_argument(
        "--start-index",
        "-s",
        type=int,
        default=0,
        help="The index n associated with the first given term (default: 0).",
    )
    parser.add_argument(
        "--timeout",
        "-t",
        type=float,
        default=4.0,
        help="Wall-clock seconds allotted to the deep search engines, shared across all of them (default: 4.0). "
        "Never used if a fast heuristic already found a confident exact match.",
    )
    parser.add_argument(
        "--top",
        "-k",
        type=int,
        default=3,
        help="Number of ranked candidates to display (default: 3).",
    )
    parser.add_argument(
        "--predict",
        "-p",
        type=int,
        default=5,
        help="Number of out-of-bounds terms to predict using the best candidate (default: 5; use 0 to disable).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the result as JSON instead of a human-readable report.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print extra diagnostic information (timing, which search tier ran).",
    )
    parser.add_argument(
        "--oeis",
        action="store_true",
        help="Enable the OEIS (oeis.org) online lookup deep surf. Requires outbound network access; "
        "silently contributes nothing if the request fails.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Maximum number of concurrent deep-surf worker threads (default: one per registered deep surf).",
    )
    return parser


def main(argv: List[str] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        tokens = _read_sequence_tokens(args)
        result = analyze(
            tokens,
            start_index=args.start_index,
            deep_surf_timeout=args.timeout,
            top_k=args.top,
            predict_n=args.predict,
            enable_oeis=args.oeis,
            max_deep_workers=args.workers,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result.to_dict(predict_n=args.predict, top_k=args.top), indent=2))
    else:
        _print_human(result, top_k=args.top, predict_n=args.predict, verbose=args.verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
