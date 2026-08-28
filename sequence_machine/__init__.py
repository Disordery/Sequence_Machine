"""
Sequence Machine
================

A library and CLI for discovering the closed-form formula, recurrence
relation, or generative rule behind a sequence of numbers.

Public API
----------
    analyze(raw_values, ...)   -> AnalysisResult   # main entry point
    Sequence                                        # input container
    Candidate, CandidateKind                        # a discovered rule
    AnalysisResult                                  # ranked output

Example
-------
    >>> from sequence_machine import analyze
    >>> result = analyze([1, 1, 2, 3, 5, 8, 13])
    >>> print(result.best.description)
    a(n) = a(n-1) + a(n-2)
"""

from .core import (
    Sequence,
    Candidate,
    CandidateKind,
    AnalysisResult,
    Const,
    Index,
    BinOp,
    UnaryOp,
    Expr,
    RecurrenceRule,
)
from .engine import analyze, SequenceMachine

__all__ = [
    "analyze",
    "SequenceMachine",
    "Sequence",
    "Candidate",
    "CandidateKind",
    "AnalysisResult",
    "Expr",
    "Const",
    "Index",
    "BinOp",
    "UnaryOp",
    "RecurrenceRule",
]

__version__ = "0.1.0"
