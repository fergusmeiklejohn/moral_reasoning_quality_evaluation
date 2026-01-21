"""
Ethics Bowl for Language Models.

A system for orchestrating structured ethical reasoning exchanges between LLMs,
with evaluation by a third model serving as judge.
"""

from .schemas import (
    PhaseType,
    RoundStatus,
    ConsistencyCase,
    EBDilemma,
    Phase,
    Scores,
    Judgment,
    Round,
    RoundConfig,
    TournamentManifest,
    TournamentConfig,
)
from .dilemma_loader import EBDilemmaLoader
from .prompt_builder import PromptBuilder
from .judgment_parser import JudgmentParser
from .storage import EBStorage
from .round_runner import RoundRunner
from .tournament_runner import TournamentRunner
from .analysis import TournamentAnalyzer
from .pattern_extractor import PatternExtractor, ResponsePatterns
from .pattern_aggregator import PatternAggregator
from .review_sampler import ReviewSampler

__all__ = [
    # Schemas
    "PhaseType",
    "RoundStatus",
    "ConsistencyCase",
    "EBDilemma",
    "Phase",
    "Scores",
    "Judgment",
    "Round",
    "RoundConfig",
    "TournamentManifest",
    "TournamentConfig",
    # Components
    "EBDilemmaLoader",
    "PromptBuilder",
    "JudgmentParser",
    "EBStorage",
    "RoundRunner",
    "TournamentRunner",
    "TournamentAnalyzer",
    # Pattern Analysis
    "PatternExtractor",
    "ResponsePatterns",
    "PatternAggregator",
    "ReviewSampler",
]
