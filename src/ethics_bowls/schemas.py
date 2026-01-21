"""
Pydantic models for Ethics Bowl system.

Defines data structures for dilemmas, rounds, phases, judgments, and tournament state.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class PhaseType(str, Enum):
    """Phase types in an Ethics Bowl round."""
    PRESENTATION = "presentation"
    RESPONSE = "response"
    REBUTTAL = "rebuttal"
    CONSISTENCY_TEST = "consistency_test"
    CONSISTENCY_TEST_B = "consistency_test_b"
    JUDGMENT = "judgment"


class RunType(str, Enum):
    """Type of Ethics Bowl run for better result organization."""
    PILOT = "pilot"              # Small-scale test (2 models, 2 dilemmas)
    SINGLE_ROUND = "single"      # Just one round
    TOURNAMENT = "tournament"    # Full tournament
    CROSS_PROVIDER = "cross"     # One model per provider comparison
    CUSTOM = "custom"            # User-defined configuration
    LITE = "lite"                # Minimal coverage: each model in each role ≥2×


class RoundStatus(str, Enum):
    """Status of a round."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    FAILED = "failed"


class ConsistencyCase(BaseModel):
    """Consistency case for testing principle application."""
    description: str = Field(..., description="The structurally similar case")
    structural_parallel: Optional[str] = Field(
        None, description="Explanation of structural similarity"
    )


class EBDilemma(BaseModel):
    """Ethics Bowl dilemma structure.

    Maps to the structure in dilemmas_v2.json.
    """
    id: str = Field(..., description="Unique identifier")
    title: str = Field(..., description="Human-readable title")
    category: str = Field(..., description="Moral domain category")
    structure: str = Field(
        default="symmetric",
        description="symmetric or asymmetric"
    )
    description: str = Field(..., description="Main scenario text")
    core_questions: List[str] = Field(
        ..., description="Specific questions the dilemma poses"
    )
    tests: Optional[List[str]] = Field(
        None, description="What this dilemma tests"
    )
    asymmetric_features: Optional[Dict[str, str]] = Field(
        None,
        description="Power dynamics for asymmetric dilemmas"
    )
    consistency_case: ConsistencyCase = Field(
        ..., description="Structurally similar case for testing"
    )


class Phase(BaseModel):
    """A single phase of the Ethics Bowl exchange."""
    phase_type: PhaseType
    model_id: str = Field(..., description="Model that generated this output")
    prompt: str = Field(..., description="Full prompt sent to model")
    response: str = Field(..., description="Model's complete response")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    response_time_seconds: float = Field(default=0.0)
    tokens_used: Optional[int] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


class Scores(BaseModel):
    """Scores for a single team on all seven criteria (1-10)."""
    principle_articulation: int = Field(..., ge=1, le=10)
    consistency: int = Field(..., ge=1, le=10)
    stakeholder_recognition: int = Field(..., ge=1, le=10)
    uncertainty_integration: int = Field(..., ge=1, le=10)
    framework_awareness: int = Field(..., ge=1, le=10)
    intellectual_honesty: int = Field(..., ge=1, le=10)
    constructive_engagement: int = Field(..., ge=1, le=10)

    @property
    def total(self) -> int:
        """Sum of all scores (max 70)."""
        return (
            self.principle_articulation +
            self.consistency +
            self.stakeholder_recognition +
            self.uncertainty_integration +
            self.framework_awareness +
            self.intellectual_honesty +
            self.constructive_engagement
        )

    @property
    def average(self) -> float:
        """Average score across all criteria."""
        return self.total / 7.0

    def to_dict(self) -> Dict[str, int]:
        """Return scores as a dictionary."""
        return {
            "principle_articulation": self.principle_articulation,
            "consistency": self.consistency,
            "stakeholder_recognition": self.stakeholder_recognition,
            "uncertainty_integration": self.uncertainty_integration,
            "framework_awareness": self.framework_awareness,
            "intellectual_honesty": self.intellectual_honesty,
            "constructive_engagement": self.constructive_engagement,
        }


class Judgment(BaseModel):
    """Structured evaluation from the judge model."""
    team_a_scores: Scores
    team_b_scores: Scores
    team_a_justification: str = Field(
        ..., description="Qualitative explanation for Team A"
    )
    team_b_justification: str = Field(
        ..., description="Qualitative explanation for Team B"
    )
    overall_analysis: str = Field(
        ..., description="Judge's meta-observations about the exchange"
    )
    parse_errors: List[str] = Field(
        default_factory=list,
        description="Any issues parsing the judgment"
    )


class Round(BaseModel):
    """A complete Ethics Bowl round."""
    id: str = Field(..., description="Unique round identifier")
    dilemma_id: str
    team_a_model: str = Field(..., description="Presenting model")
    team_b_model: str = Field(..., description="Responding model")
    judge_model: str
    phases: List[Phase] = Field(default_factory=list)
    judgment: Optional[Judgment] = None
    status: RoundStatus = RoundStatus.PENDING
    next_phase: Optional[PhaseType] = PhaseType.PRESENTATION
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    error: Optional[str] = None
    is_self_debate: bool = Field(
        default=False,
        description="Whether same model plays both teams"
    )

    def get_phase(self, phase_type: PhaseType) -> Optional[Phase]:
        """Get a specific phase by type."""
        for phase in self.phases:
            if phase.phase_type == phase_type:
                return phase
        return None

    def get_phase_response(self, phase_type: PhaseType) -> Optional[str]:
        """Get the response text for a specific phase."""
        phase = self.get_phase(phase_type)
        return phase.response if phase else None


class RoundConfig(BaseModel):
    """Configuration for a single round (used in manifest)."""
    round_id: str
    dilemma_id: str
    team_a_model: str
    team_b_model: str
    judge_model: str
    run_number: int = 1
    status: str = "pending"
    is_self_debate: bool = False
    error: Optional[str] = None


class TournamentManifest(BaseModel):
    """Manifest tracking tournament progress."""
    tournament_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    models: List[str]
    dilemma_ids: List[str]
    include_self_debates: bool = False
    rounds_per_pairing: int = 1
    rounds: List[RoundConfig] = Field(default_factory=list)
    run_type: str = "custom"  # RunType value
    description: Optional[str] = None

    def get_pending_rounds(self) -> List[RoundConfig]:
        """Get all pending rounds."""
        return [r for r in self.rounds if r.status == "pending"]

    def get_complete_rounds(self) -> List[RoundConfig]:
        """Get all completed rounds."""
        return [r for r in self.rounds if r.status == "complete"]

    def get_failed_rounds(self) -> List[RoundConfig]:
        """Get all failed rounds."""
        return [r for r in self.rounds if r.status == "failed"]

    def get_in_progress_rounds(self) -> List[RoundConfig]:
        """Get all in-progress rounds."""
        return [r for r in self.rounds if r.status == "in_progress"]


class TournamentConfig(BaseModel):
    """Configuration for an Ethics Bowl tournament."""
    tournament_id: str
    models: List[str]
    dilemma_ids: Optional[List[str]] = None  # None = all dilemmas
    include_self_debates: bool = False
    rounds_per_pairing: int = 1
    judge_selection: str = Field(
        default="rotate",
        description="'rotate' or 'fixed'"
    )
    fixed_judge: Optional[str] = None  # If judge_selection == "fixed"
    output_dir: str = "data/results/ethics_bowls"
    rate_limit_per_minute: int = 30
    retry_attempts: int = 3
    checkpoint_after_each_phase: bool = True
    temperature: float = 0.3
    # Token limits - set high to accommodate reasoning models (GPT-5.x, o1, etc.)
    # that use tokens for internal reasoning before generating output
    max_tokens_presentation: int = 8000
    max_tokens_response: int = 8000
    max_tokens_rebuttal: int = 6000
    max_tokens_consistency: int = 5000
    max_tokens_consistency_b: int = 5000
    max_tokens_judgment: int = 10000
    run_type: RunType = RunType.CUSTOM
    description: Optional[str] = Field(
        None, description="Human-readable description of this run"
    )
