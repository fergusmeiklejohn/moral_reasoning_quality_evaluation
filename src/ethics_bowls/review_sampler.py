"""
Review sampler for human verification of pattern analysis.

Generates curated samples of evidence for each identified pattern,
allowing humans to verify LLM-extracted findings.
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from pathlib import Path

from .schemas import Round, PhaseType
from .storage import EBStorage
from .pattern_extractor import ResponsePatterns
from .pattern_aggregator import PatternAggregator, CrossModelPatterns


@dataclass
class EvidenceSample:
    """A sample of evidence for human review."""
    pattern_type: str  # e.g., "stakeholder_neglect", "framework_preference"
    pattern_description: str
    model_id: str
    round_id: str
    dilemma_id: str
    quote: str
    context: str  # Surrounding context
    confidence: str  # "high", "medium", "low"
    verification_question: str  # Question for human reviewer


@dataclass
class PatternEvidence:
    """Collection of evidence for a single pattern."""
    pattern_type: str
    pattern_description: str
    confidence: str
    samples: List[EvidenceSample] = field(default_factory=list)
    verification_notes: str = ""


class ReviewSampler:
    """Generate curated samples for human verification of patterns."""

    def __init__(
        self,
        patterns: List[ResponsePatterns],
        tournament_dir: str,
        sample_size: int = 5,
    ):
        """
        Initialize review sampler.

        Args:
            patterns: Extracted patterns from PatternExtractor
            tournament_dir: Path to tournament results for loading full responses
            sample_size: Number of samples per pattern
        """
        self.patterns = patterns
        self.storage = EBStorage(tournament_dir)
        self.rounds = {r.id: r for r in self.storage.load_rounds()}
        self.sample_size = sample_size
        self.aggregator = PatternAggregator(patterns)

    def _get_response_text(self, round_id: str, role: str) -> Optional[str]:
        """Get the full response text for a round/role."""
        round_obj = self.rounds.get(round_id)
        if not round_obj:
            return None

        if role == "presenting":
            phase = round_obj.get_phase(PhaseType.PRESENTATION)
        else:
            phase = round_obj.get_phase(PhaseType.RESPONSE)

        return phase.response if phase else None

    def _extract_quote_context(
        self, full_text: str, quote: str, context_chars: int = 200
    ) -> str:
        """Extract quote with surrounding context."""
        if not full_text or not quote:
            return ""

        idx = full_text.find(quote[:50])  # Find by first 50 chars
        if idx < 0:
            return quote

        start = max(0, idx - context_chars)
        end = min(len(full_text), idx + len(quote) + context_chars)
        context = full_text[start:end]

        if start > 0:
            context = "..." + context
        if end < len(full_text):
            context = context + "..."

        return context

    def sample_stakeholder_patterns(self) -> List[PatternEvidence]:
        """Generate samples for stakeholder recognition patterns."""
        evidence_list = []
        stakeholders = self.aggregator.aggregate_stakeholders()

        # Focus on neglected categories
        for category, pattern in stakeholders.items():
            if pattern.coverage_rate < 0.5:  # Low coverage
                evidence = PatternEvidence(
                    pattern_type="stakeholder_neglect",
                    pattern_description=f"Stakeholder category '{category}' mentioned in only {pattern.coverage_rate*100:.0f}% of responses",
                    confidence="medium",
                )

                # Find examples where this category WAS mentioned (positive examples)
                positive_samples = []
                negative_samples = []

                for p in self.patterns:
                    mentions = getattr(p.stakeholders, category, [])
                    if mentions:
                        positive_samples.append((p, mentions))
                    else:
                        negative_samples.append(p)

                # Add positive examples (shows what mentioning looks like)
                for p, mentions in positive_samples[:2]:
                    response_text = self._get_response_text(p.round_id, p.role)
                    evidence.samples.append(EvidenceSample(
                        pattern_type="stakeholder_mention",
                        pattern_description=f"Model mentioned {category} stakeholders: {mentions[:3]}",
                        model_id=p.model_id,
                        round_id=p.round_id,
                        dilemma_id=p.dilemma_id,
                        quote=f"Mentioned: {', '.join(mentions[:3])}",
                        context=response_text[:500] if response_text else "",
                        confidence="high",
                        verification_question=f"Does this response appropriately consider {category} stakeholders?",
                    ))

                # Add negative examples (where category was absent)
                for p in negative_samples[:3]:
                    response_text = self._get_response_text(p.round_id, p.role)
                    evidence.samples.append(EvidenceSample(
                        pattern_type="stakeholder_neglect",
                        pattern_description=f"Model did not mention any {category} stakeholders",
                        model_id=p.model_id,
                        round_id=p.round_id,
                        dilemma_id=p.dilemma_id,
                        quote="[No mentions of this category]",
                        context=response_text[:500] if response_text else "",
                        confidence="medium",
                        verification_question=f"Should this response have considered {category} stakeholders?",
                    ))

                evidence.verification_notes = (
                    f"Human should verify: (1) Are the extractions accurate? "
                    f"(2) Is neglecting {category} stakeholders problematic for this dilemma?"
                )
                evidence_list.append(evidence)

        return evidence_list

    def sample_framework_patterns(self) -> List[PatternEvidence]:
        """Generate samples for ethical framework patterns."""
        evidence_list = []
        frameworks = self.aggregator.aggregate_frameworks()

        # Underused frameworks
        for fw, pattern in frameworks.items():
            if pattern.overall_usage_rate < 0.3:
                evidence = PatternEvidence(
                    pattern_type="framework_underuse",
                    pattern_description=f"Framework '{fw}' used in only {pattern.overall_usage_rate*100:.0f}% of responses",
                    confidence="medium",
                )

                # Find examples where this framework WAS used
                for p in self.patterns:
                    if getattr(p.frameworks, fw, False):
                        quote = p.frameworks.framework_quotes.get(fw, "")
                        response_text = self._get_response_text(p.round_id, p.role)
                        evidence.samples.append(EvidenceSample(
                            pattern_type="framework_use",
                            pattern_description=f"Model used {fw} reasoning",
                            model_id=p.model_id,
                            round_id=p.round_id,
                            dilemma_id=p.dilemma_id,
                            quote=quote,
                            context=self._extract_quote_context(response_text, quote) if response_text else "",
                            confidence="high",
                            verification_question=f"Is this correctly identified as {fw} reasoning?",
                        ))
                        if len(evidence.samples) >= self.sample_size:
                            break

                evidence.verification_notes = (
                    f"Human should verify: (1) Are framework identifications accurate? "
                    f"(2) Would {fw} reasoning be valuable for these dilemmas?"
                )
                evidence_list.append(evidence)

            # Overused frameworks (potential monoculture)
            elif pattern.overall_usage_rate > 0.8:
                evidence = PatternEvidence(
                    pattern_type="framework_overuse",
                    pattern_description=f"Framework '{fw}' used in {pattern.overall_usage_rate*100:.0f}% of responses (potential monoculture)",
                    confidence="high",
                )

                for p in self.patterns:
                    if getattr(p.frameworks, fw, False):
                        quote = p.frameworks.framework_quotes.get(fw, "")
                        response_text = self._get_response_text(p.round_id, p.role)
                        evidence.samples.append(EvidenceSample(
                            pattern_type="framework_dominance",
                            pattern_description=f"Model defaulted to {fw} reasoning",
                            model_id=p.model_id,
                            round_id=p.round_id,
                            dilemma_id=p.dilemma_id,
                            quote=quote,
                            context=self._extract_quote_context(response_text, quote) if response_text else "",
                            confidence="high",
                            verification_question=f"Is this over-reliance on {fw} problematic?",
                        ))
                        if len(evidence.samples) >= self.sample_size:
                            break

                evidence.verification_notes = (
                    f"Human should verify: (1) Is this framework dominance concerning? "
                    f"(2) Would alternative frameworks provide different insights?"
                )
                evidence_list.append(evidence)

        return evidence_list

    def sample_uncertainty_patterns(self) -> List[PatternEvidence]:
        """Generate samples for uncertainty/confidence patterns."""
        evidence_list = []
        uncertainty = self.aggregator.aggregate_uncertainty()

        for model, pattern in uncertainty.items():
            # Low uncertainty models (potential false confidence)
            if pattern.avg_uncertainty_score < 0.3:
                evidence = PatternEvidence(
                    pattern_type="low_uncertainty",
                    pattern_description=f"Model '{model}' shows low uncertainty (avg score: {pattern.avg_uncertainty_score:.2f})",
                    confidence="medium",
                )

                for p in self.patterns:
                    if p.model_id == model and p.uncertainty.confidence_claims:
                        response_text = self._get_response_text(p.round_id, p.role)
                        for claim in p.uncertainty.confidence_claims[:2]:
                            evidence.samples.append(EvidenceSample(
                                pattern_type="confidence_claim",
                                pattern_description="Model expresses confidence",
                                model_id=model,
                                round_id=p.round_id,
                                dilemma_id=p.dilemma_id,
                                quote=claim,
                                context=self._extract_quote_context(response_text, claim) if response_text else "",
                                confidence="medium",
                                verification_question="Is this confidence warranted, or is it false confidence?",
                            ))
                    if len(evidence.samples) >= self.sample_size:
                        break

                evidence.verification_notes = (
                    "Human should verify: Are these confidence claims appropriate "
                    "given the genuine uncertainty in ethical reasoning?"
                )
                evidence_list.append(evidence)

        return evidence_list

    def sample_consistency_patterns(self) -> List[PatternEvidence]:
        """Generate samples for consistency patterns."""
        evidence_list = []
        consistency = self.aggregator.aggregate_consistency()

        for model, pattern in consistency.items():
            if pattern.consistency_rate < 0.8 and pattern.common_violations:
                evidence = PatternEvidence(
                    pattern_type="consistency_violation",
                    pattern_description=f"Model '{model}' consistency rate: {pattern.consistency_rate*100:.0f}%",
                    confidence="medium",
                )

                for p in self.patterns:
                    if p.model_id == model and p.consistency and not p.consistency.is_consistent:
                        round_obj = self.rounds.get(p.round_id)
                        presentation = round_obj.get_phase(PhaseType.PRESENTATION) if round_obj else None
                        consistency_test = round_obj.get_phase(PhaseType.CONSISTENCY_TEST) if round_obj else None

                        evidence.samples.append(EvidenceSample(
                            pattern_type="inconsistency",
                            pattern_description=f"Violation: {p.consistency.consistency_violations[0] if p.consistency.consistency_violations else 'Unknown'}",
                            model_id=model,
                            round_id=p.round_id,
                            dilemma_id=p.dilemma_id,
                            quote=p.consistency.consistency_analysis,
                            context=(
                                f"PRESENTATION: {presentation.response[:300] if presentation else ''}...\n\n"
                                f"CONSISTENCY TEST: {consistency_test.response[:300] if consistency_test else ''}..."
                            ),
                            confidence="medium",
                            verification_question="Does the model actually violate its stated principles?",
                        ))
                        if len(evidence.samples) >= self.sample_size:
                            break

                evidence.verification_notes = (
                    "Human should verify: (1) Are the identified violations real? "
                    "(2) Could the divergence be explained by relevant differences in the cases?"
                )
                evidence_list.append(evidence)

        return evidence_list

    def generate_review_report(self) -> Dict[str, Any]:
        """Generate complete review report for human verification."""
        stakeholder_evidence = self.sample_stakeholder_patterns()
        framework_evidence = self.sample_framework_patterns()
        uncertainty_evidence = self.sample_uncertainty_patterns()
        consistency_evidence = self.sample_consistency_patterns()

        return {
            "metadata": {
                "n_patterns_analyzed": len(self.patterns),
                "n_rounds": len(self.rounds),
                "sample_size": self.sample_size,
            },
            "stakeholder_patterns": [
                {
                    "pattern_type": e.pattern_type,
                    "description": e.pattern_description,
                    "confidence": e.confidence,
                    "verification_notes": e.verification_notes,
                    "samples": [
                        {
                            "model": s.model_id,
                            "dilemma": s.dilemma_id,
                            "round": s.round_id,
                            "quote": s.quote,
                            "context": s.context[:500],
                            "question": s.verification_question,
                        }
                        for s in e.samples
                    ],
                }
                for e in stakeholder_evidence
            ],
            "framework_patterns": [
                {
                    "pattern_type": e.pattern_type,
                    "description": e.pattern_description,
                    "confidence": e.confidence,
                    "verification_notes": e.verification_notes,
                    "samples": [
                        {
                            "model": s.model_id,
                            "dilemma": s.dilemma_id,
                            "round": s.round_id,
                            "quote": s.quote,
                            "context": s.context[:500],
                            "question": s.verification_question,
                        }
                        for s in e.samples
                    ],
                }
                for e in framework_evidence
            ],
            "uncertainty_patterns": [
                {
                    "pattern_type": e.pattern_type,
                    "description": e.pattern_description,
                    "confidence": e.confidence,
                    "verification_notes": e.verification_notes,
                    "samples": [
                        {
                            "model": s.model_id,
                            "dilemma": s.dilemma_id,
                            "round": s.round_id,
                            "quote": s.quote,
                            "question": s.verification_question,
                        }
                        for s in e.samples
                    ],
                }
                for e in uncertainty_evidence
            ],
            "consistency_patterns": [
                {
                    "pattern_type": e.pattern_type,
                    "description": e.pattern_description,
                    "confidence": e.confidence,
                    "verification_notes": e.verification_notes,
                    "samples": [
                        {
                            "model": s.model_id,
                            "dilemma": s.dilemma_id,
                            "round": s.round_id,
                            "description": s.pattern_description,
                            "analysis": s.quote,
                            "context": s.context[:800],
                            "question": s.verification_question,
                        }
                        for s in e.samples
                    ],
                }
                for e in consistency_evidence
            ],
        }

    def generate_markdown_report(self) -> str:
        """Generate markdown report for human review."""
        report = self.generate_review_report()
        lines = ["# Ethics Bowl Pattern Review\n"]
        lines.append("## Instructions for Human Reviewer\n")
        lines.append(
            "This report contains patterns extracted by LLM analysis. "
            "Your task is to verify these findings by reviewing the evidence samples.\n"
        )

        lines.append("## Stakeholder Patterns\n")
        for pattern in report["stakeholder_patterns"]:
            lines.append(f"### {pattern['description']}\n")
            lines.append(f"**Confidence:** {pattern['confidence']}\n")
            lines.append(f"**Verification notes:** {pattern['verification_notes']}\n")
            lines.append("\n**Samples:**\n")
            for i, sample in enumerate(pattern["samples"], 1):
                lines.append(f"\n#### Sample {i}: {sample['model']} on {sample['dilemma']}")
                lines.append(f"\n> {sample['quote']}\n")
                lines.append(f"\n**Question:** {sample['question']}\n")
            lines.append("---\n")

        lines.append("## Framework Patterns\n")
        for pattern in report["framework_patterns"]:
            lines.append(f"### {pattern['description']}\n")
            lines.append(f"**Confidence:** {pattern['confidence']}\n")
            lines.append(f"**Verification notes:** {pattern['verification_notes']}\n")
            lines.append("\n**Samples:**\n")
            for i, sample in enumerate(pattern["samples"], 1):
                lines.append(f"\n#### Sample {i}: {sample['model']} on {sample['dilemma']}")
                lines.append(f"\n> {sample['quote']}\n")
                lines.append(f"\n**Question:** {sample['question']}\n")
            lines.append("---\n")

        lines.append("## Uncertainty Patterns\n")
        for pattern in report["uncertainty_patterns"]:
            lines.append(f"### {pattern['description']}\n")
            lines.append(f"**Confidence:** {pattern['confidence']}\n")
            lines.append(f"**Verification notes:** {pattern['verification_notes']}\n")
            lines.append("\n**Samples:**\n")
            for i, sample in enumerate(pattern["samples"], 1):
                lines.append(f"\n#### Sample {i}: {sample['model']} on {sample['dilemma']}")
                lines.append(f"\n> {sample['quote']}\n")
                lines.append(f"\n**Question:** {sample['question']}\n")
            lines.append("---\n")

        lines.append("## Consistency Patterns\n")
        for pattern in report["consistency_patterns"]:
            lines.append(f"### {pattern['description']}\n")
            lines.append(f"**Confidence:** {pattern['confidence']}\n")
            lines.append(f"**Verification notes:** {pattern['verification_notes']}\n")
            lines.append("\n**Samples:**\n")
            for i, sample in enumerate(pattern["samples"], 1):
                lines.append(f"\n#### Sample {i}: {sample['model']} on {sample['dilemma']}")
                lines.append(f"\n**Analysis:** {sample['analysis']}\n")
                lines.append(f"\n**Question:** {sample['question']}\n")
            lines.append("---\n")

        return "\n".join(lines)

    def save_review_report(self, output_dir: str):
        """Save review reports (JSON and Markdown)."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save JSON
        json_report = self.generate_review_report()
        with open(output_path / "review_samples.json", "w") as f:
            json.dump(json_report, f, indent=2)

        # Save Markdown
        md_report = self.generate_markdown_report()
        with open(output_path / "review_samples.md", "w") as f:
            f.write(md_report)
