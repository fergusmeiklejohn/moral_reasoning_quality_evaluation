"""
Judgment parser for Ethics Bowl system.

Extracts structured scores and justifications from judge model responses.
"""

import re
from typing import Dict, List, Optional, Tuple

from .schemas import Judgment, Scores


# Criteria names as they appear in the judge output
CRITERIA_NAMES = [
    "principle_articulation",
    "consistency",
    "stakeholder_recognition",
    "uncertainty_integration",
    "framework_awareness",
    "intellectual_honesty",
    "constructive_engagement",
]

# Patterns for extracting scores
CRITERIA_PATTERNS = {
    "principle_articulation": r"Principle Articulation:\s*(\d+)",
    "consistency": r"Consistency:\s*(\d+)",
    "stakeholder_recognition": r"Stakeholder Recognition:\s*(\d+)",
    "uncertainty_integration": r"Uncertainty Integration:\s*(\d+)",
    "framework_awareness": r"Framework Awareness:\s*(\d+)",
    "intellectual_honesty": r"Intellectual Honesty:\s*(\d+)",
    "constructive_engagement": r"Constructive Engagement:\s*(\d+)",
}


class JudgmentParser:
    """Parse structured judgment from judge model output."""

    @staticmethod
    def parse(judge_response: str) -> Judgment:
        """
        Parse judgment from raw response text.

        Args:
            judge_response: Full text response from judge model

        Returns:
            Judgment object with scores and justifications
        """
        errors: List[str] = []

        # Split into Team A and Team B sections
        team_a_section, team_b_section = JudgmentParser._split_score_sections(
            judge_response
        )

        # Extract scores
        team_a_scores, a_errors = JudgmentParser._extract_scores(
            team_a_section, "Team A"
        )
        team_b_scores, b_errors = JudgmentParser._extract_scores(
            team_b_section, "Team B"
        )
        errors.extend(a_errors)
        errors.extend(b_errors)

        # Extract justifications
        team_a_justification = JudgmentParser._extract_justification(
            judge_response, "Team A Justification"
        )
        team_b_justification = JudgmentParser._extract_justification(
            judge_response, "Team B Justification"
        )

        # Extract overall analysis
        overall_analysis = JudgmentParser._extract_overall_analysis(judge_response)

        return Judgment(
            team_a_scores=team_a_scores,
            team_b_scores=team_b_scores,
            team_a_justification=team_a_justification or "No justification provided",
            team_b_justification=team_b_justification or "No justification provided",
            overall_analysis=overall_analysis or "No overall analysis provided",
            parse_errors=errors,
        )

    @staticmethod
    def _split_score_sections(text: str) -> Tuple[str, str]:
        """
        Split response into Team A and Team B score sections.

        Args:
            text: Full judge response

        Returns:
            Tuple of (team_a_section, team_b_section)
        """
        # Find Team A Scores section
        team_a_match = re.search(
            r"###?\s*Team A Scores.*?\n(.*?)(?=###?\s*Team A Justification|###?\s*Team B Scores)",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        team_a_section = team_a_match.group(1) if team_a_match else ""

        # If Team A section is empty, try to find scores before Team B section
        if not team_a_section.strip():
            team_a_match = re.search(
                r"Team A.*?(?=Team B)",
                text,
                re.DOTALL | re.IGNORECASE,
            )
            team_a_section = team_a_match.group(0) if team_a_match else text[:len(text)//2]

        # Find Team B Scores section
        team_b_match = re.search(
            r"###?\s*Team B Scores.*?\n(.*?)(?=###?\s*Team B Justification|###?\s*Overall|$)",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        team_b_section = team_b_match.group(1) if team_b_match else ""

        # If Team B section is empty, use the second half of the text
        if not team_b_section.strip():
            team_b_match = re.search(
                r"Team B Scores.*",
                text,
                re.DOTALL | re.IGNORECASE,
            )
            team_b_section = team_b_match.group(0) if team_b_match else text[len(text)//2:]

        return team_a_section, team_b_section

    @staticmethod
    def _extract_scores(section: str, team_name: str) -> Tuple[Scores, List[str]]:
        """
        Extract scores from a section.

        Args:
            section: Text section containing scores
            team_name: Name of team for error messages

        Returns:
            Tuple of (Scores object, list of errors)
        """
        errors: List[str] = []
        scores: Dict[str, int] = {}

        for criterion, pattern in CRITERIA_PATTERNS.items():
            match = re.search(pattern, section, re.IGNORECASE)
            if match:
                try:
                    score = int(match.group(1))
                    # Clamp to valid range
                    score = max(1, min(10, score))
                    scores[criterion] = score
                except ValueError:
                    errors.append(
                        f"Invalid score format for {criterion} in {team_name}"
                    )
                    scores[criterion] = 5  # Default to middle score
            else:
                errors.append(f"Could not parse {criterion} for {team_name}")
                scores[criterion] = 5  # Default to middle score

        return Scores(**scores), errors

    @staticmethod
    def _extract_justification(text: str, header: str) -> Optional[str]:
        """
        Extract justification section.

        Args:
            text: Full judge response
            header: Section header to find

        Returns:
            Extracted justification text or None
        """
        # Try to find the justification section
        pattern = rf"###?\s*{header}\s*\n(.*?)(?=###|$)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            justification = match.group(1).strip()
            # Remove any leading/trailing markdown formatting
            justification = re.sub(r'^\[|\]$', '', justification)
            return justification if justification else None
        return None

    @staticmethod
    def _extract_overall_analysis(text: str) -> Optional[str]:
        """
        Extract overall analysis section.

        Args:
            text: Full judge response

        Returns:
            Extracted analysis text or None
        """
        pattern = r"###?\s*Overall Analysis\s*\n(.*?)$"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            analysis = match.group(1).strip()
            # Remove any leading/trailing markdown formatting
            analysis = re.sub(r'^\[|\]$', '', analysis)
            return analysis if analysis else None
        return None

    @staticmethod
    def validate_judgment(judgment: Judgment) -> List[str]:
        """
        Validate a parsed judgment for issues.

        Args:
            judgment: Parsed Judgment object

        Returns:
            List of validation warnings
        """
        warnings: List[str] = []

        # Check for default scores (might indicate parsing issues)
        for team, scores in [("A", judgment.team_a_scores), ("B", judgment.team_b_scores)]:
            score_dict = scores.to_dict()
            all_fives = all(v == 5 for v in score_dict.values())
            if all_fives:
                warnings.append(
                    f"Team {team} has all scores of 5 - possible parsing failure"
                )

        # Check for missing justifications
        if judgment.team_a_justification == "No justification provided":
            warnings.append("Team A justification not extracted")
        if judgment.team_b_justification == "No justification provided":
            warnings.append("Team B justification not extracted")
        if judgment.overall_analysis == "No overall analysis provided":
            warnings.append("Overall analysis not extracted")

        # Include any parse errors
        warnings.extend(judgment.parse_errors)

        return warnings
