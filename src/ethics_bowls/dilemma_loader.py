"""
Dilemma loader for Ethics Bowl system.

Loads dilemmas from dilemmas_v2.json for use in Ethics Bowl rounds.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from .schemas import EBDilemma, ConsistencyCase


class EBDilemmaLoader:
    """Load and manage Ethics Bowl dilemmas."""

    def __init__(self, dilemmas_source: Optional[Path] = None):
        """
        Initialize loader.

        Args:
            dilemmas_source: Path to dilemmas file (JSON).
                           Defaults to data/dilemmas/dilemmas_v2.json
        """
        if dilemmas_source is None:
            project_root = Path(__file__).parent.parent.parent
            dilemmas_source = project_root / "data" / "dilemmas" / "dilemmas_v2.json"

        self.dilemmas_source = Path(dilemmas_source)
        self.dilemmas: Dict[str, EBDilemma] = {}
        self._load()

    def _load(self) -> None:
        """Load dilemmas from JSON file."""
        if not self.dilemmas_source.exists():
            raise FileNotFoundError(
                f"Dilemmas file not found: {self.dilemmas_source}"
            )

        with open(self.dilemmas_source, encoding="utf-8") as f:
            data = json.load(f)

        for d in data.get("dilemmas", []):
            # Parse consistency case
            cc_data = d.get("consistency_case", {})
            consistency_case = ConsistencyCase(
                description=cc_data.get("description", ""),
                structural_parallel=cc_data.get("structural_parallel"),
            )

            dilemma = EBDilemma(
                id=d["id"],
                title=d["title"],
                category=d.get("category", "general"),
                structure=d.get("structure", "symmetric"),
                description=d["description"],
                core_questions=d.get("core_questions", []),
                tests=d.get("tests"),
                asymmetric_features=d.get("asymmetric_features"),
                consistency_case=consistency_case,
            )
            self.dilemmas[dilemma.id] = dilemma

    def get_dilemma(self, dilemma_id: str) -> EBDilemma:
        """
        Get a specific dilemma by ID.

        Args:
            dilemma_id: The dilemma identifier

        Returns:
            EBDilemma object

        Raises:
            KeyError: If dilemma not found
        """
        if dilemma_id not in self.dilemmas:
            available = ", ".join(self.dilemmas.keys())
            raise KeyError(
                f"Dilemma '{dilemma_id}' not found. "
                f"Available dilemmas: {available}"
            )
        return self.dilemmas[dilemma_id]

    def get_all_dilemma_ids(self) -> List[str]:
        """Get list of all dilemma IDs."""
        return list(self.dilemmas.keys())

    def get_all_dilemmas(self) -> List[EBDilemma]:
        """Get list of all dilemmas."""
        return list(self.dilemmas.values())

    def get_dilemmas_by_category(self, category: str) -> List[EBDilemma]:
        """
        Get dilemmas filtered by category.

        Args:
            category: Category to filter by

        Returns:
            List of matching dilemmas
        """
        return [d for d in self.dilemmas.values() if d.category == category]

    def get_dilemmas_by_structure(self, structure: str) -> List[EBDilemma]:
        """
        Get dilemmas filtered by structure (symmetric/asymmetric).

        Args:
            structure: 'symmetric' or 'asymmetric'

        Returns:
            List of matching dilemmas
        """
        return [d for d in self.dilemmas.values() if d.structure == structure]

    def __len__(self) -> int:
        """Return number of loaded dilemmas."""
        return len(self.dilemmas)

    def __contains__(self, dilemma_id: str) -> bool:
        """Check if dilemma ID exists."""
        return dilemma_id in self.dilemmas
