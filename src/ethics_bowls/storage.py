"""
Storage for Ethics Bowl rounds and tournament state.

Provides persistent storage for rounds, checkpoints, and tournament manifests.
Uses human-readable directory and file naming for easier navigation.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any

import jsonlines

from .schemas import Round, TournamentManifest, RoundStatus, TournamentConfig


class EBStorage:
    """Manages storage of Ethics Bowl data with human-readable naming."""

    # Model name shortcuts for readable filenames
    MODEL_SHORTS = {
        "claude-sonnet-4-5": "claude4.5",
        "claude-sonnet-4-20250514": "claude4",
        "claude-3-5-sonnet": "claude3.5",
        "gpt-5.2": "gpt5.2",
        "gpt-4o": "gpt4o",
        "gpt-4-turbo": "gpt4t",
        "gemini-3-pro-preview": "gemini3",
        "gemini-2.0-flash": "gemini2",
        "grok-4-1-fast-reasoning": "grok4",
        "qwen3-max": "qwen3",
    }

    def __init__(
        self,
        output_dir: str = "data/results/ethics_bowls",
        tournament_config: Optional[TournamentConfig] = None,
    ):
        """
        Initialize storage.

        Args:
            output_dir: Base directory for storing results
            tournament_config: Optional config for creating descriptive directory
        """
        self.base_output_dir = Path(output_dir)
        self.tournament_config = tournament_config
        self._round_counter = 0

        # If we have a config, create a descriptive tournament directory
        if tournament_config:
            self.output_dir = self._get_tournament_dir(tournament_config)
        else:
            self.output_dir = self.base_output_dir

        self._ensure_directories()

    def _shorten_model_name(self, model_name: str) -> str:
        """Shorten model name for directory/file naming."""
        if model_name in self.MODEL_SHORTS:
            return self.MODEL_SHORTS[model_name]

        # Generic shortening
        short = model_name.lower()
        short = re.sub(r'-preview$', '', short)
        short = re.sub(r'-instruct$', '', short)
        short = re.sub(r'-chat$', '', short)
        return short[:12]

    def _get_tournament_dir(self, config: TournamentConfig) -> Path:
        """
        Generate descriptive tournament directory path.

        Format: {run_type}_{YYYY-MM-DD_HH-MM}_{models_summary}/
        Example: pilot_2025-01-19_14-30_claude4.5-gpt5.2/
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M")

        # Create models summary
        model_shorts = [self._shorten_model_name(m) for m in config.models[:3]]
        models_summary = "-".join(model_shorts)
        if len(config.models) > 3:
            models_summary += f"-etc{len(config.models)}"

        run_type = config.run_type.value if hasattr(config.run_type, 'value') else str(config.run_type)
        dir_name = f"{run_type}_{timestamp}_{models_summary}"

        return self.base_output_dir / dir_name

    def _ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "rounds").mkdir(exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)

    def _generate_round_filename(self, round_obj: Round) -> str:
        """
        Generate descriptive filename for a round.

        Format: {counter:03d}_{dilemma}_{teamA}-vs-{teamB}_judge-{judge}.json
        Example: 001_gradient-entity_claude4.5-vs-gpt5.2_judge-grok4.json
        """
        self._round_counter += 1

        # Shorten dilemma ID
        dilemma_short = round_obj.dilemma_id.replace("_", "-")

        # Shorten model names
        team_a = self._shorten_model_name(round_obj.team_a_model)
        team_b = self._shorten_model_name(round_obj.team_b_model)
        judge = self._shorten_model_name(round_obj.judge_model)

        if round_obj.is_self_debate:
            matchup = f"{team_a}-self"
        else:
            matchup = f"{team_a}-vs-{team_b}"

        return f"{self._round_counter:03d}_{dilemma_short}_{matchup}_judge-{judge}.json"

    def _serialize_round(self, round_obj: Round) -> dict:
        """
        Serialize round to JSON-compatible dict.

        Args:
            round_obj: Round to serialize

        Returns:
            Dictionary representation
        """
        data = round_obj.model_dump(mode="json")
        # Ensure datetime fields are strings
        if "timestamp" in data and isinstance(data["timestamp"], datetime):
            data["timestamp"] = data["timestamp"].isoformat()
        for phase in data.get("phases", []):
            if "timestamp" in phase and isinstance(phase["timestamp"], datetime):
                phase["timestamp"] = phase["timestamp"].isoformat()
        return data

    def save_round(self, round_obj: Round) -> Path:
        """
        Save completed round with descriptive filename.

        Saves to both JSONL (for efficient querying) and individual JSON file
        (with human-readable name for easy browsing).

        Args:
            round_obj: Completed round to save

        Returns:
            Path to the saved JSON file
        """
        data = self._serialize_round(round_obj)

        # Append to JSONL file
        rounds_file = self.output_dir / "rounds" / "all_rounds.jsonl"
        with jsonlines.open(rounds_file, mode="a") as writer:
            writer.write(data)

        # Save with descriptive filename
        filename = self._generate_round_filename(round_obj)
        round_file = self.output_dir / "rounds" / filename
        with open(round_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

        # Clean up checkpoint if it exists
        checkpoint_file = self.output_dir / "checkpoints" / f"{round_obj.id}.json"
        if checkpoint_file.exists():
            checkpoint_file.unlink()

        return round_file

    def save_round_checkpoint(self, round_obj: Round) -> Path:
        """
        Save round checkpoint for resume capability.

        Args:
            round_obj: Round in progress

        Returns:
            Path to checkpoint file
        """
        data = self._serialize_round(round_obj)
        checkpoint_file = self.output_dir / "checkpoints" / f"{round_obj.id}.json"
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        return checkpoint_file

    def load_round_checkpoint(self, round_id: str) -> Optional[Round]:
        """
        Load round checkpoint if it exists.

        Args:
            round_id: ID of round to load

        Returns:
            Round object or None if no checkpoint exists
        """
        checkpoint_file = self.output_dir / "checkpoints" / f"{round_id}.json"
        if not checkpoint_file.exists():
            return None
        with open(checkpoint_file, encoding="utf-8") as f:
            data = json.load(f)
        return Round(**data)

    def load_round(self, round_id: str) -> Optional[Round]:
        """
        Load a completed round by ID.

        Args:
            round_id: ID of round to load

        Returns:
            Round object or None if not found
        """
        # Search in JSONL file for the round
        rounds_file = self.output_dir / "rounds" / "all_rounds.jsonl"
        if rounds_file.exists():
            with jsonlines.open(rounds_file) as reader:
                for obj in reader:
                    if obj.get("id") == round_id:
                        return Round(**obj)
        return None

    def save_manifest(self, manifest: TournamentManifest) -> Path:
        """
        Save tournament manifest.

        Args:
            manifest: Tournament manifest to save

        Returns:
            Path to manifest file
        """
        data = manifest.model_dump(mode="json")
        # Ensure datetime fields are strings
        for key in ["created_at", "updated_at"]:
            if key in data and isinstance(data[key], datetime):
                data[key] = data[key].isoformat()

        # Save as manifest.json in tournament directory (simpler naming)
        manifest_file = self.output_dir / "manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        return manifest_file

    def load_manifest(self, tournament_id: str) -> TournamentManifest:
        """
        Load tournament manifest.

        Args:
            tournament_id: ID of tournament to load (not used if manifest.json exists)

        Returns:
            TournamentManifest object

        Raises:
            FileNotFoundError: If manifest doesn't exist
        """
        # Try new location first
        manifest_file = self.output_dir / "manifest.json"
        if not manifest_file.exists():
            # Fall back to old naming convention
            manifest_file = self.output_dir / f"manifest_{tournament_id}.json"
        if not manifest_file.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_file}")
        with open(manifest_file, encoding="utf-8") as f:
            data = json.load(f)
        return TournamentManifest(**data)

    def manifest_exists(self, tournament_id: str) -> bool:
        """
        Check if a manifest exists.

        Args:
            tournament_id: ID of tournament to check

        Returns:
            True if manifest exists
        """
        # Check new location first
        if (self.output_dir / "manifest.json").exists():
            return True
        # Fall back to old naming
        manifest_file = self.output_dir / f"manifest_{tournament_id}.json"
        return manifest_file.exists()

    def load_rounds(
        self,
        tournament_id: Optional[str] = None,
        dilemma_id: Optional[str] = None,
        model: Optional[str] = None,
        status: Optional[RoundStatus] = None,
    ) -> List[Round]:
        """
        Load rounds with optional filtering.

        Args:
            tournament_id: Filter by tournament (not implemented yet)
            dilemma_id: Filter by dilemma
            model: Filter by model (either team or judge)
            status: Filter by round status

        Returns:
            List of matching rounds
        """
        rounds_file = self.output_dir / "rounds" / "all_rounds.jsonl"
        if not rounds_file.exists():
            return []

        rounds: List[Round] = []
        with jsonlines.open(rounds_file) as reader:
            for obj in reader:
                round_obj = Round(**obj)

                # Apply filters
                if dilemma_id and round_obj.dilemma_id != dilemma_id:
                    continue
                if model and model not in (
                    round_obj.team_a_model,
                    round_obj.team_b_model,
                    round_obj.judge_model,
                ):
                    continue
                if status and round_obj.status != status:
                    continue

                rounds.append(round_obj)

        return rounds

    def get_completed_round_ids(self) -> set:
        """
        Get set of all completed round IDs.

        Returns:
            Set of round IDs
        """
        rounds_dir = self.output_dir / "rounds"
        if not rounds_dir.exists():
            return set()

        round_ids = set()
        for json_file in rounds_dir.glob("*.json"):
            if json_file.name != "all_rounds.jsonl":
                # Extract UUID from the file (stored in JSON)
                try:
                    with open(json_file, encoding="utf-8") as f:
                        data = json.load(f)
                        if "id" in data:
                            round_ids.add(data["id"])
                except Exception:
                    # Fall back to using filename stem if JSON parsing fails
                    round_ids.add(json_file.stem)
        return round_ids

    def get_checkpoint_round_ids(self) -> set:
        """
        Get set of all checkpoint round IDs.

        Returns:
            Set of round IDs with checkpoints
        """
        checkpoints_dir = self.output_dir / "checkpoints"
        if not checkpoints_dir.exists():
            return set()

        return {f.stem for f in checkpoints_dir.glob("*.json")}

    def cleanup_checkpoints(self) -> int:
        """
        Remove all checkpoint files.

        Returns:
            Number of checkpoints removed
        """
        checkpoints_dir = self.output_dir / "checkpoints"
        if not checkpoints_dir.exists():
            return 0

        count = 0
        for checkpoint_file in checkpoints_dir.glob("*.json"):
            checkpoint_file.unlink()
            count += 1
        return count

    def write_summary(self, summary_content: str) -> Path:
        """
        Write a markdown summary file for the tournament.

        Args:
            summary_content: Markdown content for the summary

        Returns:
            Path to the summary file
        """
        summary_file = self.output_dir / "README.md"
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(summary_content)
        return summary_file

    def get_tournament_dir_name(self) -> str:
        """Get the tournament directory name (for display)."""
        return self.output_dir.name
