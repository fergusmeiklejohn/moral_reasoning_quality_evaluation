"""
Tournament runner for Ethics Bowl system.

Orchestrates full tournaments across all dilemmas and model pairings.
"""

import uuid
import logging
from datetime import datetime
from itertools import permutations
from typing import List, Optional

from tqdm import tqdm

from ..config.loader import ConfigLoader
from .schemas import (
    TournamentConfig,
    TournamentManifest,
    RoundConfig,
    Round,
    RunType,
)
from .dilemma_loader import EBDilemmaLoader
from .round_runner import RoundRunner
from .storage import EBStorage


logger = logging.getLogger(__name__)


class TournamentRunner:
    """Runs a complete Ethics Bowl tournament."""

    def __init__(
        self,
        config: TournamentConfig,
        config_loader: Optional[ConfigLoader] = None,
        dilemma_loader: Optional[EBDilemmaLoader] = None,
        storage: Optional[EBStorage] = None,
    ):
        """
        Initialize tournament runner.

        Args:
            config: Tournament configuration
            config_loader: Configuration loader
            dilemma_loader: Dilemma loader
            storage: Storage instance
        """
        self.config = config
        self.config_loader = config_loader or ConfigLoader()
        self.dilemma_loader = dilemma_loader or EBDilemmaLoader()
        # Create storage with config for descriptive directory naming
        self.storage = storage or EBStorage(config.output_dir, tournament_config=config)
        self.round_runner = RoundRunner(config, self.config_loader, self.storage)

    def _get_dilemma_ids(self) -> List[str]:
        """Get list of dilemma IDs to use."""
        if self.config.dilemma_ids:
            return self.config.dilemma_ids
        return self.dilemma_loader.get_all_dilemma_ids()

    def _select_judge(
        self,
        team_a: str,
        team_b: str,
        models: List[str],
        round_idx: int,
    ) -> str:
        """
        Select judge for a round.

        Args:
            team_a: Team A model
            team_b: Team B model
            models: All available models
            round_idx: Index for rotation

        Returns:
            Selected judge model
        """
        if self.config.judge_selection == "fixed" and self.config.fixed_judge:
            return self.config.fixed_judge

        # Rotate judge from models not participating
        remaining = [m for m in models if m not in (team_a, team_b)]
        if remaining:
            # Rotate through remaining models
            return remaining[round_idx % len(remaining)]
        else:
            # Only 2 models total - responder also judges
            return team_b

    def generate_round_configs(self) -> List[RoundConfig]:
        """
        Generate all round configurations for the tournament.

        Returns:
            List of RoundConfig objects
        """
        rounds: List[RoundConfig] = []
        dilemma_ids = self._get_dilemma_ids()
        models = self.config.models
        round_idx = 0

        for dilemma_id in dilemma_ids:
            # All ordered pairs of distinct models
            for team_a, team_b in permutations(models, 2):
                judge = self._select_judge(team_a, team_b, models, round_idx)

                for run_num in range(self.config.rounds_per_pairing):
                    rounds.append(
                        RoundConfig(
                            round_id=str(uuid.uuid4()),
                            dilemma_id=dilemma_id,
                            team_a_model=team_a,
                            team_b_model=team_b,
                            judge_model=judge,
                            run_number=run_num + 1,
                            status="pending",
                            is_self_debate=False,
                        )
                    )
                    round_idx += 1

            # Self-debates if enabled
            if self.config.include_self_debates:
                for model in models:
                    # Need a different judge
                    judges = [m for m in models if m != model]
                    judge = judges[round_idx % len(judges)] if judges else model

                    for run_num in range(self.config.rounds_per_pairing):
                        rounds.append(
                            RoundConfig(
                                round_id=str(uuid.uuid4()),
                                dilemma_id=dilemma_id,
                                team_a_model=model,
                                team_b_model=model,
                                judge_model=judge,
                                run_number=run_num + 1,
                                status="pending",
                                is_self_debate=True,
                            )
                        )
                        round_idx += 1

        return rounds

    def generate_lite_round_configs(self) -> List[RoundConfig]:
        """
        Generate minimal round configs for lite tournament mode.

        Uses a rotation scheme to ensure each model plays each role
        (Team A, Team B, Judge) at least twice across all dilemmas.
        Produces 2 rounds per dilemma.

        Returns:
            List of RoundConfig objects (2 per dilemma)
        """
        rounds: List[RoundConfig] = []
        dilemma_ids = self._get_dilemma_ids()
        models = self.config.models
        n = len(models)

        if n < 3:
            logger.warning(
                f"Lite mode requires at least 3 models for distinct roles, got {n}. "
                "Falling back to standard generation."
            )
            return self.generate_round_configs()

        # Generate 2 rounds per dilemma using rotation
        for d_idx, dilemma_id in enumerate(dilemma_ids):
            for r in range(2):
                # Rotate through models to ensure balanced role distribution
                offset = (d_idx * 2 + r) % n
                team_a = models[offset]
                team_b = models[(offset + 1) % n]
                judge = models[(offset + 2) % n]

                rounds.append(
                    RoundConfig(
                        round_id=str(uuid.uuid4()),
                        dilemma_id=dilemma_id,
                        team_a_model=team_a,
                        team_b_model=team_b,
                        judge_model=judge,
                        run_number=1,
                        status="pending",
                        is_self_debate=False,
                    )
                )

        return rounds

    def create_manifest(self) -> TournamentManifest:
        """
        Create tournament manifest.

        Returns:
            TournamentManifest object
        """
        # Dispatch to appropriate round generator based on run type
        if self.config.run_type == RunType.LITE:
            rounds = self.generate_lite_round_configs()
        else:
            rounds = self.generate_round_configs()
        run_type_value = self.config.run_type.value if hasattr(self.config.run_type, 'value') else str(self.config.run_type)
        manifest = TournamentManifest(
            tournament_id=self.config.tournament_id,
            models=self.config.models,
            dilemma_ids=self._get_dilemma_ids(),
            include_self_debates=self.config.include_self_debates,
            rounds_per_pairing=self.config.rounds_per_pairing,
            rounds=rounds,
            run_type=run_type_value,
            description=self.config.description,
        )
        self.storage.save_manifest(manifest)
        return manifest

    def run_tournament(self, resume: bool = False) -> str:
        """
        Run the full tournament.

        Args:
            resume: Whether to resume from existing manifest

        Returns:
            Tournament ID
        """
        if resume:
            manifest = self.storage.load_manifest(self.config.tournament_id)
            logger.info(f"Resuming tournament {self.config.tournament_id}")
        else:
            manifest = self.create_manifest()
            logger.info(f"Created tournament {self.config.tournament_id}")

        pending = manifest.get_pending_rounds()
        in_progress = manifest.get_in_progress_rounds()
        total = len(manifest.rounds)
        completed = len(manifest.get_complete_rounds())

        # Treat in_progress rounds as pending (need to resume)
        rounds_to_run = pending + in_progress

        print(f"\n{'='*80}")
        print(f"Ethics Bowl Tournament: {self.config.tournament_id}")
        print(f"{'='*80}")
        print(f"Models: {', '.join(self.config.models)}")
        print(f"Dilemmas: {len(manifest.dilemma_ids)}")
        print(f"Total rounds: {total}")
        print(f"Completed: {completed}")
        print(f"Remaining: {len(rounds_to_run)}")
        print(f"{'='*80}\n")

        with tqdm(total=len(rounds_to_run), desc="Tournament Progress") as pbar:
            for round_config in rounds_to_run:
                try:
                    # Update status to in_progress
                    round_config.status = "in_progress"
                    manifest.updated_at = datetime.utcnow()
                    self.storage.save_manifest(manifest)

                    # Load dilemma
                    dilemma = self.dilemma_loader.get_dilemma(round_config.dilemma_id)

                    # Check for existing checkpoint
                    checkpoint = self.storage.load_round_checkpoint(round_config.round_id)

                    # Run the round
                    result = self.round_runner.run_round(
                        dilemma=dilemma,
                        team_a_model=round_config.team_a_model,
                        team_b_model=round_config.team_b_model,
                        judge_model=round_config.judge_model,
                        round_id=round_config.round_id,
                        resume_from=checkpoint,
                    )

                    # Update manifest
                    round_config.status = "complete"
                    manifest.updated_at = datetime.utcnow()
                    self.storage.save_manifest(manifest)

                except Exception as e:
                    logger.error(f"Error in round {round_config.round_id}: {e}")
                    round_config.status = "failed"
                    round_config.error = str(e)
                    manifest.updated_at = datetime.utcnow()
                    self.storage.save_manifest(manifest)

                pbar.update(1)
                pbar.set_postfix(
                    completed=len(manifest.get_complete_rounds()),
                    failed=len(manifest.get_failed_rounds()),
                )

        # Print summary
        completed = len(manifest.get_complete_rounds())
        failed = len(manifest.get_failed_rounds())

        print(f"\n{'='*80}")
        print(f"Tournament Complete: {self.config.tournament_id}")
        print(f"{'='*80}")
        print(f"Completed rounds: {completed}")
        print(f"Failed rounds: {failed}")
        if failed > 0:
            print("Failed round IDs:")
            for r in manifest.get_failed_rounds():
                print(f"  - {r.round_id}: {r.error}")
        print(f"Results: {self.storage.output_dir}")
        print(f"{'='*80}\n")

        # Generate markdown summary
        self._generate_summary(manifest)

        return self.config.tournament_id

    def _generate_summary(self, manifest: TournamentManifest) -> None:
        """
        Generate markdown summary file for the tournament.

        Args:
            manifest: Tournament manifest with results
        """
        completed = manifest.get_complete_rounds()
        failed = manifest.get_failed_rounds()
        run_type = manifest.run_type.upper()

        # Build summary content
        lines = [
            f"# Ethics Bowl {run_type}: {self.storage.get_tournament_dir_name()}",
            "",
            f"**Run Type**: {run_type}",
            f"**Created**: {manifest.created_at.strftime('%Y-%m-%d %H:%M UTC') if manifest.created_at else 'Unknown'}",
            f"**Models**: {', '.join(manifest.models)}",
            f"**Dilemmas**: {len(manifest.dilemma_ids)}",
            "",
        ]

        if manifest.description:
            lines.extend([f"**Description**: {manifest.description}", ""])

        lines.extend([
            "## Summary",
            "",
            f"- Total rounds: {len(manifest.rounds)}",
            f"- Completed: {len(completed)}",
            f"- Failed: {len(failed)}",
            f"- Self-debates: {'Yes' if manifest.include_self_debates else 'No'}",
            "",
        ])

        # Add results table if we have completed rounds
        if completed:
            lines.extend([
                "## Results Overview",
                "",
                "| # | Dilemma | Team A | Team B | Judge | A Score | B Score | Winner |",
                "|---|---------|--------|--------|-------|---------|---------|--------|",
            ])

            for idx, r in enumerate(completed, 1):
                round_obj = self.storage.load_round(r.round_id)
                if round_obj and round_obj.judgment:
                    j = round_obj.judgment
                    a_score = j.team_a_scores.total
                    b_score = j.team_b_scores.total
                    winner = "Team A" if a_score > b_score else ("Team B" if b_score > a_score else "Tie")
                    lines.append(
                        f"| {idx} | {r.dilemma_id} | {r.team_a_model} | {r.team_b_model} | "
                        f"{r.judge_model} | {a_score}/70 | {b_score}/70 | {winner} |"
                    )

            lines.append("")

        # Add dilemma list
        lines.extend([
            "## Dilemmas",
            "",
        ])
        for d_id in manifest.dilemma_ids:
            try:
                dilemma = self.dilemma_loader.get_dilemma(d_id)
                lines.append(f"- **{d_id}**: {dilemma.title}")
            except Exception:
                lines.append(f"- **{d_id}**")

        lines.extend(["", "---", f"*Generated by Ethics Bowl system*"])

        summary_content = "\n".join(lines)
        self.storage.write_summary(summary_content)
        logger.info(f"Summary written to {self.storage.output_dir / 'README.md'}")

    def get_tournament_summary(self) -> dict:
        """
        Get summary of tournament progress.

        Returns:
            Dictionary with summary statistics
        """
        if not self.storage.manifest_exists(self.config.tournament_id):
            return {
                "tournament_id": self.config.tournament_id,
                "status": "not_started",
            }

        manifest = self.storage.load_manifest(self.config.tournament_id)
        return {
            "tournament_id": self.config.tournament_id,
            "models": manifest.models,
            "dilemmas": manifest.dilemma_ids,
            "total_rounds": len(manifest.rounds),
            "completed": len(manifest.get_complete_rounds()),
            "pending": len(manifest.get_pending_rounds()),
            "in_progress": len(manifest.get_in_progress_rounds()),
            "failed": len(manifest.get_failed_rounds()),
            "include_self_debates": manifest.include_self_debates,
            "rounds_per_pairing": manifest.rounds_per_pairing,
            "created_at": manifest.created_at.isoformat() if manifest.created_at else None,
            "updated_at": manifest.updated_at.isoformat() if manifest.updated_at else None,
        }
