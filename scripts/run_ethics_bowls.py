#!/usr/bin/env python3
"""
CLI for Ethics Bowl system.

Commands for running rounds, tournaments, and generating analysis.
"""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ethics_bowls.schemas import TournamentConfig, TournamentManifest, RunType
from src.ethics_bowls.dilemma_loader import EBDilemmaLoader
from src.ethics_bowls.round_runner import RoundRunner
from src.ethics_bowls.tournament_runner import TournamentRunner
from src.ethics_bowls.storage import EBStorage


app = typer.Typer(
    name="ethics-bowls",
    help="Ethics Bowl CLI for language model moral reasoning evaluation",
)


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


@app.command()
def run_round(
    team_a: str = typer.Option(..., "--team-a", "-a", help="Model for presenting team"),
    team_b: str = typer.Option(..., "--team-b", "-b", help="Model for responding team"),
    judge: str = typer.Option(..., "--judge", "-j", help="Model for judging"),
    dilemma: str = typer.Option(..., "--dilemma", "-d", help="Dilemma ID"),
    output_dir: str = typer.Option(
        "data/results/ethics_bowls", "--output-dir", "-o", help="Output directory"
    ),
    temperature: float = typer.Option(0.3, "--temperature", "-t", help="Sampling temperature"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Run a single Ethics Bowl round."""
    setup_logging(verbose)

    # Deduplicate models list (judge may be same as team_a or team_b)
    unique_models = list(dict.fromkeys([team_a, team_b, judge]))

    config = TournamentConfig(
        tournament_id=f"single_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        models=unique_models,
        dilemma_ids=[dilemma],
        output_dir=output_dir,
        temperature=temperature,
        run_type=RunType.SINGLE_ROUND,
        description=f"Single round: {dilemma}",
    )

    # Create storage with descriptive naming
    storage = EBStorage(output_dir, tournament_config=config)
    loader = EBDilemmaLoader()

    try:
        dilemma_obj = loader.get_dilemma(dilemma)
    except KeyError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    runner = RoundRunner(config, storage=storage)

    typer.echo(f"\nRunning Ethics Bowl round:")
    typer.echo(f"  Dilemma: {dilemma_obj.title}")
    typer.echo(f"  Team A (Presenting): {team_a}")
    typer.echo(f"  Team B (Responding): {team_b}")
    typer.echo(f"  Judge: {judge}")
    typer.echo(f"  Output: {storage.output_dir}")
    typer.echo("")

    try:
        result = runner.run_round(dilemma_obj, team_a, team_b, judge)
    except Exception as e:
        typer.echo(f"Error running round: {e}", err=True)
        raise typer.Exit(1)

    typer.echo(f"\nRound complete: {result.id}")
    typer.echo(f"Team A ({team_a}) scores: {result.judgment.team_a_scores.total}/70")
    typer.echo(f"Team B ({team_b}) scores: {result.judgment.team_b_scores.total}/70")
    typer.echo(f"Results saved to: {storage.output_dir}")


@app.command()
def run_tournament(
    models: str = typer.Option(
        ..., "--models", "-m", help="Comma-separated list of model IDs"
    ),
    dilemmas: str = typer.Option(
        "all", "--dilemmas", "-d", help="Comma-separated dilemma IDs or 'all'"
    ),
    output_dir: str = typer.Option(
        "data/results/ethics_bowls", "--output-dir", "-o", help="Output directory"
    ),
    run_type: str = typer.Option(
        "tournament", "--type", help="Run type: pilot, tournament, cross, custom, lite"
    ),
    description: Optional[str] = typer.Option(
        None, "--description", help="Human-readable description of the run"
    ),
    include_self_debates: bool = typer.Option(
        False, "--include-self-debates", help="Include self-debate rounds"
    ),
    rounds_per_pairing: int = typer.Option(
        1, "--rounds-per-pairing", "-r", help="Number of rounds per model pairing"
    ),
    temperature: float = typer.Option(0.3, "--temperature", "-t", help="Sampling temperature"),
    rate_limit: int = typer.Option(
        30, "--rate-limit", help="Requests per minute"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Run a full Ethics Bowl tournament."""
    setup_logging(verbose)

    model_list = [m.strip() for m in models.split(",")]
    dilemma_list = None if dilemmas == "all" else [d.strip() for d in dilemmas.split(",")]

    if len(model_list) < 2:
        typer.echo("Error: At least 2 models required for a tournament", err=True)
        raise typer.Exit(1)

    # Map run_type string to enum
    run_type_map = {
        "pilot": RunType.PILOT,
        "tournament": RunType.TOURNAMENT,
        "cross": RunType.CROSS_PROVIDER,
        "custom": RunType.CUSTOM,
        "lite": RunType.LITE,
    }
    run_type_enum = run_type_map.get(run_type.lower(), RunType.TOURNAMENT)

    tournament_id = f"{run_type}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    config = TournamentConfig(
        tournament_id=tournament_id,
        models=model_list,
        dilemma_ids=dilemma_list,
        include_self_debates=include_self_debates,
        rounds_per_pairing=rounds_per_pairing,
        output_dir=output_dir,
        temperature=temperature,
        rate_limit_per_minute=rate_limit,
        run_type=run_type_enum,
        description=description,
    )

    runner = TournamentRunner(config)
    runner.run_tournament()

    typer.echo(f"\nTournament complete: {runner.storage.output_dir}")


@app.command()
def resume_tournament(
    tournament_dir: str = typer.Option(
        ..., "--tournament-dir", "-d", help="Path to tournament directory"
    ),
    tournament_id: str = typer.Option(
        ..., "--tournament-id", "-i", help="Tournament ID to resume"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Resume an interrupted tournament."""
    setup_logging(verbose)

    storage = EBStorage(tournament_dir)

    try:
        manifest = storage.load_manifest(tournament_id)
    except FileNotFoundError:
        typer.echo(f"Error: Tournament manifest not found for '{tournament_id}'", err=True)
        raise typer.Exit(1)

    # Reconstruct config from manifest
    config = TournamentConfig(
        tournament_id=tournament_id,
        models=manifest.models,
        dilemma_ids=manifest.dilemma_ids,
        include_self_debates=manifest.include_self_debates,
        rounds_per_pairing=manifest.rounds_per_pairing,
        output_dir=tournament_dir,
    )

    runner = TournamentRunner(config)
    runner.run_tournament(resume=True)


@app.command()
def analyze(
    tournament_dir: str = typer.Option(
        ..., "--tournament-dir", "-d", help="Path to tournament directory"
    ),
    output_format: str = typer.Option(
        "markdown", "--output-format", "-f", help="Output format: json or markdown"
    ),
    output_file: Optional[str] = typer.Option(
        None, "--output-file", "-o", help="Output file (stdout if not specified)"
    ),
) -> None:
    """Generate analysis from completed rounds."""
    from src.ethics_bowls.analysis import TournamentAnalyzer

    try:
        analyzer = TournamentAnalyzer(tournament_dir)
    except Exception as e:
        typer.echo(f"Error loading tournament: {e}", err=True)
        raise typer.Exit(1)

    report = analyzer.generate_report(format=output_format)

    if output_file:
        Path(output_file).write_text(report)
        typer.echo(f"Analysis saved to: {output_file}")
    else:
        typer.echo(report)


@app.command()
def list_models() -> None:
    """Show available model configurations."""
    from src.config.loader import ConfigLoader

    try:
        loader = ConfigLoader()
        config = loader.load_models_config()
    except Exception as e:
        typer.echo(f"Error loading config: {e}", err=True)
        raise typer.Exit(1)

    typer.echo("\nAvailable Models:")
    for provider, provider_config in config.items():
        if "models" in provider_config and provider_config["models"]:
            for model_name in provider_config["models"]:
                typer.echo(f"  - {model_name} ({provider})")

    typer.echo("\nNote: You can also use model names directly if they can be inferred:")
    typer.echo("  - gpt-* -> openai")
    typer.echo("  - claude-* -> anthropic")
    typer.echo("  - gemini-* -> google")
    typer.echo("  - grok-* -> grok")
    typer.echo("  - qwen-* -> qwen")


@app.command()
def list_dilemmas() -> None:
    """Show available dilemmas."""
    try:
        loader = EBDilemmaLoader()
    except Exception as e:
        typer.echo(f"Error loading dilemmas: {e}", err=True)
        raise typer.Exit(1)

    typer.echo("\nAvailable Dilemmas:")
    for dilemma in loader.get_all_dilemmas():
        structure = dilemma.structure.capitalize()
        typer.echo(f"  - {dilemma.id}: {dilemma.title}")
        typer.echo(f"    Category: {dilemma.category.replace('_', ' ').title()}")
        typer.echo(f"    Structure: {structure}")
        typer.echo("")


@app.command()
def status(
    tournament_dir: str = typer.Option(
        "data/results/ethics_bowls", "--tournament-dir", "-d", help="Tournament directory"
    ),
    tournament_id: str = typer.Option(
        None, "--tournament-id", "-i", help="Specific tournament ID"
    ),
) -> None:
    """Show tournament status."""
    tournament_dir_path = Path(tournament_dir)

    if tournament_id:
        storage = EBStorage(tournament_dir)
        try:
            manifest = storage.load_manifest(tournament_id)
        except FileNotFoundError:
            typer.echo(f"Tournament '{tournament_id}' not found", err=True)
            raise typer.Exit(1)

        typer.echo(f"\nTournament: {tournament_id}")
        typer.echo(f"  Models: {', '.join(manifest.models)}")
        typer.echo(f"  Dilemmas: {len(manifest.dilemma_ids)}")
        typer.echo(f"  Total rounds: {len(manifest.rounds)}")
        typer.echo(f"  Completed: {len(manifest.get_complete_rounds())}")
        typer.echo(f"  Pending: {len(manifest.get_pending_rounds())}")
        typer.echo(f"  Failed: {len(manifest.get_failed_rounds())}")
    else:
        # Find all tournaments (both old and new naming schemes)
        # New scheme: subdirectories with manifest.json
        new_manifests = list(tournament_dir_path.glob("*/manifest.json"))
        # Old scheme: manifest_*.json in base dir
        old_manifests = list(tournament_dir_path.glob("manifest_*.json"))

        if not new_manifests and not old_manifests:
            typer.echo("No tournaments found")
            return

        typer.echo("\nTournaments:")

        # List new-style tournaments (descriptive subdirectories)
        for manifest_file in sorted(new_manifests, reverse=True):
            try:
                with open(manifest_file, encoding="utf-8") as f:
                    import json
                    data = json.load(f)
                manifest = TournamentManifest(**data)
                completed = len(manifest.get_complete_rounds())
                total = len(manifest.rounds)
                run_type = getattr(manifest, 'run_type', 'unknown').upper()
                dir_name = manifest_file.parent.name
                typer.echo(f"  {dir_name}/")
                typer.echo(f"    Type: {run_type} | {completed}/{total} rounds | Models: {', '.join(manifest.models[:2])}{'...' if len(manifest.models) > 2 else ''}")
            except Exception as e:
                typer.echo(f"  {manifest_file.parent.name}/: (error: {e})")

        # List old-style tournaments
        for manifest_file in old_manifests:
            tid = manifest_file.stem.replace("manifest_", "")
            try:
                storage = EBStorage(tournament_dir)
                manifest = storage.load_manifest(tid)
                completed = len(manifest.get_complete_rounds())
                total = len(manifest.rounds)
                typer.echo(f"  [old] {tid}: {completed}/{total} rounds complete")
            except Exception:
                typer.echo(f"  [old] {tid}: (error loading)")


if __name__ == "__main__":
    app()
