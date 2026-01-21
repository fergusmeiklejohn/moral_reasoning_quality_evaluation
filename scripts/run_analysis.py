#!/usr/bin/env python3
"""
Analysis CLI for Ethics Bowl tournament results.

Provides commands for quantitative analysis, pattern extraction, and report generation.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ethics_bowls.analysis import TournamentAnalyzer
from src.ethics_bowls.storage import EBStorage


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def cmd_quantitative(args):
    """Run quantitative analysis only (no API calls)."""
    logger.info(f"Running quantitative analysis on {args.tournament_dir}")

    try:
        analyzer = TournamentAnalyzer(args.tournament_dir)
    except ValueError as e:
        logger.error(f"Error: {e}")
        return 1

    logger.info(f"Loaded {len(analyzer.rounds)} completed rounds")

    # Generate report
    if args.format == "json":
        report = analyzer.generate_report(format="json")
    else:
        report = analyzer.generate_report(format="markdown")

    # Output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)
        logger.info(f"Report saved to {args.output}")
    else:
        print(report)

    return 0


def cmd_extract_patterns(args):
    """Extract patterns using LLM analysis."""
    from src.ethics_bowls.pattern_extractor import PatternExtractor

    logger.info(f"Extracting patterns from {args.tournament_dir}")
    logger.info(f"Using model: {args.model}")

    extractor = PatternExtractor(analyzer_model=args.model)

    def progress(current, total):
        logger.info(f"Processing round {current}/{total}")

    patterns = extractor.extract_from_tournament(
        args.tournament_dir,
        extract_stakeholders=not args.skip_stakeholders,
        extract_frameworks=not args.skip_frameworks,
        extract_uncertainty=not args.skip_uncertainty,
        extract_consistency=not args.skip_consistency,
        progress_callback=progress,
    )

    logger.info(f"Extracted patterns from {len(patterns)} responses")

    # Save extractions - derive output dir from tournament name
    output_dir = _derive_output_dir(args.tournament_dir, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "extracted_patterns.json"

    extractor.save_extractions(patterns, str(output_file))
    logger.info(f"Saved extractions to {output_file}")

    return 0


def cmd_aggregate_patterns(args):
    """Aggregate extracted patterns to find cross-model patterns."""
    from src.ethics_bowls.pattern_aggregator import PatternAggregator

    logger.info(f"Aggregating patterns from {args.patterns_file}")

    aggregator = PatternAggregator.from_json(args.patterns_file)

    logger.info(f"Loaded {len(aggregator.patterns)} patterns")
    logger.info(f"Models: {aggregator.models}")
    logger.info(f"Dilemmas: {aggregator.dilemmas}")

    # Generate summary
    summary = aggregator.generate_summary()

    # Output - patterns_file is inside tournament output dir, so go up one level
    patterns_path = Path(args.patterns_file)
    output_dir = patterns_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "pattern_summary.json"
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary to {output_file}")

    # Print key findings
    print("\n=== KEY FINDINGS ===\n")

    cross_model = summary.get("cross_model_patterns", {})
    if cross_model.get("neglected_stakeholders"):
        print("Neglected Stakeholder Categories:")
        for s in cross_model["neglected_stakeholders"]:
            print(f"  - {s['category']}: {s['coverage_rate']*100:.0f}% coverage")

    if cross_model.get("underused_frameworks"):
        print("\nUnderused Frameworks:")
        for f in cross_model["underused_frameworks"]:
            print(f"  - {f['framework']}: {f['usage_rate']*100:.0f}% usage")

    if cross_model.get("overused_frameworks"):
        print("\nOverused Frameworks (potential monoculture):")
        for f in cross_model["overused_frameworks"]:
            print(f"  - {f['framework']}: {f['usage_rate']*100:.0f}% usage")

    return 0


def cmd_generate_review(args):
    """Generate human review samples."""
    from src.ethics_bowls.pattern_aggregator import PatternAggregator
    from src.ethics_bowls.review_sampler import ReviewSampler

    logger.info(f"Generating review samples from {args.patterns_file}")

    aggregator = PatternAggregator.from_json(args.patterns_file)
    sampler = ReviewSampler(
        aggregator.patterns,
        args.tournament_dir,
        sample_size=args.sample_size,
    )

    # Derive output dir from tournament name
    output_dir = _derive_output_dir(args.tournament_dir, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sampler.save_review_report(str(output_dir))

    logger.info(f"Saved review reports to {output_dir}")
    logger.info(f"  - {output_dir}/review_samples.json")
    logger.info(f"  - {output_dir}/review_samples.md")

    return 0


def _derive_output_dir(tournament_dir: str, base_output_dir: str) -> Path:
    """Derive output directory from tournament directory name."""
    tournament_path = Path(tournament_dir)
    tournament_name = tournament_path.name
    return Path(base_output_dir) / tournament_name


def cmd_full_analysis(args):
    """Run complete analysis pipeline."""
    logger.info("Running full analysis pipeline")
    logger.info(f"Tournament: {args.tournament_dir}")

    # Derive output directory from tournament name
    output_dir = _derive_output_dir(args.tournament_dir, args.output_dir)
    logger.info(f"Output: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Quantitative analysis
    logger.info("\n=== Step 1: Quantitative Analysis ===")
    try:
        analyzer = TournamentAnalyzer(args.tournament_dir)
        quant_report = analyzer.generate_report(format="markdown")
        quant_file = output_dir / "quantitative_analysis.md"
        with open(quant_file, "w") as f:
            f.write(quant_report)
        logger.info(f"Quantitative analysis saved to {quant_file}")

        # Also save JSON
        json_report = analyzer.generate_report(format="json")
        with open(output_dir / "quantitative_analysis.json", "w") as f:
            f.write(json_report)
    except ValueError as e:
        logger.error(f"Quantitative analysis failed: {e}")
        return 1

    # Step 2: Pattern extraction (requires LLM)
    if not args.skip_extraction:
        logger.info("\n=== Step 2: Pattern Extraction (LLM) ===")
        from src.ethics_bowls.pattern_extractor import PatternExtractor

        extractor = PatternExtractor(analyzer_model=args.model)

        def progress(current, total):
            if current % 5 == 0 or current == total:
                logger.info(f"Processing round {current}/{total}")

        patterns = extractor.extract_from_tournament(
            args.tournament_dir,
            progress_callback=progress,
        )
        patterns_file = output_dir / "extracted_patterns.json"
        extractor.save_extractions(patterns, str(patterns_file))
        logger.info(f"Pattern extraction saved to {patterns_file}")

        # Step 3: Pattern aggregation
        logger.info("\n=== Step 3: Pattern Aggregation ===")
        from src.ethics_bowls.pattern_aggregator import PatternAggregator

        aggregator = PatternAggregator(patterns)
        summary = aggregator.generate_summary()
        with open(output_dir / "pattern_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Pattern summary saved to {output_dir}/pattern_summary.json")

        # Step 4: Generate review samples
        logger.info("\n=== Step 4: Review Samples ===")
        from src.ethics_bowls.review_sampler import ReviewSampler

        sampler = ReviewSampler(
            patterns,
            args.tournament_dir,
            sample_size=args.sample_size,
        )
        sampler.save_review_report(str(output_dir))
        logger.info(f"Review samples saved to {output_dir}")

    # Generate final summary report
    logger.info("\n=== Generating Final Report ===")
    _generate_final_report(output_dir, args.tournament_dir)

    logger.info(f"\nAnalysis complete! Results in {output_dir}")
    return 0


def _generate_final_report(output_dir: Path, tournament_dir: str):
    """Generate a combined final report."""
    lines = [f"# Ethics Bowl Analysis Report"]
    lines.append(f"\nGenerated: {datetime.now().isoformat()}")
    lines.append(f"\nTournament: {tournament_dir}\n")

    # Include quantitative highlights
    quant_file = output_dir / "quantitative_analysis.json"
    if quant_file.exists():
        with open(quant_file) as f:
            quant = json.load(f)

        lines.append("## Quantitative Summary\n")
        if "criterion_weakness" in quant:
            lines.append("### Improvement Priorities (weakest criteria first)")
            for p in quant["criterion_weakness"].get("improvement_priorities", [])[:3]:
                lines.append(f"- {p}")
            lines.append("")

    # Include pattern highlights
    pattern_file = output_dir / "pattern_summary.json"
    if pattern_file.exists():
        with open(pattern_file) as f:
            patterns = json.load(f)

        lines.append("## Pattern Summary\n")

        cross_model = patterns.get("cross_model_patterns", {})
        if cross_model.get("neglected_stakeholders"):
            lines.append("### Neglected Stakeholder Categories")
            for s in cross_model["neglected_stakeholders"]:
                lines.append(f"- **{s['category']}**: {s['coverage_rate']*100:.0f}% coverage")
            lines.append("")

        if cross_model.get("underused_frameworks"):
            lines.append("### Underused Ethical Frameworks")
            for f in cross_model["underused_frameworks"]:
                lines.append(f"- **{f['framework']}**: {f['usage_rate']*100:.0f}% usage")
            lines.append("")

        if cross_model.get("overused_frameworks"):
            lines.append("### Potential Framework Monoculture")
            for f in cross_model["overused_frameworks"]:
                lines.append(f"- **{f['framework']}**: {f['usage_rate']*100:.0f}% usage")
            lines.append("")

    # Add verification reminder
    lines.append("## Human Verification Required\n")
    lines.append("Review the samples in `review_samples.md` to verify these patterns.\n")
    lines.append("The patterns above were extracted by LLM analysis and may contain errors.\n")

    final_report = output_dir / "ANALYSIS_REPORT.md"
    with open(final_report, "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Ethics Bowl Tournament Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Quantitative analysis
    quant_parser = subparsers.add_parser(
        "quantitative",
        help="Run quantitative analysis only (no API calls)",
    )
    quant_parser.add_argument(
        "--tournament-dir", "-t",
        required=True,
        help="Path to tournament results directory",
    )
    quant_parser.add_argument(
        "--output", "-o",
        help="Output file path (prints to stdout if not specified)",
    )
    quant_parser.add_argument(
        "--format", "-f",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format (default: markdown)",
    )

    # Pattern extraction
    extract_parser = subparsers.add_parser(
        "extract-patterns",
        help="Extract patterns using LLM analysis",
    )
    extract_parser.add_argument(
        "--tournament-dir", "-t",
        required=True,
        help="Path to tournament results directory",
    )
    extract_parser.add_argument(
        "--output-dir", "-o",
        default="data/analysis",
        help="Base output directory; tournament name appended (default: data/analysis)",
    )
    extract_parser.add_argument(
        "--model", "-m",
        default="claude-opus-4-5",
        help="Model to use for extraction (default: claude-opus-4-5)",
    )
    extract_parser.add_argument(
        "--skip-stakeholders",
        action="store_true",
        help="Skip stakeholder extraction",
    )
    extract_parser.add_argument(
        "--skip-frameworks",
        action="store_true",
        help="Skip framework extraction",
    )
    extract_parser.add_argument(
        "--skip-uncertainty",
        action="store_true",
        help="Skip uncertainty extraction",
    )
    extract_parser.add_argument(
        "--skip-consistency",
        action="store_true",
        help="Skip consistency extraction",
    )

    # Pattern aggregation
    agg_parser = subparsers.add_parser(
        "aggregate",
        help="Aggregate extracted patterns",
    )
    agg_parser.add_argument(
        "--patterns-file", "-p",
        required=True,
        help="Path to extracted patterns JSON file (output saved alongside)",
    )

    # Generate review
    review_parser = subparsers.add_parser(
        "generate-review",
        help="Generate human review samples",
    )
    review_parser.add_argument(
        "--patterns-file", "-p",
        required=True,
        help="Path to extracted patterns JSON file",
    )
    review_parser.add_argument(
        "--tournament-dir", "-t",
        required=True,
        help="Path to tournament results directory",
    )
    review_parser.add_argument(
        "--output-dir", "-o",
        default="data/analysis",
        help="Base output directory; tournament name appended (default: data/analysis)",
    )
    review_parser.add_argument(
        "--sample-size", "-s",
        type=int,
        default=5,
        help="Number of samples per pattern (default: 5)",
    )

    # Full analysis
    full_parser = subparsers.add_parser(
        "analyze",
        help="Run complete analysis pipeline",
    )
    full_parser.add_argument(
        "--tournament-dir", "-t",
        required=True,
        help="Path to tournament results directory",
    )
    full_parser.add_argument(
        "--output-dir", "-o",
        default="data/analysis",
        help="Base output directory; tournament name appended (default: data/analysis)",
    )
    full_parser.add_argument(
        "--model", "-m",
        default="claude-opus-4-5",
        help="Model to use for extraction (default: claude-opus-4-5)",
    )
    full_parser.add_argument(
        "--sample-size", "-s",
        type=int,
        default=5,
        help="Number of review samples per pattern (default: 5)",
    )
    full_parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip LLM pattern extraction (only run quantitative analysis)",
    )

    args = parser.parse_args()

    if args.command == "quantitative":
        return cmd_quantitative(args)
    elif args.command == "extract-patterns":
        return cmd_extract_patterns(args)
    elif args.command == "aggregate":
        return cmd_aggregate_patterns(args)
    elif args.command == "generate-review":
        return cmd_generate_review(args)
    elif args.command == "analyze":
        return cmd_full_analysis(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
