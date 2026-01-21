"""
Analysis and reporting for Ethics Bowl tournaments.

Generates comparison matrices, model rankings, and dilemma difficulty analysis.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from .schemas import Round, RoundStatus
from .storage import EBStorage


# Criteria names for display
CRITERIA_DISPLAY = {
    "principle_articulation": "Principle Articulation",
    "consistency": "Consistency",
    "stakeholder_recognition": "Stakeholder Recognition",
    "uncertainty_integration": "Uncertainty Integration",
    "framework_awareness": "Framework Awareness",
    "intellectual_honesty": "Intellectual Honesty",
    "constructive_engagement": "Constructive Engagement",
}


class TournamentAnalyzer:
    """Generate analysis from completed Ethics Bowl rounds."""

    def __init__(self, tournament_dir: str):
        """
        Initialize analyzer.

        Args:
            tournament_dir: Path to tournament results directory
        """
        self.storage = EBStorage(tournament_dir)
        self.rounds = self.storage.load_rounds(status=RoundStatus.COMPLETE)

        if not self.rounds:
            raise ValueError(f"No completed rounds found in {tournament_dir}")

    def model_comparison_table(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate average scores by model across all criteria.

        Returns:
            Dictionary mapping model names to their average scores
        """
        scores_by_model: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for round_obj in self.rounds:
            if round_obj.judgment:
                # Team A scores (presenting)
                a_scores = round_obj.judgment.team_a_scores.to_dict()
                for criterion, score in a_scores.items():
                    scores_by_model[round_obj.team_a_model][criterion].append(score)
                    scores_by_model[round_obj.team_a_model]["all"].append(score)

                # Team B scores (responding)
                b_scores = round_obj.judgment.team_b_scores.to_dict()
                for criterion, score in b_scores.items():
                    scores_by_model[round_obj.team_b_model][criterion].append(score)
                    scores_by_model[round_obj.team_b_model]["all"].append(score)

        # Compute averages
        result = {}
        for model, criteria in scores_by_model.items():
            result[model] = {
                criterion: sum(scores) / len(scores) if scores else 0
                for criterion, scores in criteria.items()
            }
            # Add total/average calculations
            all_scores = criteria.get("all", [])
            result[model]["average"] = sum(all_scores) / len(all_scores) if all_scores else 0
            result[model]["rounds"] = len(all_scores) // 7  # Approximate rounds

        return result

    def model_role_comparison(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Compare model performance by role (presenting vs responding).

        Returns:
            Dictionary with scores broken down by role
        """
        scores_by_role: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: {"presenting": [], "responding": []}
        )

        for round_obj in self.rounds:
            if round_obj.judgment:
                # Team A is presenting
                a_total = round_obj.judgment.team_a_scores.total
                scores_by_role[round_obj.team_a_model]["presenting"].append(a_total)

                # Team B is responding
                b_total = round_obj.judgment.team_b_scores.total
                scores_by_role[round_obj.team_b_model]["responding"].append(b_total)

        result = {}
        for model, roles in scores_by_role.items():
            result[model] = {
                "presenting_avg": sum(roles["presenting"]) / len(roles["presenting"]) if roles["presenting"] else 0,
                "presenting_count": len(roles["presenting"]),
                "responding_avg": sum(roles["responding"]) / len(roles["responding"]) if roles["responding"] else 0,
                "responding_count": len(roles["responding"]),
            }
        return result

    def head_to_head_matrix(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Generate head-to-head comparison between model pairs.

        Returns:
            Dictionary with win/loss/margin data for each pairing
        """
        h2h: Dict[str, Dict[str, Dict[str, any]]] = defaultdict(
            lambda: defaultdict(lambda: {"wins": 0, "losses": 0, "ties": 0, "margins": []})
        )

        for round_obj in self.rounds:
            if round_obj.judgment:
                a_total = round_obj.judgment.team_a_scores.total
                b_total = round_obj.judgment.team_b_scores.total
                margin = a_total - b_total

                h2h[round_obj.team_a_model][round_obj.team_b_model]["margins"].append(margin)

                if a_total > b_total:
                    h2h[round_obj.team_a_model][round_obj.team_b_model]["wins"] += 1
                elif b_total > a_total:
                    h2h[round_obj.team_a_model][round_obj.team_b_model]["losses"] += 1
                else:
                    h2h[round_obj.team_a_model][round_obj.team_b_model]["ties"] += 1

        # Compute average margins
        for model_a in h2h:
            for model_b in h2h[model_a]:
                margins = h2h[model_a][model_b]["margins"]
                h2h[model_a][model_b]["avg_margin"] = sum(margins) / len(margins) if margins else 0
                del h2h[model_a][model_b]["margins"]  # Remove raw data

        return dict(h2h)

    def dilemma_difficulty(self) -> Dict[str, Dict[str, float]]:
        """
        Analyze difficulty of each dilemma based on average scores.

        Returns:
            Dictionary with difficulty metrics for each dilemma
        """
        scores_by_dilemma: Dict[str, List[float]] = defaultdict(list)

        for round_obj in self.rounds:
            if round_obj.judgment:
                avg_score = (
                    round_obj.judgment.team_a_scores.average +
                    round_obj.judgment.team_b_scores.average
                ) / 2
                scores_by_dilemma[round_obj.dilemma_id].append(avg_score)

        result = {}
        for dilemma_id, scores in scores_by_dilemma.items():
            result[dilemma_id] = {
                "avg_score": sum(scores) / len(scores) if scores else 0,
                "std_dev": self._std_dev(scores),
                "min_score": min(scores) if scores else 0,
                "max_score": max(scores) if scores else 0,
                "n_rounds": len(scores),
            }

        # Sort by average score (lowest = hardest)
        result = dict(sorted(result.items(), key=lambda x: x[1]["avg_score"]))
        return result

    def judge_analysis(self) -> Dict[str, Dict[str, float]]:
        """
        Analyze judging patterns by model.

        Returns:
            Dictionary with judging statistics for each model
        """
        judge_stats: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: {"scores_given": [], "team_a_bias": []}
        )

        for round_obj in self.rounds:
            if round_obj.judgment:
                judge = round_obj.judge_model
                a_total = round_obj.judgment.team_a_scores.total
                b_total = round_obj.judgment.team_b_scores.total

                judge_stats[judge]["scores_given"].extend([a_total, b_total])
                judge_stats[judge]["team_a_bias"].append(a_total - b_total)

        result = {}
        for judge, stats in judge_stats.items():
            scores = stats["scores_given"]
            biases = stats["team_a_bias"]
            result[judge] = {
                "avg_score_given": sum(scores) / len(scores) if scores else 0,
                "avg_team_a_bias": sum(biases) / len(biases) if biases else 0,
                "rounds_judged": len(biases),
                "score_std_dev": self._std_dev(scores),
            }

        return result

    def self_debate_analysis(self) -> Dict[str, Dict[str, float]]:
        """
        Analyze self-debate rounds where the same model plays both teams.

        Returns:
            Dictionary with self-debate statistics
        """
        self_debates: Dict[str, List[Dict]] = defaultdict(list)

        for round_obj in self.rounds:
            if round_obj.is_self_debate and round_obj.judgment:
                model = round_obj.team_a_model
                a_total = round_obj.judgment.team_a_scores.total
                b_total = round_obj.judgment.team_b_scores.total

                self_debates[model].append({
                    "score_diff": abs(a_total - b_total),
                    "a_total": a_total,
                    "b_total": b_total,
                })

        result = {}
        for model, debates in self_debates.items():
            if debates:
                diffs = [d["score_diff"] for d in debates]
                a_totals = [d["a_total"] for d in debates]
                b_totals = [d["b_total"] for d in debates]

                result[model] = {
                    "avg_score_diff": sum(diffs) / len(diffs),
                    "avg_presenting_score": sum(a_totals) / len(a_totals),
                    "avg_responding_score": sum(b_totals) / len(b_totals),
                    "n_self_debates": len(debates),
                }

        return result

    def generate_report(self, format: str = "markdown") -> str:
        """
        Generate full analysis report.

        Args:
            format: Output format ('markdown' or 'json')

        Returns:
            Formatted report string
        """
        model_table = self.model_comparison_table()
        role_comparison = self.model_role_comparison()
        h2h = self.head_to_head_matrix()
        difficulty = self.dilemma_difficulty()
        judge_stats = self.judge_analysis()
        self_debate = self.self_debate_analysis()
        weakness = self.criterion_weakness_analysis()
        variance = self.variance_analysis()
        patterns = self.cross_model_patterns()

        if format == "json":
            return json.dumps({
                "model_comparison": model_table,
                "role_comparison": role_comparison,
                "head_to_head": h2h,
                "dilemma_difficulty": difficulty,
                "judge_analysis": judge_stats,
                "self_debate_analysis": self_debate,
                "criterion_weakness": weakness,
                "variance_analysis": variance,
                "cross_model_patterns": patterns,
            }, indent=2)
        else:
            return self._format_markdown(
                model_table, role_comparison, h2h, difficulty, judge_stats, self_debate,
                weakness, variance, patterns
            )

    def _format_markdown(
        self,
        model_table: Dict,
        role_comparison: Dict,
        h2h: Dict,
        difficulty: Dict,
        judge_stats: Dict,
        self_debate: Dict,
        weakness: Optional[Dict] = None,
        variance: Optional[Dict] = None,
        patterns: Optional[Dict] = None,
    ) -> str:
        """Format report as markdown."""
        lines = ["# Ethics Bowl Tournament Analysis\n"]

        # Summary
        lines.append("## Summary\n")
        lines.append(f"- Total completed rounds: {len(self.rounds)}")
        lines.append(f"- Models evaluated: {len(model_table)}")
        lines.append(f"- Dilemmas used: {len(difficulty)}")
        lines.append("")

        # Model Comparison
        lines.append("## Model Comparison (Average Scores)\n")
        lines.append("| Model | Avg Score | Rounds |")
        lines.append("|-------|-----------|--------|")
        for model in sorted(model_table.keys(), key=lambda m: -model_table[m].get("average", 0)):
            scores = model_table[model]
            lines.append(f"| {model} | {scores.get('average', 0):.2f} | {scores.get('rounds', 0)} |")
        lines.append("")

        # Detailed Scores by Criterion
        lines.append("## Scores by Criterion\n")
        lines.append("| Model | Principle | Consistency | Stakeholder | Uncertainty | Framework | Honesty | Engagement |")
        lines.append("|-------|-----------|-------------|-------------|-------------|-----------|---------|------------|")
        for model in sorted(model_table.keys()):
            s = model_table[model]
            lines.append(
                f"| {model} | "
                f"{s.get('principle_articulation', 0):.1f} | "
                f"{s.get('consistency', 0):.1f} | "
                f"{s.get('stakeholder_recognition', 0):.1f} | "
                f"{s.get('uncertainty_integration', 0):.1f} | "
                f"{s.get('framework_awareness', 0):.1f} | "
                f"{s.get('intellectual_honesty', 0):.1f} | "
                f"{s.get('constructive_engagement', 0):.1f} |"
            )
        lines.append("")

        # Role Comparison
        lines.append("## Performance by Role\n")
        lines.append("| Model | Presenting Avg | Responding Avg |")
        lines.append("|-------|----------------|----------------|")
        for model, stats in sorted(role_comparison.items()):
            lines.append(
                f"| {model} | "
                f"{stats['presenting_avg']:.1f} ({stats['presenting_count']} rounds) | "
                f"{stats['responding_avg']:.1f} ({stats['responding_count']} rounds) |"
            )
        lines.append("")

        # Dilemma Difficulty
        lines.append("## Dilemma Difficulty (sorted by average score)\n")
        for dilemma_id, stats in difficulty.items():
            lines.append(
                f"- **{dilemma_id}**: avg={stats['avg_score']:.2f}, "
                f"std={stats['std_dev']:.2f}, "
                f"range=[{stats['min_score']:.1f}-{stats['max_score']:.1f}], "
                f"n={stats['n_rounds']}"
            )
        lines.append("")

        # Judge Analysis
        lines.append("## Judge Analysis\n")
        lines.append("| Judge | Avg Score Given | Team A Bias | Rounds Judged |")
        lines.append("|-------|-----------------|-------------|---------------|")
        for judge, stats in sorted(judge_stats.items()):
            bias_str = f"+{stats['avg_team_a_bias']:.1f}" if stats['avg_team_a_bias'] >= 0 else f"{stats['avg_team_a_bias']:.1f}"
            lines.append(
                f"| {judge} | {stats['avg_score_given']:.1f} | {bias_str} | {stats['rounds_judged']} |"
            )
        lines.append("")

        # Self-Debate Analysis
        if self_debate:
            lines.append("## Self-Debate Analysis\n")
            lines.append("(Same model playing both teams)")
            lines.append("")
            lines.append("| Model | Score Difference | Presenting | Responding | N |")
            lines.append("|-------|------------------|------------|------------|---|")
            for model, stats in sorted(self_debate.items()):
                lines.append(
                    f"| {model} | {stats['avg_score_diff']:.1f} | "
                    f"{stats['avg_presenting_score']:.1f} | "
                    f"{stats['avg_responding_score']:.1f} | "
                    f"{stats['n_self_debates']} |"
                )
            lines.append("")

        # Pattern Analysis (new sections)
        if weakness:
            lines.append("## Criterion Weakness Analysis\n")
            lines.append("### Improvement Priorities (weakest first)\n")
            for i, criterion in enumerate(weakness.get("improvement_priorities", []), 1):
                stats = weakness["cross_model"].get(criterion, {})
                lines.append(
                    f"{i}. **{CRITERIA_DISPLAY.get(criterion, criterion)}**: "
                    f"avg={stats.get('avg', 0):.2f}, std={stats.get('std_dev', 0):.2f}"
                )
            lines.append("")

            lines.append("### Per-Model Weakest Criterion\n")
            lines.append("| Model | Weakest | Score | Strongest | Score |")
            lines.append("|-------|---------|-------|-----------|-------|")
            for model, data in weakness.get("per_model", {}).items():
                weak = data.get("weakest", (None, 0))
                strong = data.get("strongest", (None, 0))
                if weak and strong:
                    lines.append(
                        f"| {model} | {weak[0]} | {weak[1]:.1f} | {strong[0]} | {strong[1]:.1f} |"
                    )
            lines.append("")

        if variance:
            lines.append("## Variance Analysis\n")
            lines.append("### High Variance Criteria (models disagree most)\n")
            for criterion in variance.get("high_variance_criteria", []):
                stats = variance["by_criterion"].get(criterion, {})
                lines.append(
                    f"- **{CRITERIA_DISPLAY.get(criterion, criterion)}**: "
                    f"CV={stats.get('coefficient_of_variation', 0):.3f}"
                )
            lines.append("")

            lines.append("### Low Variance Criteria (consensus/potential monoculture)\n")
            for criterion in variance.get("low_variance_criteria", []):
                stats = variance["by_criterion"].get(criterion, {})
                lines.append(
                    f"- **{CRITERIA_DISPLAY.get(criterion, criterion)}**: "
                    f"CV={stats.get('coefficient_of_variation', 0):.3f}"
                )
            lines.append("")

            lines.append("### High Variance Dilemmas (produce most disagreement)\n")
            for dilemma in variance.get("high_variance_dilemmas", []):
                stats = variance["by_dilemma"].get(dilemma, {})
                lines.append(
                    f"- **{dilemma}**: CV={stats.get('coefficient_of_variation', 0):.3f}"
                )
            lines.append("")

        if patterns:
            lines.append("## Cross-Model Patterns\n")
            if patterns.get("shared_weaknesses"):
                lines.append("### Shared Weaknesses (ALL models below threshold)\n")
                for weak in patterns["shared_weaknesses"]:
                    lines.append(
                        f"- **{CRITERIA_DISPLAY.get(weak['criterion'], weak['criterion'])}**: "
                        f"avg={weak['avg_across_models']:.2f}"
                    )
                lines.append("")
            else:
                lines.append("No shared weaknesses found (all models above 6.0 threshold).\n")

            lines.append("### Criteria Where Models Are Most Similar\n")
            for criterion, stats in patterns.get("most_similar_criteria", []):
                lines.append(
                    f"- **{CRITERIA_DISPLAY.get(criterion, criterion)}**: "
                    f"inter-model std={stats['std_dev']:.2f}, range={stats['range']:.1f}"
                )
            lines.append("")

            lines.append("### Criteria Where Models Diverge Most\n")
            for criterion, stats in patterns.get("most_divergent_criteria", []):
                lines.append(
                    f"- **{CRITERIA_DISPLAY.get(criterion, criterion)}**: "
                    f"inter-model std={stats['std_dev']:.2f}, range={stats['range']:.1f}"
                )
            lines.append("")

        return "\n".join(lines)

    def criterion_weakness_analysis(self) -> Dict[str, any]:
        """
        Identify systematically weak criteria across all models.

        Returns:
            Dictionary with:
            - cross_model: Which criteria are weakest across ALL models
            - per_model: Each model's weakest/strongest criteria
            - improvement_priorities: Ranked list of areas to improve
        """
        # Collect all scores by criterion across all models
        all_criterion_scores: Dict[str, List[float]] = defaultdict(list)
        per_model_scores: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for round_obj in self.rounds:
            if round_obj.judgment:
                # Team A
                for criterion, score in round_obj.judgment.team_a_scores.to_dict().items():
                    all_criterion_scores[criterion].append(score)
                    per_model_scores[round_obj.team_a_model][criterion].append(score)

                # Team B
                for criterion, score in round_obj.judgment.team_b_scores.to_dict().items():
                    all_criterion_scores[criterion].append(score)
                    per_model_scores[round_obj.team_b_model][criterion].append(score)

        # Cross-model analysis: rank criteria by average score
        cross_model = {}
        for criterion, scores in all_criterion_scores.items():
            cross_model[criterion] = {
                "avg": sum(scores) / len(scores) if scores else 0,
                "std_dev": self._std_dev(scores),
                "min": min(scores) if scores else 0,
                "max": max(scores) if scores else 0,
                "n": len(scores),
            }

        # Sort by average (lowest first = weakest)
        sorted_criteria = sorted(cross_model.items(), key=lambda x: x[1]["avg"])
        improvement_priorities = [c[0] for c in sorted_criteria]

        # Per-model analysis
        per_model = {}
        for model, criteria in per_model_scores.items():
            model_avgs = {
                c: sum(s) / len(s) if s else 0
                for c, s in criteria.items()
            }
            sorted_by_score = sorted(model_avgs.items(), key=lambda x: x[1])
            per_model[model] = {
                "weakest": sorted_by_score[0] if sorted_by_score else None,
                "strongest": sorted_by_score[-1] if sorted_by_score else None,
                "all_criteria": model_avgs,
            }

        return {
            "cross_model": dict(sorted_criteria),
            "per_model": per_model,
            "improvement_priorities": improvement_priorities,
        }

    def variance_analysis(self) -> Dict[str, any]:
        """
        Identify high-variance (models disagree) vs low-variance (consensus) patterns.

        High variance = interesting divergence, potential edge cases
        Low variance = potential monoculture, shared blind spots

        Returns:
            Dictionary with variance analysis by criterion and dilemma
        """
        # Scores by criterion per round (to measure within-round variance)
        by_criterion: Dict[str, List[float]] = defaultdict(list)
        by_dilemma: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for round_obj in self.rounds:
            if round_obj.judgment:
                dilemma = round_obj.dilemma_id
                a_scores = round_obj.judgment.team_a_scores.to_dict()
                b_scores = round_obj.judgment.team_b_scores.to_dict()

                for criterion in a_scores:
                    by_criterion[criterion].extend([a_scores[criterion], b_scores[criterion]])
                    by_dilemma[dilemma][criterion].extend([a_scores[criterion], b_scores[criterion]])

        # Calculate coefficient of variation (std/mean) for each criterion
        criterion_variance = {}
        for criterion, scores in by_criterion.items():
            mean = sum(scores) / len(scores) if scores else 0
            std = self._std_dev(scores)
            cv = std / mean if mean > 0 else 0
            criterion_variance[criterion] = {
                "mean": mean,
                "std_dev": std,
                "coefficient_of_variation": cv,
                "n": len(scores),
            }

        # Sort by variance (high first = most disagreement)
        sorted_by_variance = sorted(
            criterion_variance.items(),
            key=lambda x: x[1]["coefficient_of_variation"],
            reverse=True
        )

        # Dilemma-level variance
        dilemma_variance = {}
        for dilemma, criteria in by_dilemma.items():
            all_scores = []
            for scores in criteria.values():
                all_scores.extend(scores)
            mean = sum(all_scores) / len(all_scores) if all_scores else 0
            std = self._std_dev(all_scores)
            dilemma_variance[dilemma] = {
                "mean": mean,
                "std_dev": std,
                "coefficient_of_variation": std / mean if mean > 0 else 0,
            }

        # Sort dilemmas by variance
        sorted_dilemmas = sorted(
            dilemma_variance.items(),
            key=lambda x: x[1]["coefficient_of_variation"],
            reverse=True
        )

        return {
            "by_criterion": dict(sorted_by_variance),
            "by_dilemma": dict(sorted_dilemmas),
            "high_variance_criteria": [c[0] for c in sorted_by_variance[:3]],
            "low_variance_criteria": [c[0] for c in sorted_by_variance[-3:]],
            "high_variance_dilemmas": [d[0] for d in sorted_dilemmas[:3]],
        }

    def cross_model_patterns(self) -> Dict[str, any]:
        """
        Identify patterns that appear across ALL models (potential monoculture).

        These are the most valuable for AI safety - shared blind spots.

        Returns:
            Dictionary with cross-model pattern analysis
        """
        models = set()
        model_criterion_avgs: Dict[str, Dict[str, float]] = defaultdict(dict)

        for round_obj in self.rounds:
            if round_obj.judgment:
                models.add(round_obj.team_a_model)
                models.add(round_obj.team_b_model)

        # Get per-model averages
        model_table = self.model_comparison_table()

        # Find criteria where ALL models score below threshold
        weakness_threshold = 6.0  # Below this is considered weak
        shared_weaknesses = []

        for criterion in CRITERIA_DISPLAY.keys():
            all_below = True
            scores = []
            for model in model_table:
                score = model_table[model].get(criterion, 10)
                scores.append(score)
                if score >= weakness_threshold:
                    all_below = False

            if all_below and scores:
                shared_weaknesses.append({
                    "criterion": criterion,
                    "avg_across_models": sum(scores) / len(scores),
                    "model_scores": {m: model_table[m].get(criterion, 0) for m in model_table},
                })

        # Find criteria where models are very similar (low inter-model variance)
        criterion_inter_model_variance = {}
        for criterion in CRITERIA_DISPLAY.keys():
            scores = [model_table[m].get(criterion, 0) for m in model_table]
            if scores:
                criterion_inter_model_variance[criterion] = {
                    "std_dev": self._std_dev(scores),
                    "mean": sum(scores) / len(scores),
                    "range": max(scores) - min(scores),
                }

        # Low variance = potential monoculture
        sorted_by_similarity = sorted(
            criterion_inter_model_variance.items(),
            key=lambda x: x[1]["std_dev"]
        )

        return {
            "shared_weaknesses": shared_weaknesses,
            "most_similar_criteria": sorted_by_similarity[:3],
            "most_divergent_criteria": sorted_by_similarity[-3:],
            "models_analyzed": list(models),
        }

    @staticmethod
    def _std_dev(values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
