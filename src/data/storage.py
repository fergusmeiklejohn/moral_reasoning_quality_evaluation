"""
Data storage and management for experiment results.

Handles saving, loading, and organizing experiment data with automatic backups.
"""

import json
import jsonlines
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
import shutil

from .schemas import (
    ExperimentRun,
    ExperimentConfig,
    AnalysisResult,
    ExperimentRunV2,
    ExperimentConfigV2,
    V2ScoringResult,
    V2AnalysisResult,
)


class ExperimentStorage:
    """Manages storage of experiment data."""

    def __init__(self, results_dir: Optional[Path] = None):
        """
        Initialize experiment storage.

        Args:
            results_dir: Directory for storing results. If None, uses default location.
        """
        if results_dir is None:
            # Default location
            project_root = Path(__file__).parent.parent.parent
            results_dir = project_root / "data" / "results"

        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def create_experiment_directory(self, experiment_id: str) -> Path:
        """
        Create a directory for an experiment.

        Args:
            experiment_id: Unique experiment identifier

        Returns:
            Path to experiment directory
        """
        exp_dir = self.results_dir / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (exp_dir / "runs").mkdir(exist_ok=True)
        (exp_dir / "analysis").mkdir(exist_ok=True)
        (exp_dir / "backups").mkdir(exist_ok=True)

        return exp_dir

    def save_experiment_config(
        self,
        experiment_id: str,
        config: ExperimentConfig
    ) -> Path:
        """
        Save experiment configuration.

        Args:
            experiment_id: Experiment identifier
            config: Experiment configuration

        Returns:
            Path to saved config file
        """
        exp_dir = self.create_experiment_directory(experiment_id)
        config_file = exp_dir / "config.json"

        with open(config_file, 'w') as f:
            json.dump(config.model_dump(), f, indent=2, default=str)

        return config_file

    def save_run(
        self,
        experiment_id: str,
        run: ExperimentRun
    ) -> Path:
        """
        Save a single experiment run.

        Uses JSON Lines format for efficient append operations.

        Args:
            experiment_id: Experiment identifier
            run: Experiment run data

        Returns:
            Path to runs file
        """
        exp_dir = self.create_experiment_directory(experiment_id)
        runs_file = exp_dir / "runs" / "all_runs.jsonl"

        # Append to JSONL file
        with jsonlines.open(runs_file, mode='a') as writer:
            writer.write(run.model_dump(mode='json'))

        return runs_file

    def save_runs_batch(
        self,
        experiment_id: str,
        runs: List[ExperimentRun]
    ) -> Path:
        """
        Save multiple experiment runs at once.

        Args:
            experiment_id: Experiment identifier
            runs: List of experiment runs

        Returns:
            Path to runs file
        """
        exp_dir = self.create_experiment_directory(experiment_id)
        runs_file = exp_dir / "runs" / "all_runs.jsonl"

        with jsonlines.open(runs_file, mode='a') as writer:
            for run in runs:
                writer.write(run.model_dump(mode='json'))

        return runs_file

    def load_runs(
        self,
        experiment_id: str,
        model_name: Optional[str] = None,
        dilemma_id: Optional[str] = None
    ) -> List[ExperimentRun]:
        """
        Load experiment runs with optional filtering.

        Args:
            experiment_id: Experiment identifier
            model_name: Filter by model name (optional)
            dilemma_id: Filter by dilemma ID (optional)

        Returns:
            List of experiment runs

        Raises:
            FileNotFoundError: If experiment not found
        """
        exp_dir = self.results_dir / experiment_id
        runs_file = exp_dir / "runs" / "all_runs.jsonl"

        if not runs_file.exists():
            raise FileNotFoundError(f"No runs found for experiment '{experiment_id}'")

        runs = []
        with jsonlines.open(runs_file) as reader:
            for obj in reader:
                run = ExperimentRun(**obj)

                # Apply filters
                if model_name and run.model_name != model_name:
                    continue
                if dilemma_id and run.dilemma_id != dilemma_id:
                    continue

                runs.append(run)

        return runs

    def load_experiment_config(self, experiment_id: str) -> ExperimentConfig:
        """
        Load experiment configuration.

        Args:
            experiment_id: Experiment identifier

        Returns:
            Experiment configuration

        Raises:
            FileNotFoundError: If config not found
        """
        exp_dir = self.results_dir / experiment_id
        config_file = exp_dir / "config.json"

        if not config_file.exists():
            raise FileNotFoundError(
                f"Config not found for experiment '{experiment_id}'"
            )

        with open(config_file, 'r') as f:
            config_data = json.load(f)

        return ExperimentConfig(**config_data)

    def create_backup(
        self,
        experiment_id: str,
        backup_name: Optional[str] = None
    ) -> Path:
        """
        Create a backup of experiment data.

        Args:
            experiment_id: Experiment identifier
            backup_name: Optional backup name (default: timestamp)

        Returns:
            Path to backup directory
        """
        exp_dir = self.results_dir / experiment_id

        if not exp_dir.exists():
            raise FileNotFoundError(f"Experiment '{experiment_id}' not found")

        if backup_name is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            backup_name = f"backup_{timestamp}"

        backup_dir = exp_dir / "backups" / backup_name
        backup_dir.mkdir(parents=True, exist_ok=True)

        # Copy runs file
        runs_file = exp_dir / "runs" / "all_runs.jsonl"
        if runs_file.exists():
            shutil.copy2(runs_file, backup_dir / "all_runs.jsonl")

        # Copy config
        config_file = exp_dir / "config.json"
        if config_file.exists():
            shutil.copy2(config_file, backup_dir / "config.json")

        return backup_dir

    def save_analysis_result(
        self,
        experiment_id: str,
        analysis: AnalysisResult,
        analysis_name: Optional[str] = None
    ) -> Path:
        """
        Save analysis results.

        Args:
            experiment_id: Experiment identifier
            analysis: Analysis result
            analysis_name: Optional name for analysis file

        Returns:
            Path to analysis file
        """
        exp_dir = self.create_experiment_directory(experiment_id)

        if analysis_name is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            analysis_name = f"analysis_{timestamp}.json"

        analysis_file = exp_dir / "analysis" / analysis_name

        with open(analysis_file, 'w') as f:
            json.dump(analysis.model_dump(), f, indent=2, default=str)

        return analysis_file

    def get_experiment_summary(self, experiment_id: str) -> Dict[str, Any]:
        """
        Get summary statistics for an experiment.

        Args:
            experiment_id: Experiment identifier

        Returns:
            Dictionary with summary statistics
        """
        runs = self.load_runs(experiment_id)

        if not runs:
            return {
                "experiment_id": experiment_id,
                "total_runs": 0,
                "status": "no data"
            }

        # Count by various dimensions
        models = set(run.model_name for run in runs)
        dilemmas = set(run.dilemma_id for run in runs)
        temperatures = set(run.temperature for run in runs)

        # Count errors
        errors = sum(1 for run in runs if run.response.parsed_choice.value == "ERROR")
        refusals = sum(1 for run in runs if run.response.parsed_choice.value == "REFUSE")

        return {
            "experiment_id": experiment_id,
            "total_runs": len(runs),
            "models": list(models),
            "num_models": len(models),
            "dilemmas": list(dilemmas),
            "num_dilemmas": len(dilemmas),
            "temperatures": sorted(temperatures),
            "errors": errors,
            "refusals": refusals,
            "error_rate": errors / len(runs) if runs else 0,
            "refusal_rate": refusals / len(runs) if runs else 0,
        }

    def list_experiments(self) -> List[str]:
        """
        List all experiment IDs.

        Returns:
            List of experiment IDs
        """
        if not self.results_dir.exists():
            return []

        experiments = [
            d.name for d in self.results_dir.iterdir()
            if d.is_dir() and not d.name.startswith('.')
        ]

        return sorted(experiments)

    def __repr__(self) -> str:
        return f"ExperimentStorage(results_dir={self.results_dir})"

    # =========================================================================
    # VERSION 2 STORAGE METHODS
    # =========================================================================

    def save_experiment_config_v2(
        self,
        experiment_id: str,
        config: ExperimentConfigV2
    ) -> Path:
        """
        Save V2 experiment configuration.

        Args:
            experiment_id: Experiment identifier
            config: V2 experiment configuration

        Returns:
            Path to saved config file
        """
        exp_dir = self.create_experiment_directory(experiment_id)
        config_file = exp_dir / "config_v2.json"

        with open(config_file, 'w') as f:
            json.dump(config.model_dump(), f, indent=2, default=str)

        return config_file

    def save_run_v2(
        self,
        experiment_id: str,
        run: ExperimentRunV2
    ) -> Path:
        """
        Save a single V2 experiment run.

        Uses JSON Lines format for efficient append operations.

        Args:
            experiment_id: Experiment identifier
            run: V2 experiment run data

        Returns:
            Path to runs file
        """
        exp_dir = self.create_experiment_directory(experiment_id)
        runs_file = exp_dir / "runs" / "v2_runs.jsonl"

        with jsonlines.open(runs_file, mode='a') as writer:
            writer.write(run.model_dump(mode='json'))

        return runs_file

    def save_runs_batch_v2(
        self,
        experiment_id: str,
        runs: List[ExperimentRunV2]
    ) -> Path:
        """
        Save multiple V2 experiment runs at once.

        Args:
            experiment_id: Experiment identifier
            runs: List of V2 experiment runs

        Returns:
            Path to runs file
        """
        exp_dir = self.create_experiment_directory(experiment_id)
        runs_file = exp_dir / "runs" / "v2_runs.jsonl"

        with jsonlines.open(runs_file, mode='a') as writer:
            for run in runs:
                writer.write(run.model_dump(mode='json'))

        return runs_file

    def load_runs_v2(
        self,
        experiment_id: str,
        model_name: Optional[str] = None,
        dilemma_id: Optional[str] = None
    ) -> List[ExperimentRunV2]:
        """
        Load V2 experiment runs with optional filtering.

        Args:
            experiment_id: Experiment identifier
            model_name: Filter by model name (optional)
            dilemma_id: Filter by dilemma ID (optional)

        Returns:
            List of V2 experiment runs

        Raises:
            FileNotFoundError: If experiment not found
        """
        exp_dir = self.results_dir / experiment_id
        runs_file = exp_dir / "runs" / "v2_runs.jsonl"

        if not runs_file.exists():
            raise FileNotFoundError(
                f"No V2 runs found for experiment '{experiment_id}'"
            )

        runs = []
        with jsonlines.open(runs_file) as reader:
            for obj in reader:
                run = ExperimentRunV2(**obj)

                # Apply filters
                if model_name and run.model_name != model_name:
                    continue
                if dilemma_id and run.dilemma_id != dilemma_id:
                    continue

                runs.append(run)

        return runs

    def load_experiment_config_v2(self, experiment_id: str) -> ExperimentConfigV2:
        """
        Load V2 experiment configuration.

        Args:
            experiment_id: Experiment identifier

        Returns:
            V2 experiment configuration

        Raises:
            FileNotFoundError: If config not found
        """
        exp_dir = self.results_dir / experiment_id
        config_file = exp_dir / "config_v2.json"

        if not config_file.exists():
            raise FileNotFoundError(
                f"V2 config not found for experiment '{experiment_id}'"
            )

        with open(config_file, 'r') as f:
            config_data = json.load(f)

        return ExperimentConfigV2(**config_data)

    def save_scoring_results_v2(
        self,
        experiment_id: str,
        results: List[V2ScoringResult]
    ) -> Path:
        """
        Save V2 scoring results.

        Stores full judge output for auditing purposes.

        Args:
            experiment_id: Experiment identifier
            results: List of scoring results

        Returns:
            Path to scoring results file
        """
        exp_dir = self.create_experiment_directory(experiment_id)
        scoring_file = exp_dir / "analysis" / "scoring_results_v2.jsonl"

        with jsonlines.open(scoring_file, mode='a') as writer:
            for result in results:
                writer.write(result.model_dump(mode='json'))

        return scoring_file

    def load_scoring_results_v2(
        self,
        experiment_id: str,
        model_name: Optional[str] = None,
        dilemma_id: Optional[str] = None
    ) -> List[V2ScoringResult]:
        """
        Load V2 scoring results with optional filtering.

        Args:
            experiment_id: Experiment identifier
            model_name: Filter by model name (optional)
            dilemma_id: Filter by dilemma ID (optional)

        Returns:
            List of scoring results

        Raises:
            FileNotFoundError: If scoring results not found
        """
        exp_dir = self.results_dir / experiment_id
        scoring_file = exp_dir / "analysis" / "scoring_results_v2.jsonl"

        if not scoring_file.exists():
            raise FileNotFoundError(
                f"No V2 scoring results found for experiment '{experiment_id}'"
            )

        results = []
        with jsonlines.open(scoring_file) as reader:
            for obj in reader:
                result = V2ScoringResult(**obj)

                # Apply filters
                if model_name and result.model_name != model_name:
                    continue
                if dilemma_id and result.dilemma_id != dilemma_id:
                    continue

                results.append(result)

        return results

    def save_analysis_result_v2(
        self,
        experiment_id: str,
        analysis: V2AnalysisResult,
        analysis_name: Optional[str] = None
    ) -> Path:
        """
        Save V2 analysis results.

        Args:
            experiment_id: Experiment identifier
            analysis: V2 analysis result
            analysis_name: Optional name for analysis file

        Returns:
            Path to analysis file
        """
        exp_dir = self.create_experiment_directory(experiment_id)

        if analysis_name is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            analysis_name = f"analysis_v2_{timestamp}.json"

        analysis_file = exp_dir / "analysis" / analysis_name

        with open(analysis_file, 'w') as f:
            json.dump(analysis.model_dump(), f, indent=2, default=str)

        return analysis_file

    def create_backup_v2(
        self,
        experiment_id: str,
        backup_name: Optional[str] = None
    ) -> Path:
        """
        Create a backup of V2 experiment data.

        Args:
            experiment_id: Experiment identifier
            backup_name: Optional backup name (default: timestamp)

        Returns:
            Path to backup directory
        """
        exp_dir = self.results_dir / experiment_id

        if not exp_dir.exists():
            raise FileNotFoundError(f"Experiment '{experiment_id}' not found")

        if backup_name is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            backup_name = f"backup_v2_{timestamp}"

        backup_dir = exp_dir / "backups" / backup_name
        backup_dir.mkdir(parents=True, exist_ok=True)

        # Copy V2 runs file
        runs_file = exp_dir / "runs" / "v2_runs.jsonl"
        if runs_file.exists():
            shutil.copy2(runs_file, backup_dir / "v2_runs.jsonl")

        # Copy V2 config
        config_file = exp_dir / "config_v2.json"
        if config_file.exists():
            shutil.copy2(config_file, backup_dir / "config_v2.json")

        # Copy scoring results
        scoring_file = exp_dir / "analysis" / "scoring_results_v2.jsonl"
        if scoring_file.exists():
            shutil.copy2(scoring_file, backup_dir / "scoring_results_v2.jsonl")

        return backup_dir

    def get_experiment_summary_v2(self, experiment_id: str) -> Dict[str, Any]:
        """
        Get summary statistics for a V2 experiment.

        Args:
            experiment_id: Experiment identifier

        Returns:
            Dictionary with summary statistics
        """
        try:
            runs = self.load_runs_v2(experiment_id)
        except FileNotFoundError:
            return {
                "experiment_id": experiment_id,
                "total_runs": 0,
                "status": "no data"
            }

        if not runs:
            return {
                "experiment_id": experiment_id,
                "total_runs": 0,
                "status": "no data"
            }

        # Count by various dimensions
        models = set(run.model_name for run in runs)
        dilemmas = set(run.dilemma_id for run in runs)
        structures = set(run.dilemma_structure.value for run in runs)

        # Count errors
        errors = sum(1 for run in runs if run.error is not None)

        # Count scored runs
        scored = sum(1 for run in runs if run.scoring is not None)

        return {
            "experiment_id": experiment_id,
            "experiment_type": "v2_moral_reasoning",
            "total_runs": len(runs),
            "scored_runs": scored,
            "models": list(models),
            "num_models": len(models),
            "dilemmas": list(dilemmas),
            "num_dilemmas": len(dilemmas),
            "structures": list(structures),
            "errors": errors,
            "error_rate": errors / len(runs) if runs else 0,
        }
